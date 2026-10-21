# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Forgetting-gated self-imitation on verl 0.6's RayPPOTrainer.

Per step t, one rollout call generates
    - n rollouts of this step's prompts with the current weights        -> the GRPO rows
    - m rollouts of step t+lag's prompts with the same weights           -> replay for step t+lag
so the replay a step uses was sampled by the policy `lag_steps` updates ago, without a second
inference engine or any weight swapping. See recipe/self_imitation/README.md.
"""

import uuid
from collections import defaultdict
from pprint import pprint

import numpy as np
import ray
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer

from .replay import (
    PREFIX,
    ROW_CUR,
    build_train_batch,
    current_rows_mask,
    current_rows_unscaled,
    lagged_schedule,
    mask_padding,
    replay_drift_metrics,
    rollout_is_weights_current_only,
    stash_replay,
    token_advantages,
    train_view,
)


class _PromptBatch:
    """One dataloader batch, plus the replay rollouts generated for it `lag_steps` steps earlier."""

    def __init__(self, batch_dict: dict, dataloader_state: dict):
        self.batch_dict = batch_dict
        # The dataloader's position right after this batch: what a checkpoint taken at this step
        # must save, since the live dataloader has already been read `lag_steps` batches further.
        self.dataloader_state = dataloader_state
        self.uids = np.array([str(uuid.uuid4()) for _ in range(len(batch_dict["input_ids"]))], dtype=object)
        # Replay rows live in the object store while they wait: ~150 MB per step at 16k context.
        self.replay_ref = None
        self.p_old: dict[str, float] | None = None

    def expand(self, trainer: RayPPOTrainer, times: int) -> tuple[DataProto, DataProto]:
        """(non-generation fields, generation input), each repeated `times` per prompt."""
        batch = DataProto.from_single_dict(self.batch_dict)
        batch.non_tensor_batch["uid"] = self.uids.copy()
        gen_batch = trainer._get_gen_batch(batch)  # pops the prompt tensors out of `batch`
        return batch.repeat(times, interleave=True), gen_batch.repeat(times, interleave=True)


class RaySelfImitationTrainer(RayPPOTrainer):
    def _batch_stream(self):
        """(batch_dict, dataloader state right after that batch), across epochs."""
        for _ in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                yield batch_dict, self.train_dataloader.state_dict()

    def _save_checkpoint(self):
        # The base class saves train_dataloader.state_dict(), the live position -- `lag_steps`
        # batches ahead of training. Save the position right after the batch just trained on
        # instead, so a resume re-reads the prefetched batches rather than skipping them.
        if self._trained_dataloader_state is None:
            return super()._save_checkpoint()
        self.train_dataloader.state_dict = lambda: self._trained_dataloader_state
        try:
            super()._save_checkpoint()
        finally:
            del self.train_dataloader.state_dict

    def _score(self, batch: DataProto) -> tuple[DataProto, dict]:
        result = self.reward_fn(batch, return_dict=True)
        batch.batch["token_level_scores"] = result["reward_tensor"]
        batch.batch["token_level_rewards"] = result["reward_tensor"]
        return batch, result.get("reward_extra_info", {})

    def _rollout(self, cur: _PromptBatch, ahead: _PromptBatch | None, timing_raw: dict):
        """Generate cur x n and ahead x m in one call; stash ahead's replay on it, return cur's scored rollouts."""
        n = self.config.actor_rollout_ref.rollout.n
        m = self.config.self_imitation.num_replay
        cur_rest, cur_gen = cur.expand(self, n)
        gens = [cur_gen]
        if ahead is not None:
            ahead_rest, ahead_gen = ahead.expand(self, m)
            gens.append(ahead_gen)
        gen_batch = DataProto.concat(gens)
        gen_batch.meta_info["global_steps"] = self.global_steps

        with marked_timer("gen", timing_raw, "red"):
            out = self.actor_rollout_wg.generate_sequences(gen_batch)
            timing_raw.update(out.meta_info["timing"])
            out.meta_info.pop("timing", None)

        with marked_timer("reward", timing_raw, "yellow"):
            cur_batch, extra = self._score(cur_rest.union(out[: len(cur_gen)]))
            if ahead is not None:
                replay_batch, _ = self._score(ahead_rest.union(out[len(cur_gen) :]))
                rows, ahead.p_old = stash_replay(replay_batch)
                ahead.replay_ref = ray.put(rows) if rows is not None else None
        return cur_batch, extra

    def _train_step(self, cur: _PromptBatch, cur_batch: DataProto, metrics: dict, timing_raw: dict) -> DataProto:
        si = self.config.self_imitation
        world_size = self.actor_rollout_wg.world_size
        num_mini_batches = self.config.data.train_batch_size // self.config.actor_rollout_ref.actor.ppo_mini_batch_size

        replay = ray.get(cur.replay_ref) if cur.replay_ref is not None else None
        cur.replay_ref = None
        batch, n_replay, si_metrics = build_train_batch(
            train_view(cur_batch, ROW_CUR),
            replay,
            cur.p_old,
            tau=si.gate_tau,
            coef=si.replay_coef,
            row_divisor=world_size * num_mini_batches,
        )
        metrics.update(si_metrics)

        batch.batch["response_mask"] = compute_response_mask(batch)
        mask_padding(batch)
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)
        batch.meta_info["global_token_num"] = batch.batch["attention_mask"].sum(-1).tolist()

        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropy = agg_loss(
                loss_mat=old_log_prob.batch.pop("entropys"),
                loss_mask=current_rows_mask(batch),
                loss_agg_mode=self.config.actor_rollout_ref.actor.loss_agg_mode,
            )
            metrics["actor/entropy"] = entropy.item()
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, "olive"):
                batch = batch.union(self.ref_policy_wg.compute_ref_log_prob(batch))

        algo = self.config.algorithm
        if algo.rollout_is_threshold is not None and "rollout_log_probs" in batch.batch:
            weights, is_metrics = rollout_is_weights_current_only(batch, algo)
            metrics.update(is_metrics)
            if algo.get("rollout_is", False):
                batch.batch["rollout_is_weights"] = weights
        if n_replay:
            metrics.update(replay_drift_metrics(batch, self.config.actor_rollout_ref.actor.clip_ratio_low))

        with marked_timer("adv", timing_raw, "brown"):
            advantages, scale = token_advantages(batch)
            batch.batch["advantages"] = advantages
            batch.batch["returns"] = advantages
            metrics[f"{PREFIX}/adv_scale"] = scale

        if n_replay:
            # The batch grew by a variable number of rows: keep the same number of optimizer steps.
            batch.meta_info["num_mini_batches"] = num_mini_batches
        with marked_timer("update_actor", timing_raw, "red"):
            actor_output = self.actor_rollout_wg.update_actor(batch)
        metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))
        # Data metrics describe the current policy's rollouts; replay and padding rows are excluded.
        metrics.update(compute_data_metrics(batch=current_rows_unscaled(batch, scale), use_critic=False))
        return batch

    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._trained_dataloader_state = None
        self._load_checkpoint()
        last_saved_step = self.global_steps

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None

        lag = self.config.self_imitation.lag_steps
        generate_replay = self.config.self_imitation.num_replay > 0
        if self.global_steps > 1 and generate_replay and lag > 0:
            # Checkpoints save the dataloader position of the trained batch (_save_checkpoint), so
            # no prompts are skipped; only the in-flight replay is lost.
            print(
                f"WARNING: resumed at step {self.global_steps}: pending replay is not checkpointed, so "
                f"steps {self.global_steps}..{self.global_steps + lag - 1} train without replay."
            )
        prompt_batches = (_PromptBatch(*item) for item in self._batch_stream())
        for cur, ahead in lagged_schedule(prompt_batches, lag):
            is_last_step = self.global_steps >= self.total_training_steps
            self._trained_dataloader_state = cur.dataloader_state
            # No replay for steps that will never run.
            if not generate_replay or self.global_steps + lag > self.total_training_steps:
                ahead = None

            metrics = {}
            timing_raw = defaultdict(float)
            with marked_timer("step", timing_raw):
                cur_batch, _ = self._rollout(cur, ahead, timing_raw)
                batch = self._train_step(cur, cur_batch, metrics, timing_raw)

            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with marked_timer("testing", timing_raw, "green"):
                    val_metrics = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                with marked_timer("save_checkpoint", timing_raw, "green"):
                    self._save_checkpoint()
                last_saved_step = self.global_steps

            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1

        # The data ran out before total_training_steps, so the loop never saw is_last_step: save the last
        # trained step (global_steps is one past it) unless the save_freq schedule already did. No log:
        # that step's metrics were logged in the loop, and trackers reject a repeated step.
        self.global_steps -= 1
        if self.config.trainer.save_freq > 0 and last_saved_step != self.global_steps:
            self._save_checkpoint()
