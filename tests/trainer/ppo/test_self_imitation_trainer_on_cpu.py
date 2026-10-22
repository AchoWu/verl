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

"""CPU test of the self_imitation trainer's step wiring, with fake workers in place of Ray/vLLM/FSDP.

The fake rollout writes the step that generated a response into its tokens, and the fake reward
calls a response correct iff it was generated at step 1. With lag 2, step 3 then trains on replay
sampled at step 1 (all correct, while its current rollouts are all wrong), so the gate must open.
"""

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import recipe.self_imitation.self_imitation_trainer as sit
from recipe.self_imitation.replay import ROW_CUR, ROW_REPLAY, lagged_schedule, row_kinds
from verl import DataProto

PROMPT_LEN, RESP_LEN, N_PROMPTS, N, M, LAG = 3, 4, 4, 2, 2, 2


class FakeWorkers:
    world_size = 2

    def __init__(self):
        self.generated, self.updates = [], []

    def generate_sequences(self, gen: DataProto) -> DataProto:
        bsz = len(gen)
        self.generated.append(bsz)
        responses = torch.full((bsz, RESP_LEN), gen.meta_info["global_steps"])
        attention_mask = torch.cat([gen.batch["attention_mask"], torch.ones(bsz, RESP_LEN, dtype=torch.long)], -1)
        return DataProto.from_dict(
            tensors={
                "prompts": gen.batch["input_ids"],
                "responses": responses,
                "input_ids": torch.cat([gen.batch["input_ids"], responses], -1),
                "attention_mask": attention_mask,
                "position_ids": torch.arange(PROMPT_LEN + RESP_LEN).expand(bsz, -1).clone(),
                "rollout_log_probs": torch.zeros(bsz, RESP_LEN),
            },
            meta_info={"timing": {}},
        )

    def compute_log_prob(self, batch: DataProto) -> DataProto:
        zeros = torch.zeros(len(batch), RESP_LEN)
        return DataProto.from_dict(tensors={"old_log_probs": zeros, "entropys": zeros.clone()})

    def update_actor(self, batch: DataProto) -> DataProto:
        self.updates.append(batch)
        return DataProto(meta_info={"metrics": {}})


def fake_reward_fn(batch: DataProto, return_dict=False):
    scores = torch.zeros(len(batch), RESP_LEN)
    scores[:, -1] = (batch.batch["responses"][:, 0] == 1).float()
    return {"reward_tensor": scores, "reward_extra_info": {}}


def batch_dict(i: int) -> dict:
    raw_prompt_ids = np.empty(N_PROMPTS, dtype=object)
    raw_prompt_ids[:] = [[i, j] for j in range(N_PROMPTS)]
    return {
        "input_ids": torch.full((N_PROMPTS, PROMPT_LEN), i),
        "attention_mask": torch.ones(N_PROMPTS, PROMPT_LEN, dtype=torch.long),
        "position_ids": torch.arange(PROMPT_LEN).expand(N_PROMPTS, -1).clone(),
        "raw_prompt_ids": raw_prompt_ids,
        "data_source": np.array(["math"] * N_PROMPTS, dtype=object),
    }


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setattr(sit, "ray", SimpleNamespace(put=lambda x: x, get=lambda x: x))
    trainer = sit.RaySelfImitationTrainer.__new__(sit.RaySelfImitationTrainer)
    trainer.config = OmegaConf.create(
        {
            "self_imitation": {"lag_steps": LAG, "num_replay": M, "replay_coef": 1.0, "gate_tau": 0.375},
            "data": {"train_batch_size": N_PROMPTS},
            "actor_rollout_ref": {
                "rollout": {"n": N},
                "actor": {"ppo_mini_batch_size": 2, "loss_agg_mode": "token-mean", "clip_ratio_low": 0.2},
            },
            "trainer": {"balance_batch": False},
            "algorithm": {
                "rollout_is": True,
                "rollout_is_level": "token",
                "rollout_is_mode": "mask",
                "rollout_is_threshold": 5.0,
                "rollout_is_threshold_lower": 0.5,
                "rollout_is_veto_threshold": None,
            },
        }
    )
    trainer.actor_rollout_wg = FakeWorkers()
    trainer.reward_fn = fake_reward_fn
    trainer.async_rollout_mode = False
    trainer.use_reference_policy = False
    return trainer


def test_replay_reaches_the_step_lag_updates_later(trainer):
    all_metrics = []
    schedule = lagged_schedule((sit._PromptBatch(batch_dict(i), {}) for i in range(4)), LAG)
    for step, (cur, ahead) in enumerate(schedule, start=1):
        trainer.global_steps = step
        metrics, timing_raw = {}, defaultdict(float)
        cur_batch, _ = trainer._rollout(cur, ahead, timing_raw)
        trainer._train_step(cur, cur_batch, metrics, timing_raw)
        all_metrics.append(metrics)

    workers = trainer.actor_rollout_wg
    # Steps 1-2 also sample replay for steps 3-4; steps 3-4 have nothing left to sample ahead for.
    assert workers.generated == [N_PROMPTS * (N + M)] * 2 + [N_PROMPTS * N] * 2

    # Steps 1, 2: no replay generated for them -> plain GRPO batch, default mini-batching.
    for update, metrics in zip(workers.updates[:2], all_metrics[:2], strict=True):
        assert len(update) == N_PROMPTS * N and "num_mini_batches" not in update.meta_info
        assert "self_imitation/gate_open_frac" not in metrics

    # Step 3: replay sampled at step 1 (all correct) vs current rollouts (all wrong).
    update, metrics = workers.updates[2], all_metrics[2]
    kinds = row_kinds(update)
    assert (kinds == ROW_REPLAY).sum() == N_PROMPTS * M and (kinds == ROW_CUR).sum() == N_PROMPTS * N
    assert (update.batch["responses"][kinds == ROW_REPLAY] == 1).all()
    assert (update.batch["responses"][kinds == ROW_CUR] == 3).all()
    assert update.meta_info["num_mini_batches"] == N_PROMPTS // 2
    assert metrics["self_imitation/gate_open_frac"] == 1.0
    # K_o 2, K_c 0 of M = N = 2: T = 2 opens only on a 2:0 split, 1 of C(4, 2) = 6.
    assert metrics["self_imitation/gate_null_frac"] == pytest.approx(1 / 6)
    assert metrics["self_imitation/rescued_all_wrong_frac"] == 1.0
    # A = coef * (1 - p_cur) = 1 on replay rows, times the N_all / N_cur denominator fix.
    scale = metrics["self_imitation/adv_scale"]
    assert scale == pytest.approx((N + M) / N)
    torch.testing.assert_close(
        update.batch["advantages"][kinds == ROW_REPLAY], torch.full((N_PROMPTS * M, RESP_LEN), scale)
    )
    assert (update.batch["rollout_is_weights"][kinds == ROW_REPLAY] == 1).all()
    # Data metrics: current rows only (all wrong, p_cur = 0 -> zero advantage), not rescaled.
    assert metrics["critic/advantages/max"] == 0 and metrics["critic/score/mean"] == 0

    # Step 4: replay sampled at step 2 is all wrong -> nothing stored, gate closed (p_old = p_cur = 0).
    update, metrics = workers.updates[3], all_metrics[3]
    assert len(update) == N_PROMPTS * N and "num_mini_batches" not in update.meta_info
    assert metrics["self_imitation/gate_open_frac"] == 0.0


def test_checkpoint_saves_the_trained_batch_position_not_the_prefetched_one(trainer, tmp_path):
    from torchdata.stateful_dataloader import StatefulDataLoader

    trainer.train_dataloader = StatefulDataLoader(list(range(12)), batch_size=2)
    trainer.config = OmegaConf.merge(
        trainer.config,
        {"trainer": {"total_epochs": 1, "default_local_dir": str(tmp_path), "default_hdfs_dir": None}},
    )
    trainer.actor_rollout_wg.save_checkpoint = lambda *args, **kwargs: None
    trainer.use_critic = False
    trainer.global_steps = 2

    schedule = lagged_schedule(trainer._batch_stream(), LAG)
    for _ in range(2):  # train on batches 0 and 1; the dataloader has already yielded batch 3
        (_, trainer._trained_dataloader_state), _ = next(schedule)
    trainer._save_checkpoint()

    resumed = StatefulDataLoader(list(range(12)), batch_size=2)
    resumed.load_state_dict(torch.load(tmp_path / "global_step_2" / "data.pt", weights_only=False))
    assert next(iter(resumed)).tolist() == [4, 5]  # batch 2: nothing skipped
    # The live dataloader is left as it was.
    assert "state_dict" not in vars(trainer.train_dataloader)
    assert next(schedule)[0][0].tolist() == [4, 5]


class _ListLoader:
    """A dataloader stand-in: fixed batches, a state_dict, nothing else."""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def state_dict(self):
        return {}


@pytest.mark.parametrize("save_freq, expected_saves", [(3, [3, 4]), (2, [2, 4]), (0, [])])
def test_fit_saves_the_last_step_once_when_data_runs_out(trainer, monkeypatch, save_freq, expected_saves):
    import verl.utils.tracking

    logged_steps, saved_steps = [], []
    monkeypatch.setattr(
        verl.utils.tracking,
        "Tracking",
        lambda **kwargs: SimpleNamespace(log=lambda data, step: logged_steps.append(step)),
    )
    trainer.config = OmegaConf.merge(
        trainer.config,
        {
            "trainer": {
                "project_name": "p",
                "experiment_name": "e",
                "logger": ["console"],
                "total_epochs": 1,
                "save_freq": save_freq,
                "test_freq": 0,
                "val_before_train": False,
            }
        },
    )
    # 4 batches of data, but 10 steps requested: the loop ends on data exhaustion, not is_last_step.
    trainer.train_dataloader = _ListLoader([batch_dict(i) for i in range(4)])
    trainer.total_training_steps = 10
    trainer.val_reward_fn = None
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 2)
    trainer._load_checkpoint = lambda: None
    trainer._save_checkpoint = lambda: saved_steps.append(trainer.global_steps)

    trainer.fit()

    assert saved_steps == expected_saves
    assert logged_steps == [1, 2, 3, 4]  # one log per step, none repeated by the tail save
