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

"""Forgetting-gated self-imitation: the batch-building logic, kept free of Ray and workers so it
can be unit-tested on CPU. The trainer (`self_imitation_trainer.py`) only moves data around.

Every row of a training batch is one of three kinds:

    ROW_CUR     a rollout of the current policy pi_prox -> plain Dr. GRPO, A = r - p_cur
    ROW_REPLAY  a correct rollout of pi_lag (the policy `lag_steps` updates ago) on a prompt whose
                pass rate dropped by at least `gate_tau`        -> A = coef * (r - p_cur)_+
    ROW_PAD     a copy of a short current row with response_mask zeroed, only there to make the row
                count divisible by (dp world size * number of mini-batches). Carries no gradient.

Both kinds of trained rows share the baseline p_cur, the current policy's pass rate on the prompt
(see RL_Idea_与过去的自己比较的策略优化.md, section 3).
"""

from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from typing import TypeVar

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights

ROW_CUR, ROW_REPLAY, ROW_PAD = 0, 1, 2

# Everything update_actor / compute_log_prob / the metrics need. Current and replay rows must carry
# the same keys, because DataProto.concat cats TensorDicts key by key.
_TRAIN_KEYS = (
    "prompts",
    "responses",
    "input_ids",
    "attention_mask",
    "position_ids",
    "rollout_log_probs",
    "token_level_scores",
    "token_level_rewards",
)

# p_old - p_cur is a difference of two k/n fractions; absorb float rounding at the threshold.
_GATE_EPS = 1e-6

PREFIX = "self_imitation"

T = TypeVar("T")


def lagged_schedule(batches: Iterable[T], lag: int) -> Iterator[tuple[T, T | None]]:
    """Yield (cur, ahead) per step: train on `cur`, and sample replay for `ahead` with the same weights.

    `ahead` is the batch `lag` steps later (None once the data runs out), so the replay a step
    trains on was sampled by the policy `lag` updates earlier. With lag=0, ahead is cur itself. The
    first `lag` steps train on batches whose replay was never generated.
    """
    batches = iter(batches)
    pending = deque()
    while len(pending) < lag and (item := next(batches, None)) is not None:
        pending.append(item)
    while True:
        ahead = next(batches, None)
        if ahead is not None:
            pending.append(ahead)
        if not pending:
            return
        yield pending.popleft(), ahead


def seq_scores(batch: DataProto) -> np.ndarray:
    return batch.batch["token_level_scores"].sum(-1).float().numpy()


def group_mean(uids: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    sums, counts = defaultdict(float), defaultdict(int)
    for uid, score in zip(uids, scores, strict=True):
        sums[uid] += float(score)
        counts[uid] += 1
    return {uid: sums[uid] / counts[uid] for uid in sums}


def train_view(batch: DataProto, kind: int) -> DataProto:
    """The training-relevant slice of a scored rollout batch, tagged with its row kind."""
    keys = [k for k in _TRAIN_KEYS if k in batch.batch.keys()]
    view = batch.select(batch_keys=keys, non_tensor_batch_keys=["uid"])
    view.meta_info = {}
    view.non_tensor_batch["row_kind"] = np.full(len(view), kind, dtype=np.int64)
    return view


def stash_replay(batch: DataProto) -> tuple[DataProto | None, dict[str, float]]:
    """Reduce a scored pi_lag rollout to what a later step can use.

    Returns the per-prompt pass rate p_old (from all rows) and only the rows with r > 0: with
    rewards in [0, 1] and p_cur >= 0, (r - p_cur)_+ is zero for every other row, so they would
    never be trained on and are not worth holding for `lag_steps` steps.
    """
    scores = seq_scores(batch)
    p_old = group_mean(batch.non_tensor_batch["uid"], scores)
    keep = np.flatnonzero(scores > 0)
    rows = train_view(batch[keep], ROW_REPLAY) if keep.size else None
    return rows, p_old


def gate_open(p_cur: dict[str, float], p_old: dict[str, float], tau: float) -> dict[str, bool]:
    """Forgetting gate: replay a prompt only if its pass rate dropped by at least tau."""
    return {uid: uid in p_old and p_old[uid] - p_cur[uid] >= tau - _GATE_EPS for uid in p_cur}


def build_train_batch(
    cur: DataProto,
    replay: DataProto | None,
    p_old: dict[str, float] | None,
    *,
    tau: float,
    coef: float,
    row_divisor: int,
) -> tuple[DataProto, int, dict[str, float]]:
    """Assemble current rows, admitted replay rows and padding into one batch.

    `cur` and `replay` are `train_view`s. `p_old` is None during the first `lag_steps` steps (no
    replay was generated for those prompts yet). Sets `seq_advantages`; returns the batch, the number
    of admitted replay rows, and metrics.
    """
    cur_uids = cur.non_tensor_batch["uid"]
    cur_scores = seq_scores(cur)
    p_cur = group_mean(cur_uids, cur_scores)
    cur_adv = cur_scores - np.array([p_cur[u] for u in cur_uids])

    pc = np.array(list(p_cur.values()))
    metrics = {
        f"{PREFIX}/cur_all_wrong_frac": float(np.mean(pc == 0)),
        f"{PREFIX}/cur_all_correct_frac": float(np.mean(pc == 1)),
    }
    parts, advs = [cur], [cur_adv]

    if p_old is not None:
        # Both come from the same _PromptBatch; a mismatch means replay was paired with the wrong step.
        assert p_old.keys() == p_cur.keys(), "replay pass rates belong to a different prompt batch"
        gate = gate_open(p_cur, p_old, tau)
        deltas = np.array([p_old[u] - p_cur[u] for u in p_cur])
        metrics.update(
            {
                f"{PREFIX}/gate_open_frac": float(np.mean(list(gate.values()))),
                f"{PREFIX}/delta_mean": float(deltas.mean()),
                f"{PREFIX}/delta_q95": float(np.quantile(deltas, 0.95)),
                f"{PREFIX}/p_old_mean": float(np.mean(list(p_old.values()))),
            }
        )
        # Logged whenever the gate was evaluated, so "nothing rescued" reads 0 rather than missing.
        rescued = set()
        if replay is not None and coef > 0:
            rep_uids = replay.non_tensor_batch["uid"]
            rep_scores = seq_scores(replay)
            rep_adv = np.array(
                [
                    coef * max(r - p_cur[u], 0.0) if gate.get(u, False) else 0.0
                    for u, r in zip(rep_uids, rep_scores, strict=True)
                ]
            )
            keep = np.flatnonzero(rep_adv > 0)
            if keep.size:
                parts.append(replay[keep])
                advs.append(rep_adv[keep])
                rescued = {u for u in rep_uids[keep] if p_cur[u] == 0}
        n_all_wrong = int(np.sum(pc == 0))
        metrics[f"{PREFIX}/rescued_all_wrong_frac"] = len(rescued) / n_all_wrong if n_all_wrong else 0.0

    n_replay = sum(len(p) for p in parts[1:])
    # With no replay rows the batch is exactly the GRPO batch; the trainer then leaves the
    # mini-batching alone, so coef=0 reproduces plain Dr. GRPO bit for bit.
    n_pad = (-(len(cur) + n_replay)) % row_divisor if n_replay else 0
    if n_pad:
        # The shortest current rows: padding costs a forward/backward pass, so make it cheap.
        order = torch.argsort(cur.batch["attention_mask"].sum(-1)).numpy()
        pad = cur[order[np.arange(n_pad) % len(cur)]]
        pad.non_tensor_batch["row_kind"] = np.full(n_pad, ROW_PAD, dtype=np.int64)
        parts.append(pad)
        advs.append(np.zeros(n_pad))

    metrics[f"{PREFIX}/replay_rows"] = n_replay
    metrics[f"{PREFIX}/pad_rows"] = n_pad

    batch = DataProto.concat(parts)
    batch.batch["seq_advantages"] = torch.from_numpy(np.concatenate(advs)).float()
    return batch, n_replay, metrics


def row_kinds(batch: DataProto) -> torch.Tensor:
    return torch.from_numpy(np.asarray(batch.non_tensor_batch["row_kind"], dtype=np.int64))


def mask_padding(batch: DataProto) -> None:
    """Zero the response mask of padding rows: no loss, no token-mean denominator, no metrics."""
    is_pad = (row_kinds(batch) == ROW_PAD)[:, None]
    batch.batch["response_mask"] = batch.batch["response_mask"] * (~is_pad).to(batch.batch["response_mask"].dtype)


def current_rows_mask(batch: DataProto) -> torch.Tensor:
    """response_mask restricted to the current-policy rows."""
    mask = batch.batch["response_mask"]
    return mask * (row_kinds(batch) == ROW_CUR)[:, None].to(mask.dtype)


def token_advantages(batch: DataProto) -> tuple[torch.Tensor, float]:
    """Broadcast `seq_advantages` over response tokens.

    token-mean divides by every token in the batch, so the admitted replay rows would shrink the
    GRPO term by N_cur / N_all -- a step-to-step varying learning-rate cut that reaches the update
    through grad_clip and Adam's moments. Rescaling by N_all / N_cur makes the denominator the
    current-policy token count: replay becomes a purely additive term (coef is its weight).
    """
    mask = batch.batch["response_mask"]
    tokens = mask.sum(-1).float()
    n_cur = tokens[row_kinds(batch) == ROW_CUR].sum()
    scale = (tokens.sum() / n_cur).item()
    adv = batch.batch["seq_advantages"][:, None] * mask.to(torch.float32) * scale
    return adv, scale


def current_rows_unscaled(batch: DataProto, scale: float) -> DataProto:
    """The current-policy rows with advantages / returns as GRPO computed them, i.e. without the
    N_all / N_cur denominator fix, so the data metrics stay comparable to a plain GRPO run."""
    rows = batch[np.flatnonzero(row_kinds(batch).numpy() == ROW_CUR)]
    for key in ("advantages", "returns"):
        rows.batch[key] = rows.batch[key] / scale
    return rows


def rollout_is_weights_current_only(batch: DataProto, algo: DictConfig) -> tuple[torch.Tensor, dict]:
    """IcePop / rollout IS on the current rows; weight 1 on replay rows.

    For a current row the ratio pi_prox / pi_rollout is the train-inference mismatch, which is what
    rollout IS is for. For a replay row it would be pi_prox / pi_lag, and forgotten tokens are exactly
    the ones where that is small -- masking or down-weighting them would undo the replay. Leaving
    them uncorrected is the deliberate bias of self-imitation (idea doc, section 3.5). The mismatch
    metrics are computed over current rows only, so they stay comparable to a plain GRPO run.
    """
    mask = batch.batch["response_mask"]
    is_cur = (row_kinds(batch) == ROW_CUR)[:, None]
    proto, metrics = compute_rollout_importance_weights(
        old_log_prob=batch.batch["old_log_probs"],
        rollout_log_prob=batch.batch["rollout_log_probs"],
        response_mask=current_rows_mask(batch),
        rollout_is_level=algo.rollout_is_level,
        rollout_is_mode=algo.rollout_is_mode,
        rollout_is_threshold=algo.rollout_is_threshold,
        rollout_is_threshold_lower=algo.rollout_is_threshold_lower,
        rollout_is_veto_threshold=algo.rollout_is_veto_threshold,
    )
    weights = proto.batch["rollout_is_weights"]
    weights = torch.where(is_cur, weights, mask.to(weights.dtype))
    return weights, metrics


def replay_drift_metrics(batch: DataProto, clip_ratio_low: float) -> dict[str, float]:
    """How far pi_prox has drifted from pi_lag on the replayed tokens.

    log(pi_prox / pi_lag) < 0 means the current policy no longer writes what it wrote `lag_steps`
    ago. "Forgotten" tokens are those below the PPO lower clip, 1 - clip_ratio_low. pi_lag's
    log-probs come from vLLM, so this includes train-inference mismatch.
    """
    kinds = row_kinds(batch)
    is_rep = kinds == ROW_REPLAY
    if not is_rep.any():
        return {}
    mask = batch.batch["response_mask"]
    rep_mask = mask[is_rep].bool()
    log_ratio = (batch.batch["old_log_probs"] - batch.batch["rollout_log_probs"])[is_rep][rep_mask]
    return {
        f"{PREFIX}/replay_log_ratio_mean": log_ratio.mean().item(),
        f"{PREFIX}/replay_forgotten_token_frac": (log_ratio < np.log(1 - clip_ratio_low)).float().mean().item(),
        f"{PREFIX}/replay_token_frac": (rep_mask.sum() / mask.sum()).item(),
    }


def validate_self_imitation_config(config: DictConfig) -> None:
    """Configs that would make the method silently not be there. Raise before any worker starts."""
    si = config.self_imitation
    if si.lag_steps < 0 or si.num_replay < 0 or si.replay_coef < 0:
        raise ValueError(f"self_imitation.{{lag_steps,num_replay,replay_coef}} must be >= 0, got {dict(si)}")
    # tau <= 0 is legitimate (the ungated ablation); tau > 1 can never be reached by p_old - p_cur.
    if si.gate_tau > 1:
        raise ValueError(f"self_imitation.gate_tau={si.gate_tau} > 1: the forgetting gate could never open.")
    total_steps = config.trainer.get("total_training_steps")
    if si.num_replay > 0 and total_steps is not None and si.lag_steps >= total_steps:
        raise ValueError(
            f"self_imitation.lag_steps={si.lag_steps} >= trainer.total_training_steps={total_steps}: no step "
            "would ever receive replay. Lower lag_steps, or set num_replay=0 for plain Dr. GRPO."
        )
    if si.num_replay > 0 and not config.actor_rollout_ref.rollout.get("calculate_log_probs", False):
        raise ValueError(
            "self_imitation needs pi_lag's log-probs from the rollout for its replay drift metrics "
            "(replay_forgotten_token_frac): set actor_rollout_ref.rollout.calculate_log_probs=True."
        )
    if OmegaConf.select(config, "critic.enable"):
        raise ValueError("self_imitation uses group baselines (p_cur), not a critic: set critic.enable=False.")
    if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
        raise NotImplementedError(
            "self_imitation pins the number of mini-batches through meta_info['num_mini_batches'], which only "
            "the FSDP actor (dp_actor.py) reads; megatron would split the variable-size batch by row count."
        )
    if config.actor_rollout_ref.rollout.mode != "sync":
        raise NotImplementedError("self_imitation generates current and replay rollouts in one sync rollout call.")
    if config.algorithm.use_kl_in_reward:
        raise NotImplementedError("use_kl_in_reward is not wired into the self_imitation trainer.")
    if config.reward_model.enable:
        raise NotImplementedError("self_imitation scores with the rule-based reward_fn only.")
    if si.replay_coef > 0 and OmegaConf.select(config, "reward_model.overlong_buffer.enable", default=False):
        raise ValueError(
            "The overlong penalty makes rewards negative, but replay keeps only r > 0 rows assuming rewards in "
            "[0, 1]. Disable reward_model.overlong_buffer, or set self_imitation.replay_coef=0."
        )
    if config.trainer.get("rollout_data_dir"):
        raise NotImplementedError("trainer.rollout_data_dir is not supported by the self_imitation trainer.")
    if config.data.train_batch_size % config.actor_rollout_ref.actor.ppo_mini_batch_size:
        raise ValueError("data.train_batch_size must be a multiple of actor.ppo_mini_batch_size.")
    # generate_sequences chunks its input over the rollout data-parallel ranks without padding. A step
    # generates bsz * (n + m) rows, or bsz * n once nothing is left to sample replay for.
    rollout_dp = (
        config.trainer.n_gpus_per_node * config.trainer.nnodes
    ) // config.actor_rollout_ref.rollout.tensor_model_parallel_size
    bsz, n = config.data.train_batch_size, config.actor_rollout_ref.rollout.n
    for rows in {bsz * n, bsz * (n + si.num_replay)}:
        if rows % rollout_dp:
            raise ValueError(
                f"A rollout call of {rows} rows (data.train_batch_size={bsz}, rollout.n={n}, "
                f"self_imitation.num_replay={si.num_replay}) does not split over {rollout_dp} rollout DP ranks."
            )
