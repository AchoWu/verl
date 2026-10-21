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

"""CPU tests for recipe/self_imitation/replay.py (forgetting-gated self-imitation)."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from recipe.self_imitation.replay import (
    ROW_CUR,
    ROW_PAD,
    ROW_REPLAY,
    build_train_batch,
    current_rows_unscaled,
    gate_open,
    group_mean,
    lagged_schedule,
    mask_padding,
    rollout_is_weights_current_only,
    row_kinds,
    stash_replay,
    token_advantages,
    train_view,
    validate_self_imitation_config,
)
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_response_mask

PROMPT_LEN, RESP_LEN = 3, 6


def make_rollout(uids, scores, resp_lens=None, kind=ROW_CUR):
    """A scored rollout batch in the layout the vLLM rollout + reward_fn produce."""
    bsz = len(uids)
    resp_lens = resp_lens or [RESP_LEN] * bsz
    attention_mask = torch.zeros(bsz, PROMPT_LEN + RESP_LEN, dtype=torch.long)
    attention_mask[:, :PROMPT_LEN] = 1
    scores_tok = torch.zeros(bsz, RESP_LEN)
    for i, length in enumerate(resp_lens):
        attention_mask[i, PROMPT_LEN : PROMPT_LEN + length] = 1
        scores_tok[i, length - 1] = scores[i]
    responses = torch.randint(0, 100, (bsz, RESP_LEN))
    batch = DataProto.from_dict(
        tensors={
            "prompts": torch.randint(0, 100, (bsz, PROMPT_LEN)),
            "responses": responses,
            "input_ids": torch.randint(0, 100, (bsz, PROMPT_LEN + RESP_LEN)),
            "attention_mask": attention_mask,
            "position_ids": torch.arange(PROMPT_LEN + RESP_LEN).expand(bsz, -1).clone(),
            "rollout_log_probs": -torch.rand(bsz, RESP_LEN),
            "token_level_scores": scores_tok,
            "token_level_rewards": scores_tok.clone(),
        },
        non_tensors={"uid": np.array(uids, dtype=object), "data_source": np.array(["x"] * bsz, dtype=object)},
    )
    return train_view(batch, kind)


def test_group_mean_and_gate_threshold_is_inclusive():
    p = group_mean(np.array(["a", "a", "b", "b"]), np.array([1.0, 0.0, 1.0, 1.0]))
    assert p == {"a": 0.5, "b": 1.0}
    # 5/8 - 2/8 == 3/8 must open a tau=3/8 gate; thirds do not land exactly on the threshold.
    assert gate_open({"a": 2 / 8}, {"a": 5 / 8}, 0.375) == {"a": True}
    assert gate_open({"a": 1 / 3}, {"a": 2 / 3}, 1 / 3) == {"a": True}
    assert gate_open({"a": 3 / 8}, {"a": 5 / 8}, 0.375) == {"a": False}
    # A prompt without replay (first lag steps) never opens.
    assert gate_open({"a": 0.0}, {}, 0.0) == {"a": False}


def test_stash_replay_keeps_only_correct_rows_but_pass_rate_uses_all():
    rep = make_rollout(["a"] * 4 + ["b"] * 4, [1, 0, 0, 1, 0, 0, 0, 0], kind=ROW_REPLAY)
    rows, p_old = stash_replay(rep)
    assert p_old == {"a": 0.5, "b": 0.0}
    assert len(rows) == 2
    assert list(rows.non_tensor_batch["uid"]) == ["a", "a"]
    assert (row_kinds(rows) == ROW_REPLAY).all()

    rows, _ = stash_replay(make_rollout(["a", "a"], [0, 0], kind=ROW_REPLAY))
    assert rows is None


def test_no_replay_is_exactly_dr_grpo():
    cur = make_rollout(["a"] * 4 + ["b"] * 4, [1, 1, 0, 0, 1, 1, 1, 1])
    for p_old, coef in [(None, 1.0), ({"a": 1.0, "b": 1.0}, 0.0)]:
        replay, _ = stash_replay(make_rollout(["a"] * 4, [1, 1, 1, 1], kind=ROW_REPLAY))
        batch, n_replay, metrics = build_train_batch(cur, replay, p_old, tau=0.375, coef=coef, row_divisor=16)
        assert n_replay == 0 and len(batch) == 8 and metrics["self_imitation/pad_rows"] == 0
        torch.testing.assert_close(batch.batch["seq_advantages"], torch.tensor([0.5, 0.5, -0.5, -0.5, 0, 0, 0, 0]))


def test_rescued_metric_is_logged_as_zero_when_the_gate_stays_shut():
    cur = make_rollout(["a"] * 2, [0, 0])
    replay, _ = stash_replay(make_rollout(["a"] * 2, [1, 0], kind=ROW_REPLAY))
    _, n_replay, metrics = build_train_batch(cur, replay, {"a": 0.25}, tau=0.375, coef=1.0, row_divisor=1)
    assert n_replay == 0
    assert metrics["self_imitation/gate_open_frac"] == 0.0
    assert metrics["self_imitation/rescued_all_wrong_frac"] == 0.0


def test_replay_from_another_batch_is_rejected():
    cur = make_rollout(["a"] * 2, [1, 0])
    with pytest.raises(AssertionError, match="different prompt batch"):
        build_train_batch(cur, None, {"z": 1.0}, tau=0.375, coef=1.0, row_divisor=1)


def test_replay_admitted_only_on_forgotten_prompts_with_positive_advantage():
    # a: p_cur 1/4, p_old 3/4 -> forgotten.  b: p_cur 3/4, p_old 1 -> drop of 1/4 < tau.
    cur = make_rollout(["a"] * 4 + ["b"] * 4, [1, 0, 0, 0, 1, 1, 1, 0])
    replay, _ = stash_replay(make_rollout(["a"] * 4 + ["b"] * 4, [1, 1, 1, 0, 1, 1, 1, 1], kind=ROW_REPLAY))
    p_old = {"a": 0.75, "b": 1.0}
    batch, n_replay, metrics = build_train_batch(cur, replay, p_old, tau=0.375, coef=0.5, row_divisor=4)

    kinds = row_kinds(batch)
    assert n_replay == 3
    assert (kinds == ROW_REPLAY).sum() == 3 and (kinds == ROW_PAD).sum() == 1  # 8 + 3 -> pad to 12
    assert set(batch.non_tensor_batch["uid"][kinds.numpy() == ROW_REPLAY]) == {"a"}
    torch.testing.assert_close(batch.batch["seq_advantages"][kinds == ROW_REPLAY], torch.full((3,), 0.5 * 0.75))
    assert batch.batch["seq_advantages"][kinds == ROW_PAD].abs().sum() == 0
    assert metrics["self_imitation/gate_open_frac"] == 0.5
    assert metrics["self_imitation/delta_mean"] == pytest.approx((0.5 + 0.25) / 2)


def test_padding_uses_shortest_rows_and_is_masked_out():
    cur = make_rollout(["a"] * 4, [1, 0, 0, 0], resp_lens=[6, 2, 5, 4])
    replay, _ = stash_replay(make_rollout(["a"] * 4, [1, 1, 1, 1], kind=ROW_REPLAY))
    batch, _, _ = build_train_batch(cur, replay, {"a": 1.0}, tau=0.375, coef=1.0, row_divisor=5)
    batch.batch["response_mask"] = compute_response_mask(batch)
    mask_padding(batch)

    is_pad = row_kinds(batch) == ROW_PAD
    assert is_pad.sum() == 2  # 4 + 4 -> 10
    assert batch.batch["response_mask"][is_pad].sum() == 0
    assert batch.batch["response_mask"][~is_pad].sum() > 0
    # Padding copies the two shortest current rows (response lengths 2 and 4).
    assert sorted(batch.batch["attention_mask"][is_pad].sum(-1).tolist()) == [PROMPT_LEN + 2, PROMPT_LEN + 4]


def test_token_advantages_make_the_denominator_the_current_token_count():
    cur = make_rollout(["a"] * 4, [1, 0, 0, 0], resp_lens=[6, 2, 5, 4])
    replay, _ = stash_replay(make_rollout(["a"] * 2, [1, 1], resp_lens=[3, 6], kind=ROW_REPLAY))
    batch, _, _ = build_train_batch(cur, replay, {"a": 1.0}, tau=0.375, coef=1.0, row_divisor=2)
    batch.batch["response_mask"] = compute_response_mask(batch)
    mask_padding(batch)

    adv, scale = token_advantages(batch)
    mask = batch.batch["response_mask"].float()
    n_cur = 6 + 2 + 5 + 4
    assert scale == pytest.approx((n_cur + 3 + 6) / n_cur)
    # token-mean over the whole batch == sum over rows / current tokens only.
    expected = (batch.batch["seq_advantages"][:, None] * mask).sum() / n_cur
    torch.testing.assert_close((adv * mask).sum() / mask.sum(), expected)

    # Data metrics see the current rows with the GRPO advantages, not the rescaled ones.
    batch.batch["advantages"] = adv
    batch.batch["returns"] = adv
    rows = current_rows_unscaled(batch, scale)
    assert (row_kinds(rows) == ROW_CUR).all() and len(rows) == 4
    torch.testing.assert_close(
        rows.batch["advantages"], rows.batch["seq_advantages"][:, None] * rows.batch["response_mask"]
    )
    torch.testing.assert_close(batch.batch["advantages"], adv)  # the training batch is untouched


def test_rollout_is_applies_to_current_rows_only():
    cur = make_rollout(["a"] * 2, [1, 0])
    replay, _ = stash_replay(make_rollout(["a"] * 2, [1, 1], kind=ROW_REPLAY))
    batch, _, _ = build_train_batch(cur, replay, {"a": 1.0}, tau=0.375, coef=1.0, row_divisor=1)
    batch.batch["response_mask"] = compute_response_mask(batch)
    # Every token 100x more likely under pi_rollout than pi_prox: IcePop [0.5, 5] masks them all.
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"] - np.log(100.0)
    algo = OmegaConf.create(
        {
            "rollout_is_level": "token",
            "rollout_is_mode": "mask",
            "rollout_is_threshold": 5.0,
            "rollout_is_threshold_lower": 0.5,
            "rollout_is_veto_threshold": None,
        }
    )
    weights, metrics = rollout_is_weights_current_only(batch, algo)
    kinds = row_kinds(batch)
    assert weights[kinds == ROW_CUR].sum() == 0
    torch.testing.assert_close(weights[kinds == ROW_REPLAY], batch.batch["response_mask"][kinds == ROW_REPLAY].float())
    assert metrics["mismatch/rollout_is_masked_fraction"] == pytest.approx(1.0)


def _config(**overrides):
    base = {
        "self_imitation": {"lag_steps": 16, "num_replay": 8, "replay_coef": 1.0, "gate_tau": 0.375},
        "critic": {"enable": False},
        "actor_rollout_ref": {
            "actor": {"strategy": "fsdp", "ppo_mini_batch_size": 16},
            "rollout": {"mode": "sync", "calculate_log_probs": True, "n": 8, "tensor_model_parallel_size": 1},
        },
        "algorithm": {"use_kl_in_reward": False},
        "reward_model": {"enable": False, "overlong_buffer": {"enable": False}},
        "trainer": {"rollout_data_dir": None, "total_training_steps": 1000, "n_gpus_per_node": 8, "nnodes": 1},
        "data": {"train_batch_size": 64},
    }
    config = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(config, key, value)
    return config


def test_validate_config_accepts_the_recipe_defaults():
    validate_self_imitation_config(_config())
    validate_self_imitation_config(
        _config(**{"self_imitation.replay_coef": 0.0, "reward_model.overlong_buffer.enable": True})
    )
    validate_self_imitation_config(
        _config(**{"self_imitation.num_replay": 0, "actor_rollout_ref.rollout.calculate_log_probs": False})
    )
    # tau <= 0 is the ungated ablation; lag_steps is irrelevant without replay; unset step counts pass.
    validate_self_imitation_config(_config(**{"self_imitation.gate_tau": -1.0}))
    validate_self_imitation_config(_config(**{"self_imitation.num_replay": 0, "self_imitation.lag_steps": 5000}))
    validate_self_imitation_config(_config(**{"trainer.total_training_steps": None}))
    # 64 prompts on 8 ranks split for any m.
    validate_self_imitation_config(_config(**{"self_imitation.num_replay": 3}))


@pytest.mark.parametrize(
    "overrides",
    [
        {"critic.enable": True},
        {"actor_rollout_ref.actor.strategy": "megatron"},
        {"actor_rollout_ref.rollout.mode": "async"},
        {"actor_rollout_ref.rollout.calculate_log_probs": False},
        {"reward_model.overlong_buffer.enable": True},
        {"self_imitation.lag_steps": -1},
        {"data.train_batch_size": 60},
        {"trainer.rollout_data_dir": "/tmp/x"},
        {"self_imitation.gate_tau": 1.5},
        {"self_imitation.lag_steps": 1000},
        # 4 x 8 = 32 rows split over 8 ranks, but 4 x (8 + 1) = 36 do not.
        {"data.train_batch_size": 4, "actor_rollout_ref.actor.ppo_mini_batch_size": 4, "self_imitation.num_replay": 1},
    ],
)
def test_validate_config_rejects_silent_failures(overrides):
    with pytest.raises((ValueError, NotImplementedError)):
        validate_self_imitation_config(_config(**overrides))


@pytest.mark.parametrize("lag", [0, 2])
def test_lagged_schedule(lag):
    pairs = list(lagged_schedule(range(5), lag))
    assert [cur for cur, _ in pairs] == [0, 1, 2, 3, 4]
    assert [ahead for _, ahead in pairs] == [i + lag if i + lag < 5 else None for i in range(5)]
    assert list(lagged_schedule(range(1), 3)) == [(0, None)]
