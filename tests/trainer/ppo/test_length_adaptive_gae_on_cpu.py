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
"""Tests for the JustRL-II length-adaptive decoupled GAE estimator."""

import pytest
import torch

from verl.trainer.ppo.core_algos import (
    compute_length_adaptive_gae_advantage_return,
    length_adaptive_lambda,
)


class _Cfg:
    def __init__(self, k=0.5):
        self.gae_lambda_k = k


def _batch(lengths, terminal_rewards, width=None, value_fill=0.0):
    """Build (rewards, values, mask) with the terminal reward on the last valid token."""
    width = width or max(lengths)
    bs = len(lengths)
    mask = torch.zeros(bs, width)
    rewards = torch.zeros(bs, width)
    for i, (length, reward) in enumerate(zip(lengths, terminal_rewards, strict=True)):
        mask[i, :length] = 1
        rewards[i, length - 1] = reward
    values = torch.full((bs, width), value_fill) * mask
    return rewards, values, mask


def _raw_gae(rewards, values, mask, gamma, lam):
    """GAE without the masked_whiten that `compute_gae_advantage_return` applies."""
    nextvalues, lastgaelam, out = 0, 0, []
    for t in reversed(range(rewards.shape[-1])):
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam_ = delta + gamma * lam * lastgaelam
        nextvalues = values[:, t] * mask[:, t] + (1 - mask[:, t]) * nextvalues
        lastgaelam = lastgaelam_ * mask[:, t] + (1 - mask[:, t]) * lastgaelam
        out.append(lastgaelam)
    return torch.stack(out[::-1], dim=1)


class TestLengthAdaptiveLambda:
    @pytest.mark.parametrize("length", [1, 16, 256, 2048, 14336, 131072])
    @pytest.mark.parametrize("k", [0.1, 0.5, 0.9])
    def test_terminal_credit_is_length_independent(self, length, k):
        """The defining property: lambda_i ** L_i == k for every length.

        The tolerance scales with length because exponentiating by L amplifies lambda's fp32
        representation error L-fold: at L=131072, lambda = 1 - 5.3e-6 carries ~6e-8 of relative
        error, so lambda ** L lands a few per mil off k. That is the accuracy floor of fp32 and
        is harmless (bf16, by contrast, rounds lambda to exactly 1.0 and loses the mechanism
        entirely -- see `test_computed_in_fp32`).
        """
        lam = length_adaptive_lambda(k, torch.tensor([length])).item()
        assert lam**length == pytest.approx(k, rel=max(1e-3, length * 1e-7))

    def test_longer_responses_get_lambda_closer_to_one(self):
        lam = length_adaptive_lambda(0.5, torch.tensor([16, 256, 2048, 14336]))
        assert torch.all(lam[1:] > lam[:-1])
        assert torch.all(lam < 1.0)

    def test_computed_in_fp32(self):
        """bf16 rounds a 14k-length lambda to exactly 1.0, which erases the mechanism."""
        lam = length_adaptive_lambda(0.5, torch.tensor([14336]))
        assert lam.dtype == torch.float32
        assert lam.item() < 1.0
        assert lam.to(torch.bfloat16).item() == 1.0

    def test_empty_response_gives_zero(self):
        """L == 0 rows are fully masked downstream, so the value only has to be finite."""
        lam = length_adaptive_lambda(0.5, torch.tensor([0, 10]))
        assert lam[0].item() == 0.0
        assert lam[1].item() > 0.0

    @pytest.mark.parametrize("k", [0.0, -0.1, 1.5])
    def test_k_out_of_range_raises(self, k):
        with pytest.raises(ValueError, match="gae_lambda_k"):
            length_adaptive_lambda(k, torch.tensor([10]))


class TestDtypeHandling:
    """The estimator must not inherit the batch dtype for its lambda or its scans.

    The critic's values arrive from an autocast(bf16) forward, and bf16 rounds
    `lambda = k ** (1/L)` to exactly 1.0 for every L >= 512. Casting lambda to the batch dtype
    therefore degrades this estimator into plain lambda=1 GAE with no error and no metric --
    which is exactly the bug these tests exist to prevent recurring.
    """

    @pytest.mark.parametrize("length", [128, 512, 2048, 14336])
    def test_bf16_values_match_fp32(self, length):
        rewards, _, mask = _batch([length], [1.0])
        values = torch.full((1, length), 0.52) * mask

        adv32, ret32 = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards,
            values=values,
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
        )
        adv_bf, ret_bf = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards.to(torch.bfloat16),
            values=values.to(torch.bfloat16),
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
        )
        # Only input quantization should differ; the lambda and the scan stay in fp32.
        torch.testing.assert_close(adv32, adv_bf, atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(ret32, ret_bf, atol=1e-4, rtol=1e-3)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_outputs_are_fp32(self, dtype):
        rewards, _, mask = _batch([2048], [1.0])
        advantages, returns = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards.to(dtype),
            values=(torch.full((1, 2048), 0.52) * mask).to(dtype),
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
        )
        assert advantages.dtype == torch.float32
        assert returns.dtype == torch.float32

    def test_bf16_values_do_not_collapse_the_advantage(self):
        """The regression itself: bf16 lambda made every advantage exactly zero at L=14336."""
        length = 14336
        rewards, _, mask = _batch([length], [1.0])
        advantages, _ = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards.to(torch.bfloat16),
            values=(torch.full((1, length), 0.52) * mask).to(torch.bfloat16),
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
        )
        assert advantages.abs().max().item() > 0.1

    def test_bf16_returns_keep_the_terminal_reward(self):
        """A 14k-element cumsum in bf16 would round the single terminal reward away."""
        length = 14336
        rewards, _, mask = _batch([length], [1.0])
        _, returns = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards.to(torch.bfloat16),
            values=(torch.full((1, length), 0.52) * mask).to(torch.bfloat16),
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
        )
        torch.testing.assert_close(returns[0, 0], torch.tensor(1.0))
        torch.testing.assert_close(returns[0, -1], torch.tensor(1.0))

    @pytest.mark.parametrize("length", [512, 2048, 14336])
    def test_first_token_credit_survives_bf16_inputs(self, length):
        """End-to-end: with zero values the first token must still receive ~k, not ~1."""
        rewards, _, mask = _batch([length], [1.0])
        advantages, _ = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards.to(torch.bfloat16),
            values=torch.zeros(1, length, dtype=torch.bfloat16),
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
        )
        # masked_whiten makes the absolute scale meaningless, so compare the shape of the decay
        # instead: with lambda=1 every token would carry the same advantage.
        spread = advantages[0, 0].item() - advantages[0, -1].item()
        assert abs(spread) > 0.1, f"advantage is flat ({spread=}), lambda collapsed to 1.0"


class TestLengthAdaptiveGAE:
    def test_returns_are_lambda_one_suffix_sum(self):
        """With a single terminal reward, the lambda=1 return is that reward at every token."""
        lengths, terminal = [16, 256, 2048], [1.0, 0.0, 1.0]
        rewards, values, mask = _batch(lengths, terminal, value_fill=0.52)

        _, returns = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards,
            values=values,
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
        )

        for i, (length, reward) in enumerate(zip(lengths, terminal, strict=True)):
            row = returns[i, :length]
            torch.testing.assert_close(row, torch.full_like(row, reward))

    def test_returns_are_independent_of_values(self):
        """The critic target must not be `advantages + values` (that is the biased GAE return)."""
        rewards, _, mask = _batch([128, 512], [1.0, 0.0])
        _, returns_a = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards,
            values=torch.zeros_like(rewards),
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
        )
        _, returns_b = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards,
            values=torch.full_like(rewards, 0.9) * mask,
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
        )
        torch.testing.assert_close(returns_a, returns_b)

    def test_returns_zero_outside_mask(self):
        rewards, values, mask = _batch([16, 2048], [1.0, 1.0], value_fill=0.52)
        _, returns = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards,
            values=values,
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
        )
        assert torch.all(returns * (1 - mask) == 0)

    def test_critic_rewards_override_only_returns(self):
        """--critic-exclude-overlong-penalty: critic regresses r, actor still sees r - penalty."""
        # One correct-but-overlong sample: shaped 0.815, unshaped 1.0.
        rewards, values, mask = _batch([2048], [0.815], value_fill=0.52)
        critic_rewards = rewards.clone()
        critic_rewards[0, 2047] = 1.0

        adv_shaped, returns_shaped = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards,
            values=values,
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
        )
        adv_split, returns_split = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards,
            values=values,
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
            critic_token_level_rewards=critic_rewards,
        )

        # The critic's target loses the penalty ...
        assert returns_shaped[0, 0].item() == pytest.approx(0.815, abs=1e-5)
        assert returns_split[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        # ... while the actor's advantage is untouched.
        torch.testing.assert_close(adv_shaped, adv_split)

    def test_first_token_credit_matches_k(self):
        """End-to-end: with zero values the first token's raw advantage is ~k for every length."""
        lengths = [16, 256, 2048, 14336]
        rewards, values, mask = _batch(lengths, [1.0] * len(lengths))
        lam = length_adaptive_lambda(0.5, mask.sum(-1))
        raw = _raw_gae(rewards, values, mask, gamma=1.0, lam=lam)
        for i in range(len(lengths)):
            # lambda ** (L-1) rather than ** L, since the terminal token itself is the reward.
            assert raw[i, 0].item() == pytest.approx(0.5, rel=0.05)

    def test_per_sample_lambda_differs_from_scalar(self):
        """Guard against the (bs,) lambda silently collapsing to a scalar broadcast."""
        lengths = [16, 2048]
        rewards, values, mask = _batch(lengths, [1.0, 1.0], value_fill=0.3)
        lam_rowwise = length_adaptive_lambda(0.5, mask.sum(-1))
        adaptive = _raw_gae(rewards, values, mask, 1.0, lam_rowwise)
        uniform = _raw_gae(rewards, values, mask, 1.0, lam_rowwise[0])
        assert (adaptive[1] - uniform[1]).abs().max() > 1e-3

    def test_advantages_are_whitened(self):
        rewards, values, mask = _batch([64, 512, 2048], [1.0, 0.0, 1.0], value_fill=0.52)
        advantages, _ = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards,
            values=values,
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
        )
        selected = advantages[mask.bool()]
        assert selected.mean().abs().item() < 1e-4
        assert selected.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-3)

    def test_gamma_must_be_one(self):
        rewards, values, mask = _batch([32], [1.0])
        with pytest.raises(ValueError, match="gamma == 1.0"):
            compute_length_adaptive_gae_advantage_return(
                token_level_rewards=rewards,
                values=values,
                response_mask=mask,
                gamma=0.99,
                lam=1.0,
                config=_Cfg(),
            )

    @pytest.mark.parametrize("config", [None, _Cfg(k=None)])
    def test_missing_k_raises(self, config):
        rewards, values, mask = _batch([32], [1.0])
        with pytest.raises(ValueError, match="gae_lambda_k"):
            compute_length_adaptive_gae_advantage_return(
                token_level_rewards=rewards,
                values=values,
                response_mask=mask,
                gamma=1.0,
                lam=1.0,
                config=config,
            )

    def test_registered_under_its_name(self):
        from verl.trainer.ppo.core_algos import AdvantageEstimator, get_adv_estimator_fn

        assert get_adv_estimator_fn("gae_length_adaptive") is compute_length_adaptive_gae_advantage_return
        assert get_adv_estimator_fn(AdvantageEstimator.GAE_LENGTH_ADAPTIVE) is (
            compute_length_adaptive_gae_advantage_return
        )

    def test_tolerates_registry_dispatch_kwargs(self):
        """`compute_advantage` passes `index` to every registry-dispatched estimator.

        This estimator has no group baseline, so it must accept and ignore those extra keys
        rather than raising TypeError the first time a batch carries a `uid`.
        """
        import numpy as np

        rewards, values, mask = _batch([32, 64], [1.0, 0.0])
        advantages, returns = compute_length_adaptive_gae_advantage_return(
            token_level_rewards=rewards,
            values=values,
            response_mask=mask,
            gamma=1.0,
            lam=1.0,
            config=_Cfg(),
            index=np.array(["a", "a"], dtype=object),
        )
        assert advantages.shape == rewards.shape
        assert returns.shape == rewards.shape
