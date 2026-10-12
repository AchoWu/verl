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
"""Tests for JustRL-II critic value-head seeding (zero weight, bias = mean reward)."""

import pytest
import torch

from verl.utils.model import find_scalar_value_head, seed_scalar_value_head

HIDDEN = 256


class _TokenClassificationLike(torch.nn.Module):
    """`*ForTokenClassification` with num_labels=1 -- the FSDP critic's default class."""

    def __init__(self, bias=True):
        super().__init__()
        self.score = torch.nn.Linear(HIDDEN, 1, bias=bias)


class _TrlValueHeadLike(torch.nn.Module):
    """`AutoModelForCausalLMWithValueHead` -- the TRL fallback in `load_valuehead_model`."""

    def __init__(self):
        super().__init__()
        self.v_head = torch.nn.Module()
        self.v_head.summary = torch.nn.Linear(HIDDEN, 1)


class _LanguageModelLike(torch.nn.Module):
    """An actor: the head projects to the vocabulary, so there is nothing to seed."""

    def __init__(self):
        super().__init__()
        self.score = torch.nn.Linear(HIDDEN, 1024)


class TestFindScalarValueHead:
    @pytest.mark.parametrize("model_cls", [_TokenClassificationLike, _TrlValueHeadLike])
    def test_finds_both_critic_layouts(self, model_cls):
        head = find_scalar_value_head(model_cls())
        assert isinstance(head, torch.nn.Linear)
        assert head.out_features == 1

    def test_ignores_non_scalar_head(self):
        assert find_scalar_value_head(_LanguageModelLike()) is None

    def test_returns_none_without_a_head(self):
        assert find_scalar_value_head(torch.nn.Module()) is None


class TestSeedScalarValueHead:
    @pytest.mark.parametrize("model_cls", [_TokenClassificationLike, _TrlValueHeadLike])
    def test_value_is_constant_and_equals_bias(self, model_cls):
        """With a zero weight V == bias for every token, so the bias *is* the critic's prior."""
        model = model_cls()
        head = find_scalar_value_head(model)
        torch.nn.init.normal_(head.weight, mean=0.0, std=0.02)

        assert seed_scalar_value_head(model, 0.52)

        values = head(torch.randn(64, HIDDEN)).flatten()
        torch.testing.assert_close(values, torch.full_like(values, 0.52))
        assert head.weight.abs().max().item() == 0.0

    def test_replaces_the_default_init(self):
        """normal(0, 0.02) puts V far outside the [0, 1] range the value targets live in."""
        model = _TokenClassificationLike()
        head = find_scalar_value_head(model)
        torch.nn.init.normal_(head.weight, mean=0.0, std=0.02)
        hidden = torch.randn(512, HIDDEN)
        spread_before = head(hidden).flatten().std().item()

        seed_scalar_value_head(model, 0.52)
        spread_after = head(hidden).flatten().std().item()

        assert spread_before > 0.1
        assert spread_after == 0.0

    def test_creates_a_missing_bias(self):
        """mcore's `LinearForLastLayer` is built with bias=False; the prior needs one."""
        model = _TokenClassificationLike(bias=False)
        head = find_scalar_value_head(model)
        assert head.bias is None

        assert seed_scalar_value_head(model, 0.52)

        assert head.bias is not None
        values = head(torch.randn(8, HIDDEN)).flatten()
        torch.testing.assert_close(values, torch.full_like(values, 0.52))

    def test_gradient_still_flows_from_step_zero(self):
        """Zero weight does not freeze the head: dV/dw = h and dV/db = 1 are both alive."""
        model = _TokenClassificationLike()
        head = find_scalar_value_head(model)
        seed_scalar_value_head(model, 0.52)

        hidden = torch.randn(16, HIDDEN)
        head(hidden).sum().backward()

        assert head.weight.grad.abs().max().item() > 0
        assert head.bias.grad.abs().max().item() > 0

    @pytest.mark.parametrize("bias_init", [0.0, 0.52, 1.0])
    def test_honours_the_requested_bias(self, bias_init):
        model = _TokenClassificationLike()
        seed_scalar_value_head(model, bias_init)
        head = find_scalar_value_head(model)
        assert head.bias.item() == pytest.approx(bias_init)

    def test_no_op_without_a_scalar_head(self):
        """An actor module must be left alone rather than silently mangled."""
        model = _LanguageModelLike()
        before = model.score.weight.detach().clone()
        assert not seed_scalar_value_head(model, 0.52)
        torch.testing.assert_close(model.score.weight, before)

    def test_skips_meta_tensors(self):
        """Non-zero FSDP ranks build on meta and receive rank 0's weights via state sync."""
        with torch.device("meta"):
            model = _TokenClassificationLike()
        assert not seed_scalar_value_head(model, 0.52)
