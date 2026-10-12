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
"""The 0/1 reward wrapper that makes the critic's V a probability.

Without it, math_dapo's +/-1 reward makes V an affine image of P(correct) rather than the
probability itself, and the value-head seeding gain shrinks (E[r^2] = 1 regardless on +/-1).
"""

import importlib.util
import pathlib

import pytest

_MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "recipe" / "justrl2" / "reward_score.py"


def _load():
    spec = importlib.util.spec_from_file_location("justrl2_reward_score", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reward_score = _load()


class TestToUnitInterval:
    @pytest.mark.parametrize(
        ("raw", "acc", "expected"),
        [
            # math_dapo reports acc alongside its +/-1 score, so this is a relabelling.
            (1.0, True, 1.0),
            (-1.0, False, 0.0),
            (1.0, 1.0, 1.0),
            (-1.0, 0.0, 0.0),
            # No acc: fall back to an affine map of the raw score.
            (1.0, None, 1.0),
            (-1.0, None, 0.0),
            (0.0, None, 0.0),
            # An already-0/1 (or partial-credit) scorer must pass through untouched.
            (0.5, None, 0.5),
            (0.25, None, 0.25),
        ],
    )
    def test_maps_known_conventions(self, raw, acc, expected):
        assert reward_score._to_unit_interval(raw, acc) == pytest.approx(expected)

    def test_clamps_above_one(self):
        assert reward_score._to_unit_interval(1.5, None) == pytest.approx(1.0)

    def test_ignores_a_string_acc(self):
        """Some scorers put a label in `acc`; float() would raise, so fall back to the score."""
        assert reward_score._to_unit_interval(-1.0, "incorrect") == pytest.approx(0.0)


class TestComputeScore:
    @pytest.mark.parametrize("data_source", ["aime-2024", "aime-2025", "aime-2026", "math_dapo"])
    def test_rescales_math_dapo(self, data_source):
        correct = reward_score.compute_score(
            data_source=data_source, solution_str="work\nAnswer: 42", ground_truth="42"
        )
        wrong = reward_score.compute_score(data_source=data_source, solution_str="work\nAnswer: 7", ground_truth="42")
        assert correct["score"] == pytest.approx(1.0)
        assert wrong["score"] == pytest.approx(0.0)

    def test_passes_through_an_already_binary_scorer(self):
        correct = reward_score.compute_score(
            data_source="openai/gsm8k", solution_str="work\n#### 42", ground_truth="42"
        )
        wrong = reward_score.compute_score(data_source="openai/gsm8k", solution_str="work\n#### 7", ground_truth="42")
        assert correct["score"] == pytest.approx(1.0)
        assert wrong["score"] == pytest.approx(0.0)

    def test_preserves_extra_fields(self):
        """`acc` must survive: the DAPO group filter reads it as filter_groups.metric."""
        result = reward_score.compute_score(data_source="aime-2024", solution_str="work\nAnswer: 42", ground_truth="42")
        assert "acc" in result
        assert result["acc"]
        assert "pred" in result

    @pytest.mark.parametrize("data_source", ["aime-2024", "openai/gsm8k"])
    def test_score_always_in_unit_interval(self, data_source):
        for solution in ("work\nAnswer: 42", "work\n#### 42", "no answer at all", ""):
            score = reward_score.compute_score(data_source=data_source, solution_str=solution, ground_truth="42")[
                "score"
            ]
            assert 0.0 <= score <= 1.0, f"{data_source} / {solution!r} -> {score}"

    def test_loadable_through_verls_custom_reward_hook(self):
        """verl resolves the scorer by (path, name); keep both discoverable."""
        from omegaconf import OmegaConf

        from verl.trainer.ppo.reward import get_custom_reward_fn

        cfg = OmegaConf.create(
            {
                "reward": {
                    "custom_reward_function": {"path": str(_MODULE_PATH), "name": "compute_score"},
                }
            }
        )
        fn = get_custom_reward_fn(cfg)
        assert fn is not None
        result = fn(data_source="aime-2024", solution_str="work\nAnswer: 42", ground_truth="42")
        assert result["score"] == pytest.approx(1.0)
