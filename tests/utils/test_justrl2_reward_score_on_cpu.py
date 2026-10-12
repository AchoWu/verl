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
"""JustRL-II's boxed math reward, and the prompt format that has to go with it.

The grader and the prompt are a pair: every training and eval prompt asks for \\boxed{}, so the
scorer must read \\boxed{} -- verl's stock math_dapo reads an "Answer:" line and would mark
correct answers wrong. These tests pin both halves.
"""

import importlib.util
import pathlib
import sys

import pytest

_MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "recipe" / "justrl2" / "reward_score.py"


def _load():
    spec = importlib.util.spec_from_file_location("justrl2_reward_score", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reward_score = _load()


def _score(solution, ground_truth, data_source="ultradata-rl-math"):
    return reward_score.compute_score(data_source=data_source, solution_str=solution, ground_truth=ground_truth)


class TestComputeScore:
    @pytest.mark.parametrize(
        ("solution", "ground_truth"),
        [
            (r"so the answer is \boxed{10}.", "10"),
            (r"<think>long</think> \boxed{10}", "10"),
            (r"\boxed{\frac{1}{2}}", "1/2"),
            (r"\boxed{0.5}", r"\frac{1}{2}"),
            (r"\boxed{\$1.15}", "$1.15"),
            (r"\boxed{4 \times 8}", r"4 \times 8"),
            (r"\boxed{025}", "25"),  # the Qwen AIME mirror zero-pads
        ],
    )
    def test_accepts_equivalent_boxed_answers(self, solution, ground_truth):
        assert _score(solution, ground_truth) == {"score": 1.0, "acc": True, "pred": _boxed(solution)}

    @pytest.mark.parametrize(
        ("solution", "ground_truth"),
        [
            (r"\boxed{7}", "10"),
            # No \boxed{}: the DAPO "Answer:" format scores 0, as it does upstream.
            ("work\nAnswer: 10", "10"),
            # Truncated mid-box (overlong response): nothing to extract.
            (r"\boxed{10", "10"),
            ("", "10"),
        ],
    )
    def test_rejects_wrong_or_unboxed_answers(self, solution, ground_truth):
        result = _score(solution, ground_truth)
        assert result["score"] == 0.0
        assert result["acc"] is False

    def test_grades_the_last_box(self):
        assert _score(r"first \boxed{3}, then \boxed{10}", "10")["score"] == 1.0
        assert _score(r"first \boxed{10}, then \boxed{3}", "10")["score"] == 0.0

    @pytest.mark.parametrize("data_source", ["ultradata-rl-math", "aime-2024", "aime-2025", "aime-2026"])
    def test_one_grader_for_every_data_source(self, data_source):
        """JustRL-II grades train and eval with the same rule; data_source only labels metrics."""
        assert _score(r"\boxed{10}", "10", data_source)["score"] == 1.0
        assert _score("work\nAnswer: 10", "10", data_source)["score"] == 0.0

    def test_reports_acc_for_the_group_filter_and_val_metrics(self):
        result = _score(r"\boxed{10}", "10")
        assert result["acc"] is True and result["pred"] == "10"

    @pytest.mark.skipif(sys.platform == "win32", reason="math-verify fallback workers use select() on pipes")
    def test_math_verify_fallback_flips_a_string_grader_miss(self):
        """Reordered terms defeat mathd/sympy string grading; the math-verify fallback catches it."""
        pytest.importorskip("math_verify")
        assert _score(r"\boxed{\frac{3\sqrt{7}-\sqrt{3}}{2}}", r"\frac{-\sqrt{3}+3\sqrt{7}}{2}")["score"] == 1.0

    def test_stock_math_dapo_rejects_boxed_answers(self):
        """Why the custom scorer is mandatory: math_dapo parses "Answer:", not \\boxed{}."""
        from verl.utils.reward_score import default_compute_score

        assert default_compute_score("math_dapo", r"so the answer is \boxed{10}.", "10")["acc"] is False

    def test_loadable_through_verls_custom_reward_hook(self):
        """verl resolves the scorer by (path, name); keep both discoverable."""
        from omegaconf import OmegaConf

        from verl.trainer.ppo.reward import get_custom_reward_fn

        # On verl 0.6 the key is top-level, not nested under `reward`.
        cfg = OmegaConf.create({"custom_reward_function": {"path": str(_MODULE_PATH), "name": "compute_score"}})
        fn = get_custom_reward_fn(cfg)
        assert fn is not None
        assert fn(data_source="aime-2024", solution_str=r"\boxed{42}", ground_truth="42")["score"] == 1.0


def _boxed(solution):
    from recipe.justrl2.math_utils import extract_answer

    return extract_answer(solution)


ANSWER_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."


class TestPromptFormat:
    """Prompts must be what JustRL-II feeds miles: the raw text as one user turn, no system."""

    def test_training_row_is_the_raw_query_as_one_user_turn(self):
        from recipe.justrl2.prepare_ultradata_math import to_row

        query = "What is 2+3?\n" + ANSWER_SUFFIX
        row = to_row(7, {"uuid": "Math_00008", "query": query, "ground_truth": "5", "source": "s", "domain": "Math"})
        assert row["prompt"] == [{"role": "user", "content": query}]
        assert row["reward_model"]["ground_truth"] == "5"
        assert row["data_source"] == "ultradata-rl-math"
        assert row["extra_info"]["uuid"] == "Math_00008"

    @pytest.mark.parametrize("item", [{"query": "", "ground_truth": "5"}, {"query": "q", "ground_truth": " "}])
    def test_training_rows_without_query_or_label_are_skipped(self, item):
        from recipe.justrl2.prepare_ultradata_math import to_row

        assert to_row(0, item) is None

    def test_aime_prompt_appends_the_training_instruction(self):
        from recipe.justrl2 import prepare_aime_data

        assert prepare_aime_data.ANSWER_SUFFIX == ANSWER_SUFFIX
        row = prepare_aime_data.to_row("aime-2025", 0, {"problem": "  Find x.  ", "answer": 70})
        assert row["prompt"] == [{"role": "user", "content": "Find x.\n" + ANSWER_SUFFIX}]
        assert row["data_source"] == "aime-2025"
        assert row["reward_model"]["ground_truth"] == "70"

    @pytest.mark.parametrize(("answer", "label"), [("025", "25"), (70, "70"), ("0", "0"), ("3/4", "3/4")])
    def test_aime_labels_are_plain_integers(self, answer, label):
        from recipe.justrl2.prepare_aime_data import normalize_label

        assert normalize_label(answer) == label

    def test_stale_aime_parquets_are_detected(self, tmp_path):
        """A parquet from the earlier DAPO "Answer:" template must be rebuilt, not reused."""
        import pandas as pd

        from recipe.justrl2 import prepare_aime_data

        stale = tmp_path / "stale.parquet"
        pd.DataFrame([{"prompt": [{"role": "user", "content": 'Remember ... after "Answer:".'}]}]).to_parquet(stale)
        current = tmp_path / "current.parquet"
        pd.DataFrame([prepare_aime_data.to_row("aime-2024", 0, {"problem": "p", "answer": "1"})]).to_parquet(current)

        assert not prepare_aime_data._is_current(str(stale))
        assert prepare_aime_data._is_current(str(current))
