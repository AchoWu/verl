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
"""JustRL-II's math reward on verl: last ``\\boxed{}`` graded 0/1, the same rule for every set.

JustRL-II trains and evaluates with miles' ``--rm-type math``: extract the last ``\\boxed{...}``
in the response, compare it to the label with mathd normalisation and sympy, and fall back to
math-verify only when those say "wrong" (so the fallback can flip 0 -> 1, never the reverse).
The grader is vendored unchanged in ``math_utils.py`` / ``math_verify_fallback.py``; this file
only adapts it to verl's custom-reward signature.

That pairs with the prompt format: every UltraData-RL-Math query, and every AIME prompt built by
``prepare_aime_data.py``, ends with "Please reason step by step, and put your final answer
within \\boxed{}." verl's stock ``math_dapo`` scorer instead parses an ``Answer: ...`` line, so
it would mark nearly every boxed answer wrong -- which is why this scorer is not optional.

The reward is 0/1, which is also what makes the critic's ``V(s)`` read as
``P(this prefix ends up correct)`` and lets ``value_head_bias_init`` be the plain pass rate.

``data_source`` is deliberately ignored: JustRL-II applies one grader to the training set and to
every eval set alike. verl still keys validation metrics by ``data_source``, so per-year AIME
curves are unaffected.

Wire it up with (both keys are required; verl reads them together):

    custom_reward_function.path=recipe/justrl2/reward_score.py \
    custom_reward_function.name=compute_score

`run_justrl2_16k.sh` and `run_measure_pass_rate.sh` both pass these.
"""

from recipe.justrl2.math_utils import extract_answer, grade_answer_union


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Grade one response the way JustRL-II's ``--rm-type math`` does.

    Args:
        data_source: dataset identifier; unused (one grader for every set, as upstream).
        solution_str: the model's response, special tokens already stripped by the reward manager.
        ground_truth: the reference answer.
        extra_info: unused.
        **kwargs: ignored (e.g. sandbox_fusion_url, which only code scorers read).

    Returns:
        dict with "score" in {0.0, 1.0}, "acc" (bool, what the DAPO group filter and the val
        metrics read) and "pred" (the extracted boxed answer, "" when there is none).
    """
    correct = bool(grade_answer_union(solution_str, ground_truth))
    return {
        "score": 1.0 if correct else 0.0,
        "acc": correct,
        "pred": extract_answer(solution_str) or "",
    }
