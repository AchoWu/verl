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
"""A 0/1 reward wrapper, so the critic's V is a probability rather than an affine image of one.

verl's `math_dapo` scorer -- which DAPO-Math-17k and the AIME sets dispatch to -- returns
+1 for correct and -1 for incorrect. That is fine for a group-baseline method like GRPO, where
only reward *differences* within a group matter, but it costs the critic its interpretation:

    0/1 reward   ->  V(s) = P(this trajectory ends up correct | prefix)
    +/-1 reward  ->  V(s) = 2 * P(correct) - 1

With 0/1, `V` read at the prompt prefix *is* the model's estimate of how hard that problem is,
`value_head_bias_init` is the pass rate, and the value-head seeding argument works at full
strength: `E[r^2] = E[r]` for a Bernoulli reward, so seeding at the mean takes the initial loss
from `E[r]` down to `Var(r) = p(1-p)` -- a factor of `1/(1-p)`. On a +/-1 reward `E[r^2] = 1`
regardless, so seeding only removes the `mean^2` term and buys far less.

Rescaling rather than editing `math_dapo.compute_score` keeps every other recipe that depends
on its +/-1 convention untouched.

Wire it up with (both keys are required; verl reads them together):

    reward.custom_reward_function.path=recipe/justrl2/reward_score.py \
    reward.custom_reward_function.name=compute_score

`run_justrl2_16k.sh` passes these by default; set REWARD_SCALE=pm1 there to fall back to the
stock +/-1 scorer.
"""

from verl.utils.reward_score import default_compute_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Score with verl's default dispatch, then map the result onto [0, 1].

    Args:
        data_source: dataset identifier; selects the underlying scorer.
        solution_str: the model's response.
        ground_truth: the reference answer.
        extra_info: passed through to the underlying scorer.
        **kwargs: forwarded too (e.g. sandbox_fusion_url, reward_router_address).

    Returns:
        dict with "score" in [0, 1], plus whatever extra fields the underlying scorer
        reported (notably "acc", which the DAPO group filter reads as its metric).
    """
    result = default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )

    if isinstance(result, dict):
        extra = dict(result)
        raw = extra.pop("score", 0.0)
    else:
        extra, raw = {}, result

    score = _to_unit_interval(float(raw), extra.get("acc"))
    return {**extra, "score": score}


def _to_unit_interval(raw, acc=None):
    """Map a scorer's output onto [0, 1].

    Prefers the scorer's own correctness flag when it reports one: `math_dapo` returns `acc`
    alongside its +/-1 score, so this is a relabelling rather than a guess. Falls back to an
    affine map of the raw score, which handles the +/-1 convention exactly and leaves an
    already-0/1 scorer untouched.
    """
    if acc is not None and not isinstance(acc, str):
        # bool, or a 0/1 float from a scorer that grades partially
        return float(acc)
    if raw < 0.0:
        # +/-1 (or [-1, 1]) convention -> (raw + 1) / 2
        return (raw + 1.0) / 2.0
    return min(raw, 1.0)
