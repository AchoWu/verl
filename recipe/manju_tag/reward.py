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
"""Set F1 between the labels in the final <answer> and the gold labels, for all four categories.

Hooked in with
    custom_reward_function.path=recipe/manju_tag/reward.py custom_reward_function.name=compute_score
"""

import re

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
# Anything shaped like a label that is not one of this category's candidates: a label from another
# category or a made-up one. Counted as a wrong pick, so it costs precision.
LABEL_LIKE_RE = re.compile(r"(?:频道|情节|角色|时代)_[\u4e00-\u9fff]+")


def extract_labels(answer: str, candidates) -> set[str]:
    picked = set()
    # Longest first, so a candidate that contains a shorter one is not also counted as the shorter.
    for label in sorted(candidates, key=len, reverse=True):
        if label in answer:
            picked.add(label)
            answer = answer.replace(label, " ")
    picked.update(LABEL_LIKE_RE.findall(answer))
    return picked


def compute_score(data_source, solution_str, ground_truth, extra_info=None, require_tool_call=True, **kwargs):
    extra_info = extra_info or {}
    gold = set(ground_truth)

    # Only the model's text after the last tool reply: the tool's reply itself mentions <answer></answer>.
    final_turn = solution_str.rsplit("</tool_response>", 1)[-1]
    answers = ANSWER_RE.findall(final_turn)
    pred = extract_labels(answers[-1], extra_info["candidates"]) if answers else set()

    tp = len(pred & gold)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0

    # Without the content, the majority labels alone score well (频道_男频 58%, 时代_现代 64%,
    # 情节_逆袭打脸 92% of collections), so a trajectory that never ran the tool gets 0 however good
    # its guess. num_turns counts turns ToolAgentLoop actually executed -- 2 is prompt + one reply --
    # so text the model writes to look like a tool response does not count.
    num_turns = extra_info.get("num_turns")
    if num_turns is None and require_tool_call:
        # Scoring everything 0 would look like a model that never calls the tool.
        raise ValueError("extra_info has no num_turns; run with the tool agent loop or set require_tool_call=False")
    called_tool = num_turns is not None and num_turns > 2

    return {
        "score": f1 if called_tool or not require_tool_call else 0.0,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "called_tool": float(called_tool),
        "has_answer": float(bool(answers)),
    }
