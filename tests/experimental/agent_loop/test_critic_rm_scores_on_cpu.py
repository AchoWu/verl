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
"""`AgentLoopOutput.as_dict` laying out the JustRL-II critic reward as `critic_rm_scores`.

The DAPO reward manager emits `critic_reward_score` (the reward *without* the soft overlong
punishment) inside `reward_extra_info`; `as_dict` turns it into a token-level tensor shaped like
`rm_scores` so the trainer can use it as the critic's regression target.
"""

import pytest
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput


def _make_output(reward_score=0.815, reward_extra_info=None, response_len=3):
    """Build an output, passing `extra_fields` only when asked.

    Whether `extra_fields` is set at construction decides which branch of `as_dict` runs:
    set it, and `model_dump(exclude_unset=True)` emits a deep copy; leave it unset, and
    `as_dict` falls back to `self.extra_fields.copy()` (shallow). The real pipeline takes the
    second path, so tests that need to exercise aliasing must leave it unset here and mutate
    `output.extra_fields` afterwards.
    """
    kwargs = {} if reward_extra_info is None else {"extra_fields": {"reward_extra_info": reward_extra_info}}
    return AgentLoopOutput(
        prompt_ids=[101, 102],
        response_ids=list(range(11, 11 + response_len)),
        response_mask=[1] * response_len,
        reward_score=reward_score,
        metrics=AgentLoopMetrics(),
        **kwargs,
    )


def test_critic_rm_scores_mirrors_the_rm_scores_layout():
    """Both rewards sit on the last response token; only the value differs."""
    fields = _make_output(reward_score=0.815, reward_extra_info={"acc": 1.0, "critic_reward_score": 1.0}).as_dict()

    assert "critic_rm_scores" in fields
    assert fields["critic_rm_scores"].shape == fields["rm_scores"].shape
    # actor sees the shaped reward, critic the pre-penalty one
    assert fields["rm_scores"][-1].item() == pytest.approx(0.815)
    assert fields["critic_rm_scores"][-1].item() == pytest.approx(1.0)
    assert torch.all(fields["critic_rm_scores"][:-1] == 0)


def test_absent_without_the_key():
    """exclude_from_critic=False (or a non-DAPO manager) must not produce the field."""
    fields = _make_output(reward_extra_info={"acc": 1.0}).as_dict()
    assert "critic_rm_scores" not in fields


def test_absent_without_reward_extra_info():
    fields = _make_output().as_dict()
    assert "critic_rm_scores" not in fields


def test_absent_without_a_reward_score():
    """No rm_scores to mirror yet -- the reward has not been computed."""
    fields = _make_output(reward_score=None, reward_extra_info={"critic_reward_score": 1.0}).as_dict()
    assert "rm_scores" not in fields
    assert "critic_rm_scores" not in fields


def test_does_not_mutate_shared_reward_extra_info():
    """Regression: a multi-output session shares ONE reward_extra_info dict across outputs.

    `_agent_loop_postprocess` assigns `final_output.extra_fields["reward_extra_info"]` to every
    other output *by reference, after construction*. That ordering matters: because the field
    was not set at construction, `model_dump(exclude_unset=True)` omits it and `as_dict` falls
    back to `self.extra_fields.copy()` -- a shallow copy that still aliases the inner dict. A
    destructive read here would leave only the first output with a critic target.

    Mutating post-construction (rather than passing `extra_fields=` to the constructor) is
    therefore load-bearing: the constructor path goes through pydantic's deep copy and would
    hide the bug.
    """
    shared = {"acc": 1.0, "overlong_reward": -0.185, "critic_reward_score": 1.0}
    outputs = [_make_output(reward_extra_info=None) for _ in range(3)]
    for output in outputs:
        output.extra_fields["reward_extra_info"] = shared
    assert outputs[0].extra_fields["reward_extra_info"] is outputs[-1].extra_fields["reward_extra_info"]

    rendered = [output.as_dict() for output in outputs]

    for i, fields in enumerate(rendered):
        assert "critic_rm_scores" in fields, f"output {i} lost its critic target"
        assert fields["critic_rm_scores"][-1].item() == pytest.approx(1.0)
    assert shared["critic_reward_score"] == 1.0, "the shared dict was mutated"


def test_as_dict_is_idempotent():
    """The same output rendered twice must give the same result."""
    output = _make_output(reward_extra_info={"critic_reward_score": 1.0})

    first = output.as_dict()
    second = output.as_dict()

    assert "critic_rm_scores" in first
    assert "critic_rm_scores" in second
    torch.testing.assert_close(first["critic_rm_scores"], second["critic_rm_scores"])
