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
"""A tool that hands back the problem statement, which the prompt withholds."""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GetQuestionTool(BaseTool):
    """Returns the question for this sample. Nothing about the answer is exposed.

    The prompt deliberately does not contain the question: prepare_data.py puts it in
    ``extra_info.tools_kwargs.get_question.create_kwargs.question`` instead, so a trajectory that
    never calls the tool has nothing to solve and scores 0. That makes tool use necessary rather
    than optional -- unlike gsm8k_tool, which grades a candidate answer and so both leaks the
    ground truth and lets the model guess its way to a reward.

    ToolAgentLoop._call_tool runs create -> execute -> release per call, so the stored question
    only has to survive one call.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        question = kwargs.get("create_kwargs", {}).get("question") or ""
        if not question:
            logger.warning("No question in create_kwargs -- check tools_kwargs in the dataset rows.")
        self._instance_dict[instance_id] = question
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        question = self._instance_dict.get(instance_id) or ""
        if not question:
            return ToolResponse(text="The problem statement is unavailable."), 0.0, {}
        return ToolResponse(text=question), 0.0, {}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)
