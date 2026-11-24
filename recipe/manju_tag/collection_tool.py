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
"""A tool that looks up a manju collection's content by collection_rowkey."""

import json
import logging
import os
from typing import Any

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GetCollectionInfoTool(BaseTool):
    """Returns title, subject category, plot summary and script of the collection asked for.

    Content comes from the collections.jsonl that prepare_data.py writes -- an offline stand-in for
    the eproxy lookup plus ASR-to-script generation that produced it, which is too slow to run per
    rollout. Unlike get_question, the argument matters: the model has to pass the rowkey from the
    user's message, and a wrong one gets "not found" rather than some other collection's content.

    Config:
        collections_path: path to collections.jsonl (``~`` is expanded).
        max_script_chars: scripts longer than this are cut, as in the single-turn data.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.max_script_chars = config.get("max_script_chars", 4000)
        path = os.path.expanduser(config["collections_path"])
        self._collections = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    self._collections[record["collection_rowkey"]] = record
        logger.info(f"Loaded {len(self._collections)} collections from {path}")

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        rowkey = str(parameters.get("collection_rowkey", "")).strip()
        record = self._collections.get(rowkey)
        if record is None:
            return ToolResponse(text=f"未找到合集 {rowkey!r}，请检查 collection_rowkey 是否正确。"), 0.0, {}

        script = record["script"]
        if len(script) > self.max_script_chars:
            script = script[: self.max_script_chars] + "..."
        # The closing reminder restates what the user message asked for, because ~3k tokens of
        # script now sit between that message and the answer: in the first smoke run 77% of
        # trajectories had no <answer> (those inspected answered in bare 【...】 after </think>),
        # and 84% called the tool again.
        text = (
            f"合集ID：{rowkey}\n"
            f"标题：{record['title']}\n"
            f"分类：{record['subject_category']}\n\n"
            f"剧情背景：\n{record['cc_ai_video_summary']}\n\n"
            f"漫剧剧本：\n{script}\n\n"
            "（以上是该合集的全部内容，无需再次调用工具。请据此从候选标签中选择，"
            "并把选出的标签写在 <answer></answer> 内。）"
        )
        return ToolResponse(text=text), 0.0, {}
