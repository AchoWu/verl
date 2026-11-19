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
"""GSM8K with the question withheld from the prompt and served by the get_question tool.

Mirrors examples/data_preprocess/gsm8k_tool_agent_loop.py, except the question goes into
tools_kwargs instead of the user message, so the model has to call the tool to see it.
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs

SYSTEM_PROMPT = (
    "You are a math expert. The problem you have to solve is NOT shown to you: call the "
    "`get_question` tool to retrieve it. Once you have the problem, reason step by step and "
    "put your final answer in the format of `#### <answer>`."
)

USER_PROMPT = (
    "Retrieve the math problem with the `get_question` tool and solve it. Let's think step by "
    "step and output the final answer after `####`."
)

# Every row's prompt would otherwise be byte-identical, since the question -- the only thing that
# differs between samples -- now lives in tools_kwargs. vLLM died with a CUDA illegal memory
# access partway through the first rollout on such a batch (all 8 engines in the same second,
# at the same generation progress rather than the same wall-clock time), and survived the same
# config with the varied prompts of the answer-checker dataset. Root cause is still unknown, so
# this ID is a probe as much as a fix: it is the one thing the model cannot derive, and it leaks
# nothing about the question or its answer.
PROBLEM_ID_PREFIX = "Problem ID: {split}-{idx:05d}\n\n"


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k_get_question", help="The save directory for the dataset."
    )
    args = parser.parse_args()

    data_source = "openai/gsm8k"
    dataset = datasets.load_dataset(args.local_dataset_path or data_source, "main")

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            return {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": PROBLEM_ID_PREFIX.format(split=split, idx=idx) + USER_PROMPT,
                    },
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                    "need_tools_kwargs": True,
                    # The only copy of the question the rollout can reach.
                    "tools_kwargs": {
                        "get_question": {"create_kwargs": {"question": question_raw}},
                    },
                },
            }

        return process_fn

    train_dataset = dataset["train"].map(function=make_map_fn("train"), with_indices=True)
    test_dataset = dataset["test"].map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    print(f"Wrote {len(train_dataset)} train / {len(test_dataset)} test rows to {local_save_dir}")

    # multi_turn.max_tool_response_length truncates the tool's text, i.e. the question itself.
    longest = max(len(r["question"]) for r in train_dataset["extra_info"])
    print(f"Longest question: {longest} chars -- keep max_tool_response_length above this.")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
