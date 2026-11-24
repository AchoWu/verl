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
"""Manju (漫剧) tagging, single turn: the prompt carries the content itself, no tool.

The single-turn tagging template, minus its <think>...</think> instructions: Qwen3 thinks in its
own <think> block, so only the <answer> is asked for. One row per (collection, category), split by
collection the same way as prepare_data.py.

    python recipe/manju_tag/prepare_data_inline.py \
        --input /path/to/260914_train.jsonl --tags /path/to/tags.txt \
        --local_save_dir ~/data/manju_tag_inline

Writes train.parquet and test.parquet. They hold internal content and the internal tag taxonomy:
keep them out of the repository.
"""

import argparse
import os
import random

import datasets
from prepare_data import CATEGORIES, DATA_SOURCE, load_collections, load_tags

PROMPT = """你是一位专业的漫剧标签专家，请根据漫剧信息帮我从下列候选标签中选出合适的标签。
标题：{title}

分类：{subject_category}

剧情背景：
{cc_ai_video_summary}

漫剧剧本：
{script}

下面是供选择的候选标签：
{tags_desc}

注意：
1. 你必须至少选择一个标签，如果所有标签都不贴切，则从兜底标签中选择最接近的；
2. 最后把选出的【标签】写在 <answer></answer> 内，多个标签用顿号分隔，例如：<answer>【标签A】、【标签B】</answer>"""


def make_rows(collections, tags, split, max_script_chars):
    candidates = {en: [label for label, _ in tags[en]] for en in CATEGORIES}
    tags_desc = {en: "\n".join(f"【{label}】：{desc}" for label, desc in tags[en]) for en in CATEGORIES}
    rows = []
    for collection in collections:
        script = collection["script"]
        if len(script) > max_script_chars:
            script = script[:max_script_chars] + "..."
        for en in CATEGORIES:
            gold = collection[en]
            unknown = set(gold) - set(candidates[en])
            if not gold or unknown:
                raise ValueError(f"{collection['collection_rowkey']}/{en}: bad labels {gold}, unknown {unknown}")
            content = PROMPT.format(
                title=collection["title"],
                subject_category=collection["subject_category"],
                cc_ai_video_summary=collection["cc_ai_video_summary"],
                script=script,
                tags_desc=tags_desc[en],
            )
            rows.append(
                {
                    "data_source": DATA_SOURCE,
                    "prompt": [{"role": "user", "content": content}],
                    "ability": "tagging",
                    "reward_model": {"style": "rule", "ground_truth": gold},
                    "extra_info": {
                        "split": split,
                        "index": len(rows),
                        "collection_rowkey": collection["collection_rowkey"],
                        "category": en,
                        "candidates": candidates[en],
                    },
                }
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Single-turn tagging data, e.g. 260914_train.jsonl")
    parser.add_argument("--tags", required=True, help="tags.txt: '<label>\\t<description>' per line")
    parser.add_argument("--local_save_dir", default="~/data/manju_tag_inline")
    parser.add_argument("--num_val_collections", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_script_chars", type=int, default=4000)
    parser.add_argument("--tokenizer", default=None, help="Optional model path to report prompt lengths in tokens")
    args = parser.parse_args()

    tags = load_tags(args.tags)
    collections = load_collections(args.input)

    # Split by collection: the four rows of a collection share content and labels.
    random.Random(args.seed).shuffle(collections)
    val_collections = collections[: args.num_val_collections]
    train_collections = collections[args.num_val_collections :]
    train_rows = make_rows(train_collections, tags, "train", args.max_script_chars)
    val_rows = make_rows(val_collections, tags, "test", args.max_script_chars)

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    datasets.Dataset.from_list(train_rows).to_parquet(os.path.join(save_dir, "train.parquet"))
    datasets.Dataset.from_list(val_rows).to_parquet(os.path.join(save_dir, "test.parquet"))
    print(f"{len(collections)} collections -> {len(train_collections)} train / {len(val_collections)} val")
    print(f"Wrote {len(train_rows)} train / {len(val_rows)} test rows to {save_dir}")

    rows = train_rows + val_rows
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["extra_info"]["category"], []).append(r)
    for en, cat_rows in by_cat.items():
        longest = max(cat_rows, key=lambda r: len(r["prompt"][0]["content"]))
        print(f"  prompt chars [{en}]: max {len(longest['prompt'][0]['content'])}")
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        for en, cat_rows in by_cat.items():
            # Content length varies per collection here, so tokenize the longest row of each category.
            longest = max(cat_rows, key=lambda r: len(r["prompt"][0]["content"]))
            n = len(tok.apply_chat_template(longest["prompt"], add_generation_prompt=True, tokenize=True))
            print(f"  prompt tokens [{en}]: max {n} (keep data.max_prompt_length above this)")
