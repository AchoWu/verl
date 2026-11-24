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
"""Manju (漫剧) tagging as a tool-use task.

The prompt names a collection_rowkey and one tag category, nothing else. Title, subject category,
plot summary and script are only reachable through the get_collection_info tool, which looks them
up by rowkey in the collections.jsonl written here. One row per (collection, category), as in the
single-turn tagging data this is converted from.

    python recipe/manju_tag/prepare_data.py \
        --input /path/to/260914_train.jsonl --tags /path/to/tags.txt \
        --local_save_dir ~/data/manju_tag

Writes train.parquet, test.parquet and collections.jsonl. These hold internal content and the
internal tag taxonomy: keep them out of the repository.
"""

import argparse
import json
import os
import random

import datasets

DATA_SOURCE = "manju_tag"

CATEGORIES = {"pindao": "频道", "qingjie": "情节", "juese": "角色", "shidai": "时代"}

# Fields served by the tool. Everything else in the input rows (ASR text, urls, labels) is not.
CONTENT_FIELDS = ("title", "subject_category", "cc_ai_video_summary", "script")

SYSTEM_PROMPT = (
    "你是一位专业的漫剧标签专家。用户只会告诉你漫剧合集的 collection_rowkey，不会给出漫剧内容："
    "请调用 `get_collection_info` 工具获取该合集的标题、分类、剧情背景和剧本，"
    "再根据内容从候选标签中选出合适的标签。"
)

# The single-turn template asked for <think>...</think><answer>...</answer>. Qwen3 thinks in its own
# <think> block and, asked for <answer> anyway, often put the labels straight after </think>; that
# is now the format, and reward.py reads labels from there.
USER_PROMPT = """请为合集 {collection_rowkey} 打【{category_cn}】类标签。

下面是供选择的候选标签：
{tags_desc}

注意：
1. 先调用 `get_collection_info` 获取合集内容，不要凭空猜测；该工具只需调用一次；
2. 你必须至少选择一个标签，如果所有标签都不贴切，则从兜底标签中选择最接近的；
3. 思考结束后，直接输出选出的【标签】，多个标签用顿号分隔，不要输出其他内容。"""


def load_tags(path):
    """Candidate labels per category, as (label, description) in file order."""
    tags = {en: [] for en in CATEGORIES}
    prefix_to_en = {cn: en for en, cn in CATEGORIES.items()}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            label, _, desc = line.partition("\t")
            en = prefix_to_en.get(label.split("_", 1)[0])
            if en is None:
                raise ValueError(f"Tag {label!r} has no known category prefix {list(prefix_to_en)}")
            tags[en].append((label, desc))
    return tags


def load_collections(path):
    """One record per collection: content plus the labels of all four categories.

    The input has one row per (collection, category) with identical content and labels across the
    four, so the first row of each collection is kept and the rest are checked against it.
    """
    collections = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rowkey = row["collection_rowkey"]
            record = {"collection_rowkey": rowkey}
            record.update({k: row.get(k) or "" for k in CONTENT_FIELDS})
            record.update({en: list(row.get(en) or []) for en in CATEGORIES})
            seen = collections.setdefault(rowkey, record)
            if seen != record:
                raise ValueError(f"Rows of collection {rowkey} disagree on content or labels")
    return list(collections.values())


def make_rows(collections, tags, split):
    candidates = {en: [label for label, _ in tags[en]] for en in CATEGORIES}
    tags_desc = {en: "\n".join(f"【{label}】：{desc}" for label, desc in tags[en]) for en in CATEGORIES}
    rows = []
    for collection in collections:
        for en, cn in CATEGORIES.items():
            gold = collection[en]
            unknown = set(gold) - set(candidates[en])
            if not gold or unknown:
                raise ValueError(f"{collection['collection_rowkey']}/{en}: bad labels {gold}, unknown {unknown}")
            user = USER_PROMPT.format(
                collection_rowkey=collection["collection_rowkey"], category_cn=cn, tags_desc=tags_desc[en]
            )
            rows.append(
                {
                    "data_source": DATA_SOURCE,
                    "agent_name": "tool_agent",
                    "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                    "ability": "tagging",
                    "reward_model": {"style": "rule", "ground_truth": gold},
                    "extra_info": {
                        "split": split,
                        "index": len(rows),
                        "collection_rowkey": collection["collection_rowkey"],
                        "category": en,
                        # reward.py matches the answer against these, so a label written without
                        # a separator after it is still found.
                        "candidates": candidates[en],
                    },
                }
            )
    return rows


def report_lengths(rows, tokenizer_path):
    """Prompt and tool-response sizes, to size max_prompt_length / max_tool_response_length."""
    by_cat = {}
    for r in rows:
        chars = sum(len(m["content"]) for m in r["prompt"])
        by_cat.setdefault(r["extra_info"]["category"], []).append(chars)
    for en, lens in by_cat.items():
        print(f"  prompt chars [{en}]: max {max(lens)}")
    if tokenizer_path:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_path)
        longest = {}
        for r in rows:
            en = r["extra_info"]["category"]
            if en not in longest:
                longest[en] = len(tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=True))
        for en, n in longest.items():
            # The candidate list dominates and is the same for every row of a category, so one row
            # per category is representative. The tool schema adds ~150 tokens on top.
            print(f"  prompt tokens [{en}]: {n} (+ tool schema)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Single-turn tagging data, e.g. 260914_train.jsonl")
    parser.add_argument("--tags", required=True, help="tags.txt: '<label>\\t<description>' per line")
    parser.add_argument("--local_save_dir", default="~/data/manju_tag")
    parser.add_argument("--num_val_collections", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default=None, help="Optional model path to report prompt lengths in tokens")
    args = parser.parse_args()

    tags = load_tags(args.tags)
    collections = load_collections(args.input)

    # Split by collection: the four rows of a collection share content and labels, so splitting by
    # row would leak validation collections into training.
    random.Random(args.seed).shuffle(collections)
    val_collections = collections[: args.num_val_collections]
    train_collections = collections[args.num_val_collections :]
    train_rows = make_rows(train_collections, tags, "train")
    val_rows = make_rows(val_collections, tags, "test")

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    datasets.Dataset.from_list(train_rows).to_parquet(os.path.join(save_dir, "train.parquet"))
    datasets.Dataset.from_list(val_rows).to_parquet(os.path.join(save_dir, "test.parquet"))
    # The tool's lookup table covers every collection, so a trajectory gets the same answer for its
    # rowkey no matter which split it is in.
    with open(os.path.join(save_dir, "collections.jsonl"), "w", encoding="utf-8") as f:
        for c in collections:
            f.write(json.dumps({k: c[k] for k in ("collection_rowkey", *CONTENT_FIELDS)}, ensure_ascii=False) + "\n")

    print(f"{len(collections)} collections -> {len(train_collections)} train / {len(val_collections)} val")
    print(f"Wrote {len(train_rows)} train / {len(val_rows)} test rows and collections.jsonl to {save_dir}")
    report_lengths(train_rows + val_rows, args.tokenizer)
    longest = max(len(c["script"]) for c in collections)
    print(f"  longest script: {longest} chars (the tool truncates to its max_script_chars)")
