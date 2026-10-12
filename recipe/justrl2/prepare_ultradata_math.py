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
"""Build the JustRL-II training set -- UltraData-RL-2609, config Math -- in verl's RL schema.

    python3 -m recipe.justrl2.prepare_ultradata_math --out ~/verl/data/ultradata-rl-math.parquet

Mirrors JustRL-II's justrl2/prepare_data.py, which writes the same rows for miles:

* **The prompt is the raw `query`, as a single user turn, with no system prompt.** JustRL-II
  passes `--input-key prompt --apply-chat-template` and `prompt = query`; miles wraps that as
  `[{"role": "user", "content": query}]` and applies the model's own chat template. verl does
  the same with the `prompt` column below. Every query already ends with
  "\\nPlease reason step by step, and put your final answer within \\boxed{}." -- that line is
  part of the data, so nothing is added here.
* **The label is `ground_truth`**, graded by recipe/justrl2/reward_score.py (JustRL-II's
  `--rm-type math`).
* **Rows with an empty query or label are skipped; everything else is kept**, duplicates
  included (32,412 rows, 32,142 distinct queries), as upstream does.
"""

import argparse
import os

import pandas as pd

TRAIN_REPO = "openbmb/UltraData-RL-2609"
TRAIN_CONFIG = "Math"
DATA_SOURCE = "ultradata-rl-math"


def to_row(index, item):
    """One UltraData row -> one verl RL row, or None when the query or label is empty."""
    query = item.get("query")
    label = item.get("ground_truth")
    if not query or label is None or str(label).strip() == "":
        return None
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": query}],
        "ability": "MATH",
        "reward_model": {"style": "rule", "ground_truth": str(label)},
        "extra_info": {
            "index": index,
            "uuid": item.get("uuid"),
            "source": item.get("source"),
            "split": "train",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare UltraData-RL-2609 (Math) as a verl parquet")
    parser.add_argument("--out", required=True, help="Output parquet path")
    parser.add_argument("--repo", default=TRAIN_REPO)
    parser.add_argument("--config", default=TRAIN_CONFIG)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing parquet")
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} exists, skipping (use --force to rebuild)")
        return

    import datasets

    print(f"loading {args.repo} ({args.config}/train) ...")
    ds = datasets.load_dataset(args.repo, args.config, split="train")
    rows = [row for row in (to_row(i, item) for i, item in enumerate(ds)) if row is not None]
    skipped = len(ds) - len(rows)
    if skipped:
        print(f"  ({skipped} rows skipped: empty query or ground_truth)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.out)
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
