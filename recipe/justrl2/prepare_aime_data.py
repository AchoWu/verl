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
"""Build AIME 2024 / 2025 / 2026 validation parquets in verl's RL schema.

    python3 -m recipe.justrl2.prepare_aime_data --out-dir ~/verl/data

Why this exists rather than just downloading three files:

1. **Per-year metrics.** verl keys validation metrics off `data_source`, and the ready-made
   `BytedTsinghua-SIA/AIME-2024` parquet uses `math_dapo` -- the same value every other math
   set uses. Three files all labelled `math_dapo` collapse into one blended number. This script
   stamps `aime-2024` / `aime-2025` / `aime-2026`, which still routes to the math_dapo scorer
   (`data_source.startswith("aime")` in verl/utils/reward_score/__init__.py) while reporting
   `val-core/aime-2025/...` separately per year.

2. **Identical prompt format.** The instruction wrapper decides how the model answers, so a
   2025 file phrased differently from the 2024 one produces scores that cannot be compared.
   PROMPT_TEMPLATE below is copied verbatim from BytedTsinghua-SIA/AIME-2024.

3. **Upstream schemas differ.** aime_2026 ships (problem, answer, problem_idx); aime_2025 ships
   (problem, answer, solution, url, ...) with the answer as a string. Both are normalised here.

Each year has 30 problems. `--repeat N` writes each problem N times, which is how the
BytedTsinghua AIME-2024 parquet reaches 960 rows: with `val_kwargs.n=1` that is how you get
avg@N. Leave it at 1 when you set `val_kwargs.n` instead -- doing both multiplies.
"""

import argparse
import os

import datasets
import pandas as pd

# Verbatim from BytedTsinghua-SIA/AIME-2024. Do not reword: it decides the answer format the
# math_dapo scorer parses ("Answer: $Answer" on its own last line).
PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your response should be "
    "of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)

# Upstream dataset -> (hf repo, config, split, problem column, answer column)
SOURCES = {
    "aime-2024": ("Maxwell-Jia/AIME_2024", "default", "train", "Problem", "Answer"),
    "aime-2025": ("yentinglin/aime_2025", "default", "train", "problem", "answer"),
    "aime-2026": ("MathArena/aime_2026", "default", "train", "problem", "answer"),
}


def build(name, repeat):
    repo, config, split, problem_key, answer_key = SOURCES[name]
    print(f"[{name}] loading {repo} ({config}/{split}) ...")
    ds = datasets.load_dataset(repo, config, split=split)

    missing = [k for k in (problem_key, answer_key) if k not in ds.column_names]
    if missing:
        raise SystemExit(
            f"[{name}] {repo} has columns {ds.column_names}, missing {missing}. The upstream "
            f"schema changed; update SOURCES in this file."
        )

    rows = []
    for index, item in enumerate(ds):
        problem = str(item[problem_key]).strip()
        # AIME answers are integers 0-999; normalise away float/str formatting ("42.0" -> "42")
        # so string comparison in the scorer does not fail on formatting alone.
        answer = str(item[answer_key]).strip()
        if answer.endswith(".0"):
            answer = answer[:-2]
        row = {
            "data_source": name,
            "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(problem=problem)}],
            "ability": "MATH",
            "reward_model": {"style": "rule-lighteval/MATH_v2", "ground_truth": answer},
            "extra_info": {"index": index, "raw_problem": problem, "split": None},
        }
        rows.extend([row] * repeat)

    df = pd.DataFrame(rows)
    print(f"[{name}] {len(ds)} problems x {repeat} = {len(df)} rows")
    return df


def main():
    parser = argparse.ArgumentParser(description="Prepare AIME 2024/2025/2026 validation parquets")
    parser.add_argument("--out-dir", required=True, help="Directory to write <year>.parquet into")
    parser.add_argument("--years", nargs="+", default=sorted(SOURCES), choices=sorted(SOURCES))
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Duplicate each problem N times for avg@N. Keep at 1 if you use val_kwargs.n.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquets")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for name in args.years:
        path = os.path.join(args.out_dir, f"{name}.parquet")
        if os.path.exists(path) and not args.force:
            print(f"[{name}] {path} exists, skipping (use --force to rebuild)")
            written.append(path)
            continue
        build(name, args.repeat).to_parquet(path)
        written.append(path)
        print(f"[{name}] wrote {path}")

    print("\nval_files for the training script:")
    print("  [" + ",".join(f"'{p}'" for p in written) + "]")


if __name__ == "__main__":
    main()
