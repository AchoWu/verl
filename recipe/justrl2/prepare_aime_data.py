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
"""Build AIME 2024 / 2025 / 2026 validation parquets in verl's RL schema, as JustRL-II does.

    python3 -m recipe.justrl2.prepare_aime_data --out-dir ~/verl/data

Mirrors JustRL-II's justrl2/prepare_eval_data.py, so the scores are comparable:

1. **Same sources.** aime-2024 from the Qwen2.5-Math evaluation mirror, aime-2025 from
   `math-ai/aime25`, aime-2026 from `math-ai/aime26` -- 30 problems each.

2. **The training prompt.** `prompt = f"{problem}\\n{ANSWER_SUFFIX}"` as a single user turn with
   no system prompt: the same instruction line every UltraData-RL-Math query ends with, so the
   model answers in `\\boxed{}`, which recipe/justrl2/reward_score.py grades.

3. **Per-year metrics.** verl keys validation metrics off `data_source`, so each file is stamped
   `aime-2024` / `aime-2025` / `aime-2026` and reports `val-core/aime-2025/...` separately. The
   custom scorer ignores `data_source`, so this is labelling only.

4. **Plain-integer labels.** The Qwen mirror zero-pads AIME answers ("025"); they are stored as
   "25". The grader accepts either form; this is tidiness, not correctness.

`--repeat N` writes each problem N times (avg@N with `val_kwargs.n=1`). Leave it at 1 when you
set `val_kwargs.n` instead -- doing both multiplies.
"""

import argparse
import json
import os
import urllib.request

import pandas as pd

# The instruction line every training prompt ends with (JustRL-II prepare_eval_data.ANSWER_SUFFIX).
ANSWER_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."

_QWEN = "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/main/evaluation/data/{name}/test.jsonl"
_HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

SOURCES = {
    "aime-2024": _QWEN.format(name="aime24"),
    "aime-2025": _HF.format(repo="math-ai/aime25", path="test.jsonl"),
    "aime-2026": _HF.format(repo="math-ai/aime26", path="aime2026.jsonl"),
}
EXPECTED_PROBLEMS = 30


def _fetch_jsonl(url):
    req = urllib.request.Request(url, headers={"User-Agent": "verl-justrl2-prepare-aime"})
    with urllib.request.urlopen(req) as resp:
        text = resp.read().decode("utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def build_prompt(problem):
    return f"{problem.strip()}\n{ANSWER_SUFFIX}"


def normalize_label(answer):
    label = str(answer).strip()
    if label.isdigit():
        label = str(int(label))
    return label


def to_row(name, index, item):
    problem = str(item["problem"]).strip()
    return {
        "data_source": name,
        "prompt": [{"role": "user", "content": build_prompt(problem)}],
        "ability": "MATH",
        "reward_model": {"style": "rule", "ground_truth": normalize_label(item["answer"])},
        "extra_info": {"index": index, "raw_problem": problem, "split": "test"},
    }


def build(name, repeat):
    url = SOURCES[name]
    print(f"[{name}] fetching {url} ...")
    raw = _fetch_jsonl(url)
    if len(raw) != EXPECTED_PROBLEMS:
        raise SystemExit(f"[{name}] expected {EXPECTED_PROBLEMS} problems, got {len(raw)}: the upstream file changed.")
    missing = [k for k in ("problem", "answer") if k not in raw[0]]
    if missing:
        raise SystemExit(f"[{name}] rows have keys {sorted(raw[0])}, missing {missing}: the upstream schema changed.")

    rows = []
    for index, item in enumerate(raw):
        rows.extend([to_row(name, index, item)] * repeat)
    df = pd.DataFrame(rows)
    print(f"[{name}] {len(raw)} problems x {repeat} = {len(df)} rows")
    return df


def _is_current(path):
    """Whether an existing parquet was built with this prompt format (earlier versions of this
    script wrote the DAPO "Answer: $Answer" template, which the boxed grader cannot score)."""
    try:
        prompt = pd.read_parquet(path, columns=["prompt"])["prompt"].iloc[0]
        return str(prompt[0]["content"]).endswith(ANSWER_SUFFIX)
    except Exception:
        return False


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
            if _is_current(path):
                print(f"[{name}] {path} exists, skipping (use --force to rebuild)")
                written.append(path)
                continue
            print(f"[{name}] {path} has a different prompt format (pre-boxed template?), rebuilding")
        build(name, args.repeat).to_parquet(path)
        written.append(path)
        print(f"[{name}] wrote {path}")

    print("\nval_files for the training script:")
    print("  [" + ",".join(f"'{p}'" for p in written) + "]")


if __name__ == "__main__":
    main()
