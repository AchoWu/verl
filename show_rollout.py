#!/usr/bin/env python3
"""在终端查看 trainer.rollout_data_dir 落盘的轨迹。

用法:
    python show_rollout.py <dump_dir>/1.jsonl              # 工具调用最多的那条
    python show_rollout.py <dump_dir>/1.jsonl -i 5         # 第 5 条
    python show_rollout.py <dump_dir>/1.jsonl --stats      # 只看统计
"""

import argparse
import json


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="<rollout_data_dir>/<step>.jsonl")
    p.add_argument("-i", "--index", type=int, default=None, help="看第几条；默认挑工具调用最多的")
    p.add_argument("--stats", action="store_true", help="只打印统计，不打印正文")
    args = p.parse_args()

    rows = load(args.path)
    for r in rows:
        r["_calls"] = r["output"].count("<tool_call>")

    n_with_tool = sum(1 for r in rows if r["_calls"] > 0)
    scores = [r["score"] for r in rows]
    print(f"共 {len(rows)} 条轨迹")
    print(f"调用过工具: {n_with_tool} 条 ({n_with_tool / len(rows):.1%})")
    print(f"工具调用次数: 最多 {max(r['_calls'] for r in rows)}，平均 {sum(r['_calls'] for r in rows) / len(rows):.2f}")
    print(f"score: 均值 {sum(scores) / len(scores):.3f}，范围 [{min(scores)}, {max(scores)}]")
    if args.stats:
        return

    if args.index is not None:
        row = rows[args.index]
    else:
        row = max(rows, key=lambda r: r["_calls"])
        print(f"\n(自动挑选了工具调用最多的一条: {row['_calls']} 次)")

    print("\n" + "=" * 70 + "\nINPUT (prompt)\n" + "=" * 70)
    print(row["input"])
    print("\n" + "=" * 70 + "\nOUTPUT (模型输出 + 工具调用 + 工具回复，按发生顺序)\n" + "=" * 70)
    print(row["output"])
    print("\n" + "=" * 70)
    print(f"score = {row['score']}   ground_truth = {row.get('gts')}")


if __name__ == "__main__":
    main()
