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
        # 真实执行次数 = 框架拼回来的工具回复数。<tool_call> 的个数会高估：一轮里模型可能写
        # 多个，但只有前 max_parallel_calls 个会被执行，而且模型还会自己幻觉出 <tool_call>。
        r["_execs"] = r["output"].count("<tool_response>")
        r["_emitted"] = r["output"].count("<tool_call>")

    n = len(rows)
    n_with_tool = sum(1 for r in rows if r["_execs"] > 0)
    execs = [r["_execs"] for r in rows]
    emitted = [r["_emitted"] for r in rows]
    scores = [r["score"] for r in rows]
    print(f"共 {n} 条轨迹")
    print(f"实际执行过工具: {n_with_tool} 条 ({n_with_tool / n:.1%})")
    print(f"实际执行次数: 最多 {max(execs)}，平均 {sum(execs) / n:.2f}")
    print(f"模型写出的 <tool_call>: 最多 {max(emitted)}，平均 {sum(emitted) / n:.2f}（高于执行数=有被丢弃或幻觉的）")
    print(f"score: 均值 {sum(scores) / n:.3f}，范围 [{min(scores)}, {max(scores)}]")

    no_ans = [r for r in rows if "####" not in r["output"]]
    print(f"没写 ####: {len(no_ans)} 条 ({len(no_ans) / n:.1%})")

    # 思考模式（Qwen3 等）。stop 按字符串 </tool_call> 触发，<think> 里提到这个标签也会被截断：
    # 能解析成合法调用就照常执行工具、进下一轮；解析不出来则整条轨迹直接结束（下面的“截断即结束”）。
    if any("<think>" in r["output"] for r in rows):
        unclosed = [r for r in rows if r["output"].count("<think>") > r["output"].count("</think>")]
        cut = [r for r in unclosed if r["output"].rstrip().endswith("</tool_call>")]
        print(f"[思考] 含 <think>: {sum('<think>' in r['output'] for r in rows) / n:.1%}")
        print(f"[思考] <think> 多于 </think>: {len(unclosed)} 条 ({len(unclosed) / n:.1%})（下限，后续轮次可能补上 </think>）")
        print(f"[思考]   其中以 </tool_call> 结尾，截断即结束: {len(cut)} 条 ({len(cut) / n:.1%})")
    if args.stats:
        return

    if args.index is not None:
        row = rows[args.index]
    else:
        row = max(rows, key=lambda r: r["_execs"])
        print(f"\n(自动挑选了实际执行工具最多的一条: {row['_execs']} 次)")

    print("\n" + "=" * 70 + "\nINPUT (prompt)\n" + "=" * 70)
    print(row["input"])
    print("\n" + "=" * 70 + "\nOUTPUT (模型输出 + 工具调用 + 工具回复，按发生顺序)\n" + "=" * 70)
    print(row["output"])
    print("\n" + "=" * 70)
    print(f"score = {row['score']}   ground_truth = {row.get('gts')}")


if __name__ == "__main__":
    main()
