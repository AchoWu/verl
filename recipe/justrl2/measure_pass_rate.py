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
"""Measure the base model's mean reward on the training set, to seed the critic's value head.

`critic.model.value_head_bias_init` should be E[r] under the *rollout* distribution, because
with a zero-initialized head V == bias for every token at step 0 and the value loss then opens
at Var(r) instead of E[r^2]. See recipe/justrl2/README.md section 2.

That makes the sampling config part of the measurement, not a detail: a greedy (temperature=0)
accuracy is the wrong number. This script samples with the same temperature / top_p / n as
training and reports the mean over all samples.

    # 1. generate with the TRAINING sampling params
    python3 -m verl.trainer.main_generation_server \
        --config-path="$PWD/recipe/justrl2/config" \
        --config-name='justrl2_trainer.yaml' \
        data.train_files="$TRAIN_FILE" \
        data.output_path=/tmp/passrate_gen.parquet \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.response_length=14336 \
        trainer.n_gpus_per_node=8 trainer.nnodes=1

    # 2. score it
    python3 -m recipe.justrl2.measure_pass_rate --path /tmp/passrate_gen.parquet

Or run both in one go with `bash recipe/justrl2/run_measure_pass_rate.sh`.

Scoring reuses verl's own `default_compute_score` (dispatched on `data_source`), so the number
matches what the reward manager will produce during training. Pass `--custom-reward-path` to
use the same custom scorer you train with.

NOTE the overlong penalty is deliberately NOT applied here: the critic's regression target
excludes it (`exclude_from_critic=True`), so the bias must match the unshaped reward.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

_SCORER = None


def _init_scorer(custom_path, custom_name):
    """Resolve the scoring function once per worker process."""
    global _SCORER
    if custom_path:
        from verl.utils.import_utils import load_extern_object

        _SCORER = load_extern_object(module_path=custom_path, object_name=custom_name)
    else:
        from verl.utils.reward_score import default_compute_score

        _SCORER = default_compute_score


def _score_one(args):
    """Score every sample of one prompt. Returns (data_source, [scores...])."""
    data_source, responses, ground_truth, extra_info = args
    scores = []
    for response in responses:
        try:
            result = _SCORER(
                data_source=data_source,
                solution_str=response,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
        except Exception as exc:  # a scorer crash must not sink the whole run
            print(f"[warn] scorer raised on data_source={data_source}: {type(exc).__name__}: {exc}")
            result = 0.0
        if isinstance(result, dict):
            result = result.get("score", 0.0)
        scores.append(float(result))
    return data_source, scores


def _wilson_interval(successes, total, z=1.96):
    """Wilson score interval -- behaves near 0 and 1, where the normal approximation does not."""
    if total == 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / total + z**2 / (4 * total**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def main():
    parser = argparse.ArgumentParser(
        description="Measure mean reward on the training set to seed critic.model.value_head_bias_init",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Parquet file written by main_generation_server (must contain a responses column)",
    )
    parser.add_argument("--response-key", default="responses")
    parser.add_argument("--data-source-key", default="data_source")
    parser.add_argument("--reward-model-key", default="reward_model")
    parser.add_argument("--extra-info-key", default="extra_info")
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--custom-reward-path", default=None, help="Use the custom scorer you train with")
    parser.add_argument("--custom-reward-name", default="compute_score")
    parser.add_argument("--max-prompts", type=int, default=-1, help="Score only the first N prompts (a quick look)")
    parser.add_argument("--output-json", default=None, help="Also write the metrics here")
    args = parser.parse_args()

    from verl.utils.fs import copy_to_local

    dataset = pd.read_parquet(copy_to_local(args.path))
    if args.response_key not in dataset.columns:
        raise SystemExit(
            f"column {args.response_key!r} not in {args.path} (columns: {list(dataset.columns)}). "
            f"Generate responses first -- see this module's docstring."
        )
    if args.max_prompts > 0:
        dataset = dataset.iloc[: args.max_prompts]

    has_extra = args.extra_info_key in dataset.columns
    tasks = [
        (
            dataset[args.data_source_key].iloc[i],
            list(dataset[args.response_key].iloc[i]),
            dataset[args.reward_model_key].iloc[i]["ground_truth"],
            dataset[args.extra_info_key].iloc[i] if has_extra else None,
        )
        for i in range(len(dataset))
    ]

    n_samples = len(tasks[0][1])
    print(f"scoring {len(tasks)} prompts x {n_samples} samples with {args.num_workers} workers ...")

    per_source = defaultdict(list)  # data_source -> list of per-prompt score lists
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=_init_scorer,
        initargs=(args.custom_reward_path, args.custom_reward_name),
    ) as pool:
        futures = [pool.submit(_score_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            data_source, scores = future.result()
            per_source[data_source].append(scores)
            if done % 200 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    all_scores = np.array([s for lists in per_source.values() for scores in lists for s in scores], dtype=np.float64)
    # The group mean per prompt -- this is what the critic sees as the target spread.
    group_means = np.array([np.mean(scores) for lists in per_source.values() for scores in lists])

    mean_reward = float(all_scores.mean())
    is_binary = bool(np.all((all_scores == 0) | (all_scores == 1)))
    low, high = _wilson_interval(float(all_scores.sum()), all_scores.size) if is_binary else (float("nan"),) * 2

    print("\n" + "=" * 68)
    print("critic value-head bias")
    print("=" * 68)
    print(f"  mean reward over all samples : {mean_reward:.4f}   <-- value_head_bias_init")
    if is_binary:
        print(f"  95% CI (Wilson)              : [{low:.4f}, {high:.4f}]")
    print(f"  samples                      : {all_scores.size} ({len(group_means)} prompts x {n_samples})")
    print(f"  std of individual rewards    : {all_scores.std():.4f}")

    # Why the bias matters, in the two numbers the value loss actually opens at.
    e_r2 = float((all_scores**2).mean())
    var_r = float(all_scores.var())
    print("\n  initial value loss:")
    print(f"    bias = 0.0            -> E[r^2]  = {e_r2:.4f}")
    print(f"    bias = {mean_reward:.4f}         -> Var(r)  = {var_r:.4f}   ({e_r2 / max(var_r, 1e-9):.2f}x lower)")

    # Group-level diagnostics: DAPO drops zero-variance groups, so a high all-same fraction
    # means most generation batches get filtered and the effective batch size collapses.
    all_right = float(np.mean(group_means == 1.0))
    all_wrong = float(np.mean(group_means == 0.0))
    print("\n  group diagnostics (relevant to algorithm.filter_groups):")
    print(f"    all-correct groups : {all_right:.1%}")
    print(f"    all-wrong groups   : {all_wrong:.1%}")
    print(f"    kept by DAPO       : {1 - all_right - all_wrong:.1%}")
    if all_right + all_wrong > 0.8:
        print("    [warn] >80% of groups carry no gradient signal; raise data.gen_batch_size")
        print("           or pick a training set better matched to this model's difficulty.")

    if len(per_source) > 1:
        print("\n  per data_source:")
        for source, lists in sorted(per_source.items()):
            flat = np.array([s for scores in lists for s in scores])
            print(f"    {source:45s} {flat.mean():.4f}  (n={flat.size})")

    print("\n" + "=" * 68)
    print(f"  critic.model.value_head_bias_init={mean_reward:.2f}")
    print("=" * 68)

    if args.output_json:
        metrics = {
            "value_head_bias_init": round(mean_reward, 4),
            "mean_reward": mean_reward,
            "std_reward": float(all_scores.std()),
            "var_reward": var_r,
            "e_r_squared": e_r2,
            "num_samples": int(all_scores.size),
            "num_prompts": int(len(group_means)),
            "n_samples_per_prompt": n_samples,
            "all_correct_group_frac": all_right,
            "all_wrong_group_frac": all_wrong,
            "per_data_source": {
                source: float(np.mean([s for scores in lists for s in scores])) for source, lists in per_source.items()
            },
        }
        if is_binary:
            metrics["wilson_ci_95"] = [low, high]
        with open(args.output_json, "w") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"\nwrote {args.output_json}")


if __name__ == "__main__":
    main()
