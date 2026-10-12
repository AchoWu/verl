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
"""Check a Python env against what this verl-0.6 checkout was built for, before a long run.

    python3 -m recipe.justrl2.check_env              # from the repo root
    python3 -m recipe.justrl2.check_env --offline    # skip the PyPI release-date scan

This branch sits on verl commit 22d082f9 (2025-10-15), whose image is
verl0.6-cu128-torch2.8.0-fa2.7.4 with vLLM 0.11.0. An env assembled later picks up newer
releases, and three of them already broke this code path at runtime (uvicorn >= 0.41 closing the
HTTP listener, vLLM 0.11's `non_block`, uvloop >= 0.22's get_event_loop). The checks:

1. version ranges for the packages the training path depends on;
2. the three known incompatibilities: if the triggering version is installed, the fix must be in
   the source;
3. importing every module the training run loads, plus CUDA / flash-attn;
4. every installed distribution whose installed version was released after the commit date --
   the candidates for the next incompatibility. Needs pypi.org; --offline skips it.

Exit status is 1 if any FAIL line was printed.
"""

import argparse
import datetime
import importlib
import importlib.metadata as md
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from packaging.specifiers import SpecifierSet
from packaging.version import Version

COMMIT_DATE = datetime.datetime(2025, 10, 15, tzinfo=datetime.timezone.utc)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# distribution -> (allowed versions, why)
EXPECTED = {
    "torch": ("==2.8.*", "verl 0.6 image; vLLM 0.11.0 is built against torch 2.8"),
    "vllm": ("==0.11.0", "verl 0.6 image; the async server patches target 0.11.0"),
    "transformers": (">=4.56,<5", "vLLM 0.11.0 needs >=4.56; verl 0.6 imports AutoModelForVision2Seq, gone in 5"),
    "tensordict": (">=0.8.0,<=0.10.0,!=0.9.0", "verl setup.py"),
    "numpy": ("<2", "verl setup.py"),
    "ray": (">=2.41.0", "verl setup.py"),
    "pyarrow": (">=19.0.0", "verl setup.py"),
    "flash-attn": (">=2.7.4", "remove-padding path; must be built for this torch"),
    "math-verify": ("", "JustRL-II grader fallback; missing = silent under-scoring"),
    "pylatexenc": ("", "JustRL-II grader"),
    "sympy": ("", "JustRL-II grader"),
}

# Modules the training run and the scorer import.
MODULES = [
    "flash_attn",
    "flash_attn.bert_padding",
    "vllm",
    "verl.trainer.ppo.ray_trainer",
    "verl.workers.fsdp_workers",
    "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
    "verl.workers.rollout.vllm_rollout.vllm_async_server",
    "verl.workers.reward_manager.dapo",
    "recipe.dapo.dapo_ray_trainer",
    "recipe.dapo.main_dapo",
    "recipe.justrl2.reward_score",
]

failures = []


def report(level, msg):
    if level == "FAIL":
        failures.append(msg)
    print(f"  {level:<4} {msg}")


def installed(dist):
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def source(relpath):
    with open(os.path.join(REPO_ROOT, relpath), encoding="utf-8") as f:
        return f.read()


def check_versions():
    print("1. versions")
    py = ".".join(map(str, sys.version_info[:3]))
    report("OK" if (3, 10) <= sys.version_info[:2] <= (3, 12) else "WARN", f"python {py} (image: 3.11)")
    for dist, (spec, why) in EXPECTED.items():
        version = installed(dist)
        if version is None:
            report("FAIL", f"{dist} not installed -- {why}")
        elif spec and Version(version) not in SpecifierSet(spec):
            report("FAIL", f"{dist} {version}, expected {spec} -- {why}")
        else:
            report("OK", f"{dist} {version}")


def check_known_breakages():
    print("2. known incompatibilities")

    uvicorn_v = installed("uvicorn")
    patched = "_serve_until_started" in source("verl/workers/rollout/utils.py")
    if uvicorn_v and Version(uvicorn_v) >= Version("0.41") and not patched:
        report("FAIL", f"uvicorn {uvicorn_v} closes run_unvicorn's listener; rollout/utils.py is unpatched")
    else:
        report("OK", f"uvicorn {uvicorn_v} (run_unvicorn patched: {patched})")

    uvloop_v = installed("uvloop")
    patched = "uvloop >= 0.22" in source("verl/workers/fsdp_workers.py")
    if uvloop_v and Version(uvloop_v) >= Version("0.22") and not patched:
        report("FAIL", f"uvloop {uvloop_v} raises in get_event_loop(); verl/workers/fsdp_workers.py is unpatched")
    else:
        report("OK", f"uvloop {uvloop_v} (init_model loop fallback patched: {patched})")

    try:
        import inspect

        from vllm.v1.executor.abstract import Executor

        needs = "non_block" in inspect.signature(Executor.collective_rpc).parameters
        patched = "non_block: bool = False" in source("verl/workers/rollout/vllm_rollout/vllm_async_server.py")
        if needs and not patched:
            report("FAIL", "vLLM passes collective_rpc(non_block=...); verl's ZeroMQ executor is unpatched")
        else:
            report("OK", f"vLLM collective_rpc non_block (needed: {needs}, patched: {patched})")
    except Exception as e:
        report("FAIL", f"cannot inspect vLLM's Executor: {type(e).__name__}: {e}")

    # A second verl on the path silently runs other code (seen once already on this host).
    import verl

    verl_root = os.path.dirname(os.path.dirname(os.path.realpath(verl.__file__)))
    if os.path.realpath(verl_root) != os.path.realpath(REPO_ROOT):
        report("FAIL", f"verl imports from {verl_root}, not this checkout {REPO_ROOT}")
    else:
        report("OK", f"verl imports from this checkout ({REPO_ROOT})")


def check_imports():
    print("3. imports / CUDA")
    for name in MODULES:
        try:
            importlib.import_module(name)
            report("OK", f"import {name}")
        except Exception as e:
            first = str(e).splitlines()[0] if str(e) else ""
            report("FAIL", f"import {name}: {type(e).__name__}: {first}")
    try:
        import torch

        if torch.cuda.is_available():
            report("OK", f"CUDA {torch.version.cuda}, {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}")
        else:
            report("FAIL", "torch.cuda.is_available() is False")
    except Exception as e:
        report("FAIL", f"torch: {type(e).__name__}: {e}")


def _upload_time(dist_version):
    dist, version = dist_version
    url = f"https://pypi.org/pypi/{dist}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            files = json.load(resp).get("urls", [])
        times = [f["upload_time_iso_8601"] for f in files if f.get("upload_time_iso_8601")]
        if not times:
            return dist, version, None
        first = min(datetime.datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times)
        return dist, version, first
    except Exception:
        return dist, version, None


def check_release_dates(show_all):
    print(f"4. installed versions released after the verl commit ({COMMIT_DATE.date()})")
    dists = sorted({(d.metadata["Name"].lower(), d.version) for d in md.distributions() if d.metadata["Name"]})
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_upload_time, dists))
    known = [r for r in results if r[2] is not None]
    if len(known) < len(results) / 2:
        report("WARN", f"PyPI unreachable for {len(results) - len(known)}/{len(results)} packages; rerun with access")
    newer = sorted((r for r in known if r[2] > COMMIT_DATE), key=lambda r: r[2])
    on_path = [r for r in newer if _canonical(r[0]) in RUNTIME_PATH]
    for dist, version, when in on_path:
        report("WARN", f"{dist} {version} (released {when.date()}) -- on the training/serving path")
    rest = [r for r in newer if _canonical(r[0]) not in RUNTIME_PATH]
    if show_all:
        for dist, version, when in rest:
            report("INFO", f"{dist} {version} (released {when.date()})")
    elif rest:
        report("INFO", f"{len(rest)} more newer packages off the training path (--all lists them)")
    if not newer:
        report("OK", f"none of {len(known)} resolved packages is newer than the commit")


def _canonical(name):
    return name.lower().replace("_", "-").replace(".", "-")


# Distributions the training run, the rollout servers or the scorer actually exercise.
RUNTIME_PATH = {
    _canonical(n)
    for n in (
        "torch triton vllm flashinfer-python xformers flash-attn transformers tokenizers safetensors "
        "huggingface-hub accelerate peft datasets tensordict torchdata numpy pandas pyarrow ray "
        "cloudpickle msgpack msgspec pyzmq aiohttp fastapi starlette uvicorn uvloop pydantic "
        "pydantic-core openai hydra-core omegaconf codetiming wandb liger-kernel math-verify "
        "latex2sympy2-extended antlr4-python3-runtime sympy pylatexenc"
    ).split()
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="skip the PyPI release-date scan")
    parser.add_argument("--all", action="store_true", help="also list newer packages off the training path")
    args = parser.parse_args()

    check_versions()
    check_known_breakages()
    check_imports()
    if not args.offline:
        check_release_dates(args.all)

    print()
    if failures:
        print(f"{len(failures)} FAIL -- fix these before training.")
        sys.exit(1)
    print("no FAIL. WARN lines in section 4 are candidates to watch, not known breakages.")


if __name__ == "__main__":
    main()
