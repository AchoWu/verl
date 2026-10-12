#!/usr/bin/env bash
# Measure the base model's mean reward on the training set -> critic.model.value_head_bias_init
#
#   MODEL_PATH=/path/to/base-model TRAIN_FILE=/path/to/train.parquet \
#     bash recipe/justrl2/run_measure_pass_rate.sh            # 300 random prompts (default)
#
#   MAX_PROMPTS=1000 bash recipe/justrl2/run_measure_pass_rate.sh   # tighter estimate
#
#   # reuse an existing generation (skip the GPU step)
#   SKIP_GEN=1 GEN_OUTPUT=/tmp/passrate_gen.parquet bash recipe/justrl2/run_measure_pass_rate.sh
#
# Two stages: generate with the TRAINING sampling params, then score with the training scorer.
# The sampling params are not incidental -- the bias must equal E[r] under the rollout
# distribution, so a greedy decode would give the wrong number. Defaults below mirror
# run_justrl2_16k.sh; change them together with that script, never only here.
#
# Cost: N_SAMPLES x MAX_PROMPTS generations at the full response length. MAX_PROMPTS prompts are
# drawn at random (seed SAMPLE_SEED) BEFORE generation. 300 prompts put the standard error of
# the mean at <= 0.029 even in the worst case (95% CI <= +/-0.057), which is far tighter than
# the bias needs. Do not shorten RESPONSE_LENGTH instead: that truncates answers and understates
# the pass rate. MAX_PROMPTS=-1 generates for the whole training set (32,412 x 8 requests).
set -xeuo pipefail
trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python /group/40143/howu/verl/test_gpu.py
' EXIT INT TERM

# Import verl and recipe/ from THIS checkout (run from the repo root), ahead of any other
# `pip install -e` of verl in the env; see run_justrl2_16k.sh.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
verl_dir=$(python3 -c 'import os, verl; print(os.path.dirname(os.path.dirname(os.path.realpath(verl.__file__))))')
if [ "${verl_dir}" != "$(realpath "${PWD}")" ]; then
    echo "FATAL: verl resolves to ${verl_dir}, not ${PWD}. Run from the repo root." >&2
    exit 1
fi

# ---- must match run_justrl2_16k.sh ------------------------------------------------------
PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-14336}
N_SAMPLES=${N_SAMPLES:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}

# ---- paths ------------------------------------------------------------------------------
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/llms/DeepSeek-R1-Distill-Qwen-1.5B"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/ultradata-rl-math.parquet"}
GEN_INPUT=${GEN_INPUT:-"/tmp/justrl2_passrate_prompts.parquet"}
GEN_OUTPUT=${GEN_OUTPUT:-"/tmp/justrl2_passrate_gen.parquet"}
METRICS_JSON=${METRICS_JSON:-"/tmp/justrl2_passrate.json"}

# ---- resources --------------------------------------------------------------------------
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}

# ---- sampling / scoring -----------------------------------------------------------------
MAX_PROMPTS=${MAX_PROMPTS:-300}
SAMPLE_SEED=${SAMPLE_SEED:-42}
NUM_WORKERS=${NUM_WORKERS:-$(( $(nproc 2>/dev/null || echo 8) / 2 ))}
# Must be the scorer training uses (run_justrl2_16k.sh's REWARD_FN_PATH), or the measured mean is
# not the reward the critic will regress and the bias is wrong.
CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-"${PWD}/recipe/justrl2/reward_score.py"}
CUSTOM_REWARD_NAME=${CUSTOM_REWARD_NAME:-compute_score}

SKIP_GEN=${SKIP_GEN:-0}

if [ "${SKIP_GEN}" != "1" ]; then
    # Subsample BEFORE generating. main_generation_server generates for every row of the file it
    # is given, so passing the full training set means 32,412 x N_SAMPLES requests.
    n_prompts=$(TRAIN_FILE="${TRAIN_FILE}" GEN_INPUT="${GEN_INPUT}" MAX_PROMPTS="${MAX_PROMPTS}" \
        SAMPLE_SEED="${SAMPLE_SEED}" python3 - <<'PY'
import os

import pandas as pd

df = pd.read_parquet(os.environ["TRAIN_FILE"])
k = int(os.environ["MAX_PROMPTS"])
if 0 < k < len(df):
    df = df.sample(n=k, random_state=int(os.environ["SAMPLE_SEED"])).reset_index(drop=True)
df.to_parquet(os.environ["GEN_INPUT"])
print(len(df))
PY
)
    echo "generating for ${n_prompts} prompts x ${N_SAMPLES} samples -> ${GEN_OUTPUT}"

    # main_generation_server keeps up to VERL_GEN_MAX_CONNECTIONS (default 256) connections open
    # per replica, so the client needs about replicas x 256 file descriptors on top of its own.
    # Raise the soft limit for this shell and its children (the Ray cluster ray.init starts
    # below); stop any already-running cluster first, since its processes keep their old limit.
    fds_needed=$(( NNODES * NGPUS_PER_NODE / ROLLOUT_TP * ${VERL_GEN_MAX_CONNECTIONS:-256} + 4096 ))
    fds_hard=$(ulimit -Hn)
    if [ "${fds_hard}" != "unlimited" ] && [ "${fds_hard}" -lt "${fds_needed}" ]; then
        echo "FATAL: need ~${fds_needed} open files, but the hard limit is ${fds_hard}. Lower" \
             "VERL_GEN_MAX_CONNECTIONS or raise the hard limit (ulimit -Hn)." >&2
        exit 1
    fi
    if [ "$(ulimit -Sn)" != "unlimited" ] && [ "$(ulimit -Sn)" -lt "${fds_needed}" ]; then
        ulimit -Sn "${fds_needed}"
    fi

    # load_format=auto: generation-only runs have no trainer to push weights in, so the
    # rollout engine must load them itself (the training default is `dummy`).
    python3 -m verl.trainer.main_generation_server \
        trainer.nnodes="${NNODES}" \
        trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
        data.train_files="${GEN_INPUT}" \
        data.prompt_key=prompt \
        +data.output_path="${GEN_OUTPUT}" \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.load_format=auto \
        actor_rollout_ref.rollout.n="${N_SAMPLES}" \
        actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
        actor_rollout_ref.rollout.top_p="${TOP_P}" \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.prompt_length="${PROMPT_LENGTH}" \
        actor_rollout_ref.rollout.response_length="${RESPONSE_LENGTH}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
        actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
        actor_rollout_ref.rollout.max_num_batched_tokens=$((PROMPT_LENGTH + RESPONSE_LENGTH))
fi

score_args=(
    --path "${GEN_OUTPUT}"
    --num-workers "${NUM_WORKERS}"
    --output-json "${METRICS_JSON}"
    --custom-reward-path "${CUSTOM_REWARD_PATH}"
    --custom-reward-name "${CUSTOM_REWARD_NAME}"
)
# The generation already holds exactly the sampled prompts; MAX_PROMPTS only trims a reused
# (SKIP_GEN=1) generation.
if [ "${SKIP_GEN}" = "1" ]; then
    score_args+=(--max-prompts "${MAX_PROMPTS}")
fi

python3 -m recipe.justrl2.measure_pass_rate "${score_args[@]}" "$@"
