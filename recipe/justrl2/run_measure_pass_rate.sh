#!/usr/bin/env bash
# Measure the base model's mean reward on the training set -> critic.model.value_head_bias_init
#
#   MODEL_PATH=/path/to/base-model TRAIN_FILE=/path/to/train.parquet \
#     bash recipe/justrl2/run_measure_pass_rate.sh
#
#   # a quick look first: 200 prompts instead of the whole set
#   MAX_PROMPTS=200 bash recipe/justrl2/run_measure_pass_rate.sh
#
#   # reuse an existing generation (skip the GPU step)
#   SKIP_GEN=1 GEN_OUTPUT=/tmp/passrate_gen.parquet bash recipe/justrl2/run_measure_pass_rate.sh
#
# Two stages: generate with the TRAINING sampling params, then score with verl's own scorer.
# The sampling params are not incidental -- the bias must equal E[r] under the rollout
# distribution, so a greedy decode would give the wrong number. Defaults below mirror
# run_justrl2_16k.sh; change them together with that script, never only here.
#
# Cost: N_SAMPLES x prompts generations at the full response length. Subsample with
# MAX_PROMPTS (a few hundred prompts already pins the mean to ~±0.03) rather than
# shortening RESPONSE_LENGTH, which would truncate answers and understate the pass rate.
set -xeuo pipefail

# ---- must match run_justrl2_16k.sh ------------------------------------------------------
PROMPT_LENGTH=${PROMPT_LENGTH:-2048}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-14336}
N_SAMPLES=${N_SAMPLES:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}

# ---- paths ------------------------------------------------------------------------------
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/llms/DeepSeek-R1-Distill-Qwen-1.5B"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
GEN_OUTPUT=${GEN_OUTPUT:-"/tmp/justrl2_passrate_gen.parquet"}
METRICS_JSON=${METRICS_JSON:-"/tmp/justrl2_passrate.json"}

# ---- resources --------------------------------------------------------------------------
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}

# ---- scoring ----------------------------------------------------------------------------
MAX_PROMPTS=${MAX_PROMPTS:--1}
NUM_WORKERS=${NUM_WORKERS:-$(( $(nproc 2>/dev/null || echo 8) / 2 ))}
# Must match the scorer training uses, or the measured mean is on the wrong scale and the bias
# is wrong. run_justrl2_16k.sh defaults to the 0/1 wrapper; REWARD_SCALE=pm1 drops back to the
# stock +/-1 scorer in both scripts.
REWARD_SCALE=${REWARD_SCALE:-unit}
if [ "${REWARD_SCALE}" = "unit" ]; then
    CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-"${PWD}/recipe/justrl2/reward_score.py"}
else
    CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-}
fi
CUSTOM_REWARD_NAME=${CUSTOM_REWARD_NAME:-compute_score}

SKIP_GEN=${SKIP_GEN:-0}

if [ "${SKIP_GEN}" != "1" ]; then
    # load_format=auto: generation-only runs have no trainer to push weights in, so the
    # rollout engine must load them itself (the training default is `dummy`).
    python3 -m verl.trainer.main_generation_server \
        trainer.nnodes="${NNODES}" \
        trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
        data.train_files="${TRAIN_FILE}" \
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
    --max-prompts "${MAX_PROMPTS}"
    --output-json "${METRICS_JSON}"
)
if [ -n "${CUSTOM_REWARD_PATH}" ]; then
    score_args+=(--custom-reward-path "${CUSTOM_REWARD_PATH}" --custom-reward-name "${CUSTOM_REWARD_NAME}")
fi

python3 -m recipe.justrl2.measure_pass_rate "${score_args[@]}" "$@"
