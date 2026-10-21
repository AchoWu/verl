#!/usr/bin/env bash
# Forgetting-gated self-imitation on verl 0.6, with JustRL-II's 16k setup (recipe/justrl2) for
# everything except the advantage: Dr. GRPO on the current rollouts (no critic, no GAE), plus
# replay of the policy's own correct solutions from LAG_STEPS updates ago on prompts it forgot.
#
#   bash recipe/self_imitation/run_self_imitation_16k.sh                    # start / resume
#   LAG_STEPS=32 bash recipe/self_imitation/run_self_imitation_16k.sh       # change a knob
#   bash recipe/self_imitation/run_self_imitation_16k.sh trainer.save_freq=100   # any override
#
# Controls (same script, same data, same rollout budget unless noted):
#   NUM_REPLAY=0                         plain Dr. GRPO-8 (half the rollouts)
#   NUM_REPLAY=0 N_RESP_PER_PROMPT=16    Dr. GRPO-16: equal rollout budget, the key control
#   REPLAY_COEF=0                        generates replay and logs the gate, trains on none of it
#   LAG_STEPS=0 REPLAY_COEF=0            null run: gate statistics when nothing was forgotten;
#                                        use self_imitation/delta_q95 to set GATE_TAU
#
# Data and reward are JustRL-II's, from recipe/justrl2 (prepare once with
# `bash recipe/justrl2/prepare_data.sh`; this recipe only reads them).
set -xeuo pipefail

trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python /group/40143/howu/verl/test_gpu.py
' EXIT INT TERM

project_name=${PROJECT_NAME:-'verl-self-imitation'}

# ---- self-imitation ---------------------------------------------------------------------
lag_steps=${LAG_STEPS:-16}            # k: replay is sampled by the policy k updates ago
num_replay=${NUM_REPLAY:-8}           # m: replay rollouts per prompt (0 = off)
replay_coef=${REPLAY_COEF:-1.0}       # lambda: replay term weight relative to the GRPO term
gate_tau=${GATE_TAU:-0.375}           # tau: replay iff p_old - p_cur >= tau

exp_name=${EXP_NAME:-"si-16k-r1-distill-1.5b-k${lag_steps}-m${num_replay}-coef${replay_coef}-tau${gate_tau}"}

# ---- 16k context budget (as justrl2) ------------------------------------------------------
max_prompt_length=$((1024 * 2))      # 2048
max_response_length=$((1024 * 14))   # 14336

# ---- PPO (as justrl2) ---------------------------------------------------------------------
actor_lr=1e-6
clip_ratio_low=0.2
clip_ratio_high=0.28            # clip-higher
clip_ratio_c=10.0
use_kl_loss=False
kl_loss_coef=0.0
entropy_coeff=0.0
loss_agg_mode="token-mean"
adam_betas="[0.9,0.98]"         # JustRL-II's beta2
# IcePop on the current rollouts only. Replay rows are deliberately not IS-corrected; see README.
rollout_is_threshold=5.0
rollout_is_threshold_lower=0.5

# ---- rollout / batching (as justrl2) -------------------------------------------------------
# 64 prompts x 8, 4 mini-batches of 16 prompts -> 4 actor updates per step. Admitted replay rows are
# spread over the same 4 updates. Each step also generates 64 x NUM_REPLAY replay rollouts.
# The mini-batch count is part of the method, not just a memory knob: the PPO clip on pi_theta/pi_prox
# only bounds the replay term from the second update of a rollout on. With 1 mini-batch (mini == bsz)
# dp_actor runs on-policy, the ratio is always 1, and replay becomes an unclipped imitation step.
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-64}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-16}
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.95

# ---- performance (as justrl2) -------------------------------------------------------------
sp_size=${SP_SIZE:-1}
use_dynamic_bsz=True
offload=True
gen_tp=${GEN_TP:-1}
max_num_seqs=${MAX_NUM_SEQS:-64}
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))

# ---- schedule (as justrl2) -----------------------------------------------------------------
# 1000 steps over 2 epochs of UltraData-RL-Math (506 steps / epoch).
total_epochs=${TOTAL_EPOCHS:-2}
total_training_steps=${TOTAL_TRAINING_STEPS:-1000}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:-50}

# ---- Ray / paths ------------------------------------------------------------------------
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# Import verl and recipe/ from THIS checkout (see run_justrl2_16k.sh).
export PYTHONPATH="${WORKING_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
verl_dir=$(python3 -c 'import os, verl; print(os.path.dirname(os.path.dirname(os.path.realpath(verl.__file__))))')
if [ "${verl_dir}" != "$(realpath "${WORKING_DIR}")" ]; then
    echo "FATAL: verl resolves to ${verl_dir}, not ${WORKING_DIR}. Run from the repo root." >&2
    exit 1
fi
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/recipe/dapo/runtime_env.yaml"}
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
RAY_DATA_HOME=${RAY_DATA_HOME:-"/group/40143/howu/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/llms/DeepSeek-R1-Distill-Qwen-1.5B"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/ultradata-rl-math.parquet"}
AIME24_FILE=${AIME24_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
AIME25_FILE=${AIME25_FILE:-"${RAY_DATA_HOME}/data/aime-2025.parquet"}
AIME26_FILE=${AIME26_FILE:-"${RAY_DATA_HOME}/data/aime-2026.parquet"}
TEST_FILES=${TEST_FILES:-"['${AIME24_FILE}','${AIME25_FILE}','${AIME26_FILE}']"}

LOGGER=${LOGGER:-'["console","wandb","file"]'}

# JustRL-II's 0/1 boxed math reward, shared with recipe/justrl2.
REWARD_FN_PATH=${REWARD_FN_PATH:-"${WORKING_DIR}/recipe/justrl2/reward_score.py"}

# ---- launch ------------------------------------------------------------------------------
SUBMIT=${SUBMIT:-0}
if [ "${SUBMIT}" = "1" ]; then
    launcher=(ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" --working-dir "${WORKING_DIR}" --)
else
    launcher=()
fi

${launcher[@]+"${launcher[@]}"} python3 -m recipe.self_imitation.main_self_imitation \
    --config-path="${WORKING_DIR}/recipe/self_imitation/config" \
    --config-name='self_imitation_trainer.yaml' \
    self_imitation.lag_steps=${lag_steps} \
    self_imitation.num_replay=${num_replay} \
    self_imitation.replay_coef=${replay_coef} \
    self_imitation.gate_tau=${gate_tau} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILES}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_is=True \
    algorithm.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_is_threshold_lower=${rollout_is_threshold_lower} \
    algorithm.rollout_is_mode=mask \
    algorithm.rollout_is_level=token \
    algorithm.rollout_is_veto_threshold=null \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.betas="${adam_betas}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.72 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    reward_model.reward_manager=dapo \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name=compute_score \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto "$@"
