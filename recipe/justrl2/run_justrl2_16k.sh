#!/usr/bin/env bash
# JustRL-II on verl: PPO with a scalar value-head critic, length-adaptive GAE
# (lambda_i = k ** (1 / L_i)) and a mean-reward-seeded value head.
#
#   bash recipe/justrl2/run_justrl2_16k.sh                 # start / resume
#   NNODES=2 bash recipe/justrl2/run_justrl2_16k.sh        # multi-node
#   bash recipe/justrl2/run_justrl2_16k.sh critic.optim.lr=1e-5   # override any knob
#
# Runs in the foreground by default: main_ppo calls ray.init() itself, so no cluster has to be
# started first, output is live, and Ctrl-C stops it. Set SUBMIT=1 to dispatch through
# `ray job submit` against an existing cluster at RAY_ADDRESS instead (needed for multi-node).
#
# Before the first run, measure the critic's prior on your own data:
#   bash recipe/justrl2/run_measure_pass_rate.sh
# and pass the number it prints as value_head_bias_init (see README.md section 2).
#
# 16k context (2k prompt + 14k response). NOTE this is a shortened-context variant of the
# published 128k recipe, not a reproduction: lambda_i = k ** (1 / L_i) takes the response
# length as its input, so the same k acts on a different length distribution and the curves
# are not comparable to the 32k / 128k results. See recipe/justrl2/README.md.
#
# The four mechanisms and where they live:
#   1. scalar value-head critic   critic.enable + model_type=value_model (trainer-side)
#   2. value-head seeding         critic.model.value_head_bias_init
#   3. length-adaptive GAE        algorithm.adv_estimator + algorithm.gae_lambda_k
#   4. critic-only warmup         trainer.critic_warmup
#   5. penalty kept out of critic reward.reward_kwargs.overlong_buffer_cfg.exclude_from_critic
set -xeuo pipefail

project_name='verl-justrl2'
exp_name='justrl2-16k-r1-distill-1.5b'

# ---- JustRL-II core ---------------------------------------------------------------------
adv_estimator=gae_length_adaptive
gamma=1.0                       # must be 1: the decoupled returns are the suffix reward sum
gae_lambda_k=0.5                # fraction of terminal credit reaching the first token
critic_warmup=30                # rollouts where only the critic is updated
# The critic's prior: E[r] under the training sampling params. On this recipe's 0/1 reward
# scale that is just the base model's pass rate, and V(s) reads as "probability this prefix
# ends up correct". 0.52 is JustRL-II's number for THEIR model and data -- measure yours with
# `bash recipe/justrl2/run_measure_pass_rate.sh`. A wrong value is not fatal but wastes the
# warmup. See README.md section 2.
value_head_bias_init=${VALUE_HEAD_BIAS_INIT:-0.52}

# ---- 16k context budget -----------------------------------------------------------------
max_prompt_length=$((1024 * 2))      # 2048
max_response_length=$((1024 * 14))   # 14336
enable_overlong_buffer=True
overlong_buffer_len=2867             # 20% of the response budget (DAPO soft overlong)
# The penalty is subtracted from the reward, so its weight is relative to the reward span. On
# the 0/1 scale this recipe uses, factor=1.0 means a maximally-overlong correct answer scores
# 0 -- exactly as bad as a wrong one. That matches JustRL-II, whose reward is also 0/1. If you
# switch to REWARD_SCALE=pm1 the same 1.0 only costs half the span, so the two knobs have to
# move together.
overlong_penalty_factor=1.0
exclude_overlong_from_critic=True    # critic regresses reward WITHOUT the length shaping

# ---- PPO --------------------------------------------------------------------------------
actor_lr=1e-6
critic_lr=5e-6                  # 5x the actor: the critic chases a moving policy
critic_lr_warmup_steps=10
clip_ratio_low=0.2
clip_ratio_high=0.28            # clip-higher
cliprange_value=0.2
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
entropy_coeff=0.0
loss_agg_mode="token-mean"

# ---- rollout / batching -----------------------------------------------------------------
# GBS = 32 * 8 = 256. JustRL-II's release recipe is 60 * 8 = 480 on 16 nodes; its 1-node 16k
# variant drops to 8 * 8 = 64, which is a memory concession for MiniCPM-2B at 16k rather than
# an algorithmic choice. 256 keeps the recipe's large-batch intent at a cost a single node can
# carry with half its GPUs on the critic.
train_prompt_bsz=32
n_resp_per_prompt=8
train_prompt_mini_bsz=32
# Over-sampling only pays off when groups get dropped. With filtering off, every generated
# group trains, so generating extra prompts is wasted rollout time -- keep gen == train.
enable_filter_groups=False
filter_groups_metric=acc
max_num_gen_batches=10
gen_prompt_bsz=${train_prompt_bsz}
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.95

# ---- performance ------------------------------------------------------------------------
# sp_size=1 for a 1.5B model: it fits without sequence parallelism, and skipping it doubles the
# actor's data-parallel width instead of spending it on intra-sequence communication.
sp_size=${SP_SIZE:-1}
use_dynamic_bsz=True
offload=True
gen_tp=${GEN_TP:-1}
# Rollout concurrency. JustRL-II caps this at 24 because MiniCPM-2B's KV cache at 16k barely
# fits; R1-Distill-1.5B uses GQA with 2 KV heads, so one 16k sequence costs ~0.44 GB and the
# pool holds well over a hundred. 24 would throttle a 256-sample step to a fifth of capacity.
max_num_seqs=${MAX_NUM_SEQS:-64}
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
# The critic forward is cheaper than the actor's (a [1, H] head instead of [vocab, H]), so it
# tolerates a larger token budget per GPU.
critic_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2 / sp_size))

# ---- schedule ---------------------------------------------------------------------------
total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:-25}
# Aligned with save_freq so every checkpoint has a matching eval point.
test_freq=${TEST_FREQ:-25}

# ---- Ray / paths ------------------------------------------------------------------------
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/llms/DeepSeek-R1-Distill-Qwen-1.5B"}
# The critic starts from the same base weights; HF builds its scalar head fresh (the policy
# checkpoint has no [1, H] head), which is why seeding at construction is enough on FSDP.
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-"${MODEL_PATH}"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
# Validation suite: three AIME years. verl keys val metrics by `data_source`, so each file
# reports its own score (val-core/<data_source>/...) instead of one blended number. The scorer
# is picked from that same field, and anything starting with "aime" dispatches to math_dapo
# (verl/utils/reward_score/__init__.py), so these files work without extra wiring.
AIME24_FILE=${AIME24_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
AIME25_FILE=${AIME25_FILE:-"${RAY_DATA_HOME}/data/aime-2025.parquet"}
AIME26_FILE=${AIME26_FILE:-"${RAY_DATA_HOME}/data/aime-2026.parquet"}
TEST_FILES=${TEST_FILES:-"['${AIME24_FILE}','${AIME25_FILE}','${AIME26_FILE}']"}

# ---- tracking -----------------------------------------------------------------------------
# wandb logs under trainer.project_name / trainer.experiment_name (there is no separate wandb
# project knob). Export WANDB_API_KEY, or `wandb login`, before submitting -- the Ray workers
# inherit it. LOGGER='["console"]' turns wandb off.
LOGGER=${LOGGER:-'["console","wandb"]'}

# ---- reward scale -------------------------------------------------------------------------
# 0/1 instead of math_dapo's native +/-1, so the critic's V(s) reads as "probability this
# prefix ends up correct" -- at the prompt that is a difficulty estimate, and the value-head
# seeding argument works at full strength (E[r^2] = E[r] for a Bernoulli reward, versus
# E[r^2] = 1 regardless on +/-1). REWARD_SCALE=pm1 reverts to the stock scorer.
REWARD_SCALE=${REWARD_SCALE:-unit}
reward_fn_args=()
if [ "${REWARD_SCALE}" = "unit" ]; then
    reward_fn_args=(
        reward.custom_reward_function.path="${WORKING_DIR}/recipe/justrl2/reward_score.py"
        reward.custom_reward_function.name=compute_score
    )
fi

# ---- launch ------------------------------------------------------------------------------
# Default: run in the foreground. main_ppo calls ray.init() itself, so a local run needs no
# pre-started cluster. SUBMIT=1 goes through `ray job submit` against RAY_ADDRESS instead,
# which is what multi-node needs (and what detaches the job from this shell).
SUBMIT=${SUBMIT:-0}
if [ "${SUBMIT}" = "1" ]; then
    launcher=(ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" --working-dir "${WORKING_DIR}" --)
else
    launcher=()
fi

# ${launcher[@]+"${launcher[@]}"} rather than "${launcher[@]}": expanding an empty array trips
# `set -u` on bash < 4.4, which some cluster images still ship.
${launcher[@]+"${launcher[@]}"} python3 -m verl.trainer.main_ppo \
    --config-path="${WORKING_DIR}/recipe/justrl2/config" \
    --config-name='justrl2_trainer.yaml' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILES}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.gamma=${gamma} \
    algorithm.gae_lambda_k=${gae_lambda_k} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
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
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    critic.enable=True \
    critic.model.path="${CRITIC_MODEL_PATH}" \
    critic.model.value_head_bias_init=${value_head_bias_init} \
    critic.model.use_remove_padding=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.optim.lr=${critic_lr} \
    critic.optim.lr_warmup_steps=${critic_lr_warmup_steps} \
    critic.optim.weight_decay=0.1 \
    critic.cliprange_value=${cliprange_value} \
    critic.loss_agg_mode=${loss_agg_mode} \
    critic.use_dynamic_bsz=${use_dynamic_bsz} \
    critic.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    critic.ppo_max_token_len_per_gpu=${critic_ppo_max_token_len} \
    critic.ulysses_sequence_parallel_size=${sp_size} \
    critic.model.fsdp_config.param_offload=${offload} \
    critic.model.fsdp_config.optimizer_offload=${offload} \
    reward.reward_manager.name=dapo \
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    reward.reward_kwargs.overlong_buffer_cfg.log=True \
    reward.reward_kwargs.overlong_buffer_cfg.exclude_from_critic=${exclude_overlong_from_critic} \
    reward.reward_kwargs.max_resp_len=${max_response_length} \
    ${reward_fn_args[@]+"${reward_fn_args[@]}"} \
    trainer.critic_warmup=${critic_warmup} \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto "$@"
