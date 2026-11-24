#!/usr/bin/env bash
# Manju (漫剧) tagging, single turn: the prompt carries title, summary, script and the candidate
# labels, the model answers in <answer>, scored by set F1 (recipe/manju_tag/reward.py). No tool:
# standard GRPO as in examples/grpo_trainer/run_qwen3-8b.sh -- default ppo_trainer config, sync
# rollout, no agent loop.
#
#   python recipe/manju_tag/prepare_data_inline.py --input .../260914_train.jsonl --tags .../tags.txt \
#       --tokenizer /dev/shm/llms/Qwen3-4B            # once; prints prompt lengths in tokens
#   bash recipe/manju_tag/run_manju_tag_inline.sh
#   TOTAL_STEPS=1 bash recipe/manju_tag/run_manju_tag_inline.sh trainer.val_before_train=False   # smoke test
#
# Data lives in DATA_DIR, not in the repo: it holds internal content and the tag taxonomy.
set -xeuo pipefail

trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python /group/40143/howu/verl/test_gpu.py
' EXIT INT TERM

# The reward file is given as a repo-relative path.
cd "$(dirname "$(readlink -f "$0")")/../.."

MODEL_PATH=${MODEL_PATH:-/dev/shm/llms/Qwen3-4B}
DATA_DIR=${DATA_DIR:-$HOME/data/manju_tag_inline}
DUMP_DIR=${DUMP_DIR:-/group/40143/howu/rollout_dump/manju_tag_inline-$(basename "${MODEL_PATH}")}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TOTAL_STEPS=${TOTAL_STEPS:-null}

# ---- lengths --------------------------------------------------------------------------------
# The 情节 prompt is the longest: up to ~4300 chars of content plus 60 candidate descriptions
# (~5000 chars). Overlong prompts fail loudly instead of being filtered, which would drop 情节
# rows only.
max_prompt_length=${MAX_PROMPT_LEN:-8192}
max_response_length=${RESP_LEN:-4096}

# ---- batch ----------------------------------------------------------------------------------
train_batch_size=${TRAIN_BSZ:-128}    # ~6300 train rows -> ~49 steps per epoch
n_resp_per_prompt=${N_RESP:-8}
ppo_mini_batch_size=${MINI_BSZ:-64}
micro_bsz=${MICRO_BSZ:-4}             # sequences reach ~11k tokens

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz} \
    custom_reward_function.path="$(pwd)/recipe/manju_tag/reward.py" \
    custom_reward_function.name=compute_score \
    +custom_reward_function.reward_kwargs.require_tool_call=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=manju_tag \
    trainer.experiment_name="manju_tag_inline-$(basename "${MODEL_PATH}")-n${n_resp_per_prompt}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=-1 \
    trainer.rollout_data_dir="${DUMP_DIR}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_STEPS} \
    "$@"
