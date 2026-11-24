#!/usr/bin/env bash
# Manju (漫剧) tagging with a tool: the prompt gives a collection_rowkey and a tag category, the
# model has to call get_collection_info for the content, then answers with labels scored by set F1.
#
#   python recipe/manju_tag/prepare_data.py --input .../260914_train.jsonl --tags .../tags.txt \
#       --tokenizer /dev/shm/llms/Qwen3-4B            # once; prints prompt lengths in tokens
#   bash recipe/manju_tag/run_manju_tag.sh
#   TOTAL_STEPS=1 bash recipe/manju_tag/run_manju_tag.sh trainer.val_before_train=False   # smoke test
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

# Tool class and reward file are given as repo-relative paths.
cd "$(dirname "$(readlink -f "$0")")/../.."

MODEL_PATH=${MODEL_PATH:-/dev/shm/llms/Qwen3-4B}
DATA_DIR=${DATA_DIR:-$HOME/data/manju_tag}
# tool_config.yaml reads collections.jsonl from ~/data/manju_tag; edit it there if DATA_DIR moves.
TOOL_CONFIG=${TOOL_CONFIG:-recipe/manju_tag/tool_config.yaml}
DUMP_DIR=${DUMP_DIR:-/group/40143/howu/rollout_dump/manju_tag-$(basename "${MODEL_PATH}")}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TOTAL_STEPS=${TOTAL_STEPS:-null}

# ---- lengths --------------------------------------------------------------------------------
# The 情节 prompt carries 60 candidate descriptions (~5000 chars), the longest of the four.
# Overlong prompts fail loudly instead of being filtered: filtering would drop 情节 rows only.
max_prompt_length=${MAX_PROMPT_LEN:-5120}
# The tool's reply (up to ~3k tokens of script) sits inside the response, before the answer.
max_response_length=${RESP_LEN:-8192}
# In chars, applied to the tool's text. The tool cuts the script at 4000, its reply stays < 4400.
max_tool_response_length=5000

# ---- batch ----------------------------------------------------------------------------------
train_batch_size=${TRAIN_BSZ:-128}    # ~6300 train rows -> ~49 steps per epoch
n_resp_per_prompt=${N_RESP:-8}
ppo_mini_batch_size=${MINI_BSZ:-64}
micro_bsz=${MICRO_BSZ:-2}             # sequences reach ~13k tokens

python3 -m verl.trainer.main_ppo \
    --config-path="$(pwd)/examples/sglang_multiturn/config" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
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
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz} \
    actor_rollout_ref.rollout.trace.backend=null \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$(pwd)/${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${max_tool_response_length} \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=3 \
    custom_reward_function.path="$(pwd)/recipe/manju_tag/reward.py" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=manju_tag \
    trainer.experiment_name="manju_tag-$(basename "${MODEL_PATH}")-n${n_resp_per_prompt}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=-1 \
    trainer.rollout_data_dir="${DUMP_DIR}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_STEPS} \
    "$@"
