set -euo pipefail

trap '
    echo ""
    echo "============================================================"
    echo "[$(date "+%F %T")] Script exiting, start GPU occupation..."
    echo "============================================================"
    python /group/40143/howu/verl/test_gpu.py
' EXIT INT TERM

# 示例脚本用 $(pwd) 拼工具配置的路径，必须在仓库根目录下运行
cd "$(dirname "$(readlink -f "$0")")"

# Qwen3-4B 默认开思考（不传 enable_thinking 即为开）。思考很长，所以响应长度放到 4096，
# micro batch 相应减到 8。换回 Qwen2.5：
#   MODEL_PATH=/dev/shm/llms/Qwen2.5-3B-Instruct RESP_LEN=2048 MICRO_BSZ=16 bash train_agent_demo.sh
MODEL_PATH=${MODEL_PATH:-/dev/shm/llms/Qwen3-4B}
RESP_LEN=${RESP_LEN:-4096}
MICRO_BSZ=${MICRO_BSZ:-8}

# 每步的完整轨迹（prompt / 模型输出 / 工具调用 / 工具回复）会写成 <step>.jsonl。
# 按模型分目录，避免覆盖之前那次的 1.jsonl
DUMP_DIR=${DUMP_DIR:-/group/40143/howu/rollout_dump/$(basename "${MODEL_PATH}")}

bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files=$HOME/data/gsm8k_tool/train.parquet \
    data.val_files=$HOME/data/gsm8k_tool/test.parquet \
    trainer.logger='["console"]' \
    actor_rollout_ref.rollout.trace.backend=null \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    data.max_response_length=${RESP_LEN} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=11 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.rollout_data_dir="${DUMP_DIR}" \
    trainer.total_training_steps=1 \
    trainer.n_gpus_per_node=8