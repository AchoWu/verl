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

# 默认跑 get_question：题面不在 prompt 里，模型必须调用工具才能拿到题目。
# 换回原来的 gsm8k 答案检查器：
#   TOOL_CONFIG=examples/sglang_multiturn/config/tool_config/gsm8k_tool_config.yaml \
#   DATA_DIR=$HOME/data/gsm8k_tool bash train_agent_demo.sh
TOOL_CONFIG=${TOOL_CONFIG:-recipe/get_question/tool_config.yaml}
DATA_DIR=${DATA_DIR:-$HOME/data/gsm8k_get_question}
# 工具回复长度上限，超过会被从中间截断（框架默认 256，这里放到 1024 装得下 gsm8k 题面）
TOOL_RESP_LEN=${TOOL_RESP_LEN:-1024}

# TOOL_CONFIG 允许传绝对路径；相对路径按仓库根目录解析
case "${TOOL_CONFIG}" in
    /*) TOOL_CONFIG_PATH="${TOOL_CONFIG}" ;;
    *)  TOOL_CONFIG_PATH="$(pwd)/${TOOL_CONFIG}" ;;
esac

# 每步的完整轨迹（prompt / 模型输出 / 工具调用 / 工具回复）会写成 <step>.jsonl。
# 按模型分目录，避免覆盖之前那次的 1.jsonl
DUMP_DIR=${DUMP_DIR:-/group/40143/howu/rollout_dump/$(basename "${MODEL_PATH}")}

bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    trainer.logger='["console"]' \
    actor_rollout_ref.rollout.trace.backend=null \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    data.max_response_length=${RESP_LEN} \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=11 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${TOOL_RESP_LEN} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ} \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.rollout_data_dir="${DUMP_DIR}" \
    trainer.total_training_steps=3 \
    trainer.n_gpus_per_node=8 \
    "$@"

