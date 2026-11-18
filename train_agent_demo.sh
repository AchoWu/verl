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

# 每步的完整轨迹（prompt / 模型输出 / 工具调用 / 工具回复）会写成 <step>.jsonl
DUMP_DIR=${DUMP_DIR:-/group/40143/howu/rollout_dump}

bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.model.path=/dev/shm/llms/Qwen2.5-3B-Instruct \
    data.train_files=$HOME/data/gsm8k_tool/train.parquet \
    data.val_files=$HOME/data/gsm8k_tool/test.parquet \
    trainer.logger='["console"]' \
    actor_rollout_ref.rollout.trace.backend=null \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.rollout_data_dir="${DUMP_DIR}" \
    trainer.total_training_steps=1 \
    trainer.n_gpus_per_node=8