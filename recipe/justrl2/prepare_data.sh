#!/usr/bin/env bash
# Fetch everything recipe/justrl2 needs: JustRL-II's training set (UltraData-RL-2609, config
# Math) and AIME 2024/25/26, both in the prompt format JustRL-II uses.
#
#   RAY_DATA_HOME=/path/to/data-root bash recipe/justrl2/prepare_data.sh
#
# Writes into $RAY_DATA_HOME/data/, which is where run_justrl2_16k.sh looks by default.
# Already-present files are skipped; OVERWRITE=1 rebuilds them.
#
# The AIME files are BUILT rather than downloaded ready-made, for two reasons:
#
#   * They must match JustRL-II's eval sets: the same three sources, each problem followed by the
#     training prompt's "Please reason step by step, and put your final answer within \boxed{}."
#     line, so the boxed grader can score them and the numbers are comparable to upstream.
#   * verl keys validation metrics off `data_source`; each year gets its own, so the three
#     report separate curves instead of one blended score.
#
# prepare_aime_data.py handles both; see its docstring. AIME parquets written by an earlier
# version of it (DAPO "Answer:" template) are detected and rebuilt automatically.
set -euo pipefail

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
DATA_DIR="${RAY_DATA_HOME}/data"
OVERWRITE=${OVERWRITE:-0}

mkdir -p "${DATA_DIR}"

# ---- training set -------------------------------------------------------------------------
# The raw `query` becomes a single user turn, no system prompt -- exactly what JustRL-II feeds
# miles. See prepare_ultradata_math.py.
TRAIN_FILE="${DATA_DIR}/ultradata-rl-math.parquet"
force_flag=()
[ "${OVERWRITE}" = "1" ] && force_flag=(--force)
python3 -m recipe.justrl2.prepare_ultradata_math --out "${TRAIN_FILE}" ${force_flag[@]+"${force_flag[@]}"}

# ---- validation sets ----------------------------------------------------------------------
python3 -m recipe.justrl2.prepare_aime_data --out-dir "${DATA_DIR}" ${force_flag[@]+"${force_flag[@]}"}

echo
echo "data ready in ${DATA_DIR}:"
ls -la "${TRAIN_FILE}" "${DATA_DIR}"/aime-202*.parquet
echo
echo "next:"
echo "  MODEL_PATH=<base model> RAY_DATA_HOME=${RAY_DATA_HOME} \\"
echo "    MAX_PROMPTS=300 bash recipe/justrl2/run_measure_pass_rate.sh"
