#!/usr/bin/env bash
# Fetch everything recipe/justrl2 needs: the DAPO-Math-17k training set and AIME 2024/25/26.
#
#   RAY_DATA_HOME=/path/to/data-root bash recipe/justrl2/prepare_data.sh
#
# Writes into $RAY_DATA_HOME/data/, which is where run_justrl2_16k.sh looks by default.
# Already-present files are skipped; OVERWRITE=1 rebuilds them.
#
# The AIME files are BUILT rather than downloaded ready-made, for two reasons:
#
#   * verl keys validation metrics off `data_source`, and the ready-made AIME-2024 parquet uses
#     `math_dapo` -- the value every other math set uses too. Three files sharing it collapse
#     into one blended score instead of one curve per year.
#   * The instruction wrapper around each problem decides how the model answers, so the three
#     years have to carry byte-identical wrappers or their scores are not comparable.
#
# prepare_aime_data.py handles both; see its docstring.
set -euo pipefail

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
DATA_DIR="${RAY_DATA_HOME}/data"
OVERWRITE=${OVERWRITE:-0}

mkdir -p "${DATA_DIR}"

# ---- training set -------------------------------------------------------------------------
TRAIN_FILE="${DATA_DIR}/dapo-math-17k.parquet"
if [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" = "1" ]; then
    echo "downloading DAPO-Math-17k -> ${TRAIN_FILE}"
    # -O to a temp name first: a half-written parquet left behind by an interrupted wget would
    # otherwise be treated as present on the next run and fail much later, inside the trainer.
    wget -q --show-progress -O "${TRAIN_FILE}.part" \
        "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
    mv "${TRAIN_FILE}.part" "${TRAIN_FILE}"
else
    echo "exists, skipping: ${TRAIN_FILE}"
fi

# ---- validation sets ----------------------------------------------------------------------
force_flag=()
[ "${OVERWRITE}" = "1" ] && force_flag=(--force)
python3 -m recipe.justrl2.prepare_aime_data --out-dir "${DATA_DIR}" ${force_flag[@]+"${force_flag[@]}"}

echo
echo "data ready in ${DATA_DIR}:"
ls -la "${DATA_DIR}"/dapo-math-17k.parquet "${DATA_DIR}"/aime-202*.parquet
echo
echo "next:"
echo "  MODEL_PATH=<base model> RAY_DATA_HOME=${RAY_DATA_HOME} \\"
echo "    MAX_PROMPTS=300 bash recipe/justrl2/run_measure_pass_rate.sh"
