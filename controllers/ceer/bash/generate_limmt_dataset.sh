#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=/home/dexlab/retargetted/LIMMT/amass_train_filtered
OUT_DIR=data/dataset/limmt_amass_train_filtered
PYTHON=${PYTHON:-/home/dexlab/miniconda3/envs/gentle/bin/python}

mkdir -p "$OUT_DIR"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" controllers/ceer/scripts/data_process/generate_dataset.py \
  --dataset-root "$DATASET_ROOT" \
  --mem-path "$OUT_DIR"
