#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=/home/dexlab/retargetted/AMASS
OUT_DIR=data/dataset/limmt_no_foot_gentle_amass_full
ALLOWLIST=scripts/data_process/allowlist_limmt_no_foot_gentle_amass.json
PYTHON=${PYTHON:-/home/dexlab/miniconda3/envs/gentle/bin/python}

mkdir -p "$OUT_DIR"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" scripts/data_process/generate_dataset.py \
  --dataset-root "$DATASET_ROOT" \
  --allowlist "$ALLOWLIST" \
  --mem-path "$OUT_DIR"
