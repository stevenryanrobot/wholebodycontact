#!/usr/bin/env bash
# Track C2: wait for (a) the MuJoCo-domain h5 (Track C1) and (b) the r2 sweep
# champion (bash/wbc_v4_r2.sh -> champion.json), then fine-tune the champion on
# the mixed Isaac+MuJoCo data and export force_sensor_v4c.onnx.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source scripts/start_gentle_local.sh

MJ_H5="data/wbc/datasets/wbc_v4_mujoco.h5"
CHAMPION="data/wbc/sweep_v4/champion.json"

echo "[c2] waiting for $MJ_H5 and $CHAMPION ..."
until [ -f "$MJ_H5" ] && [ -f "$CHAMPION" ]; do sleep 30; done
# champion.json is written atomically at the very end of the r2 chain; give the
# GPU a moment to be released by the r2 python process.
sleep 60
echo "[c2] prerequisites ready at $(date +%H:%M:%S); starting fine-tune"

python -u forcesense/train/finetune.py \
    --champion "$CHAMPION" --mj_data "$MJ_H5" \
    --out data/wbc/sweep_v4/force_sensor_v4c.pt \
    2>&1 | tee data/wbc/logs/v4c_finetune.log \
    || { echo "[c2] FATAL: fine-tune failed"; exit 1; }

echo "[c2] ===== DONE ====="
