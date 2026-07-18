#!/usr/bin/env bash
# Whole-Body Contact v4 pipeline (run inside tmux session `wbc_v4`):
#   1. motion-tracking collection  (30 links, long holds)   -> data/wbc/datasets/wbc_v4_motion.h5
#   2. static-standing collection  (web-demo scenario)      -> data/wbc/datasets/wbc_v4_static.h5
#   3. sweep: MLP baselines + GRU H=50 (+impedance-norm)    -> data/wbc/sweep_v4/
#   4. ONNX export of best overall + best-MLP fallback
#
# Stages are resumable: a stage is skipped if its output already exists and
# passes the sample-count check, so the chain can be relaunched after a crash
# without recollecting hours of data.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source start_gentle_local.sh

RUN_PATH="luoxinyuan-duke-university/gentle_humanoid/gentle_finetune_3point_amass_limmt_full_stiff30"
OVERRIDES="cfg/wbc/collect_v4.yaml"
NUM_ENVS=128
MOTION_STEPS=35000        # -> 35001*128 = 4.48M samples (~5.7 GB)
STATIC_STEPS=15000        # -> 15001*128 = 1.92M samples (~2.5 GB, ~30% of total)
MOTION_H5="data/wbc/datasets/wbc_v4_motion.h5"
STATIC_H5="data/wbc/datasets/wbc_v4_static.h5"
SWEEP_DIR="data/wbc/sweep_v4"

check_h5() {  # check_h5 <path> <min_samples>
    python - "$1" "$2" <<'PY'
import sys, h5py
path, need = sys.argv[1], int(sys.argv[2])
try:
    with h5py.File(path, "r") as h:
        n = h["X"].shape[0]
        k = int(h.attrs["num_bodies"])
except Exception as e:
    print(f"[check] {path}: unreadable ({e})"); sys.exit(1)
print(f"[check] {path}: {n} samples, K={k}")
sys.exit(0 if n >= need else 1)
PY
}

stage_collect() {  # stage_collect <h5> <steps> <log> [extra args...]
    local h5="$1" steps="$2" log="$3"; shift 3
    local need=$(( steps * NUM_ENVS ))
    if check_h5 "$h5" "$need" 2>/dev/null; then
        echo "[pipeline] $h5 already complete, skipping collection"
        return 0
    fi
    echo "[pipeline] collecting -> $h5 ($steps steps x $NUM_ENVS envs) $*"
    python -u forcesense/collect/isaac.py \
        -r "$RUN_PATH" --overrides "$OVERRIDES" \
        -n "$steps" --num_envs "$NUM_ENVS" -o "$h5" "$@" 2>&1 | tee "$log"
    # Isaac shutdown is occasionally unclean; trust the data check, not the exit code.
    check_h5 "$h5" "$need" || { echo "[pipeline] FATAL: $h5 incomplete"; exit 1; }
}

echo "[pipeline] ===== stage 1/4: motion collection ====="
stage_collect "$MOTION_H5" "$MOTION_STEPS" data/wbc/logs/v4_collect_motion.log --seed 0

echo "[pipeline] ===== stage 2/4: static collection ====="
stage_collect "$STATIC_H5" "$STATIC_STEPS" data/wbc/logs/v4_collect_static.log \
    --seed 1 --static_command --static_height 0.80

echo "[pipeline] ===== stage 3/4: sweep ====="
python -u forcesense/train/sweep_v4.py --data "$MOTION_H5" "$STATIC_H5" \
    --out_dir "$SWEEP_DIR" 2>&1 | tee data/wbc/logs/v4_sweep.log \
    || { echo "[pipeline] FATAL: sweep failed"; exit 1; }
[ -f "$SWEEP_DIR/force_sensor_v4_best.pt" ] || { echo "[pipeline] FATAL: no best model"; exit 1; }

echo "[pipeline] ===== stage 4/4: ONNX export ====="
python -u forcesense/export.py --ckpt "$SWEEP_DIR/force_sensor_v4_best.pt" \
    2>&1 | tee data/wbc/logs/v4_export.log
if [ -f "$SWEEP_DIR/force_sensor_v4_best_mlp.pt" ]; then
    python -u forcesense/export.py --ckpt "$SWEEP_DIR/force_sensor_v4_best_mlp.pt" \
        2>&1 | tee -a data/wbc/logs/v4_export.log
fi

echo "[pipeline] ===== DONE ====="
ls -la "$SWEEP_DIR"
