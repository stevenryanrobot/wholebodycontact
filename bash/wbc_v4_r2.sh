#!/usr/bin/env bash
# Whole-Body Contact v4 round 2 — Track B (GRU recipe fix) + champion export.
# Run in tmux (window r2 of session wbc_v4). Track A (deploy re-eval of the
# round-1 checkpoints) is expected to have already appended to
# data/wbc/sweep_v4/deploy_leaderboard.jsonl.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source start_gentle_local.sh

python -u forcesense/train/sweep_v4_r2.py \
    --data data/wbc/datasets/wbc_v4_motion.h5 data/wbc/datasets/wbc_v4_static.h5 \
    --out_dir data/wbc/sweep_v4 2>&1 | tee data/wbc/logs/v4_r2_sweep.log \
    || { echo "[r2] FATAL: sweep failed"; exit 1; }

python -u forcesense/eval/eval_deploy.py \
    --data data/wbc/datasets/wbc_v4_motion.h5 data/wbc/datasets/wbc_v4_static.h5 \
    --ckpts 'data/wbc/sweep_v4/model_gru50r2_*.pt' \
    --out data/wbc/sweep_v4/deploy_leaderboard.jsonl 2>&1 | tee -a data/wbc/logs/v4_deploy_eval.log \
    || { echo "[r2] FATAL: deploy eval failed"; exit 1; }

python -u forcesense/train/champion.py 2>&1 | tee data/wbc/logs/v4_r2_champion.log

echo "[r2] ===== DONE ====="
