#!/usr/bin/env bash
set -euo pipefail

LOW_PROJECT_PATH="${LOW_PROJECT_PATH:-luoxinyuan-duke-university/gentle_humanoid}"
HL_WANDB_PROJECT="${HL_WANDB_PROJECT:-gentle_humanoid_high_level}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29501}"
NPROC="${NPROC:-4}"
SCRIPT="scripts/train.py"

LOW_RUN_PATH="${LOW_RUN_PATH:-${LOW_PROJECT_PATH}/gentle_finetune_root_wrist_12}"
TAG="${TAG:-hl_root_hold}"
SUFFIX="${SUFFIX:-root_wrist_12}"

if [[ $# -eq 1 ]]; then
  SUFFIX="$1"
elif [[ $# -ge 2 ]]; then
  TAG="$1"
  SUFFIX="$2"
fi

RUN_ID="${RUN_ID:-${TAG}_${SUFFIX}}"

cmd=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" "$SCRIPT"
  task=G1/G1_hl_root_hold
  algo=root_ppo
  task.action.low_policy.run_path="$LOW_RUN_PATH"
  wandb.project="$HL_WANDB_PROJECT"
  wandb.id="$RUN_ID"
)

echo ">>> ${cmd[*]}"
"${cmd[@]}"
