#!/usr/bin/env bash
set -euo pipefail

LOW_PROJECT_PATH="luoxinyuan-duke-university/gentle_humanoid"
HL_WANDB_PROJECT="gentle_humanoid_high_level"
CUDA_VISIBLE_DEVICES="0,1,2,3"
MASTER_PORT="29502"
NPROC="4"

TASK="G1/G1_hl_root_hold"
LOW_RUN_PATH="${LOW_PROJECT_PATH}/gentle_finetune_root_wrist_12"
RUN_ID="hl_root_hold_root_wrist_12"

export CUDA_VISIBLE_DEVICES

cmd=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" controllers/ceer/scripts/train.py
  task="$TASK"
  algo=root_ppo
  task.action.low_policy.run_path="$LOW_RUN_PATH"
  wandb.project="$HL_WANDB_PROJECT"
  wandb.id="$RUN_ID"
)

echo ">>> ${cmd[*]}"
"${cmd[@]}"
