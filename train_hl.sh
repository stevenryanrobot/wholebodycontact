#!/usr/bin/env bash
set -euo pipefail

LOW_PROJECT_PATH="luoxinyuan-duke-university/gentle_humanoid"
HL_WANDB_PROJECT="gentle_humanoid_high_level"
CUDA_VISIBLE_DEVICES="0,1,2,3"
MASTER_PORT="29502"
NPROC="4"

TASK="G1/G1_hl_force_resist_feet"
LOW_RUN_PATH="${LOW_PROJECT_PATH}/gentle_5kp_finetune_limmt_full_stiff30"
RUN_ID="hl_force_resist_root_feet_5kp_limmt_full_stiff30"

export CUDA_VISIBLE_DEVICES

cmd=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" scripts/train.py
  task="$TASK"
  algo=root_ppo
  task.action.low_policy.run_path="$LOW_RUN_PATH"
  wandb.project="$HL_WANDB_PROJECT"
  wandb.id="$RUN_ID"
)

echo ">>> ${cmd[*]}"
"${cmd[@]}"
