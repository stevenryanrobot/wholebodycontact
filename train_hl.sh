#!/usr/bin/env bash
set -euo pipefail

LOW_PROJECT_PATH="luoxinyuan-duke-university/gentle_humanoid"
HL_WANDB_PROJECT="gentle_humanoid_high_level"
CUDA_VISIBLE_DEVICES="0,1"
MASTER_PORT="29501"
NPROC="2"

ALGO="root_ppo"
TASK="G1/G1_hl_ee_compliance"
LOW_RUN_PATH="${LOW_PROJECT_PATH}/gentle_finetune_3point_amass_limmt_full_stiff30"
RUN_NAME="hl_ee_compliance_stiff200_3kp_priv_amass_limmt_full_stiff600"
RUN_ID="${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"

export CUDA_VISIBLE_DEVICES

cmd=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" scripts/train.py
  task="$TASK"
  algo="$ALGO"
  task.action.low_policy.run_path="$LOW_RUN_PATH"
  wandb.project="$HL_WANDB_PROJECT"
  wandb.id="$RUN_ID"
)

echo ">>> ${cmd[*]}"
"${cmd[@]}"
