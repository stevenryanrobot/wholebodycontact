#!/usr/bin/env bash
set -euo pipefail

LOW_PROJECT_PATH="luoxinyuan-duke-university/gentle_humanoid"
HL_WANDB_PROJECT="gentle_humanoid_high_level"
CUDA_VISIBLE_DEVICES="0"
MASTER_PORT="29504"
NPROC="1"

ALGO="root_ppo"
TASK="G1/G1_hl_ee_compliance"
LOW_RUN_PATH="${LOW_PROJECT_PATH}/gentle_finetune_3point_amass_limmt_full_stiff30"
RUN_NAME="whole_body_test"
RUN_ID="${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"

# The G1 task default is num_envs=16384, which OOMs a single 24GB RTX 4090.
# Cap it to a value that fits on one local GPU.
NUM_ENVS="${NUM_ENVS:-4096}"

export CUDA_VISIBLE_DEVICES

cmd=(torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" scripts/train.py
  task="$TASK"
  algo="$ALGO"
  task.num_envs="$NUM_ENVS"
  task.action.low_policy.run_path="$LOW_RUN_PATH"
  wandb.project="$HL_WANDB_PROJECT"
  wandb.id="$RUN_ID"
)

echo ">>> ${cmd[*]}"
"${cmd[@]}"
