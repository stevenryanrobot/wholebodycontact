#!/usr/bin/env bash
set -euo pipefail

# ===== Global Configuration =====
PROJECT="luoxinyuan-duke-university/gentle_humanoid"
export CUDA_VISIBLE_DEVICES=0,1,2,3
MASTER_PORT=29500
NPROC=4
SCRIPT="scripts/train.py"

run_pipeline() {
  local TASK="$1" TAG="$2" SUFFIX="$3"

  local ID_TRAIN="${TAG}_train_${SUFFIX}"
  local ID_ADAPT="${TAG}_adapt_${SUFFIX}"
  local ID_FINETUNE="${TAG}_finetune_${SUFFIX}"

  # ---------- TRAIN ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port=${MASTER_PORT} "$SCRIPT"
    task="$TASK" +exp=train
    wandb.id="$ID_TRAIN"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- ADAPT ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port=${MASTER_PORT} "$SCRIPT"
    task="$TASK" +exp=adapt
    checkpoint_path="run:${PROJECT}/${ID_TRAIN}"
    wandb.id="$ID_ADAPT"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"

  # ---------- FINETUNE ----------
  cmd=(torchrun --nproc_per_node="$NPROC" --master_port=${MASTER_PORT} "$SCRIPT"
    task="$TASK" +exp=finetune
    checkpoint_path="run:${PROJECT}/${ID_ADAPT}"
    wandb.id="$ID_FINETUNE"
  )
  echo ">>> ${cmd[*]}"; "${cmd[@]}"
}

# run_pipeline "G1/G1_gentle" "gentle" "ablation"
run_pipeline "G1/G1_gentle" "gentle" "5point"
# run_pipeline "G1/G1_gentle" "gentle" "safe_hand"

# run_pipeline "G1/G1_no_force" "noforce" "motion_tracking_RL"
# run_pipeline "G1/G1_extreme_force" "extremeforce" "1215"
