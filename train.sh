#!/usr/bin/env bash
set -euo pipefail

# ===== Global Configuration =====
PROJECT="luoxinyuan-duke-university/gentle_humanoid"
export CUDA_VISIBLE_DEVICES=0
MASTER_PORT=29500
NPROC=1
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
run_pipeline "G1/G1_gentle" "gentle" "test1939"
# run_pipeline "G1/G1_gentle" "gentle" "safe_hand"

# run_pipeline "G1/G1_no_force" "noforce" "motion_tracking_RL"
# run_pipeline "G1/G1_extreme_force" "extremeforce" "1215"



# #!/usr/bin/env bash
# set -euo pipefail

# export ISAACLAB_PATH=/home/xl521/IsaacLab
# export PYTHONPATH="$PYTHONPATH:$ISAACLAB_PATH/source/isaaclab"
# export LD_LIBRARY_PATH="/home/xl521/.local/lib:$CONDA_PREFIX/lib:/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# ISAACLAB_PY="$ISAACLAB_PATH/isaaclab.sh"

# # ===== Global Configuration =====
# export MEMPATH=/home/xl521/ee-gentle-humanoid/
# NPROC=1
# PROJECT="gentle_humanoid"
# SCRIPT="scripts/train.py"

# run_pipeline() {
#   local TASK="$1"
#   local TAG="$2"
#   local SUFFIX="$3"

#   local ID_TRAIN="${TAG}_train_${SUFFIX}"
#   local ID_ADAPT="${TAG}_adapt_${SUFFIX}"
#   local ID_FINETUNE="${TAG}_finetune_${SUFFIX}"

#   # ---------- TRAIN ----------
#   cmd=("$ISAACLAB_PY" -p "$SCRIPT" \
#     task="$TASK" +exp=train \
#     wandb.project="$PROJECT" \
#     wandb.id="$ID_TRAIN" \
#     headless=True)
#   echo ">>> ${cmd[@]}"
#   "${cmd[@]}"

#   # ---------- ADAPT ----------
#   cmd=("$ISAACLAB_PY" -p "$SCRIPT" \
#     task="$TASK" +exp=adapt \
#     checkpoint_path="run:${PROJECT}/${ID_TRAIN}" \
#     wandb.project="$PROJECT" \
#     wandb.id="$ID_ADAPT" \
#     headless=True)
#   echo ">>> ${cmd[@]}"
#   "${cmd[@]}"

#   # ---------- FINETUNE ----------
#   cmd=("$ISAACLAB_PY" -p "$SCRIPT" \
#     task="$TASK" +exp=finetune \
#     checkpoint_path="run:${PROJECT}/${ID_ADAPT}" \
#     wandb.project="$PROJECT" \
#     wandb.id="$ID_FINETUNE" \
#     headless=True)
#   echo ">>> ${cmd[@]}"
#   "${cmd[@]}"
# }


# run_pipeline "G1/G1_gentle" "gentle" "test"
# # run_pipeline "G1/G1_gentle" "gentle" "twist"

# # run_pipeline "G1/G1_gentle" "gentle" "lafan"
# # run_pipeline "G1/G1_gentle" "gentle" "interx"
# # run_pipeline "G1/G1_no_force" "noforce" "1215"
# # run_pipeline "G1/G1_extreme_force" "extremeforce" "1215"
