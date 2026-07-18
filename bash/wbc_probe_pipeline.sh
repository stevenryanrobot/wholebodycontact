#!/usr/bin/env bash
# ProACT existence proof: collect STATIC (passive) + WIGGLE (active squat)
# matched datasets, then train+compare leg/trunk localization.
set -e
cd ~/Documents/wholebodycontact
source start_gentle_local.sh >/dev/null 2>&1
RUN=luoxinyuan-duke-university/gentle_humanoid/gentle_finetune_3point_amass_limmt_full_stiff30
ENVS=64
STEPS=8000

echo "===== [1/3] collect STATIC (passive baseline) ====="
python -u forcesense/collect/isaac.py -r "$RUN" \
    --overrides cfg/wbc/collect_v4.yaml --static_command \
    --num_envs $ENVS --num_steps $STEPS --warmup 200 \
    -o data/wbc/probe/probe_static.h5

echo "===== [2/3] collect WIGGLE (active probe: squat) ====="
python -u forcesense/collect/isaac.py -r "$RUN" \
    --overrides cfg/wbc/collect_v4.yaml --static_command --probe \
    --num_envs $ENVS --num_steps $STEPS --warmup 200 \
    -o data/wbc/probe/probe_wiggle.h5

echo "===== [3/3] train decoders + compare ====="
python -u experiments/proact/probe_experiment.py \
    --static data/wbc/probe/probe_static.h5 \
    --wiggle data/wbc/probe/probe_wiggle.h5 \
    --n_envs $ENVS --epochs 40 --out data/wbc/probe/probe_result.json
echo "===== ProACT pipeline done ====="
