# GentleHumanoid: Whole Body Motion Tracking with Compliance - Training

[![Home Page](https://img.shields.io/badge/Project-Website-C27185.svg)](https://gentle-humanoid.axell.top/#/) 
[![arXiv](https://img.shields.io/badge/Arxiv-2511.04679-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2511.04679) 
[![Video](https://img.shields.io/badge/Video-Demo-FF0000.svg?logo=youtube)](https://www.youtube.com/watch?v=rF6N2o0IQJg)
[![Online Demo](https://img.shields.io/badge/Online-Demo-3B82F6.svg?logo=demo)](https://gentle-humanoid.axell.top/#/demo)
[![Whole-Body Contact Demo](https://img.shields.io/badge/Web-Contact_Sensing_Demo-3ecf6e.svg)](https://stevenryanrobot.github.io/wholebodycontact/)

This repository contains the official implementation of GentleHumanoid. For additional details, please refer to the [Project](https://gentle-humanoid.axell.top) page.

> **🕹️ Whole-body contact-sensing web demo (this fork):
> <https://stevenryanrobot.github.io/wholebodycontact/>** — the Unitree G1
> infers *where* it is being touched and *how hard*, from proprioception alone.
> Left-drag on the robot to push it; a green arrow / heatmap shows the sensor's
> prediction. Fully client-side (MuJoCo WASM + ONNX Runtime Web + three.js).
> Source and build in [`experiments/web_demo/`](experiments/web_demo/); auto-deployed to GitHub Pages by
> [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml).

Main features:

*  A **universal** whole-body motion tracking policy,  training and evaluation pipeline.

* Enabling training **with or without compliance**.

* Dataset support including preprocessing for AMASS, Inter-X, and LAFAN.

A **demo** of the pretrained policies, shows different force control settings, and a single model generalizing across diverse motions, is available [here](https://gentle-humanoid.axell.top/#/demo).

Instructions for **real-robot deployment** and the use of **pretrained models** on new motion sequences are available [here](https://github.com/Axellwppr/gentle-humanoid/).

## Installation

1. Create a Conda environment.
```bash
conda create -n gentle python=3.10
conda activate gentle
```

2. Install Torch.
```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

3. Install Isaac Sim. Support Ubuntu 22.04 only. For 20.04, please check [here](https://isaac-sim.github.io/IsaacLab/v2.1.0/source/setup/installation/binaries_installation.html).
```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
# Test Isaacsim
isaacsim
```
4. Install Isaac Lab.
```
cd <where you want to install IsaacLab>
git clone git@github.com:isaac-sim/IsaacLab.git
cd IsaacLab
git checkout v2.2.0
./isaaclab.sh -i none
```

5. Install GentleHumanoid.
```
cd <where you want to install repo>
git clone https://github.com/Axellwppr/gentle-humanoid-training
cd gentle-humanoid-training
pip install -e .
```

## Motion Dataset Preparation

### Retargeting with GMR

We use GMR to retarget the [AMASS](https://amass.is.tue.mpg.de/), [Inter-X](https://github.com/liangxuy/Inter-X), and [LAFAN](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) datasets. The output format is a dataset containing a series of npz files with the following fields:

- `fps`: Frame rate
- `root_pos`: Root position
- `root_rot`: Root rotation in quaternion format (xyzw)
- `dof_pos`: Degrees of freedom positions
- `local_body_pos`: Local body positions
- `local_body_rot`: Local body rotations
- `body_names`: List of body names
- `joint_names`: List of joint names

You can use the [modified version of GMR](https://github.com/Axellwppr/GMR) to directly export npz files that meet the requirements.

Note: When processing the Inter-X dataset, you need to handle the coordinate axis definition differences from AMASS and the missing framerate. Please refer to [here](https://github.com/Axellwppr/GMR/blob/c8970d755519d9bb6e79786a3fb43649b71198fa/scripts/smplx_to_robot_dataset.py#L155) for details.

You should organize the processed datasets in the following structure:
```
<dataset_root>/
    AMASS/ACCAD/Female1General_c3d/A1_-_Stand_stageii.npz
    ...
    LAFAN/walk1_subject1.npz
    ...
    InterX/G001T000A000R000/P1.npz
```

### Dataset Building

Modify `DATASET_ROOT` in `controllers/ceer/bash/generate_dataset.sh` to point to your dataset root directory, then run the script (from the repo root) to generate the dataset:
```
bash controllers/ceer/bash/generate_dataset.sh
```

The dataset will be generated in the `dataset/` directory, and the code will automatically load these datasets. You can also use the `MEMATH` environment variable to specify the dataset root path.

## Training

You can use the provided `controllers/ceer/bash/train.sh` script to run the full training pipeline. Modify the global configuration section in `controllers/ceer/bash/train.sh` to set your WandB account and other parameters, then run (from the repo root):

```bash
bash controllers/ceer/bash/train.sh
```

By default, `controllers/ceer/bash/train.sh` trains the GentleHumanoid policy (with compliance). If you want to train a baseline tracking policy (without compliance), please uncomment the corresponding lines in `controllers/ceer/bash/train.sh`:

```bash
run_pipeline "G1/G1_gentle" "gt" "<date>" # by default, GentleHumanoid policy with compliance
run_pipeline "G1/G1_no_force" "noforce" "<date>" # baseline tracking policy without force perturbation
run_pipeline "G1/G1_extreme_force" "extremeforce" "<date>" # baseline tracking policy with force perturbation
```

Under standard settings, training takes approximately 5 hours on 4× A100 GPUs.
If GPU memory is constrained, it is recommended to appropriately tune the `NPROC` and `num_envs` parameters in `controllers/ceer/bash/train.sh` and `controllers/ceer/cfg/task/G1/G1.yaml`, respectively.
Such adjustments may increase training time and could affect training performance to some extent.

## Evaluation

```bash
python controllers/ceer/scripts/eval.py --run_path ${wandb_run_path} -p # p for play
python controllers/ceer/scripts/eval.py --run_path ${wandb_run_path} -p --export # export the policy to onnx (sim2real)
```

## Whole-Body Contact: proprioceptive force sensing

This fork adds the ability for the robot to *sense* where on its body an
external force is applied, **using proprioception only** (no force/torque
sensor). The locomotion policy is **not** trained — only a small supervised MLP
that maps proprioception → (which body link is pushed, 3D force vector). When
the PD controller fights an external push, the deviation in joint torques and
tracking errors carries the signal; the MLP learns to decode it.

Workflow (run inside the `gentle` env; `source scripts/start_gentle_local.sh` first):

```bash
# 1. Collect data: drive a frozen (preferably stiff) low-level policy under
#    random whole-body forces and log (proprioception, force-label) to HDF5.
python forcesense/collect/isaac.py -r ${low_level_wandb_run_path} \
    -n 30000 --num_envs 64 -o data/wbc/wbc_train.h5

# 2. Train the force-sensing MLP (no Isaac needed).
python scripts/train_force_sensor.py --data data/wbc/wbc_train.h5 \
    --out data/wbc/force_sensor.pt --epochs 40
```

- Force application and labels reuse the existing `net_pull` external-force mode
  (see `forcesense/cfg/collect.yaml`); `net_pull_force_priv` is the ground-truth label.
- Use a stiff, non-compliant low-level policy for collection — a compliant
  controller gives way and washes out the proprioceptive signal.

## Repository layout

```
controllers/         low-level policies (swappable base controllers)
  ceer/              CEER / GentleHumanoid framework: active_adaptation/ (code), cfg/ (Hydra
                     configs), scripts/ (train.py, eval.py, utils/, data_process/), bash/
                     (train.sh, train_hl*.sh, generate_*.sh), checkpoints/
forcesense/          force-sensing module: models, common/, train/, collect/, eval/, sim2sim,
                     viz/, cfg/ (collect configs), bash/ (wbc_* pipelines), assets/
experiments/         research + apps: crosspolicy/, proact/, maze/, web_demo/
archive/              superseded v1 code
data/                datasets + checkpoints, gitignored (data/wbc/, data/dataset/)
docs/                docs + paper/
scripts/start_gentle_local.sh   machine-local env entry point — `source` it before running
```
