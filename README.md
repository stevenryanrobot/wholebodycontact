# Whole-Body Contact — proprioceptive force sensing for humanoids

Estimate **where** a humanoid is being touched/pushed and **how hard**, across the
whole body, on a Unitree G1 — **from proprioception only** (no force/torque
sensors). The estimator reads a **controller-invariant residual channel**, so the
same sensor plugs into any low-level policy **without retraining** (plug-and-play).

- **No extra hardware.** Uses joint states + applied torques the robot already has.
- **Controller-agnostic.** A residual from the equation of motion cancels the
  controller's own torques, so soft/stiff/any policy reads the same at contact.
- **Whole body incl. legs.** Subtracting the ground-reaction term un-masks the
  legs during stance.

The low-level walking/tracking policy is a fork of
[GentleHumanoid](https://arxiv.org/abs/2511.04679) (here called **CEER**); the
force-sensing module is our contribution on top.

---

## Repository layout

```
controllers/         low-level policies (the swappable base controllers)
  ceer/              CEER / GentleHumanoid framework:
    active_adaptation/  policy code (envs, MDP terms, PPO) — importable as `active_adaptation`
    cfg/                Hydra configs (algo/ task/ exp/ objects/ + train/eval yaml)
    scripts/            train.py, eval.py, utils/, data_process/
    bash/               train.sh, train_hl*.sh, generate_*.sh
    checkpoints/        exported ONNX + wandb policy-run cache (gitignored)
  README.md          how to add a new low-level policy (e.g. Sonic) as a plugin
forcesense/          the force-sensing module (our contribution)
  models.py          ForceSensorV3 (MLP), ContactGRUv4
  common/            shared lib: regions.py, data.py, metrics.py
  train/             core.py (trainer), sweep_v3/v4/v4_r2.py, finetune.py, champion.py
  collect/           isaac.py (Isaac Lab collection), mujoco.py (fast MuJoCo collection)
  sim2sim.py         MuJoCo deploy: runs the policy + sensor, computes the residual
  eval/  export.py  viz/  figs.py
  cfg/               collect.yaml, collect_v4.yaml (force-collection observation layout)
  bash/              wbc_* pipelines (collect -> train -> eval)
  assets/            G1 MuJoCo model (xml + meshes)
experiments/         research + apps
  crosspolicy/       plug-and-play / cross-controller transfer study
  proact/            active sensing (ProACT probes)
  maze/              blind-navigation maze app
  web_demo/          browser demo (MuJoCo-in-WASM)
data/                datasets + checkpoints (gitignored): data/wbc/, data/dataset/
docs/                method notes, reports, and paper/ (LaTeX)
archive/             superseded v1 code (gitignored, kept for reference)
scripts/             start_gentle_local.sh (machine-local env entry point)
```

## Environment

Isaac Sim work (policy training, Isaac force-data collection) runs in the
`gentle` conda env. **`source` the env script first** — it activates conda, puts
Isaac Lab on `PYTHONPATH`, and enters the repo:

```bash
source scripts/start_gentle_local.sh      # note: source, not ./
pip install -e .                          # once — registers the `active_adaptation` package
```

MuJoCo-only work (`forcesense/sim2sim.py`, `forcesense/collect/mujoco.py`, the
web demo) does **not** need Isaac Sim.

## Quickstart

**1. Train / obtain a low-level policy (CEER).** From the repo root:
```bash
bash controllers/ceer/bash/train.sh                 # low-level policy
# high-level / compliance variants: controllers/ceer/bash/train_hl*.sh
```
Evaluate or export a trained run:
```bash
python controllers/ceer/scripts/eval.py --run_path <wandb_run> -p           # play
python controllers/ceer/scripts/eval.py --run_path <wandb_run> --export     # -> onnx
```

**2. Collect force-contact data.**
```bash
# Isaac Lab (drives the real policy + scripted external pushes):
python forcesense/collect/isaac.py -r <wandb_run> ...
# Fast MuJoCo collection (~22x realtime, parallel workers):
python forcesense/collect/mujoco.py --workers 12 --num_steps 60000 ...
```
Datasets land in `data/wbc/datasets/`.

**3. Train the force sensor.**
```bash
python forcesense/train/core.py --data data/wbc/datasets/wbc_v4_motion.h5 \
    data/wbc/datasets/wbc_v4_static.h5 --window 6 --regions
# grid sweeps: forcesense/train/sweep_v4.py ; finetune: forcesense/train/finetune.py
```

**4. Deploy in MuJoCo (sim2sim).**
```bash
python forcesense/sim2sim.py            # runs policy + sensor, prints contact estimates
```

**5. Experiments.** Plug-and-play study `experiments/crosspolicy/`, active
sensing `experiments/proact/`, maze app `experiments/maze/`, browser demo
`experiments/web_demo/`. Pipelines that chain collect -> train -> eval live in
`forcesense/bash/wbc_*.sh`.

## Adding another low-level policy (Sonic, BeyondMimic, …)

The residual channel is controller-invariant, so a new controller needs **no
sensor retraining** — just make it drivable in sim. See
[`controllers/README.md`](controllers/README.md): drop a `controllers/<name>/`
folder with the policy's weights + a small `adapter.py` (obs/action convention).

## The method, in one line

`tau_ext = M*qdd + c - tau_act - tau_passive - tau_con` — the external joint
torque from the equation of motion. It is ~0 in contact-free stance **for any
controller** (the controller's `tau_act` cancels), and equals `Jᵀ F_ext` at
contact, so it transfers across controllers and un-masks the legs. Full
derivation: [`docs/residual_method_explained.md`](docs/residual_method_explained.md);
papers in [`docs/paper/`](docs/paper/).

## Notes

- `scripts/start_gentle_local.sh` is machine-local (absolute paths) and
  **gitignored** — each machine keeps its own.
- [`CHANGELOG.md`](CHANGELOG.md) is a dated, time-stamped work log.
- On real hardware the only sim-privileged term is `tau_con` (ground reaction);
  the G1 has no foot F/T sensors, so legs need a floating-base momentum observer
  (MOB). Arms/torso are fully proprioceptive.
