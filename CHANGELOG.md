# Changelog

All notable changes to this fork of GentleHumanoid are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — branch `wholebodycontact`

### Added — Whole-Body Contact force sensing (Plan A)
Proprioceptive external-force estimation: the robot learns to *sense* where on
its body an external force is applied, from proprioception only (no force/torque
sensor). The locomotion policy is **not** trained — only a supervised MLP.

- `cfg/wbc/collect.yaml` — data-collection overrides: whole-body random
  single-point external force (reuses the `net_pull` probe over 12 candidate
  links, full 3D directions) plus two raw observation groups, `wbc_input_`
  (proprioception) and `wbc_label_` (`net_pull_force_priv` ground truth). The
  `_` suffix keeps them out of VecNorm so values are stored raw.
- `scripts/collect_force_data.py` — drive a frozen low-level policy under random
  whole-body forces and log `(proprioception, force-label)` pairs to HDF5.
- `scripts/train_force_sensor.py` — supervised MLP with a body-link
  classification head (+ a "no-contact" class) and a 3D force-vector regression
  head; reports accuracy, confusion matrix, force cosine similarity and
  magnitude error.

Planned (Plan B, not yet implemented): feed the force estimate to the
high-level policy as an observation and train compliance with `root_ppo` while
the low-level stays frozen.

### Changed — repository tidy-up
- Moved launch/training shell scripts into `bash/` (`train.sh`, `train_hl*.sh`,
  `generate_*.sh`). Run them from the repo root, e.g. `bash bash/train.sh`.
  `start_gentle_local.sh` stays at the root as the (machine-local) env entry
  point you `source`.
- README: documented repository layout and the whole-body contact workflow;
  updated script paths.
- `.gitignore`: ignore `data/` and `*.h5` (collected datasets / trained sensors).

## Prior work (high-level control, on this fork)

- High-level teacher–student PPO and EE-compliance training
  (`bash/train_hl*.sh`, `cfg/task/G1/G1_hl_*`).
- Unified end-effector external-force handling in evaluation
  (`scripts/eval_manipulation.py`).
- Stiff low-level configs (`G1_gentle_3kp_stiff`) and motion-replay teleop.

See `git log` for the full history; this fork builds on upstream
[GentleHumanoid](https://gentle-humanoid.axell.top).
