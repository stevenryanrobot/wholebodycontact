# Changelog

All notable changes to this fork of GentleHumanoid are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — branch `wholebodycontact`

### 2026-07-18 — repository reorganization, data tidy, docs

Newest first; times are rough. The reorg changed **no behavior** — pure
relocation + import/path rewrites, validated by importing all 25 moved modules
and re-composing Hydra under the `gentle` env.

- **15:21** Split the flat `scripts/` pile into clear top-level modules:
  `controllers/` (low-level policies), `forcesense/` (the force-sensing module),
  `experiments/` (research + apps). Nested the GentleHumanoid framework at
  `controllers/ceer/active_adaptation/` (importable name kept via `setup.py`
  `package_dir`; run `pip install -e .`). Renamed the repo folder to
  `wholebodycontact`. _(b4a2bb4)_
- **15:26** Moved `start_gentle_local.sh` into `scripts/`; fixed references. _(4ffcd6d)_
- **15:30-16:00** Data + asset tidy: `data/wbc/` sorted into
  `datasets/ probe/ models/ logs/ smoke/`; `dataset/` -> `data/dataset/`; the
  low-level-policy wandb cache -> `controllers/ceer/checkpoints/wandb/`;
  `assets_sim2sim/` -> `forcesense/assets/`; `scripts/exports/` ->
  `controllers/ceer/checkpoints/`; `paper/` -> `docs/paper/`. Fixed
  reorg-induced runtime path bugs (`motion.py` dataset root, `motion_tracking.py`
  hand-samples, `isaac.py`/`play.py` `DEFAULT_OVERRIDES` + wandb cache).
- **16:06** Reorg phase 2 - co-located each area's config + scripts: CEER
  `cfg/ scripts/ bash/` -> `controllers/ceer/`; force-sensing `cfg/wbc` ->
  `forcesense/cfg/`, `bash/wbc_*` -> `forcesense/bash/`; top-level `scripts/`
  keeps only `start_gentle_local.sh`; `legacy/` -> `archive/` (gitignored).
  Hydra `config_path` + script `sys.path` are relative, so cfg+scripts moving
  together kept working. _(ae123dc)_
- **16:30** Rewrote `README.md` for this codebase; converted this changelog to a
  dated, time-stamped log; dropped Claude co-author trailers from commits.
- **~17:15** Pointed `origin` at `stevenryanrobot/wholebodycontact` and force-pushed
  the reorg onto `main` (rewrote history to drop Claude co-author trailers from all
  commits). Runtime smoke tests pass: force-sensor training and MuJoCo collection
  both run. Fixed the GitHub Pages deploy workflow (`web_demo` -> `experiments/web_demo`);
  renamed `forcesense/collect/mujoco.py` -> `mujoco_collect.py` (old name shadowed the
  `mujoco` package when run as a script -- caught by the smoke test); restored the
  web-demo ONNX models to version control so the deployed site has them.

Key module moves (old -> new): `scripts/wbc_train_v3.py` ->
`forcesense/{models.py, common/data.py, train/core.py}`; `wbc_sim2sim.py` ->
`forcesense/sim2sim.py`; `collect_force_data.py` -> `forcesense/collect/isaac.py`;
`wbc_collect_mujoco.py` -> `forcesense/collect/mujoco_collect.py`; `wbc_deploy_metrics.py`
-> `forcesense/common/metrics.py`; `wbc_eval_deploy.py` ->
`forcesense/eval/eval_deploy.py`; `wbc_export_v4.py` -> `forcesense/export.py`;
`wbc_crosspolicy*.py` -> `experiments/crosspolicy/*`; `wbc_probe_experiment.py`
-> `experiments/proact/probe_experiment.py`.

### Earlier — force-sensing research (June-July 2026, pre-reorg)

_These entries predate the reorg; file paths below use the old flat `scripts/`
layout (see the moves list above for where each lives now)._

### Added — cross-policy (plug-and-play) generalization test
Phase-0 existence check for a controller-agnostic force estimator: does a
proprioception→contact estimator trained under one downstream controller
transfer to a different one? "Different policy" is emulated cheaply by scaling
the deploy PD gains (base kp×1.0 / soft kp×0.6 / stiff kp×1.5) under the SAME
net_pull GT force protocol, so the controller is the only variable.
- `scripts/wbc_collect_mujoco.py`: added `--kp_scale` / `--kd_scale` (recorded
  in the h5 attrs) to collect controller-variant datasets.
- `scripts/wbc_crosspolicy.py` (new): dual-head (detection + 5-region) MLP,
  W=6 window; trains on one or more controllers (stacked along the worker-column
  axis) and evaluates in-domain vs on unseen controllers. Datasets in
  `data/wbc/cross/`, results `result_train*.json`.
- Finding: **the plug-and-play gap is real.** Train-on-base → base regAcc 0.78,
  but soft 0.52 / stiff 0.40 (−26 to −38 pts) and detection precision collapses
  0.97→0.65 (a different τ baseline reads as contact).
- **Controller-invariant residual channel (the fix):** `wbc_sim2sim.py` gained
  `ext_torque_residual()` — external joint-torque estimate from the equation of
  motion `M q̈ + qfrc_bias − qfrc_actuator − qfrc_passive − qfrc_constraint`
  (≈ Jᵀ F_ext), ~0 in contact-free stance regardless of gains, independent of
  how τ was produced. Stored as a separate `R` dataset by the collector; the
  cross-policy trainer takes `--feat {proprio,resid,both}`.
- **Result:** with the residual channel ALONE, a single-controller estimator
  transfers to unseen controllers with almost no loss — regAcc 0.919 in-domain
  vs 0.878 (soft) / 0.913 (stiff), precision 0.98 across all three — no
  randomization needed. `both`+randomization is the best overall (in-domain
  0.90–0.98, unseen 0.91). Controller randomization also helps raw proprioception
  (unseen stiff 0.40→0.70) but still trails the residual (0.93).
- `scripts/wbc_paper_figs.py` renders the two paper figures. **Paper draft** in
  `paper_plugandplay/` (IEEEtran, 4 pp, compiles to PDF) — SEPARATE from the
  ProACT paper. references.bib entries still marked VERIFY (from working notes).

### Added — plug-and-play optimization: robustness, legs, realizable residual (2026-07-14)
- 4 more controllers with independent/wider gains: `cross_{D_softud(0.4,0.7),
  E_vstiff(2.0,1.5),F_underd(1.0,0.5),G_overd(0.7,1.5)}.h5`, to stress the
  gain-scaling proxy with harder extrapolation.
- `scripts/wbc_crosspolicy_sweep.py` (new): leave-one-out robustness over all 7
  controllers. **residual is the robust winner** — worst/mean regAcc 0.905/0.931,
  precision 0.97/0.99; vs proprioception 0.417/0.653 (randomization helps
  interpolation, not extreme extrapolation) and `both` 0.777/0.897 (best in-domain
  but re-inherits the proprio shortcut on extremes). `sweep_result.json`.
- `scripts/wbc_crosspolicy_perregion.py` (new): per-region recall. **The residual
  rescues the LEGS: 0.56→0.89** (near arm level) by explicitly subtracting the
  ground reaction, removing double-support stance masking. Arms 0.80→0.98.
- `scripts/wbc_crosspolicy_dropout.py` (new): `both`+proprio-channel dropout does
  NOT beat resid on extreme controllers (worst 0.74–0.76) — the misleading OOD
  proprioception is still fed at test time. Resid-alone remains the recipe.
- **Realizability:** `wbc_sim2sim.py:ext_torque_residual(realizable=True)` drops
  the only sim-privileged term (`qfrc_constraint`, the GRF); collector
  `--resid_mode {full,realizable}`. Without GRF: arms 0.94 (fully realizable,
  proprioception-only) but legs 0.62 / trunk 0.42.
- **GRF-noise robustness:** `wbc_sim2sim.py:grf_torque()` + collector stores a
  `G` dataset; `scripts/wbc_crosspolicy_grfnoise.py` (new) sweeps a noisy GRF
  estimate `resid − (G+α·ε)`. Legs degrade gracefully with GRF error (0.92@0
  → 0.84@10% → 0.79@25% → 0.73@50%), arms immune (~0.97). Legs are realizable
  with a foot F/T sensor.
- **Paper rewritten** (`paper/plugandplay/`): 7-controller robustness + legs +
  realizability story; TikZ method/overview figure + 4 result figures
  (`scripts/wbc_plugandplay_figs.py`); 5 pp, compiles clean.
- **references.bib verified** against arXiv: SixthSense (2605.01427),
  GentleHumanoid (2511.04679), MOB-Net (2402.11221). Corrected a mis-citation —
  the "NEXT" free-space torque model is actually **FACTR 2** (2606.12406).

### Added — ProACT active-sensing existence proof (2026-07-10)
- `scripts/collect_force_data.py`: `--probe` / `--probe_mode
  {squat,shift,both,directed}` / `--probe_vy` / `--probe_lean_sign` inject a
  scripted probe (squat / lateral weight-shift / oracle contact-directed unload)
  into the frozen low-level policy during a sustained net_pull hold.
  `motion_tracking.py` gained `static_linvel_b` (commanded planar velocity in
  static mode) so the collector can weight-shift without touching the policy.
- `scripts/wbc_probe_experiment.py` (new) + `bash/wbc_probe_pipeline.sh`: passive
  (static) vs active (probe) per-region localization comparison.
- Findings: open-loop probes net ≈0 on legs/trunk (squat doesn't unload; lateral
  sway is asymmetric and just equalizes an incidental static asymmetry), but a
  **closed-loop probe directed to unload the contacted leg gives a net leg/trunk
  gain +0.073, dose-responsive** (left leg 0.39→0.50). Existence proof for active
  proprioceptive sensing; oracle upper bound (uses GT contact side). Skeleton
  paper in `paper/proact/`.

### Changed — paper folder reorganization (2026-07-14)
- `paper/` now holds two self-contained subfolders: `paper/proact/` and
  `paper/plugandplay/` (each `.tex` + `references.bib` + `figs/`/build). The old
  top-level `paper_plugandplay/` was moved under `paper/plugandplay/`.

### Added — web demo: walking + 24-link HUD + trimmed model picker
- Walking: the velocity-command low-level policy is now driven by a nonzero
  base-frame target velocity. WASD / arrow keys and an on-screen D-pad
  (`#walk-pad`, hold to walk, mouse + touch) move the robot; camera follows.
  Headless `test/walk_test.mjs` confirms forward/back/strafe/diagonal stay
  upright (pelvis z ~0.76–0.79). Drag-to-push + sensing work while walking.
- HUD localization bars are now rebuilt per model head: 24 per-link bars for
  the 24-link model (scrollable), 5 region bars for the 5-region models.
- Model picker trimmed to the three MuJoCo-calibrated models (v4c links W=10,
  v4c 5-region W=6, v4c restboost); uncalibrated v4 and v3 removed from the UI.

### Added — public web demo + GitHub Pages deploy
- `experiments/web_demo/` browser demo (MuJoCo WASM + ONNX Runtime Web + three.js) is now
  published at <https://stevenryanrobot.github.io/wholebodycontact/>.
- `.github/workflows/deploy.yml` builds `experiments/web_demo` with
  `--base=/wholebodycontact/` and deploys to GitHub Pages on every push to
  `main`. Root `README.md` links the live demo.
- `.gitignore`: exclude `mjlab_maze/{logs,data,results}` and web-demo
  `node_modules/`, `dist/` from the published repo.

### Added — v4 collection + retrain (in flight)
Data re-collection + retraining of the whole-body contact sensor fixing three
v3 limitations: 12-link granularity, no static-standing coverage (the web-demo
scenario), and short 0.8–2 s holds (magnitude overshot ~50% on sustained drags).

- `cfg/wbc/collect_v4.yaml` — 30 force-application link patterns (SixthSense-
  style: 4 trunk + 12 leg + 14 arm; `*_mimic` proxies skipped; 24 effective —
  waist + wrist pitch/yaw links are absent from the AMASS-retarget body list
  and are dropped by `_match_indices`), hold range widened
  `[40,100]` → `[40,400]` steps (up to 8 s). `wbc_input_`/`wbc_label_` groups
  identical to v3 (same 320-dim input; the deployed feature pipeline is
  unchanged).
- `scripts/collect_force_data.py` — new `--static_command` / `--static_height`
  flags: collect with a static standing root command via
  `cmd.set_static_root_command()` (matches the MuJoCo/web-demo deployment);
  replaces the motion-tracking `cum_error` termination with a fall-only check
  (tilt > ~60° or root < 0.45 m — a static robot ignores the reference motion),
  sets `disable_motion_finish`, and re-aims the static heading of any env that
  resets (init yaw is motion-sampled). Also stores a `static_command` h5 attr.
- `scripts/wbc_train_v3.py` — v4 upgrades, backward compatible: dynamic label
  onehot width `K` from the h5 attrs (legacy 12-link files still load);
  multi-file `--data` (motion + static h5 concatenated along time, windows
  never span a file boundary); `ContactGRUv4` (`arch="gru"`, GRU over the last
  W=50 raw frames, same det/loc/dir/mag heads, 1:4 pos:neg batch sampling +
  balanced-val metrics); optional impedance-aware torque normalization
  (`imp_norm`, SixthSense Eq. 10: `tau/(kp|q_des−q|+kd|qd|+eps)` with the G1
  constants from `wbc_sim2sim.py`, layout verified against the v3 dataset,
  flag + constants stored in the checkpoint for deployment).
- `scripts/wbc_sweep_v4.py` — 8-run sweep (MLP W∈{6,10} × {links,regions};
  GRU H=50 × {links,regions} × {raw, impedance-norm}); exports
  `force_sensor_v4_best.pt`, `force_sensor_v4_best_mlp.pt` (fallback) and
  `region_map.json` (30-link → 5-region aggregation for the web demo).
- `scripts/wbc_export_v4.py` — ONNX + meta-json export for both archs (MLP
  input `[1, W*320]` as v3; GRU input `[1, W, 320]`; impedance constants in
  the meta when applicable) with an onnxruntime parity check.
- `bash/wbc_v4_pipeline.sh` — resumable driver (motion 35 k×128 + static
  15 k×128 ≈ 6.4 M samples ≈ v3 scale, ~30% static → sweep → export), run in
  tmux session `wbc_v4`. Outputs `data/wbc/wbc_v4_{motion,static}.h5`,
  `data/wbc/sweep_v4/`.

### Added — v4 round 2: deployment metrics + GRU recipe fix (DONE)
Outcome: the recipe fixes cured the loc-degradation trend, but all three GRU
retrains topped out at deploy-F1 ~0.766 with region acc well below the MLPs —
champion by deployment metric: **`mlp_w6_regions`** (near-tie broken by region
accuracy). Verdict on this data: long windows buy ~+0.01 detection and lose
localization; MLPs win.
- `scripts/wbc_deploy_metrics.py` — deployment-semantics evaluation shared lib:
  honest labels (positive = |F| > 5 N dead-zone; sub-dead-zone contact and 1 s
  post-release ring-down excluded from negatives), detection-threshold sweep,
  k-frame persistence + hysteresis debouncing over per-env probability
  sequences, and a training-time `deploy_score` (debounced honest F1 on
  per-segment val slices) for model selection.
- `scripts/wbc_eval_deploy.py` — post-hoc re-eval of all sweep checkpoints
  with the above + motion/static split + region accuracy during
  confirmed-contact frames → `data/wbc/sweep_v4/deploy_leaderboard.jsonl`.
  Round-1 result: honest per-frame F1 0.76–0.79 (vs 0.60 at raw labels/0.5
  threshold — the flat 0.46 precision was label regime, not model); residual
  precision ceiling ~0.62–0.67 is locomotion-dynamics FPs (flat vs
  time-since-release), static regime better (MLP links: P≈0.67 R≈0.89,
  region acc 0.64–0.65). GRU round-1 loc degradation confirmed (region acc
  0.38–0.45).
- `scripts/wbc_train_v3.py` — round-2 options: natural-distribution sampling
  with fixed `steps_per_epoch`; `sep_loc_batch` (loc/dir/mag heads trained on
  their own uniformly-sampled contact batch so 1:4 det sampling cannot starve
  them); `deploy_select` (checkpoint selection by debounced honest F1).
- `scripts/wbc_sweep_v4_r2.py` + `bash/wbc_v4_r2.sh` (tmux session
  `wbc_v4_r2`) — three GRU-regions retrains (natural / sep-loc / sep+512),
  lr 3e-4, 60 epochs, then deploy re-eval and champion export via
  `scripts/wbc_r2_champion.py` (`force_sensor_v4_deploy_champion.onnx` +
  `champion.json` with the recommended threshold/debounce config; near-ties
  in debounced F1 broken by region accuracy).

### Added — v4 Track C: MuJoCo-domain data + champion fine-tune (DONE)
The web demo found v4 sensors read MuJoCo standing-at-rest as contact
(rest FP 0.97 at th 0.5) — an Isaac→MuJoCo proprioception domain gap.
- `scripts/wbc_collect_mujoco.py` — CPU-parallel MuJoCo collection reusing
  `wbc_sim2sim.Sim2Sim` headless: scripted net_pull-style episodes replicate
  the `collect_v4.yaml` profile (same 24-link onehot order read from the Isaac
  h5, same torso net-wrench limiter, 30% long-rest cycles for standing
  negatives, fall-reset), Isaac-schema h5 out (`data/wbc/wbc_v4_mujoco.h5`,
  1.2 M samples = 12 workers × 100 k steps, ~4 min at ~500 steps/s/proc).
  Fidelity verified: the new frames reproduce the browser pathology on the
  round-1 model (rest FP@0.5 0.999, push det 0.936) — the gap is in the data
  domain, and this data captures it.
- `scripts/wbc_finetune_v4c.py` + `bash/wbc_v4_c2.sh` (tmux `wbc_v4_r2:c2`) —
  fine-tunes the r2 deployment champion on Isaac motion+static + MuJoCo (3×
  oversampled), lr 1e-4, selection by debounced honest deploy-F1 on a held-out
  MuJoCo slice; reports before/after demo gates (rest plateau < 0.2, push det
  ≥ 0.85, trunk det) + Isaac val forgetting check; exports
  `data/wbc/sweep_v4/force_sensor_v4c.onnx` + meta with the recommended
  operating point.
- **Result (C2b, shipped)**: at the recommended debounced gate (th 0.6/0.45,
  k=3) MuJoCo rest FP **97% → 1.5%** (0.000 measured in the deployed demo
  pipeline), trunk-push det **0.997** (was eaten by the 0.85 stopgap gate),
  push det 0.878, deploy-F1 **0.925**; mild Isaac forgetting (det_f1
  0.595→0.575). `force_sensor_v4c_restboost.onnx` variant: rest FP ~0, push
  det −3 pts. Remaining known weakness: knee pushes detected (det 1.00) but
  mislocalized to trunk.

### Changed — web demo ships v4c_links as default (2026-07-08 morning)
- The same MuJoCo-domain fine-tune applied to the 24-link checkpoint
  (`force_sensor_v4c_links.onnx`) turned out to be the best of both worlds:
  link-level localization 0.259 → **0.669** (2.6×; standing pushes have much
  cleaner proprioceptive signatures than Isaac motion-tracking frames),
  region-aggregated 0.891, rest FP 4.4% debounced, trunk det 0.996, deploy-F1
  0.913. Now the demo default (picker: v4c_links / v4c 5-region / restboost /
  v4 / v3). Browser E2E: rest gate 0%, trunk drag ON 100% argmax=trunk, wrist
  drag resolves the exact link.

### Changed — web demo ships v4c (2026-07-08)
- `experiments/web_demo/` — v4c is the default sensor (5-region decode); model picker now
  offers v4c / v4c-restboost / v4 (24-link, keeps the two-level link+region
  highlight) / v3. Per-model operating points read from each meta's
  `recommended` field (fallback static map); det-bar threshold markers move on
  model switch. Headless Chrome E2E: rest 0% phantom contact, wrist drag
  det 0.99 with 18.9 N predicted vs 19.0 N applied, **torso drag gate ON 100%
  of hold with trunk highlighted** (impossible pre-v4c), picker round-trip
  clean.

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

### Added — v3 sensor, sweep & visualization (Plan A)
- `scripts/wbc_train_v3.py` — v3 trainer. Loads the full dataset once onto the
  GPU as a `[T, E, D]` grid (reused across a whole sweep, ~1s/epoch), feeds a
  temporal window of the last `W` frames, and uses **four heads**: detection
  (contact vs none), localization (which body/region, contact frames only),
  unit force-direction (cosine loss) and magnitude (relative-error). Optional
  `--regions` collapses the 12 links to 5 body regions.
- `scripts/wbc_sweep_v3.py` — grid-reuse sweep over window / region / cleaning
  options. Best model **`w6_regions`** (W=6, 5 regions) → `data/wbc/sweep_v3/`.
- `scripts/wbc_train.py`, `scripts/wbc_sweep.py` — the earlier per-link (v1)
  trainer + sweep these supersede.
- `scripts/viz_force_sensor.py` (v1) and `scripts/viz_force_sensor_v3.py` (v3) —
  offline plots: row-normalized confusion matrix, per-region detection vs
  localization, and a true-vs-predicted force-magnitude timeline. The v3 script
  evaluates on held-out val envs so numbers match the sweep leaderboard.
- `scripts/play_force_sensor.py` (v1) and `scripts/play_force_sensor_v3.py`
  (v3) — live Isaac Sim viewers. RED arrow = applied force, GREEN = MLP
  prediction (the whole predicted region lights up + a force arrow). v3 keeps a
  rolling `W`-frame window and gates direction/magnitude by the detection head.
  Supports both the region (5-class) and the finer link (12-class) v3 models.
  v3 extras: `--interactive` keyboard-driven force (keys 1-5 jump to a region,
  `[`/`]` cycle every candidate link, WASD/QE set push direction, `-`/`=` adjust
  magnitude, **R** resets), an
  in-viewport live RED/GREEN `|F|` curve drawn with `debug_draw`, and `--slow` /
  `--demo_force` for the random mode. In interactive mode auto-reset is disabled
  (no episode timeout / motion-end / fall reset) — only the R key resets.

Planned (Plan B, not yet implemented): feed the force estimate to the
high-level policy as an observation and train compliance with `root_ppo` while
the low-level stays frozen.

### Added — MuJoCo sim2sim prototype (Plan A, 2026-07-07)
- `scripts/wbc_sim2sim.py` — standalone plain-MuJoCo (no Isaac) sim2sim: runs the
  exported low-level ONNX policy (`scripts/exports/G1GENTLE-07-07_18-28`),
  rebuilds the 257-dim `policy` obs and 320-dim `wbc_input_` features, runs the
  v3 force sensor, applies known pushes via `xfrc_applied` and reports predicted
  vs ground-truth contact (region / direction / magnitude). Smoke test: stands
  10 s (rest FP 4%); arm pushes detected ≥0.9 with correct region and |F| within
  25%; trunk det 0.5, knee detected but mislocalized as trunk (static-standing
  posture is off the Isaac collection manifold). Validates Isaac→MuJoCo transfer
  ahead of the web demo. Key gotcha: `root_and_wrist_6d` must use the dataset's
  hand_mimic standing reference (constants baked in, extracted from
  `dataset/limmt_no_foot_gentle_amass_full`).
- `assets_sim2sim/g1/` — G1 MJCF + meshes copied from the gentle-humanoid deploy
  repo (untouched).

### Added — browser demo (Plan A, 2026-07-07)
- `experiments/web_demo/` — client-side interactive demo (inspired by the GentleHumanoid
  live demo): MuJoCo 3.10 WASM (official `@mujoco/mujoco` npm bindings) simulates
  the G1 at 200 Hz in the browser, three.js renders geometry mirrored from
  `mjModel` (visual == physical), ONNX Runtime Web runs both the low-level
  policy (`policy.onnx`, ~14 ms) and the v3 force sensor (~0.6 ms). Drag the
  robot to apply a force (red arrow); the sensor's prediction is drawn as a
  green arrow + per-region heatmap tint, with det/region probability bars and a
  10 s applied-vs-predicted |F| rolling curve. Run: `npm install && npm run dev`;
  headless checks: `npm run smoke`, `node test/browser_check.mjs`.
- `experiments/web_demo/src/{isaac,sim2sim}.js` — faithful JS port of the validated
  `scripts/wbc_sim2sim.py` pipeline (Isaac joint order + regex gain/scale maps,
  per-substep EMA(0.9), absolute joint-pos histories, 257-dim policy obs,
  320-dim sensor frame, link-origin force + 20 N·m torso torque limiter, wrist
  'still' reference). `USE_POLICY=true`: the real ONNX policy balances the robot
  in-browser at 200 Hz. Headless Chrome E2E: stands at z≈0.78; wrist drag 19 N →
  det 100% of hold, correct arm region, green arrow + heatmap render; torso det
  weak as in the Python prototype (known collection-manifold limitation). Drag
  force capped at 40 N (sensor training range).
- `data/wbc/sweep_v3/force_sensor_v3_best.onnx` (+ `.meta.json` with
  normalization stats) — v3 sensor exported for the web demo; onnxruntime parity
  vs PyTorch < 4e-6.

### Fixed
- `scripts/eval.py` — strip `sys.argv` after argparse so Isaac Kit's own CLI
  parser no longer chokes on our flags (`-p -e` previously crashed AppLauncher
  with "Failed to parse command line arguments").

### Added — docs
- `docs/maze_navigation_plan.md` — design doc for the north-star goal: a real G1
  navigating out of a maze **blind, using whole-body contact sensing only**
  (proprioceptive, no contact/force sensor, no vision). Feasibility assessment,
  3-layer architecture (locomotion / contact perception / navigation), and a
  staged roadmap with go/no-go gates. Plan A is the bottom perception layer and
  needs re-collection on *walking-into-walls* contact (not net_pull point forces).
- `docs/blind_maze_research_and_method.md` — **supersedes the technical part of
  the maze plan.** Deep research report (9 subagents, 2 waves: 5 surveys + 4
  reproduction-level paper reads) covering the full solution space for
  proprioception-only whole-body contact sensing (sim-trained learned estimators
  / momentum-observer + learned residual / free-space self-supervised residual
  à la NEXT / implicit-in-policy), contact-only maze navigation algorithms
  (Pledge, reactive state machines, RL à la Bresa, noise debouncing), G1
  platform facts (tau_est, ankle linkage), a comparison matrix, our hybrid
  method (GRU H=50, 6-sector contact output, momentum-residual input option,
  Pledge-wrapped reactive + RL navigation), novelty claims vs SixthSense
  (sustained wall contact, closed perception→navigation loop, real-time), and a
  phase-gated code roadmap (wall_collect / wbc_train_v4 / maze env / nav_*).

- `docs/framework_and_training_details.md` — full methods reference (in
  Chinese), verified line-by-line against current code: the two parallel
  stacks (mjlab blind-maze pipeline / Isaac hierarchical-compliance stack),
  all three control layers of each, and every model's exact
  observation/action/label dimensions, architectures, losses and training
  hyperparameters (walking policy, ContactGRU, low-level tracking policy,
  WBC force sensor v3, HL root_ppo, Pledge state machine).

### Added — mjlab blind-maze pipeline (`mjlab_maze/`, overnight 2026-07-05)
End-to-end sim demo of the north-star goal in **mjlab** (MuJoCo-Warp): G1 walks
mazes blind, sensing walls via whole-body contact. New isolated venv
`~/venvs/mjlab` (torch cu126). Components:
- `maze_gen.py` — perfect-maze generator (recursive backtracker) → wall boxes
  (1.6 m, above G1 shoulders so arms/torso engage) + MJCF emitter.
- `pledge.py` — contact-only navigation: 12 world-frame 30° azimuth bins
  (axis-aligned wall bearings land on bin centers), k-of-n debounce in the
  WORLD frame (walls don't rotate with the robot), left-hand wall following
  with yaw-servo wall acquisition (open-loop turn to remembered wall bearing),
  lost-wall bail-out, RAMBLER-style random kicks, stuck recovery. Pledge
  departure kept but disabled for perfect mazes (pure left-hand is complete).
- `test_pledge_2d.py` — kinematic validation: 96% clean / 98% noisy (det .85,
  FA .15) on 4x4; 87-90% on 6x6.
- `maze_env.py` — Mjlab-Velocity-Flat-Unitree-G1 + walls via `SceneCfg.spec_fn`
  + whole-body `ContactSensorCfg` vs wall geoms only + twist-command override.
- `features.py` / `collect_sensor_data.py` — 96-D proprioception (q/dq/tau/IMU,
  no commands) + net wall-contact labels; random-walk collection in mazes
  (real *sustained walking contact* — the distribution no prior work covers).
- `sensor_model.py` / `train_sensor.py` — ContactGRU (H=50) det/azimuth/mag
  heads + RollingEstimator (proprio → 12 world bins for the controller).
- `nav_run.py` / `record_video.py` — closed-loop maze eval (GT or estimated
  contact) + mp4 recording.
- G1 velocity walking policy trained from scratch in mjlab (rsl_rl).

**Overnight results (2026-07-05):** contact sensor det F1 **0.949**, azimuth
error **8.3°**, false-alarm 0.2%, cross-maze F1 0.943 — sustained
walking-contact is far more learnable than transient pushes (the literature
gap our research doc identified). **3×3 maze escaped blind on video twice:
with GT contact (409 s) and with the pure learned proprioceptive sensor
(90 s)** — the north-star loop (proprioception → whole-body contact estimate →
wall-following → maze exit) closed in simulation. Batch reliability (52
episodes, thick walls): 17–25%; sensor-loop ≥ GT-loop, so perception is NOT
the limiter — all falls traced to the contact-naive walking policy wrestling
walls (ACQUIRE/STUCK turns), fix = contact-curriculum retraining of the
low-level. Full analysis: `docs/overnight_experiment_report.md`.

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
