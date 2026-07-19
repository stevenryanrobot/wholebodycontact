# G1 Contact-Force-Sensing Web Demo

Browser-based demo of proprioceptive whole-body contact force sensing on the
Unitree G1, in the spirit of <https://gentle-humanoid.axell.top/#/demo>:
MuJoCo compiled to WASM + ONNX Runtime Web + three.js, fully client-side.

The robot is balanced by the exported low-level ONNX policy. Two force-sensing
methods are selectable in the HUD: the **proprioceptive** sensors (v3/v4c) that
read the raw 320-dim obs frame, and the **residual** sensor that reads a 29-dim
controller-invariant external-torque channel computed from the equation of
motion (see below). Left-drag on the robot to pull it
(up to 40 N, red arrow); the sensor's prediction is shown as a green arrow
(direction * magnitude at the predicted-region centroid), a per-region heatmap
on the robot, HUD probability bars, and a rolling applied-vs-predicted force
curve.

The control/observation pipeline is a faithful JS port of the **validated
Python prototype `forcesense/sim2sim.py`** (read that file for the provenance
of every Isaac-semantics decision — port, don't re-derive):

- 257-dim policy obs: `boot(1) | command(6) | root_and_wrist_6d(12) |
  root_ang_vel(3) | projected_gravity(3) | joint_pos_history[0..4](145) |
  prev_actions*3(87)`; joint-pos obs are ABSOLUTE; histories newest-first;
  obs updated at 50 Hz control ticks with 2-substep joint-pos averaging.
- `root_and_wrist_6d`: baked-in "still" standing wrist reference
  (`WRIST_REF_STILL` in `src/isaac.js`, copied verbatim).
- command: root height 0.80, zero linvel target, heading [1,0] in the yaw
  frame, force_limit 30.
- actions: clip ±10, EMA alpha 0.9 applied PER 200 Hz substep, per-joint
  regex scale map, `q_des = default_pose + scale*filtered`, PD with base
  G1 gains, torque clipped to effort limits (and 0.8*ctrlrange).
- passive joint damping/frictionloss zeroed (Isaac's implicit PD has none).
- sensor: 320-dim frame `applied_torque | joint_pos_history[0..4] |
  q_des | gyro hist 5 | gravity hist 5 | prev_actions*3`, per-frame
  (x-mean)/std normalization, W-frame window flattened OLDEST-FIRST (window,
  base_dim, head width all read from the model's meta json — models are
  drop-in), magnitude = mag_head * force_max(40). The dir head is in the
  BASE frame; it is rotated to world with the root quat for the arrow.

### Sensor models (HUD picker)

Detection is debounced by `ContactGate` (hysteresis + 3-frame persistence);
the operating point is read from each model meta's `recommended` field when
present (v4c models), falling back to the static `OP_POINTS` map in
`src/main.js`. The green arrow, heatmap, and the curve's predicted |F| are
gated by the debounced state; the HUD det bar always shows the raw
probability with marker lines at the active thresholds.

- **v4c (default)** `models/force_sensor_v4c.onnx`: 5-region head, W=6,
  fine-tuned overnight on MuJoCo-domain data (Track C). Fixes the v4 rest-FP
  pathology: at its recommended gate (th_hi=0.6 / th_lo=0.45 / k=3) measured
  rest FP raw 0.000 and debounced rest ON 0.000 in `npm run smoke`, with
  trunk-push det 1.00 (v3/v4 could not detect trunk pushes reliably).
- **v4c restboost** `models/force_sensor_v4c_restboost.onnx`: quietest-rest
  variant (recommended gate 0.4/0.25), ~3 pts lower push detection.
- **v4** `models/force_sensor_v4.onnx` (`model_mlp_w10_links` export): W=10,
  24-link `loc_logits` head. Link probs are aggregated to the 5 regions via
  `models/region_map.json`, and the argmax link's meshes get a stronger tint
  than its region (two-level heatmap — only this model exercises that path).
  HISTORICAL ISSUE (2026-07-08, kept for reference): uncalibrated v4
  checkpoints read the sim2sim standing regime as ~70% contact (rest FP
  0.93-0.97; identical in the Python prototype, so not a JS bug) — hence its
  fallback gate sits above the rest plateau at 0.85/0.75. Superseded by v4c.
- **v3** `models/force_sensor_v3.onnx`: W=6, 5-region head, rest FP 0.042.
- **residual** `models/force_sensor_resid.onnx`: W=6, 5-region head, but the
  input is NOT proprioception — it is the 29-dim controller-invariant residual
  channel (`meta.input_source == "resid"`, base_dim 29 vs 320). The demo
  detects this flag and feeds `Sim2Sim.residChannel()` instead of `wbcObs()`;
  everything downstream (window, per-feature norm from meta, det/loc/dir/mag
  decode, heatmap, green arrow, curve) is unchanged. Trained on
  `data/wbc/cross/cross_A_base.h5` (base-gain controller, matching the demo's
  policy) via `forcesense.train.core --feat resid`. `node test/resid_check.mjs`
  (real pipeline): rest FP raw 0.000; det 1.00 for torso/arm/leg 25 N pushes;
  arm dir cos ~0.99. The visible differentiator vs proprioception: a left-knee
  push localizes to **left_leg** (region acc 1.00), where v4c mislocalizes it
  to trunk — the residual explicitly cancels the ground-reaction term, so
  stance masking on the legs is removed. (Under the single demo controller the
  no-false-positive and controller-invariance properties are guaranteed by
  construction; the full plug-and-play win needs a controller change, which
  this demo does not expose.)

  ### Residual channel (how it's computed in JS)

  Training defines the residual (`forcesense/sim2sim.py:ext_torque_residual`,
  see `docs/residual_method_explained.md`) as
  `M(q) q̈ + qfrc_bias − qfrc_actuator − qfrc_passive − qfrc_constraint`,
  the external generalized force `J_cᵀ F_ext` on the 29 actuated dofs. MuJoCo's
  equation of motion gives `M q̈ = qfrc_smooth + qfrc_constraint`, so the
  constraint (GRF) term cancels and the demo computes the identical quantity as
  `qfrc_smooth + qfrc_bias − qfrc_actuator − qfrc_passive` — no `mj_fullM` /
  `mj_mulM` and no `qfrc_constraint` needed, all live TypedArray heap views.
  Verified against the Python prototype to < 1.2e-7 abs over 300 control ticks.
  Read post-step (last substep's forward dynamics), same cadence as the
  collected R channel, indexed at the ISAAC-order joint dof addresses (`vadr`).
- drag force: applied at the LINK FRAME ORIGIN (CoM-offset torque correction)
  with the 20 N·m net-torque-about-torso limiter, capped at 40 N (sensor
  training range 10-40 N).
- policy ONNX: raw obs in (VecNorm folded into the graph); the `action`
  output selected BY POSITION via `models/policy.json` `out_keys` (graph
  output names collide: `loc`/`action` are both `linear_9`, numerically equal).

## Run

```sh
cd web_demo
npm install
npm run dev          # dev server (prints the local URL)
npm run build        # production build into dist/
npm run preview      # serve dist/
npm run smoke        # headless Node test of the REAL pipeline:
                     #   10 s policy standing + 4 instrumented 25 N pushes
node test/resid_check.mjs   # same pipeline, the RESIDUAL sensor:
                     #   rest FP + torso/arm/LEG pushes off residChannel()
node test/browser_check.mjs http://localhost:5173   # drives system Chrome:
                     # boot, policy standing, torso + arm drags with live
                     # sensor readouts, Reset recovery, screenshots
```

Node >= 18 required (developed on 18.19 / npm 9).

## MuJoCo WASM choice

Uses the **official `@mujoco/mujoco` npm package (MuJoCo 3.10.0)**, the
canonical JS/TS bindings developed by Google DeepMind (moved to npm in late
2025). Chosen over the older community port `zalo/mujoco_wasm` (MuJoCo 2.3.x)
because it is:

- an npm package with prebuilt single-threaded `.wasm` (no COOP/COEP headers
  needed) plus TypeScript declarations,
- MuJoCo >= 3.0 (tracks upstream releases),
- has a documented virtual-FS API: `new mujoco.MjVFS()` + `vfs.addBuffer(name,
  bytes)` + `MjModel.from_xml_string(xml, vfs)` for XML + mesh assets,
- exposes live TypedArray views into the WASM heap (`qpos`, `qvel`, `ctrl`,
  `xfrc_applied`, `geom_xpos/xmat`, `mesh_vert/face`, ...), which is exactly
  what the renderer and control loop need.

Rendering mirrors geometry straight out of `mjModel` (`mesh_vert`/`mesh_face`
from the WASM heap, per-geom `geom_xpos`/`geom_xmat` each frame) — no separate
STL loading, so visual == physical. Only group-1 (visual) geoms are mirrored.

## Verified (all green)

Headless Node smoke (`npm run smoke`, exact same modules the browser runs;
v4c sensor at its recommended gate, v4 + v3 alongside for the rest comparison):

```
stand test PASS; final z=0.780; rest FP raw: v4c=0.000 v4=0.970 v3=0.042
v4c debounced gate(0.6/0.45) rest ON=0.000; ~16x realtime
push torso_link        (trunk)     25N: det=1.00 argmax=trunk     |F|=28.5N cos=0.95  <- recovered
push left_wrist_roll   (left_arm)  25N: det=1.00 argmax=left_arm  |F|=28.0N cos=0.97
push right_elbow       (right_arm) 25N: det=1.00 argmax=right_arm |F|=26.1N cos=0.95
push left_knee         (left_leg)  25N: det=1.00 argmax=trunk (mislocalized) |F|=25.7N cos=0.95
```

Headless Chrome end-to-end (dev AND production preview builds):

- policy standing: pelvis z stable at 0.777-0.81; **6 s rest: debounced gate
  ON 0.0%** (mean raw det 0.36 — no phantom contact).
- torso drag ~26 N / 2 s (the v4c headline): **gate ON 100% of the hold**,
  argmax region trunk, mean det 0.92-0.96, predicted ~29-31 N.
- left-wrist drag ~19 N / 2 s: det 0.99 for 100% of the hold, argmax
  left_arm, gate ON 100%, predicted ~18.9 N (vs 19.0 N applied).
- model picker round-trip v4c -> v4 -> v3 -> v4c, each with its own gate and
  threshold markers (v4 rest det 0.70 but gate off; v3 0.39; v4c 0.37).
- Reset restores the stand (z=0.78); zero page errors.

Timing: 50 Hz control ticks (policy inference ~1-2 ms warm, sensor ~0.5 ms)
each running 4 MuJoCo substeps at dt=0.005; the sim loop uses accumulator
catch-up (max 4 ticks/wakeup) so sim time tracks the wall clock under load.

## Flags (src/main.js)

- `USE_POLICY = true` — the real pipeline. Set false to fall back to the
  legacy scaffold statue (stiff PD + pelvis base-assist spring, also the
  automatic fallback if ONNX fails to load).
- `USE_FAKE_SENSOR = false` — set true to drive the viz from the ground-truth
  applied force instead of the model (stub kept from the scaffold pass).

## Layout

```
index.html            HUD/canvas layout + styles
vite.config.js        optimizeDeps excludes for the two wasm packages
src/main.js           boot, 50 Hz async sim loop, render loop, flags, __demo test hook
src/isaac.js          Isaac constants (joint order, gains, wrist ref) + quat math
src/sim2sim.js        JS port of wbc_sim2sim.py Sim2Sim (obs/action semantics)
src/mujocoLoader.js   wasm init, MjVFS asset loading, model introspection
src/control.js        legacy scaffold statue (computePD, base assist) — fallback only
src/robotRenderer.js  mjModel -> three.js mirror, two-level region/link tinting
src/drag.js           raycast grab, camera-plane force (cap 40 N), red arrow
src/sensorViz.js      decode (link->region aggregation, base->world dir),
                      ContactGate (hysteresis+persistence), green arrow, curve
src/onnx.js           ort sessions, action-by-position, meta-driven SensorRunner,
                      loadSensor/switchSensor for the HUD model picker
src/hud.js            DOM panel + model picker
test/node_smoke.mjs   headless REAL-pipeline test (port of the Python harness)
test/resid_check.mjs  headless REAL-pipeline test of the residual sensor
test/browser_check.mjs headless-Chrome end-to-end check
public/assets/g1/     G1 MJCF + 35 referenced STL meshes
public/models/        policy.onnx + policy.json,
                      force_sensor_v4c.onnx + meta.json (default, 5 regions, W=6,
                        MuJoCo-domain-calibrated; meta carries `recommended` gate),
                      force_sensor_v4c_restboost.onnx + meta.json (quietest rest),
                      force_sensor_v4.onnx + meta.json (24 links, W=10),
                      force_sensor_v3.onnx + meta.json (5 regions, W=6),
                      force_sensor_resid.onnx + meta.json (residual channel,
                        5 regions, W=6, base_dim 29, input_source "resid"),
                      region_map.json (24 links -> 5 regions)
```

## Gotchas discovered (do not regress)

- `onnxruntime-web` must run with `ort.env.wasm.numThreads = 1`: with threads
  it spawns a Worker on its own module URL, which after `vite build` is the
  app bundle itself — the app re-executes in the worker and session creation
  hangs.
- Don't set `ort.env.wasm.wasmPaths` to a `public/` folder: vite dev returns
  500 for JS imports of public assets. Let ort resolve via `import.meta.url`.
- `optimizeDeps.exclude: ['@mujoco/mujoco', 'onnxruntime-web']` is required so
  their `import.meta.url` wasm resolution survives the dev server.
- The policy ONNX graph has duplicate output names; select outputs by position
  from `policy.json` `out_keys`, never by `outputNames[0]`.
- Keep the MJCF passive damping/frictionloss zeroing — with the MJCF defaults
  (damping 2, frictionloss 0.2) the arms are over-damped and off the sensor's
  training manifold.
