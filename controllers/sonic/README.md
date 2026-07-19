# controllers/sonic — Sonic (GR00T-WholeBodyControl) as a plug-in policy

External low-level policy from **NVlabs/GR00T-WholeBodyControl** (HF
`nvidia/GEAR-SONIC`, public). Integrated as a `controllers/` plugin so the
force-sensing residual can be tested for plug-and-play across a genuinely
different controller (not just kp-scaled variants of CEER).

**What it is.** An encoder→decoder **motion-tracking** policy: the decoder maps a
994-dim observation (incl. a 64-dim token) to 29 joint-**position** deltas;
torque is produced by a PD law (`τ = Kp(q_target−q) − Kd·q̇`) that we run in the
sim. It needs a motion reference (the encoder token), synthesized here by
`planner_sonic.onnx` (idle-stand). 50 Hz control.

## Files
- `sonic_policy.py` — the adapter: 994-dim decoder obs assembly, name-verified
  IsaacLab↔MuJoCo joint remap, Sonic PD gains / default pose / action scale
  (baked from the deploy `policy_parameters.hpp`), `SonicPolicy` (loads the
  encoder/decoder ONNX).
- `sonic_planner.py` — wraps `planner_sonic.onnx` to build a standing reference
  → encoder token.
- `checkpoints/` — `model_encoder.onnx`, `model_decoder.onnx`,
  `planner_sonic.onnx`, config yaml (gitignored; from HF `nvidia/GEAR-SONIC`).
- Sim-side driving glue lives in `forcesense/sim2sim.py`
  (`Sim2Sim.setup_sonic` / `reset_sonic` / `control_step_sonic`) — drives OUR
  `g1.xml` with Sonic's PD while emitting the 320-dim proprioception + residual.
- M1 test harness: `experiments/sonic_m1_drive.py`.

## Status — WIP (closed-loop driving not yet stable)
Verified piecewise: remap is a name-matched bijection, default pose stands under
Sonic's PD, decoder ≈0 action at the default+upright state, planner emits a valid
idle trajectory. **Blocker:** closed-loop driving winds up (action 0.8→29) and
collapses at ~0.7 s. Localized to a **heading-canonicalization frame convention**
in Sonic's C++ deploy stack (`UpdateHeadingState` / `ComputeApplyDeltaHeading`)
not yet reproduced in the obs assembly — every other convention is verified.

**Next step:** canonicalize the reference + decoder obs against the robot's live
heading (remove first-frame heading + horizontal position) exactly as the deploy
code does; reference: `~/wbc_external/sonic_groot/gear_sonic_deploy/` (upstream
clone, not vendored).

## How to add other external policies
Same pattern: `controllers/<name>/` with the policy's weights (`checkpoints/`,
gitignored), an adapter mapping our sim state ⇄ its obs/action convention, and a
README. The residual sensor is controller-invariant, so no sensor retraining.
