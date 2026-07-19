# Task 1 — position-based vs current-based force sensing

**Question.** Our proprioceptive force sensor's input includes the applied joint
torque, which on real hardware needs motor current (`τ ≈ K_t·i`, noisy). Can we
drop current and reconstruct torque from **position** (the PD law, encoders only)
with little accuracy loss?

**Answer: yes — negligible loss.** Two independent measures on the Isaac v4 data.

## 1. Torque-reconstruction fidelity (no training)
Per-joint correlation between measured torque and the position reconstruction
`τ̂ = kp·(q_des − q) − kd·q̇` (uses only encoder position + known gains):

- **median 0.974, mean 0.954**; all 29 joints ≥ 0.80, 26/29 ≥ 0.90.

Position reconstructs each joint's torque almost perfectly, without current.

## 2. End-to-end sensor accuracy
Train the contact sensor with 3 torque-channel variants, everything else equal
(`wbc_v4_motion + wbc_v4_static`, MLP, W=6, 30 epochs):

| variant | active_acc | det_f1 | force_cos |
|---|---|---|---|
| **A** measured torque (current proxy) | 0.440 | 0.602 | 0.313 |
| **B** PD-reconstructed from position (no current) | **0.441** | 0.597 | 0.310 |
| **C** no torque (pure kinematics) | 0.416 | 0.596 | 0.287 |

**B ≈ A** (gap ≈ 0). Even **C** (torque dropped entirely) is only marginally
worse → the torque channel is largely redundant with the position channels.
(`task1_compare.png`.)

## Takeaway
The force sensor does **not** need current-based torque. Encoder position + a
known-gain PD reconstruction gives essentially the same accuracy — and is more
realizable on hardware (encoders are clean and cheap; current-based torque is
noisy: friction, `K_t` drift, gearbox losses).

## Caveats / honest notes
- Absolute accuracy here is modest (quick MLP config, not the tuned champion);
  the **A/B/C comparison is controlled**, so the *relative* result is robust.
- **Stiffness dependence** (does position-only degrade for stiff controllers,
  where a given force causes a smaller joint deviation?) was **not** reliably
  measured tonight: the MuJoCo cross-controller datasets use different actuator
  gains than `IMP_KP`, so the reconstruction there is confounded (a flat cross-
  joint correlation looked like it dropped with stiffness, but that is a
  gains-mismatch artifact, not a validated law). A proper multi-stiffness test
  with matched gains + encoder-noise sweep is the natural next step.

Code: `experiments/posforce/run.py` (A/B/C), `cross_stiffness.py` /
correlation analysis (stiffness, WIP). Data written to `data/wbc/posforce/`.
