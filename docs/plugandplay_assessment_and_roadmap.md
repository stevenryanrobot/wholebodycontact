# Plug-and-Play Proprioceptive Force Sensing — Assessment & Roadmap

*Status as of 2026-07-14. Companion to `paper/plugandplay/` and the memory notes
`crosspolicy-plugandplay-result`, `sixthsense-comparison-and-direction`,
`proact-experiment-results`.*

---

## 1. The method (one-liner + architecture)

**A proprioception-only whole-body external-force estimator that plugs into any
low-level controller and senses the legs, by feeding the network a
controller-invariant physics residual instead of (only) raw proprioception.**

Pipeline / architecture (all in sim today):
- **Collection**: MuJoCo sim2sim, ~22× realtime, 12 parallel workers. A frozen
  low-level policy stands/moves; `net_pull` applies known external forces
  (ground-truth labels). `scripts/wbc_collect_mujoco.py` (`--kp_scale/--kd_scale`
  emulate different controllers).
- **Input channels**:
  - `proprio` (320): joint torque, joint-pos history, commanded target, IMU
    history, prev actions.
  - `resid` (29): **policy-invariant external-joint-torque residual** from the
    equations of motion `M q̈ + qfrc_bias − qfrc_actuator − qfrc_passive −
    qfrc_constraint ≈ Jᵀ F_ext` (`Sim2Sim.ext_torque_residual()`).
  - `both` = concat.
- **Model**: W=6 window, small MLP, dual head (contact detection + 5-region
  localization). `scripts/wbc_crosspolicy.py`.
- **Output**: per-region contact (5 regions: L/R arm, L/R leg, trunk) + detection.

## 2. Positioning vs SixthSense (our differentiation)

SixthSense (arXiv 2605.01427) = passive proprioceptive whole-body wrench on ONE
real G1 / ONE pipeline; standing/walking/tracking link-loc 62/58/37%.

| Axis | SixthSense | Ours |
|---|---|---|
| Controller-invariance (plug-and-play) | not tested (one pipeline) | **core contribution** |
| Legs under stance-masking | link-loc ~58% | **residual → 0.89 region recall** |
| Real-robot recalibration | — | residual is controller-agnostic + NEXT-style free-space calib |
| Sensors | proprioception only | proprioception only (same moat) |
| Robot / hardware | real G1 | **sim only (gap)** |

## 3. Results so far (verified)

**Cross-controller robustness** — leave-one-out over **7 controllers** (kp 0.4–2.0,
kd 0.5–1.5, independent), train on 6 / eval on unseen (`sweep_result.json`):

| feature | worst regAcc | mean regAcc | worst prec | mean prec |
|---|---|---|---|---|
| proprio | 0.417 | 0.653 | 0.53 | 0.80 |
| **resid** | **0.905** | **0.931** | **0.969** | **0.987** |
| both | 0.777 | 0.897 | 0.671 | 0.883 |
| both + proprio-dropout | 0.735–0.760 | ~0.88 | ~0.65 | ~0.89 |

→ **RESID-ALONE is the robust recipe.** Controller randomization helps proprio's
interpolation but not extreme extrapolation; `both`/dropout re-inherit the
misleading OOD proprio at test time. Physics residual is controller-invariant by
construction.

**Per-region recall (cross-controller, 7-fold avg)** — *the legs* (`cross_perregion`):

| | left_arm | right_arm | left_leg | right_leg | trunk | LEGS | ARMS |
|---|---|---|---|---|---|---|---|
| proprio | 0.807 | 0.794 | 0.559 | 0.563 | 0.427 | 0.561 | 0.801 |
| **resid** | 0.982 | 0.976 | **0.894** | **0.884** | 0.855 | **0.889** | 0.979 |

→ **The residual solves the legs too** (0.56 → 0.89): the EOM residual explicitly
subtracts the ground-reaction (`qfrc_constraint`), removing the double-support
stance-masking that hides leg forces in raw proprioception. One mechanism fixes
both plug-and-play AND legs.

## 4. Optimization points (prioritized) — what a reviewer will attack

### ⚠️ Load-bearing (must close before the paper is credible)

1. **Realizable residual (BIGGEST hole).** The residual uses `qfrc_constraint`,
   a **sim-privileged** ground-reaction term (free in sim, absent on hardware).
   Both headline results (plug-and-play robustness *and* legs) rest entirely on
   this. **Must show the residual still works when the ground reaction is
   estimated by a realizable floating-base momentum observer (MOB)** — not sim
   ground truth. Tractable in sim right now (recompute residual without
   `qfrc_constraint`; add noise). If it survives → strong paper; if not → rethink.
   **Recommended to do FIRST.**

2. **A genuinely different policy (not gain-scaling).** "Different controller" is
   currently scaled kp/kd — a mild proxy. A structurally different net (different
   action space / objective / reference) is a bigger shift; the plug-and-play
   claim's credibility hinges on at least one such policy. Higher effort (train/
   obtain a new controller). Until then, the measured gap is a **lower bound**.

### Nice-to-have (needed to fully match SixthSense's table)

3. **Force-vector metrics** — SixthSense reports force RMSE ~2N; we report only
   region+detection. Add magnitude/direction quality.
4. **Dynamic tasks** — SixthSense has standing/walking/tracking; we are
   standing-heavy. Add walking.
5. **Finer output** — SixthSense is a 30-link wrench field; we are 5-region.
   Either go per-link or justify the granularity.
6. **Statistical rigor** — multiple seeds / error bars.
7. **Any hardware at all** — a proprioception paper with zero real robot is weak
   for a robotics venue; even a small real-robot demo would help a lot.

## 5. Recommendation / roadmap

1. **Now**: close optimization **#1 (realizable residual / MOB)** in sim — it is
   the single load-bearing test and is doable immediately. Re-run the 7-controller
   sweep + per-region with the realizable residual.
2. **If #1 holds**: write the paper around "realizable residual → plug-and-play +
   legs", with #2 as the key remaining validation and #3–#7 in limitations/future.
3. **Then / bigger**: a genuinely different policy (#2), force metrics (#3),
   walking (#4), and eventually hardware (#7 / MOB on real G1).

**Bottom line**: the idea is novel and well-motivated (plug-and-play + legs via
one physics channel, an axis SixthSense never tests), and the sim numbers are
strong — but the whole thing currently rests on a sim-privileged residual.
Make the residual *realizable* first; that decides whether this is a cute sim
result or a credible contribution.
