"""M1: drive OUR G1 (g1.xml) with the external Sonic policy in standalone MuJoCo.

Assembles the Sonic decoder obs from OUR MuJoCo state, remaps joint order by
name, applies Sonic's PD, and checks the robot holds the default standing pose
for N seconds without exploding. Saves a positions-over-time rollout as proof.

    conda run -n gentle python experiments/sonic_m1_drive.py --seconds 8
"""
import os
import sys
import argparse
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default=os.path.join(REPO, "forcesense/assets/g1/g1.xml"))
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--root_z", type=float, default=0.793)
    p.add_argument("--out", default=os.path.join(REPO, "data/wbc/sonic/m1_rollout.npz"))
    p.add_argument("--token", default=os.path.join(REPO, "data/wbc/sonic/tok_stand_opt.npy"),
                   help="optimized standing token .npy (falls back to encoder if missing)")
    p.add_argument("--use-planner", action="store_true",
                   help="use the kinematic planner (mode 0 idle) for a real token")
    p.add_argument("--replan-every", type=int, default=25,
                   help="re-plan+re-encode the token every N control steps (planner mode)")
    args = p.parse_args()

    from forcesense.sim2sim import Sim2Sim
    from forcesense.collect.sonic_policy import (
        SonicPolicy, SONIC_DEFAULT_MJ, SONIC_KP_MJ, SONIC_KD_MJ,
        SONIC_ACTION_SCALE_MJ, SONIC_EFFORT_MJ)

    sim = Sim2Sim(args.xml, wrist_ref_mode="still")
    sim.setup_sonic(SONIC_DEFAULT_MJ, SONIC_KP_MJ, SONIC_KD_MJ,
                    SONIC_ACTION_SCALE_MJ, SONIC_EFFORT_MJ)
    sim.reset_sonic(root_z=args.root_z)
    policy = SonicPolicy(root_z=args.root_z, token_path=args.token,
                         use_planner=args.use_planner)
    policy.prime(sim)

    print(f"[m1] token norm={np.linalg.norm(policy.token):.3f}  "
          f"nq={sim.m.nq} nu={sim.m.nu}")
    steps = int(args.seconds * Sim2Sim.CTRL_HZ)
    zs, tilts, actnorms, taunorms = [], [], [], []
    qpos_log = []
    fell_at = None
    for k in range(steps):
        if args.use_planner and args.replan_every > 0 and k > 0 and k % args.replan_every == 0:
            policy.refresh_token(sim)
        act_mj = policy.act(sim)
        sim.control_step_sonic(act_mj)
        policy.record(sim)
        z = float(sim.d.qpos[2])
        grav = sim._proj_grav_b()
        tilt = float(np.degrees(np.arccos(np.clip(-grav[2], -1, 1))))
        zs.append(z); tilts.append(tilt)
        actnorms.append(float(np.linalg.norm(act_mj)))
        taunorms.append(float(np.abs(sim.applied_torque).mean()))
        qpos_log.append(sim.d.qpos[sim.sonic_qadr].copy())
        if z < 0.4 or tilt > 70:
            fell_at = k / Sim2Sim.CTRL_HZ
            print(f"[m1] FELL at t={fell_at:.2f}s  z={z:.3f} tilt={tilt:.1f}deg")
            break
        if (k + 1) % 50 == 0:
            print(f"  t={(k+1)//50:2d}s  z={z:.3f}  tilt={tilt:4.1f}deg  "
                  f"|act|={actnorms[-1]:.2f}  |tau|={taunorms[-1]:.1f}")

    ok = fell_at is None and zs[-1] > 0.55 and tilts[-1] < 30
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, z=np.array(zs), tilt=np.array(tilts),
                        act_norm=np.array(actnorms), tau_mean=np.array(taunorms),
                        qpos=np.array(qpos_log), ok=ok, fell_at=fell_at or -1.0)
    print(f"[m1] {'PASS' if ok else 'FAIL'}  final z={zs[-1]:.3f} tilt={tilts[-1]:.1f}deg  "
          f"steps={len(zs)}  -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
