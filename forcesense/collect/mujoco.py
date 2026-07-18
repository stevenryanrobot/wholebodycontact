"""Track C1: MuJoCo-domain collection for the whole-body contact sensor.

Why: v4 sensors trained on Isaac read MuJoCo standing-at-rest as contact
(rest det_p ~0.7, FP 0.97 at th 0.5) — Isaac static standing != MuJoCo static
standing proprioception. Cure: same-schema data from the deployment domain.

What: N parallel worker processes, each a headless forcesense.sim2sim.Sim2Sim
(the verified MuJoCo replica of the Isaac obs/action pipeline) standing under
the exported ONNX policy while a scripted net_pull-style scheduler replicates
the forcesense/cfg/collect_v4.yaml force profile:
  rest 20-60 (30% of cycles: 200-500 for extra pure-rest negatives)
  -> ramp_up 10-30 -> hold 40-400 -> ramp_down 10-30, |F| ~ U[10,40] N,
  isotropic 3D direction, one random link of the SAME 24-link set as the
  Isaac v4 files (order read from the reference h5 so onehot indices match),
  force applied at the link origin with the same torso net-wrench limiter
  (inside Sim2Sim.control_step). Falls (pelvis z<0.45 or tilt>60deg) reset the
  worker (with small joint-noise init), mirroring Isaac collection.

Output: one h5 with the exact Isaac collection schema —
  X [N,320] wbc frames, Y [N, 9+24+6] labels
  ([point_b(3) | force_b/40(3) | force_w/40(3) | onehot(24) | phase(4) |
    timer(1) | mag(1)]), rows interleaved as [t*W + w] so the trainer's
  [T,E,D] reshape (n_envs=W) keeps per-column temporal continuity.
  attrs: domain='mujoco', n_envs=W, body_names, force_max=40.0, ...

Usage:
    python forcesense/collect/mujoco.py --measure          # throughput probe
    python forcesense/collect/mujoco.py --workers 12 --num_steps 60000 \
        -o data/wbc/datasets/wbc_v4_mujoco.h5
"""
import os
import sys
import time
import argparse
import multiprocessing as mp

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

DEFAULT_ONNX = os.path.join(REPO, "controllers/ceer/checkpoints/G1GENTLE-07-07_18-28/policy.onnx")
DEFAULT_XML = os.path.join(REPO, "forcesense/assets/g1/g1.xml")
DEFAULT_REF_H5 = os.path.join(REPO, "data/wbc/datasets/wbc_v4_motion.h5")

LABEL_PREFIX = 9
LABEL_SUFFIX = 6
FORCE_RANGE = (10.0, 40.0)
RAMP_UP = (10, 30)
HOLD = (40, 400)
RAMP_DOWN = (10, 30)
REST = (20, 60)
REST_LONG = (200, 500)      # 30% of cycles: extra pure-rest negatives
REST_LONG_PROB = 0.3
FALL_Z = 0.45
FALL_GRAV_Z = -0.5          # projected gravity z > -0.5 -> tilted > ~60 deg


def load_body_names(ref_h5):
    import h5py
    with h5py.File(ref_h5, "r") as h:
        names = [b.decode() if isinstance(b, bytes) else str(b)
                 for b in h.attrs["body_names"]]
        fmax = float(h.attrs["force_max"])
    return names, fmax


class ForceScheduler:
    """net_pull phase machine (motion_tracking.py:net_pull_schedule) for one env."""
    PHASES = ("rest", "ramp_up", "hold", "ramp_down")

    def __init__(self, rng, num_bodies):
        self.rng = rng
        self.num_bodies = num_bodies
        self.reset(first=True)

    def _rest_steps(self):
        lo, hi = REST_LONG if self.rng.random() < REST_LONG_PROB else REST
        return int(self.rng.integers(lo, hi + 1))

    def reset(self, first=False):
        self.phase = 0
        # long first rest -> guaranteed settle frames after (re)init
        self.timer = int(self.rng.integers(100, 201)) if first else self._rest_steps()
        self.body = 0
        self.dir_w = np.zeros(3)
        self.mag_target = 0.0
        self.mag = 0.0
        self.ramp_steps = 1

    def step(self):
        """Advance one control step; returns (force_w[3], phase_idx, timer, mag)."""
        self.timer -= 1
        if self.timer < 0:
            if self.phase == 0:                      # rest -> ramp_up
                self.phase = 1
                self.body = int(self.rng.integers(0, self.num_bodies))
                v = self.rng.normal(size=3)
                self.dir_w = v / (np.linalg.norm(v) + 1e-9)
                self.mag_target = float(self.rng.uniform(*FORCE_RANGE))
                self.ramp_steps = int(self.rng.integers(RAMP_UP[0], RAMP_UP[1] + 1))
                self.timer = self.ramp_steps
                self.ramp_from = self.mag
            elif self.phase == 1:                    # ramp_up -> hold
                self.phase = 2
                self.timer = int(self.rng.integers(HOLD[0], HOLD[1] + 1))
            elif self.phase == 2:                    # hold -> ramp_down
                self.phase = 3
                self.ramp_steps = int(self.rng.integers(RAMP_DOWN[0], RAMP_DOWN[1] + 1))
                self.timer = self.ramp_steps
                self.ramp_from = self.mag
            else:                                    # ramp_down -> rest
                self.phase = 0
                self.timer = self._rest_steps()
                self.dir_w = np.zeros(3)
        # magnitude (TemporalLerp equivalent: linear ramps, hold at target)
        if self.phase == 1:
            frac = 1.0 - max(self.timer, 0) / max(self.ramp_steps, 1)
            self.mag = self.ramp_from + (self.mag_target - self.ramp_from) * frac
        elif self.phase == 2:
            self.mag = self.mag_target
        elif self.phase == 3:
            frac = 1.0 - max(self.timer, 0) / max(self.ramp_steps, 1)
            self.mag = self.ramp_from * (1.0 - frac)
        else:
            self.mag = 0.0
        return self.dir_w * self.mag, self.phase, self.timer, self.mag


def worker(rank, seed, num_steps, out_npz, xml, onnx, body_names, force_max, quiet,
           kp_scale=1.0, kd_scale=1.0, realizable=False):
    from forcesense.sim2sim import Sim2Sim, OnnxPolicy, quat_apply_inverse
    rng = np.random.default_rng(seed)
    sim = Sim2Sim(xml, seed=seed, wrist_ref_mode="still")
    # controller variant: scale the deploy PD gains to emulate a DIFFERENT
    # downstream policy's torque distribution (cross-policy generalization test).
    sim.kp = sim.kp * kp_scale
    sim.kd = sim.kd * kd_scale
    policy = OnnxPolicy(onnx)
    K = len(body_names)
    bid = {}
    for n in body_names:
        b = sim.mujoco.mj_name2id(sim.m, sim.mujoco.mjtObj.mjOBJ_BODY, n)
        assert b >= 0, f"body {n} not in MJCF"
        bid[n] = b
    sched = ForceScheduler(rng, K)

    def reinit():
        sim.reset()
        # small joint-noise init for pose diversity (Isaac resets sample motion
        # poses; the MuJoCo demo always starts near default, so keep it mild)
        sim.d.qpos[sim.qadr] += rng.normal(0.0, 0.03, size=len(sim.qadr))
        sim.mujoco.mj_forward(sim.m, sim.d)
        sched.reset()

    X = np.empty((num_steps, 320), dtype=np.float32)
    R = np.empty((num_steps, 29), dtype=np.float32)   # policy-invariant residual
    G = np.empty((num_steps, 29), dtype=np.float32)   # GRF (constraint) projection
    Y = np.empty((num_steps, LABEL_PREFIX + K + LABEL_SUFFIX), dtype=np.float32)
    falls = 0
    t0 = time.time()
    for i in range(num_steps):
        F_w, phase, timer, mag = sched.step()
        body = body_names[sched.body]
        obs = sim.policy_obs()
        act = policy(obs)
        sim.control_step(act, ext_force_w=F_w if mag > 1e-9 else None, ext_body=body)

        X[i] = sim.wbc_obs()
        R[i] = sim.ext_torque_residual(realizable=realizable)
        G[i] = sim.grf_torque()
        root_p, root_q = sim.d.qpos[0:3].copy(), sim.root_quat()
        point_b = quat_apply_inverse(root_q, sim.d.xpos[bid[body]] - root_p)
        f_b = quat_apply_inverse(root_q, F_w)
        y = np.zeros(LABEL_PREFIX + K + LABEL_SUFFIX, dtype=np.float32)
        y[0:3] = point_b
        y[3:6] = f_b / force_max
        y[6:9] = F_w / force_max
        y[LABEL_PREFIX + sched.body] = 1.0
        y[LABEL_PREFIX + K + phase] = 1.0
        y[LABEL_PREFIX + K + 4] = max(timer, 0) / 250.0
        y[LABEL_PREFIX + K + 5] = mag / force_max
        Y[i] = y

        if sim.d.qpos[2] < FALL_Z or sim.grav_hist[0][2] > FALL_GRAV_Z:
            falls += 1
            reinit()
        if not quiet and rank == 0 and (i + 1) % 5000 == 0:
            r = (i + 1) / (time.time() - t0)
            print(f"[mj-collect w0] {i+1}/{num_steps} steps ({r:.0f} steps/s, "
                  f"falls={falls})", flush=True)
    np.savez_compressed(out_npz, X=X, R=R, G=G, Y=Y, falls=falls)
    print(f"[mj-collect w{rank}] done: {num_steps} steps, {falls} falls, "
          f"{num_steps/(time.time()-t0):.0f} steps/s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output", default=os.path.join(REPO, "data/wbc/datasets/wbc_v4_mujoco.h5"))
    p.add_argument("--onnx", default=DEFAULT_ONNX)
    p.add_argument("--xml", default=DEFAULT_XML)
    p.add_argument("--ref_h5", default=DEFAULT_REF_H5,
                   help="Isaac v4 h5 whose body_names order defines the onehot.")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--num_steps", type=int, default=60000,
                   help="Control steps PER WORKER (samples = workers*num_steps).")
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--measure", action="store_true",
                   help="Single-process 1500-step throughput probe, then exit.")
    p.add_argument("--kp_scale", type=float, default=1.0,
                   help="Scale deploy PD kp to emulate a different controller.")
    p.add_argument("--kd_scale", type=float, default=1.0,
                   help="Scale deploy PD kd to emulate a different controller.")
    p.add_argument("--tmp_dir", default=None)
    p.add_argument("--resid_mode", choices=["full", "realizable"], default="full",
                   help="'full' subtracts qfrc_constraint (sim-privileged GRF); "
                        "'realizable' drops it (hardware-computable residual).")
    args = p.parse_args()
    _realizable = args.resid_mode == "realizable"

    body_names, force_max = load_body_names(args.ref_h5)
    print(f"[mj-collect] {len(body_names)} bodies (order from {args.ref_h5})", flush=True)

    if args.measure:
        t0 = time.time()
        worker(0, args.seed, 1500, os.path.join("/tmp", "wbc_mj_measure.npz"),
               args.xml, args.onnx, body_names, force_max, quiet=False,
               kp_scale=args.kp_scale, kd_scale=args.kd_scale)
        dt = time.time() - t0
        print(f"[mj-collect] MEASURE: 1500 steps in {dt:.1f}s -> "
              f"{1500/dt:.0f} steps/s/proc (sim-time x{1500/50/dt:.1f} realtime)", flush=True)
        return

    tmp_dir = args.tmp_dir or os.path.join(os.path.dirname(args.output), "mj_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    ctx = mp.get_context("spawn")
    procs = []
    outs = []
    for w in range(args.workers):
        out_npz = os.path.join(tmp_dir, f"worker_{w}.npz")
        outs.append(out_npz)
        pr = ctx.Process(target=worker,
                         args=(w, args.seed + w, args.num_steps, out_npz,
                               args.xml, args.onnx, body_names, force_max, False,
                               args.kp_scale, args.kd_scale, _realizable))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
    bad = [w for w, pr in enumerate(procs) if pr.exitcode != 0]
    assert not bad, f"workers failed: {bad}"

    # merge: interleave rows as t*W + w so reshape(T, W, D) keeps per-column
    # (per-worker) temporal continuity — same layout as Isaac collection.
    import h5py
    W = args.workers
    Xs = [np.load(o)["X"] for o in outs]
    Rs = [np.load(o)["R"] for o in outs]
    Gs = [np.load(o)["G"] for o in outs]
    Ys = [np.load(o)["Y"] for o in outs]
    falls = sum(int(np.load(o)["falls"]) for o in outs)
    T = min(x.shape[0] for x in Xs)
    X = np.stack([x[:T] for x in Xs], axis=1).reshape(T * W, -1)
    R = np.stack([r[:T] for r in Rs], axis=1).reshape(T * W, -1)
    G = np.stack([g[:T] for g in Gs], axis=1).reshape(T * W, -1)
    Y = np.stack([y[:T] for y in Ys], axis=1).reshape(T * W, -1)
    with h5py.File(args.output, "w") as h:
        h.create_dataset("X", data=X, chunks=(4096, X.shape[1]), dtype="f4")
        h.create_dataset("R", data=R, chunks=(4096, R.shape[1]), dtype="f4")
        h.create_dataset("G", data=G, chunks=(4096, G.shape[1]), dtype="f4")
        h.create_dataset("Y", data=Y, chunks=(4096, Y.shape[1]), dtype="f4")
        h.attrs["num_bodies"] = len(body_names)
        h.attrs["force_max"] = force_max
        h.attrs["body_names"] = np.array(body_names, dtype=h5py.special_dtype(vlen=str))
        h.attrs["input_dim"] = X.shape[1]
        h.attrs["label_dim"] = Y.shape[1]
        h.attrs["label_prefix"] = LABEL_PREFIX
        h.attrs["label_suffix"] = LABEL_SUFFIX
        h.attrs["run_path"] = "mujoco_sim2sim"
        h.attrs["domain"] = "mujoco"
        h.attrs["n_envs"] = W
        h.attrs["static_command"] = True
        h.attrs["total_falls"] = falls
        h.attrs["kp_scale"] = args.kp_scale
        h.attrs["kd_scale"] = args.kd_scale
    for o in outs:
        os.remove(o)
    print(f"[mj-collect] merged {T}x{W} = {T*W} samples ({falls} falls) -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
