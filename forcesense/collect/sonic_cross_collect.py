"""Collect cross_H_sonic.h5: external-push force-sensing dataset with the robot
driven by the EXTERNAL Sonic (GR00T-WholeBodyControl) decoder policy instead of
our CEER ONNX. Same X[N,320]/R[N,29]/G[N,29]/Y[N,39] format and same push
scheduler as forcesense/collect/mujoco_collect.py (the "H" cross-controller).

Only the action source changes: SonicPolicy.act -> control_step_sonic. This is
the cross-policy axis SixthSense never tests (a fully different controller with
its own gains, defaults and action scale). Requires the Sonic obs-builder fix
(m2i joint gather) verified in data/wbc/sonic/make_ref_and_diff.py.

    conda run -n gentle python forcesense/collect/sonic_cross_collect.py \
        -o data/wbc/cross/cross_H_sonic.h5 --workers 12 --num_steps 30000
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from forcesense.collect.mujoco_collect import (  # noqa: E402
    DEFAULT_REF_H5, DEFAULT_XML, FALL_GRAV_Z, FALL_Z, LABEL_PREFIX, LABEL_SUFFIX,
    ForceScheduler, load_body_names,
)

ROOT_Z = 0.755  # feet-on-floor pelvis height for OUR g1.xml (see sonic_m1_drive)


def _standing_token(policy, root_z):
    """Constant hold-default-pose reference token (planner-obs encoder path,
    IsaacLab joint order) — the lowest-action standing reference (see M1)."""
    from controllers.sonic.sonic_planner import build_encoder_obs_from_qpos
    from controllers.sonic.sonic_policy import SONIC_DEFAULT_MJ
    qpos = np.zeros(36, np.float32)
    qpos[2] = root_z
    qpos[3] = 1.0
    qpos[7:36] = SONIC_DEFAULT_MJ.astype(np.float32)
    enc_obs = build_encoder_obs_from_qpos(np.tile(qpos, (64, 1)), qpos[3:7],
                                          joint_order="isaaclab")
    return policy.enc.run(None, {"obs_dict": enc_obs})[0].ravel().astype(np.float32)


def worker(rank, seed, num_steps, out_npz, xml, body_names, force_max, quiet,
           realizable=False):
    from forcesense.sim2sim import Sim2Sim, quat_apply_inverse
    from controllers.sonic.sonic_policy import (
        SonicPolicy, SONIC_ACTION_SCALE_MJ, SONIC_DEFAULT_MJ, SONIC_EFFORT_MJ,
        SONIC_KD_MJ, SONIC_KP_MJ)

    rng = np.random.default_rng(seed)
    sim = Sim2Sim(xml, seed=seed, wrist_ref_mode="still")
    sim.setup_sonic(SONIC_DEFAULT_MJ, SONIC_KP_MJ, SONIC_KD_MJ,
                    SONIC_ACTION_SCALE_MJ, SONIC_EFFORT_MJ)
    sim.reset_sonic(root_z=ROOT_Z)
    policy = SonicPolicy(root_z=ROOT_Z, token_path="__none__")
    policy.token = _standing_token(policy, ROOT_Z)
    policy.prime(sim)

    K = len(body_names)
    bid = {}
    for n in body_names:
        b = sim.mujoco.mj_name2id(sim.m, sim.mujoco.mjtObj.mjOBJ_BODY, n)
        assert b >= 0, f"body {n} not in MJCF"
        bid[n] = b
    sched = ForceScheduler(rng, K)

    def reinit():
        sim.reset_sonic(root_z=ROOT_Z)
        sim.d.qpos[sim.sonic_qadr] += rng.normal(0.0, 0.03, size=len(sim.sonic_qadr))
        sim.mujoco.mj_forward(sim.m, sim.d)
        policy.prime(sim)
        sched.reset()

    X = np.empty((num_steps, 320), dtype=np.float32)
    R = np.empty((num_steps, 29), dtype=np.float32)
    G = np.empty((num_steps, 29), dtype=np.float32)
    Y = np.empty((num_steps, LABEL_PREFIX + K + LABEL_SUFFIX), dtype=np.float32)
    falls = 0
    t0 = time.time()
    for i in range(num_steps):
        F_w, phase, timer, mag = sched.step()
        body = body_names[sched.body]
        act_mj = policy.act(sim)
        sim.control_step_sonic(act_mj, ext_force_w=F_w if mag > 1e-9 else None,
                               ext_body=body)
        policy.record(sim)

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
            print(f"[sonic-collect w0] {i+1}/{num_steps} ({r:.0f} steps/s, "
                  f"falls={falls})", flush=True)
    np.savez_compressed(out_npz, X=X, R=R, G=G, Y=Y, falls=falls)
    print(f"[sonic-collect w{rank}] done: {num_steps} steps, {falls} falls, "
          f"{num_steps/(time.time()-t0):.0f} steps/s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--output",
                   default=os.path.join(REPO, "data/wbc/cross/cross_H_sonic.h5"))
    p.add_argument("--xml", default=DEFAULT_XML)
    p.add_argument("--ref_h5", default=DEFAULT_REF_H5)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--num_steps", type=int, default=30000)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--measure", action="store_true")
    p.add_argument("--tmp_dir", default=None)
    p.add_argument("--resid_mode", choices=["full", "realizable"], default="full")
    args = p.parse_args()
    _realizable = args.resid_mode == "realizable"

    body_names, force_max = load_body_names(args.ref_h5)
    print(f"[sonic-collect] {len(body_names)} bodies (order from {args.ref_h5})",
          flush=True)

    if args.measure:
        t0 = time.time()
        worker(0, args.seed, 1500, os.path.join("/tmp", "sonic_measure.npz"),
               args.xml, body_names, force_max, quiet=False, realizable=_realizable)
        dt = time.time() - t0
        print(f"[sonic-collect] MEASURE: 1500 steps in {dt:.1f}s -> "
              f"{1500/dt:.0f} steps/s/proc", flush=True)
        return

    tmp_dir = args.tmp_dir or os.path.join(os.path.dirname(args.output), "mj_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    ctx = mp.get_context("spawn")
    procs, outs = [], []
    for w in range(args.workers):
        out_npz = os.path.join(tmp_dir, f"sonic_worker_{w}.npz")
        outs.append(out_npz)
        pr = ctx.Process(target=worker,
                         args=(w, args.seed + w, args.num_steps, out_npz, args.xml,
                               body_names, force_max, False, _realizable))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
    bad = [w for w, pr in enumerate(procs) if pr.exitcode != 0]
    assert not bad, f"workers failed: {bad}"

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
        h.attrs["controller"] = "sonic_groot_wbc"
        h.attrs["n_envs"] = W
        h.attrs["static_command"] = True
        h.attrs["total_falls"] = falls
        h.attrs["kp_scale"] = 1.0
        h.attrs["kd_scale"] = 1.0
    for o in outs:
        os.remove(o)
    print(f"[sonic-collect] merged {T}x{W} = {T*W} samples ({falls} falls) -> "
          f"{args.output}", flush=True)


if __name__ == "__main__":
    main()
