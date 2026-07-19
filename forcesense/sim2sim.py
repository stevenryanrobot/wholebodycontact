"""Whole-Body Contact — standalone MuJoCo sim2sim prototype.

Runs the exported low-level G1 ONNX policy (controllers/ceer/checkpoints/G1GENTLE-07-07_18-28)
in plain MuJoCo (no Isaac), rebuilds its 257-dim "policy" observation and the
320-dim "wbc_input_" sensor features, runs the v3 contact-force sensor
(data/wbc/sweep_v3/force_sensor_v3_best.pt) on top, applies known external
forces via xfrc_applied, and reports predicted vs ground-truth contact.

Usage (headless smoke test):
    source scripts/start_gentle_local.sh
    python forcesense/sim2sim.py                      # stand test + 3 pushes
    python forcesense/sim2sim.py --push-body torso_link --push-force 25
    DISPLAY=:1 python forcesense/sim2sim.py --viewer  # interactive-ish view

Every Isaac-semantics assumption is marked with "ISAAC:" comments and was
verified against the repo source (file:line refs in comments).
"""
import os
import re
import sys
import json
import argparse
import numpy as np

# --------------------------------------------------------------------------- #
# constants verified against the training stack
# --------------------------------------------------------------------------- #
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ISAAC: articulation (PhysX BFS) joint order. Source: scratchpad
# gentle-humanoid/config/tracking.yaml action_joint_names + controller.yaml
# isaac_joint_names_state (identical); this is the order of ALL joint-space
# observations and of the policy action output.
ISAAC_JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
NJ = len(ISAAC_JOINTS)

# ISAAC: default joint pos = G1_CFG.init_state.joint_pos
# (active_adaptation/assets/humanoid.py:39-56). This is default_joint_pos in
# the action manager: q_des = default + action_scale * filtered_action.
DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.28,
    ".*_knee_joint": 0.5,
    ".*_ankle_pitch_joint": -0.23,
    ".*_elbow_joint": 0.87,
    "left_shoulder_roll_joint": 0.16,
    "left_shoulder_pitch_joint": 0.35,
    "right_shoulder_roll_joint": -0.16,
    "right_shoulder_pitch_joint": 0.35,
}
# ISAAC: action scaling regex map (run cfg task.action.action_scaling ==
# policy.json action_scaling).
ACTION_SCALING = {
    ".*elbow_joint": 1.0, ".*shoulder.*": 1.0, ".*wrist.*": 1.0,
    ".*hip_roll.*": 0.25, ".*hip_yaw.*": 0.25, ".*hip_pitch.*": 0.5,
    ".*knee.*": 0.5, ".*waist.*": 0.25, ".*ankle.*": 0.5,
}
# ISAAC: PD gains / effort limits from G1_CFG actuators (humanoid.py:59-148).
# NOTE: the run '3kp_amass_limmt_full_stiff600' uses robot g1_col_full which is
# a deepcopy of G1_CFG with only the USD path changed — there is NO 3x-kp gain
# variant in the repo (verified: assets/__init__.py, run cfg has no override;
# only motor_params_implicit randomization 0.9-1.1x). Base gains apply. They
# also match gentle-humanoid config/controller.yaml kps_real/kds_real exactly.
KP = {
    ".*_hip_yaw_joint": 40.17923847137318, ".*_hip_pitch_joint": 40.17923847137318,
    ".*_hip_roll_joint": 99.09842777666113, ".*_knee_joint": 99.09842777666113,
    "waist_yaw_joint": 40.17923847137318,
    "waist_roll_joint": 28.50124619574858, "waist_pitch_joint": 28.50124619574858,
    ".*_ankle_pitch_joint": 28.50124619574858, ".*_ankle_roll_joint": 28.50124619574858,
    ".*_shoulder_pitch_joint": 14.25062309787429, ".*_shoulder_roll_joint": 14.25062309787429,
    ".*_shoulder_yaw_joint": 14.25062309787429, ".*_elbow_joint": 14.25062309787429,
    ".*_wrist_roll_joint": 14.25062309787429,
    ".*_wrist_pitch_joint": 16.77832748089279, ".*_wrist_yaw_joint": 16.77832748089279,
}
KD = {
    ".*_hip_yaw_joint": 2.5578897650279457, ".*_hip_pitch_joint": 2.5578897650279457,
    ".*_hip_roll_joint": 6.3088018534966395, ".*_knee_joint": 6.3088018534966395,
    "waist_yaw_joint": 2.5578897650279457,
    "waist_roll_joint": 1.814445686584846, "waist_pitch_joint": 1.814445686584846,
    ".*_ankle_pitch_joint": 1.814445686584846, ".*_ankle_roll_joint": 1.814445686584846,
    ".*_shoulder_pitch_joint": 0.907222843292423, ".*_shoulder_roll_joint": 0.907222843292423,
    ".*_shoulder_yaw_joint": 0.907222843292423, ".*_elbow_joint": 0.907222843292423,
    ".*_wrist_roll_joint": 0.907222843292423,
    ".*_wrist_pitch_joint": 1.06814150219, ".*_wrist_yaw_joint": 1.06814150219,
}
EFFORT_LIMIT = {
    ".*_hip_yaw_joint": 88.0, ".*_hip_roll_joint": 139.0, ".*_hip_pitch_joint": 88.0,
    ".*_knee_joint": 139.0, "waist_yaw_joint": 88.0, "waist_roll_joint": 50.0,
    "waist_pitch_joint": 50.0, ".*_ankle_pitch_joint": 50.0, ".*_ankle_roll_joint": 50.0,
    ".*_shoulder_pitch_joint": 25.0, ".*_shoulder_roll_joint": 25.0,
    ".*_shoulder_yaw_joint": 25.0, ".*_elbow_joint": 25.0, ".*_wrist_roll_joint": 25.0,
    ".*_wrist_pitch_joint": 5.0, ".*_wrist_yaw_joint": 5.0,
}

# ISAAC: cfg force_safe_bounds=[30,30], force_safe_default=30 for this run
# (scripts/wandb/3kp_amass_limmt_full_stiff600/files/cfg.yaml:33-36), so the
# command's force_limit entry was constant 30.0 N (raw Newtons, not normalized;
# motion_tracking.py:2434 uses force_safe_limit_tl.current directly).
FORCE_LIMIT_CMD = 30.0

# ISAAC: net-wrench limiter about the torso (motion_tracking.py
# _limit_net_wrench_about_torso, run cfg net_force_limit=120, net_torque_limit=20).
NET_FORCE_LIMIT = 120.0
NET_TORQUE_LIMIT = 20.0

# root_and_wrist_6d reference for STANDING, extracted from the actual training
# motion dataset (data/dataset/limmt_no_foot_gentle_amass_full).
# Layout (motion_tracking.py:638-706): [l_pos_b(3), r_pos_b(3), l_axis_angle_b(3),
# r_axis_angle_b(3)] of left/right_hand_mimic in the FULL root frame.
#
# "still": mean over ~47k frames with |root_lin_vel|<0.05 and mean|joint_vel|<0.1
# (the dataset's natural standing-still stance, arms relaxed at pelvis height).
# Empirically this keeps the sensor's rest false-positive rate at ~0 — with the
# default-pose variant below, the arms hold a low-hands posture whose steady
# tracking error reads as phantom arm contact (det_p~0.68 at rest).
WRIST_REF_STILL = np.array([
    0.1395, 0.2587, -0.0435,     # left_hand_mimic pos_b
    0.1488, -0.2562, 0.0042,     # right_hand_mimic pos_b
    0.5543, 0.8831, -0.0440,     # left hand axis-angle (root-relative)
    -0.5530, 0.7476, -0.0262,    # right hand axis-angle
], dtype=np.float32)
# mean over the 500 frames whose joint_pos is closest to the DEFAULT pose
WRIST_REF_DEFAULT_POSE = np.array([
    0.0780, 0.2356, -0.1707,
    0.1136, -0.2594, -0.1525,
    0.2570, 1.3039, -0.0100,
    -0.2832, 1.1930, -0.0111,
], dtype=np.float32)
# Command root height. 0.79-0.80 matches HierarchicalRootCommand's
# nominal_root_height (action.py:168) and gives the least-crouched equilibrium
# (a lower command makes the policy crouch to reach it, pushing the leg state
# off the sensor's training manifold).
ROOT_HEIGHT_STANDING = 0.80

# push body -> ground-truth region (forcesense/common/regions.py region_of)
def region_of(name):
    side = "left" if name.startswith("left") else ("right" if name.startswith("right") else "")
    if any(k in name for k in ("shoulder", "elbow", "wrist", "hand")):
        return f"{side}_arm"
    if any(k in name for k in ("hip", "knee", "ankle")):
        return f"{side}_leg"
    return "trunk"


def resolve_regex_map(regex_map, names):
    """First-full-match wins; mirrors isaaclab resolve_matching_names_values
    (fullmatch semantics; the maps above are disjoint so order is irrelevant)."""
    out = np.zeros(len(names), dtype=np.float64)
    for i, n in enumerate(names):
        hits = [v for pat, v in regex_map.items() if re.fullmatch(pat, n)]
        if len(hits) != 1:
            raise ValueError(f"joint {n}: {len(hits)} regex matches")
        out[i] = hits[0]
    return out


def default_joint_pos_isaac():
    out = np.zeros(NJ)
    for i, n in enumerate(ISAAC_JOINTS):
        for pat, v in DEFAULT_JOINT_POS.items():
            if re.fullmatch(pat, n):
                out[i] = v
    return out


# ---------------------------- quaternion helpers (wxyz, matches Isaac/MuJoCo)
def quat_apply(q, v):
    w, x, y, z = q
    qv = np.array([x, y, z])
    return v + 2.0 * np.cross(qv, np.cross(qv, v) + w * v)

def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_apply_inverse(q, v):
    return quat_apply(quat_conj(q), v)

def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def yaw_quat(q):
    # ISAAC: active_adaptation.utils.math.yaw_quat — keep only yaw component.
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

def axis_angle_from_quat(q):
    # ISAAC: axis_angle_from_quat (isaaclab math): axis * angle, angle in [0, pi]
    q = q / (np.linalg.norm(q) + 1e-9)
    if q[0] < 0:
        q = -q
    angle = 2.0 * np.arccos(np.clip(q[0], -1.0, 1.0))
    s = np.sqrt(max(1.0 - q[0] * q[0], 1e-12))
    axis = q[1:4] / s
    if angle < 1e-6:
        return np.zeros(3)
    return axis * angle


# --------------------------------------------------------------------------- #
class Sim2Sim:
    """MuJoCo replica of the Isaac low-level env loop (obs/action semantics)."""

    CTRL_HZ = 50
    PHYS_HZ = 200          # ISAAC: isaac_physics_dt=0.005, step_dt=0.02 -> decimation 4
    DECIMATION = 4
    ALPHA = 0.9            # ISAAC: action alpha=[0.9,0.9]; the EMA lerp runs EVERY
                           # physics substep (action.py:138 inside __call__ which is
                           # called per substep) — after 4 substeps the filtered
                           # action is within (1-0.9)^4 = 1e-4 of the raw action.
    ACTION_CLIP = 10.0     # ISAAC: action.py:122 raw_action.clamp(-10, 10)

    def __init__(self, xml, keep_passive=False, root_height_cmd=ROOT_HEIGHT_STANDING,
                 seed=0, wrist_ref_mode="still"):
        self.wrist_ref_mode = wrist_ref_mode
        import mujoco
        self.mujoco = mujoco
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.m.opt.timestep = 1.0 / self.PHYS_HZ   # mirror gentle-humanoid sim2sim
        self.d = mujoco.MjData(self.m)
        self.rng = np.random.default_rng(seed)

        if not keep_passive:
            # ISAAC: Isaac's implicit PD has NO passive joint damping/friction
            # (damping enters only via actuator kd). The MJCF's blanket
            # damping=2 / frictionloss=0.2 default would over-damp the arms
            # (kd~0.9), so we zero passive terms for the hinge dofs.
            self.m.dof_damping[6:] = 0.0
            self.m.dof_frictionloss[6:] = 0.0

        # ---- index maps: Isaac order <-> MJCF ----
        self.qadr = np.zeros(NJ, dtype=int)   # qpos address per Isaac joint
        self.vadr = np.zeros(NJ, dtype=int)   # qvel/dof address per Isaac joint
        self.aid = np.zeros(NJ, dtype=int)    # actuator id per Isaac joint
        for i, n in enumerate(ISAAC_JOINTS):
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
            assert j >= 0, f"joint {n} not in MJCF"
            self.qadr[i] = self.m.jnt_qposadr[j]
            self.vadr[i] = self.m.jnt_dofadr[j]
            a = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            assert a >= 0, f"actuator {n} not in MJCF"
            self.aid[i] = a

        self.default_qj = default_joint_pos_isaac()
        self.kp = resolve_regex_map(KP, ISAAC_JOINTS)
        self.kd = resolve_regex_map(KD, ISAAC_JOINTS)
        self.effort = resolve_regex_map(EFFORT_LIMIT, ISAAC_JOINTS)
        self.scale = resolve_regex_map(ACTION_SCALING, ISAAC_JOINTS)
        # 0.8*ctrlrange clip from gentle-humanoid sim2sim (±160N·m) — the Isaac
        # effort limits (<=139) always bind first, kept for parity anyway.
        self.ctrl_lo = 0.8 * self.m.actuator_ctrlrange[self.aid, 0]
        self.ctrl_hi = 0.8 * self.m.actuator_ctrlrange[self.aid, 1]

        self.torso_bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.root_height_cmd = root_height_cmd

        # ---- Sonic (MuJoCo-order) address maps: g1.xml joints 1..29 in MJCF
        # order are identical by NAME to Sonic's "MuJoCo order" table, so we
        # build qpos/qvel/actuator address arrays in that order for the Sonic
        # driving path (control_step_sonic). Kept separate from the OUR-Isaac
        # maps above so the CEER path is untouched.
        from controllers.sonic.sonic_policy import SONIC_MJ_JOINTS
        self.sonic_qadr = np.zeros(NJ, dtype=int)
        self.sonic_vadr = np.zeros(NJ, dtype=int)
        self.sonic_aid = np.zeros(NJ, dtype=int)
        for i, n in enumerate(SONIC_MJ_JOINTS):
            j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
            assert j >= 0, f"sonic joint {n} not in MJCF"
            self.sonic_qadr[i] = self.m.jnt_qposadr[j]
            self.sonic_vadr[i] = self.m.jnt_dofadr[j]
            a = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            assert a >= 0, f"sonic actuator {n} not in MJCF"
            self.sonic_aid[i] = a
        # map from OUR-Isaac joint index -> MuJoCo(Sonic) joint index (by name),
        # used to express Sonic quantities in OUR-Isaac order for wbc_obs().
        _mj_name_to_pos = {n: k for k, n in enumerate(SONIC_MJ_JOINTS)}
        self.isaac_to_sonicmj = np.array(
            [_mj_name_to_pos[n] for n in ISAAC_JOINTS], dtype=int)

        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self):
        mujoco, m, d = self.mujoco, self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = 0.0
        d.qpos[2] = 0.78          # gentle-humanoid sim2sim settles here after gantry
        d.qpos[3] = 1.0           # identity quat (wxyz)
        d.qpos[self.qadr] = self.default_qj
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        # action pipeline state (ISAAC: action.py reset -> zeros)
        self.action_buf = np.zeros((3, NJ), dtype=np.float32)   # newest first
        self.applied_filt = np.zeros(NJ, dtype=np.float32)      # EMA state
        self.q_des = self.default_qj.copy()                     # joint_pos_target
        self.applied_torque = np.zeros(NJ, dtype=np.float32)

        # obs history buffers, newest first (ISAAC: observations.py reset fills
        # the whole buffer with the current value; joint pos stored ABSOLUTE —
        # the "offset" subtracted in joint_pos_history.compute is the
        # random_joint_offset randomization (±0.01), zero at deploy, NOT the
        # default pose (verified: action.py:86 offset=zeros;
        # randomizations.py:245 is the only writer).
        self.jpos_hist = np.tile(d.qpos[self.qadr], (5, 1)).astype(np.float32)
        self.angvel_hist = np.tile(self._root_ang_vel_b(), (5, 1)).astype(np.float32)
        self.grav_hist = np.tile(self._proj_grav_b(), (5, 1)).astype(np.float32)

        # frozen standing wrist reference for root_and_wrist_6d ("track a
        # constant standing pose"). See WRIST_REF_* comments above.
        self.wrist_ref = {
            "still": WRIST_REF_STILL,
            "default-pose": WRIST_REF_DEFAULT_POSE,
        }.get(self.wrist_ref_mode)
        self.wrist_ref = (self.wrist_ref.copy() if self.wrist_ref is not None
                          else self._wrist_6d_now())   # "fk" fallback
        self.heading_w = np.array([1.0, 0.0, 0.0])  # initial world heading target
        self.step_count = 0

    # root state helpers ---------------------------------------------------
    def root_quat(self):
        return self.d.qpos[3:7].copy()

    def _root_ang_vel_b(self):
        # ISAAC: root_ang_vel_b is angular velocity in the base (pelvis) frame.
        # MuJoCo free-joint qvel[3:6] is ALREADY the body-local angular velocity
        # (MuJoCo convention: free joint = global linear + local angular).
        return self.d.qvel[3:6].copy()

    def _proj_grav_b(self):
        return quat_apply_inverse(self.root_quat(), np.array([0.0, 0.0, -1.0]))

    def _wrist_6d_now(self):
        """root_and_wrist_6d (motion_tracking.py:638-706):
        [l_wrist_pos_b(3), r_wrist_pos_b(3), l_axis_angle_b(3), r_axis_angle_b(3)]
        of the 'hand_mimic' keypoints in the FULL root frame.
        ASSUMPTION: hand_mimic ~ wrist_yaw_link frame + 0.08 m along local +x
        (center of the rubber hand; the mimic body only exists in the Isaac
        USD / motion dataset, not in this MJCF). For a standing robot we freeze
        the reference at the current (default) pose = "track where you are".
        """
        mujoco, d = self.mujoco, self.d
        root_p, root_q = d.qpos[0:3].copy(), self.root_quat()
        pos, aa = [], []
        for side in ("left", "right"):
            b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
            p_w = d.xpos[b].copy()
            q_w = d.xquat[b].copy()
            p_w = p_w + quat_apply(q_w, np.array([0.08, 0.0, 0.0]))
            pos.append(quat_apply_inverse(root_q, p_w - root_p))
            aa.append(axis_angle_from_quat(quat_mul(quat_conj(root_q), q_w)))
        return np.concatenate(pos + aa).astype(np.float32)

    # ---------------------------------------------------------------- step
    def control_step(self, raw_action, ext_force_w=None, ext_body=None):
        """One 50 Hz control step = 4 physics substeps @200 Hz.
        raw_action: [29] policy 'action' output, Isaac order, pre-clip.
        ext_force_w: [3] world-frame force on body `ext_body` (name) or None.
        """
        mujoco, m, d = self.mujoco, self.m, self.d
        raw = np.clip(np.asarray(raw_action, dtype=np.float32).reshape(NJ),
                      -self.ACTION_CLIP, self.ACTION_CLIP)
        # ISAAC: action_buf rolled once per control step (substep 0), newest first
        self.action_buf = np.roll(self.action_buf, 1, axis=0)
        self.action_buf[0] = raw

        jp_sub = np.zeros((2, NJ))
        for sub in range(self.DECIMATION):
            # ISAAC: EMA every substep; delay=0 at deploy (max_delay randomized
            # 0..4 substeps in training — we use the no-delay nominal).
            self.applied_filt += self.ALPHA * (raw - self.applied_filt)
            # ISAAC: q_des = default_joint_pos + offset(=0) + scale * filtered
            self.q_des = self.default_qj + self.scale * self.applied_filt

            q = d.qpos[self.qadr]
            dq = d.qvel[self.vadr]
            tau = self.kp * (self.q_des - q) - self.kd * dq
            tau = np.clip(tau, -self.effort, self.effort)      # ISAAC: effort_limit_sim
            tau = np.clip(tau, self.ctrl_lo, self.ctrl_hi)
            d.ctrl[self.aid] = tau
            self.applied_torque = tau.astype(np.float32)        # = asset.data.applied_torque

            # external force, mirroring Isaac's apply_forces_and_torques_at_position
            # at the LINK FRAME ORIGIN in world coords (base.py:444-460) plus the
            # net-wrench-about-torso limiter (motion_tracking.py:2197-2224).
            d.xfrc_applied[:] = 0.0
            if ext_force_w is not None and np.linalg.norm(ext_force_w) > 1e-9:
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ext_body)
                F = np.asarray(ext_force_w, dtype=np.float64)
                # xfrc acts at the body CoM (xipos); shift to the link origin (xpos)
                r_com = d.xpos[b] - d.xipos[b]
                d.xfrc_applied[b, :3] = F
                d.xfrc_applied[b, 3:] = np.cross(r_com, F)
                # torso wrench limiter: moment of F about torso_link origin
                r = d.xpos[b] - d.xpos[self.torso_bid]
                M = np.cross(r, F)
                Mn = np.linalg.norm(M)
                if Mn > NET_TORQUE_LIMIT:
                    dM = M * (NET_TORQUE_LIMIT / Mn - 1.0)
                    d.xfrc_applied[self.torso_bid, 3:] += dM
                Fn = np.linalg.norm(F)
                if Fn > NET_FORCE_LIMIT:  # never binds at <=40 N, kept for parity
                    d.xfrc_applied[b, :3] *= NET_FORCE_LIMIT / Fn

            mujoco.mj_step(m, d)
            if sub >= 2:
                jp_sub[sub - 2] = d.qpos[self.qadr]

        # ---- per-control-step obs buffer update (ISAAC: obs.update() after all
        # substeps; joint_pos_history stores the MEAN of the last two substeps
        # (observations.py:257-268, joint_pos[:, substep % 2].mean)).
        self.jpos_hist = np.roll(self.jpos_hist, 1, axis=0)
        self.jpos_hist[0] = jp_sub.mean(0)
        self.angvel_hist = np.roll(self.angvel_hist, 1, axis=0)
        self.angvel_hist[0] = self._root_ang_vel_b()
        self.grav_hist = np.roll(self.grav_hist, 1, axis=0)
        self.grav_hist[0] = self._proj_grav_b()
        self.step_count += 1

    # -------------------------------------------------- Sonic driving path
    def setup_sonic(self, sonic_default_mj, kp_mj, kd_mj, scale_mj, effort_mj):
        """Install Sonic's PD gains / default pose / action scale (all MuJoCo
        order) and re-init the robot to Sonic's default standing pose. Call
        once after constructing Sim2Sim, before the Sonic loop."""
        self.sonic_default_mj = np.asarray(sonic_default_mj, dtype=np.float64)
        self.sonic_kp_mj = np.asarray(kp_mj, dtype=np.float64)
        self.sonic_kd_mj = np.asarray(kd_mj, dtype=np.float64)
        self.sonic_scale_mj = np.asarray(scale_mj, dtype=np.float64)
        self.sonic_effort_mj = np.asarray(effort_mj, dtype=np.float64)
        self.reset_sonic()

    # Sonic's default_angles put the pelvis at 0.793 m on *their* g1_29dof.xml
    # (foot spheres exactly on the floor there).  OUR g1.xml has a different
    # ankle/foot geometry: at pelvis 0.793 the foot spheres float 3.6 cm above
    # the floor, so the robot free-falls at spawn.  At pelvis 0.755 the 8 foot
    # spheres (4/foot) firmly contact the floor (verified: ncon=8).  Use that
    # feet-on-floor height as the Sonic spawn default.
    SONIC_STAND_Z = 0.755

    def reset_sonic(self, root_z=None):
        """Reset to Sonic's default standing pose (MuJoCo-order default angles).
        Resets the OUR-Isaac obs history buffers too (they key the wbc_obs).
        root_z defaults to SONIC_STAND_Z (feet-on-floor for OUR g1.xml)."""
        if root_z is None:
            root_z = self.SONIC_STAND_Z
        mujoco, m, d = self.mujoco, self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = 0.0
        d.qpos[2] = root_z
        d.qpos[3] = 1.0
        d.qpos[self.sonic_qadr] = self.sonic_default_mj
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        # Sonic q_des starts at default; torques zero.
        self.sonic_q_des_mj = self.sonic_default_mj.copy()
        self.applied_torque = np.zeros(NJ, dtype=np.float32)   # OUR-Isaac order
        self.action_buf = np.zeros((3, NJ), dtype=np.float32)  # OUR-Isaac order raw action
        self.q_des = d.qpos[self.qadr].astype(np.float32)      # OUR-Isaac order q_des
        # OUR-Isaac obs history (jpos in OUR-Isaac order; used by wbc_obs)
        self.jpos_hist = np.tile(d.qpos[self.qadr], (5, 1)).astype(np.float32)
        self.angvel_hist = np.tile(self._root_ang_vel_b(), (5, 1)).astype(np.float32)
        self.grav_hist = np.tile(self._proj_grav_b(), (5, 1)).astype(np.float32)
        self.step_count = 0

    def control_step_sonic(self, action_mj, ext_force_w=None, ext_body=None):
        """One 50 Hz control step driven by Sonic. action_mj: 29-dim scaled
        action delta ALREADY in MuJoCo(g1.xml) order (SonicPolicy.act output).
        Applies Sonic PD (no EMA, no clip — Sonic uses raw q_target each tick):
            q_target[mj] = default_mj + action_mj * scale_mj
            tau = kp*(q_target-q) - kd*qd,  clipped to Sonic effort limits.
        Updates wbc_obs feature state expressed in OUR-Isaac order."""
        mujoco, m, d = self.mujoco, self.m, self.d
        action_mj = np.asarray(action_mj, dtype=np.float64).reshape(NJ)
        # roll OUR-Isaac-order raw action buffer (action_buf feeds wbc_obs).
        self.action_buf = np.roll(self.action_buf, 1, axis=0)
        self.action_buf[0] = action_mj[self.isaac_to_sonicmj].astype(np.float32)

        q_target_mj = self.sonic_default_mj + action_mj * self.sonic_scale_mj
        self.sonic_q_des_mj = q_target_mj
        jp_sub = np.zeros((2, NJ))
        for sub in range(self.DECIMATION):
            q = d.qpos[self.sonic_qadr]
            dq = d.qvel[self.sonic_vadr]
            tau_mj = self.sonic_kp_mj * (q_target_mj - q) - self.sonic_kd_mj * dq
            tau_mj = np.clip(tau_mj, -self.sonic_effort_mj, self.sonic_effort_mj)
            # honor the MJCF 0.8*ctrlrange clip in MuJoCo order too
            lo = 0.8 * self.m.actuator_ctrlrange[self.sonic_aid, 0]
            hi = 0.8 * self.m.actuator_ctrlrange[self.sonic_aid, 1]
            tau_mj = np.clip(tau_mj, lo, hi)
            d.ctrl[self.sonic_aid] = tau_mj

            d.xfrc_applied[:] = 0.0
            if ext_force_w is not None and np.linalg.norm(ext_force_w) > 1e-9:
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ext_body)
                F = np.asarray(ext_force_w, dtype=np.float64)
                r_com = d.xpos[b] - d.xipos[b]
                d.xfrc_applied[b, :3] = F
                d.xfrc_applied[b, 3:] = np.cross(r_com, F)
                r = d.xpos[b] - d.xpos[self.torso_bid]
                M = np.cross(r, F)
                Mn = np.linalg.norm(M)
                if Mn > NET_TORQUE_LIMIT:
                    dM = M * (NET_TORQUE_LIMIT / Mn - 1.0)
                    d.xfrc_applied[self.torso_bid, 3:] += dM
                Fn = np.linalg.norm(F)
                if Fn > NET_FORCE_LIMIT:
                    d.xfrc_applied[b, :3] *= NET_FORCE_LIMIT / Fn

            mujoco.mj_step(m, d)
            if sub >= 2:
                jp_sub[sub - 2] = d.qpos[self.qadr]   # OUR-Isaac order for wbc_obs

        # applied_torque / q_des expressed in OUR-Isaac order (wbc_obs layout).
        self.applied_torque = tau_mj[self.isaac_to_sonicmj].astype(np.float32)
        self.q_des = q_target_mj[self.isaac_to_sonicmj].astype(np.float32)
        # OUR-Isaac obs history update (mirrors control_step).
        self.jpos_hist = np.roll(self.jpos_hist, 1, axis=0)
        self.jpos_hist[0] = jp_sub.mean(0)
        self.angvel_hist = np.roll(self.angvel_hist, 1, axis=0)
        self.angvel_hist[0] = self._root_ang_vel_b()
        self.grav_hist = np.roll(self.grav_hist, 1, axis=0)
        self.grav_hist[0] = self._proj_grav_b()
        self.step_count += 1

    # ---------------------------------------------------------------- obs
    def command_obs(self):
        """ISAAC: command() = [root_height, target_linvel_b_xy(2),
        target_heading_b_xy(2), force_limit] (motion_tracking.py:2397-2443).
        Standing: height=const, linvel target=0, heading = initial world heading
        expressed in the current yaw frame, force_limit = trained constant 30 N.
        """
        yq = yaw_quat(self.root_quat())
        hb = quat_apply_inverse(yq, self.heading_w)
        return np.array([self.root_height_cmd, 0.0, 0.0, hb[0], hb[1],
                         FORCE_LIMIT_CMD], dtype=np.float32)

    def policy_obs(self):
        """257-dim 'policy' group, cfg insertion order (run cfg observation.policy;
        ObsGroup._compute concatenates funcs in cfg order, base.py:60-65):
        boot(1) | command(6) | root_and_wrist_6d(12) | root_ang_vel[0](3) |
        projected_gravity[0](3) | joint_pos_history[0..4](145) | prev_actions*3(87).
        """
        return np.concatenate([
            # ISAAC: boot_indicator_state = boot_indicator/25, counts 25->0 over
            # the first 0.5 s after reset. The real-robot deploy stack feeds a
            # constant 0.0 (gentle-humanoid src/observation.py BootIndicator);
            # we do the same (robot starts already standing).
            np.array([0.0], dtype=np.float32),
            self.command_obs(),
            self.wrist_ref,
            self.angvel_hist[0],
            self.grav_hist[0],
            self.jpos_hist.reshape(-1),        # steps [0..4], newest first
            self.action_buf.reshape(-1),       # steps 3, newest first, raw actions
        ]).astype(np.float32)

    def ext_torque_residual(self, realizable=False):
        """Policy-INVARIANT external joint-torque estimate (~ J^T F_ext on the 29
        actuated dofs), from the post-step dynamics via the equation of motion:
            resid = M(q) qacc + qfrc_bias - qfrc_actuator - qfrc_passive
                    - qfrc_constraint
        This recovers the external generalized force WITHOUT depending on HOW the
        actuator torque was produced (any controller / policy / PD gain) -> a
        controller-agnostic input channel for the force estimator.

        qfrc_constraint (foot ground-reaction) is the ONLY sim-privileged term
        (M, qfrc_bias, actuator torque, passive are all hardware-computable).
        realizable=True DROPS it: resid_real = J^T F_ext + qfrc_constraint, i.e.
        clean on non-contact links (arms/trunk) but contaminated by the
        ground-reaction on the legs -> quantifies how much the leg result relies
        on a GRF estimate (a real robot would supply it via foot F/T or a
        floating-base momentum observer)."""
        m, d = self.m, self.d
        nv = m.nv
        Mmat = np.zeros((nv, nv))
        self.mujoco.mj_fullM(m, d, Mmat)
        resid = Mmat @ d.qacc + d.qfrc_bias - d.qfrc_actuator - d.qfrc_passive
        if not realizable:
            resid = resid - d.qfrc_constraint
        return resid[self.vadr].astype(np.float32)      # 29 actuated dofs

    def grf_torque(self):
        """Joint-space projection of the ground-reaction (constraint) force on
        the 29 actuated dofs. On hardware this is what a foot F/T sensor (or a
        floating-base MOB) provides; storing it lets us add a realistic GRF-
        estimation error to the residual: resid_noisy = resid_real - (G + noise)."""
        return self.d.qfrc_constraint[self.vadr].astype(np.float32)

    def wbc_obs(self):
        """320-dim 'wbc_input_' group (forcesense/cfg/collect.yaml order):
        applied_torque(29) | joint_pos_history[0..4](145) | applied_action(29) |
        root_ang_vel[0..4](15) | projected_gravity[0..4](15) | prev_actions*3(87).
        applied_action = asset.data.joint_pos_target = the ABSOLUTE q_des
        (observations.py:477-487)."""
        return np.concatenate([
            self.applied_torque,
            self.jpos_hist.reshape(-1),
            self.q_des.astype(np.float32),
            self.angvel_hist.reshape(-1),
            self.grav_hist.reshape(-1),
            self.action_buf.reshape(-1),
        ]).astype(np.float32)


# --------------------------------------------------------------------------- #
class OnnxPolicy:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        meta = json.load(open(os.path.join(os.path.dirname(onnx_path), "policy.json")))
        out_keys = meta["out_keys"]
        n_out = len(self.sess.get_outputs())
        assert n_out == len(out_keys), f"{n_out} outputs vs out_keys {out_keys}"
        # graph output names are node names (and 'loc'/'action' collide as
        # 'linear_9' — the export is deterministic, action == loc, verified
        # numerically); select by POSITION from policy.json out_keys.
        self.action_idx = out_keys.index("action")
        self.in_name = self.sess.get_inputs()[0].name  # 'policy', [1,257], RAW obs
                                                       # (VecNorm folded into the graph)

    def __call__(self, obs):
        outs = self.sess.run(None, {self.in_name: obs.reshape(1, -1).astype(np.float32)})
        return outs[self.action_idx][0]


class ForceSensorV3Runner:
    """v3 dual-head sensor: rolling window of W wbc frames, newest LAST
    (wbc_train_v3.gather_window stacks oldest..newest; play_force_sensor_v3
    keeps buf[-1]=newest). Per-frame stats tiled W times, normalization applied
    AFTER stacking (equivalent to per-frame)."""

    def __init__(self, ckpt_path):
        import torch
        sys.path.insert(0, REPO)
        from forcesense.models import ForceSensorV3
        self.torch = torch
        ck = torch.load(ckpt_path, map_location="cpu")
        assert ck.get("arch") == "v3"
        self.W = ck["window"]
        self.base_dim = ck["base_dim"]
        self.K = ck["num_bodies"]
        self.class_names = list(ck["body_names"]) + ["none"]
        self.det_thresh = ck["det_thresh"]
        self.force_max = ck["force_max"]
        self.model = ForceSensorV3(ck["in_dim"], self.K, ck["hidden"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.xm = ck["x_mean"].numpy().reshape(-1)   # [base_dim]
        self.xs = ck["x_std"].numpy().reshape(-1)
        self.buf = None

    def reset(self):
        self.buf = None

    def __call__(self, frame):
        if self.buf is None:
            self.buf = np.tile(frame, (self.W, 1))
        else:
            self.buf = np.roll(self.buf, -1, axis=0)
            self.buf[-1] = frame
        x = ((self.buf - self.xm) / self.xs).reshape(1, -1)
        with self.torch.no_grad():
            det, loc, dirh, magh = self.model(self.torch.from_numpy(x.astype(np.float32)))
        det_p = float(self.torch.sigmoid(det)[0, 0])
        loc_idx = int(loc[0].argmax())
        dir_b = dirh[0].numpy()
        dir_b = dir_b / (np.linalg.norm(dir_b) + 1e-9)
        mag = max(float(magh[0, 0]), 0.0) * self.force_max   # Newtons
        active = det_p > self.det_thresh
        pred_cls = loc_idx if active else self.K
        return {"det_p": det_p, "active": active, "cls": pred_cls,
                "cls_name": self.class_names[pred_cls], "loc_idx": loc_idx,
                "dir_b": dir_b, "mag": mag}


# --------------------------------------------------------------------------- #
def force_profile(step, mag, ramp_up=20, hold=100, ramp_down=20):
    """Collection-style profile in 50 Hz steps (collect.yaml midpoints:
    ramp_up~20, hold~70-100, ramp_down~20). Returns (magnitude, phase)."""
    if step < ramp_up:
        return mag * step / ramp_up, "ramp_up"
    if step < ramp_up + hold:
        return mag, "hold"
    if step < ramp_up + hold + ramp_down:
        return mag * (1 - (step - ramp_up - hold) / ramp_down), "ramp_down"
    return 0.0, "rest"


def run_stand_test(sim, policy, sensor, seconds, viewer=None, verbose=True):
    """Zero-force standing; returns (ok, per-second heights, false-positive rate)."""
    heights, fp = [], []
    steps = int(seconds * Sim2Sim.CTRL_HZ)
    for k in range(steps):
        obs = sim.policy_obs()
        act = policy(obs)
        sim.control_step(act)
        pred = sensor(sim.wbc_obs()) if sensor is not None else None
        if pred is not None:
            fp.append(float(pred["active"]))
        if viewer is not None:
            viewer.sync()
        h = sim.d.qpos[2]
        if h < 0.5:
            print(f"  FELL at t={k / 50:.2f}s (pelvis z={h:.3f})")
            return False, heights, np.mean(fp) if fp else 0.0
        if (k + 1) % 50 == 0:
            heights.append(h)
            if verbose:
                g = sim.grav_hist[0]
                print(f"  t={(k + 1) // 50:2d}s  z={h:.3f}  grav_b=({g[0]:+.2f},{g[1]:+.2f},{g[2]:+.2f})"
                      f"  |act|={np.abs(sim.action_buf[0]).mean():.2f}"
                      f"  |tau|={np.abs(sim.applied_torque).mean():.1f}"
                      + (f"  det_p={pred['det_p']:.2f}" if pred else ""))
    return True, heights, float(np.mean(fp)) if fp else 0.0


def run_push(sim, policy, sensor, body, mag, direction, viewer=None):
    """Rest 1s -> ramp 0.4s -> hold 2s -> ramp-down 0.4s -> rest 1s.
    Metrics accumulated over the HOLD phase. Returns a result dict."""
    true_region = region_of(body)
    dir_w = np.asarray(direction, dtype=np.float64)
    dir_w /= np.linalg.norm(dir_w)
    ramp_up, hold, ramp_down, rest = 20, 100, 20, 50
    total = rest + ramp_up + hold + ramp_down + rest

    det_hits, cls_hits, mags, coss, det_ps, cls_counts = [], [], [], [], [], {}
    fell = False
    for k in range(total):
        ps = k - rest  # profile step (negative during leading rest)
        f_mag, phase = (0.0, "rest") if ps < 0 else force_profile(ps, mag, ramp_up, hold, ramp_down)
        F_w = dir_w * f_mag
        obs = sim.policy_obs()
        act = policy(obs)
        sim.control_step(act, ext_force_w=F_w if f_mag > 0 else None, ext_body=body)
        pred = sensor(sim.wbc_obs())
        if viewer is not None:
            viewer.sync()
        if sim.d.qpos[2] < 0.5:
            fell = True
            print(f"  !! fell during push ({body}, {mag}N, phase={phase})")
            break
        if phase == "hold":
            # ground truth in BASE frame (labels: net_pull_force_b =
            # quat_apply_inverse(root_quat, force_w), full quat)
            f_b = quat_apply_inverse(sim.root_quat(), F_w)
            f_b_unit = f_b / (np.linalg.norm(f_b) + 1e-9)
            det_hits.append(float(pred["active"]))
            det_ps.append(pred["det_p"])
            cls_hits.append(float(pred["active"] and pred["cls_name"] == true_region))
            cls_counts[pred["cls_name"]] = cls_counts.get(pred["cls_name"], 0) + 1
            if pred["active"]:
                mags.append(pred["mag"])
                coss.append(float(np.dot(pred["dir_b"], f_b_unit)))
    top_cls = max(cls_counts, key=cls_counts.get) if cls_counts else "n/a"
    return {
        "body": body, "region": true_region, "applied_N": mag,
        "dir_w": dir_w.tolist(), "fell": fell,
        "det_rate": float(np.mean(det_hits)) if det_hits else 0.0,
        "mean_det_p": float(np.mean(det_ps)) if det_ps else 0.0,
        "region_acc": float(np.mean(cls_hits)) if cls_hits else 0.0,
        "argmax_region": top_cls,
        "pred_mag_N": float(np.mean(mags)) if mags else 0.0,
        "dir_cos": float(np.mean(coss)) if coss else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx", default=os.path.join(REPO, "controllers/ceer/checkpoints/G1GENTLE-07-07_18-28/policy.onnx"))
    p.add_argument("--sensor", default=os.path.join(REPO, "data/wbc/sweep_v3/force_sensor_v3_best.pt"))
    p.add_argument("--xml", default=os.path.join(REPO, "forcesense/assets/g1/g1.xml"))
    p.add_argument("--push-body", default=None,
                   help="only push this body (default: torso_link, left_wrist_roll_link, left_knee_link)")
    p.add_argument("--push-force", type=float, default=25.0, help="push magnitude in N")
    p.add_argument("--push-dir", default=None, help="world dir 'x,y,z' (default random horizontal)")
    p.add_argument("--duration", type=float, default=10.0, help="stand-test seconds")
    p.add_argument("--viewer", action="store_true", help="mujoco.viewer.launch_passive")
    p.add_argument("--keep-passive-dynamics", action="store_true",
                   help="keep the MJCF's passive joint damping/frictionloss")
    p.add_argument("--root-height-cmd", type=float, default=ROOT_HEIGHT_STANDING)
    p.add_argument("--wrist-ref", choices=["still", "default-pose", "fk"], default="still",
                   help="root_and_wrist_6d source: dataset standing-still reference "
                        "(default; rest FP ~0), dataset near-default-pose reference, "
                        "or a MuJoCo FK approximation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", default=None, help="write a markdown report here")
    args = p.parse_args()

    sim = Sim2Sim(args.xml, keep_passive=args.keep_passive_dynamics,
                  root_height_cmd=args.root_height_cmd, seed=args.seed,
                  wrist_ref_mode=args.wrist_ref)
    policy = OnnxPolicy(args.onnx)
    sensor = ForceSensorV3Runner(args.sensor)
    print(f"[sim2sim] model nq={sim.m.nq} nu={sim.m.nu}; sensor W={sensor.W} "
          f"classes={sensor.class_names} det_thresh={sensor.det_thresh}")

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.m, sim.d)

    # ---- phase 1: stand test -------------------------------------------- #
    print(f"[sim2sim] stand test: {args.duration:.0f}s, zero external force")
    ok, heights, fp_rate = run_stand_test(sim, policy, sensor, args.duration, viewer)
    print(f"[sim2sim] stand test {'PASS' if ok else 'FAIL'}; "
          f"final z={sim.d.qpos[2]:.3f}; sensor false-positive rate={fp_rate:.3f}")
    results = []
    if ok:
        # ---- phase 2: pushes -------------------------------------------- #
        bodies = [args.push_body] if args.push_body else \
                 ["torso_link", "left_wrist_roll_link", "right_elbow_link", "left_knee_link"]
        for body in bodies:
            if args.push_dir:
                d = np.array([float(x) for x in args.push_dir.split(",")])
            else:
                ang = sim.rng.uniform(0, 2 * np.pi)   # horizontal, like net_pull_xy_only
                d = np.array([np.cos(ang), np.sin(ang), 0.0])
            print(f"[sim2sim] push {body} ({region_of(body)}): {args.push_force:.0f}N dir_w={np.round(d, 2)}")
            r = run_push(sim, policy, sensor, body, args.push_force, d, viewer)
            results.append(r)
            print(f"  det_rate={r['det_rate']:.2f}  argmax_region={r['argmax_region']} "
                  f"(acc={r['region_acc']:.2f})  pred|F|={r['pred_mag_N']:.1f}N "
                  f"dir_cos={r['dir_cos']:.2f}  fell={r['fell']}")

        hdr = f"{'body':<24}{'region':<11}{'F(N)':>5} {'det':>5} {'argmax':<11}{'acc':>5} {'|F|pred':>8} {'cos':>6}"
        print("\n" + hdr); print("-" * len(hdr))
        for r in results:
            print(f"{r['body']:<24}{r['region']:<11}{r['applied_N']:>5.0f} "
                  f"{r['det_rate']:>5.2f} {r['argmax_region']:<11}{r['region_acc']:>5.2f} "
                  f"{r['pred_mag_N']:>8.1f} {r['dir_cos']:>6.2f}")

    if args.report:
        with open(args.report, "w") as f:
            f.write("# WBC sim2sim smoke test (MuJoCo, no Isaac)\n\n")
            f.write(f"- policy: `{args.onnx}`\n- sensor: `{args.sensor}`\n- xml: `{args.xml}`\n")
            f.write(f"- stand test ({args.duration:.0f}s): **{'PASS' if ok else 'FAIL'}**, "
                    f"final pelvis z={sim.d.qpos[2]:.3f}, per-second z={np.round(heights, 3).tolist()}\n")
            f.write(f"- sensor false-positive rate while standing: {fp_rate:.3f}\n\n")
            if results:
                f.write("| body | region | F applied (N) | det rate | argmax region | region acc | pred \\|F\\| (N) | dir cos |\n")
                f.write("|---|---|---|---|---|---|---|---|\n")
                for r in results:
                    f.write(f"| {r['body']} | {r['region']} | {r['applied_N']:.0f} | {r['det_rate']:.2f} "
                            f"| {r['argmax_region']} | {r['region_acc']:.2f} | {r['pred_mag_N']:.1f} "
                            f"| {r['dir_cos']:.2f} |\n")
        print(f"[sim2sim] report -> {args.report}")

    if viewer is not None:
        viewer.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
