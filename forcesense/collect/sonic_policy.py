"""Sonic (GR00T-WholeBodyControl / GEAR-SONIC) policy adapter for OUR G1 MuJoCo sim.

Drives OUR forcesense/assets/g1/g1.xml through the external Sonic encoder-decoder
ONNX policy, remapping joint order by NAME between Sonic's IsaacLab order and OUR
MJCF (== Sonic "MuJoCo") order, applying Sonic's own PD gains / default angles /
action scale.

VERIFIED against:
  ~/wbc_external/sonic_groot/gear_sonic_deploy/.../policy_parameters.hpp
  ~/wbc_external/sonic_groot/gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp
    - action apply (l.3123-3125):  q_target[mj] = default_angles[mj]
                                    + floatarr[isaaclab_to_mujoco[mj]] * g1_action_scale[mj]
      last_action[i] = floatarr[i]                      (IsaacLab order)
    - decoder obs joint entry (l.2830): body_q[i_isaaclab]
                                    = q[mujoco_to_isaaclab[i]] - default_angles[mujoco_to_isaaclab[i]]
    - decoder proprio history: 10 frames, OLDEST->NEWEST (newest_first=false),
      layout [ang_vel 10x3 | (q-default) 10x29 | dq 10x29 | last_action 10x29 |
              gravity 10x3], all joint blocks in IsaacLab order. No obs norm.
    - encoder_mode_4 = [mode_id, 0,0,0]; g1 mode_id = 0  -> all zeros.

OUR MJCF joint order (g1.xml joints 1..29) is IDENTICAL by name to Sonic's
"MuJoCo order" table, so `isaaclab_to_mujoco`/`mujoco_to_isaaclab` map directly
onto our qpos/qvel address arrays.  Control 50 Hz, physics 200 Hz (decimation 4).
"""
import os
import numpy as np

# --------------------------------------------------------------------------- #
# Sonic constants (policy_parameters.hpp), all MuJoCo-order arrays unless noted.
# --------------------------------------------------------------------------- #
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28], dtype=int)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28], dtype=int)

# Sonic MuJoCo joint order == OUR g1.xml joint order (verified by name).
SONIC_MJ_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# --- motor constants -> gains/scale (exact reproduction of the .hpp math) ---
_ARM_5020, _ARM_7520_14, _ARM_7520_22, _ARM_4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
_W = 10 * 2.0 * np.pi
_STF_5020 = _ARM_5020 * _W * _W
_STF_7520_14 = _ARM_7520_14 * _W * _W
_STF_7520_22 = _ARM_7520_22 * _W * _W
_STF_4010 = _ARM_4010 * _W * _W
_DMP_5020 = 2.0 * 2 * _ARM_5020 * _W
_DMP_7520_14 = 2.0 * 2 * _ARM_7520_14 * _W
_DMP_7520_22 = 2.0 * 2 * _ARM_7520_22 * _W
_DMP_4010 = 2.0 * 2 * _ARM_4010 * _W
_EFF_5020, _EFF_7520_14, _EFF_7520_22, _EFF_4010 = 25.0, 88.0, 139.0, 5.0

# Per-joint motor type in MuJoCo order (from the .hpp comments, joint-by-joint).
# tuple: (stiffness, damping, effort, kp_mult, kd_mult)  where kp = kp_mult*stiffness
_MJ = [
    (_STF_7520_22, _DMP_7520_22, _EFF_7520_22, 1, 1),  # left_hip_pitch
    (_STF_7520_22, _DMP_7520_22, _EFF_7520_22, 1, 1),  # left_hip_roll
    (_STF_7520_14, _DMP_7520_14, _EFF_7520_14, 1, 1),  # left_hip_yaw
    (_STF_7520_22, _DMP_7520_22, _EFF_7520_22, 1, 1),  # left_knee
    (_STF_5020, _DMP_5020, _EFF_5020, 2, 2),           # left_ankle_pitch
    (_STF_5020, _DMP_5020, _EFF_5020, 2, 2),           # left_ankle_roll
    (_STF_7520_22, _DMP_7520_22, _EFF_7520_22, 1, 1),  # right_hip_pitch
    (_STF_7520_22, _DMP_7520_22, _EFF_7520_22, 1, 1),  # right_hip_roll
    (_STF_7520_14, _DMP_7520_14, _EFF_7520_14, 1, 1),  # right_hip_yaw
    (_STF_7520_22, _DMP_7520_22, _EFF_7520_22, 1, 1),  # right_knee
    (_STF_5020, _DMP_5020, _EFF_5020, 2, 2),           # right_ankle_pitch
    (_STF_5020, _DMP_5020, _EFF_5020, 2, 2),           # right_ankle_roll
    (_STF_7520_14, _DMP_7520_14, _EFF_7520_14, 1, 1),  # waist_yaw
    (_STF_5020, _DMP_5020, _EFF_5020, 2, 2),           # waist_roll
    (_STF_5020, _DMP_5020, _EFF_5020, 2, 2),           # waist_pitch
    (_STF_5020, _DMP_5020, _EFF_5020, 1, 1),           # left_shoulder_pitch
    (_STF_5020, _DMP_5020, _EFF_5020, 1, 1),           # left_shoulder_roll
    (_STF_5020, _DMP_5020, _EFF_5020, 1, 1),           # left_shoulder_yaw
    (_STF_5020, _DMP_5020, _EFF_5020, 1, 1),           # left_elbow
    (_STF_5020, _DMP_5020, _EFF_5020, 1, 1),           # left_wrist_roll
    (_STF_4010, _DMP_4010, _EFF_4010, 1, 1),           # left_wrist_pitch
    (_STF_4010, _DMP_4010, _EFF_4010, 1, 1),           # left_wrist_yaw
    (_STF_5020, _DMP_5020, _EFF_5020, 1, 1),           # right_shoulder_pitch
    (_STF_5020, _DMP_5020, _EFF_5020, 1, 1),           # right_shoulder_roll
    (_STF_5020, _DMP_5020, _EFF_5020, 1, 1),           # right_shoulder_yaw
    (_STF_5020, _DMP_5020, _EFF_5020, 1, 1),           # right_elbow
    (_STF_5020, _DMP_5020, _EFF_5020, 1, 1),           # right_wrist_roll
    (_STF_4010, _DMP_4010, _EFF_4010, 1, 1),           # right_wrist_pitch
    (_STF_4010, _DMP_4010, _EFF_4010, 1, 1),           # right_wrist_yaw
]
SONIC_KP_MJ = np.array([s * km for (s, d, e, km, kd) in _MJ])                 # MuJoCo order
SONIC_KD_MJ = np.array([d * kd for (s, d, e, km, kd) in _MJ])
SONIC_EFFORT_MJ = np.array([e for (s, d, e, km, kd) in _MJ])
SONIC_ACTION_SCALE_MJ = np.array([0.25 * e / s for (s, d, e, km, kd) in _MJ])  # MuJoCo order

# Default standing angles, MuJoCo order (policy_parameters.hpp default_angles).
SONIC_DEFAULT_MJ = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,          # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,          # right leg
    0.0, 0.0, 0.0,                                  # waist yaw/roll/pitch
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,             # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,            # right arm
])

# Encoder obs layout (mode g1 = 0), offsets from minimal_inference.py / obs config.
ENC_DIM = 1762
ENC_MODE = 0
_OFF_MODE = 0                    # encoder_mode_4 (4)  -> [0,0,0,0]
_OFF_JPOS = 4                    # motion_joint_positions_10frame_step5 (290) MuJoCo order
_OFF_JVEL = 294                  # motion_joint_velocities_10frame_step5 (290)
_OFF_ROOTZ10 = 584               # motion_root_z_position_10frame_step5 (10)
_OFF_ROOTZ = 594                 # motion_root_z_position (1)
_OFF_ANCHOR1 = 595               # motion_anchor_orientation (6)
_OFF_ANCHOR10 = 601              # motion_anchor_orientation_10frame_step5 (60)
DEC_DIM = 994


def build_static_encoder_obs(default_mj=SONIC_DEFAULT_MJ, root_z=0.793):
    """Encoder obs (1762) for 'hold the default standing pose' (mode g1).

    The encoder motion_joint_positions are the raw reference joint angles in
    MuJoCo order (info.txt joint_pos is MuJoCo order).  For a static stand every
    future frame equals the default pose, velocities zero, and the anchor
    orientation diff is identity -> 6D rot = [1,0, 0,1, 0,0] (row-wise first two
    cols of I).  root_z = default pelvis height (~0.793 from squat info sample).
    """
    obs = np.zeros((1, ENC_DIM), dtype=np.float32)
    # encoder_mode_4 already zeros (mode id 0).
    jpos10 = np.tile(default_mj.astype(np.float32), 10)      # 10x29 oldest->newest, all default
    obs[0, _OFF_JPOS:_OFF_JPOS + 290] = jpos10
    # velocities zero (already).
    obs[0, _OFF_ROOTZ10:_OFF_ROOTZ10 + 10] = root_z
    obs[0, _OFF_ROOTZ] = root_z
    identity6 = np.array([1., 0., 0., 1., 0., 0.], dtype=np.float32)
    obs[0, _OFF_ANCHOR1:_OFF_ANCHOR1 + 6] = identity6
    obs[0, _OFF_ANCHOR10:_OFF_ANCHOR10 + 60] = np.tile(identity6, 10)
    return obs


class SonicPolicy:
    """Encoder-decoder Sonic policy driving OUR G1 via a shared MuJoCo history.

    Usage mirrors OUR OnnxPolicy but the decoder needs proprio history, so this
    class owns a 10-frame ring of (base_ang_vel, q-default, dq, last_action,
    gravity) in IsaacLab order and is fed by the Sim2Sim loop each control step.

    Call order per control step:
        act_mj = policy.act(sim)          # remapped 29-dim action, OUR MJCF order
        sim.control_step_sonic(act_mj, ...)   (Sim2Sim applies Sonic PD)
        policy.record(sim)                # push newest proprio frame post-step
    """

    HIST = 10

    def __init__(self, enc_path=None, dec_path=None, root_z=0.793,
                 default_mj=None, token_path=None, use_planner=False,
                 planner_mode=0, planner_qpos=None,
                 providers=("CPUExecutionProvider",)):
        import onnxruntime as ort
        base = os.path.expanduser("~/wbc_external/sonic_groot/downloads")
        enc_path = enc_path or os.path.join(base, "model_encoder.onnx")
        dec_path = dec_path or os.path.join(base, "model_decoder.onnx")
        self.enc = ort.InferenceSession(enc_path, providers=list(providers))
        self.dec = ort.InferenceSession(dec_path, providers=list(providers))
        self.default_mj = SONIC_DEFAULT_MJ if default_mj is None else np.asarray(default_mj)
        self.i2m = ISAACLAB_TO_MUJOCO
        self.m2i = MUJOCO_TO_ISAACLAB
        # Motion-command token. The example reference dirs shipped with the repo
        # contain only text summaries (no joint_pos.csv/body_quat.csv arrays) and
        # there is no planner ONNX in downloads/, so a real tracking reference
        # cannot be encoded. Instead we use a "hold default standing pose" token:
        # optimized (via token_path) so the decoder outputs ~zero action at the
        # default+upright+still state -> the policy holds the Sonic default stance.
        # This is a legitimate use of the token interface (the deploy stack allows
        # external tokens via ZMQ: token_state_data_ set directly).
        if use_planner:
            # Real trackable reference via the released kinematic planner. Mode 0
            # = idle stand; the reference is synthesized from the current qpos
            # (defaults to Sonic default standing pose at root_z). IsaacLab joint
            # order for motion_joint_positions (verified: gives ~0 action at
            # default, vs 6.1 for MuJoCo order).
            from .sonic_planner import SonicPlanner, build_encoder_obs_from_qpos
            self._planner = SonicPlanner(providers=providers)
            self._build_enc_obs = build_encoder_obs_from_qpos
            if planner_qpos is None:
                planner_qpos = np.zeros(36, np.float32)
                planner_qpos[2] = root_z
                planner_qpos[3] = 1.0
                planner_qpos[7:36] = self.default_mj.astype(np.float32)
            self._planner_mode = planner_mode
            frames = self._planner.plan(planner_qpos, mode=planner_mode)
            base_quat = np.asarray(planner_qpos[3:7], np.float32)
            enc_obs = self._build_enc_obs(frames, base_quat, joint_order="isaaclab")
            self.token = self.enc.run(None, {"obs_dict": enc_obs})[0].ravel().astype(np.float32)
        elif token_path and os.path.exists(token_path):
            self.token = np.load(token_path).ravel().astype(np.float32)
        else:
            enc_obs = build_static_encoder_obs(self.default_mj, root_z=root_z)
            self.token = self.enc.run(None, {"obs_dict": enc_obs})[0].ravel().astype(np.float32)
        # proprio history rings, oldest->newest, IsaacLab order.
        self.h_angvel = np.zeros((self.HIST, 3), np.float32)
        self.h_qrel = np.zeros((self.HIST, 29), np.float32)
        self.h_dq = np.zeros((self.HIST, 29), np.float32)
        self.h_lastact = np.zeros((self.HIST, 29), np.float32)
        self.h_grav = np.zeros((self.HIST, 3), np.float32)
        self.last_action_il = np.zeros(29, np.float32)   # IsaacLab order raw action
        self._primed = False

    def refresh_token(self, sim):
        """Re-plan the reference from the robot's live MuJoCo qpos and re-encode
        the token (the planner/encoder run at a low rate in the real stack).
        No-op unless constructed with use_planner=True."""
        if getattr(self, "_planner", None) is None:
            return
        qpos = sim.d.qpos[:36].astype(np.float32).copy()
        frames = self._planner.plan(qpos, mode=self._planner_mode)
        base_quat = qpos[3:7]
        enc_obs = self._build_enc_obs(frames, base_quat, joint_order="isaaclab")
        self.token = self.enc.run(None, {"obs_dict": enc_obs})[0].ravel().astype(np.float32)

    # -- proprio frame from OUR Sim2Sim state, in IsaacLab order ---------------
    def _frame(self, sim):
        q_mj = sim.d.qpos[sim.sonic_qadr].astype(np.float32)     # MuJoCo order
        dq_mj = sim.d.qvel[sim.sonic_vadr].astype(np.float32)
        qrel_mj = q_mj - self.default_mj.astype(np.float32)
        qrel_il = qrel_mj[self.i2m]      # -> IsaacLab order
        dq_il = dq_mj[self.i2m]
        angvel = sim._root_ang_vel_b().astype(np.float32)        # base frame rad/s
        grav = sim._proj_grav_b().astype(np.float32)             # projected gravity, base frame
        return angvel, qrel_il, dq_il, grav

    def prime(self, sim):
        """Fill all 10 history frames with the current state (call after reset)."""
        angvel, qrel_il, dq_il, grav = self._frame(sim)
        self.h_angvel[:] = angvel
        self.h_qrel[:] = qrel_il
        self.h_dq[:] = dq_il
        self.h_lastact[:] = 0.0
        self.h_grav[:] = grav
        self.last_action_il[:] = 0.0
        self._primed = True

    def record(self, sim):
        """Push the newest proprio frame (call AFTER control_step)."""
        angvel, qrel_il, dq_il, grav = self._frame(sim)
        self.h_angvel = np.roll(self.h_angvel, -1, axis=0); self.h_angvel[-1] = angvel
        self.h_qrel = np.roll(self.h_qrel, -1, axis=0);     self.h_qrel[-1] = qrel_il
        self.h_dq = np.roll(self.h_dq, -1, axis=0);         self.h_dq[-1] = dq_il
        self.h_grav = np.roll(self.h_grav, -1, axis=0);     self.h_grav[-1] = grav
        self.h_lastact = np.roll(self.h_lastact, -1, axis=0); self.h_lastact[-1] = self.last_action_il

    def _decoder_obs(self):
        obs = np.zeros((1, DEC_DIM), np.float32)
        obs[0, 0:64] = self.token
        obs[0, 64:94] = self.h_angvel.reshape(-1)        # 10x3
        obs[0, 94:384] = self.h_qrel.reshape(-1)         # 10x29
        obs[0, 384:674] = self.h_dq.reshape(-1)          # 10x29
        obs[0, 674:964] = self.h_lastact.reshape(-1)     # 10x29
        obs[0, 964:994] = self.h_grav.reshape(-1)        # 10x3
        return obs

    def act(self, sim):
        """Run the decoder; return action remapped to OUR MJCF (MuJoCo) order.

        Also stores the raw IsaacLab-order action for the next history push and
        for the WBC feature vector (remapped by the caller)."""
        if not self._primed:
            self.prime(sim)
        action_il = self.dec.run(None, {"obs_dict": self._decoder_obs()})[0].ravel().astype(np.float32)
        self.last_action_il = action_il
        action_mj = action_il[self.i2m]                  # IsaacLab -> MuJoCo order
        return action_mj

    def last_action_mujoco(self):
        return self.last_action_il[self.i2m]
