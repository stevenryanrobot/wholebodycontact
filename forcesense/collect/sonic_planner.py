"""Sonic kinematic-planner reference builder.

The Sonic decoder needs a motion-command TOKEN from the encoder, which needs a
trackable motion REFERENCE (future whole-body poses).  The example reference
dirs shipped in the repo contain only text summaries (no joint_pos/body_quat
arrays), so we generate the reference online with the released kinematic planner
ONNX (planner_sonic.onnx):

    planner(context 4x qpos, mode, dirs, ...) -> mujoco_qpos [1,N,36]  (world frame)

For mode 0 (idle / standing still) fed the robot's current qpos, the planner
emits a dynamically-consistent standing trajectory.  We convert those qpos frames
into the encoder's motion observation (joint positions/velocities in IsaacLab
order + anchor orientation), run the encoder, and get a token with a real basin
of attraction (unlike a hand-built static token).

I/O spec: ~/wbc_external/sonic_groot/docs/source/references/planner_onnx.md
qpos layout [36]: [0:3] root pos (xyz), [3:7] root quat (wxyz), [7:36] 29 joint
angles in MuJoCo body-tree order (== our g1.xml order).
"""
import os
import numpy as np

from .sonic_policy import (
    ISAACLAB_TO_MUJOCO, MUJOCO_TO_ISAACLAB, SONIC_DEFAULT_MJ,
    build_static_encoder_obs, ENC_DIM,
    _OFF_JPOS, _OFF_JVEL, _OFF_ROOTZ10, _OFF_ROOTZ, _OFF_ANCHOR1, _OFF_ANCHOR10,
)

PLANNER_PATH = os.path.expanduser("~/wbc_external/sonic_groot/downloads/planner_sonic.onnx")

MODE_IDLE = 0
MODE_WALK = 2


def _quat_to_rot6d_rel(q_base_wxyz, q_ref_wxyz):
    """First two columns of R = R(base)^T R(ref), flattened row-wise (6D).
    Matches GatherMotionAnchorOrientationMutiFrame orientation_mode 0."""
    from scipy.spatial.transform import Rotation as Rot
    # scipy uses xyzw
    def wxyz2xyzw(q):
        return np.array([q[1], q[2], q[3], q[0]])
    Rb = Rot.from_quat(wxyz2xyzw(q_base_wxyz)).as_matrix()
    Rr = Rot.from_quat(wxyz2xyzw(q_ref_wxyz)).as_matrix()
    Rrel = Rb.T @ Rr
    return np.array([Rrel[0, 0], Rrel[0, 1],
                     Rrel[1, 0], Rrel[1, 1],
                     Rrel[2, 0], Rrel[2, 1]], dtype=np.float32)


class SonicPlanner:
    def __init__(self, planner_path=None, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort
        planner_path = planner_path or PLANNER_PATH
        self.sess = ort.InferenceSession(planner_path, providers=list(providers))
        self.in_names = {i.name: i for i in self.sess.get_inputs()}
        self.out_names = [o.name for o in self.sess.get_outputs()]
        # allowed_pred_num_tokens mask length K
        k_in = self.in_names.get("allowed_pred_num_tokens")
        self.K = int(k_in.shape[-1]) if k_in is not None and isinstance(k_in.shape[-1], int) else 11

    def plan(self, qpos36, mode=MODE_IDLE, target_vel=-1.0,
             move_dir=(1., 0., 0.), face_dir=(1., 0., 0.), height=-1.0, seed=0):
        """qpos36: current MuJoCo qpos [36] (or [4,36] context). Returns valid
        future qpos frames [num_pred_frames, 36] in world frame."""
        qpos36 = np.asarray(qpos36, dtype=np.float32)
        if qpos36.ndim == 1:
            ctx = np.tile(qpos36, (4, 1))[None]           # [1,4,36]
        else:
            ctx = qpos36[None].astype(np.float32)
        feed = {}
        for name in self.in_names:
            if name == "context_mujoco_qpos":
                feed[name] = ctx
            elif name == "target_vel":
                feed[name] = np.array([target_vel], np.float32)
            elif name == "mode":
                feed[name] = np.array([mode], np.int64)
            elif name == "movement_direction":
                feed[name] = np.asarray(move_dir, np.float32)[None]
            elif name == "facing_direction":
                feed[name] = np.asarray(face_dir, np.float32)[None]
            elif name == "height":
                feed[name] = np.array([height], np.float32)
            elif name == "random_seed":
                feed[name] = np.array([seed], np.int64)
            elif name == "has_specific_target":
                feed[name] = np.zeros((1, 1), np.int64)
            elif name == "specific_target_positions":
                feed[name] = np.zeros((1, 4, 3), np.float32)
            elif name == "specific_target_headings":
                feed[name] = np.zeros((1, 4), np.float32)
            elif name == "allowed_pred_num_tokens":
                feed[name] = np.ones((1, self.K), np.int64)
            else:
                # unknown input: fill zeros of its shape
                shp = [d if isinstance(d, int) else 1 for d in self.in_names[name].shape]
                dt = np.int64 if "int" in self.in_names[name].type else np.float32
                feed[name] = np.zeros(shp, dt)
        outs = self.sess.run(None, feed)
        out = dict(zip(self.out_names, outs))
        qpos = out["mujoco_qpos"][0]                      # [N,36]
        n = int(np.asarray(out["num_pred_frames"]).ravel()[0])
        return qpos[:max(n, 1)]


def build_encoder_obs_from_qpos(frames_qpos, base_quat_wxyz, mode_id=0,
                                joint_order="isaaclab"):
    """Build the 1762-dim encoder obs (g1 mode) from planner qpos frames.

    frames_qpos: [F,36] future poses (world frame) from the planner.
    base_quat_wxyz: robot current base quat (for anchor orientation diff).
    joint_order: 'isaaclab' or 'mujoco' for motion_joint_positions layout.

    Uses 10 future frames at step 5 (indices 0,5,10,...,45), clamped to F.
    """
    obs = np.zeros((1, ENC_DIM), dtype=np.float32)
    F = frames_qpos.shape[0]
    idxs = [min(f * 5, F - 1) for f in range(10)]
    m2i = MUJOCO_TO_ISAACLAB
    jpos10, jvel10, anchor10, rootz10 = [], [], [], []
    prev_j = None
    for k, fi in enumerate(idxs):
        qp = frames_qpos[fi]
        j_mj = qp[7:36].astype(np.float32)                # MuJoCo order joints
        j = j_mj[m2i] if joint_order == "isaaclab" else j_mj
        jpos10.append(j)
        # velocity ~ finite diff over the step (50 Hz frames -> dt=0.02*5)
        if prev_j is None:
            jvel10.append(np.zeros(29, np.float32))
        else:
            jvel10.append((j - prev_j) / (0.02 * 5))
        prev_j = j
        rootz10.append(qp[2])
        anchor10.append(_quat_to_rot6d_rel(base_quat_wxyz, qp[3:7]))
    obs[0, _OFF_JPOS:_OFF_JPOS + 290] = np.concatenate(jpos10)
    obs[0, _OFF_JVEL:_OFF_JVEL + 290] = np.concatenate(jvel10)
    obs[0, _OFF_ROOTZ10:_OFF_ROOTZ10 + 10] = np.array(rootz10, np.float32)
    obs[0, _OFF_ROOTZ] = frames_qpos[0, 2]
    obs[0, _OFF_ANCHOR1:_OFF_ANCHOR1 + 6] = anchor10[0]
    obs[0, _OFF_ANCHOR10:_OFF_ANCHOR10 + 60] = np.concatenate(anchor10)
    return obs
