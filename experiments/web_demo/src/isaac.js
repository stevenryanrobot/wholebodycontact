// Isaac-semantics constants + math, ported verbatim from the validated
// prototype forcesense/sim2sim.py (see the "ISAAC:" comments there for the
// provenance of every value). Pure JS — no MuJoCo / three.js imports.

// ISAAC: articulation (PhysX BFS) joint order — the order of ALL joint-space
// observations and of the policy action output.
export const ISAAC_JOINTS = [
  'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
  'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
  'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
  'left_knee_joint', 'right_knee_joint',
  'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',
  'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
  'left_shoulder_roll_joint', 'right_shoulder_roll_joint',
  'left_ankle_roll_joint', 'right_ankle_roll_joint',
  'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
  'left_elbow_joint', 'right_elbow_joint',
  'left_wrist_roll_joint', 'right_wrist_roll_joint',
  'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
  'left_wrist_yaw_joint', 'right_wrist_yaw_joint',
];
export const NJ = ISAAC_JOINTS.length; // 29

// ISAAC: default joint pos = G1_CFG.init_state.joint_pos.
// q_des = default + action_scale * filtered_action.
export const DEFAULT_JOINT_POS = {
  '.*_hip_pitch_joint': -0.28,
  '.*_knee_joint': 0.5,
  '.*_ankle_pitch_joint': -0.23,
  '.*_elbow_joint': 0.87,
  'left_shoulder_roll_joint': 0.16,
  'left_shoulder_pitch_joint': 0.35,
  'right_shoulder_roll_joint': -0.16,
  'right_shoulder_pitch_joint': 0.35,
};

// ISAAC: action scaling regex map (== policy.json action_scaling).
export const ACTION_SCALING = {
  '.*elbow_joint': 1.0, '.*shoulder.*': 1.0, '.*wrist.*': 1.0,
  '.*hip_roll.*': 0.25, '.*hip_yaw.*': 0.25, '.*hip_pitch.*': 0.5,
  '.*knee.*': 0.5, '.*waist.*': 0.25, '.*ankle.*': 0.5,
};

// ISAAC: PD gains / effort limits from G1_CFG actuators (base gains — the
// trained run has no gain override; see prototype comment).
export const KP = {
  '.*_hip_yaw_joint': 40.17923847137318, '.*_hip_pitch_joint': 40.17923847137318,
  '.*_hip_roll_joint': 99.09842777666113, '.*_knee_joint': 99.09842777666113,
  'waist_yaw_joint': 40.17923847137318,
  'waist_roll_joint': 28.50124619574858, 'waist_pitch_joint': 28.50124619574858,
  '.*_ankle_pitch_joint': 28.50124619574858, '.*_ankle_roll_joint': 28.50124619574858,
  '.*_shoulder_pitch_joint': 14.25062309787429, '.*_shoulder_roll_joint': 14.25062309787429,
  '.*_shoulder_yaw_joint': 14.25062309787429, '.*_elbow_joint': 14.25062309787429,
  '.*_wrist_roll_joint': 14.25062309787429,
  '.*_wrist_pitch_joint': 16.77832748089279, '.*_wrist_yaw_joint': 16.77832748089279,
};
export const KD = {
  '.*_hip_yaw_joint': 2.5578897650279457, '.*_hip_pitch_joint': 2.5578897650279457,
  '.*_hip_roll_joint': 6.3088018534966395, '.*_knee_joint': 6.3088018534966395,
  'waist_yaw_joint': 2.5578897650279457,
  'waist_roll_joint': 1.814445686584846, 'waist_pitch_joint': 1.814445686584846,
  '.*_ankle_pitch_joint': 1.814445686584846, '.*_ankle_roll_joint': 1.814445686584846,
  '.*_shoulder_pitch_joint': 0.907222843292423, '.*_shoulder_roll_joint': 0.907222843292423,
  '.*_shoulder_yaw_joint': 0.907222843292423, '.*_elbow_joint': 0.907222843292423,
  '.*_wrist_roll_joint': 0.907222843292423,
  '.*_wrist_pitch_joint': 1.06814150219, '.*_wrist_yaw_joint': 1.06814150219,
};
export const EFFORT_LIMIT = {
  '.*_hip_yaw_joint': 88.0, '.*_hip_roll_joint': 139.0, '.*_hip_pitch_joint': 88.0,
  '.*_knee_joint': 139.0, 'waist_yaw_joint': 88.0, 'waist_roll_joint': 50.0,
  'waist_pitch_joint': 50.0, '.*_ankle_pitch_joint': 50.0, '.*_ankle_roll_joint': 50.0,
  '.*_shoulder_pitch_joint': 25.0, '.*_shoulder_roll_joint': 25.0,
  '.*_shoulder_yaw_joint': 25.0, '.*_elbow_joint': 25.0, '.*_wrist_roll_joint': 25.0,
  '.*_wrist_pitch_joint': 5.0, '.*_wrist_yaw_joint': 5.0,
};

// ISAAC: command force_limit was constant 30.0 N for this run.
export const FORCE_LIMIT_CMD = 30.0;
// ISAAC: net-wrench limiter about the torso.
export const NET_FORCE_LIMIT = 120.0;
export const NET_TORQUE_LIMIT = 20.0;

// root_and_wrist_6d reference for STANDING ("still" dataset mean — keeps the
// sensor's rest false-positive rate ~0). Layout: [l_pos_b(3), r_pos_b(3),
// l_axis_angle_b(3), r_axis_angle_b(3)] in the FULL root frame.
export const WRIST_REF_STILL = Float32Array.from([
  0.1395, 0.2587, -0.0435,
  0.1488, -0.2562, 0.0042,
  0.5543, 0.8831, -0.0440,
  -0.5530, 0.7476, -0.0262,
]);

export const ROOT_HEIGHT_STANDING = 0.80;

/** First-full-match wins; mirrors isaaclab resolve_matching_names_values. */
export function resolveRegexMap(regexMap, names) {
  const out = new Float64Array(names.length);
  const entries = Object.entries(regexMap).map(([p, v]) => [new RegExp(`^(?:${p})$`), v]);
  names.forEach((n, i) => {
    const hits = entries.filter(([re]) => re.test(n)).map(([, v]) => v);
    if (hits.length !== 1) throw new Error(`joint ${n}: ${hits.length} regex matches`);
    out[i] = hits[0];
  });
  return out;
}

export function defaultJointPosIsaac() {
  const out = new Float64Array(NJ);
  const entries = Object.entries(DEFAULT_JOINT_POS).map(([p, v]) => [new RegExp(`^(?:${p})$`), v]);
  ISAAC_JOINTS.forEach((n, i) => {
    for (const [re, v] of entries) if (re.test(n)) out[i] = v;
  });
  return out;
}

// ------------------------- quaternion helpers (wxyz, matches Isaac/MuJoCo) --
export function quatApply(q, v, out = new Float64Array(3)) {
  // v + 2*qv x (qv x v + w*v)
  const w = q[0], x = q[1], y = q[2], z = q[3];
  const ix = y * v[2] - z * v[1] + w * v[0];
  const iy = z * v[0] - x * v[2] + w * v[1];
  const iz = x * v[1] - y * v[0] + w * v[2];
  out[0] = v[0] + 2 * (y * iz - z * iy);
  out[1] = v[1] + 2 * (z * ix - x * iz);
  out[2] = v[2] + 2 * (x * iy - y * ix);
  return out;
}

export function quatConj(q, out = new Float64Array(4)) {
  out[0] = q[0]; out[1] = -q[1]; out[2] = -q[2]; out[3] = -q[3];
  return out;
}

const _qc = new Float64Array(4);
export function quatApplyInverse(q, v, out = new Float64Array(3)) {
  return quatApply(quatConj(q, _qc), v, out);
}

export function quatMul(a, b, out = new Float64Array(4)) {
  const [w1, x1, y1, z1] = a, [w2, x2, y2, z2] = b;
  out[0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
  out[1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
  out[2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
  out[3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
  return out;
}

/** ISAAC: yaw_quat — keep only the yaw component. */
export function yawQuat(q, out = new Float64Array(4)) {
  const [w, x, y, z] = q;
  const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
  out[0] = Math.cos(yaw / 2); out[1] = 0; out[2] = 0; out[3] = Math.sin(yaw / 2);
  return out;
}
