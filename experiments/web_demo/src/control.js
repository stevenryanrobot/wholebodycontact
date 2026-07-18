// PD control in actuated-joint space.
// Pure functions only — no MuJoCo / three.js imports here.

/**
 * Joint-space PD torque: tau = kp*(qDes - qpos) - kd*qvel, clamped to ctrlRange.
 *
 * All arrays are indexed by actuator (length nu). `qpos`/`qvel` must already be
 * gathered into actuator order (i.e. exclude the floating base) by the caller.
 *
 * @param {ArrayLike<number>} qDes      desired joint positions [nu]
 * @param {ArrayLike<number>} qpos      current joint positions  [nu]
 * @param {ArrayLike<number>} qvel      current joint velocities [nu]
 * @param {ArrayLike<number>} kp        proportional gains [nu]
 * @param {ArrayLike<number>} kd        derivative gains   [nu]
 * @param {ArrayLike<number>} ctrlRange flat [lo0, hi0, lo1, hi1, ...] (2*nu)
 * @param {Float64Array}     [out]      optional output buffer [nu]
 * @returns {Float64Array} torques [nu]
 */
export function computePD(qDes, qpos, qvel, kp, kd, ctrlRange, out) {
  const n = qDes.length;
  const tau = out ?? new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let t = kp[i] * (qDes[i] - qpos[i]) - kd[i] * qvel[i];
    const lo = ctrlRange[2 * i];
    const hi = ctrlRange[2 * i + 1];
    if (t < lo) t = lo;
    else if (t > hi) t = hi;
    tau[i] = t;
  }
  return tau;
}

// ---------------------------------------------------------------------------
// Placeholder gains + default standing pose for the Unitree G1 (29 actuators).
// Keyed by substring of the joint name; a later pass will replace these with
// the gains the policy was trained with.
// ---------------------------------------------------------------------------

const GAIN_TABLE = [
  // [name substring, kp, kd]
  ['hip_yaw',        150, 5],
  ['hip_roll',       150, 5],
  ['hip_pitch',      150, 5],
  ['knee',           200, 6],
  // High ankle stiffness is what keeps the unbalanced PD statue upright
  // (verified headlessly: kp<200 tips over within ~2s). Revisit when the
  // balancing policy takes over.
  ['ankle',          300, 10],
  ['waist',          200, 5],
  ['shoulder',        60, 2],
  ['elbow',           60, 2],
  ['wrist',           25, 1],
];

const DEFAULT_POSE_TABLE = [
  // slight crouch: keeps the CoM low and the PD hold robust
  ['hip_pitch',   -0.2],
  ['knee',         0.4],
  ['ankle_pitch', -0.2],
];

// ---------------------------------------------------------------------------
// Placeholder "base assist": a weak world-frame spring-damper wrench on the
// pelvis that stands in for the balancing policy (USE_POLICY=false). Without
// it the PD statue topples from any push > ~20 N. Disable when the policy
// takes over. Returns a 6D wrench [fx fy fz tx ty tz] (world frame).
// ---------------------------------------------------------------------------

export const BASE_ASSIST_GAINS = {
  kxy: 150, dxy: 40,   // horizontal position spring [N/m], damper [N s/m]
  kz: 0, dz: 0,        // vertical handled by the legs
  // NOTE: keep dR small — the wrench acts on the low-inertia pelvis link and
  // explicit rotational damping above ~5 N m s/rad blows up at dt=0.005
  // (verified headlessly).
  kR: 120, dR: 3,      // upright orientation spring [Nm/rad], damper
};

/**
 * @param {ArrayLike<number>} qpos free-joint pose [x y z qw qx qy qz ...]
 * @param {ArrayLike<number>} qvel free-joint vel  [vx vy vz wx wy wz ...]
 * @param {[number, number, number]} anchor world xy(z) the pelvis is pulled to
 * @param {object} g gains (BASE_ASSIST_GAINS)
 * @param {Float64Array} [out] 6-vector
 */
export function computeBaseAssist(qpos, qvel, anchor, g = BASE_ASSIST_GAINS, out) {
  const w = out ?? new Float64Array(6);
  w[0] = -g.kxy * (qpos[0] - anchor[0]) - g.dxy * qvel[0];
  w[1] = -g.kxy * (qpos[1] - anchor[1]) - g.dxy * qvel[1];
  w[2] = -g.kz * (qpos[2] - anchor[2]) - g.dz * qvel[2];
  // Tilt error: body z-axis vs world z-axis. body z in world frame from quat.
  const qw = qpos[3], qx = qpos[4], qy = qpos[5], qz = qpos[6];
  const zx = 2 * (qx * qz + qw * qy);
  const zy = 2 * (qy * qz - qw * qx);
  // axis to rotate bodyZ onto worldZ ~ cross(bodyZ, worldZ) = (zy, -zx, 0)
  // (small-angle). Angular velocity of a MuJoCo free joint is body-local;
  // rotate it to world for damping (approximate with small-tilt: use as-is
  // for x/y which is adequate for an assist).
  w[3] = g.kR * zy - g.dR * qvel[3];
  w[4] = -g.kR * zx - g.dR * qvel[4];
  w[5] = -g.dR * qvel[5]; // damp yaw only
  return w;
}

function lookup(table, name, fallback) {
  for (const row of table) {
    if (name.includes(row[0])) return row.slice(1);
  }
  return fallback;
}

/**
 * Build kp/kd/qDes arrays for a list of actuated joint names (actuator order).
 * @param {string[]} jointNames
 * @returns {{kp: Float64Array, kd: Float64Array, qDes: Float64Array}}
 */
export function defaultGainsAndPose(jointNames) {
  const n = jointNames.length;
  const kp = new Float64Array(n);
  const kd = new Float64Array(n);
  const qDes = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const [p, d] = lookup(GAIN_TABLE, jointNames[i], [50, 2]);
    kp[i] = p;
    kd[i] = d;
    const [q] = lookup(DEFAULT_POSE_TABLE, jointNames[i], [0]);
    qDes[i] = q;
  }
  return { kp, kd, qDes };
}
