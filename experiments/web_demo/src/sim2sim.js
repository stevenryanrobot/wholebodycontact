// Faithful JS port of forcesense/sim2sim.py `Sim2Sim` — the MuJoCo replica
// of the Isaac low-level env loop (obs/action semantics). Validated in Python;
// port, don't re-derive. Works both in the browser and under Node (smoke test).

import {
  ISAAC_JOINTS, NJ, KP, KD, EFFORT_LIMIT, ACTION_SCALING,
  FORCE_LIMIT_CMD, NET_TORQUE_LIMIT, NET_FORCE_LIMIT,
  WRIST_REF_STILL, ROOT_HEIGHT_STANDING,
  resolveRegexMap, defaultJointPosIsaac,
  quatApplyInverse, yawQuat,
} from './isaac.js';

export const CTRL_HZ = 50;
export const PHYS_HZ = 200;      // ISAAC: isaac_physics_dt=0.005, decimation 4
export const DECIMATION = 4;
const ALPHA = 0.9;               // ISAAC: action EMA runs EVERY physics substep
const ACTION_CLIP = 10.0;        // ISAAC: raw_action.clamp(-10, 10)

export class Sim2Sim {
  /**
   * @param mujoco loaded WASM module
   * @param model  MjModel (timestep must already be 0.005)
   * @param data   MjData
   */
  constructor(mujoco, model, data, { rootHeightCmd = ROOT_HEIGHT_STANDING } = {}) {
    this.mujoco = mujoco;
    this.m = model;
    this.d = data;
    this.rootHeightCmd = rootHeightCmd;

    // ISAAC: Isaac's implicit PD has NO passive joint damping/frictionloss;
    // zero the MJCF's blanket damping=2 / frictionloss=0.2 on the hinge dofs.
    const dofDamping = model.dof_damping;
    const dofFriction = model.dof_frictionloss;
    for (let i = 6; i < model.nv; i++) { dofDamping[i] = 0; dofFriction[i] = 0; }

    // ---- index maps: Isaac order <-> MJCF ----
    this.qadr = new Int32Array(NJ);
    this.vadr = new Int32Array(NJ);
    this.aid = new Int32Array(NJ);
    const jntQposAdr = model.jnt_qposadr, jntDofAdr = model.jnt_dofadr;
    for (let i = 0; i < NJ; i++) {
      const n = ISAAC_JOINTS[i];
      const j = model.jnt(n), a = model.actuator(n);
      if (!j || !a) throw new Error(`joint/actuator ${n} not in MJCF`);
      this.qadr[i] = jntQposAdr[j.id];
      this.vadr[i] = jntDofAdr[j.id];
      this.aid[i] = a.id;
      j.delete?.(); a.delete?.();
    }

    this.defaultQj = defaultJointPosIsaac();
    this.kp = resolveRegexMap(KP, ISAAC_JOINTS);
    this.kd = resolveRegexMap(KD, ISAAC_JOINTS);
    this.effort = resolveRegexMap(EFFORT_LIMIT, ISAAC_JOINTS);
    this.scale = resolveRegexMap(ACTION_SCALING, ISAAC_JOINTS);
    // 0.8*ctrlrange clip (gentle-humanoid sim2sim parity; effort limits bind first)
    const cr = model.actuator_ctrlrange;
    this.ctrlLo = new Float64Array(NJ);
    this.ctrlHi = new Float64Array(NJ);
    for (let i = 0; i < NJ; i++) {
      this.ctrlLo[i] = 0.8 * cr[2 * this.aid[i]];
      this.ctrlHi[i] = 0.8 * cr[2 * this.aid[i] + 1];
    }

    const torso = model.body('torso_link');
    this.torsoBid = torso.id;
    torso.delete?.();

    // scratch buffers
    this._jpSub = [new Float64Array(NJ), new Float64Array(NJ)];
    this._v3a = new Float64Array(3);
    this._v3b = new Float64Array(3);
    this._q4 = new Float64Array(4);
    this._policyObs = new Float32Array(257);
    this._wbcObs = new Float32Array(320);

    this.reset();
  }

  // ------------------------------------------------------------------ state
  reset() {
    const { mujoco, m, d } = this;
    mujoco.mj_resetData(m, d);
    const qpos = d.qpos;
    qpos.fill(0);
    qpos[2] = 0.78;      // gentle-humanoid sim2sim settles here after gantry
    qpos[3] = 1.0;       // identity quat (wxyz)
    for (let i = 0; i < NJ; i++) qpos[this.qadr[i]] = this.defaultQj[i];
    d.qvel.fill(0);
    mujoco.mj_forward(m, d);

    // action pipeline state (ISAAC: reset -> zeros)
    this.actionBuf = [new Float32Array(NJ), new Float32Array(NJ), new Float32Array(NJ)]; // newest first
    this.appliedFilt = new Float32Array(NJ);   // EMA state
    this.qDes = Float64Array.from(this.defaultQj); // joint_pos_target
    this.appliedTorque = new Float32Array(NJ);

    // obs history buffers, newest first. Joint pos stored ABSOLUTE (the
    // "offset" in Isaac is the ±0.01 randomization, zero at deploy, NOT the
    // default pose).
    const jp = new Float32Array(NJ);
    for (let i = 0; i < NJ; i++) jp[i] = qpos[this.qadr[i]];
    this.jposHist = [0, 1, 2, 3, 4].map(() => Float32Array.from(jp));
    const av = this.#rootAngVelB();
    this.angvelHist = [0, 1, 2, 3, 4].map(() => Float32Array.from(av));
    const gv = this.#projGravB();
    this.gravHist = [0, 1, 2, 3, 4].map(() => Float32Array.from(gv));

    // frozen standing wrist reference ("still" dataset mean)
    this.wristRef = WRIST_REF_STILL;
    this.headingW = Float64Array.from([1, 0, 0]);
    // base-frame target linear velocity command [vx(forward), vy(left)] m/s.
    // 0,0 = stand still (default). Set via setTargetVel() to make it walk.
    this.targetVelB = this.targetVelB ?? new Float64Array([0, 0]);
    this.stepCount = 0;
  }

  /** Set the base-frame walk command: vx forward (+), vy left (+), m/s. */
  setTargetVel(vx, vy) {
    if (!this.targetVelB) this.targetVelB = new Float64Array(2);
    this.targetVelB[0] = vx;
    this.targetVelB[1] = vy;
  }

  // root state helpers ------------------------------------------------------
  rootQuat() { const q = this.d.qpos; return [q[3], q[4], q[5], q[6]]; }

  #rootAngVelB() {
    // MuJoCo free-joint qvel[3:6] is ALREADY body-local angular velocity.
    const v = this.d.qvel;
    return [v[3], v[4], v[5]];
  }

  #projGravB() {
    return Array.from(quatApplyInverse(this.rootQuat(), [0, 0, -1], this._v3a));
  }

  // ------------------------------------------------------------------ step
  /**
   * One 50 Hz control step = 4 physics substeps @200 Hz.
   * @param rawAction Float32Array[29] policy 'action' output (Isaac order, pre-clip)
   * @param extForceW [fx,fy,fz] world-frame force on body extBodyId, or null
   * @param extBodyId MuJoCo body id the force acts on (at the LINK FRAME ORIGIN)
   */
  controlStep(rawAction, extForceW = null, extBodyId = -1) {
    const { mujoco, m, d } = this;
    // clip + roll action_buf (newest first), once per control step
    const raw = new Float32Array(NJ);
    for (let i = 0; i < NJ; i++) {
      raw[i] = Math.min(ACTION_CLIP, Math.max(-ACTION_CLIP, rawAction[i]));
    }
    this.actionBuf.pop();
    this.actionBuf.unshift(raw);

    const qpos = d.qpos, qvel = d.qvel, ctrl = d.ctrl;
    const filt = this.appliedFilt, qDes = this.qDes, tauOut = this.appliedTorque;

    const hasForce = extForceW &&
      (extForceW[0] ** 2 + extForceW[1] ** 2 + extForceW[2] ** 2) > 1e-18 && extBodyId >= 0;

    for (let sub = 0; sub < DECIMATION; sub++) {
      // ISAAC: EMA every substep (no actuation delay at deploy)
      for (let i = 0; i < NJ; i++) {
        filt[i] += ALPHA * (raw[i] - filt[i]);
        // ISAAC: q_des = default_joint_pos + offset(=0) + scale * filtered
        qDes[i] = this.defaultQj[i] + this.scale[i] * filt[i];
        const q = qpos[this.qadr[i]];
        const dq = qvel[this.vadr[i]];
        let tau = this.kp[i] * (qDes[i] - q) - this.kd[i] * dq;
        const e = this.effort[i];
        if (tau > e) tau = e; else if (tau < -e) tau = -e;      // effort_limit_sim
        if (tau > this.ctrlHi[i]) tau = this.ctrlHi[i];
        else if (tau < this.ctrlLo[i]) tau = this.ctrlLo[i];
        ctrl[this.aid[i]] = tau;
        tauOut[i] = tau;                                        // = applied_torque
      }

      // external force, mirroring Isaac's apply_forces_and_torques_at_position
      // at the LINK FRAME ORIGIN (world coords) + net-wrench-about-torso limiter.
      const xfrc = d.xfrc_applied;
      xfrc.fill(0);
      if (hasForce) {
        const b = extBodyId;
        const xpos = d.xpos, xipos = d.xipos;
        let Fx = extForceW[0], Fy = extForceW[1], Fz = extForceW[2];
        const Fn = Math.hypot(Fx, Fy, Fz);
        if (Fn > NET_FORCE_LIMIT) { const s = NET_FORCE_LIMIT / Fn; Fx *= s; Fy *= s; Fz *= s; }
        // xfrc acts at the body CoM (xipos); shift to the link origin (xpos)
        const rcx = xpos[3 * b] - xipos[3 * b];
        const rcy = xpos[3 * b + 1] - xipos[3 * b + 1];
        const rcz = xpos[3 * b + 2] - xipos[3 * b + 2];
        xfrc[6 * b + 0] = Fx; xfrc[6 * b + 1] = Fy; xfrc[6 * b + 2] = Fz;
        xfrc[6 * b + 3] = rcy * Fz - rcz * Fy;
        xfrc[6 * b + 4] = rcz * Fx - rcx * Fz;
        xfrc[6 * b + 5] = rcx * Fy - rcy * Fx;
        // torso wrench limiter: moment of F about torso_link origin
        const t = this.torsoBid;
        const rx = xpos[3 * b] - xpos[3 * t];
        const ry = xpos[3 * b + 1] - xpos[3 * t + 1];
        const rz = xpos[3 * b + 2] - xpos[3 * t + 2];
        const Mx = ry * Fz - rz * Fy, My = rz * Fx - rx * Fz, Mz = rx * Fy - ry * Fx;
        const Mn = Math.hypot(Mx, My, Mz);
        if (Mn > NET_TORQUE_LIMIT) {
          const s = NET_TORQUE_LIMIT / Mn - 1.0;
          xfrc[6 * t + 3] += Mx * s;
          xfrc[6 * t + 4] += My * s;
          xfrc[6 * t + 5] += Mz * s;
        }
      }

      mujoco.mj_step(m, d);
      if (sub >= 2) {
        const jp = this._jpSub[sub - 2];
        for (let i = 0; i < NJ; i++) jp[i] = qpos[this.qadr[i]];
      }
    }

    // per-control-step obs buffer update: joint_pos_history stores the MEAN of
    // the last two substeps; histories are newest first.
    const jpNew = this.jposHist.pop();
    for (let i = 0; i < NJ; i++) jpNew[i] = 0.5 * (this._jpSub[0][i] + this._jpSub[1][i]);
    this.jposHist.unshift(jpNew);

    const avNew = this.angvelHist.pop();
    const av = this.#rootAngVelB();
    avNew[0] = av[0]; avNew[1] = av[1]; avNew[2] = av[2];
    this.angvelHist.unshift(avNew);

    const gvNew = this.gravHist.pop();
    const gv = this.#projGravB();
    gvNew[0] = gv[0]; gvNew[1] = gv[1]; gvNew[2] = gv[2];
    this.gravHist.unshift(gvNew);

    this.stepCount += 1;
  }

  // ------------------------------------------------------------------ obs
  /** command() = [root_height, target_linvel_b_xy(2), heading_b_xy(2), force_limit] */
  #commandObs(out, o) {
    const yq = yawQuat(this.rootQuat(), this._q4);
    const hb = quatApplyInverse(yq, this.headingW, this._v3b);
    out[o] = this.rootHeightCmd;
    out[o + 1] = this.targetVelB ? this.targetVelB[0] : 0;
    out[o + 2] = this.targetVelB ? this.targetVelB[1] : 0;
    out[o + 3] = hb[0]; out[o + 4] = hb[1];
    out[o + 5] = FORCE_LIMIT_CMD;
  }

  /**
   * 257-dim 'policy' obs, cfg insertion order:
   * boot(1) | command(6) | root_and_wrist_6d(12) | root_ang_vel[0](3) |
   * projected_gravity[0](3) | joint_pos_history[0..4](145) | prev_actions*3(87).
   */
  policyObs() {
    const out = this._policyObs;
    let o = 0;
    out[o++] = 0.0;                       // boot indicator (deploy: constant 0)
    this.#commandObs(out, o); o += 6;
    out.set(this.wristRef, o); o += 12;
    out.set(this.angvelHist[0], o); o += 3;
    out.set(this.gravHist[0], o); o += 3;
    for (const h of this.jposHist) { out.set(h, o); o += NJ; }   // newest first
    for (const a of this.actionBuf) { out.set(a, o); o += NJ; }  // newest first
    if (o !== 257) throw new Error(`policy obs ${o} != 257`);
    return out;
  }

  /**
   * 320-dim 'wbc_input_' frame:
   * applied_torque(29) | joint_pos_history[0..4](145) | applied_action=q_des(29) |
   * root_ang_vel[0..4](15) | projected_gravity[0..4](15) | prev_actions*3(87).
   */
  wbcObs() {
    const out = this._wbcObs;
    let o = 0;
    out.set(this.appliedTorque, o); o += NJ;
    for (const h of this.jposHist) { out.set(h, o); o += NJ; }
    for (let i = 0; i < NJ; i++) out[o + i] = this.qDes[i];      // ABSOLUTE q_des
    o += NJ;
    for (const h of this.angvelHist) { out.set(h, o); o += 3; }
    for (const h of this.gravHist) { out.set(h, o); o += 3; }
    for (const a of this.actionBuf) { out.set(a, o); o += NJ; }
    if (o !== 320) throw new Error(`wbc obs ${o} != 320`);
    return out;
  }
}
