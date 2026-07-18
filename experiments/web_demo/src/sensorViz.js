// Prediction visualization: green arrow + region heatmap + rolling force curve.
//
// Normally fed the REAL v3 sensor-head outputs (det_logit/loc_logits/dir/mag);
// the dir head is in the BASE (pelvis) frame and is rotated to world with the
// root quaternion. fakeSensorFromApplied() remains as a stub fallback
// (USE_FAKE_SENSOR flag / before the first control tick).

import * as THREE from 'three';
import { REGIONS, regionOfBody } from './robotRenderer.js';
import { quatApply } from './isaac.js';

export function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

export function softmax(logits) {
  let mx = -Infinity;
  for (const v of logits) if (v > mx) mx = v;
  let sum = 0;
  const out = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) { out[i] = Math.exp(logits[i] - mx); sum += out[i]; }
  for (let i = 0; i < logits.length; i++) out[i] /= sum;
  return out;
}

/**
 * Fabricate sensor-head outputs from the applied force (ground truth).
 * Same tensor layout as the real model: det_logit[1], loc_logits[nRegions],
 * dir[3], mag[1] (normalized by forceMax).
 */
export function fakeSensorFromApplied(applied, regions, forceMax) {
  const det_logit = new Float32Array(1);
  const loc_logits = new Float32Array(regions.length);
  const dir = new Float32Array(3);
  const mag = new Float32Array(1);
  if (!applied || applied.force.length() < 0.5) {
    det_logit[0] = -4 + Math.random() * 0.5;
    loc_logits.fill(0);
    return { det_logit, loc_logits, dir, mag };
  }
  const f = applied.force;
  const region = regionOfBody(applied.bodyName);
  const ridx = regions.indexOf(region);
  det_logit[0] = 2.5 + Math.random() * 0.8;
  for (let i = 0; i < regions.length; i++) {
    loc_logits[i] = (i === ridx ? 3.0 : 0) + (Math.random() - 0.5) * 0.6;
  }
  const n = f.clone().normalize();
  dir[0] = n.x + (Math.random() - 0.5) * 0.15;
  dir[1] = n.y + (Math.random() - 0.5) * 0.15;
  dir[2] = n.z + (Math.random() - 0.5) * 0.15;
  mag[0] = Math.min(1.5, (f.length() / forceMax) * (0.85 + Math.random() * 0.2));
  return { det_logit, loc_logits, dir, mag };
}

/**
 * Hysteresis + persistence detection gate (tonight's calibrated deployment
 * operating point): contact turns ON after k consecutive frames with
 * sigmoid(det) >= thHi, and OFF after k consecutive frames below thLo.
 * Update once per 50 Hz sensor tick (NOT per render frame).
 */
export class ContactGate {
  constructor({ thHi = 0.5, thLo = 0.35, k = 3 } = {}) {
    this.thHi = thHi; this.thLo = thLo; this.k = k;
    this.reset();
  }

  reset() { this.on = false; this._cnt = 0; }

  /** @param detP sigmoid(det_logit). @returns debounced contact state */
  update(detP) {
    if (!this.on) {
      this._cnt = detP >= this.thHi ? this._cnt + 1 : 0;
      if (this._cnt >= this.k) { this.on = true; this._cnt = 0; }
    } else {
      this._cnt = detP < this.thLo ? this._cnt + 1 : 0;
      if (this._cnt >= this.k) { this.on = false; this._cnt = 0; }
    }
    return this.on;
  }
}

/**
 * Decode raw head outputs for display.
 * @param raw   {det_logit, loc_logits, dir, mag}; loc_logits is model.names-wide
 *              (5 regions for v3, 24 links for v4)
 * @param model {names, forceMax, bodyToRegion|null} — when bodyToRegion is
 *              given, names are link names and per-region probabilities are
 *              aggregated by summing link probs per region.
 * @param rootQuat wxyz root quaternion; when given, the dir head (BASE frame,
 *        matching the training labels) is rotated into world/MuJoCo coords.
 * @returns {det, probs(region->p, det-gated), linkProbs(link->p, det-gated)|null,
 *           argmaxLink|null, argmaxRegion, dir, magN(UNgated Newtons)}
 */
export function decodeSensor(raw, model, rootQuat = null) {
  const det = sigmoid(raw.det_logit[0]);
  const sm = softmax(raw.loc_logits);
  const names = model.names;

  const probs = Object.fromEntries(REGIONS.map((r) => [r, 0]));
  let linkProbs = null;
  let argmaxLink = null;
  if (model.bodyToRegion) {
    // link head: aggregate per region, keep per-link probabilities
    linkProbs = {};
    names.forEach((n, i) => {
      linkProbs[n] = sm[i] * det;
      const r = model.bodyToRegion[n] ?? regionOfBody(n);
      probs[r] = (probs[r] ?? 0) + sm[i] * det;
    });
    argmaxLink = names[sm.indexOf(Math.max(...sm))];
  } else {
    names.forEach((r, i) => { probs[r] = sm[i] * det; });
  }
  const argmaxRegion = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0];

  let d = [raw.dir[0], raw.dir[1], raw.dir[2]];
  if (rootQuat) d = quatApply(rootQuat, d);
  const dir = new THREE.Vector3(d[0], d[1], d[2]);
  if (dir.lengthSq() > 1e-9) dir.normalize();
  const magN = Math.max(0, raw.mag[0]) * model.forceMax;
  return { det, probs, linkProbs, argmaxLink, argmaxRegion, dir, magN };
}

export class SensorViz {
  /**
   * @param robot RobotRenderer
   * @param curveCanvas HTMLCanvasElement for the rolling force plot
   * @param meta sensor meta json ({force_max, det_thresh, body_names, ...})
   * @param regionMap region_map.json ({body_to_region}) or null
   */
  constructor(robot, curveCanvas, meta, regionMap = null) {
    this.robot = robot;
    this.setModel(meta, regionMap);

    this.arrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.3, 0x3ecf6e, 0.08, 0.045);
    this.arrow.visible = false;
    robot.root.add(this.arrow);

    this.canvas = curveCanvas;
    this.ctx = curveCanvas.getContext('2d');
    this.samples = []; // {t, applied, predicted}
    this.windowSec = 10;

    this._centroid = new THREE.Vector3();
    this.last = { det: 0, probs: Object.fromEntries(REGIONS.map((r) => [r, 0])), magN: 0, contactOn: false };
  }

  /** (Re)configure for a sensor model's meta — supports the HUD model picker. */
  setModel(meta, regionMap = null) {
    this.names = meta?.body_names ?? REGIONS;
    this.forceMax = meta?.force_max ?? 40;
    this.detThresh = meta?.det_thresh ?? 0.5;
    // Link-head models (names are MJCF bodies, not regions) need aggregation.
    const isLinkHead = this.names.some((n) => !REGIONS.includes(n));
    this.isLinkHead = isLinkHead;
    this.bodyToRegion = isLinkHead ? (regionMap?.body_to_region ?? {}) : null;
    this.model = { names: this.names, forceMax: this.forceMax, bodyToRegion: this.bodyToRegion };
  }

  /**
   * @param applied  output of DragForce.current() (ground truth), or null
   * @param rawOut   real sensor-head outputs; pass null to use the fake stub
   * @param t        sim time [s]
   * @param rootQuat wxyz root quat for the base->world dir transform (real
   *                 outputs only; the fake stub already produces world dirs)
   * @param contactOn debounced ContactGate state (50 Hz); null -> fall back to
   *                 raw det >= detThresh (fake-stub path)
   */
  update(applied, rawOut, t, rootQuat = null, contactOn = null) {
    const raw = rawOut ?? fakeSensorFromApplied(applied, REGIONS, this.forceMax);
    const dec = rawOut
      ? decodeSensor(raw, this.model, rootQuat)
      : decodeSensor(raw, { names: REGIONS, forceMax: this.forceMax, bodyToRegion: null }, null);
    const on = contactOn ?? dec.det >= this.detThresh;
    const magN = on ? dec.magN : 0;
    this.last = { ...dec, magN, contactOn: on };

    // heatmap (dark while the debounced gate is OFF). Link-head models show
    // their true K-way granularity: each candidate link glows by its own
    // probability (region models fall back to region flooding).
    if (on) {
      if (dec.linkProbs) {
        this.robot.setLinkTint(dec.linkProbs, dec.probs, dec.det);
      } else {
        this.robot.setRegionTint(dec.probs, dec.argmaxLink, dec.det);
      }
    } else {
      this.robot.setRegionTint(this.constructor.ZERO_PROBS);
    }

    // green predicted-force arrow. Link-head models anchor it at the predicted
    // LINK (the one that glows); region models use the region centroid.
    if (on && magN > 0.5) {
      if (dec.linkProbs && dec.argmaxLink) {
        this.robot.bodyCentroid(dec.argmaxLink, this._centroid);
      } else {
        this.robot.regionCentroid(dec.argmaxRegion, this._centroid);
      }
      this.arrow.visible = true;
      this.arrow.position.copy(this._centroid);
      this.arrow.setDirection(dec.dir);
      this.arrow.setLength(0.15 + 0.5 * Math.min(1, magN / this.forceMax), 0.07, 0.04);
    } else {
      this.arrow.visible = false;
    }

    // rolling curve (predicted gated by the debounced state)
    const appliedN = applied ? applied.force.length() : 0;
    this.samples.push({ t, applied: appliedN, predicted: magN });
    const cutoff = t - this.windowSec;
    while (this.samples.length && this.samples[0].t < cutoff) this.samples.shift();
    this.#drawCurve(t);
  }

  static ZERO_PROBS = Object.fromEntries(REGIONS.map((r) => [r, 0]));

  #drawCurve(tNow) {
    const c = this.canvas, ctx = this.ctx;
    const dpr = window.devicePixelRatio || 1;
    const W = c.clientWidth, H = c.clientHeight;
    if (c.width !== W * dpr || c.height !== H * dpr) { c.width = W * dpr; c.height = H * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const padL = 34, padB = 14, padT = 6, padR = 6;
    const plotW = W - padL - padR, plotH = H - padT - padB;

    let yMax = 20;
    for (const s of this.samples) yMax = Math.max(yMax, s.applied, s.predicted);
    yMax = Math.ceil(yMax / 10) * 10;

    // axes
    ctx.strokeStyle = '#4a5462'; ctx.lineWidth = 1;
    ctx.strokeRect(padL, padT, plotW, plotH);
    ctx.fillStyle = '#8b96a3'; ctx.font = '10px system-ui';
    ctx.textAlign = 'right';
    for (const frac of [0, 0.5, 1]) {
      const yv = yMax * frac, y = padT + plotH * (1 - frac);
      ctx.fillText(`${yv.toFixed(0)} N`, padL - 4, y + 3);
      if (frac > 0 && frac < 1) {
        ctx.strokeStyle = '#333c48';
        ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(padL + plotW, y); ctx.stroke();
      }
    }
    ctx.textAlign = 'center';
    ctx.fillText(`last ${this.windowSec}s`, padL + plotW / 2, H - 3);

    const xOf = (t) => padL + plotW * (1 - (tNow - t) / this.windowSec);
    const yOf = (v) => padT + plotH * (1 - Math.min(v, yMax) / yMax);

    const drawSeries = (key, color) => {
      ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
      let started = false;
      for (const s of this.samples) {
        const x = xOf(s.t), y = yOf(s[key]);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };
    drawSeries('applied', '#e14b4b');
    drawSeries('predicted', '#3ecf6e');

    // legend
    ctx.textAlign = 'left'; ctx.font = '10px system-ui';
    ctx.fillStyle = '#e14b4b'; ctx.fillText('applied |F|', padL + 6, padT + 11);
    ctx.fillStyle = '#3ecf6e'; ctx.fillText('predicted |F|', padL + 70, padT + 11);
  }
}
