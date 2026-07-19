// Entry point: three.js scene + the REAL pipeline (JS port of the validated
// forcesense/sim2sim.py): ONNX policy at 50 Hz driving 200 Hz MuJoCo physics,
// v3 force sensor on the wbc obs window, drag-to-push interaction.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

import { initMujoco, fetchG1Assets, compileG1, getActuatorInfo, getBodyNames, resetToStand } from './mujocoLoader.js';
import { computePD, defaultGainsAndPose, computeBaseAssist } from './control.js';
import { Sim2Sim, CTRL_HZ, DECIMATION } from './sim2sim.js';
import { RobotRenderer, REGIONS } from './robotRenderer.js';
import { DragForce } from './drag.js';
import { SensorViz, ContactGate, sigmoid } from './sensorViz.js';
import { initOnnx } from './onnx.js';
import { Hud } from './hud.js';

// ---------------------------------------------------------------------------
// Flags
// ---------------------------------------------------------------------------
const USE_POLICY = true;      // ONNX policy control (validated pipeline)
const BASE_ASSIST = false;    // legacy pelvis-spring helper — only relevant to
                              // the fallback statue (USE_POLICY=false / ONNX failure)
const USE_FAKE_SENSOR = false; // fabricate sensor outputs instead of the model

const CTRL_DT_MS = 1000 / CTRL_HZ; // 20 ms

const state = { paused: false };

const hud = new Hud(
  REGIONS,
  () => (state.paused = !state.paused),
  () => resetSim(),
  (name) => switchModel(name),
);

let mujoco, model, data, sim, onnx, robot, drag, viz;
let legacy = null;     // fallback-statue state when the policy is unavailable
let policyMode = false;
let latestRaw = null;  // newest raw sensor-head outputs (50 Hz)
let simTime = 0;

// Calibrated deployment operating points. Preferred source is the model
// meta's `recommended` field (present on the domain-calibrated v4c models);
// this static map is the fallback for older metas. v4's Isaac-collected
// static data reads MuJoCo standing as det≈0.70 (sim2sim domain gap) while
// pushes reach 0.9+, so its gate sits above the rest plateau (measured rest
// ON = 0.000 at 0.85/0.75 with wrist holds still gating ON). v3 rests ~0.04.
const OP_POINTS = {
  force_sensor_v4c_links: { thHi: 0.6, thLo: 0.45, k: 3 },
  force_sensor_v4c: { thHi: 0.6, thLo: 0.45, k: 3 },
  force_sensor_v4c_restboost: { thHi: 0.4, thLo: 0.25, k: 3 },
  force_sensor_v4: { thHi: 0.85, thLo: 0.75, k: 3 },
  force_sensor_v3: { thHi: 0.5, thLo: 0.35, k: 3 },
  force_sensor_resid: { thHi: 0.5, thLo: 0.35, k: 3 },
};

/** Operating point for a model: meta.recommended if present, else the map. */
function opPointFor(name, meta) {
  const rec = meta?.recommended;
  if (rec && rec.th_hi != null && rec.th_lo != null) {
    return { thHi: rec.th_hi, thLo: rec.th_lo, k: rec.k ?? 3 };
  }
  return OP_POINTS[name] ?? OP_POINTS.force_sensor_v3;
}

let gate = new ContactGate(OP_POINTS.force_sensor_v4c_links);
hud.setThresholds(gate.thLo, gate.thHi);
let contactOn = false;

// --- walk command (base-frame target velocity, m/s) driven by WASD/arrows ---
const walkKeys = new Set();
const WALK_SPEED = 0.6, BACK_SPEED = 0.4, STRAFE_SPEED = 0.4;
function walkKeyOf(key) {
  switch (key) {
    case 'w': case 'W': case 'ArrowUp': return 'w';
    case 's': case 'S': case 'ArrowDown': return 's';
    case 'a': case 'A': case 'ArrowLeft': return 'a';
    case 'd': case 'D': case 'ArrowRight': return 'd';
    default: return null;
  }
}
function applyWalkCommand() {
  if (!sim || !policyMode) return;
  let vx = 0, vy = 0;
  if (walkKeys.has('w')) vx += WALK_SPEED;
  if (walkKeys.has('s')) vx -= BACK_SPEED;
  if (walkKeys.has('a')) vy += STRAFE_SPEED;  // base-frame +y = left
  if (walkKeys.has('d')) vy -= STRAFE_SPEED;
  sim.setTargetVel(vx, vy);
}
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const k = walkKeyOf(e.key);
  if (!k) return;
  e.preventDefault();
  if (!e.repeat) { walkKeys.add(k); applyWalkCommand(); }
});
window.addEventListener('keyup', (e) => {
  const k = walkKeyOf(e.key);
  if (!k) return;
  walkKeys.delete(k);
  applyWalkCommand();
});
// On-screen directional pad: press-and-hold (mouse + touch) to walk.
for (const btn of document.querySelectorAll('.wp-btn')) {
  const dir = btn.dataset.dir;
  const press = (e) => {
    e.preventDefault();
    walkKeys.add(dir); btn.classList.add('on'); applyWalkCommand();
    btn.setPointerCapture?.(e.pointerId);
  };
  const release = () => {
    if (!walkKeys.has(dir)) return;
    walkKeys.delete(dir); btn.classList.remove('on'); applyWalkCommand();
  };
  btn.addEventListener('pointerdown', press);
  btn.addEventListener('pointerup', release);
  btn.addEventListener('pointerleave', release);
  btn.addEventListener('pointercancel', release);
}

/** Rebuild the HUD localization bars for the active model's head (5 vs 24). */
function shortLinkLabel(name) {
  return name.replace(/_link$/, '').replace(/^left_/, 'L ').replace(/^right_/, 'R ').replace(/_/g, ' ');
}
function refreshHudBars() {
  if (!viz) return;
  if (viz.isLinkHead) {
    hud.setBars(viz.names.map((n) => ({ key: n, label: shortLinkLabel(n) })));
  } else {
    hud.setBars(REGIONS.map((r) => ({ key: r, label: r })));
  }
}

/** Configure viz + gate + HUD bars for a (just-loaded) sensor model. */
function applySensorModel(meta, name) {
  viz.setModel(meta, onnx.regionMap);
  gate = new ContactGate(opPointFor(name, meta));
  hud.setThresholds(gate.thLo, gate.thHi);
  latestRaw = null;
  contactOn = false;
  refreshHudBars();
}

function resetSim() {
  if (!mujoco) return;
  walkKeys.clear();
  if (policyMode) {
    sim.reset();
    sim.setTargetVel(0, 0);
    onnx.sensor?.reset();
  } else if (legacy) {
    resetToStand(mujoco, model, data, legacy.act, legacy.qDes);
  }
  latestRaw = null;
  gate.reset();
  contactOn = false;
}

let switching = false;
let userPickedSensor = false;  // suppress the background default once user picks
async function switchModel(name) {
  if (!onnx || switching) return;
  switching = true;
  userPickedSensor = true;
  hud.status(`loading ${name}…`);
  try {
    const runner = await onnx.switchSensor(name);
    applySensorModel(runner.meta, name);
    hud.hideStatus();
    console.log(`[onnx] switched sensor to ${name} (W=${runner.W}, K=${runner.meta.num_bodies})`);
  } catch (err) {
    hud.status(`sensor switch failed: ${err.message}`, true);
  } finally {
    switching = false;
  }
}

// ---------------------------------------------------------------------------
// three.js scene
// ---------------------------------------------------------------------------
const canvas = document.getElementById('scene-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x10141a);
scene.fog = new THREE.Fog(0x10141a, 8, 30);

const camera = new THREE.PerspectiveCamera(45, 2, 0.05, 100);
camera.position.set(2.2, 1.4, 2.2); // three coords (Y up)

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.75, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.maxPolarAngle = Math.PI * 0.52;
controls.minDistance = 0.8;
controls.maxDistance = 12;

scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x30363f, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(3, 6, 2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -3; sun.shadow.camera.right = 3;
sun.shadow.camera.top = 3; sun.shadow.camera.bottom = -3;
scene.add(sun);
const fill = new THREE.DirectionalLight(0x8fb2ff, 0.35);
fill.position.set(-4, 3, -3);
scene.add(fill);

function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function boot() {
  try {
    hud.status('loading MuJoCo + models…');
    // Kick off every independent download at once so they overlap: the MuJoCo
    // WASM compile, the G1 meshes (~19 MB), and the ONNX runtime + policy all
    // race instead of running one-after-another. The sensor model then streams
    // in the background (onnx.sensorReady) after the robot is already standing.
    const g1AssetsP = fetchG1Assets(`${import.meta.env.BASE_URL}assets/g1`);
    let onnxP = null;
    try {
      onnxP = Promise.race([
        initOnnx(),
        new Promise((_, rej) => setTimeout(() => rej(new Error('timeout after 20s')), 20000)),
      ]);
    } catch { onnxP = null; }

    mujoco = await initMujoco();
    ({ model, data } = compileG1(mujoco, await g1AssetsP));
    const bodyNames = getBodyNames(model);

    try {
      // Never let a wedged ONNX backend block the simulation.
      onnx = await onnxP;
      console.log('[onnx] policy ready; sensor loading in background');
    } catch (err) {
      hud.status(`ONNX load failed — falling back to PD statue: ${err.message}`, true);
      console.error(err);
      onnx = null;
    }

    policyMode = USE_POLICY && onnx !== null;
    if (policyMode) {
      sim = new Sim2Sim(mujoco, model, data);
    } else {
      // Legacy scaffold statue: stiff PD + pelvis base-assist spring.
      const act = getActuatorInfo(mujoco, model);
      const gains = defaultGainsAndPose(act.names);
      legacy = { act, ...gains, pelvisId: bodyNames.indexOf('pelvis') };
      resetToStand(mujoco, model, data, act, gains.qDes);
    }

    hud.status('building scene…');
    robot = new RobotRenderer(model, data, bodyNames, scene);
    drag = new DragForce(camera, renderer.domElement, controls, robot, data,
      document.getElementById('drag-label'));
    viz = new SensorViz(robot, document.getElementById('curve-canvas'),
      onnx?.meta ?? null, onnx?.regionMap ?? null);
    refreshHudBars();  // starts as 5 region bars until the sensor streams in

    // Sensor finished downloading in the background: switch viz/gate/HUD to it
    // (unless the user already picked a different model from the dropdown).
    if (onnx) {
      onnx.sensorReady
        .then((s) => {
          if (!userPickedSensor) applySensorModel(s.meta, 'force_sensor_v4c_links');
          console.log('[onnx] sensor ready (background)');
        })
        .catch((err) => hud.status(`sensor load failed: ${err.message}`, true));
    }

    // Debug/testing hook (used by test/browser_check.mjs to aim drags).
    window.__demo = {
      state,
      policyMode,
      screenPos(bodyName) {
        const b = bodyNames.indexOf(bodyName);
        if (b < 0) return null;
        const v = new THREE.Vector3(data.xpos[3 * b], data.xpos[3 * b + 1], data.xpos[3 * b + 2]);
        robot.root.localToWorld(v);
        v.project(camera);
        return {
          x: (v.x * 0.5 + 0.5) * window.innerWidth,
          y: (-v.y * 0.5 + 0.5) * window.innerHeight,
        };
      },
      latest() {
        return {
          det: viz.last.det, probs: viz.last.probs, magN: viz.last.magN,
          contactOn, argmaxLink: viz.last.argmaxLink ?? null,
          argmaxRegion: viz.last.argmaxRegion ?? null, z: data.qpos[2],
        };
      },
      grabbed() { return drag.current()?.bodyName ?? null; },
    };

    if (onnx) hud.hideStatus();
    startSimLoop();
    startRenderLoop();
  } catch (err) {
    hud.status(`FATAL: ${err.message}\n${err.stack ?? ''}`, true);
    throw err;
  }
}

// ---------------------------------------------------------------------------
// Sim loop: soft-realtime 50 Hz control ticks (each = policy inference +
// 4 physics substeps @200 Hz + sensor inference), decoupled from rendering.
// ---------------------------------------------------------------------------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function currentDragForce() {
  const applied = drag.current();
  if (!applied || applied.force.length() < 1e-3) return { F: null, bodyId: -1 };
  return { F: [applied.force.x, applied.force.y, applied.force.z], bodyId: applied.bodyId };
}

async function startSimLoop() {
  // legacy-statue scratch
  const leg = legacy ? {
    qposJ: new Float64Array(legacy.act.nu),
    qvelJ: new Float64Array(legacy.act.nu),
    tau: new Float64Array(legacy.act.nu),
    assist: new Float64Array(6),
    anchor: [0, 0, 0.755],
  } : null;

  // Accumulator scheduling: when the main thread is busy (heavy rendering,
  // background-tab throttling) run up to MAX_CATCHUP ticks per wakeup so sim
  // time keeps tracking the wall clock; beyond that, drop time.
  const MAX_CATCHUP = 4;
  let due = performance.now();
  for (;;) {
    if (!state.paused) {
      let n = 0;
      while (performance.now() >= due && n < MAX_CATCHUP) {
        const { F, bodyId } = currentDragForce();
        if (policyMode) {
          const action = await onnx.runPolicy(sim.policyObs());
          sim.controlStep(action, F, bodyId);
          if (!USE_FAKE_SENSOR && !switching && onnx.sensor) {
            // residual models consume the 29-dim controller-invariant channel;
            // proprioceptive models consume the 320-dim wbc obs frame.
            const frame = onnx.meta?.input_source === 'resid'
              ? sim.residChannel()
              : sim.wbcObs();
            latestRaw = await onnx.sensor.run(frame);
            // debounced detection gate — one update per 50 Hz sensor tick
            contactOn = gate.update(sigmoid(latestRaw.det_logit[0]));
          }
        } else {
          for (let s = 0; s < DECIMATION; s++) legacyStep(leg, F, bodyId);
        }
        simTime += CTRL_DT_MS / 1000;
        due += CTRL_DT_MS;
        n++;
      }
      if (performance.now() > due) due = performance.now(); // drop unpayable debt
    } else {
      due = performance.now();
    }
    await sleep(Math.max(0, due - performance.now()));
  }
}

function legacyStep(leg, F, bodyId) {
  const { act, kp, kd, qDes, pelvisId } = legacy;
  const qpos = data.qpos, qvel = data.qvel, ctrl = data.ctrl;
  for (let i = 0; i < act.nu; i++) {
    leg.qposJ[i] = qpos[act.qposAdr[i]];
    leg.qvelJ[i] = qvel[act.dofAdr[i]];
  }
  computePD(qDes, leg.qposJ, leg.qvelJ, kp, kd, act.ctrlRange, leg.tau);
  for (let i = 0; i < act.nu; i++) ctrl[i] = leg.tau[i];
  const xfrc = data.xfrc_applied;
  xfrc.fill(0);
  if (BASE_ASSIST || !USE_POLICY) {
    computeBaseAssist(qpos, qvel, leg.anchor, undefined, leg.assist);
    for (let k = 0; k < 6; k++) xfrc[pelvisId * 6 + k] = leg.assist[k];
  }
  if (F && bodyId >= 0) {
    xfrc[6 * bodyId] += F[0];
    xfrc[6 * bodyId + 1] += F[1];
    xfrc[6 * bodyId + 2] += F[2];
  }
  mujoco.mj_step(model, data);
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------
function startRenderLoop() {
  let lastLog = 0;
  const pelvisWorld = new THREE.Vector3();
  let prevPelvis = null;

  function frame(now) {
    requestAnimationFrame(frame);

    robot.update();
    drag.updateArrow();

    // Camera follows the robot as it walks: pan camera + orbit target by the
    // pelvis' world-space delta so it stays framed (no effect while standing).
    pelvisWorld.set(data.qpos[0], data.qpos[1], data.qpos[2]);
    robot.root.localToWorld(pelvisWorld);
    if (prevPelvis) {
      const dx = pelvisWorld.x - prevPelvis.x;
      const dy = pelvisWorld.y - prevPelvis.y;
      const dz = pelvisWorld.z - prevPelvis.z;
      camera.position.set(camera.position.x + dx, camera.position.y + dy, camera.position.z + dz);
      controls.target.set(controls.target.x + dx, controls.target.y + dy, controls.target.z + dz);
      prevPelvis.copy(pelvisWorld);
    } else {
      prevPelvis = pelvisWorld.clone();
    }

    const applied = drag.current();
    // Real sensor outputs (50 Hz, newest available) + root quat so the
    // base-frame dir head can be shown in world coordinates. With
    // USE_FAKE_SENSOR (or before the first tick) viz falls back to the stub.
    const rootQuat = policyMode ? sim.rootQuat() : null;
    viz.update(applied, USE_FAKE_SENSOR ? null : latestRaw, simTime, rootQuat,
      USE_FAKE_SENSOR || !latestRaw ? null : contactOn);
    hud.update(
      viz.last.det,
      viz.isLinkHead ? (viz.last.linkProbs ?? {}) : viz.last.probs,
      applied ? applied.force.length() : 0,
      viz.last.magN,
      viz.last.contactOn,
    );
    hud.setGrabbed(applied?.bodyName ?? null);

    if (now - lastLog >= 1000) {
      lastLog = now;
      console.log(`[sim] t=${simTime.toFixed(1)}s pelvis z=${data.qpos[2].toFixed(3)} m`
        + ` det_p=${viz.last.det.toFixed(2)}${policyMode ? '' : ' (fallback statue)'}`);
    }

    controls.update();
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
}

boot();
