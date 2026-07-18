// ONNX Runtime Web session management:
//   models/policy.onnx          input 'policy' [1,257] RAW obs (VecNorm folded in)
//   models/policy.json          out_keys — selects the 'action' output BY POSITION
//   models/force_sensor_v3.onnx input 'x' [1,1920], outputs det_logit/loc_logits/dir/mag
//   models/force_sensor_v3.meta.json normalization stats etc.
//
// Environment-agnostic: pass Node fs-based loaders for the smoke test; the
// browser defaults use fetch().

import * as ort from 'onnxruntime-web';

// Force single-threaded: with threads>1 ort spawns a Worker pointing at its
// own module URL, which after vite build is the whole app bundle -> the app
// re-executes inside a worker ("document is not defined") and session
// creation hangs. Models here are small; 1 thread is plenty.
// (Also: don't set ort.env.wasm.wasmPaths to a public/ folder — vite dev
// refuses JS imports of public assets; ort resolves via import.meta.url.)
ort.env.wasm.numThreads = 1;

function browserLoaders(base) {
  return {
    json: async (name) => (await fetch(`${base}models/${name}`)).json(),
    bytes: async (name) => new Uint8Array(await (await fetch(`${base}models/${name}`)).arrayBuffer()),
  };
}

async function timedRun(session, feeds, label) {
  const t0 = performance.now();
  const out = await session.run(feeds);
  const dt = performance.now() - t0;
  console.log(`[onnx] ${label} inference: ${dt.toFixed(2)} ms; outputs:`,
    Object.fromEntries(Object.entries(out).map(([k, v]) => [k, v.dims])));
  return out;
}

/**
 * Rolling-window v3 sensor runner (port of wbc_sim2sim.py ForceSensorV3Runner):
 * keeps W=6 raw 320-dim frames, newest LAST (flattened oldest..newest), and
 * applies the per-frame (x - mean) / std normalization.
 */
export class SensorRunner {
  constructor(session, meta) {
    this.session = session;
    this.meta = meta;
    this.inName = session.inputNames.includes('x') ? 'x' : session.inputNames[0];
    this.W = meta.window;
    this.baseDim = meta.base_dim;
    this.xm = Float32Array.from(meta.x_mean[0] ?? meta.x_mean);
    this.xs = Float32Array.from(meta.x_std[0] ?? meta.x_std);
    this.detThresh = meta.det_thresh;
    this.forceMax = meta.force_max;
    this.regions = meta.body_names;
    this.buf = null; // Array<Float32Array[baseDim]>, newest last
    this._x = new Float32Array(this.W * this.baseDim);
  }

  reset() { this.buf = null; }

  /** @param frame Float32Array[320] raw wbc obs. Returns raw head outputs. */
  async run(frame) {
    if (this.buf === null) {
      this.buf = Array.from({ length: this.W }, () => Float32Array.from(frame));
    } else {
      const oldest = this.buf.shift();
      oldest.set(frame);
      this.buf.push(oldest);
    }
    const x = this._x, D = this.baseDim;
    for (let f = 0; f < this.W; f++) {
      const fr = this.buf[f];
      for (let k = 0; k < D; k++) x[f * D + k] = (fr[k] - this.xm[k]) / this.xs[k];
    }
    const out = await this.session.run({
      [this.inName]: new ort.Tensor('float32', x, [1, this.W * D]),
    });
    return {
      det_logit: out.det_logit.data,
      loc_logits: out.loc_logits.data,
      dir: out.dir.data,
      mag: out.mag.data,
    };
  }
}

export async function initOnnx(loaders) {
  loaders = loaders ?? browserLoaders(import.meta.env.BASE_URL);
  const opts = { executionProviders: ['wasm'] };

  const policyMeta = await loaders.json('policy.json');
  // 24-link -> 5-region aggregation map (used by link-head models like v4).
  const regionMap = await loaders.json('region_map.json');

  const policy = await ort.InferenceSession.create(await loaders.bytes('policy.onnx'), opts);
  console.log('[onnx] policy loaded. inputs:', policy.inputNames, 'outputs:', policy.outputNames);
  // Graph output names are node names and can COLLIDE ('loc'/'action' both map
  // to 'linear_9'; verified numerically equal in the prototype). Select the
  // action output BY POSITION from policy.json out_keys; the name-keyed fetch
  // below is safe because the colliding tensors are identical.
  const actionName = policy.outputNames[policyMeta.out_keys.indexOf('action')];
  const policyIn = policy.inputNames[0]; // 'policy'

  await timedRun(policy, {
    [policyIn]: new ort.Tensor('float32', new Float32Array(257), [1, 257]),
  }, 'policy (dummy [1,257])');

  /** Load a sensor model by base name (e.g. 'force_sensor_v4'); everything
   *  (window size, base dim, head width) adapts from its meta json. */
  async function loadSensor(name) {
    const meta = await loaders.json(`${name}.meta.json`);
    if (meta.imp_norm) {
      throw new Error(`${name}: imp_norm models need the impedance torque `
        + 'transform, which is not implemented in the JS deployment');
    }
    const session = await ort.InferenceSession.create(await loaders.bytes(`${name}.onnx`), opts);
    console.log(`[onnx] sensor '${name}' loaded. W=${meta.window} K=${meta.num_bodies}`,
      'outputs:', session.outputNames);
    const inDim = meta.in_dim ?? meta.window * meta.base_dim;
    await timedRun(session, {
      [session.inputNames[0]]: new ort.Tensor('float32', new Float32Array(inDim), [1, inDim]),
    }, `sensor ${name} (dummy [1,${inDim}])`);
    return new SensorRunner(session, meta);
  }

  const api = {
    regionMap,
    policy,
    // The sensor is loaded in the BACKGROUND (see sensorReady below) so the
    // robot is interactive — stand/walk needs only the policy — before the
    // ~7.6 MB sensor model finishes. null until ready.
    sensor: null,
    meta: null,
    /** Load an additional runner without replacing the active one. */
    loadSensor,
    /** Swap the active sensor model (fresh window buffer). */
    async switchSensor(name) {
      api.sensor = await loadSensor(name);
      api.meta = api.sensor.meta;
      return api.sensor;
    },
    /** obs: Float32Array[257] RAW policy obs -> Float32Array[29] action */
    async runPolicy(obs) {
      const out = await policy.run({
        [policyIn]: new ort.Tensor('float32', obs, [1, obs.length]),
      });
      return out[actionName].data;
    },
  };
  // Default: the domain-calibrated 24-LINK model. Kicked off now, awaited by
  // the caller via api.sensorReady; doesn't block the policy/sim from starting.
  api.sensorReady = loadSensor('force_sensor_v4c_links').then((s) => {
    if (!api.sensor) { api.sensor = s; api.meta = s.meta; }
    return s;
  });
  return api;
}
