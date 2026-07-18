// Loading MuJoCo (official @mujoco/mujoco WASM bindings, MuJoCo 3.10) and the
// G1 model with its mesh assets through a MjVFS virtual filesystem.
//
// Environment-agnostic: asset bytes are passed in as a Map, so the same code
// path runs in the browser (fetch) and in the node smoke test (fs).

import loadMujoco from '@mujoco/mujoco';

export const PHYSICS_DT = 0.005; // 200 Hz

/** Initialize the WASM module (call once). */
export async function initMujoco() {
  return await loadMujoco();
}

/** Extract mesh file names referenced by the MJCF, plus the compiler meshdir. */
export function listMeshAssets(xmlText) {
  const meshdir = (xmlText.match(/<compiler[^>]*\bmeshdir="([^"]+)"/) || [])[1] || '';
  const files = [];
  const re = /<mesh\b[^>]*\bfile="([^"]+)"/g;
  let m;
  while ((m = re.exec(xmlText)) !== null) files.push(m[1]);
  return { meshdir, files: [...new Set(files)] };
}

/** Force the physics timestep in the MJCF text to PHYSICS_DT. */
export function patchTimestep(xmlText, dt = PHYSICS_DT) {
  return xmlText.replace(/timestep\s*=\s*['"][^'"]+['"]/, `timestep='${dt}'`);
}

/**
 * Compile an MjModel from XML text + asset bytes.
 * @param mujoco  the loaded WASM module
 * @param xmlText MJCF string
 * @param assets  Map<fileName, Uint8Array> keyed by the file= attribute value
 * @param meshdir compiler meshdir ('' if none)
 */
export function makeModel(mujoco, xmlText, assets, meshdir) {
  const vfs = new mujoco.MjVFS();
  try {
    for (const [name, bytes] of assets) {
      // Register under both the raw name and the meshdir-resolved path so the
      // compiler finds it regardless of how it resolves relative paths.
      vfs.addBuffer(name, bytes);
      if (meshdir) vfs.addBuffer(`${meshdir}/${name}`, bytes);
    }
    return mujoco.MjModel.from_xml_string(patchTimestep(xmlText), vfs);
  } finally {
    vfs.delete();
  }
}

/**
 * Fetch the MJCF + all its mesh bytes from a base URL (no mujoco needed).
 * Split from compilation so the ~19 MB of meshes can download in parallel with
 * the MuJoCo/ONNX WASM instead of waiting for them.
 */
export async function fetchG1Assets(baseUrl = 'assets/g1', xmlName = 'g1.xml') {
  const xmlText = await (await fetch(`${baseUrl}/${xmlName}`)).text();
  const { meshdir, files } = listMeshAssets(xmlText);
  const buffers = await Promise.all(
    files.map(async (f) => {
      const resp = await fetch(`${baseUrl}/${meshdir ? meshdir + '/' : ''}${f}`);
      if (!resp.ok) throw new Error(`failed to fetch mesh asset ${f}: ${resp.status}`);
      return new Uint8Array(await resp.arrayBuffer());
    })
  );
  const assets = new Map(files.map((f, i) => [f, buffers[i]]));
  return { xmlText, assets, meshdir };
}

/** Compile prefetched assets into an MjModel + MjData (needs the WASM module). */
export function compileG1(mujoco, { xmlText, assets, meshdir }) {
  const model = makeModel(mujoco, xmlText, assets, meshdir);
  const data = new mujoco.MjData(model);
  return { model, data };
}

/** Browser helper: fetch the MJCF + its meshes from a base URL and compile. */
export async function loadG1(mujoco, baseUrl = 'assets/g1', xmlName = 'g1.xml') {
  return compileG1(mujoco, await fetchG1Assets(baseUrl, xmlName));
}

/**
 * Introspect actuators: names, target joint qpos/dof addresses, ctrlrange.
 * Assumes joint transmission with hinge joints (true for the G1 motors).
 */
export function getActuatorInfo(mujoco, model) {
  const nu = model.nu;
  const names = [];
  const qposAdr = new Int32Array(nu);
  const dofAdr = new Int32Array(nu);
  const ctrlRange = new Float64Array(2 * nu);
  const trnid = model.actuator_trnid; // [nu x 2]
  const jntQposAdr = model.jnt_qposadr;
  const jntDofAdr = model.jnt_dofadr;
  const cr = model.actuator_ctrlrange; // [nu x 2]
  for (let i = 0; i < nu; i++) {
    const acc = model.actuator(i);
    names.push(acc.name);
    acc.delete?.();
    const j = trnid[2 * i];
    qposAdr[i] = jntQposAdr[j];
    dofAdr[i] = jntDofAdr[j];
    ctrlRange[2 * i] = cr[2 * i];
    ctrlRange[2 * i + 1] = cr[2 * i + 1];
  }
  return { nu, names, qposAdr, dofAdr, ctrlRange };
}

/** Body names indexed by body id. */
export function getBodyNames(model) {
  const names = [];
  for (let i = 0; i < model.nbody; i++) {
    const acc = model.body(i);
    names.push(acc.name);
    acc.delete?.();
  }
  return names;
}

/**
 * Reset the sim to the default standing pose and run forward dynamics.
 * qDes is in actuator order.
 */
export function resetToStand(mujoco, model, data, act, qDes, pelvisHeight = 0.755) {
  mujoco.mj_resetData(model, data);
  const qpos = data.qpos;
  qpos[2] = pelvisHeight;            // free joint z
  qpos[3] = 1; qpos[4] = 0; qpos[5] = 0; qpos[6] = 0; // identity quat
  for (let i = 0; i < act.nu; i++) qpos[act.qposAdr[i]] = qDes[i];
  mujoco.mj_forward(model, data);
}
