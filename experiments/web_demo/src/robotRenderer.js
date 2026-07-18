// Mirrors MuJoCo's mjModel geometry into three.js meshes (visual == physical:
// vertices/faces are read straight from the WASM heap, no separate STL loads),
// and provides region-based tinting for the sensor prediction heatmap.

import * as THREE from 'three';

export const REGIONS = ['left_arm', 'left_leg', 'right_arm', 'right_leg', 'trunk'];

/** Map a MuJoCo body name to a sensor region. */
export function regionOfBody(name) {
  const side = name.startsWith('left') ? 'left' : name.startsWith('right') ? 'right' : null;
  if (side && /(shoulder|elbow|wrist|hand)/.test(name)) return `${side}_arm`;
  if (side && /(hip|knee|ankle)/.test(name)) return `${side}_leg`;
  return 'trunk';
}

const mjGEOM_PLANE = 0;
const mjGEOM_MESH = 7;

/**
 * Build the scene mirror of the model.
 * Everything lives under `root`, which carries the MuJoCo(Z-up) -> three(Y-up)
 * rotation, so all positions/directions inside it are in MuJoCo coordinates.
 */
export class RobotRenderer {
  constructor(model, data, bodyNames, scene) {
    this.model = model;
    this.data = data;
    this.bodyNames = bodyNames;

    this.root = new THREE.Group();
    this.root.name = 'mujocoRoot';
    this.root.rotation.x = -Math.PI / 2;
    scene.add(this.root);

    /** @type {THREE.Mesh[]} robot meshes (raycast targets), userData: {geomIdx, bodyId, bodyName, region} */
    this.meshes = [];
    this.bodiesByRegion = new Map(REGIONS.map((r) => [r, []]));

    this.#buildGround();
    this.#buildRobotMeshes();

    // Cached temporaries
    this._m4 = new THREE.Matrix4();
  }

  #buildGround() {
    const geo = new THREE.PlaneGeometry(60, 60); // local +z normal == MuJoCo up
    const mat = new THREE.MeshStandardMaterial({ color: 0x2b3442, roughness: 0.95, metalness: 0.0 });
    const plane = new THREE.Mesh(geo, mat);
    plane.receiveShadow = true;
    this.root.add(plane);

    const grid = new THREE.GridHelper(60, 60, 0x4a5568, 0x3a4453);
    grid.rotation.x = Math.PI / 2; // GridHelper is XZ; rotate into MuJoCo XY
    grid.position.z = 0.002;
    this.root.add(grid);
  }

  #geometryForMesh(meshId) {
    const m = this.model;
    const va = m.mesh_vertadr[meshId], vn = m.mesh_vertnum[meshId];
    const fa = m.mesh_faceadr[meshId], fn = m.mesh_facenum[meshId];
    const positions = new Float32Array(m.mesh_vert.subarray(va * 3, (va + vn) * 3));
    const indices = new Uint32Array(m.mesh_face.subarray(fa * 3, (fa + fn) * 3));
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeVertexNormals();
    return geo;
  }

  #buildRobotMeshes() {
    const m = this.model;
    const types = m.geom_type, groups = m.geom_group, dataid = m.geom_dataid;
    const bodyid = m.geom_bodyid, rgba = m.geom_rgba;
    const geoCache = new Map();

    for (let g = 0; g < m.ngeom; g++) {
      if (types[g] === mjGEOM_PLANE) continue; // our own ground stands in
      // Only mirror the visual meshes (group 1); group 0 holds collision
      // duplicates and tiny foot-contact spheres.
      if (types[g] !== mjGEOM_MESH || groups[g] !== 1) continue;

      const meshId = dataid[g];
      if (!geoCache.has(meshId)) geoCache.set(meshId, this.#geometryForMesh(meshId));
      const baseColor = new THREE.Color(rgba[4 * g], rgba[4 * g + 1], rgba[4 * g + 2]);
      const mat = new THREE.MeshStandardMaterial({
        color: baseColor.clone(),
        roughness: 0.65,
        metalness: 0.25,
      });
      const mesh = new THREE.Mesh(geoCache.get(meshId), mat);
      mesh.castShadow = true;
      mesh.receiveShadow = false;
      mesh.matrixAutoUpdate = false;

      const b = bodyid[g];
      const bodyName = this.bodyNames[b];
      const region = regionOfBody(bodyName);
      mesh.userData = { geomIdx: g, bodyId: b, bodyName, region, baseColor };
      this.root.add(mesh);
      this.meshes.push(mesh);

      const list = this.bodiesByRegion.get(region);
      if (!list.includes(b)) list.push(b);
    }
  }

  /** Sync mesh transforms from the simulation (MuJoCo coords, root-local). */
  update() {
    const xpos = this.data.geom_xpos;
    const xmat = this.data.geom_xmat;
    for (const mesh of this.meshes) {
      const g = mesh.userData.geomIdx;
      const p = 3 * g, r = 9 * g;
      this._m4.set(
        xmat[r], xmat[r + 1], xmat[r + 2], xpos[p],
        xmat[r + 3], xmat[r + 4], xmat[r + 5], xpos[p + 1],
        xmat[r + 6], xmat[r + 7], xmat[r + 8], xpos[p + 2],
        0, 0, 0, 1,
      );
      mesh.matrix.copy(this._m4);
    }
  }

  /**
   * Tint body meshes toward green by per-region probability (0..1).
   * @param {Record<string, number>} regionProbs keyed by REGIONS entries
   * Two-level highlighting: every mesh gets a soft tint from its region's
   * aggregated probability; the meshes of `hotLink` (the argmax link of a
   * link-head model) get a strong tint (`hotVal`, e.g. det probability).
   */
  setRegionTint(regionProbs, hotLink = null, hotVal = 0) {
    for (const mesh of this.meshes) {
      const regionP = regionProbs[mesh.userData.region] ?? 0;
      let p = 0.55 * regionP; // softer region-level tint
      if (hotLink !== null && mesh.userData.bodyName === hotLink) {
        p = Math.max(p, hotVal); // full tint on the predicted link
      }
      // emissive lerp toward green; base color untouched so p=0 restores.
      mesh.material.emissive.setRGB(0.02 * p, 0.55 * p, 0.12 * p);
    }
  }

  /**
   * Per-link tint: each of the K candidate links glows by ITS OWN softmax
   * probability (normalized so the argmax link reaches `det` intensity) —
   * makes the model's actual K-way granularity visible instead of flooding
   * the whole region. Bodies that are not candidate links (e.g. head, wrist
   * pitch/yaw shells) get a faint region backdrop so the limb still reads.
   * @param {Record<string, number>} linkProbs  bodyName -> det-gated prob
   * @param {Record<string, number>} regionProbs region -> aggregated prob
   * @param {number} det detection probability (peak intensity)
   */
  setLinkTint(linkProbs, regionProbs, det) {
    let maxP = 0;
    for (const v of Object.values(linkProbs)) maxP = Math.max(maxP, v);
    const scale = maxP > 1e-6 ? det / maxP : 0;
    for (const mesh of this.meshes) {
      const own = linkProbs[mesh.userData.bodyName];
      const p = own != null
        ? own * scale
        : 0.12 * (regionProbs[mesh.userData.region] ?? 0); // faint backdrop
      mesh.material.emissive.setRGB(0.02 * p, 0.55 * p, 0.12 * p);
    }
  }

  /** World position of a single body/link (MuJoCo coords), for the arrow anchor. */
  bodyCentroid(bodyName, out = new THREE.Vector3()) {
    const b = this.bodyNames.indexOf(bodyName);
    if (b < 0) return out.set(0, 0, 0);
    const xpos = this.data.xpos;
    return out.set(xpos[3 * b], xpos[3 * b + 1], xpos[3 * b + 2]);
  }

  /** Probability-weighted centroid: mean body xpos of a region (MuJoCo coords). */
  regionCentroid(region, out = new THREE.Vector3()) {
    const ids = this.bodiesByRegion.get(region) ?? [];
    const xpos = this.data.xpos;
    out.set(0, 0, 0);
    if (ids.length === 0) return out;
    for (const b of ids) out.x += xpos[3 * b], out.y += xpos[3 * b + 1], out.z += xpos[3 * b + 2];
    return out.multiplyScalar(1 / ids.length);
  }
}
