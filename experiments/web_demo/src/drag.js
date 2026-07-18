// Mouse interaction: left-drag ON the robot applies an external force to the
// grabbed body (red arrow at application point + "applied: XX N" label);
// left-drag on empty space orbits the camera (OrbitControls stays enabled).

import * as THREE from 'three';

// Sensor training range is 10-40 N; cap the drag force there so the demo
// stays in-distribution (and matches the prototype's push magnitudes).
const FORCE_PER_METER = 100; // N of force per meter of camera-plane drag
const FORCE_MAX = 40;        // N

export class DragForce {
  /**
   * @param camera   THREE.PerspectiveCamera
   * @param domEl    renderer.domElement
   * @param controls OrbitControls (disabled while grabbing the robot)
   * @param robot    RobotRenderer (for raycast targets + root transform)
   * @param data     MjData (xpos/xquat live views)
   * @param labelEl  HTML element for the "applied: XX N" readout
   */
  constructor(camera, domEl, controls, robot, data, labelEl) {
    this.camera = camera;
    this.controls = controls;
    this.robot = robot;
    this.data = data;
    this.labelEl = labelEl;

    this.raycaster = new THREE.Raycaster();
    this.active = null; // { bodyId, localPoint(THREE.Vector3, body frame) }
    this.forceMj = new THREE.Vector3();  // current force, MuJoCo coords
    this.pointMj = new THREE.Vector3();  // current application point, MuJoCo coords

    this.arrow = new THREE.ArrowHelper(
      new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0.3, 0xe14b4b, 0.08, 0.045);
    this.arrow.visible = false;
    robot.root.add(this.arrow); // MuJoCo coords

    this._ndc = new THREE.Vector2();
    this._plane = new THREE.Plane();
    this._tmp = new THREE.Vector3();
    this._grabWorld = new THREE.Vector3();

    domEl.addEventListener('pointerdown', (e) => this.#onDown(e));
    domEl.addEventListener('pointermove', (e) => this.#onMove(e));
    window.addEventListener('pointerup', (e) => this.#onUp(e));
    this.domEl = domEl;
  }

  #setNdc(e) {
    const r = this.domEl.getBoundingClientRect();
    this._ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }

  #bodyPose(bodyId) {
    const xpos = this.data.xpos, xquat = this.data.xquat;
    const p = new THREE.Vector3(xpos[3 * bodyId], xpos[3 * bodyId + 1], xpos[3 * bodyId + 2]);
    // MuJoCo quat is (w,x,y,z); THREE.Quaternion is (x,y,z,w)
    const q = new THREE.Quaternion(
      xquat[4 * bodyId + 1], xquat[4 * bodyId + 2], xquat[4 * bodyId + 3], xquat[4 * bodyId]);
    return { p, q };
  }

  #onDown(e) {
    if (e.button !== 0) return;
    this.#setNdc(e);
    this.raycaster.setFromCamera(this._ndc, this.camera);
    const hits = this.raycaster.intersectObjects(this.robot.meshes, false);
    if (hits.length === 0) return; // empty space -> OrbitControls handles it

    const hit = hits[0];
    const bodyId = hit.object.userData.bodyId;
    // Store grab point in the body frame so it rides along with the body.
    const pointMj = this.robot.root.worldToLocal(hit.point.clone());
    const { p, q } = this.#bodyPose(bodyId);
    const localPoint = pointMj.clone().sub(p).applyQuaternion(q.clone().invert());
    this.active = { bodyId, localPoint, bodyName: hit.object.userData.bodyName };

    // Drag plane: through the grab point, facing the camera.
    this._grabWorld.copy(hit.point);
    const n = this.camera.getWorldDirection(new THREE.Vector3());
    this._plane.setFromNormalAndCoplanarPoint(n, this._grabWorld);

    this.controls.enabled = false;
    this.forceMj.set(0, 0, 0);
    e.preventDefault();
  }

  #onMove(e) {
    if (!this.active) return;
    this.#setNdc(e);
    this.raycaster.setFromCamera(this._ndc, this.camera);
    if (this.raycaster.ray.intersectPlane(this._plane, this._tmp)) {
      // Camera-plane drag vector (world) -> MuJoCo coords via root rotation.
      const dragWorld = this._tmp.clone().sub(this._grabWorld);
      const q = this.robot.root.getWorldQuaternion(new THREE.Quaternion()).invert();
      const dragMj = dragWorld.applyQuaternion(q);
      this.forceMj.copy(dragMj.multiplyScalar(FORCE_PER_METER));
      if (this.forceMj.length() > FORCE_MAX) this.forceMj.setLength(FORCE_MAX);
    }
    if (this.labelEl) {
      this.labelEl.style.display = 'block';
      this.labelEl.style.left = `${e.clientX + 14}px`;
      this.labelEl.style.top = `${e.clientY - 10}px`;
      this.labelEl.textContent = `applied: ${this.forceMj.length().toFixed(1)} N`;
    }
  }

  #onUp() {
    if (!this.active) return;
    this.active = null;
    this.forceMj.set(0, 0, 0);
    this.arrow.visible = false;
    if (this.labelEl) this.labelEl.style.display = 'none';
    this.controls.enabled = true;
  }

  /**
   * Called every physics step. Returns the current application, or null.
   * @returns {null | {bodyId: number, bodyName: string, point: THREE.Vector3, force: THREE.Vector3}}
   *          point/force in MuJoCo world coordinates.
   */
  current() {
    if (!this.active) return null;
    const { p, q } = this.#bodyPose(this.active.bodyId);
    this.pointMj.copy(this.active.localPoint).applyQuaternion(q).add(p);
    return {
      bodyId: this.active.bodyId,
      bodyName: this.active.bodyName,
      point: this.pointMj,
      force: this.forceMj,
    };
  }

  /** Update the red arrow (call once per rendered frame). */
  updateArrow() {
    const cur = this.current();
    const mag = cur ? cur.force.length() : 0;
    if (!cur || mag < 1e-3) { this.arrow.visible = false; return; }
    this.arrow.visible = true;
    this.arrow.position.copy(cur.point);
    this.arrow.setDirection(this._tmp.copy(cur.force).normalize());
    this.arrow.setLength(0.15 + 0.5 * (mag / FORCE_MAX), 0.07, 0.04);
  }
}
