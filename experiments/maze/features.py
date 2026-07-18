"""Shared proprioception features + labels for the whole-body contact sensor.

Features (per env, per 50Hz control step) — deployable set, NO commands/actions
(SixthSense: exclude controller-specific inputs), NO base position/yaw:
    joint_pos (29) | joint_vel (29) | actuator torque (29) |
    imu gyro (3)   | imu upvector = projected gravity (3)  | imu lin acc (3)
    => D = 96

Labels from the wall_contact ContactSensor (net over all contacting bodies):
    contact (bool) | robot-frame azimuth az_r (rad) | planar force magnitude (N)
The sensor predicts ROBOT-frame azimuth (deployable); navigation adds IMU yaw
to get world bins.
"""
from __future__ import annotations
import torch
import mujoco

FEAT_DIM = 96


class FeatureExtractor:
    def __init__(self, raw_env):
        self.raw = raw_env
        self.robot = raw_env.scene["robot"]
        mjm = raw_env.sim.mj_model
        self._adr = {}
        for name in ("imu_ang_vel", "imu_upvector", "imu_lin_acc"):
            s = mjm.sensor(f"robot/{name}")          # entity-namespaced in the scene spec
            self._adr[name] = (int(s.adr[0]), int(s.dim[0]))

    def _sens(self, name):
        a, d = self._adr[name]
        return self.raw.sim.data.sensordata[:, a:a + d]

    def proprio(self) -> torch.Tensor:
        """[N, 96] float32."""
        # NOTE: sim.data fields are WarpBridge proxies whose __torch_function__
        # does not unwrap inside list args (torch.cat recursion) — index with
        # [:] to materialize a plain torch tensor first.
        jp = self.robot.data.joint_pos[:]         # [N,29]
        jv = self.robot.data.joint_vel[:]         # [N,29]
        tau = self.raw.sim.data.actuator_force[:]  # [N,29]
        gyro = self._sens("imu_ang_vel")          # [N,3] (sliced -> plain)
        upv = self._sens("imu_upvector")          # [N,3]
        acc = self._sens("imu_lin_acc")           # [N,3]
        return torch.cat([jp, jv, tau, gyro, upv, acc], dim=-1).float()


def contact_label(mi, force_thresh=3.0):
    """Net wall-contact label per env from the GT sensor.
    Returns (contact [N] bool, az_r [N] robot-frame rad, mag [N] Newtons)."""
    cd = mi.sensor.data
    found = cd.found                              # [N,P]
    force = cd.force                              # [N,P,3] world, robot->wall
    f2 = force[..., :2]
    active = (found > 0) & (f2.norm(dim=-1) > 0.5)     # any real touch on the body
    fnet = torch.where(active.unsqueeze(-1), f2, torch.zeros_like(f2)).sum(dim=1)  # [N,2]
    mag = fnet.norm(dim=-1)                       # [N]
    contact = mag > force_thresh
    yaw = mi.pose2d()[:, 2]
    az_w = torch.atan2(fnet[:, 1], fnet[:, 0])
    az_r = torch.atan2(torch.sin(az_w - yaw), torch.cos(az_w - yaw))
    return contact, az_r, mag
