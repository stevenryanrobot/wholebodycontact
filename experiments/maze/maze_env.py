"""mjlab maze environment for blind contact-only navigation on Unitree G1.

Builds Mjlab-Velocity-Flat-Unitree-G1 (play cfg) with:
  * maze walls injected via SceneCfg.spec_fn (boxes from maze_gen, shifted so the
    start cell is at the origin where the robot spawns),
  * a whole-body ContactSensor against wall geoms only (secondary = geom pattern
    "maze_wall_.*", so foot-ground contact never pollutes the labels),
  * per-step velocity-command override (twist.vel_command_b) for navigation.

GT extraction: ContactSensor force is global-frame, primary->secondary
(robot -> wall), i.e. it IS the obstacle bearing. World azimuth -> 12 x 30-deg
world bins (pledge.world_bin_of); the controller debounces in world frame.
"""
from __future__ import annotations
import math
import os
import sys
import torch
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.maze.maze_gen import generate, MazeSpec
from experiments.maze.pledge import sector_of

# body-level primaries for wall contact (arms, trunk, legs — feet excluded on purpose:
# feet hitting a wall reads through ankle links' knee/hip anyway, and we avoid slots blowup)
WALL_PRIMARY_PATTERN = (
    r"^(pelvis|torso_link|.*shoulder_(pitch|roll|yaw)_link|.*elbow_link|"
    r".*wrist_(roll|pitch|yaw)_link|.*hip_(pitch|roll|yaw)_link|.*knee_link)$"
)


def make_maze_spec_fn(spec_boxes, wall_h=1.6):
    def _fn(spec: mujoco.MjSpec) -> None:
        for i, (cx, cy, hx, hy) in enumerate(spec_boxes):
            g = spec.worldbody.add_geom()
            g.name = f"maze_wall_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size[:] = [hx, hy, wall_h / 2]
            g.pos[:] = [cx, cy, wall_h / 2]
            g.contype = 1
            g.conaffinity = 1
            g.rgba[:] = [0.55, 0.5, 0.45, 1.0]
    return _fn


def shifted_maze(rows=4, cols=4, cell=2.0, seed=0):
    """Maze with the start-cell center at the origin (robot spawn)."""
    spec = generate(rows, cols, cell=cell, seed=seed)
    sx, sy = spec.start_xy
    spec.boxes = [(cx - sx, cy - sy, hx, hy) for (cx, cy, hx, hy) in spec.boxes]
    spec.goal_xy = (spec.goal_xy[0] - sx, spec.goal_xy[1] - sy)
    spec.exit_xy = (spec.exit_xy[0] - sx, spec.exit_xy[1] - sy)
    spec.start_xy = (0.0, 0.0)
    return spec


def build_env(maze: MazeSpec, num_envs=1, device="cuda:0", episode_s=None, render_mode=None,
              topdown=False):
    import mjlab.tasks  # noqa: F401  (populate registry)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from mjlab.sensor import ContactSensorCfg, ContactMatch
    from mjlab.viewer.viewer_config import ViewerConfig

    task = "Mjlab-Velocity-Flat-Unitree-G1"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs
    cfg.scene.spec_fn = make_maze_spec_fn(maze.boxes)
    if episode_s is not None:
        cfg.episode_length_s = episode_s

    if topdown:
        # fixed god's-eye camera: whole maze in frame, no body tracking
        span_x = maze.cols * maze.cell
        span_y = maze.rows * maze.cell
        cx = maze.cell * (maze.cols - 1) / 2.0     # interior center (shifted frame)
        cy = maze.cell * (maze.rows - 1) / 2.0
        cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
        cfg.viewer.lookat = (cx + 0.8, cy, 0.0)    # nudge east so the exit is visible
        cfg.viewer.elevation = -90.0               # straight down
        cfg.viewer.azimuth = 90.0
        cfg.viewer.distance = max(span_x, span_y) * 1.4 + 2.0
        cfg.viewer.width = 1280
        cfg.viewer.height = 960

    wall_contact = ContactSensorCfg(
        name="wall_contact",
        primary=ContactMatch(mode="body", pattern=WALL_PRIMARY_PATTERN, entity="robot"),
        secondary=ContactMatch(mode="geom", pattern=r"^maze_wall_\d+$"),
        secondary_policy="any",
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (wall_contact,)

    # navigation drives commands; make random resampling inert
    cfg.commands["twist"].resampling_time_range = (1.0e6, 1.0e6)

    env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=render_mode)
    env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task).clip_actions)
    return env


def load_policy(env, checkpoint, device="cuda:0"):
    from dataclasses import asdict
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    task = "Mjlab-Velocity-Flat-Unitree-G1"
    runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(load_rl_cfg(task)), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device)


class MazeInterface:
    """Runtime handles: command override, GT sectors, pose, success check."""

    def __init__(self, env, maze: MazeSpec, device="cuda:0"):
        self.raw = env.unwrapped
        self.maze = maze
        self.device = device
        self.robot = self.raw.scene["robot"]
        self.twist = self.raw.command_manager.get_term("twist")
        self.sensor = self.raw.scene.sensors["wall_contact"]
        self.n_primaries = None      # resolved lazily from first read

    # ---- commands ----
    def set_cmd(self, vx, vy, wz, env_ids=None):
        cmd = self.twist.vel_command_b
        if env_ids is None:
            cmd[:, 0] = vx; cmd[:, 1] = vy; cmd[:, 2] = wz
        else:
            cmd[env_ids, 0] = vx; cmd[env_ids, 1] = vy; cmd[env_ids, 2] = wz

    def set_cmd_batch(self, cmds: torch.Tensor):
        self.twist.vel_command_b[:] = cmds

    # ---- pose ----
    def pose2d(self):
        """[N,3] x, y, yaw (world)."""
        pose = self.robot.data.root_link_pose_w        # [N,7] pos+quat(wxyz)
        x, y = pose[:, 0], pose[:, 1]
        qw, qx, qy, qz = pose[:, 3], pose[:, 4], pose[:, 5], pose[:, 6]
        yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return torch.stack([x, y, yaw], dim=-1)

    # ---- GT contact bins (WORLD frame; controller rotates to robot frame) ----
    def gt_sectors(self, force_thresh=3.0):
        """[N,12] bool — wall-contact per WORLD-frame 30° azimuth bin (force > thresh N).
        Force is global-frame primary->secondary, i.e. robot->wall = obstacle bearing.
        Debouncing/latching happens in the world frame inside PledgeController
        (walls don't rotate when the robot does)."""
        cd = self.sensor.data
        found = cd.found                               # [N,P]
        force = cd.force                               # [N,P,3]
        N, P = found.shape
        sec = torch.zeros(N, 12, dtype=torch.bool, device=found.device)   # 12 world bins
        fmag = force[..., :2].norm(dim=-1)             # planar magnitude [N,P]
        active = (found > 0) & (fmag > force_thresh)
        if active.any():
            az_w = torch.atan2(force[..., 1], force[..., 0])       # [N,P] world
            deg = torch.rad2deg(az_w) % 360.0
            idx = ((deg + 15.0) % 360.0 // 30.0).long()            # [N,P] 30° bins
            ii, jj = torch.nonzero(active, as_tuple=True)
            sec[ii, idx[ii, jj]] = True
        return sec

    def success(self):
        pose = self.pose2d()
        ex, ey = self.maze.exit_xy
        return (pose[:, 0] > ex - 0.4) & ((pose[:, 1] - ey).abs() < 1.2)

    def fell(self):
        # pelvis height low = fallen
        h = self.robot.data.root_link_pose_w[:, 2]
        return h < 0.35
