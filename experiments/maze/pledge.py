"""Pledge-wrapped reactive wall-following on contact azimuth bins.

Simulator-agnostic: consumes 12 raw WORLD-frame 30-deg azimuth bins + yaw,
debounces in the world frame, rotates to 6 robot-frame sectors internally,
and emits (vx, vy, wz) velocity commands. Design per
docs/blind_maze_research_and_method.md §3 (k-of-n debounce, EMA, Pledge
counter with heading tolerance; left-hand wall following).

Sectors (robot frame azimuth, deg):  0=FRONT(-30..30) 1=FL(30..90)
2=BL(90..150) 3=BACK(150..210) 4=BR(210..270) 5=FR(270..330).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

FRONT, FL, BL, BACK, BR, FR = range(6)


def sector_of(azimuth_rad: float) -> int:
    """Robot-frame azimuth (atan2(fy,fx) of contact direction robot->obstacle)."""
    a = math.degrees(azimuth_rad) % 360.0
    return int(((a + 30.0) % 360.0) // 60.0)


N_WORLD_BINS = 12          # 30° world-frame bins: axis-aligned wall bearings
                           # (0/90/180/270°) fall on bin CENTERS, not edges

def world_bin_of(azimuth_rad: float) -> int:
    a = math.degrees(azimuth_rad) % 360.0
    return int(((a + 15.0) % 360.0) // 30.0)


@dataclass
class PledgeCfg:
    v_walk: float = 0.7          # m/s forward (tracked ~70% by the policy)
    v_slow: float = 0.3
    w_turn: float = 0.7          # rad/s in-place turn (cmd range max; tracked ~64%)
    w_veer: float = 0.35         # rad/s while following
    k_on: int = 3                # frames to confirm contact  (15%FA -> ~0.3%)
    k_off: int = 10              # frames to confirm wall lost
    # Pledge departure: only needed for non-tree mazes / approach-from-outside.
    # In a PERFECT maze every wall connects to the outer boundary, so pure
    # left-hand following (never depart) is complete — keep departure disabled
    # (yaw_tol=0) for Gate-1 simply-connected mazes.
    yaw_tol: float = 0.0
    stuck_window: float = 4.0    # s without progress -> STUCK
    stuck_dist: float = 0.15     # m
    dt: float = 0.02             # control period (s)
    seek_curve_w: float = 0.45   # curve-left rate when re-finding wall
    lost_limit: float = 8.0      # s of no contact while FOLLOW -> give up, EXPLORE
                                 # (must exceed outer-corner wrap time ~(π/2)/seek_curve_w≈3.5s)
    follow_limit: float = 90.0   # s of continuous FOLLOW -> random-kick (RAMBLER-style
                                 # stochastic departure; breaks wall-switch cycles)
    kick_max: float = math.radians(75.0)  # random re-heading magnitude on kick
    seed: int = 0


class Debounce:
    """k-of-n persistence per sector."""
    def __init__(self, k_on, k_off, n_sectors=6):
        self.k_on, self.k_off = k_on, k_off
        self.cnt_on = [0] * n_sectors
        self.cnt_off = [0] * n_sectors
        self.state = [False] * n_sectors

    def update(self, raw):
        for i, r in enumerate(raw):
            if r:
                self.cnt_on[i] += 1; self.cnt_off[i] = 0
                if self.cnt_on[i] >= self.k_on:
                    self.state[i] = True
            else:
                self.cnt_off[i] += 1; self.cnt_on[i] = 0
                if self.cnt_off[i] >= self.k_off:
                    self.state[i] = False
        return list(self.state)


class PledgeController:
    """States: EXPLORE -> (contact) ACQUIRE -> FOLLOW(left wall) -> DEPART -> EXPLORE.
    Pledge invariant: leave the wall only when cumulative turn ≈ 0 AND heading ≈ φ0."""

    def __init__(self, cfg: PledgeCfg = None):
        import random as _random
        self.cfg = cfg or PledgeCfg()
        self.deb = Debounce(self.cfg.k_on, self.cfg.k_off, n_sectors=N_WORLD_BINS)
        self.state = "EXPLORE"
        self.phi0 = None             # preferred heading (set at reset)
        self.turn_acc = 0.0          # Pledge counter (rad, integrated yaw change while on wall)
        self.prev_yaw = None
        self.last_pos = None
        self.progress_t = 0.0
        self.stuck_t = 0.0
        self.lost_t = 0.0
        self.follow_t = 0.0
        self.acquire_target = 0.0
        self.rng = _random.Random(self.cfg.seed)
        self.log = []

    def reset(self, yaw, pos):
        self.__init__(self.cfg)
        self.phi0 = yaw
        self.prev_yaw = yaw
        self.last_pos = tuple(pos)

    # ---------------------------------------------------------------- step
    def step(self, raw_world_bins, yaw, pos):
        """raw_world_bins: N_WORLD_BINS(12) bools — obstacle azimuth bins in the
        WORLD frame (bin b covers b*30deg +/-15deg; axis-aligned walls land on
        centers). Debounce happens in the world frame (walls don't rotate when
        the robot does); flags are rotated into the 6 robot-frame sectors
        afterwards. Returns (vx, vy, wz, state)."""
        c = self.cfg
        w = self.deb.update(raw_world_bins)
        s = [False] * 6
        for b, v in enumerate(w):
            if v:
                s[sector_of(math.radians(30.0 * b) - yaw)] = True
        front, fl, bl, back, br, fr = s

        # yaw bookkeeping (wrap-safe)
        dyaw = _wrap(yaw - self.prev_yaw)
        self.prev_yaw = yaw
        if self.state in ("ACQUIRE", "FOLLOW", "DEPART"):
            self.turn_acc += dyaw

        # stuck watchdog (any state): no displacement for stuck_window
        self.progress_t += c.dt
        if _dist(pos, self.last_pos) > c.stuck_dist:
            self.last_pos = tuple(pos); self.progress_t = 0.0
        stuck = self.progress_t > c.stuck_window

        if self.state == "EXPLORE":
            if front or fl or fr:
                # remember where the wall is and turn (open-loop, yaw-servo) until
                # it sits on our LEFT: target = wall_bearing - 90deg. Contact may
                # be lost during the turn (thin shell / unpinned humanoid) — we do
                # NOT depend on it.
                bins_active = [b for b, v in enumerate(w) if v]
                nearest = min(bins_active,
                              key=lambda b: abs(_wrap(math.radians(30.0 * b) - yaw)))
                wall_bearing_w = math.radians(30.0 * nearest)
                self.acquire_target = _wrap(wall_bearing_w - math.pi / 2)
                self.state = "ACQUIRE"; self.turn_acc = 0.0
            elif stuck:
                self.state = "STUCK"; self.stuck_t = 0.0
            else:
                # steer back toward preferred heading phi0
                err = _wrap(self.phi0 - yaw)
                return (c.v_walk, 0.0, _clip(err, c.w_veer), "EXPLORE")

        if self.state == "ACQUIRE":
            if abs(_wrap(yaw - self.acquire_target)) < math.radians(12.0):
                self.state = "FOLLOW"
            elif stuck:
                self.state = "STUCK"; self.stuck_t = 0.0
            else:
                # pure in-place turn: backing up while wedged against the wall was
                # the #1 fall cause (all batch falls happened in ACQUIRE)
                return (0.0, 0.0, -c.w_turn, "ACQUIRE")

        if self.state == "FOLLOW":
            touching = front or fl or bl
            self.lost_t = 0.0 if touching else self.lost_t + c.dt
            self.follow_t += c.dt
            # RAMBLER-style stochastic departure: too long on walls without an exit
            # -> random re-heading, breaks wall-switch cycles a pure follower can enter
            if self.follow_t > c.follow_limit:
                self.phi0 = _wrap(self.phi0 + self.rng.uniform(-c.kick_max, c.kick_max))
                self.state = "EXPLORE"; self.turn_acc = 0.0
                self.follow_t = 0.0; self.lost_t = 0.0
            # Pledge departure test
            elif abs(self.turn_acc) < c.yaw_tol and abs(_wrap(yaw - self.phi0)) < c.yaw_tol \
                    and not (front or fl):
                self.state = "DEPART"
            elif self.lost_t > c.lost_limit:
                # wall truly lost (open junction): give up following, head back to phi0
                self.state = "EXPLORE"; self.turn_acc = 0.0; self.lost_t = 0.0
            elif stuck:
                self.state = "STUCK"; self.stuck_t = 0.0
            elif front:
                return (0.0, 0.0, -c.w_turn, "FOLLOW")        # inner corner: turn right
            elif fl or bl:
                # touching left wall: slide forward, veer slightly away
                return (c.v_walk, 0.0, -0.15 * c.w_veer, "FOLLOW")
            else:
                # lost the wall: curve LEFT to wrap around the corner (left-hand rule)
                return (c.v_slow, 0.0, c.seek_curve_w, "FOLLOW")

        if self.state == "DEPART":
            # walk straight along phi0 for a short burst, then EXPLORE
            if front or fl or fr:
                self.state = "ACQUIRE"; self.turn_acc = 0.0
            else:
                err = _wrap(self.phi0 - yaw)
                self.state = "EXPLORE"
                return (c.v_walk, 0.0, _clip(err, c.w_veer), "DEPART")

        if self.state == "STUCK":
            # back up + turn right for ~1.5 s, then explore again
            self.stuck_t += c.dt
            if self.stuck_t > 1.5:
                self.state = "EXPLORE"; self.progress_t = 0.0
            return (-c.v_slow, 0.0, -c.w_turn, "STUCK")

        return (0.0, 0.0, 0.0, self.state)


def _wrap(a):
    while a > math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def _clip(x, m):
    return max(-m, min(m, x))


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
