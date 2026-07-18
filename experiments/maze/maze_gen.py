"""Grid-maze generator -> wall boxes (and MuJoCo XML snippet).

Recursive-backtracker over an R x C cell grid. Output is a list of axis-aligned
wall boxes in world meters, plus start/goal cell centers. The maze is *simply
connected* (no islands) so left/right-hand wall-following provably escapes —
exactly the Gate-1 setting from docs/blind_maze_research_and_method.md.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field


@dataclass
class MazeSpec:
    rows: int = 4
    cols: int = 4
    cell: float = 2.0          # cell size (m) — corridor width
    wall_t: float = 0.3        # wall thickness (m) — thin walls get punched through
                               # by step impacts at speed (MuJoCo soft contact)
    wall_h: float = 1.6        # wall height (m) — above G1 shoulders so arms/torso engage
    seed: int = 0
    boxes: list = field(default_factory=list)   # (cx, cy, hx, hy) half-sizes
    start_xy: tuple = (0.0, 0.0)
    goal_xy: tuple = (0.0, 0.0)
    exit_side: str = "E"


def generate(rows=4, cols=4, cell=2.0, wall_t=0.3, wall_h=1.6, seed=0) -> MazeSpec:
    rng = random.Random(seed)
    # walls[r][c] = {N,S,E,W} closed
    closed = [[{"N", "S", "E", "W"} for _ in range(cols)] for _ in range(rows)]
    # recursive backtracker
    stack = [(0, 0)]
    seen = {(0, 0)}
    D = {"N": (1, 0, "S"), "S": (-1, 0, "N"), "E": (0, 1, "W"), "W": (0, -1, "E")}
    while stack:
        r, c = stack[-1]
        nbrs = []
        for d, (dr, dc, opp) in D.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:
                nbrs.append((d, nr, nc, opp))
        if not nbrs:
            stack.pop()
            continue
        d, nr, nc, opp = rng.choice(nbrs)
        closed[r][c].discard(d)
        closed[nr][nc].discard(opp)
        seen.add((nr, nc))
        stack.append((nr, nc))

    # exit: open the East wall of the last cell (top-right area)
    er, ec = rows - 1, cols - 1
    closed[er][ec].discard("E")

    spec = MazeSpec(rows, cols, cell, wall_t, wall_h, seed)
    ht = wall_t / 2

    def add(cx, cy, hx, hy):
        spec.boxes.append((cx, cy, hx, hy))

    # build unique wall segments: south & west of every cell + north of top row
    # + east of right col (skipping opened exit)
    for r in range(rows):
        for c in range(cols):
            x0, y0 = c * cell, r * cell          # cell SW corner
            if "S" in closed[r][c]:
                add(x0 + cell / 2, y0, cell / 2 + ht, ht)
            if "W" in closed[r][c]:
                add(x0, y0 + cell / 2, ht, cell / 2 + ht)
            if r == rows - 1 and "N" in closed[r][c]:
                add(x0 + cell / 2, y0 + cell, cell / 2 + ht, ht)
            if c == cols - 1 and "E" in closed[r][c]:
                add(x0 + cell, y0 + cell / 2, ht, cell / 2 + ht)

    spec.start_xy = (cell / 2, cell / 2)                      # cell (0,0) center
    spec.goal_xy = ((cols - 1) * cell + cell / 2, (rows - 1) * cell + cell / 2)
    # goal region: past the opened E wall of the exit cell
    spec.exit_xy = ((cols) * cell + 0.6, (rows - 1) * cell + cell / 2)
    return spec


def to_mjcf_geoms(spec: MazeSpec, name_prefix="maze_wall") -> str:
    """MuJoCo <geom> lines for a worldbody (static boxes)."""
    lines = []
    for i, (cx, cy, hx, hy) in enumerate(spec.boxes):
        lines.append(
            f'<geom name="{name_prefix}_{i}" type="box" '
            f'pos="{cx:.3f} {cy:.3f} {spec.wall_h / 2:.3f}" '
            f'size="{hx:.3f} {hy:.3f} {spec.wall_h / 2:.3f}" '
            f'rgba="0.5 0.45 0.4 1" contype="1" conaffinity="1"/>'
        )
    return "\n".join(lines)


if __name__ == "__main__":
    s = generate(4, 4, seed=1)
    print(f"{len(s.boxes)} walls; start={s.start_xy} goal={s.goal_xy} exit={s.exit_xy}")
    print(to_mjcf_geoms(s)[:300], "...")
