"""Empirical probe of tr87 frames: scripted actions, diff dumps."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("tr87")
f = env.reset()
def g(fr): return fr.frame[-1]
def show(grid, y0=0, y1=64):
    cs = "0123456789abcdef"
    for y in range(y0, y1):
        print("".join(cs[v % 16] for v in grid[y]))
def diff(a, b):
    out = []
    for y in range(64):
        for x in range(64):
            if a[y][x] != b[y][x]:
                out.append((x, y, a[y][x], b[y][x]))
    return out

grid = g(f)
print("=== initial frame ===")
show(grid)
prev = grid
import collections
for name, act in [("A1", GameAction.ACTION1), ("A1", GameAction.ACTION1),
                  ("A2", GameAction.ACTION2),
                  ("A3", GameAction.ACTION3), ("A4", GameAction.ACTION4),
                  ("A4", GameAction.ACTION4)]:
    f = env.step(act)
    grid = g(f)
    d = diff(prev, grid)
    xs = [c[0] for c in d]; ys = [c[1] for c in d]
    print(f"--- {name}: {len(d)} px changed, bbox x[{min(xs)},{max(xs)}] y[{min(ys)},{max(ys)}], frames={len(f.frame)}, avail={f.available_actions}")
    # group changed cells by row-region
    byrow = collections.Counter(y for _,y,_,_ in d)
    print("   rows:", sorted(byrow.items()))
    prev = grid
print("=== after probes ===")
show(grid)
