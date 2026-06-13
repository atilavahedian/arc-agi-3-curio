"""Probe dc22: click the REAL buttons (panel components) and dump masks.

Usage: .venv/bin/python scripts/probe_dc22b.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction

logging.basicConfig(level=logging.WARNING)

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
env = arc.make("dc22")
frame = env.reset()


def grid_of(fr):
    return fr.frame[-1]


def diff(a, b):
    return [(x, y) for y in range(64) for x in range(64) if a[y][x] != b[y][x]]


def act6(x, y):
    a = GameAction.ACTION6
    a.set_data({"x": x, "y": y})
    return env.step(a)


g0 = grid_of(frame)

# the two buttons sit in the right panel: color-9 cells with x >= 40, and
# color-8 cells with x >= 40
for color, label in ((9, "blrmbx"), (8, "refgps")):
    cells = [(x, y) for y in range(64) for x in range(64)
             if g0[y][x] == color and x >= 40]
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    print(f"button {label} color={color}: bbox=({min(xs)},{min(ys)})-({max(xs)},{max(ys)})")

prev = g0
for rep in range(3):
    for color, label in ((9, "blrmbx"), (8, "refgps")):
        cells = [(x, y) for y in range(64) for x in range(64)
                 if prev[y][x] == color and x >= 40]
        cx = sum(x for x, _ in cells) // len(cells)
        cy = sum(y for _, y in cells) // len(cells)
        frame = act6(cx, cy)
        g = grid_of(frame)
        d = diff(prev, g)
        interior = sorted(c for c in d if 3 <= c[0] < 61 and 3 <= c[1] < 61)
        print(f"click {label}@({cx},{cy}): nframes={len(frame.frame)} "
              f"interior_changed={len(interior)}")
        if interior:
            xs = [x for x, _ in interior]
            ys = [y for _, y in interior]
            print(f"    bbox=({min(xs)},{min(ys)})-({max(xs)},{max(ys)})  cells={interior[:20]}...")
        prev = g

# now print the bridge area around x=8-26, y=20-44 in both states
print("\nboard after toggles (rows 18..44, cols 4..30):")
for y in range(18, 45):
    print("   ", "".join(f"{prev[y][x]:X}" for x in range(4, 31)))
