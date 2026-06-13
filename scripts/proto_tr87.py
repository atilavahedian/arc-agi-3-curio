"""Offline prototype of the cursor+editor perception for tr87.

Drives the real engine, captures frames, and tests:
  1. rotation-canonical card signatures are collision-free per family
  2. ring-test card detection finds exactly the rule/input/answer cards
  3. sequence grouping + separator-gap rule pairing
  4. goal inference (greedy/DFS segmentation) for level 1
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arc_agi import OperationMode

GRID = 64


def canon_window(grid, x, y, w, h):
    """Rotation-canonical tuple of a wxh window (assumes w == h)."""
    pat = [[grid[y + dy][x + dx] for dx in range(w)] for dy in range(h)]
    best = None
    for _ in range(4):
        t = tuple(v for row in pat for v in row)
        if best is None or t < best:
            best = t
        pat = [list(r) for r in zip(*pat[::-1])]  # rotate 90
    return best


def background_of(grid):
    c = Counter()
    for row in grid:
        c.update(row)
    return c.most_common(1)[0][0]


def detect_cards(grid, pitch, background):
    """Positions whose pitch x pitch window has a monochrome non-background
    border ring and a non-uniform interior."""
    out = []
    p = pitch
    for y in range(GRID - p + 1):
        for x in range(GRID - p + 1):
            ring = grid[y][x]
            if ring == background:
                continue
            ok = True
            for dx in range(p):
                if grid[y][x + dx] != ring or grid[y + p - 1][x + dx] != ring:
                    ok = False
                    break
            if ok:
                for dy in range(1, p - 1):
                    if grid[y + dy][x] != ring or grid[y + dy][x + p - 1] != ring:
                        ok = False
                        break
            if not ok:
                continue
            if all(grid[y + dy][x + dx] == ring
                   for dy in range(1, p - 1) for dx in range(1, p - 1)):
                continue  # uniform interior: not a card
            out.append((x, y, ring, canon_window(grid, x, y, p, p)))
    return out


def sequences(cards, pitch):
    """Group cards into horizontal pitch-adjacent runs."""
    by_y = defaultdict(list)
    for x, y, ring, sig in cards:
        by_y[y].append((x, ring, sig))
    seqs = []
    for y, row in by_y.items():
        row.sort()
        run = [row[0]]
        for item in row[1:]:
            if item[0] == run[-1][0] + pitch:
                run.append(item)
            else:
                seqs.append((y, run))
                run = [item]
        seqs.append((y, run))
    return seqs


def main():
    from arcengine import GameAction
    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arc.make("tr87")
    env.reset()
    frame = env.step(GameAction.RESET)
    grid = frame.frame[-1]
    bg = background_of(grid)
    print("background:", bg)

    # 1 — collision check: canonical sigs of all cards on screen
    cards = detect_cards(grid, 7, bg)
    print(f"cards detected: {len(cards)}")
    for x, y, ring, sig in sorted(cards, key=lambda c: (c[1], c[0])):
        print(f"  ({x:2},{y:2}) ring={ring:2} sig={hash(sig) & 0xffff:5}")

    seqs = sequences(cards, 7)
    print("\nsequences:")
    for y, run in sorted(seqs):
        print(f"  y={y:2} xs={[c[0] for c in run]} rings={[c[1] for c in run]}")

    # rule pairing: same y, gap region non-uniform
    print("\ngap regions:")
    for y, run in sorted(seqs):
        pass
    by_y = defaultdict(list)
    for y, run in seqs:
        by_y[y].append(run)
    for y, runs in sorted(by_y.items()):
        runs.sort(key=lambda r: r[0][0])
        for a, b in zip(runs, runs[1:]):
            x0 = a[-1][0] + 7
            x1 = b[0][0]
            region = {grid[y + dy][x] for dy in range(7) for x in range(x0, x1)}
            print(f"  y={y:2} gap x[{x0},{x1}) colors={sorted(region)} "
                  f"-> {'RULE-PAIR' if len(region) > 1 else 'separate'}")

    # 2 — press ACTION1 8 times: mutate diffs at slot 0, ring closure
    print("\npressing ACTION1 x8 (cycle):")
    prev = grid
    sigs_seen = []
    for i in range(8):
        frame = env.step(GameAction.ACTION1)
        cur = frame.frame[-1]
        diff = [(x, y) for y in range(GRID) for x in range(GRID)
                if prev[y][x] != cur[y][x]]
        core = [(x, y) for x, y in diff if 3 <= x < 61 and 3 <= y < 61]
        if core:
            xs = [x for x, _ in core]
            ys = [y for _, y in core]
            print(f"  press {i}: interior diff bbox=({min(xs)},{min(ys)})-"
                  f"({max(xs)},{max(ys)}) n={len(core)} "
                  f"(band diff n={len(diff) - len(core)})")
        else:
            print(f"  press {i}: no interior diff (total diff {len(diff)})")
        prev = cur
    # press ACTION4 6 times: cursor moves
    print("\npressing ACTION4 x6 (selector):")
    for i in range(6):
        frame = env.step(GameAction.ACTION4)
        cur = frame.frame[-1]
        diff = [(x, y) for y in range(GRID) for x in range(GRID)
                if prev[y][x] != cur[y][x]]
        core = [(x, y) for x, y in diff if 3 <= x < 61 and 3 <= y < 61]
        xs = [x for x, _ in core] or [0]
        ys = [y for _, y in core] or [0]
        print(f"  press {i}: interior diff n={len(core)} "
              f"bbox=({min(xs)},{min(ys)})-({max(xs)},{max(ys)})")
        prev = cur


if __name__ == "__main__":
    main()
