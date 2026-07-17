"""Black-box opening-frame and one-step causal probe for an official game.

The probe uses only rendered frames and advertised actions.  It reports the
opening visual grammar, then resets before each action trial so effects are
comparable.  No environment implementation details are inspected.

Usage:
    .venv/bin/python scripts/probe_opening.py --game sk48 --ascii \
        --crop 8,8,50,44 --sequence 4,4,1,7
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction


Grid = list[list[int]]
Cell = tuple[int, int]


def grid_of(frame) -> Grid:
    return [list(map(int, row)) for row in frame.frame[-1]]


def components(grid: Grid) -> tuple[int, list[dict]]:
    counts = Counter(value for row in grid for value in row)
    background = counts.most_common(1)[0][0]
    height, width = len(grid), len(grid[0])
    seen: set[Cell] = set()
    found: list[dict] = []
    for y in range(height):
        for x in range(width):
            if (x, y) in seen or grid[y][x] == background:
                continue
            color = grid[y][x]
            cells: set[Cell] = {(x, y)}
            queue = deque([(x, y)])
            seen.add((x, y))
            while queue:
                cx, cy = queue.popleft()
                for nx, ny in ((cx + 1, cy), (cx - 1, cy),
                               (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height \
                            and (nx, ny) not in seen \
                            and grid[ny][nx] == color:
                        seen.add((nx, ny))
                        cells.add((nx, ny))
                        queue.append((nx, ny))
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            found.append({
                "color": color,
                "size": len(cells),
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "center": (round(sum(xs) / len(xs)),
                           round(sum(ys) / len(ys))),
            })
    found.sort(key=lambda rec: (-rec["size"], rec["color"], rec["bbox"]))
    return background, found


def diff_summary(before: Grid, after: Grid) -> str:
    changed = [(x, y) for y, row in enumerate(before)
               for x, value in enumerate(row) if value != after[y][x]]
    if not changed:
        return "changed=0"
    xs = [x for x, _y in changed]
    ys = [y for _x, y in changed]
    transitions = Counter(
        (before[y][x], after[y][x]) for x, y in changed
    ).most_common(8)
    return (f"changed={len(changed)} bbox=({min(xs)},{min(ys)})-"
            f"({max(xs)},{max(ys)}) transitions={transitions}")


def step(env, action_id: int, data: dict | None = None):
    action = GameAction.from_id(action_id)
    return env.step(action, data=data) if data is not None else env.step(action)


def parse_ints(raw: str, expected: int | None = None) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if expected is not None and len(values) != expected:
        raise argparse.ArgumentTypeError(
            f"expected {expected} comma-separated integers, got {raw!r}"
        )
    return values


def show_grid(grid: Grid, crop: tuple[int, int, int, int] | None) -> None:
    if crop is None:
        x0, y0, x1, y1 = 0, 0, len(grid[0]) - 1, len(grid) - 1
    else:
        x0, y0, x1, y1 = crop
    for y in range(max(0, y0), min(len(grid) - 1, y1) + 1):
        row = grid[y]
        print(f"{y:02d} " + "".join(
            format(row[x], "X")
            for x in range(max(0, x0), min(len(row) - 1, x1) + 1)
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--click-limit", type=int, default=16)
    parser.add_argument("--ascii", action="store_true")
    parser.add_argument("--crop", type=lambda raw: parse_ints(raw, 4),
                        help="inclusive x0,y0,x1,y1 ASCII crop")
    parser.add_argument("--sequence", type=parse_ints,
                        help="comma-separated simple actions to replay")
    parser.add_argument("--final-click", type=lambda raw: parse_ints(raw, 2),
                        help="x,y ACTION6 issued after --sequence")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    env = arcade.make(args.game)
    opening = env.reset()
    base = grid_of(opening)
    available = sorted(int(action) for action in opening.available_actions or [])
    background, comps = components(base)
    counts = Counter(value for row in base for value in row)

    print(f"game={args.game} state={opening.state} "
          f"levels={opening.levels_completed} actions={available}")
    print(f"background={background} colors={counts.most_common()}")
    print("components:")
    for rec in comps:
        print(f"  c={rec['color']:2d} n={rec['size']:4d} "
              f"bbox={rec['bbox']} center={rec['center']}")
    if args.ascii:
        print("opening:")
        show_grid(base, args.crop)

    if args.sequence:
        env.reset()
        print(f"sequence start: {args.sequence}")
        for index, action_id in enumerate(args.sequence, 1):
            if action_id == GameAction.ACTION6.value:
                raise SystemExit("ACTION6 sequence steps require coordinates")
            result = step(env, action_id)
            current = grid_of(result)
            print(f"sequence[{index}] action={action_id} state={result.state} "
                  f"levels={result.levels_completed}")
            show_grid(current, args.crop)
        if args.final_click:
            x, y = args.final_click
            result = step(env, GameAction.ACTION6.value, {"x": x, "y": y})
            print(f"sequence click=({x},{y}) state={result.state} "
                  f"levels={result.levels_completed}")
            show_grid(grid_of(result), args.crop)

    for action_id in available:
        if action_id == GameAction.RESET.value:
            continue
        if action_id == GameAction.ACTION6.value:
            candidates = [rec["center"] for rec in comps
                          if rec["size"] <= 512][:args.click_limit]
            for x, y in candidates:
                env.reset()
                result = step(env, action_id, {"x": x, "y": y})
                print(f"action={action_id} data=({x},{y}) frames={len(result.frame)} "
                      f"state={result.state} levels={result.levels_completed} "
                      f"{diff_summary(base, grid_of(result))}")
            continue
        env.reset()
        previous = base
        for repetition in range(1, args.repeat + 1):
            result = step(env, action_id)
            current = grid_of(result)
            print(f"action={action_id} repeat={repetition} frames={len(result.frame)} "
                  f"state={result.state} levels={result.levels_completed} "
                  f"step:{diff_summary(previous, current)} "
                  f"base:{diff_summary(base, current)}")
            previous = current
            if result.state.name in {"WIN", "GAME_OVER"}:
                break


if __name__ == "__main__":
    main()
