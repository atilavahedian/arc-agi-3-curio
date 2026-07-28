"""Inspect frame-derived descriptors around generic level completions.

This is a development probe, not a solver.  It subclasses Curio only to log
the settled frame immediately before a level advance and the first frame of
the next level.  Descriptors intentionally omit game ids, colors and absolute
coordinates so any production hypothesis can be evaluated for genuine
cross-level transfer rather than memorization.

Usage:
    CURIO_GENERIC_ONLY=1 CURIO_EXPLORER=graph \
      .venv/bin/python scripts/probe_completion_transfer.py lp85,vc33 \
      --max-steps 2000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arc_agi import OperationMode


def load_agent_module():
    spec = importlib.util.spec_from_file_location(
        "completion_probe_agent", ROOT / "agent" / "my_agent.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load agent/my_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def colorless_shape(cells: frozenset[tuple[int, int]]) -> tuple:
    """Rotation-canonical component geometry without color or position."""
    points = list(cells)
    variants = []
    for _ in range(4):
        x0 = min(x for x, _y in points)
        y0 = min(y for _x, y in points)
        variants.append(tuple(sorted((x - x0, y - y0)
                                     for x, y in points)))
        points = [(-y, x) for x, y in points]
    return min(variants)


def component_descriptor(
    comps: list[tuple[int, frozenset[tuple[int, int]]]],
    cell: tuple[int, int],
) -> Optional[dict[str, Any]]:
    target = next(((color, cells) for color, cells in comps if cell in cells),
                  None)
    if target is None:
        return None
    color, cells = target
    shape = colorless_shape(cells)
    xs = [x for x, _y in cells]
    ys = [y for _x, y in cells]
    sides = []
    if min(xs) == 0:
        sides.append("left")
    if max(xs) == 63:
        sides.append("right")
    if min(ys) == 0:
        sides.append("top")
    if max(ys) == 63:
        sides.append("bottom")
    shape_count = sum(colorless_shape(other) == shape
                      for _other_color, other in comps)
    color_count = sum(other_color == color for other_color, _cells in comps)
    return {
        "shape": shape,
        "area": len(cells),
        "bbox": [max(xs) - min(xs) + 1, max(ys) - min(ys) + 1],
        "border_sides": sides,
        "same_shape_components": shape_count,
        "same_color_components": color_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", help="comma-separated public game ids")
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "agg")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/arc3-mpl")
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger().setLevel(logging.CRITICAL)

    module = load_agent_module()
    BaseAgent = module.MyAgent

    class ProbeAgent(BaseAgent):
        completion_probe: list[dict[str, Any]]

        def __init__(self, *agent_args: Any, **agent_kwargs: Any) -> None:
            super().__init__(*agent_args, **agent_kwargs)
            self.completion_probe = []

        def _learn(self, grid, leveled=False, frames=None) -> None:
            if leveled and self._prev_grid is not None \
                    and self._prev_action is not None:
                entry: dict[str, Any] = {
                    "from_level": self._best_levels,
                    "action": ("click" if self._prev_action.startswith("6:")
                               else self._prev_action),
                    "action_counter": self.action_counter,
                }
                if self._prev_action.startswith("6:"):
                    try:
                        x, y = map(int, self._prev_action[2:].split(","))
                    except ValueError:
                        x = y = -1
                    previous = module.components(self._prev_grid)
                    descriptor = component_descriptor(previous, (x, y))
                    entry["winner"] = descriptor
                    if descriptor is not None and grid is not None:
                        shape = tuple(tuple(point)
                                      for point in descriptor["shape"])
                        next_comps = module.components(grid)
                        entry["next_shape_matches"] = [
                            component_descriptor(next_comps, min(cells))
                            for _color, cells in next_comps
                            if colorless_shape(cells) == shape
                        ]
                    sig = module.signature_under(previous, (x, y))
                    entry["exact_effects_before_win"] = \
                        list(self._click_effects.get(sig, (0, 0)))
                self.completion_probe.append(entry)
            super()._learn(grid, leveled, frames)

    ProbeAgent.MAX_ACTIONS = min(ProbeAgent.MAX_ACTIONS, args.max_steps)
    arcade = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    output = []
    for game in [g.strip() for g in args.games.split(",") if g.strip()]:
        env = arcade.make(game)
        agent = ProbeAgent(
            card_id="completion-probe", game_id=game,
            agent_name=f"Curio.completion-probe.{game}",
            ROOT_URL="http://localhost", record=False, arc_env=env,
            tags=["completion-probe"],
        )
        agent.main()
        final = agent.frames[-1]
        output.append({
            "game": game,
            "levels_completed": int(final.levels_completed),
            "actions": int(agent.action_counter),
            "completions": agent.completion_probe,
        })
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
