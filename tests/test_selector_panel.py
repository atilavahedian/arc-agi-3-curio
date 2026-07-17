"""Regression tests for the clue-driven selector-panel macro."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from arcengine import FrameData, GameAction, GameState


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"


def load_agent_module():
    sys.path.insert(0, str(FRAMEWORK))
    spec = importlib.util.spec_from_file_location(
        "curio_panel_agent_under_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SelectorPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_agent_module()
        cls.agent_class = cls.module.MyAgent

    @staticmethod
    def _grid(*, transformed: bool = False, mover_x: int = 30):
        grid = [[5 for _x in range(64)] for _y in range(64)]

        # Larger two-colour socket and smaller same-palette mover.
        for y in range(10, 16):
            for x in range(5, 10):
                grid[y][x] = 9 if x < 7 else 10
        mover_side = 2 if transformed else 4
        split = mover_x + mover_side // 2
        for y in range(11, 11 + mover_side):
            for x in range(mover_x, mover_x + mover_side):
                grid[y][x] = 9 if x < split else 10

        # Framed 3x3 panel: the minority value forms a cardinal mask.
        for y in range(42, 59):
            for x in range(22, 39):
                grid[y][x] = 3
        values = ((2, 0, 2), (0, 2, 0), (2, 0, 2))
        for row in range(3):
            for col in range(3):
                for y in range(44 + 5 * row, 47 + 5 * row):
                    for x in range(24 + 5 * col, 27 + 5 * col):
                        grid[y][x] = values[row][col]

        # Separate same-frame clue card with the identical cardinal mask.
        for y in range(45, 55):
            for x in range(5, 15):
                grid[y][x] = 3 if x in (5, 14) or y in (45, 54) else 2
        for x0, y0 in ((9, 46), (6, 49), (12, 49), (9, 52)):
            for y in range(y0, y0 + 2):
                for x in range(x0, x0 + 2):
                    grid[y][x] = 15
        return grid

    @staticmethod
    def _frame(grid, *, level: int = 0, state=GameState.NOT_FINISHED):
        return FrameData(
            frame=[grid],
            state=state,
            levels_completed=level,
            available_actions=[1, 2, 3, 4, 6],
        )

    def _agent(self):
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            return self.agent_class(
                card_id="test",
                game_id="panel-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )

    @classmethod
    def _grid_for_drive(cls, drive: GameAction):
        grid = cls._grid()
        for y in range(40):
            for x in range(64):
                if grid[y][x] in (9, 10):
                    grid[y][x] = 5
        specs = {
            GameAction.ACTION1: ((45, 5, 50, 9), (46, 25, 49, 28)),
            GameAction.ACTION2: ((45, 25, 50, 29), (46, 5, 49, 8)),
            GameAction.ACTION3: ((5, 10, 9, 15), (30, 11, 33, 14)),
            GameAction.ACTION4: ((40, 10, 44, 15), (15, 11, 18, 14)),
        }
        socket, mover = specs[drive]
        for x0, y0, x1, y1 in (socket, mover):
            split = x0 + (x1 - x0 + 1) // 2
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    grid[y][x] = 9 if x < split else 10
        return grid

    def test_detector_returns_clue_and_cardinal_panel_cells(self) -> None:
        setup = self.module.panel_macro_setup(
            self._grid(), {1, 2, 3, 4, 6})

        self.assertIsNotNone(setup)
        self.assertEqual(
            setup["clicks"],
            [(9, 46), (30, 45), (25, 50), (35, 50), (30, 55)],
        )
        self.assertEqual(setup["actor_colors"], (9, 10))
        self.assertEqual(setup["actor_pixels"], 16)
        self.assertEqual(setup["drive"], GameAction.ACTION3.value)

    def test_detector_rejects_wrong_action_profile(self) -> None:
        self.assertIsNone(
            self.module.panel_macro_setup(self._grid(), {1, 2, 3, 4, 5, 6})
        )

    def test_geometry_maps_all_four_cardinal_drives(self) -> None:
        directions = {
            GameAction.ACTION1: (0, -1),
            GameAction.ACTION2: (0, 1),
            GameAction.ACTION3: (-1, 0),
            GameAction.ACTION4: (1, 0),
        }
        for expected, direction in directions.items():
            with self.subTest(action=expected):
                setup = self.module.panel_macro_setup(
                    self._grid_for_drive(expected), {1, 2, 3, 4, 6})
                self.assertIsNotNone(setup)
                self.assertEqual(setup["drive"], expected.value)
                self.assertEqual(setup["direction"], direction)

    def test_detector_rejects_panel_clue_mask_mismatch(self) -> None:
        grid = self._grid()
        for y in range(44, 47):
            for x in range(29, 32):
                grid[y][x] = 2
        for y in range(44, 47):
            for x in range(24, 27):
                grid[y][x] = 0

        self.assertIsNone(
            self.module.panel_macro_setup(grid, {1, 2, 3, 4, 6})
        )

    def test_policy_clicks_then_revalidates_motion(self) -> None:
        agent = self._agent()
        initial = self._grid()
        frame = self._frame(initial)
        clicks = []
        for _ in range(5):
            action = agent._panel_policy(initial, frame)
            self.assertIs(action, GameAction.ACTION6)
            clicks.append((action.action_data.x, action.action_data.y))
        self.assertEqual(
            clicks, [(9, 46), (30, 45), (25, 50), (35, 50), (30, 55)])

        cast = self._grid(transformed=True, mover_x=30)
        first_drive = agent._panel_policy(cast, self._frame(cast))
        moved = self._grid(transformed=True, mover_x=28)
        second_drive = agent._panel_policy(moved, self._frame(moved))

        self.assertIs(first_drive, GameAction.ACTION3)
        self.assertIs(second_drive, GameAction.ACTION3)

        self.assertIsNone(agent._panel_policy(moved, self._frame(moved)))
        self.assertEqual(agent._panel_benched, 0)
        self.assertFalse(agent._panel_engaged)

    def test_public_dispatch_engages_panel_before_fallback(self) -> None:
        agent = self._agent()
        grid = self._grid()

        action = agent.choose_action([], self._frame(grid))

        self.assertIs(action, GameAction.ACTION6)
        self.assertEqual((action.action_data.x, action.action_data.y), (9, 46))
        self.assertTrue(agent._panel_engaged)

    def test_wrong_way_motion_benches_without_rearming(self) -> None:
        agent = self._agent()
        initial = self._grid()
        frame = self._frame(initial)
        for _ in range(5):
            agent._panel_policy(initial, frame)
        cast = self._grid(transformed=True, mover_x=30)
        self.assertIs(
            agent._panel_policy(cast, self._frame(cast)), GameAction.ACTION3)

        wrong_way = self._grid(transformed=True, mover_x=32)
        self.assertIsNone(
            agent._panel_policy(wrong_way, self._frame(wrong_way)))
        self.assertEqual(agent._panel_benched, 0)
        self.assertIsNone(
            agent._panel_policy(initial, self._frame(initial)))

    def test_game_over_clears_and_benches_active_macro(self) -> None:
        agent = self._agent()
        agent._panel_engaged = True
        agent._panel_level = 0
        agent._panel_clicks.append((30, 45))
        frame = self._frame(self._grid(), state=GameState.GAME_OVER)

        action = agent.choose_action([], frame)

        self.assertIs(action, GameAction.RESET)
        self.assertFalse(agent._panel_engaged)
        self.assertEqual(list(agent._panel_clicks), [])
        self.assertEqual(agent._panel_benched, 0)

    def test_level_up_clears_panel_bench(self) -> None:
        agent = self._agent()
        agent._panel_benched = 0
        agent._panel_level = 0
        agent._policy = lambda _grid, _frame: GameAction.ACTION1
        grid = [[5 for _x in range(64)] for _y in range(64)]

        agent.choose_action([], self._frame(grid, level=1))

        self.assertEqual(agent._panel_level, 1)
        self.assertIsNone(agent._panel_benched)
        self.assertFalse(agent._panel_engaged)


if __name__ == "__main__":
    unittest.main()
