"""Regression tests for the trusted per-level overlay goal snapshot."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from collections import Counter, deque
from pathlib import Path
from unittest.mock import patch

from arcengine import FrameData, GameAction, GameState


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"


def load_agent_class():
    sys.path.insert(0, str(FRAMEWORK))
    spec = importlib.util.spec_from_file_location(
        "curio_overlay_agent_under_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


class OverlayGoalCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_class = load_agent_class()

    @staticmethod
    def _overlay_grid(count: int = 4) -> list[list[int]]:
        grid = [[8 for _x in range(64)] for _y in range(64)]
        centers = [(12, 12), (24, 12), (12, 24), (24, 24)][:count]
        for index, (cx, cy) in enumerate(centers):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    grid[cy + dy][cx + dx] = 3
            grid[cy][cx] = 9 if index < 2 else 11
        grid[40][40] = 0
        grid[40][39] = 13
        grid[40][41] = 13
        return grid

    def _snapshot_agent(self):
        agent = self.agent_class.__new__(self.agent_class)
        agent._ov_cached_outline = None
        agent._ov_cached_anchors = {}
        agent._ov_outline_votes = Counter()
        return agent

    def test_confirmed_goal_survives_live_ring_occlusion(self) -> None:
        agent = self._snapshot_agent()
        grid = self._overlay_grid()
        first = agent._ov_goal_snapshot(grid)
        self.assertIsNotNone(first)

        # Cover one complete ring and centre with the moving piece.  Only
        # three live anchors remain, below the four-anchor confidence gate.
        for y in range(11, 14):
            for x in range(11, 14):
                grid[y][x] = 9
        second = agent._ov_goal_snapshot(grid)

        self.assertEqual(second, first)
        self.assertEqual(sum(len(v) for v in second[1].values()), 4)
        self.assertTrue(all(isinstance(v, tuple) for v in second[1].values()))

    def test_weak_overlay_is_never_cached(self) -> None:
        agent = self._snapshot_agent()

        self.assertIsNone(agent._ov_goal_snapshot(self._overlay_grid(count=3)))
        self.assertIsNone(agent._ov_cached_outline)
        self.assertEqual(agent._ov_cached_anchors, {})

    def test_rich_overlay_without_selection_is_never_cached(self) -> None:
        agent = self._snapshot_agent()
        grid = self._overlay_grid()
        grid[40][40] = 8

        self.assertIsNone(agent._ov_goal_snapshot(grid))
        self.assertIsNone(agent._ov_cached_outline)
        self.assertEqual(agent._ov_cached_anchors, {})

    def test_inflight_plan_uses_snapshot_after_goal_is_hidden(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        for action, delta in {
            1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)
        }.items():
            agent._ov_deltas[action][delta] = 3
        agent._ov_level = 0
        agent._ov_plan = deque([4, 4])
        agent._ov_target = (46, 40)
        agent._ov_last_center = None
        frame = FrameData(
            frame=[self._overlay_grid()],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            available_actions=[1, 2, 3, 4, 5],
        )

        first_grid = self._overlay_grid()
        first_action = agent._overlay_policy(first_grid, frame)
        self.assertIs(first_action, GameAction.ACTION4)
        self.assertEqual(sum(len(v) for v in agent._ov_cached_anchors.values()), 4)

        hidden = self._overlay_grid()
        hidden[40][40] = 13
        hidden[40][43] = 0
        hidden[40][42] = 13
        hidden[40][44] = 13
        for cx, cy in ((12, 12), (24, 12)):
            for y in range(cy - 1, cy + 2):
                for x in range(cx - 1, cx + 2):
                    hidden[y][x] = 13
        fresh = self._snapshot_agent()
        self.assertIsNone(fresh._ov_goal_snapshot(hidden))

        second_action = agent._overlay_policy(hidden, frame)

        self.assertIs(second_action, GameAction.ACTION4)
        self.assertEqual(list(agent._ov_plan), [])

    def test_same_level_game_over_retains_snapshot(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        agent._ov_cached_outline = 3
        agent._ov_cached_anchors = {9: ((12, 12),)}
        frame = FrameData(
            frame=[self._overlay_grid()],
            state=GameState.GAME_OVER,
            levels_completed=0,
            available_actions=[GameAction.RESET.value],
        )

        action = agent.choose_action([], frame)

        self.assertIs(action, GameAction.RESET)
        self.assertEqual(agent._ov_cached_outline, 3)
        self.assertEqual(agent._ov_cached_anchors, {9: ((12, 12),)})

    def test_level_up_clears_snapshot(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        agent._ov_cached_outline = 3
        agent._ov_cached_anchors = {9: ((12, 12),)}
        agent._policy = lambda _grid, _frame: GameAction.ACTION1
        grid = [[8 for _x in range(64)] for _y in range(64)]
        frame = FrameData(
            frame=[grid],
            state=GameState.NOT_FINISHED,
            levels_completed=1,
            available_actions=[GameAction.ACTION1.value],
        )

        agent.choose_action([], frame)

        self.assertIsNone(agent._ov_cached_outline)
        self.assertEqual(agent._ov_cached_anchors, {})

    @staticmethod
    def _hollow_piece_grid(missing_tip: bool = True) -> list[list[int]]:
        grid = [[8 for _x in range(64)] for _y in range(64)]
        cx, cy = 39, 30
        grid[cy][cx] = 0
        for dy in range(-9, 10):
            for dx in range(-9, 10):
                if abs(dx) + abs(dy) == 9:
                    grid[cy + dy][cx + dx] = 13
        if missing_tip:
            grid[cy][cx + 9] = 9
        # A previously placed piece crosses only one side of the search area;
        # raw nearest-colour and cumulative-window readers choose it wrongly.
        for distance in range(1, 10):
            grid[cy - distance][cx - distance] = 12
        # Remote same-colour anchor pixels must not join the moving body.
        for point in ((5, 5), (10, 10), (55, 55)):
            grid[point[1]][point[0]] = 13
        return grid

    def test_symmetric_shell_finds_hollow_selected_piece(self) -> None:
        agent = self._snapshot_agent()

        selected = agent._ov_selected(self._hollow_piece_grid(), outline=4)

        self.assertEqual(selected, ((39, 30), 13))

    def test_adjacent_line_pair_is_a_valid_selected_piece(self) -> None:
        agent = self._snapshot_agent()
        grid = [[8 for _x in range(64)] for _y in range(64)]
        grid[30][30] = 0
        grid[30][29] = 13
        grid[30][31] = 13
        grid[28][28] = 12  # closer-shell noise without an opposite partner

        self.assertEqual(agent._ov_selected(grid, outline=4), ((30, 30), 13))

    def test_equal_symmetric_arm_candidates_are_rejected(self) -> None:
        agent = self._snapshot_agent()
        grid = [[8 for _x in range(64)] for _y in range(64)]
        grid[30][30] = 0
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            grid[30 + dy][30 + dx] = 13
        for dx, dy in ((-2, -2), (2, 2), (-2, 2), (2, -2)):
            grid[30 + dy][30 + dx] = 12

        self.assertIsNone(agent._ov_selected(grid, outline=4))

    def test_disconnected_hollow_footprint_uses_nearest_component(self) -> None:
        grid = self._hollow_piece_grid()
        funcs = self.agent_class._ov_selected.__globals__
        footprint = funcs["piece_footprint"](grid, (39, 30), 13)
        expected = {(39, 30)} | {
            (39 + dx, 30 + dy)
            for dy in range(-9, 10) for dx in range(-9, 10)
            if abs(dx) + abs(dy) == 9 and (dx, dy) != (9, 0)
        }

        self.assertEqual(footprint, frozenset(expected))
        self.assertEqual(len(footprint), 36)
        self.assertTrue({(5, 5), (10, 10), (55, 55)}.isdisjoint(footprint))
        target = funcs["cover_centers"](
            footprint, (39, 30), [(21, 3), (27, 9), (12, 12)], 3,
            (39, 30),
        )
        self.assertEqual(target, (21, 12))

    def test_pristine_hollow_footprint_keeps_entire_ring(self) -> None:
        grid = self._hollow_piece_grid(missing_tip=False)
        footprint = self.agent_class._ov_selected.__globals__["piece_footprint"](
            grid, (39, 30), 13
        )

        self.assertEqual(len(footprint), 37)
        self.assertIn((48, 30), footprint)

    def test_connected_footprint_and_missing_arm_keep_old_behavior(self) -> None:
        func = self.agent_class._ov_selected.__globals__["piece_footprint"]
        grid = [[8 for _x in range(64)] for _y in range(64)]
        center = (30, 30)
        grid[30][30] = 0
        expected = {center}
        for distance in range(1, 5):
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                point = (30 + sx * distance, 30 + sy * distance)
                grid[point[1]][point[0]] = 13
                expected.add(point)
        self.assertEqual(func(grid, center, 13), frozenset(expected))

        empty = [[8 for _x in range(64)] for _y in range(64)]
        empty[30][30] = 0
        self.assertEqual(func(empty, center, 13), frozenset({center}))

    def test_confirmed_overlay_blocks_editor_takeover(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        agent._ov_outline_votes[4] = 1
        frame = FrameData(
            frame=[self._overlay_grid()],
            state=GameState.NOT_FINISHED,
            levels_completed=1,
            available_actions=[1, 2, 3, 4, 5],
        )

        self.assertIsNone(agent._editor_policy(self._overlay_grid(), frame))
        self.assertFalse(agent._ed_engaged)


if __name__ == "__main__":
    unittest.main()
