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


if __name__ == "__main__":
    unittest.main()
