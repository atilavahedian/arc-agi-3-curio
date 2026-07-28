"""Focused tests for the isolated Curio-v7 runtime circuit breaker."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from arcengine import FrameData, GameAction, GameState


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"


def load_agent_class():
    sys.path.insert(0, str(FRAMEWORK))
    spec = importlib.util.spec_from_file_location(
        "curio_v7_fault_contained_under_test",
        ROOT / "baselines" / "curio_v7_fault_contained.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


class CurioV7FaultContainmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_class = load_agent_class()

    @staticmethod
    def _grid(width: int = 7, height: int = 5) -> list[list[int]]:
        return [[0 for _x in range(width)] for _y in range(height)]

    @classmethod
    def _frame(
        cls,
        actions: list[int],
        *,
        grid: list[list[int]] | None = None,
        state: GameState = GameState.NOT_FINISHED,
    ) -> FrameData:
        return FrameData(
            frame=[grid if grid is not None else cls._grid()],
            state=state,
            levels_completed=0,
            available_actions=actions,
        )

    @staticmethod
    def _raise(error: Exception):
        def raising(*_args, **_kwargs):
            raise error

        return raising

    def test_success_path_returns_delegate_object_and_payload_exactly(
        self,
    ) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        expected = GameAction.ACTION6
        expected.set_data({"x": 5, "y": 3})
        agent._choose_action = lambda _frames, _latest: expected

        actual = agent.choose_action([], self._frame([6]))

        self.assertIs(actual, expected)
        self.assertEqual(
            (actual.action_data.x, actual.action_data.y), (5, 3)
        )
        self.assertFalse(hasattr(agent, "_runtime_faults"))

    def test_active_fault_uses_smallest_advertised_simple_action(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        agent._choose_action = self._raise(RuntimeError("hidden parser miss"))
        frame = self._frame([0, 6, 4, 2])

        action = agent.choose_action([], frame)

        self.assertIs(action, GameAction.ACTION2)
        self.assertIn(action.value, frame.available_actions)
        self.assertEqual(agent._prev_action, "2")
        self.assertEqual(agent._runtime_faults, 1)
        self.assertEqual(agent._last_runtime_fault, "RuntimeError")

    def test_click_only_fault_is_deterministic_and_in_bounds(self) -> None:
        frame = self._frame([0, 6], grid=self._grid(width=7, height=5))
        agent = self.agent_class.__new__(self.agent_class)
        agent._choose_action = self._raise(ValueError("unknown click layout"))

        first = agent.choose_action([], frame)
        first_xy = (first.action_data.x, first.action_data.y)
        second = agent.choose_action([], frame)
        second_xy = (second.action_data.x, second.action_data.y)

        self.assertIs(first, GameAction.ACTION6)
        self.assertIs(second, GameAction.ACTION6)
        self.assertIn(first.value, frame.available_actions)
        self.assertEqual(first_xy, (3, 2))
        self.assertEqual(second_xy, first_xy)
        self.assertLess(first_xy[0], 7)
        self.assertLess(first_xy[1], 5)
        self.assertEqual(agent._prev_action, "6:3,2")
        self.assertEqual(agent._runtime_faults, 2)

    def test_terminal_fault_resets_and_drops_stale_learning_edge(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        agent._choose_action = self._raise(IndexError("terminal cleanup miss"))
        agent._prev_grid = self._grid()
        agent._prev_key = 123
        agent._prev_action = "4"
        frame = self._frame([0], state=GameState.GAME_OVER)

        action = agent.choose_action([], frame)

        self.assertIs(action, GameAction.RESET)
        self.assertIsNone(agent._prev_grid)
        self.assertIsNone(agent._prev_key)
        self.assertIsNone(agent._prev_action)
        self.assertEqual(agent._runtime_faults, 1)


if __name__ == "__main__":
    unittest.main()
