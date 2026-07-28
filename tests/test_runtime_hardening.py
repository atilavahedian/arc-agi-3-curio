"""Failure-containment tests for Curio's top-level runtime boundary."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import deque
from pathlib import Path

from arcengine import FrameData, GameAction, GameState


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"


def load_agent_class():
    sys.path.insert(0, str(FRAMEWORK))
    spec = importlib.util.spec_from_file_location(
        "curio_runtime_agent_under_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


class RuntimeHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_class = load_agent_class()

    @staticmethod
    def _grid(size: int = 64) -> list[list[int]]:
        return [[0 for _x in range(size)] for _y in range(size)]

    @classmethod
    def _frame(
        cls, actions: list[int], *, grid=None,
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

    def test_success_path_returns_the_delegate_result_untouched(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        expected = GameAction.ACTION5
        agent._choose_action = lambda _frames, _latest: expected

        actual = agent.choose_action([], self._frame([5]))

        self.assertIs(actual, expected)
        self.assertFalse(hasattr(agent, "_runtime_faults"))

    def test_fault_prefers_smallest_advertised_simple_action_not_reset(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        agent._choose_action = self._raise(RuntimeError("parser exploded"))
        frame = self._frame([0, 6, 4, 2])

        action = agent.choose_action([], frame)

        self.assertIs(action, GameAction.ACTION2)
        self.assertIn(action.value, frame.available_actions)
        self.assertEqual(agent._prev_action, "2")
        self.assertEqual(agent._runtime_faults, 1)
        self.assertEqual(agent._last_runtime_fault, "RuntimeError")

    def test_click_only_fault_uses_visible_rare_pixel_and_is_deterministic(
        self,
    ) -> None:
        grid = [[0 for _x in range(7)] for _y in range(5)]
        grid[1][1] = 4
        grid[1][2] = 4
        grid[3][5] = 7  # unique rarest foreground class
        frame = self._frame([0, 6], grid=grid)
        agent = self.agent_class.__new__(self.agent_class)
        agent._choose_action = self._raise(ValueError("unknown layout"))

        first = agent.choose_action([], frame)
        first_data = (first.action_data.x, first.action_data.y)
        second = agent.choose_action([], frame)

        self.assertIs(first, GameAction.ACTION6)
        self.assertIs(second, GameAction.ACTION6)
        self.assertIn(first.value, frame.available_actions)
        self.assertEqual(first_data, (5, 3))
        self.assertEqual(
            (second.action_data.x, second.action_data.y), first_data)
        self.assertEqual(agent._prev_action, "6:5,3")
        self.assertEqual(agent._runtime_faults, 2)

    def test_fault_discards_partial_plans_before_fallback(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        agent._choose_action = self._raise(IndexError("partial parse"))
        agent._plan = deque([GameAction.ACTION1])
        agent._lattice_plan = deque([(1, 2)])
        agent._sort_plan = [(3, (1, 1), (2, 2))]
        agent._gx_route = deque(["4"])
        agent._wp_replay = [(123, "2")]

        action = agent.choose_action([], self._frame([3]))

        self.assertIs(action, GameAction.ACTION3)
        self.assertEqual(list(agent._plan), [])
        self.assertEqual(list(agent._lattice_plan), [])
        self.assertEqual(agent._sort_plan, [])
        self.assertEqual(list(agent._gx_route), [])
        self.assertIsNone(agent._wp_replay)

    def test_one_policy_fault_recovers_on_the_next_frame(self) -> None:
        agent = self.agent_class(
            card_id="test",
            game_id="runtime-recovery",
            agent_name="curio",
            ROOT_URL="",
            record=False,
            arc_env=None,
        )
        calls = 0

        def policy(_grid, _latest):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise LookupError("one hidden parser miss")
            return agent._step(GameAction.ACTION4)

        agent._policy = policy
        frame = self._frame([2, 4])

        first = agent.choose_action([], frame)
        second = agent.choose_action([], frame)

        self.assertIs(first, GameAction.ACTION2)
        self.assertIs(second, GameAction.ACTION4)
        self.assertEqual(agent._runtime_faults, 1)
        self.assertEqual(agent._prev_action, "4")

    def test_terminal_fault_resets_but_active_fault_does_not(self) -> None:
        terminal = self.agent_class.__new__(self.agent_class)
        terminal._choose_action = self._raise(RuntimeError("cleanup miss"))
        dead = self._frame([0], state=GameState.GAME_OVER)

        active = self.agent_class.__new__(self.agent_class)
        active._choose_action = self._raise(RuntimeError("policy miss"))
        live = self._frame([0, 7])

        self.assertIs(terminal.choose_action([], dead), GameAction.RESET)
        self.assertIs(active.choose_action([], live), GameAction.ACTION7)
        self.assertIsNone(terminal._prev_action)
        self.assertEqual(active._prev_action, "7")


if __name__ == "__main__":
    unittest.main()
