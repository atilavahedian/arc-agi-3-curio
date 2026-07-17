"""Regression tests for Curio's generic state-graph explorer."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"


def load_agent_class():
    sys.path.insert(0, str(FRAMEWORK))
    spec = importlib.util.spec_from_file_location(
        "curio_agent_under_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


class GraphResetEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_class = load_agent_class()

    def test_pending_reset_edge_keeps_state_action_order(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        agent._gx_pending_reset = (9173, "6:11,29")
        agent._transitions = {}
        agent._tried = defaultdict(set)

        agent._gx_commit_pending_reset(4401)

        self.assertEqual(agent._transitions, {(9173, "6:11,29"): 4401})
        self.assertEqual(agent._tried[9173], {"6:11,29"})
        self.assertNotIn("6:11,29", agent._tried)
        self.assertIsNone(agent._gx_pending_reset)

    def test_no_pending_reset_is_a_noop(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        agent._gx_pending_reset = None
        agent._transitions = {(1, "2"): 3}
        agent._tried = defaultdict(set, {1: {"2"}})

        agent._gx_commit_pending_reset(999)

        self.assertEqual(agent._transitions, {(1, "2"): 3})
        self.assertEqual(agent._tried[1], {"2"})


class SlideLethalEdgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_class = load_agent_class()

    def test_game_over_records_physical_node_and_action(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        agent._sl_engaged = True
        agent._prev_grid = [[0]]
        agent._prev_action = "4"
        agent._sl_lethal_edges = set()
        lattice = (6, 2, 4, {(8, 10)}, 1, 2)
        agent._sl_avatar_anchor = lambda _grid: ((7, 9), 5)
        agent._sl_body_center = lambda _grid, _anchor, _lat=None: (8, 10)
        agent._sl_lattice = lambda _grid, _center: lattice
        agent._sl_node = lambda _cell, _lat: (8, 10)

        agent._sl_note_lethal_edge()

        self.assertEqual(agent._sl_lethal_edges, {((8, 10), 4)})

    def test_route_will_not_replay_lethal_edge(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        agent._sl_dirmap = {}
        agent._sl_blocked_edges = set()
        agent._sl_lethal_edges = set()
        grid = [[0 for _x in range(64)] for _y in range(64)]
        grid[1][2] = 1
        lattice = (2, 1, 1, {(1, 1), (3, 1)}, 1, 2)
        agent._sl_avatar_cells = lambda _grid, _start, _lat: set()
        agent._sl_body_center = lambda _grid, _start, _lat=None: (1, 1)
        agent._sl_node = lambda _cell, _lat: (1, 1)
        agent._sl_exit_node = lambda _grid, _lat, _cells: (3, 1)

        self.assertEqual(agent._sl_route(grid, (1, 1), lattice, 1), ([4], (3, 1)))

        agent._sl_lethal_edges.add(((1, 1), 4))
        self.assertIsNone(agent._sl_route(grid, (1, 1), lattice, 1))

    def test_route_will_not_repeat_observed_blocked_edge(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        agent._sl_dirmap = {}
        agent._sl_blocked_edges = set()
        agent._sl_lethal_edges = set()
        grid = [[0 for _x in range(64)] for _y in range(64)]
        grid[1][2] = 1
        lattice = (2, 1, 1, {(1, 1), (3, 1)}, 1, 2)
        agent._sl_avatar_cells = lambda _grid, _start, _lat: set()
        agent._sl_body_center = lambda _grid, _start, _lat=None: (1, 1)
        agent._sl_node = lambda _cell, _lat: (1, 1)
        agent._sl_exit_node = lambda _grid, _lat, _cells: (3, 1)
        agent._sl_edge_at = lambda _grid, action: ((1, 1), action)

        agent._sl_note_blocked_edge(grid, 4)

        self.assertEqual(agent._sl_blocked_edges, {((1, 1), 4)})
        self.assertIsNone(agent._sl_route(grid, (1, 1), lattice, 1))

    def test_snapped_outcome_marks_noop_but_not_motion(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        agent._sl_blocked_edges = set()
        agent._sl_node_at = lambda node: node
        a, b = (1, 1), (3, 1)

        agent._sl_note_outcome(a, b, 4)
        self.assertEqual(agent._sl_blocked_edges, set())

        agent._sl_note_outcome(b, b, 2)
        self.assertEqual(agent._sl_blocked_edges, {(b, 2)})

    def test_failed_life_cap_benches_slide_head_for_level(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        agent._sl_benched = None
        agent._sl_level = 2
        agent._level_deaths = 4
        agent._sl_deaths_at_engage = 0
        frame = SimpleNamespace(
            available_actions=[1, 2, 3, 4], levels_completed=2)

        self.assertIsNone(agent._slide_policy([], frame))
        self.assertEqual(agent._sl_benched, 2)


if __name__ == "__main__":
    unittest.main()
