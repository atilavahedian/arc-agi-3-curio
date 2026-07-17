"""Regression tests for Curio's generic state-graph explorer."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from collections import Counter, defaultdict, deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from arcengine import FrameData, GameAction, GameState


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


class HudIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_class = load_agent_class()

    def _agent(self, graph: bool = False):
        agent = self.agent_class.__new__(self.agent_class)
        agent._frames_diffed = 32
        agent._band_frames = defaultdict(int)
        agent._hud_mask = frozenset()
        agent._hud_identity_ready = False
        agent._transitions = {(10, "1"): 11}
        agent._tried = defaultdict(set, {10: {"1"}})
        agent._state_visits = Counter({10: 2})
        agent._steps_since_novelty = 9
        agent._gx_on = graph
        agent._click_effects = {77: [1, 2]}
        agent._move_votes = defaultdict(Counter, {1: Counter({(1, 0): 3})})
        if graph:
            agent._gx_nodes = {10: {"actions": ["1"]}}
            agent._gx_route = deque(["1"])
            agent._gx_route_dest = 11
            agent._gx_start = 10
            agent._gx_pending_reset = (10, "1")
        return agent

    def test_hud_free_identity_is_ready_at_first_recompute(self) -> None:
        agent = self._agent()

        self.assertFalse(agent._recompute_hud_mask())
        self.assertTrue(agent._hud_identity_ready)
        self.assertEqual(agent._transitions, {(10, "1"): 11})

    def test_nonempty_mask_requires_two_identical_candidates(self) -> None:
        agent = self._agent()
        agent._band_frames[(0, 0)] = (1 << 32) - 1

        self.assertTrue(agent._recompute_hud_mask())
        self.assertEqual(agent._hud_mask, frozenset({(0, 0)}))
        self.assertFalse(agent._hud_identity_ready)

        self.assertFalse(agent._recompute_hud_mask())
        self.assertTrue(agent._hud_identity_ready)

    def test_mask_change_invalidates_only_keyed_geography(self) -> None:
        agent = self._agent(graph=True)
        agent._band_frames[(0, 0)] = (1 << 32) - 1

        agent._recompute_hud_mask()

        self.assertEqual(agent._transitions, {})
        self.assertEqual(agent._tried, {})
        self.assertEqual(agent._state_visits, Counter())
        self.assertEqual(agent._gx_nodes, {})
        self.assertEqual(list(agent._gx_route), [])
        self.assertIsNone(agent._gx_route_dest)
        self.assertIsNone(agent._gx_start)
        self.assertIsNone(agent._gx_pending_reset)
        self.assertEqual(agent._click_effects, {77: [1, 2]})
        self.assertEqual(agent._move_votes[1][(1, 0)], 3)

    def test_level_repaint_neither_updates_hud_nor_records_edge(self) -> None:
        agent = self._agent()
        agent._last_move = None
        agent._prev_grid = [[0]]
        agent._prev_action = "1"
        agent._prev_key = 10
        agent._plan = deque(["stale"])
        agent._update_hud_mask = lambda _grid: self.fail("HUD observed level repaint")

        agent._learn([[1]], leveled=True)

        self.assertEqual(agent._transitions, {(10, "1"): 11})
        self.assertEqual(list(agent._plan), [])

    def test_first_edge_after_mask_change_rehashes_source(self) -> None:
        agent = self._agent()
        grid = [[0 for _x in range(64)] for _y in range(64)]
        agent._last_move = None
        agent._prev_grid = grid
        agent._prev_action = "probe"
        agent._prev_key = 12345
        agent._hud_mask = frozenset({(0, 0)})
        agent._plan = deque()
        agent._update_hud_mask = lambda _grid: True
        agent._transitions.clear()
        agent._tried.clear()
        expected = agent._masked_hash(grid)

        agent._learn(grid)

        self.assertEqual(agent._transitions, {(expected, "probe"): expected})
        self.assertEqual(agent._tried[expected], {"probe"})
        self.assertNotIn((12345, "probe"), agent._transitions)


class ClickInstanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_class = load_agent_class()

    def _agent(self):
        agent = self.agent_class.__new__(self.agent_class)
        agent._gx_instance_effects = {}
        agent._gx_lethal_sig = set()
        agent._click_effects = {}
        agent._gx_nodes = {}
        agent._tried = defaultdict(set)
        return agent

    @staticmethod
    def _twin_grid():
        grid = [[0 for _x in range(64)] for _y in range(64)]
        grid[5][63] = 1
        grid[20][63] = 1
        return grid

    @staticmethod
    def _dense_twin_grid():
        grid = [[0 for _x in range(64)] for _y in range(64)]
        for i in range(17):
            grid[2 * (i // 8) + 1][2 * (i % 8) + 1] = 1
        return grid

    def test_identical_components_are_both_candidates(self) -> None:
        agent = self._agent()

        self.assertEqual(
            agent._gx_click_targets(self._twin_grid()),
            [(63, 5), (63, 20)],
        )

    def test_mixed_action_board_preserves_class_dedupe(self) -> None:
        agent = self._agent()

        self.assertEqual(len(agent._gx_click_targets(
            self._twin_grid(), {GameAction.ACTION1.value,
                                GameAction.ACTION6.value})), 1)

    def test_dense_board_preserves_one_candidate_per_class(self) -> None:
        agent = self._agent()

        self.assertEqual(len(agent._gx_click_targets(
            self._dense_twin_grid())), 1)

    def test_dense_board_preserves_global_dead_class_pruning(self) -> None:
        agent = self._agent()
        grid = self._dense_twin_grid()
        sig = agent._gx_instance_key.__globals__["shape_signature"](
            1, frozenset({(1, 1)}))
        agent._click_effects[sig] = [0, 4]

        self.assertEqual(agent._gx_click_targets(grid), [])

    def test_unknown_twins_are_exposed_one_at_a_time(self) -> None:
        agent = self._agent()
        grid = self._twin_grid()
        node = agent._gx_node(99, grid, {GameAction.ACTION6.value})

        first = agent._gx_untried(99, node)
        self.assertEqual(len(first), 1)
        agent._tried[99].add(first[0])
        second = agent._gx_untried(99, node)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first, second)

    def test_fresh_twin_beats_partially_inert_twin(self) -> None:
        agent = self._agent()
        grid = self._twin_grid()
        sig = agent._gx_instance_key.__globals__["shape_signature"](
            1, frozenset({(63, 5)}))
        agent._gx_note_instance(sig, (63, 5), changed=False)
        node = agent._gx_node(99, grid, {GameAction.ACTION6.value})

        self.assertEqual(agent._gx_untried(99, node), ["6:63,20"])

    def test_cached_node_refreshes_instance_salience(self) -> None:
        agent = self._agent()
        grid = self._twin_grid()
        node = agent._gx_node(99, grid, {GameAction.ACTION6.value})
        sig = node["sig"]["6:63,5"]
        for _ in range(4):
            agent._gx_note_instance(sig, (63, 5), changed=False)

        self.assertEqual(agent._gx_untried(99, node), ["6:63,20"])

    def test_productive_class_exposes_all_twins(self) -> None:
        agent = self._agent()
        grid = self._twin_grid()
        sig = agent._gx_instance_key.__globals__["shape_signature"](
            1, frozenset({(63, 5)}))
        agent._click_effects[sig] = [3, 3]
        node = agent._gx_node(99, grid, {GameAction.ACTION6.value})

        self.assertEqual(len(agent._gx_untried(99, node)), 2)

    def test_global_negative_history_does_not_hide_fresh_instances(self) -> None:
        agent = self._agent()
        grid = self._twin_grid()
        sig = agent._gx_instance_key.__globals__["shape_signature"](
            1, frozenset({(63, 5)}))
        agent._click_effects[sig] = [0, 4]

        self.assertEqual(
            agent._gx_click_targets(grid), [(63, 5), (63, 20)])

    def test_inert_instance_does_not_suppress_its_twin(self) -> None:
        agent = self._agent()
        grid = self._twin_grid()
        sig = agent._gx_instance_key.__globals__["shape_signature"](
            1, frozenset({(63, 5)}))
        for _ in range(4):
            agent._gx_note_instance(sig, (63, 5), changed=False)

        self.assertEqual(agent._gx_click_targets(grid), [(63, 20)])

    def test_all_inert_instances_do_not_fall_back_to_background(self) -> None:
        agent = self._agent()
        grid = self._twin_grid()
        sig = agent._gx_instance_key.__globals__["shape_signature"](
            1, frozenset({(63, 5)}))
        for coords in ((63, 5), (63, 20)):
            for _ in range(4):
                agent._gx_note_instance(sig, coords, changed=False)

        self.assertEqual(agent._gx_click_targets(grid), [])

    def test_noncentroid_click_updates_canonical_instance(self) -> None:
        agent = self._agent()
        grid = [[0 for _x in range(64)] for _y in range(64)]
        for x, y in ((62, 5), (63, 5), (62, 6),
                     (62, 20), (63, 20), (62, 21)):
            grid[y][x] = 1
        comps = agent._gx_instance_key.__globals__["components"](grid)
        sig = agent._gx_instance_key.__globals__["signature_under"](
            comps, (63, 5))
        for _ in range(4):
            agent._gx_note_instance(
                sig, (63, 5), changed=False, comps=comps)

        self.assertEqual(agent._gx_click_targets(grid), [(62, 20)])

    def test_level_up_clears_instance_evidence(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="click-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        agent._gx_instance_effects[(123, 5, 5)] = [0, 4]
        agent._policy = lambda _grid, _frame: GameAction.ACTION1
        grid = [[0 for _x in range(64)] for _y in range(64)]
        frame = FrameData(
            frame=[grid],
            state=GameState.NOT_FINISHED,
            levels_completed=1,
            available_actions=[GameAction.ACTION1.value],
        )

        agent.choose_action([], frame)

        self.assertEqual(agent._gx_instance_effects, {})

    def test_lethal_only_board_falls_back_to_reset_not_click(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="click-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        grid = [[0 for _x in range(64)] for _y in range(64)]
        grid[5][9] = 1
        sig = agent._gx_instance_key.__globals__["shape_signature"](
            1, frozenset({(9, 5)}))
        agent._gx_lethal_sig.add(sig)
        agent._hud_identity_ready = True
        frame = FrameData(
            frame=[grid],
            state=GameState.NOT_FINISHED,
            available_actions=[GameAction.ACTION6.value],
        )

        action = agent._graph_explore_policy(grid, frame)

        self.assertIs(action, GameAction.RESET)
        self.assertEqual(agent._prev_action, "0")


if __name__ == "__main__":
    unittest.main()
