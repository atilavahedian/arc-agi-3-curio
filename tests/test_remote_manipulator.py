"""Generic regression tests for Curio's frame-derived manipulator head."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"


def load_agent_module():
    sys.path.insert(0, str(FRAMEWORK))
    spec = importlib.util.spec_from_file_location(
        "curio_manipulator_under_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blank():
    return [[0 for _x in range(64)] for _y in range(64)]


def paint(grid, color, x, y, width, height):
    for py in range(y, y + height):
        for px in range(x, x + width):
            grid[py][px] = color


def panel(grid, *, second_mode=False):
    # Opposite controls intentionally share appearances.  Their anchors are
    # the only identity evidence available to a generic visual learner.
    for color, x in ((2, 10), (3, 15), (3, 20), (2, 25)):
        paint(grid, color, x, 50, 4, 4)
    paint(grid, 4, 13, 45, 9, 3)
    if second_mode:
        paint(grid, 5, 13, 56, 9, 3)


def scene(body_x, *, body_width=5, second_mode=False):
    grid = blank()
    panel(grid, second_mode=second_mode)
    paint(grid, 9, body_x, 20, body_width, 5)
    return grid


class RemoteTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_agent_module()

    def test_accepts_one_sizeable_remote_body(self):
        before = scene(30)
        after = scene(34)
        clicked = frozenset(
            (x, y) for y in range(50, 54) for x in range(10, 14)
        )

        moved = self.module.remote_translation(before, after, clicked)

        self.assertIsNotNone(moved)
        self.assertEqual(moved[0], (4, 0))
        self.assertEqual(moved[2], (30, 20))
        self.assertEqual(moved[4], (34, 20))

    def test_rejects_aggregate_motion_of_thin_strips(self):
        before = blank()
        after = blank()
        for y in (20, 30):
            paint(before, 7, 20, y, 6, 2)
            paint(after, 7, 24, y, 6, 2)

        self.assertIsNone(self.module.remote_translation(before, after))

    def test_rejects_a_body_that_was_clicked_directly(self):
        before = scene(30)
        after = scene(34)
        body = frozenset(
            (x, y) for y in range(20, 25) for x in range(30, 35)
        )

        self.assertIsNone(
            self.module.remote_translation(before, after, body)
        )


class PanelGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_agent_module()
        cls.agent_class = cls.module.MyAgent

    def test_regular_panel_keeps_lookalike_instances_distinct(self):
        grid = scene(30)
        comps = self.module.components(grid)
        agent = self.agent_class.__new__(self.agent_class)
        clicked = agent._mp_id(comps, (11, 51))

        found = agent._mp_panel(comps, clicked, (4, 0))

        self.assertIsNotNone(found)
        actions, mode = found
        self.assertEqual(len(actions), 5)
        square_ids = [ident for ident in actions if ident != mode]
        self.assertEqual(len({ident[1] for ident in square_ids}), 4)
        lookalike = [ident for ident in square_ids
                     if ident[0] == clicked[0]]
        self.assertEqual(len(lookalike), 2)
        self.assertNotEqual(lookalike[0], lookalike[1])

    def test_panel_requires_exactly_one_adjacent_mode_control(self):
        grid = scene(30, second_mode=True)
        comps = self.module.components(grid)
        agent = self.agent_class.__new__(self.agent_class)
        clicked = agent._mp_id(comps, (11, 51))

        self.assertIsNone(agent._mp_panel(comps, clicked, (4, 0)))


class ManipulatorLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_agent_module()
        cls.agent_class = cls.module.MyAgent

    def make_agent(self, before):
        agent = self.agent_class.__new__(self.agent_class)
        agent._prev_grid = before
        agent._mp_votes = defaultdict(Counter)
        agent._mp_delta = {}
        agent._mp_actions = {}
        agent._mp_mode_button = None
        agent._mp_quarantine = set()
        agent._mp_engaged = False
        agent._mp_modes = set()
        agent._mp_state = None
        agent._mp_start = None
        agent._mp_body = frozenset()
        agent._mp_edges = {}
        agent._mp_pending = None
        agent._mp_last_mode = None
        agent._mp_resync = set()
        agent._mp_blocked = set()
        agent._sw_recs = {}
        agent._sw_probe = Counter()
        agent._sw_belief = {}
        agent._sw_plan = deque()
        agent._sw_nogoal = None
        agent._walls = set()
        agent._wall_bounces = Counter()
        return agent

    def test_second_consistent_translation_claims_and_quarantines_panel(self):
        before, middle, after = scene(30), scene(34), scene(38)
        agent = self.make_agent(before)
        click = (11, 51)
        comps = self.module.components(before)
        ident = agent._mp_id(comps, click)

        self.assertFalse(agent._learn_manip(
            middle, click, comps, ident[0]
        ))
        # Simulate the tentative cyclic-switch record created after the first
        # observation.  Confirmation must remove it from that competing model.
        agent._sw_recs[ident[0]] = {
            "mask": set(), "events": [(frozenset(), (), (), ())],
            "overlap": False,
        }
        agent._prev_grid = middle
        claimed = agent._learn_manip(
            after, click, self.module.components(middle), ident[0]
        )

        self.assertTrue(claimed)
        self.assertTrue(agent._mp_engaged)
        self.assertEqual(len(agent._mp_actions), 5)
        self.assertEqual(agent._mp_delta[ident], (4, 0))
        self.assertNotIn(ident[0], agent._sw_recs)

    def test_mode_edge_repairs_only_after_next_real_translation(self):
        before, middle, after = scene(30), scene(34), scene(38)
        agent = self.make_agent(before)
        click = (11, 51)
        ident = agent._mp_id(self.module.components(before), click)
        agent._learn_manip(
            middle, click, self.module.components(before), ident[0]
        )
        agent._prev_grid = middle
        agent._learn_manip(
            after, click, self.module.components(middle), ident[0]
        )
        old_state = agent._mp_state
        mode_id = agent._mp_mode_button
        mode_click = agent._mp_actions[mode_id]

        morphed = scene(38, body_width=6)
        agent._prev_grid = after
        agent._mp_pending = (old_state, mode_id, agent._mp_body)
        self.assertTrue(agent._learn_manip(
            morphed, mode_click, self.module.components(after), mode_id[0]
        ))
        self.assertEqual(agent._mp_edges[(old_state, mode_id)], old_state)

        moved = scene(42, body_width=6)
        agent._prev_grid = morphed
        agent._mp_pending = (old_state, ident, agent._mp_body)
        self.assertTrue(agent._learn_manip(
            moved, click, self.module.components(morphed), ident[0]
        ))

        repaired = agent._mp_edges[(old_state, mode_id)]
        self.assertNotEqual(repaired, old_state)
        self.assertEqual(repaired[1], old_state[1])
        self.assertNotEqual(repaired[0], old_state[0])


if __name__ == "__main__":
    unittest.main()
