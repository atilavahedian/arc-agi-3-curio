"""Regression tests for the symbolic card-editor rule engine."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"


def load_agent_class():
    sys.path.insert(0, str(FRAMEWORK))
    spec = importlib.util.spec_from_file_location(
        "curio_editor_agent_under_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


class EditorRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_class = load_agent_class()

    def test_chained_rewrite_must_match_target_card_type(self) -> None:
        agent = self.agent_class.__new__(self.agent_class)
        pitch = 7
        grid = [[0 for _x in range(64)] for _y in range(64)]
        a_sig = (101,)
        b_sig = (202,)
        c_sig = (303,)
        wrong_sig = (404,)

        a_to_b = [(0, 10, a_sig)]
        b_result = [(14, 7, b_sig)]
        b_to_c = [(0, 7, b_sig)]
        c_result = [(14, 11, c_sig)]
        reference = [(0, 10, a_sig)]
        mutable = [(0, 11, wrong_sig)]
        rows = [
            (5, a_to_b),
            (5, b_result),
            (15, b_to_c),
            (15, c_result),
            (30, reference),
            (40, mutable),
        ]

        # Non-uniform gaps mark the first four runs as two rule pairs.
        grid[5][7] = 1
        grid[15][7] = 1

        goals = agent._ed_infer_goals(grid, pitch, rows, 40, mutable)

        self.assertEqual(goals, [(c_sig,)])
        self.assertNotIn((b_sig,), goals)

    @staticmethod
    def _sig(index: int) -> tuple[int]:
        return (index,)

    @classmethod
    def _cycle(cls) -> dict[tuple[int], tuple[int]]:
        return {
            cls._sig(i): cls._sig(i % 7 + 1) for i in range(1, 8)
        }

    @classmethod
    def _run(cls, x: int, ring: int, *values: int):
        return [
            (x + 7 * i, ring, cls._sig(value))
            for i, value in enumerate(values)
        ]

    def test_meta_marker_and_mutation_gate_cover_multi_card_side(self) -> None:
        pitch = 7
        for n in (1, 2):
            with self.subTest(cards=n):
                grid = [[2 for _x in range(64)] for _y in range(64)]
                key = (10, 20, n)
                run = self._run(20, 7, *([1] * n))
                x0, x1 = 21, 20 + n * pitch - 2
                for x in range(x0, x1 + 1):
                    grid[7][x] = 0
                    grid[19][x] = 0
                grid[8][x0] = grid[8][x1] = 0
                grid[18][x0] = grid[18][x1] = 0

                marker = self.agent_class._ed_meta_marker(
                    grid, pitch, {key: run})

                self.assertEqual(marker, key)
                self.assertTrue(self.agent_class._ed_meta_rule_hit(
                    (21, 11, x1, 15), key, pitch))
                # A normal editor mutation on the bottom answer row must not
                # activate altered-rule mode.
                self.assertFalse(self.agent_class._ed_meta_rule_hit(
                    (21, 52, 25, 56), key, pitch))

    def test_meta_depth_one_synthesizes_uniform_side_offsets(self) -> None:
        a, b = 10, 7
        keys = [(5 + 10 * (i // 2), 2 + 20 * (i % 2), 1)
                for i in range(8)]
        rules = [
            (keys[0], keys[1], self._run(2, a, 2),
             self._run(22, b, 1)),
            (keys[2], keys[3], self._run(2, a, 2),
             self._run(22, b, 6, 6)),
            (keys[4], keys[5], self._run(2, a, 3, 3),
             self._run(22, b, 3)),
            (keys[6], keys[7], self._run(2, a, 5),
             self._run(22, b, 2)),
        ]
        layout = {
            "rules": rules,
            "source": self._run(4, a, 5, 1, 4, 4, 2),
            "target": self._run(4, b, 3, 5, 5, 1, 2),
            "path": (a, b),
        }

        solutions = self.agent_class._ed_meta_solutions(
            layout, {a: self._cycle(), b: self._cycle()})

        self.assertGreaterEqual(len(solutions), 2)
        self.assertEqual(sum(solutions[0].values()), 20)
        # The repeated B6,B6 side must keep one shared offset, rather than
        # allowing its two cards to be edited independently.
        self.assertEqual(solutions[0][keys[3]], 6)

    def test_meta_depth_two_synthesizes_composed_rules(self) -> None:
        a, b, c = 10, 7, 11
        keys = [(5 + 12 * (i // 4), 2 + 10 * (i % 4), 1)
                for i in range(12)]
        rules = [
            (keys[0], keys[1], self._run(2, a, 3),
             self._run(12, b, 6, 1)),
            (keys[2], keys[3], self._run(22, b, 4),
             self._run(32, c, 2)),
            (keys[4], keys[5], self._run(2, a, 7),
             self._run(12, b, 5, 3)),
            (keys[6], keys[7], self._run(22, b, 5),
             self._run(32, c, 5)),
            (keys[8], keys[9], self._run(2, a, 5),
             self._run(12, b, 3, 3)),
            (keys[10], keys[11], self._run(22, b, 7),
             self._run(32, c, 6)),
        ]
        layout = {
            "rules": rules,
            "source": self._run(4, a, 7, 6, 1),
            "target": self._run(4, c, 3, 1, 5, 5, 1, 3),
            "path": (a, b, c),
        }

        solutions = self.agent_class._ed_meta_solutions(
            layout,
            {a: self._cycle(), b: self._cycle(), c: self._cycle()},
        )

        self.assertGreater(len(solutions), 100)
        self.assertEqual(sum(solutions[0].values()), 17)
        self.assertTrue(self.agent_class._ed_meta_cycle_ready(self._cycle()))


if __name__ == "__main__":
    unittest.main()
