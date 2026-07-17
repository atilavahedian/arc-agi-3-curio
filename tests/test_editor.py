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


if __name__ == "__main__":
    unittest.main()
