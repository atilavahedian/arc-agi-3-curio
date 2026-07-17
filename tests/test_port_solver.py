"""Focused regressions for the generic selected-piece port solver."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"


def load_agent_module():
    sys.path.insert(0, str(FRAMEWORK))
    spec = importlib.util.spec_from_file_location(
        "curio_port_agent_under_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PortSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_agent_module()
        cls.agent_class = cls.module.MyAgent

    @staticmethod
    def _blank_grid():
        return [[0 for _x in range(64)] for _y in range(64)]

    def test_known_selection_owns_a_credible_movement_diff(self) -> None:
        """A nearby piece may overlap more of a feedback-recoloured diff.

        Once a click has established the selected piece, a credible movement
        diff belongs to that piece rather than whichever piece has the largest
        incidental overlap with the union of changed pixels.
        """
        agent = self.agent_class.__new__(self.agent_class)
        distractor = {(10, 10), (11, 10), (12, 10), (13, 10)}
        selected = {(20, 20), (21, 20), (22, 20), (23, 20)}
        agent._pc_pieces = {
            0: {"cells": set(distractor), "ports": set()},
            1: {"cells": set(selected), "ports": set()},
        }
        agent._pc_level = 6
        agent._pc_sel = 1
        agent._pc_bounce = 3
        agent._pc_idprobe = 2
        agent._pc_strikes = 0
        agent._prev_action = "4"
        agent._prev_grid = self._blank_grid()
        # The distractor overlaps three diff pixels, while the selected piece
        # overlaps two; the selected overlap is nevertheless size-credible.
        union = frozenset({
            (13, 10), (14, 10), (15, 10),
            (23, 20), (24, 20),
        })
        agent._last_move = (4, (3, 0), union)
        agent._pc_note_feedback = lambda _grid: None
        agent._port_color = lambda: 9
        agent._fb_color = lambda: None

        agent._pc_sync(self._blank_grid(), level=6, rules={4: (3, 0)}, scale=3)

        self.assertEqual(agent._pc_pieces[0]["cells"], distractor)
        self.assertEqual(
            agent._pc_pieces[1]["cells"],
            {(23, 20), (24, 20), (25, 20), (26, 20)},
        )
        self.assertEqual(agent._pc_sel, 1)
        self.assertEqual(agent._pc_bounce, 0)

    def test_detected_stack_finishes_bounded_scan_before_solving(self) -> None:
        """A positively detected variant stack must be fully enumerated."""
        agent = self.agent_class.__new__(self.agent_class)
        pat = (
            frozenset({(0, 0)}),
            frozenset({(0, 0)}),
        )
        agent._pc_pieces = {
            0: {
                "cells": {(0, 0)}, "ports": {(0, 0)},
                "pats": [pat], "stack": True,
            },
            1: {
                "cells": {(3, 0)}, "ports": {(3, 0)},
                "pats": [pat], "stack": False,
            },
        }
        agent._pc_unreachable = set()
        agent._pc_failed_cfgs = set()
        agent._pc_vscout = Counter({0: self.module.XFORM_CAP - 1})

        self.assertEqual(agent._pc_solve(scale=3), [])

        agent._pc_vscout[0] = self.module.XFORM_CAP
        solutions = agent._pc_solve(scale=3)
        self.assertTrue(solutions)

    def test_solver_finds_global_cheapest_pairing_after_early_candidates(self) -> None:
        """Search must not stop before sorting an arbitrary first batch.

        Eight single-port pieces have 105 geometric perfect matchings.  The
        cheapest matching pairs adjacent positions, but its first pair is
        lexicographically last, so it lies beyond the historical first-twelve
        DFS cutoff.  Two fully scanned stacks activate the bounded deep search.
        """
        agent = self.agent_class.__new__(self.agent_class)
        positions = {
            0: 0, 7: 3,
            1: 12, 6: 15,
            2: 24, 5: 27,
            3: 36, 4: 39,
        }
        pieces = {}
        for pid, x in positions.items():
            cells = {(x, 0)}
            ports = {(x, 0)}
            pieces[pid] = {
                "cells": cells,
                "ports": ports,
                "pats": [self.module.pat_norm(cells, ports)],
                "stack": pid in {0, 7},
            }
        agent._pc_pieces = pieces
        agent._pc_unreachable = set()
        agent._pc_failed_cfgs = set()
        agent._pc_vscout = Counter({
            0: self.module.XFORM_CAP,
            7: self.module.XFORM_CAP,
        })

        solutions = agent._pc_solve(scale=3)

        self.assertTrue(solutions)
        self.assertEqual(
            {pid: target[1] for pid, target in solutions[0].items()},
            {
                0: (0, 0), 7: (0, 0),
                1: (12, 0), 6: (12, 0),
                2: (24, 0), 5: (24, 0),
                3: (36, 0), 4: (36, 0),
            },
        )

    def test_executor_handles_shape_mismatch_before_translation_only_work(self) -> None:
        """Transforms happen before moves that could create joint recolours."""
        agent = self.agent_class.__new__(self.agent_class)
        move_cells = {(2, 2)}
        move_ports = {(2, 2)}
        shape_cells = {(10, 10), (11, 10)}
        shape_ports = {(10, 10)}
        move_pat = self.module.pat_norm(move_cells, move_ports)
        target_shape = (
            frozenset({(0, 0), (0, 1)}),
            frozenset({(0, 0)}),
        )
        agent._pc_pieces = {
            0: {"cells": set(move_cells), "ports": set(move_ports)},
            1: {"cells": set(shape_cells), "ports": set(shape_ports)},
        }
        agent._pc_solution = {
            0: (move_pat, (5, 2)),       # translation only
            1: (target_shape, (10, 10)),  # transform required
        }
        agent._pc_sel = None
        agent._pc_bounce = 0
        agent._pc_xform_spent = 0
        agent._pc_strikes = 0
        agent._pc_unreachable = set()
        agent._pc_failed_cfgs = set()
        selected = []

        def click_cell(pid):
            selected.append(pid)
            return (20 + pid, 20)

        agent._pc_click_cell = click_cell
        agent._pc_click = lambda cell, why: (cell, why)

        result = agent._pc_exec(avail={5}, rules={4: (3, 0)}, scale=3)

        self.assertEqual(selected, [1])
        self.assertEqual(agent._pc_sel, 1)
        self.assertEqual(result[0], (21, 20))


if __name__ == "__main__":
    unittest.main()
