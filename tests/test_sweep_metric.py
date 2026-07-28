"""Regression tests for the public-game score scale used by sweep tooling."""
from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_sweep_module():
    spec = importlib.util.spec_from_file_location(
        "curio_sweep_under_test", ROOT / "scripts" / "sweep.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SweepMetricScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sweep = load_sweep_module()

    def test_complete_game_remains_100_percentage_points(self) -> None:
        self.assertEqual(
            self.sweep.game_score([100.0] * 6, [True] * 6), 100.0
        )

    def test_one_of_eight_complete_is_capped_at_2_7778_points(self) -> None:
        score = self.sweep.game_score(
            [100.0] + [0.0] * 7, [True] + [False] * 7
        )
        self.assertAlmostEqual(score, 100.0 / 36.0)

    def test_tracked_sweeps_store_percentage_points_without_dividing(self) -> None:
        for name in (
            "baseline_head.json",
            "generic_only_head.json",
            "sort_fix.json",
        ):
            data = json.loads((ROOT / "runs" / name).read_text())
            self.assertNotIn("leaderboard_estimate", data)
            self.assertEqual(data["public_game_score_percent"], data["aggregate"])


if __name__ == "__main__":
    unittest.main()
