"""Regression test for ACTION6 payload corruption in official Swarm threads."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ActionThreadSafetyTests(unittest.TestCase):
    def assert_thread_safe(self, source: Path) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_action_thread_safety.py"),
                "--iterations",
                "64",
                "--agent-source",
                str(source),
                "--json",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["baseline"]["first_action_mismatches"], 64
        )
        self.assertEqual(result["fixed"]["first_action_mismatches"], 0)
        self.assertEqual(result["fixed"]["second_action_mismatches"], 0)
        self.assertEqual(result["processes"]["action_mismatches"], 0)

    def test_official_request_path_is_thread_local_with_curio(self) -> None:
        self.assert_thread_safe(ROOT / "agent" / "my_agent.py")

    def test_original_v7_candidate_is_thread_local(self) -> None:
        self.assert_thread_safe(ROOT / "baselines" / "curio_v7_threadsafe.py")


if __name__ == "__main__":
    unittest.main()
