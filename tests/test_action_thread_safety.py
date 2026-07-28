"""Regression test for ACTION6 payload corruption in official Swarm threads."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ActionThreadSafetyTests(unittest.TestCase):
    def test_official_request_path_is_thread_local_with_curio(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_action_thread_safety.py"),
                "--iterations",
                "64",
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


if __name__ == "__main__":
    unittest.main()
