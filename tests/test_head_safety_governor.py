from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))
SPEC = importlib.util.spec_from_file_location(
    "head_governor_agent", ROOT / "agent" / "my_agent.py")
assert SPEC is not None and SPEC.loader is not None
AGENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGENT)


class SwitchActionProfileTests(unittest.TestCase):
    def test_cardinal_click_profile_can_activate_switch_head(self) -> None:
        self.assertTrue(AGENT.switch_action_profile_safe([1, 2, 3, 4, 6]))

    def test_extra_alias_does_not_hide_complete_cardinal_profile(self) -> None:
        self.assertTrue(
            AGENT.switch_action_profile_safe([1, 2, 3, 4, 6, 7]))

    def test_transform_profile_is_adversarially_rejected(self) -> None:
        self.assertFalse(
            AGENT.switch_action_profile_safe([1, 2, 3, 4, 5, 6]))

    def test_incomplete_cardinal_profile_is_adversarially_rejected(self) -> None:
        self.assertFalse(AGENT.switch_action_profile_safe([3, 4, 6, 7]))

    def test_enum_members_are_normalized(self) -> None:
        self.assertTrue(AGENT.switch_action_profile_safe([
            AGENT.GameAction.ACTION1,
            AGENT.GameAction.ACTION2,
            AGENT.GameAction.ACTION3,
            AGENT.GameAction.ACTION4,
            AGENT.GameAction.ACTION6,
        ]))


if __name__ == "__main__":
    unittest.main()
