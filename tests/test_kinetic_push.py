"""Focused regressions for the frame-derived selected-push solver."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"


def load_agent_module():
    sys.path.insert(0, str(FRAMEWORK))
    spec = importlib.util.spec_from_file_location(
        "curio_kinetic_push_agent_under_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KineticPushTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_agent_module()

    @staticmethod
    def _blank_grid(fill: int = 1):
        return [[fill for _x in range(64)] for _y in range(64)]

    @staticmethod
    def _draw_selector(grid, x: int, y: int, *, selected: bool = True) -> None:
        for yy in range(y, y + 3):
            for xx in range(x, x + 3):
                grid[yy][xx] = 14
        grid[y + 1][x + 1] = 0 if selected else 5

    @staticmethod
    def _draw_socket(grid, x: int, y: int) -> None:
        for yy in range(y, y + 5):
            for xx in range(x, x + 5):
                if xx in (x, x + 4) or yy in (y, y + 4):
                    grid[yy][xx] = 4

    @staticmethod
    def _entity(kind: str, pos, *, direction=None):
        return {
            "kind": kind,
            "color": 14 if kind == "control" else 13,
            "pos": pos,
            "w": 3,
            "h": 3,
            "mask": frozenset((x, y) for y in range(3) for x in range(3)),
            "dir": direction,
        }

    def _model(self, entities, *, force=frozenset(), bombs=()):
        return {
            "entities": tuple(entities),
            "controls": (0,),
            "objects": tuple(),
            "bombs": tuple(bombs),
            "control_targets": tuple(),
            "object_targets": tuple(),
            "hard": frozenset(),
            "force": frozenset(force),
            "direct_blocked": frozenset(),
        }

    def test_gate_requires_exact_actions_selected_frame_and_matching_socket(self) -> None:
        grid = self._blank_grid()
        self._draw_selector(grid, 3, 6)
        self._draw_socket(grid, 20, 18)
        grid[30][31] = 2

        setup = self.module.kp_setup(grid, {1, 2, 3, 4, 6})

        self.assertIsNotNone(setup)
        assert setup is not None
        self.assertEqual(setup["controls"], (0,))
        self.assertEqual(setup["control_targets"], ((3, 3, (21, 19)),))
        self.assertEqual(setup["start"], (((3, 6),), 0, ()))
        self.assertIn((31, 30), setup["hard"])
        self.assertIsNone(self.module.kp_setup(grid, {1, 2, 3, 4, 5, 6}))

        no_socket = self._blank_grid()
        self._draw_selector(no_socket, 3, 6)
        self.assertIsNone(
            self.module.kp_setup(no_socket, {1, 2, 3, 4, 6})
        )

    def test_gate_rejects_ambiguous_or_incidental_selector_scaffolds(self) -> None:
        grid = self._blank_grid()
        self._draw_selector(grid, 3, 6)
        self._draw_selector(grid, 9, 6)
        self._draw_socket(grid, 20, 18)
        self.assertIsNone(self.module.kp_setup(grid, {1, 2, 3, 4, 6}))

        incidental = self._blank_grid()
        self._draw_selector(incidental, 3, 6)
        incidental[6][4] = 2
        self._draw_socket(incidental, 20, 18)
        self.assertIsNone(
            self.module.kp_setup(incidental, {1, 2, 3, 4, 6})
        )

    def test_bomb_stripes_derive_direction_and_phase_from_pixels(self) -> None:
        bbox = (10, 10, 13, 13)
        cases = (
            ((0, 1), {(x, 10) for x in range(10, 14)}),
            ((0, -1), {(x, 13) for x in range(10, 14)}),
            ((1, 0), {(10, y) for y in range(10, 14)}),
            ((-1, 0), {(13, y) for y in range(10, 14)}),
        )
        for direction, stripe in cases:
            with self.subTest(direction=direction):
                grid = self._blank_grid()
                for y in range(10, 14):
                    for x in range(10, 14):
                        grid[y][x] = 12 if (x, y) in stripe else 13
                self.assertEqual(
                    self.module.kp_bomb_read(grid, bbox), (direction, 1)
                )

        uniform = self._blank_grid()
        for y in range(10, 14):
            for x in range(10, 14):
                uniform[y][x] = 13
        self.assertEqual(self.module.kp_bomb_read(uniform, bbox), (None, 0))

    def test_selected_collision_shoves_other_five_pitches(self) -> None:
        entities = [
            self._entity("control", (0, 0)),
            self._entity("object", (3, 0)),
        ]
        model = self._model(entities)
        state = (((0, 0), (3, 0)), 0, ())

        moved = self.module.kp_step(model, state, ("move", 4))

        self.assertEqual(moved[0][0], (0, 0))
        self.assertEqual(moved[0][1], (18, 0))

    def test_force_surface_extends_a_shove_until_the_entity_clears_it(self) -> None:
        entities = [
            self._entity("control", (0, 0)),
            self._entity("object", (3, 0)),
        ]
        force = frozenset((x, y) for x in range(18, 21) for y in range(3))
        model = self._model(entities, force=force)
        state = (((0, 0), (3, 0)), 0, ())

        moved = self.module.kp_step(model, state, ("move", 4))

        self.assertEqual(moved[0][0], (0, 0))
        self.assertEqual(moved[0][1], (21, 0))

    def test_selection_does_not_tick_bombs_but_cardinal_input_does(self) -> None:
        entities = [
            self._entity("control", (0, 0)),
            self._entity("control", (9, 0)),
            self._entity("bomb", (40, 40), direction=(0, 1)),
        ]
        model = self._model(entities, bombs=(2,))
        state = (((0, 0), (9, 0), (40, 40)), 0, (2,))

        selected = self.module.kp_step(model, state, ("select", 1))
        moved = self.module.kp_step(model, state, ("move", 3))

        self.assertEqual(selected, (state[0], 1, (2,)))
        self.assertEqual(moved[2], (0,))

    def test_state_validation_detects_an_unexpected_visible_boundary_bounce(self) -> None:
        grid = self._blank_grid()
        self._draw_selector(grid, 0, 6)
        entity = self._entity("control", (0, 6))
        model = self._model([entity])
        stayed = (((0, 6),), 0, ())
        predicted_offscreen = (((-3, 6),), 0, ())

        self.assertTrue(self.module.kp_state_matches(grid, model, stayed))
        self.assertFalse(
            self.module.kp_state_matches(grid, model, predicted_offscreen)
        )

    def test_solver_source_contains_no_game_or_level_identifier(self) -> None:
        source = (ROOT / "agent" / "my_agent.py").read_text()
        self.assertNotIn("ka59", source.lower())


if __name__ == "__main__":
    unittest.main()
