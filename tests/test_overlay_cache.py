"""Regression tests for the trusted per-level overlay goal snapshot."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from collections import Counter, deque
from pathlib import Path
from unittest.mock import patch

from arcengine import FrameData, GameAction, GameState


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "vendor" / "ARC-AGI-3-Agents"


def load_agent_class():
    sys.path.insert(0, str(FRAMEWORK))
    spec = importlib.util.spec_from_file_location(
        "curio_overlay_agent_under_test", ROOT / "agent" / "my_agent.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MyAgent


class OverlayGoalCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.agent_class = load_agent_class()

    @staticmethod
    def _overlay_grid(count: int = 4) -> list[list[int]]:
        grid = [[8 for _x in range(64)] for _y in range(64)]
        centers = [(12, 12), (24, 12), (12, 24), (24, 24)][:count]
        for index, (cx, cy) in enumerate(centers):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    grid[cy + dy][cx + dx] = 3
            grid[cy][cx] = 9 if index < 2 else 11
        grid[40][40] = 0
        grid[40][39] = 13
        grid[40][41] = 13
        return grid

    def _snapshot_agent(self):
        agent = self.agent_class.__new__(self.agent_class)
        agent._ov_cached_outline = None
        agent._ov_cached_anchors = {}
        agent._ov_outline_votes = Counter()
        return agent

    def test_confirmed_goal_survives_live_ring_occlusion(self) -> None:
        agent = self._snapshot_agent()
        grid = self._overlay_grid()
        first = agent._ov_goal_snapshot(grid)
        self.assertIsNotNone(first)

        # Cover one complete ring and centre with the moving piece.  Only
        # three live anchors remain, below the four-anchor confidence gate.
        for y in range(11, 14):
            for x in range(11, 14):
                grid[y][x] = 9
        second = agent._ov_goal_snapshot(grid)

        self.assertEqual(second, first)
        self.assertEqual(sum(len(v) for v in second[1].values()), 4)
        self.assertTrue(all(isinstance(v, tuple) for v in second[1].values()))

    def test_weak_overlay_is_never_cached(self) -> None:
        agent = self._snapshot_agent()

        self.assertIsNone(agent._ov_goal_snapshot(self._overlay_grid(count=3)))
        self.assertIsNone(agent._ov_cached_outline)
        self.assertEqual(agent._ov_cached_anchors, {})

    def test_rich_overlay_without_selection_is_never_cached(self) -> None:
        agent = self._snapshot_agent()
        grid = self._overlay_grid()
        grid[40][40] = 8

        self.assertIsNone(agent._ov_goal_snapshot(grid))
        self.assertIsNone(agent._ov_cached_outline)
        self.assertEqual(agent._ov_cached_anchors, {})

    def test_inflight_plan_uses_snapshot_after_goal_is_hidden(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        for action, delta in {
            1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)
        }.items():
            agent._ov_deltas[action][delta] = 3
        agent._ov_level = 0
        agent._ov_plan = deque([4, 4])
        agent._ov_target = (46, 40)
        agent._ov_last_center = None
        frame = FrameData(
            frame=[self._overlay_grid()],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            available_actions=[1, 2, 3, 4, 5],
        )

        first_grid = self._overlay_grid()
        first_action = agent._overlay_policy(first_grid, frame)
        self.assertIs(first_action, GameAction.ACTION4)
        self.assertEqual(sum(len(v) for v in agent._ov_cached_anchors.values()), 4)

        hidden = self._overlay_grid()
        hidden[40][40] = 13
        hidden[40][43] = 0
        hidden[40][42] = 13
        hidden[40][44] = 13
        for cx, cy in ((12, 12), (24, 12)):
            for y in range(cy - 1, cy + 2):
                for x in range(cx - 1, cx + 2):
                    hidden[y][x] = 13
        fresh = self._snapshot_agent()
        self.assertIsNone(fresh._ov_goal_snapshot(hidden))

        second_action = agent._overlay_policy(hidden, frame)

        self.assertIs(second_action, GameAction.ACTION4)
        self.assertEqual(list(agent._ov_plan), [])

    def test_same_level_game_over_retains_snapshot(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        agent._ov_cached_outline = 3
        agent._ov_cached_anchors = {9: ((12, 12),)}
        obstacle = self._deform_obstacle()
        agent._ov_obstacles = (obstacle,)
        agent._ov_deform_route = True
        frame = FrameData(
            frame=[self._overlay_grid()],
            state=GameState.GAME_OVER,
            levels_completed=0,
            available_actions=[GameAction.RESET.value],
        )

        action = agent.choose_action([], frame)

        self.assertIs(action, GameAction.RESET)
        self.assertEqual(agent._ov_cached_outline, 3)
        self.assertEqual(agent._ov_cached_anchors, {9: ((12, 12),)})
        self.assertEqual(agent._ov_obstacles, (obstacle,))
        self.assertFalse(agent._ov_deform_route)

    def test_level_up_clears_snapshot(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        agent._ov_cached_outline = 3
        agent._ov_cached_anchors = {9: ((12, 12),)}
        agent._ov_obstacles = (self._deform_obstacle(),)
        agent._ov_deform_route = True
        agent._ov_catalog_recolor = True
        agent._ov_recolor_cataloged = True
        agent._ov_recolor_required = frozenset({(9, (12, 12))})
        agent._ov_recolor_assignment.append(
            (11, frozenset({(-1, 0), (1, 0)}), (20, 20), 9,
             (12, 12), frozenset({(12, 12)}), (1,))
        )
        agent._policy = lambda _grid, _frame: GameAction.ACTION1
        grid = [[8 for _x in range(64)] for _y in range(64)]
        frame = FrameData(
            frame=[grid],
            state=GameState.NOT_FINISHED,
            levels_completed=1,
            available_actions=[GameAction.ACTION1.value],
        )

        agent.choose_action([], frame)

        self.assertIsNone(agent._ov_cached_outline)
        self.assertEqual(agent._ov_cached_anchors, {})
        self.assertEqual(agent._ov_obstacles, tuple())
        self.assertFalse(agent._ov_deform_route)
        self.assertFalse(agent._ov_catalog_recolor)
        self.assertFalse(agent._ov_recolor_cataloged)
        self.assertEqual(len(agent._ov_recolor_assignment), 0)
        self.assertEqual(agent._ov_recolor_required, frozenset())

    @staticmethod
    def _hollow_piece_grid(missing_tip: bool = True) -> list[list[int]]:
        grid = [[8 for _x in range(64)] for _y in range(64)]
        cx, cy = 39, 30
        grid[cy][cx] = 0
        for dy in range(-9, 10):
            for dx in range(-9, 10):
                if abs(dx) + abs(dy) == 9:
                    grid[cy + dy][cx + dx] = 13
        if missing_tip:
            grid[cy][cx + 9] = 9
        # A previously placed piece crosses only one side of the search area;
        # raw nearest-colour and cumulative-window readers choose it wrongly.
        for distance in range(1, 10):
            grid[cy - distance][cx - distance] = 12
        # Remote same-colour anchor pixels must not join the moving body.
        for point in ((5, 5), (10, 10), (55, 55)):
            grid[point[1]][point[0]] = 13
        return grid

    def test_symmetric_shell_finds_hollow_selected_piece(self) -> None:
        agent = self._snapshot_agent()

        selected = agent._ov_selected(self._hollow_piece_grid(), outline=4)

        self.assertEqual(selected, ((39, 30), 13))

    def test_adjacent_line_pair_is_a_valid_selected_piece(self) -> None:
        agent = self._snapshot_agent()
        grid = [[8 for _x in range(64)] for _y in range(64)]
        grid[30][30] = 0
        grid[30][29] = 13
        grid[30][31] = 13
        grid[28][28] = 12  # closer-shell noise without an opposite partner

        self.assertEqual(agent._ov_selected(grid, outline=4), ((30, 30), 13))

    def test_equal_symmetric_arm_candidates_are_rejected(self) -> None:
        agent = self._snapshot_agent()
        grid = [[8 for _x in range(64)] for _y in range(64)]
        grid[30][30] = 0
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            grid[30 + dy][30 + dx] = 13
        for dx, dy in ((-2, -2), (2, 2), (-2, 2), (2, -2)):
            grid[30 + dy][30 + dx] = 12

        self.assertIsNone(agent._ov_selected(grid, outline=4))

    def test_disconnected_hollow_footprint_uses_nearest_component(self) -> None:
        grid = self._hollow_piece_grid()
        funcs = self.agent_class._ov_selected.__globals__
        footprint = funcs["piece_footprint"](grid, (39, 30), 13)
        expected = {(39, 30)} | {
            (39 + dx, 30 + dy)
            for dy in range(-9, 10) for dx in range(-9, 10)
            if abs(dx) + abs(dy) == 9 and (dx, dy) != (9, 0)
        }

        self.assertEqual(footprint, frozenset(expected))
        self.assertEqual(len(footprint), 36)
        self.assertTrue({(5, 5), (10, 10), (55, 55)}.isdisjoint(footprint))
        target = funcs["cover_centers"](
            footprint, (39, 30), [(21, 3), (27, 9), (12, 12)], 3,
            (39, 30),
        )
        self.assertEqual(target, (21, 12))

    def test_pristine_hollow_footprint_keeps_entire_ring(self) -> None:
        grid = self._hollow_piece_grid(missing_tip=False)
        footprint = self.agent_class._ov_selected.__globals__["piece_footprint"](
            grid, (39, 30), 13
        )

        self.assertEqual(len(footprint), 37)
        self.assertIn((48, 30), footprint)

    def test_connected_footprint_and_missing_arm_keep_old_behavior(self) -> None:
        func = self.agent_class._ov_selected.__globals__["piece_footprint"]
        grid = [[8 for _x in range(64)] for _y in range(64)]
        center = (30, 30)
        grid[30][30] = 0
        expected = {center}
        for distance in range(1, 5):
            for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                point = (30 + sx * distance, 30 + sy * distance)
                grid[point[1]][point[0]] = 13
                expected.add(point)
        self.assertEqual(func(grid, center, 13), frozenset(expected))

        empty = [[8 for _x in range(64)] for _y in range(64)]
        empty[30][30] = 0
        self.assertEqual(func(empty, center, 13), frozenset({center}))

    def test_confirmed_overlay_blocks_editor_takeover(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        agent._ov_outline_votes[4] = 1
        frame = FrameData(
            frame=[self._overlay_grid()],
            state=GameState.NOT_FINISHED,
            levels_completed=1,
            available_actions=[1, 2, 3, 4, 5],
        )

        self.assertIsNone(agent._editor_policy(self._overlay_grid(), frame))
        self.assertFalse(agent._ed_engaged)

    def test_symmetric_mask_isolates_touching_same_color_piece(self) -> None:
        funcs = self.agent_class._ov_selected.__globals__
        grid = [[7 for _x in range(64)] for _y in range(64)]
        center = (30, 30)
        grid[30][30] = 0
        expected = set()
        for dx in range(-5, 6):
            if dx:
                grid[30][30 + dx] = 13
                expected.add((dx, 0))
        # A touching same-colour body is connected to the right endpoint, but
        # has no point-reflected partner around this selected hole.
        for dy in range(1, 6):
            grid[30 + dy][35] = 13

        raw = funcs["piece_footprint"](grid, center, 13)
        isolated = funcs["symmetric_piece_mask"](grid, center, 13)

        self.assertGreater(len(raw), len(expected) + 1)
        self.assertEqual(isolated, frozenset(expected))
        self.assertNotIn((0, 0), isolated)

    def test_symmetric_mask_recovers_hollow_body_without_center(self) -> None:
        func = self.agent_class._ov_selected.__globals__["symmetric_piece_mask"]
        grid = [[7 for _x in range(64)] for _y in range(64)]
        center = (30, 30)
        grid[30][30] = 0
        expected = {
            (dx, dy) for dy in range(-5, 6) for dx in range(-5, 6)
            if abs(dx) + abs(dy) == 5
        }
        for dx, dy in expected:
            grid[30 + dy][30 + dx] = 13
        # Symmetric singleton noise is closer than the ring, but it is not a
        # valid whole component and must not beat the hollow body.
        grid[28][30] = 13
        grid[32][30] = 13

        mask = func(grid, center, 13)

        self.assertEqual(mask, frozenset(expected))
        self.assertNotIn((0, 0), mask)

    def test_clipped_mask_restores_only_offscreen_arm(self) -> None:
        funcs = self.agent_class._ov_selected.__globals__
        grid = [[5 for _x in range(64)] for _y in range(64)]
        center = (54, 36)
        grid[36][54] = 0
        # A radius-13 plus whose right arm is clipped by the viewport.
        expected = set()
        for distance in range(1, 14):
            for dx, dy in ((distance, 0), (-distance, 0),
                           (0, distance), (0, -distance)):
                expected.add((dx, dy))
                x, y = center[0] + dx, center[1] + dy
                if 0 <= x < 64 and 0 <= y < 64:
                    grid[y][x] = 6
        # Remote same-colour paint fill must not be reflected into the piece.
        for y in range(55, 59):
            for x in range(29, 33):
                grid[y][x] = 6

        ordinary = funcs["symmetric_piece_mask"](grid, center, 6)
        restored = funcs["clipped_symmetric_piece_mask"](grid, center, 6)

        self.assertLess(len(ordinary), len(expected))
        self.assertEqual(restored, frozenset(expected))

    def test_clipped_mask_restores_hud_occluded_cross_endpoint(self) -> None:
        funcs = self.agent_class._ov_selected.__globals__
        grid = [[5 for _x in range(64)] for _y in range(64)]
        center = (24, 54)
        for x in range(6, 43):
            grid[54][x] = 7
        for y in range(45, 63):
            grid[y][24] = 7
        grid[54][24] = 0

        restored = funcs["clipped_symmetric_piece_mask"](grid, center, 7)
        expected = {
            (x - center[0], 0) for x in range(6, 43)
            if x != center[0]
        } | {
            (0, y - center[1]) for y in range(45, 64)
            if y != center[1]
        }

        self.assertEqual(restored, frozenset(expected))

    def test_paint_pad_reader_requires_exact_framed_components(self) -> None:
        func = self.agent_class._ov_selected.__globals__["overlay_paint_pads"]
        grid = [[5 for _x in range(64)] for _y in range(64)]
        for fill, x0, y0 in ((10, 4, 4), (12, 28, 4), (14, 52, 52)):
            for y in range(y0, y0 + 6):
                for x in range(x0, x0 + 6):
                    grid[y][x] = 2 if x in (x0, x0 + 5) \
                        or y in (y0, y0 + 5) else fill
        for y in range(28, 33):
            for x in range(52, 57):
                grid[y][x] = 3 if x in (52, 56) or y in (28, 32) else 11
        # A visually similar 7x6 frame is not an exact pad.
        for y in range(28, 34):
            for x in range(4, 11):
                grid[y][x] = 3 if x in (4, 10) or y in (28, 33) else 11

        pads = func(grid)

        self.assertEqual(pads, (
            (10, (4, 4, 9, 9)),
            (12, (28, 4, 33, 9)),
            (11, (52, 28, 56, 32)),
            (14, (52, 52, 57, 57)),
        ))

    @staticmethod
    def _deform_obstacle() -> tuple[
        frozenset[tuple[int, int]], tuple[int, int, int, int]
    ]:
        rows = (
            range(8), (0, 1, 2, 5, 6, 7), (0, 1, 6, 7), (0, 7),
            (0, 7), (0, 1, 6, 7), (0, 1, 2, 5, 6, 7), range(8),
        )
        cells = frozenset(
            (28 + x, 28 + y) for y, xs in enumerate(rows) for x in xs
        )
        return cells, (28, 28, 35, 35)

    def test_deform_obstacle_reader_is_structural(self) -> None:
        func = self.agent_class._ov_selected.__globals__[
            "overlay_deform_obstacles"
        ]
        grid = self._overlay_grid()
        obstacle = self._deform_obstacle()
        for x, y in obstacle[0]:
            grid[y][x] = 1
        # A dense edge component is HUD-like and must not join the tool set.
        for y in range(8):
            for x in range(8):
                grid[y][x] = 2

        found = func(grid, 3, {9, 11}, tuple())

        self.assertEqual(found, (obstacle,))

    def test_joint_deformation_finds_cross_and_frame_routes(self) -> None:
        func = self.agent_class._ov_selected.__globals__[
            "assign_overlay_deform_pieces"
        ]
        cross = frozenset(
            (dx, dy) for dy in range(-12, 13) for dx in range(-12, 13)
            if (dx == 0 or dy == 0) and (dx, dy) != (0, 0)
        )
        frame = frozenset(
            (dx, dy) for dy in range(-9, 10) for dx in range(-9, 10)
            if abs(dx) == 9 or abs(dy) == 9
        )
        pieces = [((15, 48), 11, frame), ((48, 15), 9, cross)]
        anchors = {
            9: ((12, 6), (9, 9), (30, 9), (12, 27)),
            11: ((45, 30), (54, 30), (45, 57), (54, 57)),
        }
        deltas = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}
        obstacles = (self._deform_obstacle(),)
        frame_route = (
            1, 1, 4, 4, 2, 4, 2, 4, 4, 4,
            4, 4, 4, 4, 1, 3, 4, 4, 4,
        )
        cross_route = (
            2, 2, 2, 2, 2, 3, 3, 1, 3, 2, 2, 1, 1,
            1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        )

        first = func(pieces, anchors, deltas, obstacles)
        second = func(
            pieces,
            {color: tuple(reversed(cells))
             for color, cells in reversed(tuple(anchors.items()))},
            deltas,
            tuple(reversed(obstacles)),
        )

        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertEqual([item[6] for item in first],
                         [frame_route, cross_route])
        self.assertEqual([item[4] for item in first],
                         [(50, 44), (18, 15)])
        self.assertEqual(sum(len(item[6]) for item in first), 45)
        covered = {
            (item[3], cell) for item in first for cell in item[5]
        }
        required = {
            (color, cell) for color, cells in anchors.items() for cell in cells
        }
        self.assertEqual(covered, required)

    def test_deformation_rejects_arbitrary_shapes_and_partial_goals(self) -> None:
        func = self.agent_class._ov_selected.__globals__[
            "assign_overlay_deform_pieces"
        ]
        deltas = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}
        obstacles = (self._deform_obstacle(),)
        arbitrary = frozenset({(-2, 0), (-1, 0), (1, 0), (2, 1)})

        self.assertIsNone(func(
            [((15, 15), 9, arbitrary)], {9: ((20, 20),)},
            deltas, obstacles,
        ))
        frame = frozenset(
            (dx, dy) for dy in range(-3, 4) for dx in range(-3, 4)
            if abs(dx) == 3 or abs(dy) == 3
        )
        self.assertIsNone(func(
            [((15, 15), 9, frame)], {8: ((63, 63),)},
            deltas, obstacles,
        ))

    def test_recolor_route_uses_geometry_and_avoids_late_overwrite(self) -> None:
        funcs = self.agent_class._ov_selected.__globals__
        cross = frozenset(
            (dx, dy) for distance in range(1, 14)
            for dx, dy in ((distance, 0), (-distance, 0),
                           (0, distance), (0, -distance))
        )
        anchors = {
            12: ((15, 18), (27, 30), (15, 43)),
            14: ((48, 21), (33, 24), (30, 39)),
        }
        pads = tuple(
            (color, (x0, y0, x0 + 5, y0 + 5))
            for color, x0, y0 in (
                (10, 4, 4), (11, 4, 54), (12, 28, 4),
                (13, 52, 4), (6, 28, 54), (14, 52, 54),
            )
        )
        deltas = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}

        targets = funcs["overlay_shape_targets"](
            cross, anchors, (54, 36), 3
        )
        plan = funcs["overlay_recolor_plan"](
            (54, 36), 6, cross, (15, 30), 12, deltas, pads
        )

        self.assertEqual(targets, ((12, (15, 30)),))
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan), 21)
        self.assertEqual(plan, [1] * 5 + [3] * 7 + [2] * 3 + [3] * 6)

    def test_joint_recolor_assignment_splits_same_color_anchors(self) -> None:
        func = self.agent_class._ov_selected.__globals__[
            "assign_overlay_recolor_pieces"
        ]
        x_mask = frozenset(
            (sx * distance, sy * distance)
            for distance in range(1, 12)
            for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1))
        )
        diamond = frozenset(
            (dx, dy) for dy in range(-9, 10) for dx in range(-9, 10)
            if abs(dx) + abs(dy) == 9
        )
        plus = frozenset(
            (dx, dy) for distance in range(1, 15)
            for dx, dy in ((distance, 0), (-distance, 0),
                           (0, distance), (0, -distance))
        )
        pieces = [
            ((24, 42), 11, x_mask),
            ((30, 18), 14, diamond),
            ((54, 33), 12, plus),
        ]
        anchors = {
            9: ((21, 6), (39, 6), (33, 45),
                (24, 51), (45, 51), (33, 60)),
            8: ((51, 27), (42, 36)),
        }
        pads = tuple(
            (color, bbox) for color, bbox in (
                (11, (3, 3, 8, 8)),
                (10, (54, 3, 59, 8)),
                (14, (3, 27, 8, 32)),
                (9, (3, 52, 8, 57)),
                (8, (54, 52, 59, 57)),
            )
        )
        deltas = {1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)}

        first = func(pieces, anchors, deltas, pads)
        second = func(
            pieces,
            {color: tuple(reversed(cells))
             for color, cells in reversed(tuple(anchors.items()))},
            deltas,
            tuple(reversed(pads)),
        )

        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertEqual(
            [(item[3], item[4]) for item in first],
            [(9, (30, 15)), (8, (51, 36)), (9, (33, 51))],
        )
        self.assertEqual(sum(len(item[6]) for item in first), 61)
        covered = {
            (item[3], cell) for item in first for cell in item[5]
        }
        required = {
            (color, cell) for color, cells in anchors.items() for cell in cells
        }
        self.assertEqual(covered, required)

    def test_joint_recolor_accepts_pad_aware_translation(self) -> None:
        func = self.agent_class._ov_selected.__globals__[
            "assign_overlay_recolor_pieces"
        ]
        mask = frozenset({(-1, 0), (1, 0)})
        pieces = [((10, 10), 9, mask), ((20, 20), 9, mask)]
        deltas = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        pads = ((9, (30, 30, 35, 35)),)

        # Even without a colour change, this path remains useful on a
        # pad-bearing board: its routes prove that no wrong pad is crossed.
        assignment = func(
            pieces, {9: ((9, 10), (11, 10))}, deltas, pads
        )
        self.assertIsNotNone(assignment)
        self.assertEqual(
            {(item[3], cell) for item in assignment for cell in item[5]},
            {(9, (9, 10)), (9, (11, 10))},
        )
        # Full coverage is mandatory, and the hard goal-width cap is shared
        # with the ordinary overlay assignment.
        impossible = {8: ((5, 5),)}
        self.assertIsNone(func(pieces, impossible, deltas, pads))
        too_many = {8: tuple((x + 10, 10) for x in range(17))}
        self.assertIsNone(func(pieces, too_many, deltas, pads))

    def test_joint_assignment_covers_all_same_color_anchors(self) -> None:
        func = self.agent_class._ov_selected.__globals__["assign_overlay_pieces"]
        horizontal = frozenset({(-2, 0), (-1, 0), (1, 0), (2, 0)})
        vertical = frozenset({(0, -2), (0, -1), (0, 1), (0, 2)})
        pieces = [((10, 20), 9, horizontal), ((20, 20), 9, vertical)]
        goals = (
            (8, 10), (9, 10), (11, 10), (12, 10),
            (20, 8), (20, 9), (20, 11), (20, 12),
        )

        first = func(pieces, {9: goals}, step=1)
        second = func(pieces, {9: tuple(reversed(goals))}, step=1)

        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        covered = set().union(*(placement[3] for placement in first))
        self.assertEqual(covered, set(goals))
        self.assertTrue(all((0, 0) not in placement[1] for placement in first))

    def test_joint_assignment_rejects_partial_and_color_mismatch(self) -> None:
        func = self.agent_class._ov_selected.__globals__["assign_overlay_pieces"]
        mask = frozenset({(-1, 0), (1, 0)})
        pieces = [((2, 2), 9, mask), ((4, 4), 9, mask)]

        self.assertIsNone(func(pieces, {9: ((0, 0),)}, step=2))
        self.assertIsNone(func(pieces, {10: ((3, 2),)}, step=1))
        too_many = tuple((x + 10, 10) for x in range(17))
        self.assertIsNone(func(pieces, {9: too_many}, step=1))

    def test_hidden_hole_midplan_cannot_emit_select(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        agent._ov_level = 0
        agent._ov_cached_outline = 4
        agent._ov_cached_anchors = {9: ((10, 10), (20, 20), (30, 30), (40, 40))}
        agent._ov_select = GameAction.ACTION5.value
        agent._ov_target = (30, 30)
        agent._ov_plan = deque([1, 1])
        agent._ov_deltas[1][(0, -3)] = 3
        grid = [[8 for _x in range(64)] for _y in range(64)]
        frame = FrameData(
            frame=[grid],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            available_actions=[1, 2, 3, 4, 5],
        )

        action = agent._overlay_policy(grid, frame)

        self.assertIs(action, GameAction.ACTION1)
        self.assertIsNot(action, GameAction.ACTION5)
        self.assertEqual(list(agent._ov_plan), [1])

    def test_recolor_route_can_cross_target_before_final_color(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        for action, delta in {
            1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)
        }.items():
            agent._ov_deltas[action][delta] = 3
        mask = frozenset({(-1, 0), (1, 0)})
        agent._ov_level = 0
        agent._ov_select = GameAction.ACTION5.value
        # The first leg has already reached the final coordinate, but the
        # proved route must still touch a paint pad and return before success.
        agent._ov_target = (40, 40)
        agent._ov_expected_arm = 9
        agent._ov_plan = deque([GameAction.ACTION4.value,
                                GameAction.ACTION3.value])
        agent._ov_recolor_assignment = deque([
            (13, mask, (37, 40), 9, (40, 40),
             frozenset({(12, 12)}),
             (GameAction.ACTION4.value, GameAction.ACTION4.value,
              GameAction.ACTION3.value)),
        ])
        agent._ov_recolor_required = frozenset({(9, (12, 12))})
        frame = FrameData(
            frame=[self._overlay_grid()],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            available_actions=[1, 2, 3, 4, 5],
        )

        action = agent._overlay_policy(self._overlay_grid(), frame)

        self.assertIs(action, GameAction.ACTION4)
        self.assertEqual(list(agent._ov_plan), [GameAction.ACTION3.value])
        self.assertEqual(len(agent._ov_recolor_assignment), 1)
        self.assertIsNone(agent._ov_benched)

    def test_deformation_can_change_shape_without_moving_hole(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        for action, delta in {
            1: (0, -3), 2: (0, 3), 3: (-3, 0), 4: (3, 0)
        }.items():
            agent._ov_deltas[action][delta] = 3
        mask = frozenset({(-1, 0), (1, 0)})
        agent._ov_level = 0
        agent._ov_select = GameAction.ACTION5.value
        agent._ov_target = (43, 40)
        agent._ov_expected_arm = 13
        agent._ov_plan = deque([GameAction.ACTION4.value])
        agent._ov_last_center = (40, 40)
        agent._ov_deform_route = True
        agent._ov_recolor_assignment = deque([
            (13, mask, (40, 40), 13, (43, 40),
             frozenset({(12, 12)}), (GameAction.ACTION4.value,)),
        ])
        agent._ov_recolor_required = frozenset({(13, (12, 12))})
        frame = FrameData(
            frame=[self._overlay_grid()],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            available_actions=[1, 2, 3, 4, 5],
        )

        action = agent._overlay_policy(self._overlay_grid(), frame)

        self.assertIs(action, GameAction.ACTION4)
        self.assertEqual(list(agent._ov_plan), [])
        self.assertEqual(len(agent._ov_recolor_assignment), 1)
        self.assertTrue(agent._ov_deform_route)
        self.assertIsNone(agent._ov_benched)

    def test_hidden_hole_during_select_cycle_keeps_ordinal(self) -> None:
        mask = frozenset({(-1, 0), (1, 0)})
        for state in ("catalog", "assignment"):
            with self.subTest(state=state), patch.dict(
                os.environ, {"CURIO_EXPLORER": "graph"}
            ):
                agent = self.agent_class(
                    card_id="test",
                    game_id="overlay-test",
                    agent_name="curio",
                    ROOT_URL="",
                    record=False,
                    arc_env=None,
                )
            agent._ov_level = 0
            agent._ov_cached_outline = 4
            agent._ov_cached_anchors = {
                9: ((10, 10), (20, 20), (30, 30), (40, 40))
            }
            agent._ov_select = GameAction.ACTION5.value
            agent._ov_deltas[1][(0, -3)] = 3
            if state == "catalog":
                agent._ov_catalog = [((20, 20), 9, mask)]
            else:
                agent._ov_assignment = deque([
                    (9, mask, (20, 20), frozenset({(19, 20), (21, 20)}))
                ])
            grid = [[8 for _x in range(64)] for _y in range(64)]
            frame = FrameData(
                frame=[grid],
                state=GameState.NOT_FINISHED,
                levels_completed=0,
                available_actions=[1, 2, 3, 4, 5],
            )

            action = agent._overlay_policy(grid, frame)

            self.assertIs(action, GameAction.ACTION1)
            self.assertIsNot(action, GameAction.ACTION5)

    def test_hidden_hole_after_recolor_target_advances_select(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        agent._ov_level = 0
        agent._ov_cached_outline = 4
        agent._ov_cached_anchors = {
            12: ((10, 10), (20, 20), (30, 30), (40, 40))
        }
        agent._ov_select = GameAction.ACTION5.value
        agent._ov_target = (30, 30)
        agent._ov_expected_arm = 12
        grid = [[8 for _x in range(64)] for _y in range(64)]
        frame = FrameData(
            frame=[grid],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            available_actions=[1, 2, 3, 4, 5],
        )

        action = agent._overlay_policy(grid, frame)

        self.assertIs(action, GameAction.ACTION5)
        self.assertIn(12, agent._ov_solved)
        self.assertIsNone(agent._ov_target)
        self.assertIsNone(agent._ov_expected_arm)

    def test_hidden_joint_recolor_records_only_completed_anchors(self) -> None:
        with patch.dict(os.environ, {"CURIO_EXPLORER": "graph"}):
            agent = self.agent_class(
                card_id="test",
                game_id="overlay-test",
                agent_name="curio",
                ROOT_URL="",
                record=False,
                arc_env=None,
            )
        mask = frozenset({(-1, 0), (1, 0), (0, -1), (0, 1)})
        first_hits = frozenset({(10, 10), (20, 10)})
        second_hits = frozenset({(10, 20), (20, 20)})
        agent._ov_level = 0
        agent._ov_cached_outline = 4
        agent._ov_cached_anchors = {
            9: tuple(first_hits | second_hits)
        }
        agent._ov_select = GameAction.ACTION5.value
        agent._ov_target = (15, 10)
        agent._ov_expected_arm = 9
        agent._ov_recolor_required = frozenset(
            (9, cell) for cell in first_hits | second_hits
        )
        agent._ov_recolor_assignment = deque([
            (11, mask, (15, 30), 9, (15, 10), first_hits, (1,)),
            (12, mask, (15, 40), 9, (15, 20), second_hits, (2,)),
        ])
        grid = [[8 for _x in range(64)] for _y in range(64)]
        frame = FrameData(
            frame=[grid],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            available_actions=[1, 2, 3, 4, 5],
        )

        action = agent._overlay_policy(grid, frame)

        self.assertIs(action, GameAction.ACTION5)
        self.assertEqual(len(agent._ov_recolor_assignment), 1)
        self.assertEqual(
            agent._ov_assigned_anchors,
            {(9, cell) for cell in first_hits},
        )
        self.assertNotIn(9, agent._ov_solved)
        self.assertEqual(len(agent._ov_recolor_required), 4)


if __name__ == "__main__":
    unittest.main()
