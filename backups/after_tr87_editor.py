"""Curio v2 — world-modeling explorer for ARC-AGI-3.

v1 (novelty search) plateaued: visiting new states isn't understanding.
v2 adds a learned world model on top of the v1 fallback:

  1. PERCEIVE  — segment the grid into colored connected components.
  2. LEARN     — diff consecutive frames: when a simple action translates
                 exactly one rigid group of components by a consistent
                 delta, that group is our avatar (multi-color sprites move
                 as several components) and the action is a movement command.
  3. MAP       — when a known movement action fails to move the avatar,
                 the blocked cells are walls.
  4. PLAN      — BFS over anchor positions using learned movement deltas,
                 toward (a) rare-color objects, then (b) unexplored space.
  5. FALLBACK  — games with no controllable avatar use v1 novelty search.
  6. AFFORD    — every click logs whether the clicked sprite class changed
                 the board; after the first GAME_OVER, exploration clicks
                 productive classes first and skips proven-dead ones (the
                 library survives death and level-ups: appearance is physics).
  7. LATTICE   — click-only games whose dynamics are in-place RECOLORS
                 (ft09 family) get a board model: detect a regular lattice
                 of same-shaped sprites, learn per-appearance recolor masks
                 and the palette cycle from click diffs, read static marker
                 sprites as ==/!= color constraints on their neighbours, and
                 solve the board exactly (greedy for identity masks, bounded
                 BFS otherwise).  Triple-gated: click-only + lattice +
                 learned recolor effect, so movement and translation games
                 never see this branch.
  8. ATTR      — movement games that carry a persistent attribute (ls20
                 family) get state-augmented planning: a compact patch that
                 repeatedly changes when tiles are entered — never moving
                 with the avatar — is an attribute register; per-tile-
                 appearance effects on it are learned, a blocked move that
                 plays a failure animation is a conditional gate (not a
                 wall), and the planner BFS-es the (anchor x attribute)
                 product graph: touch the shifter once more, then enter the
                 pad.  A budget strip (one masked HUD color draining at a
                 constant per-action rate) rides along in the search state —
                 refill tiles reset it and dying branches are pruned.  Gated
                 on trusted movement rules + a detected register, so click-
                 only games and plain mazes never see this branch.
  9. PORT      — games with clicks AND full 2D movement (cn04 family) get a
                 selected-piece model: clicks choose which multi-color piece
                 the arrows move, a non-movement simple action transforms it
                 in place (rotation / variant cycle), and rare accent-color
                 blobs recurring on several pieces are connection ports.
                 The goal hypothesis — overlap ports pairwise — is solved
                 exactly by a pairing DFS over (piece, orientation,
                 translation) placements; port TYPES are invisible (two
                 marker types render alike), so geometric solutions are
                 played out cheapest-first and falsified by the level not
                 advancing.  Gated on clicks + four trusted axis moves + a
                 voted port color + no attribute register, so click-only
                 games, ls20 and plain mazes never see this branch.
 10. EDIT      — simple-action-only games whose verbs are a cursor and an
                 in-place glyph cycler (tr87 family) get a factored
                 editor model: classify each action's diff as SEL_MOVE (a
                 small marker translates cleanly) or MUTATE (one compact
                 interior bbox changes in place), read the board as
                 "cards" (backing tiles with a monochrome border ring
                 holding a glyph) under rotation-canonical window
                 signatures, mine rewrite rules from same-row card runs
                 whose gap shows a separator marker, rewrite the frozen
                 reference row through them to get the goal configuration,
                 and drive cursor+cycle presses slot by slot until the
                 mutable row matches.  Gated on no clicks + a trusted
                 mutate verb + no movement rules at engage time, so click
                 games and avatar games never see this branch.
"""
from __future__ import annotations

import os
import random
import zlib
from collections import Counter, defaultdict, deque
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

GRID = 64
STUCK_LIMIT = 12
VOTE_THRESHOLD = 3   # consistent observations before trusting a movement rule
PLANNER_PATIENCE = 30   # planner steps without novelty before exploration takes over
PLANNER_COOLDOWN = 150  # exploration-only steps after the planner is benched
NOVELTY_WARMUP = 500    # pure-exploration opening; the world model learns passively
                        # (v1 solved lp85 inside 400 steps — don't let the planner
                        # preempt cheap wins exploration gets for free)
HUD_BAND = 3            # only cells within this many px of the frame edge are maskable
HUD_RATE = 0.4          # change rate above which a border cell counts as HUD chrome
HUD_WARMUP = 30         # diffed frames before the mask may activate
HUD_RECOMPUTE = 32      # mask recompute cadence, in diffed frames
HUD_FREEZE = 300        # diffed frames after which the mask is frozen for the run
GROUP_BBOX = 20         # mover-group union bbox cap (px): avatars are compact,
                        # large synchronized animations are not (cn04's
                        # measured piece is 15x18, so 12 was too tight)
WALL_TRUST = 3          # bounces before a wall survives a world-state change
FLOOD_SHARE = 0.8       # single-color frame share that marks a death flash
CLICK_PROBES = 4        # tries before a click signature may be declared dead
CLICK_DEAD = 0.1        # Laplace P(effect) below which a probed signature drops
LATTICE_MIN_CELLS = 4   # same-shape, same-phase sprites before a board is trusted
LATTICE_PROBE_CAP = 8   # probe clicks per appearance class before giving up
                        # (must outlast ft09 level 1's 4-frame hint animation,
                        # which consumes clicks and garbles their diffs)
LATTICE_DEATH_CAP = 3   # in-level GAME_OVERs after engaging before the lattice
                        # solver is benched (replays are deterministic)
LATTICE_BFS_CAP = 6000  # BFS state budget for neighborhood-mask boards
ATTR_BBOX = 16          # attribute-register bbox cap (px): a HUD glyph box
                        # is compact; death repaints (lives pips + restored
                        # rings + glyph reset together) span the frame and
                        # must never vote
ATTR_VOTES = 2          # changes at one cell before it joins the register
ATTR_EVENTS = 2         # compact register events before the model engages
GATE_GIVEUP = 8         # attributes a gate may reject before it's a wall
PROBE_TRIES = 3         # routes onto one tile class before giving up on it
                        # (blocked classes — locked pads — never enter, so
                        # the cap is what stops a probe loop)
BUDGET_OBS = 6          # consistent one-color drains before the masked
                        # strip is trusted as a step budget
RULE_TRIES = 12         # uses an available simple action gets before the
                        # attr planner may engage without a rule for it —
                        # a planner that monopolizes the policy too early
                        # starves rule discovery (measured: ACTION4 stayed
                        # untrusted for 3000 steps on ls20)
ATTR_BFS_CAP = 4000     # product-graph state budget (anchor x attribute)
PORT_VOTES = 4          # frames a color must qualify on 2+ pieces before
                        # it is trusted as the port accent color
PC_SOLS = 12            # pairing solutions kept per solve
PC_NODES = 20000        # pairing-DFS node budget
PC_SCOUT = 2            # selection clicks probing one piece for hidden ports
PC_STRIKES = 8          # exec failures / desyncs before the port planner
                        # benches the level (novelty search takes over)
XFORM_CAP = 10          # in-place transform presses chasing one pattern
                        # (ping-pong variant stacks need up to 2n-2)
ED_VOTES = 3            # classifications before an editor verb is trusted
ED_PROBES = 5           # policy presses probing one unclassified action
ED_MUT_BBOX = 24        # interior union-bbox cap for an in-place mutate
                        # (one glyph slot, or one multi-glyph rule side)
ED_CYCLE_CAP = 9        # cycle presses chasing one slot's goal signature
                        # (a 7-variant ring needs at most 6)
ED_MISS_CAP = 4         # consecutive cursor/slot losses before a strike
ED_STRIKES = 10         # model failures before the editor benches a level
ED_GOALS = 6            # goal candidates (rule segmentations) kept

Grid = list[list[int]]
Cell = tuple[int, int]


def grid_of(frame: FrameData) -> Optional[Grid]:
    return frame.frame[-1] if not frame.is_empty() else None


def grid_hash(grid: Optional[Grid]) -> int:
    return hash(tuple(tuple(row) for row in grid)) if grid else 0


def flood_color(grid: Grid) -> Optional[int]:
    """The color covering more than FLOOD_SHARE of the frame, if any."""
    counts: Counter[int] = Counter()
    for row in grid:
        counts.update(row)
    color, n = counts.most_common(1)[0]
    return color if n > GRID * GRID * FLOOD_SHARE else None


def components(grid: Grid) -> list[tuple[int, frozenset[Cell]]]:
    """4-connected components, skipping the background (most common) color."""
    counts: Counter[int] = Counter()
    for row in grid:
        counts.update(row)
    background = counts.most_common(1)[0][0]
    seen = [[False] * GRID for _ in range(GRID)]
    out: list[tuple[int, frozenset[Cell]]] = []
    for y in range(GRID):
        for x in range(GRID):
            if seen[y][x] or grid[y][x] == background:
                continue
            color = grid[y][x]
            cells, dq = [], deque([(x, y)])
            seen[y][x] = True
            while dq:
                cx, cy = dq.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)):
                    if 0 <= nx < GRID and 0 <= ny < GRID and not seen[ny][nx] \
                            and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        dq.append((nx, ny))
            out.append((color, frozenset(cells)))
    return out


def translation(a: frozenset[Cell], b: frozenset[Cell]) -> Optional[Cell]:
    """If b is exactly a shifted by (dx, dy), return that delta."""
    if len(a) != len(b):
        return None
    ax, ay = min(a)
    bx, by = min(b)
    dx, dy = bx - ax, by - ay
    if all((x + dx, y + dy) in b for x, y in a):
        return (dx, dy)
    return None


def shape_signature(color: int, cells: frozenset[Cell]) -> int:
    """Appearance hash of one component: cells normalized to the bbox
    origin, canonical min over 4 rotations, plus color — identical sprites
    hash equal wherever (and on whichever level) they appear."""
    pts = list(cells)
    variants = []
    for _ in range(4):
        mx = min(x for x, _y in pts)
        my = min(y for _x, y in pts)
        variants.append(tuple(sorted((x - mx, y - my) for x, y in pts)))
        pts = [(-y, x) for x, y in pts]  # rotate 90°
    return hash((color, min(variants)))


def signature_under(
    comps: list[tuple[int, frozenset[Cell]]], cell: Cell
) -> int:
    """Appearance hash of the component under a click; background → 0."""
    for color, cells in comps:
        if cell in cells:
            return shape_signature(color, cells)
    return 0


def moved_objects(prev: Grid, cur: Grid) -> list[tuple[int, Cell, frozenset[Cell]]]:
    """Objects translated between frames: (color, delta, new_cells)."""
    prev_by_color: dict[int, list[frozenset[Cell]]] = defaultdict(list)
    for color, cells in components(prev):
        prev_by_color[color].append(cells)
    # nearest match (not first match), consensus-tie-broken: interchangeable
    # same-shape twins (cn04's two 3x3 ports, 6px apart, stepping 3px)
    # otherwise alias to a neighbour's position and fake a second delta
    pending: list[tuple[int, frozenset[Cell], list[Cell]]] = []
    consensus: Counter[Cell] = Counter()  # delta → cells of unambiguous movers
    for color, cells in components(cur):
        for old in prev_by_color.get(color, []):
            if old == cells:
                break
        else:
            cands = [d for old in prev_by_color.get(color, [])
                     if (d := translation(old, cells)) and d != (0, 0)]
            if cands:
                pending.append((color, cells, cands))
                if len(cands) == 1:
                    consensus[cands[0]] += len(cells)
    moves = []
    for color, cells, cands in pending:
        best = min(cands, key=lambda d: (abs(d[0]) + abs(d[1]), -consensus[d]))
        moves.append((color, best, cells))
    return moves


def recolored_objects(prev: Grid, cur: Grid) -> list[tuple[frozenset[Cell], int, int]]:
    """Objects recolored IN PLACE between frames: (cells, old, new).  The
    in-place complement of moved_objects — ft09-family dynamics are pure
    recolors, which translation diffing cannot represent at all."""
    prev_by_cells: dict[frozenset[Cell], int] = {
        cells: color for color, cells in components(prev)}
    out: list[tuple[frozenset[Cell], int, int]] = []
    for color, cells in components(cur):
        old = prev_by_cells.get(cells)
        if old is not None and old != color:
            out.append((cells, old, color))
    return out


def background_of(grid: Grid) -> int:
    counts: Counter[int] = Counter()
    for row in grid:
        counts.update(row)
    return counts.most_common(1)[0][0]


def pixel_blobs(pixels: frozenset[Cell]) -> list[list[Cell]]:
    """4-connected blobs within a pixel set."""
    todo = set(pixels)
    out: list[list[Cell]] = []
    while todo:
        seed = todo.pop()
        blob, dq = [seed], deque([seed])
        while dq:
            cx, cy = dq.popleft()
            for nxt in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                if nxt in todo:
                    todo.discard(nxt)
                    blob.append(nxt)
                    dq.append(nxt)
        out.append(blob)
    return out


def blob_anchors(pixels: frozenset[Cell]) -> frozenset[Cell]:
    """Bbox corners of the 4-connected blobs in a pixel set: two same-shape
    port blobs coincide exactly when their anchors do."""
    return frozenset(
        (min(x for x, _y in blob), min(y for _x, y in blob))
        for blob in pixel_blobs(pixels))


def bbox_min(cells) -> Cell:
    return (min(x for x, _y in cells), min(y for _x, y in cells))


def pat_norm(cells, ports) -> tuple[frozenset[Cell], frozenset[Cell]]:
    """A piece's translation-free pattern: (cells, port pixels), both
    shifted so the cell bbox starts at the origin."""
    mx, my = bbox_min(cells)
    return (frozenset((x - mx, y - my) for x, y in cells),
            frozenset((x - mx, y - my) for x, y in ports))


def pat_rot(pat: tuple) -> tuple:
    """The pattern rotated 90 degrees (direction is irrelevant: planners
    consider all four)."""
    cells, ports = pat
    rc = [(-y, x) for x, y in cells]
    mx = min(x for x, _y in rc)
    my = min(y for _x, y in rc)
    return (frozenset((x - mx, y - my) for x, y in rc),
            frozenset((-y - mx, x - my) for x, y in ports))


def pat_rots(pat: tuple) -> list[tuple]:
    out = [pat]
    r = pat
    for _ in range(3):
        r = pat_rot(r)
        if r not in out:
            out.append(r)
    return out


def canon_window(grid: Grid, x: int, y: int, p: int) -> tuple:
    """Rotation-canonical pixel tuple of a p x p window: the same card
    appearance hashes equal wherever it sits and however it is rotated
    (editor games draw each glyph instance at a random 0/90/180/270)."""
    pat = [[grid[y + dy][x + dx] for dx in range(p)] for dy in range(p)]
    best = None
    for _ in range(4):
        t = tuple(v for row in pat for v in row)
        if best is None or t < best:
            best = t
        pat = [list(r) for r in zip(*pat[::-1])]  # rotate 90 degrees
    return best


class MyAgent(Agent):
    """Learns what it controls, maps the world, plans; explores otherwise."""

    MAX_ACTIONS = 10000  # ceiling only; play_local caps at min(this, --max-steps)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # crc32 is stable across processes (hash() is salted per run);
        # CURIO_SEED lets evaluation sweep seeds deliberately
        self._rng = random.Random(
            zlib.crc32(self.game_id.encode()) ^ int(os.environ.get("CURIO_SEED", "0"))
        )
        # ── v1 novelty machinery (fallback policy) ──
        self._transitions: dict[tuple[int, str], int] = {}
        self._tried: dict[int, set[str]] = defaultdict(set)
        self._state_visits: Counter[int] = Counter()
        self._steps_since_novelty = 0
        # ── v2 world model ──
        self._prev_grid: Optional[Grid] = None
        self._prev_key: Optional[int] = None
        self._prev_action: Optional[str] = None
        self._avail: set[int] = set()  # latest frame's available actions
        self._move_votes: dict[int, Counter[Cell]] = defaultdict(Counter)
        self._avatar_sigs: Counter[frozenset] = Counter()  # {(color, dx, dy)}
        self._avatar_votes: Counter[int] = Counter()  # color fallback evidence
        self._walls: set[Cell] = set()
        self._wall_bounces: Counter[Cell] = Counter()
        self._visited_anchors: set[Cell] = set()
        self._plan: deque[GameAction] = deque()
        self._planner_cooldown = 0
        self._best_levels = 0
        # ── click-affordance library: appearance → [changed, tries] ──
        # (never cleared — not even on level-up or GAME_OVER: lp85/vc33
        # buttons look identical on every level, so what a sprite class
        # DOES transfers even though geography doesn't)
        self._click_effects: dict[int, list[int]] = {}
        self._game_overs = 0
        # ── lattice recolor model (ft09 family): appearance → effect mask ──
        # what a sprite class DOES when clicked is physics (persists across
        # levels and deaths); the board, palette ring, dead sites and solver
        # bookkeeping are per-level geography
        self._click_fx: dict[tuple, Counter[frozenset[Cell]]] = {}
        self._probed_classes: set[tuple] = set()
        self._probe_sent: Counter[tuple] = Counter()
        self._probe_gaveup: set[tuple] = set()
        self._clue_polarity = 0   # 0: marker pixel 0 reads ==, else !=
        self._palette_next: dict[int, int] = {}   # per-level color cycle
        self._site_dead: set[Cell] = set()        # lattice coords, per level
        self._lattice_plan: deque[Cell] = deque()
        self._lattice_benched: Optional[int] = None  # level where solver quit
        self._lattice_engaged = False
        self._deaths_at_engage = 0
        self._polarity_flips = 0
        self._level_deaths = 0
        self._bfs_tries = 0
        # ── attribute register (ls20 family): carried state + tile effects ──
        # what a tile class DOES to the register is physics (appearance-
        # keyed, survives levels and deaths); which cells rejected which
        # attribute is per-level geography
        self._attr_votes: Counter[Cell] = Counter()   # physics
        self._attr_events = 0                         # physics
        self._tile_fx: dict[tuple, dict[tuple, tuple]] = {}  # sig→attr→attr'
        self._refill_sigs: set[tuple] = set()         # physics
        self._stepped_sigs: set[tuple] = set()        # physics
        self._probe_steps: Counter[tuple] = Counter()  # physics
        self._act_uses: Counter[int] = Counter()      # physics
        self._pad_reject: dict[Cell, set[tuple]] = {}  # per-level geography
        self._budget_obs: Counter[tuple[int, int]] = Counter()  # per level
        self._budget_rose = False
        self._budget_max = 0  # actions per full strip, per level
        # ── port alignment (cn04 family): selected-piece manipulation ──
        # the port accent color, connection-feedback color and transform
        # verb are physics (they look the same on every level); the piece
        # roster, pairing blacklist and failure counters are per-level
        # geography
        self._port_votes: Counter[int] = Counter()      # physics
        self._fb_votes: Counter[int] = Counter()        # physics
        self._xform_votes: Counter[int] = Counter()     # physics
        self._last_move: Optional[tuple[int, Cell, frozenset]] = None
        self._pc_pieces: dict[int, dict] = {}
        self._pc_level = -1
        self._pc_sel: Optional[int] = None
        self._pc_solution: Optional[dict[int, tuple]] = None
        self._pc_failed_cfgs: set[frozenset] = set()
        self._pc_unreachable: set[tuple] = set()
        self._pc_scout: Counter[int] = Counter()
        self._pc_vscout: Counter[int] = Counter()
        self._pc_known_pats: dict[int, list] = {}  # survives deaths: the
        self._pc_known_stack: set[int] = set()     # level replays verbatim
        self._pc_strikes = 0
        self._pc_bounce = 0
        self._pc_idprobe = 0
        self._pc_xform_spent = 0
        self._pc_benched: Optional[int] = None
        # ── editor model (tr87 family): cursor + in-place glyph cycling ──
        # verb classifications, marker votes, step deltas and probe spend
        # are physics (what an action DOES looks the same on every level);
        # mutate boxes, goal candidates and failure counters are geography
        self._ed_class: dict[int, Counter[str]] = defaultdict(Counter)
        self._ed_deltas: dict[int, Counter[Cell]] = defaultdict(Counter)
        self._ed_marker: dict[int, Counter[frozenset]] = defaultdict(Counter)
        self._ed_probe: Counter[int] = Counter()    # physics
        self._ed_engaged = False                    # sticky per game
        self._ed_dead = False    # two benched levels, zero wins: stand down
        self._ed_wins = 0
        self._ed_benches = 0
        self._ed_benched: Optional[int] = None      # level where editor quit
        self._ed_strikes = 0
        self._ed_boxes: list[tuple[int, int, int, int]] = []  # per level
        self._ed_goal: Optional[list[tuple]] = None            # per level
        self._ed_goal_idx = 0
        self._ed_full_seen = False  # candidate fully matched once already
        self._ed_spent = 0          # cycle presses at the current slot
        self._ed_miss = 0           # consecutive cursor/slot read failures
        # ── HUD masking: high-churn border strips excluded from state keys ──
        # (persists across level-up and GAME_OVER: change rates are physics,
        # not geography)
        self._cell_change: Counter[Cell] = Counter()
        self._band_frames: dict[Cell, int] = defaultdict(int)  # cell → frame bitmask
        self._frames_diffed = 0
        self._hud_mask: frozenset[Cell] = frozenset()

    # ── framework contract ───────────────────────────────────────────────
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if latest_frame.state is GameState.GAME_OVER \
                    and self._prev_grid is not None:
                # once per streak: this branch nulls _prev_grid, so repeated
                # GAME_OVER frames before the RESET lands don't double-count
                self._game_overs += 1
                self._level_deaths += 1
            self._prev_grid, self._prev_key, self._prev_action = None, None, None
            self._plan.clear()
            self._lattice_plan.clear()
            # the level replays from its start layout: rebuild the piece
            # roster fresh (blacklists, scout counters and the shape
            # library stay — pids are deterministic across replays) and
            # give a benched port planner a new life too: desync strikes
            # die with the run that earned them
            self._pc_pieces.clear()
            self._pc_sel = None
            self._pc_solution = None
            self._pc_bounce = 0
            self._pc_idprobe = 0
            self._pc_strikes = 0
            self._pc_benched = None
            # the editor's level replays deterministically too: keep the
            # mutate boxes and goal candidates, give a benched run a fresh
            # life (strikes die with the budget that earned them)
            self._ed_strikes = 0
            self._ed_benched = None
            self._ed_spent = 0
            self._ed_miss = 0
            self._ed_full_seen = False
            return GameAction.RESET

        grid = grid_of(latest_frame)
        self._avail = set(latest_frame.available_actions or [])
        # may refresh the HUD mask, so learn before keying; the leveled flag
        # stops cross-level repaints from being read as click physics; the
        # animation frames tell blocked-with-animation (gate) from blocked
        # (wall), and a flood INSIDE them is a death flash
        self._learn(grid, latest_frame.levels_completed != self._best_levels,
                    latest_frame.frame)
        key = self._masked_hash(grid)
        self._note_visit(key)

        if latest_frame.levels_completed > self._best_levels:
            if self._ed_engaged and self._ed_benched != self._best_levels:
                self._ed_wins += 1  # the editor was driving this level
            self._best_levels = latest_frame.levels_completed
            # new level: keep physics (move rules, avatar signature, HUD
            # mask, click-affordance library), forget geography
            self._walls.clear()
            self._wall_bounces.clear()
            self._visited_anchors.clear()
            self._tried.clear()
            self._state_visits.clear()
            self._plan.clear()
            self._steps_since_novelty = 0
            # lattice geography: the palette and board are level data
            self._palette_next.clear()
            self._site_dead.clear()
            self._lattice_plan.clear()
            self._lattice_engaged = False
            self._polarity_flips = 0
            self._level_deaths = 0
            self._bfs_tries = 0
            # attribute geography: which pads rejected what, and the budget
            # drain rate, are level data (the register itself is physics)
            self._pad_reject.clear()
            self._budget_obs.clear()
            self._budget_rose = False
            self._budget_max = 0
            # port geography: the piece roster, pairing blacklist, scout
            # and failure counters are level data (colors/verbs persist)
            self._pc_pieces.clear()
            self._pc_sel = None
            self._pc_solution = None
            self._pc_failed_cfgs.clear()
            self._pc_unreachable.clear()
            self._pc_scout.clear()
            self._pc_vscout.clear()
            self._pc_known_pats.clear()
            self._pc_known_stack.clear()
            self._pc_strikes = 0
            self._pc_bounce = 0
            self._pc_idprobe = 0
            # editor geography: the new level has its own slots, reference
            # row and rules (the verbs and the cursor marker persist)
            self._ed_boxes.clear()
            self._ed_goal = None
            self._ed_goal_idx = 0
            self._ed_full_seen = False
            self._ed_strikes = 0
            self._ed_benched = None
            self._ed_spent = 0
            self._ed_miss = 0

        action = self._policy(grid, latest_frame)
        self._prev_grid = grid
        self._prev_key = key
        return action

    # ── learning ─────────────────────────────────────────────────────────
    def _learn(
        self, grid: Optional[Grid], leveled: bool = False,
        frames: Optional[list[Grid]] = None,
    ) -> None:
        self._last_move = None  # this frame's clean translation, if any
        if self._prev_grid is None or grid is None or self._prev_action is None:
            return
        if flood_color(grid) != flood_color(self._prev_grid):
            # a color flooding in (death flash) or receding (respawn, level
            # redraw) makes this diff garbage — drop the plan, learn nothing.
            # (an event test, not a state test: cn04's NORMAL frames are >80%
            # background, so a bare flood check would disable learning there)
            self._plan.clear()
            return
        self._update_hud_mask(grid)  # before the isdigit() gate: clicks feed it too
        if self._prev_key is not None:
            self._transitions[(self._prev_key, self._prev_action)] = \
                self._masked_hash(grid)
        if not self._prev_action.isdigit():
            if self._prev_action.startswith("6:"):
                # AFFORD: log whether clicking this sprite class did anything
                # (appearance-keyed, so the fact transfers across levels)
                x, y = map(int, self._prev_action[2:].split(","))
                comps_prev = components(self._prev_grid)
                sig = signature_under(comps_prev, (x, y))
                eff = self._click_effects.setdefault(sig, [0, 0])
                if not self._same_unmasked(self._prev_grid, grid):
                    eff[0] += 1
                eff[1] += 1
                self._learn_click_fx(grid, (x, y), comps_prev, leveled)
            return  # click effects aren't positional; only model simple actions
        act = int(self._prev_action)
        self._act_uses[act] += 1
        anim = len(frames) if frames else 1
        if frames and any(flood_color(f) != flood_color(grid)
                          for f in frames[:-1]):
            # a flood INSIDE the action's animation (ls20's full-screen
            # death flash, hidden from last-frame diffs by the respawn
            # repaint): the diff straddles a teleport/reset.  Learning from
            # it teaches poison — a "bounce" at the death cell becomes a
            # phantom gate, a glyph reset becomes floor-that-shifts-
            # attributes.  Drop the plan, learn nothing.
            self._plan.clear()
            return
        reg = self._attr_register()
        self._budget_rose = False
        if reg is not None:
            self._observe_budget(self._prev_grid, grid)
        groups: dict[Cell, list[tuple[int, frozenset[Cell]]]] = defaultdict(list)
        for color, delta, cells in moved_objects(self._prev_grid, grid):
            groups[delta].append((color, cells))
        if not leveled:
            # EDIT: classify this action's diff for the editor model
            # (cross-level repaints are not action physics)
            self._ed_note(act, grid, groups)
        if len(groups) == 1:
            # all movers share one delta → one rigid group: avatar candidate.
            # Multi-color sprites (ls20's 12-over-9 player, cn04's selected
            # piece) move as several components; a single-color avatar is
            # just a group of one.  >1 deltas means ambient animation;
            # crediting those poisons the model.
            (delta, movers), = groups.items()
            union = {cell for _color, cells in movers for cell in cells}
            xs = sorted(x for x, _y in union)
            ys = sorted(y for _x, y in union)
            # the full union is a clean rigid translation whatever its size
            # (cn04's 15x21 pieces overflow GROUP_BBOX): record it for the
            # port-model sync; the bbox cap below only protects avatar/rule
            # learning from large synchronized animations
            self._last_move = (act, delta, frozenset(union))
            if xs[-1] - xs[0] < GROUP_BBOX and ys[-1] - ys[0] < GROUP_BBOX:
                ax, ay = min(union)
                self._move_votes[act][delta] += 1
                self._avatar_sigs[frozenset(
                    (color, x - ax, y - ay)
                    for color, cells in movers for x, y in cells)] += 1
                for color, _cells in movers:
                    self._avatar_votes[color] += 1
                extra = self._changed_beyond(union, delta, grid)
                clean = delta == self._movement_rules().get(act)
                if clean:
                    # the avatar ENTERED this tile class: probes stop
                    # targeting it (what it does is now on record)
                    self._stepped_sigs.add(
                        self._tile_sig(self._prev_grid, union))
                if extra:
                    self._soften_walls()
                    if clean:
                        # only clean rule-steps vote: a death-respawn
                        # teleport is also a "single group moved" diff, and
                        # its lives-pip change would build a phantom register
                        self._note_attr_event(extra)
                        reg = self._attr_register()  # may have just engaged
                if reg is not None:
                    if clean:
                        # clean rule-step onto a tile: log what the tile DID
                        # to the register (identity entries matter too — the
                        # planner walks known-inert floor without probing it)
                        sig = self._tile_sig(self._prev_grid, union)
                        before = self._attr_state(self._prev_grid, reg)
                        self._tile_fx.setdefault(sig, {})[before] = \
                            self._attr_state(grid, reg)
                        if self._budget_rose:
                            self._refill_sigs.add(sig)
                    elif extra:
                        # non-rule jump that also changed the world: a death
                        # respawn teleport — the position plan is garbage
                        self._plan.clear()
                return
        if act in self._movement_rules():
            prev_av = self._find_avatar(self._prev_grid)
            cur_av = self._find_avatar(grid)
            if not prev_av or not cur_av:
                return
            dx, dy = self._movement_rules()[act]
            if cur_av[0] == (prev_av[0][0] + dx, prev_av[0][1] + dy) \
                    and len(cur_av[1]) == len(prev_av[1]):
                # clean rule move that the single-group gate REJECTED: the
                # register repaint aliased as a second mover (ls20's rotated
                # glyph chunk "translates" within the HUD box) — anchor the
                # ATTR learning on the avatar template instead
                union = set(cur_av[1])
                if self._last_move is None:
                    self._last_move = (act, (dx, dy), frozenset(union))
                self._stepped_sigs.add(
                    self._tile_sig(self._prev_grid, union))
                extra = self._changed_beyond(union, (dx, dy), grid)
                if extra:
                    self._soften_walls()
                    self._note_attr_event(extra)
                    reg = self._attr_register()
                if reg is not None:
                    sig = self._tile_sig(self._prev_grid, union)
                    before = self._attr_state(self._prev_grid, reg)
                    self._tile_fx.setdefault(sig, {})[before] = \
                        self._attr_state(grid, reg)
                    if self._budget_rose:
                        self._refill_sigs.add(sig)
                return
            if prev_av[0] == cur_av[0]:
                # trusted movement action, avatar still at the same anchor →
                # wall ahead.  Anchor test, not frame equality: ls20's
                # blocked moves still flash failure pixels and tick the
                # masked pip strip.
                if reg is not None:
                    self._plan.clear()  # the route assumed this step landed
                    dest = (prev_av[0][0] + dx, prev_av[0][1] + dy)
                    if anim > 1:
                        # blocked WITH an animation (ls20's pad failure
                        # flash): a conditional gate, not a wall — remember
                        # the attribute it rejected, retry under another
                        self._pad_reject.setdefault(dest, set()).add(
                            self._attr_state(grid, reg))
                        return
                    if dest in self._pad_reject:
                        # a known gate bounced without its animation: count
                        # the attempt so a misread gate can't loop forever
                        self._pad_reject[dest].add(
                            self._attr_state(grid, reg))
                for x, y in prev_av[1]:
                    cell = (x + dx, y + dy)
                    self._walls.add(cell)
                    self._wall_bounces[cell] += 1

    def _changed_beyond(
        self, moved: set[Cell], delta: Cell, grid: Grid
    ) -> list[Cell]:
        """Unmasked cells outside the moved group's footprint that changed."""
        dx, dy = delta
        footprint = moved | {(x - dx, y - dy) for x, y in moved}
        out: list[Cell] = []
        for y, (prow, row) in enumerate(zip(self._prev_grid, grid)):
            if prow == row:
                continue
            for x, (pv, v) in enumerate(zip(prow, row)):
                if pv != v and (x, y) not in footprint \
                        and (x, y) not in self._hud_mask:
                    out.append((x, y))
        return out

    def _soften_walls(self) -> None:
        """The world changed beyond the avatar (ls20: a shifter tile cycled
        the carried glyph) — conditional walls may no longer hold.  Keep only
        walls confirmed WALL_TRUST+ times; the rest get re-tested, so a goal
        pad that bounced us under the wrong glyph is never blacklisted."""
        self._walls = {c for c in self._walls
                       if self._wall_bounces[c] >= WALL_TRUST}

    # ── attribute register (ls20 family): carried state + tile effects ──
    def _note_attr_event(self, extra: list[Cell]) -> None:
        """A clean avatar move ALSO changed a compact patch elsewhere — the
        signature of an attribute register (ls20's carried-glyph HUD box
        updating when a shifter tile is entered).  Border-band cells are
        stripped first: the step-pip strip lives there and changes (in a
        deceptively compact 1x2 patch) on EVERY move until the HUD mask
        warms up — the register's unmaskable interior is what counts.
        Pad unlocks repaint compact patches too (the 3x3 key glyph), but
        only once per pad, which the per-cell vote threshold absorbs."""
        if GameAction.ACTION6.value in self._avail \
                and self._port_color() is not None:
            # selected-piece games (cn04): a move that ALSO recolors cells
            # elsewhere is pairing feedback or an uncovered piece, not a
            # carried-attribute register — two same-cell pairings across a
            # type-retry would otherwise mint a phantom register and gate
            # the attr planner in.  ls20-family games expose no clicks, so
            # their voting is untouched.
            return
        core = [(x, y) for x, y in extra
                if HUD_BAND <= x < GRID - HUD_BAND
                and HUD_BAND <= y < GRID - HUD_BAND]
        if not core:
            return
        xs = [x for x, _y in core]
        ys = [y for _x, y in core]
        if max(xs) - min(xs) >= ATTR_BBOX or max(ys) - min(ys) >= ATTR_BBOX:
            return
        for cell in core:
            self._attr_votes[cell] += 1
        self._attr_events += 1

    def _attr_register(self) -> Optional[frozenset[Cell]]:
        """Cells confirmed (ATTR_VOTES+ events) to change while the avatar
        moved elsewhere: the attribute register.  None until the evidence
        is in — every ATTR policy branch keys off that gate, so games
        without a register never see this machinery."""
        if self._attr_events < ATTR_EVENTS:
            return None
        cells = [c for c, n in self._attr_votes.items() if n >= ATTR_VOTES]
        if not cells:
            return None
        xs = [x for x, _y in cells]
        ys = [y for _x, y in cells]
        if max(xs) - min(xs) >= ATTR_BBOX or max(ys) - min(ys) >= ATTR_BBOX:
            return None  # two clashing change sources: trust neither
        return frozenset(cells)

    def _attr_state(self, grid: Grid, reg: frozenset[Cell]) -> tuple:
        """The register's pixels, read off the frame: the carried attribute
        exactly as the game displays it."""
        return tuple(grid[y][x] for x, y in sorted(reg))

    @staticmethod
    def _tile_sig(grid: Grid, cells: frozenset[Cell] | set[Cell]) -> tuple:
        """Appearance of the patch under a sprite footprint, raster-ordered:
        the position-free key for per-tile-class register effects."""
        return tuple(grid[y][x]
                     for x, y in sorted(cells, key=lambda c: (c[1], c[0])))

    def _observe_budget(self, prev: Grid, cur: Grid) -> None:
        """Track the step-budget strip: within the masked HUD, one color
        draining by a constant amount on (nearly) every action is a budget;
        a jump back UP is a refill ring or a death reset."""
        if not self._hud_mask:
            return
        pc: Counter[int] = Counter()
        cc: Counter[int] = Counter()
        for x, y in self._hud_mask:
            pc[prev[y][x]] += 1
            cc[cur[y][x]] += 1
        drops = [(c, pc[c] - cc[c]) for c in pc if pc[c] > cc[c]]
        if len(drops) == 1:  # exactly one color drained: budget candidate
            self._budget_obs[drops[0]] += 1
        budget = self._budget_state()
        if budget is not None:
            color, k = budget
            self._budget_max = max(self._budget_max, cc[color] // k)
            if cc[color] - pc[color] >= 3 * k:
                self._budget_rose = True

    def _budget_state(self) -> Optional[tuple[int, int]]:
        """(color, cells drained per action) once one pair dominates."""
        if not self._budget_obs:
            return None
        (ck, n), = self._budget_obs.most_common(1)
        return ck if n >= BUDGET_OBS else None

    def _budget_remaining(self, grid: Grid) -> Optional[int]:
        """Actions left before the strip empties, by the learned rate."""
        budget = self._budget_state()
        if budget is None:
            return None
        color, k = budget
        lit = sum(1 for x, y in self._hud_mask if grid[y][x] == color)
        return lit // k

    def _plan_attr_route(
        self, grid: Grid, avatar: tuple[Cell, frozenset[Cell]],
        rules: dict[int, Cell], reg: frozenset[Cell],
    ) -> Optional[list[GameAction]]:
        """BFS over the (anchor, attribute) product graph.  Entering a tile
        whose learned effect maps the current attribute applies it, so
        'touch the shifter once more, then enter the pad' falls out of
        plain shortest-path search.  The step budget rides along in the
        search state: every move costs one, entering a refill tile resets
        it, and branches that would die are pruned — so a plan that only
        fits by routing THROUGH a refill ring is found, and budget-suicide
        plans are not produced at all.  Goals, best rank wins:
          1 — a gate (a pad that bounced with a failure animation) under an
              attribute it has not rejected yet,
          2 — a tile with unknown effect: a class that changed the register
              before (but not from THIS attribute), or a never-entered
              patterned tile — probing them grows the product graph,
          3 — an unvisited anchor (the old exploration goal, attr-aware).
        Returns None when the product graph offers nothing, so the caller
        falls through to the plain positional planner."""
        anchor, cells = avatar
        shape = sorted(((x - anchor[0], y - anchor[1]) for x, y in cells),
                       key=lambda d: (d[1], d[0]))  # raster: matches _tile_sig
        attr0 = self._attr_state(grid, reg)
        self._visited_anchors.add(anchor)
        counts: Counter[int] = Counter()
        for row in grid:
            counts.update(row)
        background = counts.most_common(1)[0][0]
        shifty = {sig for sig, fx in self._tile_fx.items()
                  if any(a != b for a, b in fx.items())}
        sig_cache: dict[Cell, Optional[tuple]] = {}

        def sig_at(pos: Cell) -> Optional[tuple]:
            if pos not in sig_cache:
                px, py = pos
                inside = all(0 <= px + dx < GRID and 0 <= py + dy < GRID
                             for dx, dy in shape)
                sig_cache[pos] = tuple(
                    grid[py + dy][px + dx] for dx, dy in shape
                ) if inside else None
            return sig_cache[pos]

        def open_floor(pos: Cell) -> bool:
            px, py = pos
            return all(
                0 <= px + dx < GRID and 0 <= py + dy < GRID
                and (px + dx, py + dy) not in self._walls
                for dx, dy in shape)

        INF = 1 << 30
        remaining = self._budget_remaining(grid)
        start_left = INF if remaining is None else remaining
        full_left = INF if remaining is None \
            else max(remaining, self._budget_max)
        goals: dict[int, list[GameAction]] = {}
        start = (anchor, attr0)
        best_left: dict[tuple[Cell, tuple], int] = {start: start_left}
        dq: deque[tuple[tuple[Cell, tuple], list[GameAction], int]] = \
            deque([(start, [], start_left)])
        popped = 0
        while dq and popped < ATTR_BFS_CAP and 1 not in goals:
            (pos, attr), path, left = dq.popleft()
            popped += 1
            for act, (dx, dy) in rules.items():
                nxt = (pos[0] + dx, pos[1] + dy)
                nleft = left - 1
                if nleft < 0:
                    continue  # this branch dies before arriving
                npath = path + [GameAction.from_id(act)]
                rej = self._pad_reject.get(nxt)
                if rej is not None:
                    # conditional gate: only enterable as a plan's FINAL
                    # step, and only under an attribute it hasn't rejected
                    if attr not in rej and len(rej) < GATE_GIVEUP:
                        goals.setdefault(1, npath)
                    continue
                if not open_floor(nxt):
                    continue
                # the start anchor reads as the avatar's own sprite in the
                # CURRENT grid — an eternally "unknown patterned tile" that
                # must never become a probe goal (out-and-back forever)
                sig = sig_at(nxt) if nxt != anchor else None
                fx = self._tile_fx.get(sig, {}) if sig is not None else {}
                if sig is not None and attr not in fx \
                        and (sig in shifty
                             or (sig not in self._tile_fx
                                 and len(set(sig)) > 1
                                 and any(v != background for v in sig))):
                    # unknown effect on a patterned tile: probing it is the
                    # goal (don't search past it — the outcome is unmodeled)
                    goals.setdefault(2, npath)
                    continue
                if sig is not None and sig in self._refill_sigs:
                    nleft = full_left
                state = (nxt, fx.get(attr, attr))
                if nleft <= best_left.get(state, -1):
                    continue  # been here with at least this much budget
                best_left[state] = nleft
                if nxt not in self._visited_anchors:
                    goals.setdefault(3, npath)
                dq.append((state, npath, nleft))
        return goals.get(1) or goals.get(2) or goals.get(3)

    def _probe_tiles(
        self, grid: Grid, avatar: tuple[Cell, frozenset[Cell]],
        rules: dict[int, Cell],
    ) -> Optional[list[GameAction]]:
        """Walk ONTO the nearest patterned tile class the avatar has never
        entered.  What standing on a tile DOES is the movement analog of
        the click-affordance library — and it's the bootstrap for the
        attribute register, which only reveals itself when a shifter tile
        is actually stepped on.  Runs strictly as a fallback (positional
        planner found nothing), so games it doesn't apply to keep their
        novelty-search behavior; PROBE_TRIES caps routes to classes that
        never let the avatar in (locked pads)."""
        anchor, cells = avatar
        shape = sorted(((x - anchor[0], y - anchor[1]) for x, y in cells),
                       key=lambda d: (d[1], d[0]))
        counts: Counter[int] = Counter()
        for row in grid:
            counts.update(row)
        background = counts.most_common(1)[0][0]

        def sig_at(pos: Cell) -> Optional[tuple]:
            px, py = pos
            if not all(0 <= px + dx < GRID and 0 <= py + dy < GRID
                       for dx, dy in shape):
                return None
            return tuple(grid[py + dy][px + dx] for dx, dy in shape)

        def open_floor(pos: Cell) -> bool:
            px, py = pos
            return all(
                0 <= px + dx < GRID and 0 <= py + dy < GRID
                and (px + dx, py + dy) not in self._walls
                for dx, dy in shape)

        seen = {anchor}
        dq: deque[tuple[Cell, list[GameAction]]] = deque([(anchor, [])])
        while dq and len(seen) < 3000:
            pos, path = dq.popleft()
            for act, (dx, dy) in rules.items():
                nxt = (pos[0] + dx, pos[1] + dy)
                if nxt in seen or not open_floor(nxt):
                    continue
                seen.add(nxt)
                npath = path + [GameAction.from_id(act)]
                sig = sig_at(nxt)
                if sig is not None and sig not in self._stepped_sigs \
                        and self._probe_steps[sig] < PROBE_TRIES \
                        and len(set(sig)) > 1 \
                        and any(v != background for v in sig):
                    self._probe_steps[sig] += 1
                    return npath
                dq.append((nxt, npath))
        return None

    # ── port alignment (cn04 family): selected-piece manipulation ───────
    def _port_color(self) -> Optional[int]:
        """The accent color marking connection ports, once trusted."""
        if not self._port_votes:
            return None
        top = self._port_votes.most_common(2)
        color, n = top[0]
        if n < PORT_VOTES or (len(top) > 1 and top[1][1] * 2 > n):
            return None
        return color

    def _fb_color(self) -> Optional[int]:
        """The color two coinciding ports render as (cn04's green)."""
        if not self._fb_votes:
            return None
        color, n = self._fb_votes.most_common(1)[0]
        return color if n >= 2 else None

    def _pc_regions(self, grid: Grid) -> list[frozenset[Cell]]:
        """8-connected non-background regions: multi-color pieces read as
        one region each.  Regions confined to the border band are HUD
        chrome (cn04's step bar), not pieces."""
        background = background_of(grid)
        seen = [[False] * GRID for _ in range(GRID)]
        out: list[frozenset[Cell]] = []
        for y in range(GRID):
            for x in range(GRID):
                if seen[y][x] or grid[y][x] == background:
                    continue
                cells, dq = [], deque([(x, y)])
                seen[y][x] = True
                while dq:
                    cx, cy = dq.popleft()
                    cells.append((cx, cy))
                    for nx in (cx - 1, cx, cx + 1):
                        for ny in (cy - 1, cy, cy + 1):
                            if 0 <= nx < GRID and 0 <= ny < GRID \
                                    and not seen[ny][nx] \
                                    and grid[ny][nx] != background:
                                seen[ny][nx] = True
                                dq.append((nx, ny))
                if len(cells) < 4:
                    continue
                xs = [cx for cx, _cy in cells]
                ys = [cy for _cx, cy in cells]
                if max(ys) < HUD_BAND or min(ys) >= GRID - HUD_BAND \
                        or max(xs) < HUD_BAND or min(xs) >= GRID - HUD_BAND:
                    continue  # band-confined: HUD chrome, not a piece
                out.append(frozenset(cells))
        return out

    def _vote_port_color(self, grid: Grid, scale: int) -> None:
        """Port candidates: a color whose every blob fits in one movement
        step, carried as a minority by 2+ separate pieces.  Body colors and
        the selection recolor fail the blob-size or minority test; the
        feedback color can't qualify because voting only runs before any
        pairing exists (fresh levels start unpaired)."""
        regs = self._pc_regions(grid)
        if len(regs) < 2:
            return
        carriers: Counter[int] = Counter()
        oversize: set[int] = set()
        for cells in regs:
            per: dict[int, set[Cell]] = defaultdict(set)
            for x, y in cells:
                per[grid[y][x]].add((x, y))
            for color, px in per.items():
                blobs = pixel_blobs(frozenset(px))
                if any(max(x for x, _y in b) - min(x for x, _y in b) >= scale
                       or max(y for _x, y in b) - min(y for _x, y in b) >= scale
                       for b in blobs):
                    oversize.add(color)
                elif len(px) * 2 <= len(cells):
                    carriers[color] += 1
        for color, n in carriers.items():
            if n >= 2 and color not in oversize:
                self._port_votes[color] += 1

    def _pc_build(self, grid: Grid, level: int) -> None:
        """(Re)build the piece roster from a frame.  At level start and
        after a death the pieces are disjoint, so regions ARE pieces; ports
        are the accent pixels inside each region (grey-masked levels reveal
        them piece by piece — sync picks those up as selections happen)."""
        port, fb = self._port_color(), self._fb_color()
        self._pc_pieces = {}
        self._pc_sel = None
        self._pc_solution = None
        self._pc_bounce = 0
        self._pc_idprobe = 0
        self._pc_level = level
        regs = sorted(self._pc_regions(grid),
                      key=lambda r: (min(y for _x, y in r),
                                     min(x for x, _y in r)))
        for pid, cells in enumerate(regs):
            ports = {(x, y) for x, y in cells
                     if grid[y][x] == port or grid[y][x] == fb}
            pats = [pat_norm(cells, ports)] if ports else []
            for pat in self._pc_known_pats.get(pid, []):
                if pat not in pats:
                    pats.append(pat)  # shapes probed in an earlier life
            self._pc_pieces[pid] = {
                "cells": set(cells), "ports": ports,
                "pats": pats,
                "stack": pid in self._pc_known_stack,
            }

    def _pc_shift(self, pid: int, delta: Cell) -> None:
        dx, dy = delta
        p = self._pc_pieces[pid]
        cells = {(x + dx, y + dy) for x, y in p["cells"]}
        if not all(0 <= x < GRID and 0 <= y < GRID for x, y in cells):
            return  # the engine clamps at borders: a real move can't exit
        p["cells"] = cells
        p["ports"] = {(x + dx, y + dy) for x, y in p["ports"]}

    def _pc_port_score(self, grid: Grid, ports: set[Cell],
                       dx: int, dy: int) -> float:
        """Fraction of a piece's port pixels that look port-ish (port or
        feedback colored) at a hypothesized offset."""
        port, fb = self._port_color(), self._fb_color()
        good = total = 0
        for x, y in ports:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID and 0 <= ny < GRID:
                total += 1
                if grid[ny][nx] == port or grid[ny][nx] == fb:
                    good += 1
        return good / total if total else 0.0

    def _pc_note_pattern(self, pid: int, p: dict) -> None:
        pat = pat_norm(p["cells"], p["ports"])
        if pat in p["pats"]:
            return
        if p["pats"] and not any(pat in pat_rots(q) for q in p["pats"]):
            # the new shape is NOT a rotation of a known one: a variant
            # stack — synthesized rotations would be unreachable lies
            p["stack"] = True
            self._pc_known_stack.add(pid)
        p["pats"].append(pat)
        mem = self._pc_known_pats.setdefault(pid, [])
        if pat not in mem:
            mem.append(pat)

    def _pc_verify_at(self, grid: Grid, pat: tuple, pos: Cell,
                      background: int, others: set[Cell]) -> bool:
        """Would this placement render exactly like the frame shows?  The
        selected piece sits on the top layer, so its every pixel is
        visible: all port pixels must show the port (or feedback) color and
        all body cells must be non-background — works even while the piece
        overlaps another one.  Exactness matters too: nothing may hang off
        the footprint except other pieces (cn04's snake variants GROW, and
        a grown snake still contains its shorter ancestor)."""
        cells, ports = pat
        port, fb = self._port_color(), self._fb_color()
        px, py = pos
        for x, y in cells:
            nx, ny = px + x, py + y
            if not (0 <= nx < GRID and 0 <= ny < GRID):
                return False
            v = grid[ny][nx]
            if v == background:
                return False
            if (x, y) in ports and v != port and v != fb:
                return False
        for x, y in cells:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (x + dx, y + dy) in cells:
                    continue
                nx, ny = px + x + dx, py + y + dy
                if 0 <= nx < GRID and 0 <= ny < GRID \
                        and grid[ny][nx] != background \
                        and (nx, ny) not in others:
                    return False
        return True

    def _pc_reextract(self, grid: Grid, pid: int, scale: int) -> bool:
        """Re-read the selected piece after an in-place transform.  Known
        orientations are verified PREDICTIVELY against the frame (reliable
        even while overlapping other pieces — the selected piece renders on
        top); genuinely new shapes (variant stacks) fall back to a flood
        re-read, trustworthy only in the clear.  True when the footprint
        changed."""
        pieces = self._pc_pieces
        p = pieces[pid]
        background = background_of(grid)
        others: set[Cell] = set()
        for q, qp in pieces.items():
            if q != pid:
                others |= qp["cells"]
        old_pat = pat_norm(p["cells"], p["ports"])
        opos = bbox_min(p["cells"])
        xs = [x for x, _y in p["cells"]]
        ys = [y for _x, y in p["cells"]]
        span = max(max(xs) - min(xs), max(ys) - min(ys)) + scale
        span -= span % scale
        cands: list[tuple] = []
        for pat in p["pats"]:
            for r in pat_rots(pat):
                if r not in cands:
                    cands.append(r)
        placements = sorted(
            ((abs(dx) + abs(dy) + (pat != old_pat), pat,
              (opos[0] + dx, opos[1] + dy))
             for pat in cands if pat[1]
             for dx in range(-span, span + 1, scale)
             for dy in range(-span, span + 1, scale)),
            key=lambda c: c[0])  # the unchanged placement verifies first
        for _d, pat, pos in placements:
            if self._pc_verify_at(grid, pat, pos, background, others):
                changed = pat != old_pat or pos != opos
                p["cells"] = {(pos[0] + x, pos[1] + y) for x, y in pat[0]}
                p["ports"] = {(pos[0] + x, pos[1] + y) for x, y in pat[1]}
                if changed:
                    self._pc_note_pattern(pid, p)
                return changed
        # unknown shape: flood re-read around the old footprint, excluding
        # other pieces' cells (truncation risk — don't record patterns when
        # the result touches another piece)
        x0, x1 = min(xs) - span, max(xs) + span
        y0, y1 = min(ys) - span, max(ys) + span
        seeds = [(x, y) for x, y in p["cells"]
                 if grid[y][x] != background and (x, y) not in others]
        if not seeds:
            return False
        cells = set(seeds)
        dq = deque(seeds)
        while dq:
            cx, cy = dq.popleft()
            for nx in (cx - 1, cx, cx + 1):
                for ny in (cy - 1, cy, cy + 1):
                    if x0 <= nx <= x1 and y0 <= ny <= y1 \
                            and 0 <= nx < GRID and 0 <= ny < GRID \
                            and (nx, ny) not in cells \
                            and (nx, ny) not in others \
                            and grid[ny][nx] != background:
                        cells.add((nx, ny))
                        dq.append((nx, ny))
        port, fb = self._port_color(), self._fb_color()
        ports = {(x, y) for x, y in cells
                 if grid[y][x] == port or grid[y][x] == fb}
        changed = cells != p["cells"]
        clean = not any((x + dx, y + dy) in others
                        for x, y in cells
                        for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        p["cells"], p["ports"] = cells, ports
        if changed and clean:
            self._pc_note_pattern(pid, p)
        return changed

    def _pc_note_feedback(self, grid: Grid) -> None:
        """Two port blobs on one cell render a fresh color: learn it from
        the model's anchor occupancy."""
        if self._fb_color() is not None:
            return
        port = self._port_color()
        background = background_of(grid)
        occ: Counter[Cell] = Counter()
        for p in self._pc_pieces.values():
            occ.update(blob_anchors(frozenset(p["ports"])))
        for (x, y), n in occ.items():
            if n == 2 and grid[y][x] != port and grid[y][x] != background:
                self._fb_votes[grid[y][x]] += 1

    def _pc_sync(self, grid: Grid, level: int,
                 rules: dict[int, Cell], scale: int) -> None:
        """Reconcile the piece roster with what the last action did."""
        if not self._pc_pieces or self._pc_level != level:
            self._pc_build(grid, level)
            return
        pa = self._prev_action
        pieces = self._pc_pieces
        if pa is not None and pa.isdigit():
            act = int(pa)
            if act in rules:
                mv = self._last_move
                handled = False
                if mv is not None and mv[0] == act:
                    _act, delta, union = mv
                    pid = max(pieces,
                              key=lambda q: len(pieces[q]["cells"] & union))
                    # credible only when the moved group accounts for the
                    # whole piece: a lone aliased port blob gliding inside
                    # a big STATIC piece fails the size test, and sparse
                    # glyphs (tiny self-overlap after one step) pass it
                    if pieces[pid]["cells"] & union \
                            and len(union) * 2 >= len(pieces[pid]["cells"]):
                        self._pc_shift(pid, delta)
                        self._pc_sel = pid
                        self._pc_bounce = 0
                        self._pc_idprobe = 0
                        handled = True
                if not handled:
                    if self._prev_grid is not None \
                            and self._same_unmasked(self._prev_grid, grid):
                        self._pc_bounce += 1  # clamped, or nothing selected
                    elif self._pc_sel in pieces:
                        # not a clean translation (pairing recolors, or an
                        # aliased twin broke the diff): score the selected
                        # piece's ports at both hypotheses, ties read moved
                        self._pc_bounce = 0
                        p = pieces[self._pc_sel]
                        dx, dy = rules[act]
                        if self._pc_port_score(grid, p["ports"], dx, dy) \
                                >= self._pc_port_score(grid, p["ports"],
                                                       0, 0):
                            self._pc_shift(self._pc_sel, (dx, dy))
                        else:
                            # the world changed but our piece stayed: the
                            # selection belief is a lie — relearn it
                            self._pc_sel = None
                            self._pc_strikes += 1
            elif self._pc_sel in pieces and self._prev_grid is not None \
                    and not self._same_unmasked(self._prev_grid, grid):
                # a non-movement simple action changed the world: in-place
                # transform of the selected piece (rotation/variant cycle)
                if self._pc_reextract(grid, self._pc_sel, scale):
                    self._xform_votes[act] += 1
        elif pa is not None and pa.startswith("6:"):
            x, y = map(int, pa[2:].split(","))
            pid = next((q for q in sorted(pieces)
                        if (x, y) in pieces[q]["cells"]), None)
            if pid is not None and pid != self._pc_sel:
                self._pc_sel = pid
                self._pc_bounce = 0
                self._pc_idprobe = 0
        self._pc_note_feedback(grid)
        # the selected piece shows its true pixels: refresh its port map
        # (this is what reads hidden ports under grey masking)
        sel = self._pc_sel
        port, fb = self._port_color(), self._fb_color()
        if sel in pieces:
            p = pieces[sel]
            ports = {(x, y) for x, y in p["cells"]
                     if grid[y][x] == port or grid[y][x] == fb}
            if len(ports) >= len(p["ports"]) and ports != p["ports"]:
                p["ports"] = ports
                self._pc_note_pattern(sel, p)

    def _pc_patterns(self, pid: int) -> list[tuple]:
        """Orientations the planner may ask of a piece: observed patterns,
        plus synthesized rotations unless the piece is a variant stack."""
        p = self._pc_pieces[pid]
        pats = list(p["pats"])
        if not p["stack"]:
            for pat in list(pats):
                for r in pat_rots(pat):
                    if r not in pats:
                        pats.append(r)
        live = [pat for pat in pats
                if (pid, pat) not in self._pc_unreachable]
        return live or list(p["pats"])

    def _pc_cfg_key(self, sol: dict) -> frozenset:
        """Position-independent fingerprint of a configuration, stable
        across deaths and re-solves: relative placements only."""
        base = min(sol)
        bx, by = sol[base][1]
        return frozenset((pid, pat, pos[0] - bx, pos[1] - by)
                         for pid, (pat, pos) in sol.items())

    def _pc_solve(self, scale: int) -> list[dict[int, tuple]]:
        """DFS over pairings: take the first unpaired port anchor, try
        every (unplaced piece, orientation, port) that lands a port there
        without ever stacking three; a complete assignment with every
        anchor paired is a candidate final configuration.  Port TYPES are
        invisible (the two marker kinds render alike), so geometric
        solutions come out cheapest-first and are falsified by playing
        them out — the level not advancing blacklists the configuration."""
        pieces = self._pc_pieces
        pids = [pid for pid in sorted(pieces) if pieces[pid]["ports"]]
        if len(pids) < 2:
            return []
        cur = {pid: (pat_norm(pieces[pid]["cells"], pieces[pid]["ports"]),
                     bbox_min(pieces[pid]["cells"])) for pid in pids}
        pats = {pid: self._pc_patterns(pid) for pid in pids}
        anchors: dict[tuple, tuple[Cell, ...]] = {}
        for pid in pids:
            for pat in pats[pid]:
                if pat not in anchors:
                    anchors[pat] = tuple(sorted(blob_anchors(pat[1])))
        order = sorted(pids, key=lambda q: (-len(blob_anchors(
            frozenset(pieces[q]["ports"]))), -len(pieces[q]["cells"]), q))
        sols: list[dict[int, tuple]] = []
        nodes = [0]

        def fits(occ: Counter, pat: tuple, pos: Cell) -> bool:
            return all(occ[(a[0] + pos[0], a[1] + pos[1])] < 2
                       for a in anchors[pat])

        def place(occ: Counter, pat: tuple, pos: Cell, k: int) -> None:
            for a in anchors[pat]:
                occ[(a[0] + pos[0], a[1] + pos[1])] += k

        def dfs(assign: dict, occ: Counter) -> None:
            nodes[0] += 1
            if nodes[0] > PC_NODES or len(sols) >= PC_SOLS:
                return
            open_cells = sorted(c for c, n in occ.items() if n == 1)
            if not open_cells:
                rest = [q for q in order if q not in assign]
                if not rest:
                    sols.append(dict(assign))
                    return
                q = rest[0]  # disjoint cluster: seed it where it stands
                pos = cur[q][1]
                for pat in pats[q]:
                    if fits(occ, pat, pos):
                        assign[q] = (pat, pos)
                        place(occ, pat, pos, 1)
                        dfs(assign, occ)
                        place(occ, pat, pos, -1)
                        del assign[q]
                return
            c = open_cells[0]
            for q in order:
                if q in assign:
                    continue
                qx, qy = cur[q][1]
                for pat in pats[q]:
                    for a in anchors[pat]:
                        pos = (c[0] - a[0], c[1] - a[1])
                        if (pos[0] - qx) % scale or (pos[1] - qy) % scale:
                            continue  # unreachable by movement steps
                        if not fits(occ, pat, pos):
                            continue
                        assign[q] = (pat, pos)
                        place(occ, pat, pos, 1)
                        dfs(assign, occ)
                        place(occ, pat, pos, -1)
                        del assign[q]

        seed = order[0]
        for pat in pats[seed]:
            occ: Counter[Cell] = Counter()
            pos = cur[seed][1]
            place(occ, pat, pos, 1)
            dfs({seed: (pat, pos)}, occ)
            if len(sols) >= PC_SOLS or nodes[0] > PC_NODES:
                break

        def cost(sol: dict) -> int:
            c = 0
            for pid, (pat, pos) in sol.items():
                cpat, cpos = cur[pid]
                steps = (abs(pos[0] - cpos[0])
                         + abs(pos[1] - cpos[1])) // scale
                if pat != cpat:
                    c += 4  # selection click + transform presses, roughly
                if steps:
                    c += 3 + steps
            return c

        live: list[dict[int, tuple]] = []
        seen_keys: set[frozenset] = set()
        for sol in sols:
            key = self._pc_cfg_key(sol)
            if key in self._pc_failed_cfgs or key in seen_keys:
                continue
            seen_keys.add(key)
            live.append(sol)
        live.sort(key=cost)
        return live

    def _pc_click_cell(self, pid: int) -> Optional[Cell]:
        """A pixel that selects this piece and nothing else: on the piece,
        not on any other piece, away from port pixels."""
        pieces = self._pc_pieces
        p = pieces[pid]
        others: set[Cell] = set()
        for q, qp in pieces.items():
            if q != pid:
                others |= qp["cells"]
        good = [c for c in p["cells"]
                if 0 <= c[0] < GRID and 0 <= c[1] < GRID]
        cands = [c for c in good
                 if c not in others and c not in p["ports"]]
        if not cands:
            cands = [c for c in good if c not in others]
        if not cands:
            return None
        xs = [x for x, _y in p["cells"]]
        ys = [y for _x, y in p["cells"]]
        cx, cy = (min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2
        return min(cands, key=lambda c: (abs(c[0] - cx) + abs(c[1] - cy), c))

    def _pc_click(self, cell: Cell, why: str) -> GameAction:
        x, y = cell
        action = GameAction.ACTION6
        action.set_data({"x": x, "y": y})
        action.reasoning = {"why": why}
        self._prev_action = f"6:{x},{y}"
        return action

    def _pc_xform_act(self, avail: set[int],
                      rules: dict[int, Cell]) -> Optional[int]:
        """The in-place transform verb: a simple action that isn't a
        movement rule, most-voted first (cn04: ACTION5)."""
        cands = [a for a in sorted(avail)
                 if a not in rules and not GameAction.from_id(a).is_complex()
                 and a != GameAction.RESET.value]
        if not cands:
            return None
        return max(cands, key=lambda a: (self._xform_votes[a], -a))

    def _pc_fail(self, sol: dict) -> None:
        self._pc_failed_cfgs.add(self._pc_cfg_key(sol))
        self._pc_solution = None
        self._pc_strikes += 1

    def _pc_probe_xform(self, avail: set[int],
                        rules: dict[int, Cell]) -> Optional[GameAction]:
        """No pairing exists with the shapes seen so far: cycle pieces
        through the transform verb to discover hidden variants (cn04's
        stacked pieces only show one shape at a time — the right one must
        be FOUND).  Sync records each new shape; the solver re-runs as the
        library grows.  Budgeted per piece and per level."""
        act = self._pc_xform_act(avail, rules)
        if act is None:
            return None
        pieces = self._pc_pieces
        sel = self._pc_sel
        order = ([sel] if sel in pieces else []) \
            + [q for q in sorted(pieces) if q != sel]
        for pid in order:
            if not pieces[pid]["ports"] \
                    or self._pc_vscout[pid] >= XFORM_CAP:
                continue
            if pid != sel:
                cell = self._pc_click_cell(pid)
                if cell is None:
                    self._pc_vscout[pid] = XFORM_CAP
                    continue
                self._pc_sel = pid
                self._pc_bounce = 0
                return self._pc_click(cell, f"select piece {pid} to probe")
            self._pc_vscout[pid] += 1
            return self._step(GameAction.from_id(act))
        return None

    def _pc_exec(self, avail: set[int], rules: dict[int, Cell],
                 scale: int) -> Optional[GameAction]:
        """Drive the current solution one action at a time: select the
        first out-of-place piece, transform it to its target orientation,
        then walk it to its target position.  Reaching the full
        configuration without a level-up means the pairing was type-wrong:
        blacklist it and re-solve."""
        sol = self._pc_solution
        pieces = self._pc_pieces
        for pid in sorted(sol):
            if pid not in pieces:
                self._pc_fail(sol)
                return None
            p = pieces[pid]
            pat_t, pos_t = sol[pid]
            if pat_norm(p["cells"], p["ports"]) == pat_t \
                    and bbox_min(p["cells"]) == pos_t:
                continue
            if self._pc_sel != pid:
                cell = self._pc_click_cell(pid)
                if cell is None:
                    self._pc_fail(sol)
                    return None
                self._pc_sel = pid
                self._pc_bounce = 0
                self._pc_xform_spent = 0
                return self._pc_click(cell, f"select piece {pid}")
            if pat_norm(p["cells"], p["ports"]) != pat_t:
                act = self._pc_xform_act(avail, rules)
                if act is None or self._pc_xform_spent >= XFORM_CAP:
                    self._pc_unreachable.add((pid, pat_t))
                    self._pc_solution = None
                    self._pc_strikes += 1
                    return None
                self._pc_xform_spent += 1
                return self._step(GameAction.from_id(act))
            self._pc_xform_spent = 0
            if self._pc_bounce >= 2:
                self._pc_fail(sol)  # the route needs a clamped move
                self._pc_bounce = 0
                return None
            cx, cy = bbox_min(p["cells"])
            ddx, ddy = pos_t[0] - cx, pos_t[1] - cy
            for act, (adx, ady) in sorted(rules.items()):
                if (ddx and adx and (ddx > 0) == (adx > 0)
                        and abs(adx) <= abs(ddx)) \
                        or (ddy and ady and (ddy > 0) == (ady > 0)
                            and abs(ady) <= abs(ddy)):
                    return self._step(GameAction.from_id(act))
            self._pc_fail(sol)  # off-lattice: no rule closes the gap
            return None
        # every piece sits on its target and the level did NOT advance
        # (the win fires on the final placing move itself): geometrically
        # right, type-wrong — blacklist and try the next pairing
        self._pc_fail(sol)
        return None

    def _port_policy(
        self, grid: Optional[Grid], latest_frame: FrameData
    ) -> Optional[GameAction]:
        """Port-alignment policy, triple-gated: clicks available + four
        trusted axis movement rules of one magnitude + a voted port accent
        color.  Click-only games never form rules, ls20/tr87 expose no
        clicks, so only selected-piece games (cn04 family) can reach the
        model."""
        if grid is None:
            return None
        avail = set(latest_frame.available_actions or [])
        if GameAction.ACTION6.value not in avail:
            return None
        # port-mode rule hygiene: alignment needs the four axis moves of
        # one magnitude — collect, per direction, the best-voted act, and
        # DROP oddball deltas instead of letting them poison the gate (a
        # variant stack's cycle diffs fake a few translation votes for the
        # transform verb, L5's ping-pong shapes shift one blob diagonally)
        cand: dict[int, dict[Cell, int]] = {}
        for a, d in self._movement_rules().items():
            s = abs(d[0]) + abs(d[1])
            if s >= 2 and (d[0] == 0 or d[1] == 0):
                dirs = cand.setdefault(s, {})
                if d not in dirs or self._move_votes[a][d] \
                        > self._move_votes[dirs[d]][d]:
                    dirs[d] = a
        full = [(sum(self._move_votes[a][d] for d, a in dirs.items()),
                 s, dirs) for s, dirs in cand.items() if len(dirs) == 4]
        if not full:
            return None  # no full 2D control on a coarse lattice yet
        _votes, scale, dirs = max(full)
        rules = {a: d for d, a in dirs.items()}
        # (no register gate needed: ls20-family games expose no clicks, and
        # _note_attr_event stands down in click+port games, so the two
        # models cannot form on the same game)
        level = latest_frame.levels_completed
        if self._pc_benched == level:
            return None
        if self._port_color() is None:
            self._vote_port_color(grid, scale)
            return None
        self._pc_sync(grid, level, rules, scale)
        if self._pc_strikes >= PC_STRIKES:
            self._pc_benched = level
            return None
        pieces = self._pc_pieces
        with_ports = [pid for pid in sorted(pieces) if pieces[pid]["ports"]]
        unscouted = [pid for pid in sorted(pieces)
                     if not pieces[pid]["ports"]
                     and self._pc_scout[pid] < PC_SCOUT]
        if len(with_ports) + len(unscouted) < 2:
            return None  # nothing to pair on this board
        # 1 — learn which piece the arrows move right now: cheap arrow
        # probes first (sync reads the mover off the diff), then a click
        # (selection is then known by construction)
        if self._pc_sel is None:
            if self._pc_idprobe < len(rules):
                acts = sorted(rules)
                act = acts[self._pc_idprobe % len(acts)]
                self._pc_idprobe += 1
                return self._step(GameAction.from_id(act))
            pid = next((q for q in sorted(pieces)
                        if self._pc_click_cell(q) is not None), None)
            if pid is None:
                self._pc_benched = level
                return None
            self._pc_sel = pid
            self._pc_bounce = 0
            self._pc_idprobe = 0
            return self._pc_click(self._pc_click_cell(pid),
                                  f"select piece {pid}")
        # 2 — grey masking hides unselected ports: select each blank piece
        # once to scout them
        for pid in unscouted:
            if pid == self._pc_sel:
                self._pc_scout[pid] = PC_SCOUT  # selected and still blank
                continue
            cell = self._pc_click_cell(pid)
            if cell is None:
                self._pc_scout[pid] = PC_SCOUT
                continue
            self._pc_scout[pid] += 1
            self._pc_sel = pid
            self._pc_bounce = 0
            return self._pc_click(cell, f"scout piece {pid}")
        # 3 — plan a full pairing; 4 — drive it one action at a time
        for _retry in range(2):
            if self._pc_solution is None:
                sols = self._pc_solve(scale)
                if not sols:
                    # maybe a piece is hiding the shape that would pair up
                    # (variant stacks): probe transforms before giving up
                    probe = self._pc_probe_xform(avail, rules)
                    if probe is not None:
                        return probe
                    self._pc_benched = level
                    return None
                self._pc_solution = sols[0]
                self._pc_xform_spent = 0
            action = self._pc_exec(avail, rules, scale)
            if action is not None:
                return action
            if self._pc_strikes >= PC_STRIKES:
                self._pc_benched = level
                return None
        return None

    # ── editor model (tr87 family): cursor + in-place glyph cycling ─────
    def _ed_note(
        self, act: int, grid: Grid,
        groups: dict[Cell, list[tuple[int, frozenset[Cell]]]],
    ) -> None:
        """Classify one simple action's frame-diff: a clean small-pattern
        translation with nothing else changing (a SEL_MOVE — the cursor
        marker stepping between slots, wraparound included), an interior
        change confined to one compact bbox with nothing translated (an
        in-place MUTATE — a glyph slot cycling under the cursor), a no-op,
        or unmodelled noise.  Border-band cells are exempt (budget bars
        tick there); click-capable games never feed this."""
        if GameAction.ACTION6.value in self._avail:
            return
        cls = self._ed_class[act]
        changed: set[Cell] = set()
        for y in range(HUD_BAND, GRID - HUD_BAND):
            prow, row = self._prev_grid[y], grid[y]
            if prow == row:
                continue
            for x in range(HUD_BAND, GRID - HUD_BAND):
                if prow[x] != row[x] and (x, y) not in self._hud_mask:
                    changed.add((x, y))
        if not changed:
            cls["noop"] += 1
            return
        if len(groups) == 1:
            # SEL_MOVE candidate — but only when the translation is REAL:
            # the source region must have vanished (a swapped glyph reads
            # as a fake "translation" from any identical unchanged twin,
            # whose source cells never changed) and source/destination
            # must be disjoint (an in-place variant swap that happens to
            # be a shifted copy of its predecessor overlaps itself), and
            # old+new cells must explain the whole diff.
            (delta, movers), = groups.items()
            union = {cell for _c, cells in movers for cell in cells}
            src = {(x - delta[0], y - delta[1]) for x, y in union}
            if union.isdisjoint(src):
                both = {(x, y) for x, y in union | src
                        if HUD_BAND <= x < GRID - HUD_BAND
                        and HUD_BAND <= y < GRID - HUD_BAND
                        and (x, y) not in self._hud_mask}
                if both == changed:
                    cls["mov"] += 1
                    self._ed_deltas[act][delta] += 1
                    ax, ay = min(union)
                    self._ed_marker[act][frozenset(
                        (grid[y][x], x - ax, y - ay)
                        for x, y in union)] += 1
                    return
        xs = [x for x, _y in changed]
        ys = [y for _x, y in changed]
        if max(xs) - min(xs) < ED_MUT_BBOX and max(ys) - min(ys) < ED_MUT_BBOX:
            cls["mut"] += 1
            self._ed_boxes.append((min(xs), min(ys), max(xs), max(ys)))
            del self._ed_boxes[:-8]
        else:
            cls["other"] += 1

    def _ed_verb(self, act: int, kind: str) -> bool:
        """Is this action a trusted 'mut' (cycle) or 'mov' (selector) verb?
        Plurality over the rival kind only: a cycle press aliases as a
        clean translation whenever the new glyph variant happens to be a
        translate of the old one (measured every 7th press on tr87), so a
        strict dominance test would flicker the verb off mid-edit; real
        selector/movement actions produce essentially zero mutate events,
        so plain majority separates the two cleanly."""
        c = self._ed_class.get(act)
        if c is None:
            return False
        n = c[kind]
        rival = c["mut"] + c["mov"] - n
        return n >= ED_VOTES and n >= rival

    def _ed_marker_sig(self, sel: list[int]) -> Optional[frozenset]:
        """The cursor's appearance: the most-voted pattern the selector
        verbs translate (tr87: the two-bracket pair as one rigid union)."""
        cnt: Counter[frozenset] = Counter()
        for a in sel:
            cnt.update(self._ed_marker[a])
        if not cnt:
            return None
        sig, n = cnt.most_common(1)[0]
        return sig if n >= 2 else None

    def _ed_sel_step(self, sel: list[int], want: int) -> GameAction:
        """The selector action whose dominant step points the wanted way
        along the row (any selector works as a fallback: slot rows wrap)."""
        for a in sel:
            votes = self._ed_deltas.get(a)
            if not votes:
                continue
            (dx, dy), _n = votes.most_common(1)[0]
            dirn = (1 if dx > 0 else -1) if abs(dx) >= abs(dy) \
                else (1 if dy > 0 else -1)
            if dirn == want:
                return GameAction.from_id(a)
        return GameAction.from_id(sel[0])

    @staticmethod
    def _ed_cards(grid: Grid, pitch: int) -> list[tuple[int, int, int, tuple]]:
        """Card sites as (x, y, ring_color, canonical_sig): every p x p
        window with a monochrome non-background border ring and a non-
        uniform interior — a backing tile holding a glyph.  Position-free,
        so rule cards, the frozen reference row and the mutable row all
        read out in one pass; misaligned windows clip glyph ink with their
        ring and drop out."""
        background = background_of(grid)
        out: list[tuple[int, int, int, tuple]] = []
        for y in range(GRID - pitch + 1):
            top, bot = grid[y], grid[y + pitch - 1]
            for x in range(GRID - pitch + 1):
                ring = top[x]
                if ring == background:
                    continue
                ok = all(top[x + d] == ring and bot[x + d] == ring
                         for d in range(pitch))
                if ok:
                    ok = all(grid[y + d][x] == ring
                             and grid[y + d][x + pitch - 1] == ring
                             for d in range(1, pitch - 1))
                if not ok:
                    continue
                if all(grid[y + dy][x + dx] == ring
                       for dy in range(1, pitch - 1)
                       for dx in range(1, pitch - 1)):
                    continue  # uniform block: no glyph inside
                out.append((x, y, ring, canon_window(grid, x, y, pitch)))
        return out

    @staticmethod
    def _ed_rows(
        cards: list[tuple[int, int, int, tuple]], pitch: int
    ) -> list[tuple[int, list[tuple[int, int, tuple]]]]:
        """Cards grouped into horizontal pitch-adjacent runs: (y, run).
        Runs chain within one x-residue class, and when two runs at the
        same y overlap in span the longer one wins: a lucky glyph variant
        can leave a clean ring for a misaligned window straddling two real
        cards (measured on tr87 level 2), and letting that phantom break
        the true chain split the mutable row in half."""
        by_y: dict[int, list[tuple[int, int, tuple]]] = defaultdict(list)
        for x, y, ring, sig in cards:
            by_y[y].append((x, ring, sig))
        rows: list[tuple[int, list[tuple[int, int, tuple]]]] = []
        for y, items in sorted(by_y.items()):
            runs: list[list[tuple[int, int, tuple]]] = []
            by_res: dict[int, list[tuple[int, int, tuple]]] = defaultdict(list)
            for it in sorted(items):
                by_res[it[0] % pitch].append(it)
            for res in sorted(by_res):
                run = [by_res[res][0]]
                for it in by_res[res][1:]:
                    if it[0] == run[-1][0] + pitch:
                        run.append(it)
                    else:
                        runs.append(run)
                        run = [it]
                runs.append(run)
            kept: list[list[tuple[int, int, tuple]]] = []
            for run in sorted(runs, key=lambda r: (-len(r), r[0][0])):
                span = (run[0][0], run[-1][0] + pitch)
                if not any(span[0] < k[-1][0] + pitch and k[0][0] < span[1]
                           for k in kept):
                    kept.append(run)
            for run in sorted(kept, key=lambda r: r[0][0]):
                rows.append((y, run))
        return rows

    def _ed_infer_goals(
        self, grid: Grid, pitch: int,
        rows: list[tuple[int, list]], mut_y: int, mut_run: list,
    ) -> list[tuple]:
        """Rule mining + analogy.  Same-row card runs whose gap region
        shows a separator marker (non-uniform pixels between them) are
        LHS→RHS rewrite examples; the unpaired run nearest the mutable row
        is the frozen reference; goal candidates are the reference
        rewritten through the rules (every DFS segmentation, depth 1 then
        2 for chained variants), accepted when the output length matches
        the mutable row."""
        rules: dict[tuple, tuple] = {}
        unpaired: list[tuple[int, list]] = []
        by_y: dict[int, list[list]] = defaultdict(list)
        for y, run in rows:
            by_y[y].append(run)
        for y, runs in sorted(by_y.items()):
            runs.sort(key=lambda r: r[0][0])
            used: set[int] = set()
            i = 0
            while i < len(runs) - 1:
                a, b = runs[i], runs[i + 1]
                if a is mut_run or b is mut_run:
                    i += 1
                    continue
                gx0 = a[-1][0] + pitch
                gx1 = b[0][0]
                gap = {grid[y + dy][x]
                       for dy in range(pitch) for x in range(gx0, gx1)}
                if 0 < gx1 - gx0 <= 3 * pitch and len(gap) > 1:
                    rules[tuple(c[2] for c in a)] = tuple(c[2] for c in b)
                    used.update((i, i + 1))
                    i += 2
                else:
                    i += 1
            for j, run in enumerate(runs):
                if j not in used and run is not mut_run:
                    unpaired.append((y, run))
        if not rules or not unpaired:
            return []
        ref = min(unpaired, key=lambda yr: abs(yr[0] - mut_y))[1]
        seq = tuple(c[2] for c in ref)
        want = len(mut_run)

        def expand(s: tuple, cap: int = 24) -> list[tuple]:
            outs: list[tuple] = []

            def dfs(i: int, acc: list) -> None:
                if len(outs) >= cap:
                    return
                if i == len(s):
                    outs.append(tuple(acc))
                    return
                for lhs, rhs in rules.items():
                    if s[i:i + len(lhs)] == lhs:
                        dfs(i + len(lhs), acc + list(rhs))
            dfs(0, [])
            return outs

        once = expand(seq)
        goals = [g for g in once if len(g) == want]
        if not goals:
            for mid in once:
                goals += [g for g in expand(mid) if len(g) == want]
        seen: set[tuple] = set()
        out: list[tuple] = []
        for g in goals:
            if g not in seen:
                seen.add(g)
                out.append(g)
        return out[:ED_GOALS]

    def _ed_bench(self, level: int) -> None:
        self._ed_benched = level
        self._ed_strikes = 0
        self._ed_benches += 1
        if self._ed_benches >= 2 and self._ed_wins == 0:
            self._ed_dead = True  # this is not an editor game after all

    def _editor_policy(
        self, grid: Optional[Grid], latest_frame: FrameData
    ) -> Optional[GameAction]:
        """Factored cursor+editor policy, triple-gated at engage time:
        simple-actions-only (no clicks) + a trusted in-place cycle verb +
        no trusted movement rules (avatar games belong to the attr/maze
        planners).  Once engaged it probes every available action to
        classify it, parses the card board, infers the goal by analogy
        and edits slot by slot; persistent failure benches the level, two
        benched levels with zero wins kill the branch for the game."""
        if grid is None or self._ed_dead:
            return None
        avail = set(latest_frame.available_actions or [])
        if GameAction.ACTION6.value in avail:
            return None
        simple = sorted(
            a for a in avail
            if a != GameAction.RESET.value
            and not GameAction.from_id(a).is_complex())
        if not simple:
            return None
        cyc = [a for a in simple if self._ed_verb(a, "mut")]
        if not self._ed_engaged:
            if not cyc or self._movement_rules():
                return None
            self._ed_engaged = True
        if not cyc:
            return None
        level = latest_frame.levels_completed
        if self._ed_benched == level:
            return None
        if self._ed_strikes >= ED_STRIKES:
            self._ed_bench(level)
            return None
        sel = [a for a in simple if self._ed_verb(a, "mov")]
        # 1 — probe: classify every available action before planning
        for a in simple:
            if not (self._ed_verb(a, "mut") or self._ed_verb(a, "mov")) \
                    and self._ed_probe[a] < ED_PROBES:
                self._ed_probe[a] += 1
                return self._step(GameAction.from_id(a))
        if not sel:
            self._ed_bench(level)  # no way to move the cursor: not editable
            return None
        # 2 — slot pitch: the selector verbs' dominant step magnitude
        votes: Counter[Cell] = Counter()
        for a in sel:
            votes.update(self._ed_deltas[a])
        (dx, dy), _n = votes.most_common(1)[0]
        pitch = max(abs(dx), abs(dy))
        if not 3 <= pitch <= 16:
            self._ed_bench(level)
            return None
        # 3 — board parse; the mutable row is the run where mutates landed
        cards = self._ed_cards(grid, pitch)
        rows = self._ed_rows(cards, pitch)
        cyc_act = max(cyc, key=lambda a: self._ed_class[a]["mut"])
        mut_y, mut_run = -1, None
        for box in reversed(self._ed_boxes):
            cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
            for y, run in rows:
                if y <= cy < y + pitch \
                        and any(x <= cx < x + pitch for x, _r, _s in run):
                    mut_y, mut_run = y, run
                    break
            if mut_run is not None:
                break
        if mut_run is None:
            # nothing seen mutating on this level yet (fresh level), or
            # perception drift: one cycle press re-reveals the live slot
            self._ed_miss += 1
            if self._ed_miss > ED_MISS_CAP:
                self._ed_miss = 0
                self._ed_strikes += 1
            return self._step(GameAction.from_id(cyc_act))
        # 4 — cursor: template-match the marker, read the selected slot
        marker = self._ed_marker_sig(sel)
        hit = self._match_signature(grid, marker) if marker else None
        sel_idx = None
        if hit is not None:
            w = max(mdx for _c, mdx, _mdy in marker) + 1
            mid = hit[0][0] + w // 2
            for i, (x, _r, _s) in enumerate(mut_run):
                if x <= mid < x + pitch:
                    sel_idx = i
                    break
        if sel_idx is None:
            # cursor invisible or parked off the mutable row: step it
            self._ed_miss += 1
            if self._ed_miss > ED_MISS_CAP:
                self._ed_miss = 0
                self._ed_strikes += 1
            return self._step(self._ed_sel_step(sel, 1))
        self._ed_miss = 0
        # 5 — goal by analogy (memoized per level)
        if self._ed_goal is None:
            self._ed_goal = self._ed_infer_goals(
                grid, pitch, rows, mut_y, mut_run)
            self._ed_goal_idx = 0
            if not self._ed_goal:
                self._ed_bench(level)
                return None
        cur = [s for _x, _r, s in mut_run]
        if any(len(g) != len(cur) for g in self._ed_goal):
            # the parsed row is shorter/longer than every candidate was
            # inferred against: a perception dropout, not evidence against
            # the goal — step the cursor (content-safe) and re-read
            self._ed_miss += 1
            if self._ed_miss > ED_MISS_CAP:
                self._ed_miss = 0
                self._ed_strikes += 1
            return self._step(self._ed_sel_step(sel, 1))
        wrong: list[int] = []
        while self._ed_goal_idx < len(self._ed_goal):
            goal = self._ed_goal[self._ed_goal_idx]
            wrong = [i for i, (c, g) in enumerate(zip(cur, goal)) if c != g]
            if wrong:
                break
            if self._ed_full_seen:
                # completed this candidate before and the level didn't
                # advance: the segmentation was wrong — falsify it
                self._ed_goal_idx += 1
                self._ed_full_seen = False
                self._ed_strikes += 1
                continue
            # the board already matches, but wins only fire on a cycle
            # press: nudge one slot off and back to force the check
            self._ed_full_seen = True
            self._ed_spent = 0
            return self._step(GameAction.from_id(cyc_act))
        if not wrong:
            self._ed_bench(level)  # every candidate falsified
            return None
        # 6 — edit: cycle the selected slot when wrong, else walk the
        # cursor to the nearest wrong slot (rows wrap around)
        if sel_idx in wrong:
            if self._ed_spent >= ED_CYCLE_CAP:
                # the goal signature never came around: model is off
                self._ed_spent = 0
                self._ed_full_seen = False
                self._ed_goal_idx += 1
                self._ed_strikes += 1
                return self._editor_policy(grid, latest_frame)
            self._ed_spent += 1
            return self._step(GameAction.from_id(cyc_act))
        self._ed_spent = 0
        n = len(cur)
        fwd = min((w - sel_idx) % n for w in wrong)
        bwd = min((sel_idx - w) % n for w in wrong)
        return self._step(self._ed_sel_step(sel, 1 if fwd <= bwd else -1))

    # ── lattice recolor model: board detection + click-effect learning ──
    def _detect_board(
        self, grid: Grid,
        comps: Optional[list[tuple[int, frozenset[Cell]]]] = None,
    ) -> Optional[tuple[int, int, int, int, int, dict]]:
        """Detect a regular lattice of same-shaped sprites (a click-puzzle
        board) as (pitch, w, h, ox, oy, sites).

        Seed: the color-blind shape group whose anchors best agree on one
        square pitch AND phase — the phase vote is what excludes ft09's
        top-left tutorial cluster, which shares the cell shape but sits at
        a different y-phase and would poison a plain gcd.  Sites are then
        sampled geometrically over the whole frame, so pattern cells, clue
        markers and junk straddling a site all get an appearance key:
        site → (ax, ay, class_key, center_color).  Class keys relabel
        colors by first appearance (background → -1), so a solid cell of
        ANY palette color is one class and effect masks transfer across
        recolors and levels."""
        if comps is None:
            comps = components(grid)
        groups: dict[tuple, list[Cell]] = defaultdict(list)
        for _color, cells in comps:
            mx = min(x for x, _y in cells)
            my = min(y for _x, y in cells)
            w = max(x for x, _y in cells) - mx + 1
            h = max(y for _x, y in cells) - my + 1
            groups[(w, h, frozenset((x - mx, y - my) for x, y in cells))] \
                .append((mx, my))
        best: Optional[tuple[int, int, int, int, int]] = None
        best_score = 0
        for (w, h, _shape), anchors in groups.items():
            if len(anchors) < LATTICE_MIN_CELLS or max(w, h) < 2:
                continue
            side = max(w, h, 4)  # pitch can't be smaller than the sprite
            pitches = {b - a
                       for vs in (sorted({x for x, _y in anchors}),
                                  sorted({y for _x, y in anchors}))
                       for a, b in zip(vs, vs[1:]) if side <= b - a <= GRID // 2}
            for p in sorted(pitches):
                phases: Counter[Cell] = Counter(
                    (x % p, y % p) for x, y in anchors)
                phase, n = phases.most_common(1)[0]
                # area-weighted: board cells dominate the frame's pixel real
                # estate, so 27 6x6 cells outrank the 30+ tiny 2x2 fragments
                # inside ft09 level 5's clue glyphs, which agree on a bogus
                # pitch-4 lattice by sheer count
                if n < LATTICE_MIN_CELLS or n * w * h <= best_score:
                    continue
                inliers = [a for a in anchors
                           if (a[0] % p, a[1] % p) == phase]
                if len({x for x, _y in inliers}) < 2 \
                        or len({y for _x, y in inliers}) < 2:
                    continue  # a 2D lattice, not a strip of repeats
                best, best_score = (p, w, h, phase[0], phase[1]), n * w * h
        if best is None:
            return None
        p, w, h, ox, oy = best
        counts: Counter[int] = Counter()
        for row in grid:
            counts.update(row)
        background = counts.most_common(1)[0][0]
        sites: dict[Cell, tuple[int, int, tuple, int]] = {}
        for j, ay in enumerate(range(oy, GRID - h + 1, p)):
            for i, ax in enumerate(range(ox, GRID - w + 1, p)):
                relabel: dict[int, int] = {}
                key = tuple(
                    -1 if (v := grid[ay + dy][ax + dx]) == background
                    else relabel.setdefault(v, len(relabel))
                    for dy in range(h) for dx in range(w))
                if relabel:  # at least one non-background pixel
                    sites[(i, j)] = (ax, ay, key,
                                     grid[ay + h // 2][ax + w // 2])
        return (p, w, h, ox, oy, sites)

    def _site_at(self, board: tuple, cell: Cell) -> Optional[Cell]:
        """Lattice coords of the site whose sprite rect contains cell."""
        p, w, h, ox, oy, sites = board
        x, y = cell
        if x < ox or y < oy or (x - ox) % p >= w or (y - oy) % p >= h:
            return None
        ij = ((x - ox) // p, (y - oy) // p)
        return ij if ij in sites else None

    def _learn_click_fx(
        self, grid: Grid, click: Cell,
        comps_prev: list[tuple[int, frozenset[Cell]]], leveled: bool,
    ) -> None:
        """When a click on a board site changed the world by in-place
        recolors ALONE, record which lattice offsets recolored, keyed by
        the clicked site's appearance class — and the color cycle old→new.
        A click that changed nothing marks the site dead (clue markers and
        decorations don't react).  Anything that doesn't diff down to clean
        one-site recolors (hint animations, death flashes, translations,
        level repaints) is dropped, not guessed at."""
        if leveled:
            return  # cross-level repaint is not click physics
        board = self._detect_board(self._prev_grid, comps_prev)
        if board is None:
            return
        site = self._site_at(board, click)
        if site is None:
            return
        ck = board[5][site][2]
        rec = recolored_objects(self._prev_grid, grid)
        if not rec:
            if self._same_unmasked(self._prev_grid, grid):
                self._probed_classes.add(ck)
                self._site_dead.add(site)
            return
        # every changed pixel in the board interior must belong to a
        # recolored object: translations (lp85), animations and repaints
        # leave strays.  (A moved_objects veto is wrong here — a recolored
        # cell aliases as a "translation" of any identical same-color
        # sprite elsewhere.)  The border band is exempt: that's where
        # budget bars tick, and they tick too RARELY for the HUD mask to
        # ever catch them in a game where most exploratory clicks are
        # no-ops (ft09 measured 12% churn vs the 40% mask threshold).
        explained: set[Cell] = set()
        for cells, _old, _new in rec:
            explained |= cells
        for y, (prow, row) in enumerate(zip(self._prev_grid, grid)):
            if prow == row:
                continue
            for x, (pv, v) in enumerate(zip(prow, row)):
                if pv != v and (x, y) not in explained \
                        and (x, y) not in self._hud_mask \
                        and HUD_BAND <= x < GRID - HUD_BAND \
                        and HUD_BAND <= y < GRID - HUD_BAND:
                    return
        offsets: set[Cell] = set()
        for cells, _old, _new in rec:
            xs = [x for x, _y in cells]
            ys = [y for _x, y in cells]
            lo = self._site_at(board, (min(xs), min(ys)))
            hi = self._site_at(board, (max(xs), max(ys)))
            if lo is None or lo != hi:
                return  # recolor outside / across sites: garbage event
            offsets.add((lo[0] - site[0], lo[1] - site[1]))
        for _cells, old, new in rec:
            self._palette_next[old] = new
        self._click_fx.setdefault(ck, Counter())[frozenset(offsets)] += 1
        self._probed_classes.add(ck)

    def _fx_mask(self, ck: tuple) -> set[Cell]:
        """Union of observed recolor offsets for an appearance class (edge
        cells truncate the mask, so the union over observations is the
        closest available estimate of the true effect pattern)."""
        out: set[Cell] = set()
        for offsets in self._click_fx.get(ck, ()):
            out |= offsets
        return out

    def _palette_ring(self) -> Optional[list[int]]:
        """The learned color cycle, or None while it hasn't closed."""
        if not self._palette_next:
            return None
        start = min(self._palette_next)
        ring = [start]
        cur = self._palette_next[start]
        while cur != start:
            if cur in ring or cur not in self._palette_next:
                return None  # rho-shaped (polluted) or still open
            ring.append(cur)
            cur = self._palette_next[cur]
        known = set(self._palette_next) | set(self._palette_next.values())
        return ring if set(ring) == known else None

    def _update_hud_mask(self, grid: Grid) -> None:
        """Track cell change rates; mask high-churn border strips (HUD chrome).

        HUDs (step bars, pips, timers) repaint near the frame edge on almost
        every action, so raw grid hashes never repeat and every frame looks
        novel.  The band restriction is the safety property: gameplay in the
        board interior can never be masked.  The mask FREEZES after
        HUD_FREEZE diffed frames so state keys stop churning mid-run.
        """
        if self._frames_diffed >= HUD_FREEZE:
            return
        bit = 1 << self._frames_diffed
        for y in range(GRID):
            prow, row = self._prev_grid[y], grid[y]
            if prow != row:
                for x in range(GRID):
                    if prow[x] != row[x]:
                        self._cell_change[(x, y)] += 1
                        if x < HUD_BAND or x >= GRID - HUD_BAND \
                                or y < HUD_BAND or y >= GRID - HUD_BAND:
                            self._band_frames[(x, y)] |= bit
        self._frames_diffed += 1
        if self._frames_diffed >= HUD_WARMUP \
                and self._frames_diffed % HUD_RECOMPUTE == 0:
            self._recompute_hud_mask()

    def _recompute_hud_mask(self) -> None:
        """Mask contiguous border-band runs whose churn rate exceeds HUD_RATE.

        Step bars SWEEP: each cell flips only ~twice per life, but some cell
        in the bar flips nearly every frame — the churn signal lives at the
        strip level, not the cell level.  A run = a maximal contiguous
        stretch of ever-changed band cells along one row/column; its churn
        rate = the fraction of diffed frames in which ANY of its cells
        changed.  Row runs claim cells first so a masked bar can't bleed
        into perpendicular gameplay (e.g. a target display just below it);
        rarely-changing neighbours (lock pips, glyph registers) stay
        unmasked because the gap between runs separates them.
        """
        cutoff = self._frames_diffed * HUD_RATE
        masked: set[Cell] = set()
        for horizontal in (True, False):
            lines: dict[int, list[int]] = defaultdict(list)
            for x, y in self._band_frames:
                if horizontal:
                    lines[y].append(x)
                elif (x, y) not in masked:  # row runs take precedence
                    lines[x].append(y)
            for fixed, pos in lines.items():
                pos.sort()
                run: list[int] = []
                for p in pos + [GRID + 2]:  # sentinel flushes the last run
                    if run and p - run[-1] > 1:
                        cells = [(q, fixed) if horizontal else (fixed, q)
                                 for q in run]
                        churn = 0
                        for cell in cells:
                            churn |= self._band_frames[cell]
                        if churn.bit_count() > cutoff:
                            masked.update(cells)
                        run = []
                    run.append(p)
        self._hud_mask = frozenset(masked)

    def _masked_hash(self, grid: Optional[Grid]) -> int:
        """grid_hash with HUD cells zeroed; plain grid_hash while mask is empty."""
        if grid is None or not self._hud_mask:
            return grid_hash(grid)
        return hash(tuple(
            tuple(0 if (x, y) in self._hud_mask else v for x, v in enumerate(row))
            for y, row in enumerate(grid)
        ))

    def _same_unmasked(self, a: Grid, b: Grid) -> bool:
        """Grid equality ignoring masked HUD cells."""
        if not self._hud_mask:
            return a == b
        return all(
            ra == rb or all(va == vb or (x, y) in self._hud_mask
                            for x, (va, vb) in enumerate(zip(ra, rb)))
            for y, (ra, rb) in enumerate(zip(a, b))
        )

    def _movement_rules(self) -> dict[int, Cell]:
        rules = {}
        for act, votes in self._move_votes.items():
            delta, n = votes.most_common(1)[0]
            if n >= VOTE_THRESHOLD:
                rules[act] = delta
        return rules

    def _avatar_color(self) -> Optional[int]:
        if not self._avatar_votes:
            return None
        color, n = self._avatar_votes.most_common(1)[0]
        return color if n >= VOTE_THRESHOLD else None

    def _avatar_signature(self) -> Optional[frozenset]:
        if not self._avatar_sigs:
            return None
        sig, n = self._avatar_sigs.most_common(1)[0]
        return sig if n >= VOTE_THRESHOLD else None

    def _match_signature(
        self, grid: Grid, sig: frozenset
    ) -> Optional[tuple[Cell, frozenset[Cell]]]:
        """Template-scan for the avatar signature, anchored on its (0,0) cell."""
        anchor_color = next(c for c, dx, dy in sig if (dx, dy) == (0, 0))
        for y in range(GRID):
            row = grid[y]
            for x in range(GRID):
                if row[x] != anchor_color:
                    continue
                if all(0 <= x + dx < GRID and 0 <= y + dy < GRID
                       and grid[y + dy][x + dx] == c for c, dx, dy in sig):
                    return ((x, y),
                            frozenset((x + dx, y + dy) for _c, dx, dy in sig))
        return None

    def _find_avatar(self, grid: Grid) -> Optional[tuple[Cell, frozenset[Cell]]]:
        """Locate the avatar as (anchor, cells): template-match the trusted
        multi-color signature; fall back to the smallest component of the
        voted color only when the scan misses (protects games where the
        avatar overlaps other objects and breaks the template)."""
        sig = self._avatar_signature()
        if sig is not None:
            hit = self._match_signature(grid, sig)
            if hit:
                return hit
        color = self._avatar_color()
        if color is None:
            return None
        comps = [cells for c, cells in components(grid) if c == color]
        if not comps:
            return None
        cells = min(comps, key=len)
        return (min(cells), cells)

    # ── planning ─────────────────────────────────────────────────────────
    def _policy(self, grid: Optional[Grid], latest_frame: FrameData) -> GameAction:
        if grid is not None:
            # triple-gated lattice solver (click-only + board + learned
            # recolor effect); returns None whenever the gate doesn't apply
            lattice = self._lattice_policy(grid, latest_frame)
            if lattice is not None:
                return lattice
            # factored cursor+editor model (tr87 family), gated inside on
            # no clicks + a trusted in-place cycle verb + no movement
            # rules at engage time.  Runs BEFORE the warmup bench: the
            # editor's own probe phase replaces novelty warmup (a 128-
            # action budget game cannot afford 500 random presses).
            editor = self._editor_policy(grid, latest_frame)
            if editor is not None:
                return editor
            # attribute-state planning (ls20 family), double-gated on
            # trusted movement rules + a detected register.  Runs BEFORE
            # the novelty-patience bench: every attr plan strictly grows
            # the model (a new gate reject, effect entry, probe or anchor),
            # so benching it for PLANNER_COOLDOWN steps just burns the step
            # budget of a 3-lives game on random moves.  The readiness test
            # keeps it from monopolizing the policy while an available
            # action still lacks a trusted rule — that starves discovery.
            reg = self._attr_register()
            rules = self._movement_rules()
            avail = set(latest_frame.available_actions or []) \
                - {GameAction.RESET.value, GameAction.ACTION6.value}
            ready = all(a in rules or self._act_uses[a] >= RULE_TRIES
                        for a in avail)
            if reg is not None and len(rules) >= 2 and ready:
                avatar = self._find_avatar(grid)
                if avatar:
                    if self._plan:
                        return self._step(self._plan.popleft())
                    plan = self._plan_attr_route(grid, avatar, rules, reg)
                    if plan:
                        self._plan.extend(plan[1:])
                        return self._step(plan[0])
            # selected-piece port alignment (cn04 family), gated inside on
            # clicks + full 2D movement + a voted port color + no register.
            # Above the warmup gate: every port action either reveals
            # hidden ports or executes an exact pairing plan.
            port = self._port_policy(grid, latest_frame)
            if port is not None:
                return port
        # warmup ends early once the model is live (2+ trusted movement rules
        # and a located avatar): 500 random steps would burn through ls20's
        # 42-pip lives and cn04's 75-action budget before planning ever
        # starts.  Click-only games never form rules, so the cheap first
        # conjunct keeps their warmup bit-identical.
        if self.action_counter < NOVELTY_WARMUP \
                and not (len(self._movement_rules()) >= 2
                         and grid is not None and self._find_avatar(grid)):
            return self._novelty_policy(grid, latest_frame)
        if self._planner_cooldown > 0:
            self._planner_cooldown -= 1
            return self._novelty_policy(grid, latest_frame)
        # a planner that stopped finding new states is stuck in a loop the
        # world model can't see — bench it and let raw exploration break out
        if self._steps_since_novelty > PLANNER_PATIENCE:
            self._plan.clear()
            self._planner_cooldown = PLANNER_COOLDOWN
            return self._novelty_policy(grid, latest_frame)
        rules = self._movement_rules()
        if grid is not None and len(rules) >= 2:
            avatar = self._find_avatar(grid)
            if avatar:
                if self._plan:
                    return self._step(self._plan.popleft())
                plan = self._plan_route(grid, avatar, rules)
                if not plan:
                    plan = self._probe_tiles(grid, avatar, rules)
                if plan:
                    self._plan.extend(plan[1:])
                    return self._step(plan[0])
        return self._novelty_policy(grid, latest_frame)

    def _plan_route(
        self, grid: Grid, avatar: tuple[Cell, frozenset[Cell]],
        rules: dict[int, Cell]
    ) -> list[GameAction]:
        anchor, cells = avatar
        shape = frozenset((x - anchor[0], y - anchor[1]) for x, y in cells)
        self._visited_anchors.add(anchor)

        targets = set()
        for _color, comp_cells in components(grid):
            if not (comp_cells & cells) and len(comp_cells) < len(cells) * 8:
                targets |= comp_cells

        def ok(pos: Cell) -> bool:
            return all(
                0 <= pos[0] + sx < GRID and 0 <= pos[1] + sy < GRID
                and (pos[0] + sx, pos[1] + sy) not in self._walls
                for sx, sy in shape
            )

        def is_goal(pos: Cell) -> bool:
            body = {(pos[0] + sx, pos[1] + sy) for sx, sy in shape}
            near = {(x + dx, y + dy) for x, y in body
                    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0), (0, 0))}
            return bool(near & targets) or (pos not in self._visited_anchors)

        best: Optional[list[GameAction]] = None
        seen = {anchor}
        dq: deque[tuple[Cell, list[GameAction]]] = deque([(anchor, [])])
        while dq and len(seen) < 3000:
            pos, path = dq.popleft()
            if path and is_goal(pos):
                best = path
                break
            for act, (dx, dy) in rules.items():
                nxt = (pos[0] + dx, pos[1] + dy)
                if nxt in seen or not ok(nxt):
                    continue
                seen.add(nxt)
                dq.append((nxt, path + [GameAction.from_id(act)]))
        return best or []

    def _step(self, action: GameAction) -> GameAction:
        action.reasoning = {"why": "planned route step"}
        self._prev_action = str(action.value)
        return action

    # ── lattice puzzle solver (ft09 family) ─────────────────────────────
    def _lattice_policy(
        self, grid: Grid, latest_frame: FrameData
    ) -> Optional[GameAction]:
        """Solve recolor click-puzzles over a detected board.  The triple
        gate is load-bearing: ACTION6-only games + a uniform-pitch lattice
        + at least one LEARNED recolor effect — lp85 clicks translate and
        vc33 has no lattice, so neither can reach this branch.  Probe each
        unseen appearance class once; read probed-static sites as ==/!=
        constraints on their neighbours; click violated cells (greedy is
        exact for identity masks); falsify a polarity that contradicts
        itself or is satisfied without a level-up, and bench the solver for
        the level when both polarities are spent or retries stop paying."""
        avail = set(latest_frame.available_actions or []) \
            - {GameAction.RESET.value}
        if avail != {GameAction.ACTION6.value} or not self._click_fx:
            return None
        level = latest_frame.levels_completed
        if self._lattice_benched == level:
            return None
        if self._lattice_engaged \
                and self._level_deaths - self._deaths_at_engage \
                >= LATTICE_DEATH_CAP:
            self._lattice_benched = level
            return None
        board = self._detect_board(grid)
        if board is None:
            return None
        sites = board[5]
        order = sorted(sites, key=lambda ij: (ij[1], ij[0]))
        for ij in order:  # probe each unseen appearance class once
            ck = sites[ij][2]
            if ck in self._probed_classes or ck in self._probe_gaveup \
                    or ij in self._site_dead:
                continue
            if self._probe_sent[ck] >= LATTICE_PROBE_CAP:
                self._probe_gaveup.add(ck)  # unlearnable: not cell, not clue
                continue
            self._probe_sent[ck] += 1
            return self._lattice_click(board, ij, "probe appearance class")
        ring = self._palette_ring()
        if ring is None:
            # the color cycle is level data and must close before clue
            # readings can be judged complete (a clue only becomes visible
            # once its target color is known to cycle): click a responsive
            # cell whose color has no learned successor yet
            for ij in order:
                cell = sites[ij]
                if cell[2] in self._click_fx and ij not in self._site_dead \
                        and cell[3] not in self._palette_next:
                    return self._lattice_click(board, ij, "expand palette ring")
        for _ in range(2):  # current polarity, then one falsification flip
            cons = self._lattice_constraints(grid, board)
            if not cons:
                return None
            viol, broken = self._lattice_violations(board, cons)
            if not broken and viol:
                return self._lattice_solve(board, cons, viol, level)
            # contradictory reading, or satisfied without a level-up (the
            # game checks the goal after every cell click): polarity is
            # wrong — but only judge it on COMPLETE knowledge, or a clue
            # whose target color just hasn't cycled into view yet would
            # falsify a correct polarity (measured on ft09 level 4)
            if ring is None:
                return None
            if self._polarity_flips >= 1:
                self._lattice_benched = level
                return None
            self._clue_polarity ^= 1
            self._polarity_flips += 1
        return None

    def _lattice_constraints(
        self, grid: Grid, board: tuple
    ) -> dict[Cell, list[tuple[bool, int]]]:
        """Read probed-static sites as clue markers: their center color is
        a target and the pixel toward each responsive neighbour encodes
        ==target (drawn 0) or !=target, under the current polarity.  Only
        sites whose center is a cycling color qualify — that drops junk
        sites (panel borders) whose centers are background or chrome."""
        p, w, h, ox, oy, sites = board
        cycle = set(self._palette_next) | set(self._palette_next.values())
        cons: dict[Cell, list[tuple[bool, int]]] = defaultdict(list)
        for (i, j), (ax, ay, ck, center) in sites.items():
            if ck in self._click_fx or ck not in self._probed_classes \
                    or center not in cycle:
                continue
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    n = (i + dx, j + dy)
                    cell = sites.get(n)
                    if cell is None or n in self._site_dead \
                            or cell[2] not in self._click_fx:
                        continue
                    px = grid[ay + min(h - 1, (dy + 1) * h // 3)] \
                             [ax + min(w - 1, (dx + 1) * w // 3)]
                    same = (px == 0) == (self._clue_polarity == 0)
                    cons[n].append((same, center))
        return dict(cons)

    def _lattice_violations(
        self, board: tuple, cons: dict[Cell, list[tuple[bool, int]]]
    ) -> tuple[list[Cell], bool]:
        """Cells whose color violates the constraints, plus a broken flag
        when the reading is self-contradictory or needs colors outside the
        learned palette ring — both falsify the current polarity."""
        sites = board[5]
        ring = self._palette_ring()
        viol: list[Cell] = []
        for n in sorted(cons, key=lambda ij: (ij[1], ij[0])):
            color = sites[n][3]
            eq = {t for same, t in cons[n] if same}
            ne = {t for same, t in cons[n] if not same}
            if len(eq) > 1 or (eq & ne):
                return [], True
            if eq:
                target = next(iter(eq))
                if ring is not None and target not in ring:
                    return [], True  # unreachable target
                if color != target:
                    viol.append(n)
            elif color in ne:
                if ring is not None and not (set(ring) - ne):
                    return [], True  # no color can satisfy
                viol.append(n)
        return viol, False

    def _lattice_solve(
        self, board: tuple, cons: dict, viol: list[Cell], level: int
    ) -> Optional[GameAction]:
        """Identity masks: fix the first violated cell — one click cycles
        only that cell, so greedy per-step re-solve is exact, self-healing
        across deaths, and discovers the palette ring as it goes.  Mixed
        boards stay greedy as long as no neighborhood-mask cell can disturb
        another one: fix those first, and the identity fix-ups afterwards
        cannot re-break them.  Truly coupled boards get a bounded BFS over
        the cell-color vector, replayed from self._lattice_plan."""
        sites = board[5]
        resp = [ij for ij in sites
                if sites[ij][2] in self._click_fx
                and ij not in self._site_dead]
        nonid = [ij for ij in resp if self._fx_mask(sites[ij][2]) - {(0, 0)}]
        coupled = any((b[0] - a[0], b[1] - a[1]) in self._fx_mask(sites[a][2])
                      for a in nonid for b in nonid if a != b)
        if not coupled:
            first = [ij for ij in viol if ij in nonid] or viol
            return self._lattice_click(board, first[0], "fix constrained cell")
        if self._lattice_plan:
            ij = self._lattice_plan.popleft()
            if ij in sites:
                return self._lattice_click(board, ij, "planned mask click")
            self._lattice_plan.clear()
            return None
        if self._bfs_tries >= 2:
            self._lattice_benched = level
            return None
        self._bfs_tries += 1
        ring = self._palette_ring()
        plan = (self._lattice_gf2(board, cons, ring)
                if ring is not None and len(ring) == 2 else None)
        if not plan:
            plan = self._lattice_bfs(board, cons)
        if not plan:
            return None
        self._lattice_plan.extend(plan[1:])
        return self._lattice_click(board, plan[0], "planned mask click")

    def _lattice_gf2(
        self, board: tuple, cons: dict, ring: list[int]
    ) -> Optional[list[Cell]]:
        """Exact solve for coupled 2-color boards (ft09 level 6: every cell
        recolors itself AND its upper neighbour).  Click counts are mod-2,
        effects are linear and commute, and a != constraint is just == the
        other color — so the whole board is one GF(2) system, solved by
        Gauss-Jordan elimination with free cells pinned to zero clicks.
        Returns the cells to click once each, or None when no assignment
        exists under the current constraint reading."""
        sites = board[5]
        cells = sorted((ij for ij in sites
                        if sites[ij][2] in self._click_fx
                        and ij not in self._site_dead),
                       key=lambda ij: (ij[1], ij[0]))
        pos = {ij: c for c, ij in enumerate(cells)}
        idx = {color: k for k, color in enumerate(ring)}
        rows: list[int] = []  # bit c = click coefficient; top bit = parity
        for n, cc in cons.items():
            if n not in pos or sites[n][3] not in idx:
                return None
            allowed = {c for c in ring
                       if all((c == t) == same for same, t in cc)}
            if not allowed:
                return None
            if allowed == set(ring):
                continue  # any color satisfies: no equation
            target = next(iter(allowed))
            row = 0
            for ij, c in pos.items():
                if (n[0] - ij[0], n[1] - ij[1]) \
                        in self._fx_mask(sites[ij][2]):
                    row |= 1 << c
            parity = (idx[target] - idx[sites[n][3]]) % 2
            rows.append(row | parity << len(cells))
        pivots: dict[int, int] = {}  # pivot column → reduced row
        for row in rows:
            for col, prow in pivots.items():
                if row >> col & 1:
                    row ^= prow
            col = next((c for c in range(len(cells)) if row >> c & 1), None)
            if col is None:
                if row:
                    return None  # 0 = 1: reading is unsatisfiable
                continue
            for c2 in pivots:
                if pivots[c2] >> col & 1:
                    pivots[c2] ^= row
            pivots[col] = row
        return [ij for c, ij in enumerate(cells)
                if pivots.get(c, 0) >> len(cells) & 1]

    def _lattice_bfs(self, board: tuple, cons: dict) -> list[Cell]:
        """Breadth-first search over the color vector of responsive cells,
        one click per edge, applying each class's learned mask.  Exact but
        budgeted: deep solutions (large all-pattern boards) are out of its
        reach by design — better to bench than to thrash."""
        sites = board[5]
        ring = self._palette_ring()
        if ring is None:
            return []
        idx = {c: k for k, c in enumerate(ring)}
        cells = sorted(
            (ij for ij in sites
             if sites[ij][2] in self._click_fx
             and ij not in self._site_dead),
            key=lambda ij: (ij[1], ij[0]))
        if any(sites[ij][3] not in idx for ij in cells):
            return []
        pos = {ij: k for k, ij in enumerate(cells)}
        masks = [[pos[n] for off in self._fx_mask(sites[ij][2])
                  if (n := (ij[0] + off[0], ij[1] + off[1])) in pos]
                 for ij in cells]
        checks = [(pos[n], cc) for n, cc in cons.items() if n in pos]
        k = len(ring)
        start = tuple(idx[sites[ij][3]] for ij in cells)

        def solved(state: tuple) -> bool:
            return all(all(same == (ring[state[ci]] == t) for same, t in cc)
                       for ci, cc in checks)

        seen = {start}
        dq: deque[tuple[tuple, list[Cell]]] = deque([(start, [])])
        popped = 0
        while dq and popped < LATTICE_BFS_CAP:
            state, path = dq.popleft()
            popped += 1
            if path and solved(state):
                return path
            for ci, ij in enumerate(cells):
                nxt = list(state)
                for t in masks[ci]:
                    nxt[t] = (nxt[t] + 1) % k
                tnxt = tuple(nxt)
                if tnxt not in seen:
                    seen.add(tnxt)
                    dq.append((tnxt, path + [ij]))
        return []

    def _lattice_click(self, board: tuple, ij: Cell, why: str) -> GameAction:
        p, w, h, ox, oy, sites = board
        ax, ay = sites[ij][0], sites[ij][1]
        x, y = ax + w // 2, ay + h // 2
        action = GameAction.ACTION6
        action.set_data({"x": x, "y": y})
        action.reasoning = {"why": f"{why} at lattice {ij}"}
        self._prev_action = f"6:{x},{y}"
        if not self._lattice_engaged:
            self._lattice_engaged = True
            self._deaths_at_engage = self._level_deaths
        return action

    # ── v1 fallback: novelty search ──────────────────────────────────────
    def _note_visit(self, key: int) -> None:
        if self._state_visits[key] == 0:
            self._steps_since_novelty = 0
        else:
            self._steps_since_novelty += 1
        self._state_visits[key] += 1

    def _click_targets(self, grid: Optional[Grid], cap: int = 24) -> list[Cell]:
        # v1's exact color-group targeting — empirically what cracks lp85.
        # Rarest colors first; centroid, plus first pixel when the centroid
        # falls off-color (spread-out color groups).
        if grid is None:
            return [(32, 32)]
        positions: dict[int, list[Cell]] = defaultdict(list)
        counts: Counter[int] = Counter()
        for y, row in enumerate(grid):
            for x, color in enumerate(row):
                positions[color].append((x, y))
                counts[color] += 1
        if not counts:
            return [(32, 32)]
        background = counts.most_common(1)[0][0]
        out: list[Cell] = []
        for color, _n in sorted(counts.items(), key=lambda kv: kv[1]):
            if color == background:
                continue
            pts = positions[color]
            cx = sum(p[0] for p in pts) // len(pts)
            cy = sum(p[1] for p in pts) // len(pts)
            out.append((cx, cy))
            if grid[cy][cx] != color:
                out.append(pts[0])
            if len(out) >= cap:
                break
        # append component centroids (vc33-style wins): same color can form
        # several objects; their individual centers are distinct click targets
        for _color, cells in sorted(components(grid), key=lambda c: len(c[1]))[:cap]:
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            out.append((sum(xs) // len(xs), sum(ys) // len(ys)))
        seen: set[Cell] = set()
        deduped = [t for t in out if not (t in seen or seen.add(t))]
        return deduped or [(32, 32)]

    def _afford_rank(
        self, grid: Grid,
        untried: list[tuple[str, GameAction, Optional[Cell]]],
    ) -> list[tuple[str, GameAction, Optional[Cell]]]:
        """Reorder untried clicks by the affordance library.  Only ever
        called after the first GAME_OVER, so the first life of any fresh
        game is bit-identical to pure enumeration.  STABLE sort on
        descending Laplace P(effect): unknown signatures all score the 0.5
        prior, so ties preserve the rarest-color order that empirically
        cracks lp85; proven-dead signatures (CLICK_PROBES+ tries with
        P(effect) < CLICK_DEAD) drop out — that's what conserves vc33's
        50-click budgets.  Non-click options never move slots."""
        comps = components(grid)
        scored: list[tuple[float, tuple[str, GameAction, Optional[Cell]]]] = []
        for opt in untried:
            if opt[2] is None:
                continue
            changed, tries = self._click_effects.get(
                signature_under(comps, opt[2]), (0, 0))
            score = (changed + 1) / (tries + 2)
            if tries >= CLICK_PROBES and score < CLICK_DEAD:
                continue  # proven dead: stop spending budget on it
            scored.append((score, opt))
        scored.sort(key=lambda so: -so[0])  # stable: ties keep current order
        fill = iter(opt for _score, opt in scored)
        merged: list[tuple[str, GameAction, Optional[Cell]]] = []
        for opt in untried:
            if opt[2] is None:
                merged.append(opt)
            else:
                nxt = next(fill, None)
                if nxt is not None:
                    merged.append(nxt)
        return merged

    def _novelty_policy(
        self, grid: Optional[Grid], latest_frame: FrameData
    ) -> GameAction:
        key = self._masked_hash(grid)
        avail = set(latest_frame.available_actions or [])
        options: list[tuple[str, GameAction, Optional[Cell]]] = []
        for action in GameAction:
            if action is GameAction.RESET:
                continue
            if avail and action.value not in avail:
                continue
            if action.is_complex():
                for x, y in self._click_targets(grid):
                    options.append((f"6:{x},{y}", action, (x, y)))
            else:
                options.append((str(action.value), action, None))
        if not options:
            self._prev_action = "0"
            return GameAction.RESET

        untried = [o for o in options if o[0] not in self._tried[key]]
        if untried and self._game_overs and grid is not None:
            untried = self._afford_rank(grid, untried)
        if untried:
            choice = (self._rng.choice(untried)
                      if self._steps_since_novelty > STUCK_LIMIT else untried[0])
        else:
            def rank(opt: tuple[str, GameAction, Optional[Cell]]) -> float:
                dest = self._transitions.get((key, opt[0]))
                if dest == key:
                    return 1e9
                return -1.0 if dest is None else float(self._state_visits[dest])
            options.sort(key=rank)
            choice = self._rng.choice(options[: max(1, len(options) // 4)])

        action_key, action, coords = choice
        self._tried[key].add(action_key)
        if coords is not None:
            action.set_data({"x": coords[0], "y": coords[1]})
            action.reasoning = {"why": f"explore object at {coords}"}
        else:
            action.reasoning = f"explore action {action.value}"
        self._prev_action = action_key
        return action
