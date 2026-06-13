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
            return GameAction.RESET

        grid = grid_of(latest_frame)
        # may refresh the HUD mask, so learn before keying; the leveled flag
        # stops cross-level repaints from being read as click physics
        self._learn(grid, latest_frame.levels_completed != self._best_levels)
        key = self._masked_hash(grid)
        self._note_visit(key)

        if latest_frame.levels_completed > self._best_levels:
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

        action = self._policy(grid, latest_frame)
        self._prev_grid = grid
        self._prev_key = key
        return action

    # ── learning ─────────────────────────────────────────────────────────
    def _learn(self, grid: Optional[Grid], leveled: bool = False) -> None:
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
        groups: dict[Cell, list[tuple[int, frozenset[Cell]]]] = defaultdict(list)
        for color, delta, cells in moved_objects(self._prev_grid, grid):
            groups[delta].append((color, cells))
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
            if xs[-1] - xs[0] < GROUP_BBOX and ys[-1] - ys[0] < GROUP_BBOX:
                ax, ay = min(union)
                self._move_votes[act][delta] += 1
                self._avatar_sigs[frozenset(
                    (color, x - ax, y - ay)
                    for color, cells in movers for x, y in cells)] += 1
                for color, _cells in movers:
                    self._avatar_votes[color] += 1
                if self._world_changed_beyond(union, delta, grid):
                    self._soften_walls()
                return
        if act in self._movement_rules():
            # trusted movement action, avatar still at the same anchor →
            # wall ahead.  Anchor test, not frame equality: ls20's blocked
            # moves still flash failure pixels and tick the masked pip strip.
            prev_av = self._find_avatar(self._prev_grid)
            cur_av = self._find_avatar(grid)
            if prev_av and cur_av and prev_av[0] == cur_av[0]:
                dx, dy = self._movement_rules()[act]
                for x, y in prev_av[1]:
                    cell = (x + dx, y + dy)
                    self._walls.add(cell)
                    self._wall_bounces[cell] += 1

    def _world_changed_beyond(
        self, moved: set[Cell], delta: Cell, grid: Grid
    ) -> bool:
        """True if unmasked cells outside the moved group's footprint changed."""
        dx, dy = delta
        footprint = moved | {(x - dx, y - dy) for x, y in moved}
        for y, (prow, row) in enumerate(zip(self._prev_grid, grid)):
            if prow == row:
                continue
            for x, (pv, v) in enumerate(zip(prow, row)):
                if pv != v and (x, y) not in footprint \
                        and (x, y) not in self._hud_mask:
                    return True
        return False

    def _soften_walls(self) -> None:
        """The world changed beyond the avatar (ls20: a shifter tile cycled
        the carried glyph) — conditional walls may no longer hold.  Keep only
        walls confirmed WALL_TRUST+ times; the rest get re-tested, so a goal
        pad that bounced us under the wrong glyph is never blacklisted."""
        self._walls = {c for c in self._walls
                       if self._wall_bounces[c] >= WALL_TRUST}

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
