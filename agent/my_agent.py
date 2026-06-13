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
 11. SWITCH    — movement games with clicks but no selected piece (dc22
                 family) get a remote-toggle model: a click signature whose
                 effect is a consistent repaint AWAY from the click site,
                 cycling through a closed set of pixel patches, is a switch;
                 footprint signatures the avatar entered (or bounced off)
                 give signature-level passability, and a BFS over the
                 (anchor x switch-phase) product graph routes "toggle the
                 bridge in, then walk it" plans toward exact overlap with
                 the rarest-color compact component.  A click that floods
                 the frame mid-animation, restores the board exactly and
                 drains the budget strip is a fall trap: banned per context
                 and never counted as click deadness.  Gated on clicks +
                 trusted movement rules + no register + no port color, so
                 click-only games, ls20/tr87 and cn04 never see this branch.
 12. FASTPATH  — efficiency layer (the metric is quadratic in actions):
                 novelty warmup ends early for click games once the
                 affordance library holds a confidently-productive
                 signature (the click analog of trusted movement rules),
                 and a level-up that lands on an already-seen start frame
                 (a known level signature) skips warmup outright; every
                 completed level's exact action sequence is recorded,
                 keyed by (level index, masked start-frame hash), and if
                 a reset ever rewinds completed progress (some games
                 punish game-over that way) the recorded win replays
                 step by step while each frame's masked hash still
                 matches the recording, aborting to normal policy on the
                 first mismatch.
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
GX_RESET_CAP = 4        # graph-explorer RESET-backtracks per level before it
                        # defers to novelty (analog of LATTICE_DEATH_CAP:
                        # bounds wasted opening re-walks; tuned on HELD-18)
GX_LETHAL_HITS = 2      # deaths on the SAME click-class before the graph
                        # explorer bans it (avoids one-off coincident-hazard
                        # false positives, and gives the affordance library a
                        # chance to record the class's effect rate so the
                        # productivity guard can spare a real control before the
                        # ban fires; HITS=1 measured worse: 0.0296 vs 0.0391 —
                        # premature bans on un-probed productive classes;
                        # appearance physics, persists)
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
CLICK_WARM_TRIES = 5    # tries a click signature needs before it may end
                        # novelty warmup early — the affordance library
                        # knowing a productive class is the click-game
                        # analog of trusted movement rules
CLICK_WARM_P = 0.8      # Laplace P(effect) above which it counts
CLICK_WARM_FRAMES = 100  # diffed frames the lattice/attr detectors get
                        # to fire before the early exit may trigger
SW_FRAMES = 30          # diffed frames before the switch planner may engage
                        # (cn04's port pre-vote always wins this race)
SW_CYCLE_CAP = 8        # longest phase cycle a switch may close
SW_PROBES = 1           # policy clicks probing one unconfirmed signature —
                        # a real switch reveals itself on the first press
                        # (and then gets SW_CYCLE_CAP tries to close its
                        # cycle); inert tiles waste only one click each
SW_BLOCK_VOTES = 2      # bounces before a footprint signature reads blocked
SW_FALL_DROP = 5        # border-band cells that must drain alongside an
                        # exact board restore to read a fall trap
SW_CLICK_COST = 2       # budget units a click edge spends (dc22 charges a
                        # button press twice; moves cost one)
SW_BFS_CAP = 6000       # (anchor x phase-vector) product-graph state budget
SW_STRIKES = 10         # model surprises before the switch planner benches
                        # the level (novelty search takes over)
SL_ACCENT_MAX = 4       # max pixels a node-maze avatar accent color may span
                        # for the slide observer to track it (a unique tiny
                        # marker, not a sprawling object)
SL_STRIKES = 6          # dry/failed slide plans before the maze head benches
                        # the level (novelty search takes over)
SL_PROBE_CAP = 1        # directional probes per action before the slide head
                        # trusts (or rejects) that action's unit step
OV_MIN_ANCHORS = 2      # hollow goal-overlay boxes a frame needs before the
                        # overlay-align head trusts the static-goal signature
                        # (re86: a single box is too weak to distinguish from
                        # an incidental ring; two fixed boxes are the family)
OV_STRIKES = 8          # dry/failed overlay plans before the head benches the
                        # level (mirrors PC_STRIKES — novelty takes over)
SORT_MIN_TARGETS = 3    # equal-size hollow boxes in a horizontal run before the
                        # sequence-match head trusts a target row (a unique,
                        # low-collision signature: 3+ distinct-border boxes)
SORT_STRIKES = 6        # contradicted placements before the sort head benches
                        # the level (mirrors SL/PC — novelty takes over)
HERD_SEL = 8            # attractor selection radius (px): a click grabs movers
                        # whose centre lies within this radius and pulls them
                        # toward the click.  Step toward the goal stays inside
                        # this radius so the mover is re-grabbed every click.
HERD_STRIKES = 24       # consecutive clicks with no board progress before the
                        # herd head benches the level (all candidates inert /
                        # geometry the head can't read) — novelty takes over

# ABLATION TOGGLE: CURIO_GENERIC_ONLY=1 disables the five family-specific
# modules (lattice/GF2, editor, attribute-state, port-align, switch) and
# their gates, leaving ONLY the generic core — object perception, movement-
# rule voting, BFS routing, novelty exploration, HUD masking, affordance.
# Unset (default) is bit-identical to the full agent.  This isolates how
# much of the score is the family heads vs. the generic substrate.
GENERIC_ONLY = os.environ.get("CURIO_GENERIC_ONLY", "") == "1"

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


def overlay_anchors(
    grid: Grid, outline_color: int
) -> dict[int, list[Cell]]:
    """Static goal-overlay anchors: hollow 3x3 boxes whose 8-neighbourhood is
    all `outline_color` and whose centre is a single other (non-background)
    pixel.  Returns {centre_colour: [centre cells]}.  Pure read — the
    overlay-align head must cover each centre with a same-colour piece pixel
    (re86's win check ignores the outline ring, matching only the centres)."""
    counts: Counter[int] = Counter()
    for row in grid:
        counts.update(row)
    bg = counts.most_common(1)[0][0]
    out: dict[int, list[Cell]] = defaultdict(list)
    for y in range(1, GRID - 1):
        row = grid[y]
        for x in range(1, GRID - 1):
            c = row[x]
            if c == outline_color or c == bg:
                continue
            if all(grid[y + dy][x + dx] == outline_color
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if (dx, dy) != (0, 0)):
                out[c].append((x, y))
    return {k: v for k, v in out.items()}


def piece_footprint(
    grid: Grid, center: Cell, arm_color: int
) -> frozenset[Cell]:
    """The 8-connected blob of the selected piece: cells equal to the piece's
    arm colour OR the selection hole (0), flood-filled from its centre.  The
    selected diagonal-X renders its centre as -1/0 intermittently, so the
    flood admits BOTH the arm colour and 0 to keep the blob whole."""
    cx, cy = center
    seen = {center}
    dq = [center]
    fp: list[Cell] = []
    while dq:
        x, y = dq.pop()
        fp.append((x, y))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID and 0 <= ny < GRID \
                        and (nx, ny) not in seen \
                        and grid[ny][nx] in (0, arm_color):
                    seen.add((nx, ny))
                    dq.append((nx, ny))
    return frozenset(fp)


def cover_centers(
    footprint: frozenset[Cell], center: Cell, anchors: list[Cell],
    step: int, cur: Cell,
) -> Optional[Cell]:
    """The step-grid centre position at which the piece footprint covers ALL
    given anchor centres, nearest to the current centre.  `footprint` and
    `center` describe the piece as currently rendered; translations are
    restricted to multiples of `step` (the learned move magnitude) so the
    result is exactly reachable by the arrow rules.  None when no translation
    on the step lattice covers every anchor (a non-axis shape, or anchors the
    arms cannot reach)."""
    if not anchors:
        return None
    cx, cy = center
    rel = frozenset((px - cx, py - cy) for px, py in footprint)
    aset = set(anchors)
    best: Optional[tuple[int, Cell]] = None
    for tdy in range(-(GRID - 1), GRID, step):
        for tdx in range(-(GRID - 1), GRID, step):
            ncx, ncy = cx + tdx, cy + tdy
            cells = {(ncx + rx, ncy + ry) for rx, ry in rel}
            if aset <= cells:
                cost = abs(ncx - cur[0]) + abs(ncy - cur[1])
                if best is None or cost < best[0]:
                    best = (cost, (ncx, ncy))
    return best[1] if best else None


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
        # ── graph explorer (extends the novelty fallback; opt-in) ──
        # CURIO_EXPLORER=graph swaps _novelty_policy for a salience-tiered
        # global state-graph explorer (BFS frontier + RESET backtracking).
        # Read once; with the var unset every hook below is dead code and
        # the executed path is byte-identical to v7 (FIT floor preserved).
        self._gx_on = os.environ.get("CURIO_EXPLORER", "") == "graph"
        if self._gx_on:
            # node record per masked-frame hash: built once per first visit,
            # idempotent within a level (geography is stable within a level)
            self._gx_nodes: dict[int, dict[str, Any]] = {}
            self._gx_route: deque[str] = deque()      # cached BFS action-keys
            self._gx_route_dest: Optional[int] = None  # the route's target node
            self._gx_start: Optional[int] = None       # level-start node hash
            self._gx_resets = 0                        # RESET-backtracks spent
            # (state, action-key) the explorer just issued as a RESET-inducing
            # action; confirmed and turned into a start edge on the next start
            self._gx_pending_reset: Optional[tuple[int, str]] = None
            # appearance signatures of clicks exhausted ANYWHERE (cross-state
            # pruning); appearance is physics, so this persists across levels
            self._gx_global_tried_sig: set[int] = set()
            # LETHAL-CLICK MEMORY (win-speed lever): appearance signatures of
            # sprites whose click was the last action before a GAME_OVER.  A
            # death restarts the CURRENT level (confirmed empirically — these
            # games never rewind to level 0), so every repeated lethal click
            # inflates that level's action count, and the squared-efficiency
            # metric punishes it quadratically.  Deadliness is appearance
            # physics, so excluding the class everywhere (like _click_dead)
            # stops the death-loop churn that bleeds the per-level score.
            # Requires GX_LETHAL_HITS deaths on the SAME class before banning
            # (a one-off death can be a coincident hazard elsewhere on the
            # board, not the clicked sprite).
            self._gx_lethal_hits: Counter[int] = Counter()
            self._gx_lethal_sig: set[int] = set()
        # ── v2 world model ──
        self._prev_grid: Optional[Grid] = None
        self._prev_key: Optional[int] = None
        self._prev_action: Optional[str] = None
        self._avail: set[int] = set()  # latest frame's available actions
        self._move_votes: dict[int, Counter[Cell]] = defaultdict(Counter)
        # SLIDE/NODE-MAZE (tu93 family) perception: NET displacement of the
        # minority-pixel avatar, voted PER ACTION in parallel to _move_votes.
        # The multi-group gate (line ~945) records nothing here — a node-maze
        # avatar (color body + accent pixel) and the corridor repaint behind
        # it move as >=2 deltas, so _move_votes never trusts a direction.  This
        # store is a STRICTLY ADDITIVE observer: it is read only by the slide
        # head, never by _movement_rules / _walls, so it cannot perturb any
        # floor game.  A zero-displacement-but-budget-spent result records a
        # blocked edge.
        self._slide_votes: dict[int, Counter[Cell]] = defaultdict(Counter)
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
        self._attr_seen: set[tuple] = set()            # per-level geography
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
        # ── overlay-align head (re86 family): selected piece placed to cover
        # a STATIC goal-overlay's anchor centres.  The outline colour and the
        # SELECT verb are physics (same on every level); the per-colour anchor
        # sets, solved-colour set and failure counters are per-level geography.
        self._ov_outline_votes: Counter[int] = Counter()  # physics
        self._ov_deltas: dict[int, Counter[Cell]] = defaultdict(Counter)  # phys
        self._ov_select: Optional[int] = None       # the cycle-selection verb
        self._ov_idprobe = 0        # arrow-probe cursor (learn the 4 deltas)
        self._ov_level = -1
        self._ov_solved: set[int] = set()           # colours already covered
        self._ov_plan: deque[int] = deque()         # queued MOVE action ids
        self._ov_target: Optional[Cell] = None      # centre we are driving to
        self._ov_strikes = 0
        self._ov_benched: Optional[int] = None
        self._ov_last_center: Optional[Cell] = None  # progress watchdog
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
        # ── switch model (dc22 family): remote cyclic toggles ──
        # footprint passability (_stepped_sigs / _block_sigs) is appearance-
        # keyed physics; a switch's mask and phase patches are positional —
        # per-level geography — as are the fall-trap bans and probe spend
        self._sw_recs: dict[int, dict] = {}        # click sig → toggle record
        self._sw_probe: Counter[int] = Counter()   # probe clicks per sig
        self._sw_carry: Counter[tuple] = Counter()  # carry probes per (sig, anchor)
        self._sw_belief: dict[int, tuple] = {}  # sig → (phase idx, direction)
        self._block_sigs: Counter[tuple] = Counter()  # physics: bounced
        self._fall_bans: set[tuple] = set()  # (sig, support sig, anchor)
        self._sw_plan: deque[tuple[str, Cell, tuple]] = deque()
        self._sw_strikes = 0
        self._sw_benched: Optional[int] = None
        self._sw_nogoal: Optional[tuple] = None  # (key, phases) of a dry BFS
        # ── slide / node-maze head (tu93 family) ──
        # A directional node-maze: each ACTION steps the avatar one corridor
        # node toward that direction (blocked moves spend budget, no motion).
        # The board is a regular binary corridor/wall lattice; the exit is a
        # rare-color component.  All state here is per-level GEOGRAPHY (the
        # board, exit and probe spend are level data) except the engaged flag
        # and the directional probe map, which are cheap to rebuild.  Strikes
        # bench the head for the level on repeated dry plans (PC/SW pattern).
        self._sl_engaged = False           # sticky once a maze is confirmed
        self._sl_probe: Counter[int] = Counter()  # directional probe spend
        self._sl_dirmap: dict[int, Cell] = {}     # act -> unit step (probed)
        self._sl_corridor: Optional[int] = None   # learned passable color
        self._sl_strikes = 0
        self._sl_benched: Optional[int] = None
        # ── sequence-match assignment head (sb26 family) ──
        # A target ROW of hollow boxes spells an ordered colour sequence; a
        # palette of solid tiles (same colour-multiset) must be dragged into
        # empty slot markers inside one or more nested containers.  The win
        # walk descends doors (a door's interior colour == the border colour
        # of the container it opens), so the slot fill order is a DFS over the
        # container tree, and target[i] -> dfs_slot[i] is a positional
        # bijection.  Drags are two ACTION6 clicks (token, slot); the final
        # slot is filled LAST so the verification auto-fires.  All state is
        # per-level GEOGRAPHY (the board is level data): the in-flight plan of
        # (token_colour, slot_cell) pairs and the placement cursor, plus a
        # strike/bench self-disable on a contradicted (no-change) placement.
        # plan entries: (token colour, token click cell, slot cell)
        self._sort_plan: list[tuple[int, Cell, Cell]] = []
        self._sort_idx = 0          # next pair in _sort_plan to emit
        self._sort_phase = 0        # 0 = click token, 1 = click slot
        self._sort_level = -1       # level the current plan was built for
        self._sort_prev_sig: Optional[int] = None  # board hash before a place
        self._sort_strikes = 0
        self._sort_benched: Optional[int] = None
        # ── attractor-herd model (su15 family): click acts as a black hole
        # that grabs movable blobs within HERD_SEL and pulls them toward the
        # click; win = herd the right blob(s) into a static goal zone.  No
        # board state is needed beyond a per-level "inert" memory (centres the
        # head has clicked toward without the blob moving — decorations of the
        # same shape as a mover) and a no-progress strike counter.  All state
        # is read off pixels via _learn's _prev_grid, so this head perturbs no
        # floor game (gate is the exclusive {6,7} action set). ──
        self._herd_inert: set[Cell] = set()   # centres proven non-responsive
        self._herd_last: Optional[Cell] = None  # candidate centre last aimed
        self._herd_strikes = 0
        self._herd_benched: Optional[int] = None
        self._herd_level = -1
        # ── win-path replay + level-start signatures (efficiency) ──
        # paths are keyed (level index, masked start-frame hash): replay
        # only ever fires when the SAME level restarts from the SAME start
        # frame after a level was already completed — engines that only
        # reset the current level never re-enter a completed level, so
        # this is strictly insurance for games whose game-over rewinds
        # completed progress, and dead weight (a dict probe per level
        # start) everywhere else
        self._wp_paths: dict[tuple[int, int], list[tuple[int, str]]] = {}
        self._wp_log: list[tuple[int, str]] = []
        self._wp_start = 0          # masked hash of the level-start frame
        self._wp_level = 0          # level index the recording started on
        self._wp_pending_start = True
        self._wp_replay: Optional[list[tuple[int, str]]] = None
        self._wp_idx = 0
        self._seen_starts: set[int] = set()  # masked level-start hashes
        self._warmup_skip = False   # level-up landed on a known start frame
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
                if self._gx_on and self._prev_key is not None \
                        and self._prev_action is not None \
                        and self._gx_pending_reset is None:
                    # CORRECT-RESET DETECTION: a death IS a reset-inducing
                    # event — the action that caused it teleports to the level
                    # start.  Stash it; the start edge is recorded on the next
                    # start frame (GAME_OVER destroys the diff, so _learn can't
                    # see it).  Guarded so a non-graph run is untouched.
                    self._gx_pending_reset = (
                        self._prev_key, self._prev_action)
                    # LETHAL-CLICK MEMORY: if the fatal action was a click,
                    # tally a death against the clicked sprite's appearance
                    # class (read off the pre-death grid, still in _prev_grid).
                    # GX_LETHAL_HITS deaths on the same class ban it — the
                    # explorer then never re-clicks that hazard, killing the
                    # death-loop that inflates the level's action count.
                    if self._prev_action.startswith("6:"):
                        try:
                            lx, ly = map(int, self._prev_action[2:].split(","))
                            lsig = signature_under(
                                components(self._prev_grid), (lx, ly))
                        except ValueError:
                            lsig = 0
                        if lsig:
                            self._gx_lethal_hits[lsig] += 1
                            # PRODUCTIVITY GUARD: a class that reliably CHANGES
                            # the world (high Laplace effect rate) is a
                            # context-dependent control, not a pure hazard —
                            # banning it globally throws away the only path to
                            # the win (measured on sc25).  Per-state _tried
                            # already stops re-clicking it from the exact death
                            # state; only ban classes that are mostly inert
                            # except for killing.
                            ch, tr = self._click_effects.get(lsig, (0, 0))
                            productive = tr > 0 and (ch + 1) / (tr + 2) > 0.5
                            if self._gx_lethal_hits[lsig] >= GX_LETHAL_HITS \
                                    and not productive:
                                self._gx_lethal_sig.add(lsig)
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
            # the overlay level replays deterministically: void the in-flight
            # plan/target and solved-colour progress, give a benched run a
            # fresh life (strikes die with the budget that earned them).  The
            # outline colour and SELECT verb are physics — kept.
            self._ov_solved.clear()
            self._ov_plan.clear()
            self._ov_target = None
            self._ov_strikes = 0
            self._ov_benched = None
            self._ov_last_center = None
            self._ov_idprobe = 0
            # the editor's level replays deterministically too: keep the
            # mutate boxes and goal candidates, give a benched run a fresh
            # life (strikes die with the budget that earned them)
            self._ed_strikes = 0
            self._ed_benched = None
            self._ed_spent = 0
            self._ed_miss = 0
            self._ed_full_seen = False
            # the switch level replays deterministically: keep the toggle
            # records (masks repeat), void the plan and the phase beliefs
            # (the board rewinds to its start phases), refresh the strikes
            self._sw_plan.clear()
            self._sw_belief.clear()
            self._sw_strikes = 0
            self._sw_benched = None
            self._sw_nogoal = None
            # the slide maze level replays deterministically: refresh the
            # strikes/bench (board rewinds to its start), keep the engaged
            # flag and direction map (physics).  The head re-plans every step
            # so there is no standing plan to void.
            self._sl_strikes = 0
            self._sl_benched = None
            # the sort level replays deterministically: void the in-flight
            # placement plan (the board rewinds to its empty start), refresh
            # the strikes/bench so a dead life's strikes don't carry over.
            self._sort_plan = []
            self._sort_idx = 0
            self._sort_phase = 0
            self._sort_level = -1
            self._sort_prev_sig = None
            self._sort_strikes = 0
            self._sort_benched = None
            # the herd level replays deterministically: void the inert memory
            # and aim (the board rewinds to its start layout) and refresh the
            # strikes so a dead life's strikes don't carry over.  The bench is
            # KEYED BY LEVEL, so it survives a same-level replay (a level the
            # head genuinely can't read should stay benched).
            self._herd_inert = set()
            self._herd_last = None
            self._herd_strikes = 0
            # the dead life's win-path recording is garbage; the frame
            # after RESET is a level start (the engine replays the level)
            self._wp_log.clear()
            self._wp_replay = None
            self._wp_pending_start = True
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

        if latest_frame.levels_completed > self._wp_level:
            # WIN-PATH: this life's recording just completed the level it
            # started on — bank it for replay should a reset ever rewind
            # here.  Keyed off _wp_level, not _best_levels: a re-completion
            # after a progress-rewinding reset re-banks under the CURRENT
            # (frozen) HUD mask, so paths recorded before the mask froze
            # heal themselves the first time they are walked again.
            if self._wp_log:
                self._wp_paths[(self._wp_level, self._wp_start)] = \
                    list(self._wp_log)
            if key in self._seen_starts:
                # PLAN-FIRST RESUMPTION: the next level starts on a frame
                # already seen as a level start (a known level signature) —
                # novelty warmup has nothing left to teach here
                self._warmup_skip = True
            self._wp_pending_start = True
        elif latest_frame.levels_completed < self._wp_level:
            # progress rewound WITHOUT passing through the GAME_OVER
            # branch: the running recording straddles a reset — void it
            self._wp_log.clear()
            self._wp_replay = None
            self._wp_pending_start = True

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
            if self._gx_on:
                # graph geography has the same per-level lifetime as _tried;
                # appearance-keyed _gx_global_tried_sig persists (it's physics)
                self._gx_nodes.clear()
                self._gx_route.clear()
                self._gx_route_dest = None
                self._gx_resets = 0
                self._gx_start = None
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
            self._attr_seen.clear()
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
            # overlay geography: the new level has its own anchor centres and
            # piece roster (outline colour and SELECT verb are physics).
            self._ov_solved.clear()
            self._ov_plan.clear()
            self._ov_target = None
            self._ov_strikes = 0
            self._ov_benched = None
            self._ov_last_center = None
            self._ov_idprobe = 0
            # switch geography: masks, phase patches, fall bans and probe
            # spend are positional level data (footprint passability — the
            # _stepped/_block signature stores — is appearance physics)
            self._sw_recs.clear()
            self._sw_probe.clear()
            self._sw_carry.clear()
            self._sw_belief.clear()
            self._fall_bans.clear()
            self._sw_plan.clear()
            self._sw_strikes = 0
            self._sw_benched = None
            self._sw_nogoal = None
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
            # slide-maze geography: the board and exit are new; the probe
            # spend, corridor colour, strikes and bench are level data
            # (engaged flag and the learned direction map persist as physics)
            self._sl_probe.clear()
            self._sl_corridor = None
            self._sl_strikes = 0
            self._sl_benched = None
            # sort geography: the new level has its own target row, palette
            # and container tree — drop the plan, reset the placement cursor
            # and refresh the strikes/bench (nothing here is physics).
            self._sort_plan = []
            self._sort_idx = 0
            self._sort_phase = 0
            self._sort_level = -1
            self._sort_prev_sig = None
            self._sort_strikes = 0
            self._sort_benched = None

        if self._wp_pending_start:
            # first frame of a level (game start, level-up, or post-death
            # restart of the same level): key the new recording on it, and
            # arm replay when this exact (level, start frame) was already
            # completed once — the slow discovered win repeats fast
            self._wp_pending_start = False
            self._wp_start = key
            self._wp_level = latest_frame.levels_completed
            self._wp_log = []
            self._seen_starts.add(key)
            self._wp_replay = self._wp_paths.get((self._wp_level, key))
            self._wp_idx = 0
            if self._gx_on:
                # canonical level-start node: RESET/death edges point here so
                # the explorer can deliberately backtrack to it (this frame is
                # the first after game start, level-up, or a respawn/RESET)
                self._gx_start = key
                if self._gx_pending_reset is not None:
                    # the action that triggered this respawn was a confirmed
                    # reset-inducing action: record its edge to the start and
                    # mark it tried so the explorer never re-issues it as new
                    rkey, rstate = self._gx_pending_reset
                    self._transitions[(rstate, rkey)] = key
                    self._tried[rstate].add(rkey)
                    self._gx_pending_reset = None

        action: Optional[GameAction] = None
        if self._wp_replay is not None:
            if self._wp_idx < len(self._wp_replay) \
                    and self._wp_replay[self._wp_idx][0] == key:
                # the frame still matches the recording: repeat its step
                action = self._emit_key(self._wp_replay[self._wp_idx][1])
                self._wp_idx += 1
            else:
                # divergence (or exhaustion without a level-up): back to
                # normal policy — and drop the per-frame model state the
                # policies could not keep in sync while replay was driving
                self._wp_replay = None
                self._plan.clear()
                self._lattice_plan.clear()
                self._sw_plan.clear()
                self._pc_pieces.clear()
                self._pc_sel = None
                self._pc_solution = None
        if action is None:
            action = self._policy(grid, latest_frame)
        if self._prev_action == "0":
            # an in-level RESET restarts the level: the recording is void
            self._wp_log.clear()
            self._wp_replay = None
            self._wp_pending_start = True
        elif self._prev_action is not None:
            self._wp_log.append((key, self._prev_action))
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
                if frames and any(flood_color(f) != flood_color(grid)
                                  for f in frames[:-1]) \
                        and self._sw_band_drop(self._prev_grid, grid) \
                        > SW_FALL_DROP \
                        and self._sw_same_inside(self._prev_grid, grid):
                    # FALL TRAP (dc22): the click toggled the floor from
                    # under the avatar — the engine flooded the frame with
                    # a fall animation, restored the board exactly and
                    # drained the budget strip.  Ban the context, and count
                    # the event as an EFFECT: the exact restore would
                    # otherwise read as "no effect" and poison the very
                    # button the planner needs toward CLICK_DEAD.
                    av = self._find_avatar(self._prev_grid)
                    if av is not None:
                        self._fall_bans.add(
                            (sig, self._tile_sig(self._prev_grid, av[1]),
                             av[0]))
                    eff = self._click_effects.setdefault(sig, [0, 0])
                    eff[0] += 1
                    eff[1] += 1
                    self._sw_plan.clear()
                    self._plan.clear()
                    return
                eff = self._click_effects.setdefault(sig, [0, 0])
                if not self._same_unmasked(self._prev_grid, grid):
                    eff[0] += 1
                eff[1] += 1
                self._learn_click_fx(grid, (x, y), comps_prev, leveled)
                if not leveled:
                    # SWITCH: record what this click repainted away from
                    # the click site (appended after the lattice path —
                    # ft09's machinery above is untouched)
                    self._learn_switch(grid, (x, y), comps_prev, sig)
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
            # SLIDE/NODE-MAZE perception (tu93 family): the avatar carries a
            # unique accent pixel (a single-pixel minority component); track
            # its NET displacement from the pre-action grid to the settled
            # frame and vote it per action.  This is a pure additive observer
            # — it writes ONLY _slide_votes, never _move_votes / rules / walls,
            # so it cannot perturb any other family.  It runs even when the
            # multi-group gate below records nothing (the corridor repaint
            # behind a moving avatar reads as a second mover).  Zero-but-spent
            # records a blocked edge (the budget ticked, the avatar held).
            pa = self._sl_avatar_anchor(self._prev_grid)
            ca = self._sl_avatar_anchor(grid)
            if pa is not None and ca is not None and pa[1] == ca[1]:
                self._slide_votes[act][(ca[0][0] - pa[0][0],
                                        ca[0][1] - pa[0][1])] += 1
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
                    self._soften_walls(grid, extra)
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
                    self._soften_walls(grid, extra)
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
                dest = {(x + dx, y + dy) for x, y in prev_av[1]}
                if all(0 <= cx < GRID and 0 <= cy < GRID for cx, cy in dest):
                    # signature-level passability (the switch planner's
                    # phase-aware complement of per-cell _walls — which
                    # keep feeding the legacy planners untouched)
                    self._block_sigs[self._tile_sig(grid, dest)] += 1
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

    def _soften_walls(
        self, grid: Optional[Grid] = None,
        extra: Optional[list[Cell]] = None,
    ) -> None:
        """The world changed beyond the avatar (ls20: a shifter tile cycled
        the carried glyph) — conditional walls may no longer hold.  Keep only
        walls confirmed WALL_TRUST+ times; the rest get re-tested, so a goal
        pad that bounced us under the wrong glyph is never blacklisted.

        Gated on the change plausibly OPENING something: a repaint outside
        the attribute register (a pad unlock), or a register value never
        carried before.  Routine register cycling used to re-soften on
        EVERY shifter step, and the planner re-bounced the same unconfirmed
        walls dozens of times per life."""
        if grid is not None and extra is not None:
            reg = self._attr_register()
            if reg is not None:
                attr = self._attr_state(grid, reg)
                novel = attr not in self._attr_seen
                self._attr_seen.add(attr)
                if not novel and all(cell in reg for cell in extra):
                    return
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
                elif len(px) * 2 <= len(cells) \
                        and any(len(b) > 1 for b in blobs):
                    # a port marker is a CONTIGUOUS accent blob; a color
                    # scattered as lone pixels (dc22's checkered tiles)
                    # is texture and must not fake a port vote
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
        by_dir: dict[tuple[int, Cell], list[tuple[int, int]]] = {}
        for a, d in self._movement_rules().items():
            s = abs(d[0]) + abs(d[1])
            if s >= 2 and (d[0] == 0 or d[1] == 0):
                by_dir.setdefault((s, d), []).append(
                    (self._move_votes[a][d], a))
        cand: dict[int, dict[Cell, int]] = {}
        for (s, d), claims in by_dir.items():
            claims.sort(reverse=True)
            if len(claims) > 1 and claims[1][0] * 2 > claims[0][0]:
                # two actions claim one direction and the vote hasn't
                # separated yet (the transform verb's cycle diffs fake a
                # few translation votes early): engaging now would hand
                # the REAL mover to the xform probe and desync the model —
                # wait the few frames it takes the true mover to pull away
                continue
            cand.setdefault(s, {})[d] = claims[0][1]
        full = [(sum(self._move_votes[a][d] for d, a in dirs.items()),
                 s, dirs) for s, dirs in cand.items() if len(dirs) == 4]
        if not full:
            # PRE-VOTE: port-color evidence is read off static frames, so
            # it doesn't need full 2D control — vote while the rules are
            # still trusting, and the planner engages the moment the
            # fourth rule lands instead of wandering PORT_VOTES more frames
            if cand and self._port_color() is None:
                self._vote_port_color(grid, max(cand))
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

    # ── overlay-align head (re86 family): cover static goal anchors ────────
    def _ov_outline(self, grid: Grid) -> Optional[int]:
        """The static goal-overlay's outline colour: the colour that forms
        the MOST hollow 3x3 boxes (a non-background colour ringing single
        centre pixels).  None when fewer than OV_MIN_ANCHORS boxes of any one
        colour exist — the family signature is absent (a plain movement game
        has no such overlay)."""
        counts: Counter[int] = Counter()
        for row in grid:
            counts.update(row)
        bg = counts.most_common(1)[0][0]
        # candidate outline colours: each forms boxes whose ring is that
        # colour.  Count boxes per outline colour by trial.
        box_count: Counter[int] = Counter()
        for y in range(1, GRID - 1):
            for x in range(1, GRID - 1):
                c = grid[y][x]
                if c == bg:
                    continue
                ring = grid[y - 1][x - 1]
                if ring == bg or ring == c:
                    continue
                if all(grid[y + dy][x + dx] == ring
                       for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                       if (dx, dy) != (0, 0)):
                    box_count[ring] += 1
        if not box_count:
            return None
        color, n = box_count.most_common(1)[0]
        return color if n >= OV_MIN_ANCHORS else None

    def _ov_select_verb(
        self, avail: set[int], rules: dict[int, Cell]
    ) -> Optional[int]:
        """The cycle-selection verb: the available simple action that is NOT a
        movement rule (re86: ACTION5 jumps the selection hole to another
        piece, translating nothing — so it never earns a move rule)."""
        cands = [a for a in sorted(avail)
                 if a not in rules and not GameAction.from_id(a).is_complex()
                 and a != GameAction.RESET.value]
        return cands[0] if len(cands) == 1 else None

    def _ov_selected(
        self, grid: Grid, outline: int
    ) -> Optional[tuple[Cell, int]]:
        """The selected piece: (centre cell, arm colour).  The centre is the
        unique colour-0 selection hole; the arm colour is the most common
        colour among the cells NEAR the hole, excluding the background, the
        outline ring and 0 itself (the arms radiate from the centre, so the
        nearest non-background non-outline colour is the piece).  None when no
        single 0 hole is present (a transient animation frame)."""
        holes = [(x, y) for y in range(GRID) for x in range(GRID)
                 if grid[y][x] == 0]
        if len(holes) != 1:
            return None
        cx, cy = holes[0]
        counts: Counter[int] = Counter()
        for row in grid:
            counts.update(row)
        bg = counts.most_common(1)[0][0]
        cnt: Counter[int] = Counter()
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                x, y = cx + dx, cy + dy
                if 0 <= x < GRID and 0 <= y < GRID:
                    v = grid[y][x]
                    if v >= 0 and v not in (0, bg, outline):
                        cnt[v] += 1
        if not cnt:
            return None
        arm = cnt.most_common(1)[0][0]
        return ((cx, cy), arm)

    def _overlay_policy(
        self, grid: Optional[Grid], latest_frame: FrameData
    ) -> Optional[GameAction]:
        """Overlay-align policy, gated on: the [1,2,3,4,5]-no-ACTION6 action
        set + a static hollow-box goal overlay (OV_MIN_ANCHORS+ boxes) + a
        single colour-0 selection hole.  Once engaged it probes ACTION1..4 to
        learn the four axis MOVE deltas from the HOLE's own displacement (the
        arms span the whole frame, so the generic GROUP_BBOX-capped vote never
        trusts them), at one magnitude >= 2; the remaining simple action is the
        SELECT verb that jumps the hole to another piece.  g50t shares the
        action set but has a scrolling obstacle and no box overlay, so the
        secondary gate keeps the head inert there.  Drives the selected piece's
        arm footprint onto its same-colour anchor centres, then cycles SELECT
        to the next colour; benches on repeated dry/blocked plans."""
        if grid is None:
            return None
        avail = set(latest_frame.available_actions or [])
        # action-set gate: re86's exact signature (move quartet + SELECT, no
        # click).  Necessary but NOT sufficient (g50t collides) — the overlay
        # signature below is the load-bearing discriminator.
        if GameAction.ACTION6.value in avail:
            return None
        if GameAction.ACTION5.value not in avail:
            return None
        outline = self._ov_outline(grid)
        if outline is None:
            return None  # no static box overlay → not this family
        self._ov_outline_votes[outline] += 1
        anchors = overlay_anchors(grid, outline)
        if not anchors:
            return None
        sel = self._ov_selected(grid, outline)
        if sel is None:
            return None  # transient frame with no clean selection hole
        center, arm = sel
        # LEARN MOVE DELTAS the head's own way: the generic move-vote path caps
        # mover bbox at GROUP_BBOX, but re86's arms span the whole frame, so
        # the arrows never earn a rule there.  Instead read the selection
        # hole's own displacement: a small one-axis step is a MOVE delta; the
        # SELECT verb jumps the hole to a far/diagonal piece (never recorded).
        if self._prev_action is not None and self._prev_action.isdigit() \
                and self._prev_grid is not None \
                and self._ov_last_center is not None:
            pa = int(self._prev_action)
            ddx = center[0] - self._ov_last_center[0]
            ddy = center[1] - self._ov_last_center[1]
            mag = abs(ddx) + abs(ddy)
            if 0 < mag <= 6 and (ddx == 0) != (ddy == 0):
                self._ov_deltas[pa][(ddx, ddy)] += 1
        # the four axis MOVE actions: those with a trusted one-axis delta.
        # SELECT is the available simple action with no such delta.
        axis: dict[int, Cell] = {}
        for a in sorted(avail):
            if a == GameAction.RESET.value \
                    or GameAction.from_id(a).is_complex():
                continue
            if self._ov_deltas[a]:
                d, n = self._ov_deltas[a].most_common(1)[0]
                if n >= VOTE_THRESHOLD:
                    axis[a] = d
        mags = {abs(d[0]) + abs(d[1]) for d in axis.values()}
        level = latest_frame.levels_completed
        if self._ov_benched == level:
            return None
        if self._ov_strikes >= OV_STRIKES:
            self._ov_benched = level
            return None
        if self._ov_level != level:
            self._ov_level = level
            self._ov_solved.clear()
            self._ov_plan.clear()
            self._ov_target = None
            self._ov_idprobe = 0
        # PROBE PHASE: arrows aren't all learned yet.  Cycle ACTION1..4 once
        # each (and SELECT can't be probed — it desyncs selection), recording
        # the hole displacement.  Bail out of probing once all four deltas are
        # trusted at one magnitude.
        simple = [a for a in sorted(avail)
                  if a != GameAction.RESET.value
                  and not GameAction.from_id(a).is_complex()
                  and a != GameAction.ACTION5.value]
        if len(axis) < 4 or len(mags) != 1:
            if self._ov_idprobe < len(simple) * VOTE_THRESHOLD:
                act = simple[self._ov_idprobe % len(simple)]
                self._ov_idprobe += 1
                self._ov_last_center = center
                return self._step(GameAction.from_id(act))
            return None  # probing exhausted without 4 clean axis deltas
        step = mags.pop()
        select = self._ov_select_verb(
            avail, {a: d for a, d in axis.items()})
        if select is None:
            return None
        self._ov_select = select
        # direction -> move action (unit axis of the learned step)
        dir_act = {(0, -1): None, (0, 1): None, (-1, 0): None, (1, 0): None}
        for a, d in axis.items():
            u = (0, -1) if d[1] < 0 else (0, 1) if d[1] > 0 \
                else (-1, 0) if d[0] < 0 else (1, 0)
            dir_act[u] = a
        # progress watchdog: an in-flight plan that left the centre unmoved
        # means the piece bounced a wall (re86 clamps at the camera edge) —
        # strike and re-plan
        if self._ov_plan and self._ov_last_center is not None \
                and center == self._ov_last_center:
            self._ov_plan.clear()
            self._ov_target = None
            self._ov_strikes += 1
            if self._ov_strikes >= OV_STRIKES:
                self._ov_benched = level
                return None
        # drive an in-flight plan toward the current target
        if self._ov_plan and self._ov_target is not None \
                and center != self._ov_target:
            act = self._ov_plan.popleft()
            self._ov_last_center = center
            return self._step(GameAction.from_id(act))
        # target reached (or no plan): this colour is placed — mark it solved
        if self._ov_target is not None and center == self._ov_target:
            self._ov_solved.add(arm)
            self._ov_target = None
            self._ov_plan.clear()
        # plan the selected colour's cover if it still has unsolved anchors
        my_anchors = anchors.get(arm, [])
        if my_anchors and arm not in self._ov_solved:
            tgt = cover_centers(frozenset(piece_footprint(grid, center, arm)),
                                center, my_anchors, step, center)
            if tgt is not None and tgt != center:
                plan: list[int] = []
                tx, ty = tgt
                dx, dy = tx - center[0], ty - center[1]
                nx = abs(dx) // step
                ny = abs(dy) // step
                ax = dir_act[(1, 0)] if dx > 0 else dir_act[(-1, 0)]
                ay = dir_act[(0, 1)] if dy > 0 else dir_act[(0, -1)]
                if dx and ax is not None:
                    plan += [ax] * nx
                if dy and ay is not None:
                    plan += [ay] * ny
                if plan:
                    self._ov_target = tgt
                    self._ov_plan = deque(plan)
                    act = self._ov_plan.popleft()
                    self._ov_last_center = center
                    return self._step(GameAction.from_id(act))
            # this colour is placed already (tgt == center) or uncoverable
            # (tgt is None — a non-axis shape this head defers): mark it done
            # so SELECT advances instead of looping on it
            self._ov_solved.add(arm)
        # nothing to drive for this colour: cycle SELECT to the next piece.
        # If every overlay colour is solved, the engine has already won; a
        # full cycle with no plan that lands here repeatedly is a dry run —
        # strike so the head eventually benches and novelty takes over.
        unsolved = [c for c in anchors if c not in self._ov_solved]
        if not unsolved:
            self._ov_strikes += 1
        self._ov_last_center = None
        self._ov_target = None
        return self._step(GameAction.from_id(select))

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
            # read the direction off the best-voted CARD-SIZED step: a
            # dominant wraparound jump points the opposite way
            dx, dy = next(
                ((sx, sy) for (sx, sy), _n in votes.most_common()
                 if 3 <= max(abs(sx), abs(sy)) <= 16),
                votes.most_common(1)[0][0])
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
            # engage on a trusted cycle verb — unless some movement rule
            # belongs to an action the editor could NOT classify (a real
            # avatar's moves).  A cursor's own steps form movement rules
            # too (balanced early sampling trusts them fast), but those
            # actions carry clean mov/mut classifications, so they don't
            # block: avatar games have no trusted mut verb in the first
            # place, and a misfire self-heals through the bench ladder.
            if not cyc or any(
                    not self._ed_verb(a, "mov") and not self._ed_verb(a, "mut")
                    for a in self._movement_rules()):
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
        # 2 — slot pitch: the selector verbs' best-voted SANE step
        # magnitude.  Wraparound jumps (the cursor stepping off a row end
        # back to the start, a multiple of the true pitch) can dominate
        # the raw vote when pre-engage sampling trapped the cursor at a
        # row end — scan the vote in descending order for the first
        # card-sized magnitude instead of trusting the top entry.
        votes: Counter[Cell] = Counter()
        for a in sel:
            votes.update(self._ed_deltas[a])
        pitch = None
        for (dx, dy), _n in votes.most_common():
            mag = max(abs(dx), abs(dy))
            if 3 <= mag <= 16:
                pitch = mag
                break
        if pitch is None:
            # only wrap jumps on record: drive a selector ourselves —
            # consecutive presses yield true steps (wraps are 1-in-N) —
            # and bench through the strike ladder if the vote never
            # sanitizes (a real non-editor game)
            self._ed_miss += 1
            if self._ed_miss > ED_MISS_CAP:
                self._ed_miss = 0
                self._ed_strikes += 1
            return self._step(GameAction.from_id(sel[0]))
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

    def _sl_avatar_anchor(
        self, grid: Grid
    ) -> Optional[tuple[Cell, int]]:
        """Anchor of the node-maze avatar's accent: the rarest non-background
        color that forms ONE tiny stable component (a single accent pixel or
        a handful).  Returns (anchor cell, color), or None when no unique
        rare accent exists (most games — this keeps the slide observer inert).
        Pure read; never mutates state."""
        counts: Counter[int] = Counter()
        for row in grid:
            counts.update(row)
        if len(counts) < 3:
            return None
        bg = counts.most_common(1)[0][0]
        # candidate accent colors: rare ones, capped tiny so a busy frame's
        # least-common-but-still-large color never qualifies
        rare = [c for c, n in counts.items()
                if c != bg and 0 < n <= SL_ACCENT_MAX]
        if not rare:
            return None
        comps = components(grid)
        best: Optional[tuple[int, Cell, int]] = None  # (size, anchor, color)
        for color in rare:
            cc = [cells for c, cells in comps if c == color]
            if len(cc) != 1:
                continue  # ambiguous: not a unique accent
            cells = cc[0]
            cand = (len(cells), min(cells), color)
            if best is None or cand < best:
                best = cand
        if best is None:
            return None
        return (best[1], best[2])

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

    # ── sequence-match assignment model (sb26 family) ──────────────────────
    def _sort_hollow_boxes(
        self, grid: Grid
    ) -> list[tuple[int, int, int, int, int]]:
        """Every same-colour component that is a HOLLOW rectangular ring (its
        cells are exactly the bounding-box perimeter, interior is some OTHER
        colour).  Returns (colour, x0, y0, x1, y1) per box.  These are the
        target-row glyphs, the container walls and the nested-door frames —
        the whole sb26 board is built from hollow boxes."""
        out: list[tuple[int, int, int, int, int]] = []
        for color, cells in components(grid):
            xs = [x for x, _y in cells]
            ys = [y for _x, y in cells]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            w, h = x1 - x0 + 1, y1 - y0 + 1
            if w < 4 or h < 4:
                continue
            ring = 2 * (w + h) - 4
            # a clean ring: every cell on the perimeter, none inside, and the
            # count matches a full rectangular outline (a 1px gap on a door's
            # mouth is tolerated by the +/-2 slack)
            if all(x in (x0, x1) or y in (y0, y1) for x, y in cells) \
                    and abs(len(cells) - ring) <= 2:
                out.append((color, x0, y0, x1, y1))
        return out

    def _sort_solid_tiles(
        self, grid: Grid, bg: int
    ) -> list[tuple[int, frozenset[Cell]]]:
        """Solid (filled, non-hollow) rectangular blocks: the palette tiles
        and the placed tokens render as a solid square of one colour."""
        out: list[tuple[int, frozenset[Cell]]] = []
        for color, cells in components(grid):
            if color == bg:
                continue
            xs = [x for x, _y in cells]
            ys = [y for _x, y in cells]
            w = max(xs) - min(xs) + 1
            h = max(ys) - min(ys) + 1
            if 2 <= w <= 8 and 2 <= h <= 8 and len(cells) == w * h:
                out.append((color, cells))
        return out

    def _sort_target_row(
        self, boxes: list[tuple[int, int, int, int, int]]
    ) -> Optional[list[tuple[int, int]]]:
        """The target sequence: the topmost horizontal run of >= 3 equal-size
        hollow boxes sharing one y-band, left-to-right.  Returns
        [(colour, centre_x), ...] or None.  Distinct border colours are NOT
        required (a target may repeat a colour) but >= 2 distinct colours are,
        so a plain grid of identical cells never reads as a target row."""
        if len(boxes) < SORT_MIN_TARGETS:
            return None
        by_band: dict[tuple, list[tuple[int, int, int, int, int]]] = \
            defaultdict(list)
        for b in boxes:
            _c, x0, y0, x1, y1 = b
            by_band[(y0, x1 - x0, y1 - y0)].append(b)
        bands = [grp for grp in by_band.values() if len(grp) >= SORT_MIN_TARGETS]
        if not bands:
            return None
        # topmost band (smallest y0) is the target row
        band = min(bands, key=lambda g: g[0][2])
        band = sorted(band, key=lambda b: b[1])
        colors = [b[0] for b in band]
        if len(set(colors)) < 2:
            return None
        return [(b[0], (b[1] + b[3]) // 2) for b in band]

    def _sort_policy(
        self, grid: Grid, latest_frame: FrameData
    ) -> Optional[GameAction]:
        """Solve a sequence-match assignment puzzle (sb26 family).

        GATE (very specific, low collision): clicks available, NO movement
        rules, NO avatar, plus the structural signature — a target ROW of
        >= 3 equal-size hollow boxes, empty slot markers (one uniform colour)
        inside larger container boxes, and a palette of solid tiles whose
        colour-multiset equals the target multiset.

        The win walk descends doors (interior colour == the opened
        container's border colour) so the fill order is a DFS over the
        container tree; target[i] -> dfs_slot[i] is a positional bijection.
        Each call emits ONE click of a two-click drag (token, then slot); the
        final slot is filled last and ACTION5 fires the verification.  A
        placement that does not change the board is a strike; SORT_STRIKES
        benches the level."""
        avail = set(latest_frame.available_actions or []) \
            - {GameAction.RESET.value}
        if GameAction.ACTION6.value not in avail:
            return None
        if self._movement_rules() or self._find_avatar(grid):
            return None
        level = latest_frame.levels_completed
        if self._sort_benched == level:
            return None

        # ── replay an in-flight plan ──────────────────────────────────────
        if self._sort_plan and self._sort_level == level \
                and self._sort_idx < len(self._sort_plan):
            return self._sort_emit(grid)

        # ── (re)build the plan from the board ─────────────────────────────
        counts: Counter[int] = Counter()
        for row in grid:
            counts.update(row)
        bg = counts.most_common(1)[0][0]
        boxes = self._sort_hollow_boxes(grid)
        targets = self._sort_target_row(boxes)
        if not targets:
            return None
        tgt_colors = [c for c, _x in targets]
        # palette tiles: solid blocks whose colour multiset == target multiset
        tiles = self._sort_solid_tiles(grid, bg)
        # group candidate palette tiles by colour; a colour may repeat
        pal_by_color: dict[int, list[Cell]] = defaultdict(list)
        # the palette sits in its own row strip (max y); take the solid tiles
        # on the lowest band so placed-token blocks inside containers don't
        # masquerade as palette
        if not tiles:
            return None
        max_y = max(min(y for _x, y in cells) for _c, cells in tiles)
        pal_tiles = [(c, cells) for c, cells in tiles
                     if min(y for _x, y in cells) >= max_y - 2]
        for c, cells in pal_tiles:
            xs = [x for x, _y in cells]
            ys = [y for _x, y in cells]
            pal_by_color[c].append(((min(xs) + max(xs)) // 2,
                                    (min(ys) + max(ys)) // 2))
        if Counter(c for c, _ in pal_tiles) != Counter(tgt_colors):
            return None

        # containers: the hollow boxes that are NOT the target row and are
        # large enough to hold slots (bigger than a target glyph)
        tgt_centers = {x for _c, x in targets}
        glyph_w = None
        for _c, x0, y0, x1, y1 in boxes:
            if (x0 + x1) // 2 in tgt_centers:
                glyph_w = x1 - x0
                tgt_y_band = y0
                break
        if glyph_w is None:
            return None
        containers = [b for b in boxes
                      if not ((b[1] + b[3]) // 2 in tgt_centers
                              and b[2] == tgt_y_band)
                      and (b[3] - b[1]) > glyph_w]
        if not containers:
            return None

        # slot markers: small uniform-colour solid blocks sitting INSIDE a
        # container (not the palette, not a target glyph).  Their colour is
        # uniform across all empty slots; doors are hollow boxes inside a
        # container whose interior colour matches some container's border.
        slot_blocks = [(c, cells) for c, cells in tiles
                       if min(y for _x, y in cells) < max_y - 2]
        in_cont = []
        for c, cells in slot_blocks:
            cx = sum(x for x, _y in cells) // len(cells)
            cy = sum(y for _x, y in cells) // len(cells)
            for _bc, x0, y0, x1, y1 in containers:
                if x0 < cx < x1 and y0 < cy < y1:
                    in_cont.append((c, (cx, cy), (x0, y0, x1, y1)))
                    break
        if not in_cont:
            return None
        # the slot marker colour is the most common colour among in-container
        # solid blocks
        slot_color = Counter(c for c, _ctr, _box in in_cont).most_common(1)[0][0]
        empty_slots = [(ctr, box) for c, ctr, box in in_cont if c == slot_color]
        if len(empty_slots) != len(tgt_colors):
            return None

        order = self._sort_dfs_slots(containers, empty_slots, boxes, bg, grid)
        if order is None or len(order) != len(tgt_colors):
            return None

        # positional bijection target[i] -> dfs_slot[i]; fill the final slot
        # LAST so the verification fires only when the board is complete
        plan: list[tuple[int, Cell, Cell]] = []
        used: dict[int, int] = defaultdict(int)
        for (tc, _tx), slot in zip(targets, order):
            picks = pal_by_color.get(tc)
            if not picks:
                return None
            cell = picks[used[tc] % len(picks)]
            used[tc] += 1
            plan.append((tc, cell, slot))
        self._sort_plan = plan
        self._sort_idx = 0
        self._sort_phase = 0
        self._sort_level = level
        self._sort_prev_sig = None
        return self._sort_emit(grid)

    def _sort_dfs_slots(
        self,
        containers: list[tuple[int, int, int, int, int]],
        empty_slots: list[tuple[Cell, tuple]],
        boxes: list[tuple[int, int, int, int, int]],
        bg: int, grid: Grid,
    ) -> Optional[list[Cell]]:
        """The fill order the win walk visits slots in: descend each
        container left-to-right; a DOOR (a hollow box inside a container whose
        interior colour equals some container's BORDER colour) recurses into
        that container, then control returns.  Returns the ordered slot cells,
        or None if the door graph is malformed."""
        # index containers by border colour (the door target key)
        cont_by_color: dict[int, tuple] = {}
        for b in containers:
            cont_by_color[b[0]] = b
        # slots per container (keyed by the container's full 5-tuple so it
        # agrees with doors_in and walk()), left-to-right.  A slot's stored
        # box is the (x0,y0,x1,y1) of the container it sits in; match it back
        # to the 5-tuple.
        slots_in: dict[tuple, list[Cell]] = defaultdict(list)
        for ctr, box in empty_slots:
            for cb in containers:
                if cb[1:5] == tuple(box):
                    slots_in[cb].append(ctr)
                    break
        # doors per container: a hollow box strictly inside a container whose
        # interior colour matches another container's border colour
        doors_in: dict[tuple, list[tuple[int, Cell]]] = defaultdict(list)
        for dc, dx0, dy0, dx1, dy1 in boxes:
            for cb in containers:
                _bc, cx0, cy0, cx1, cy1 = cb
                if cb[:5] == (dc, dx0, dy0, dx1, dy1):
                    continue
                if cx0 < dx0 and dx1 < cx1 and cy0 < dy0 and dy1 < cy1:
                    interior = grid[(dy0 + dy1) // 2][(dx0 + dx1) // 2]
                    if interior in cont_by_color and interior != cb[0]:
                        doors_in[cb].append((interior,
                                             ((dx0 + dx1) // 2,
                                              (dy0 + dy1) // 2)))
                    break

        # walk: at each container, interleave slots and doors by x-position
        order: list[Cell] = []
        visited: set[tuple] = set()

        def walk(cont: tuple) -> bool:
            if cont in visited:
                return False
            visited.add(cont)
            items: list[tuple[int, str, object]] = []
            for ctr in slots_in.get(cont, []):
                items.append((ctr[0], "slot", ctr))
            for icolor, dctr in doors_in.get(cont, []):
                items.append((dctr[0], "door", icolor))
            items.sort(key=lambda t: t[0])
            for _x, kind, payload in items:
                if kind == "slot":
                    order.append(payload)  # type: ignore[arg-type]
                else:
                    child = cont_by_color.get(payload)  # type: ignore[index]
                    if child is None or not walk(child):
                        return False
            return True

        # root = the top-most / left-most container that no door points into
        door_targets = {ic for ds in doors_in.values() for ic, _ in ds}
        roots = [c for c in containers if c[0] not in door_targets]
        if not roots:
            roots = containers
        root = min(roots, key=lambda b: (b[2], b[1]))
        if not walk(root):
            return None
        # any container not reached (disconnected door graph) → bail
        if len(order) != len(empty_slots):
            return None
        return order

    def _sort_emit(self, grid: Grid) -> Optional[GameAction]:
        """Emit one click of the current two-click drag.  Phase 0 selects the
        token, phase 1 drops it on the slot and advances the cursor; after the
        last slot, ACTION5 fires the verification.  A placement that left the
        board unchanged is a strike (mis-parsed slot/token)."""
        if self._sort_idx >= len(self._sort_plan):
            # all placed: trigger the verification look
            self._sort_idx += 1  # one-shot guard
            return self._sort_look()
        _tc, token_cell, slot_cell = self._sort_plan[self._sort_idx]
        if self._sort_phase == 0:
            # before selecting, check the prior placement actually landed
            if self._sort_prev_sig is not None:
                if grid_hash(grid) == self._sort_prev_sig:
                    self._sort_strikes += 1
                    if self._sort_strikes >= SORT_STRIKES:
                        self._sort_benched = self._sort_level
                        self._sort_plan = []
                        return None
                self._sort_prev_sig = None
            self._sort_phase = 1
            return self._sort_click(token_cell, "sort: select token")
        # phase 1: drop on the slot, advance
        self._sort_phase = 0
        self._sort_prev_sig = grid_hash(grid)
        self._sort_idx += 1
        return self._sort_click(slot_cell, "sort: place in slot")

    def _sort_look(self) -> GameAction:
        action = GameAction.ACTION5
        action.reasoning = {"why": "sort: verify placement"}
        self._prev_action = str(GameAction.ACTION5.value)
        return action

    def _sort_click(self, cell: Cell, why: str) -> GameAction:
        x, y = cell
        action = GameAction.ACTION6
        action.set_data({"x": int(x), "y": int(y)})
        action.reasoning = {"why": why}
        self._prev_action = f"6:{int(x)},{int(y)}"
        return action

    # ── attractor-herd head (su15 family) ─────────────────────────────────
    @staticmethod
    def _herd_blobs(grid: Grid) -> tuple[Optional[dict], list[dict]]:
        """Pixel-only goal + mover detection for the attractor-herd family.

        The board is a 64x64 field with thin decoration bars hugging the top/
        bottom edges, optional dotted guide specks (size-1 cells), a medium
        SOLID square-ish GOAL zone, and small compact MOVER blobs.  Returns
        (goal, movers) where each is a dict {cx, cy, size, color, bbox};
        movers are nearest-to-goal first.  Movers that overlap the goal box
        are dropped (a decoration sitting on the goal is not a thing to herd).
        Static look-alikes are filtered out at the call site via responsiveness
        (the inert memory), not here — shape alone can't separate them."""
        comps = components(grid)
        info: list[dict] = []
        for color, cells in comps:
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs) + 1, max(ys) + 1
            info.append({
                "color": color, "size": len(cells),
                "cx": sum(xs) / len(cells), "cy": sum(ys) / len(cells),
                "bbox": (x0, y0, x1, y1),
            })
        if not info:
            return None, []

        def solidity(c: dict) -> float:
            x0, y0, x1, y1 = c["bbox"]
            return c["size"] / max(1, (x1 - x0) * (y1 - y0))

        def square(c: dict) -> float:
            x0, y0, x1, y1 = c["bbox"]
            w, h = x1 - x0, y1 - y0
            return min(w, h) / max(1, max(w, h))

        def is_bar(c: dict) -> bool:
            x0, y0, x1, y1 = c["bbox"]
            w, h = x1 - x0, y1 - y0
            spans = w >= GRID - 4 or h >= GRID - 4
            edge = y0 <= 1 or y1 >= GRID - 1 or x0 <= 1 or x1 >= GRID - 1
            return spans and edge

        live = [c for c in info if c["size"] >= 2 and not is_bar(c)]
        if not live:
            return None, []
        goal_cands = [c for c in live
                      if 16 <= c["size"] <= 120
                      and solidity(c) >= 0.5 and square(c) >= 0.6]
        goal = (max(goal_cands, key=lambda c: c["size"]) if goal_cands
                else max(live, key=lambda c: c["size"]))
        gx0, gy0, gx1, gy1 = goal["bbox"]
        movers = [c for c in live
                  if c is not goal and 4 <= c["size"] <= 16
                  and solidity(c) >= 0.55 and square(c) >= 0.5
                  and not (gx0 - 1 <= c["cx"] <= gx1 + 1
                           and gy0 - 1 <= c["cy"] <= gy1 + 1)]
        movers.sort(key=lambda c: (c["cx"] - goal["cx"]) ** 2
                    + (c["cy"] - goal["cy"]) ** 2)
        return goal, movers

    def _herd_policy(
        self, grid: Grid, latest_frame: FrameData
    ) -> Optional[GameAction]:
        """Walk a mover blob into the goal zone via successive attractor
        clicks (su15 family).  GATE: the available-action set is EXACTLY
        {ACTION6, ACTION7} — empirically unique to this family, so the head
        cannot collide with any movement game or with the {6}-only click
        games (lp85/vc33/ft09/...).  One click per call; the mover the last
        click aimed at is judged for responsiveness off _prev_grid (a blob
        still exactly where the head clicked toward it is a decoration ->
        inert).  A no-progress streak benches the level."""
        avail = set(latest_frame.available_actions or []) \
            - {GameAction.RESET.value}
        if avail != {GameAction.ACTION6.value, GameAction.ACTION7.value}:
            return None
        level = latest_frame.levels_completed
        if self._herd_benched == level:
            return None
        if level != self._herd_level:
            # new level: fresh inert/aim memory and strikes
            self._herd_level = level
            self._herd_inert = set()
            self._herd_last = None
            self._herd_strikes = 0

        # ── responsiveness: did the LAST aimed candidate move? ────────────
        # _prev_grid is the frame the head last acted on (set by _learn).  If
        # the same-position blob is still there unchanged, it's a static
        # look-alike — remember it so we stop aiming at it.  Any board change
        # at all resets the no-progress strike counter.
        progressed = False
        if self._herd_last is not None and self._prev_grid is not None:
            lx, ly = self._herd_last
            if grid != self._prev_grid:
                progressed = True
            still = False
            for color, cells in components(grid):
                xs = [c[0] for c in cells]
                ys = [c[1] for c in cells]
                cx, cy = sum(xs) / len(cells), sum(ys) / len(cells)
                if abs(cx - lx) < 1.0 and abs(cy - ly) < 1.0 \
                        and len(cells) <= 16:
                    still = True
                    break
            if still:
                self._herd_inert.add((round(lx), round(ly)))
        if progressed:
            self._herd_strikes = 0
        else:
            self._herd_strikes += 1
            if self._herd_strikes >= HERD_STRIKES:
                self._herd_benched = level
                return None

        goal, movers = self._herd_blobs(grid)
        movers = [m for m in movers
                  if (round(m["cx"]), round(m["cy"])) not in self._herd_inert]
        if goal is None or not movers:
            # nothing actionable this frame: clear inert (geometry may have
            # shifted) and decline — let novelty probe instead of stalling
            self._herd_inert = set()
            self._herd_last = None
            return None

        m = movers[0]
        mcx, mcy = m["cx"], m["cy"]
        gcx, gcy = goal["cx"], goal["cy"]
        dx, dy = gcx - mcx, gcy - mcy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1.0:
            tx, ty = gcx, gcy
        else:
            r = min(HERD_SEL, dist)         # stay in grab radius of the mover
            tx, ty = mcx + dx / dist * r, mcy + dy / dist * r
        tx = max(0, min(GRID - 1, int(round(tx))))
        ty = max(0, min(GRID - 1, int(round(ty))))
        self._herd_last = (mcx, mcy)
        action = GameAction.ACTION6
        action.set_data({"x": tx, "y": ty})
        action.reasoning = {"why": f"herd mover {(round(mcx), round(mcy))} "
                                   f"-> goal {(round(gcx), round(gcy))}"}
        self._prev_action = f"6:{tx},{ty}"
        return action

    # ── planning ─────────────────────────────────────────────────────────
    def _policy(self, grid: Optional[Grid], latest_frame: FrameData) -> GameAction:
        # CURIO_GENERIC_ONLY ablation: skip every family-specific head and
        # its gate, dropping straight through to the generic core (warmup,
        # novelty, BFS routing).  `not GENERIC_ONLY` is a module-level const
        # that is True by default, so the branch below is bit-identical to
        # the full agent when the toggle is unset.
        if grid is not None and not GENERIC_ONLY:
            # triple-gated lattice solver (click-only + board + learned
            # recolor effect); returns None whenever the gate doesn't apply
            lattice = self._lattice_policy(grid, latest_frame)
            if lattice is not None:
                return lattice
            # directional node-maze solver (tu93 family).  Dispatched BEFORE
            # the editor: both share the [1,2,3,4] action set, but the slide
            # gate is far stricter (a clean binary corridor/wall lattice + a
            # unique accent avatar on a node + a rare-colour exit node), so it
            # returns None on tr87 (no lattice, in-place glyph cycle) and
            # ls20 (irregular tiles + register), leaving those heads intact —
            # while winning the race on a true maze, where the editor's loose
            # in-place-cycle gate would otherwise mis-engage on the slide
            # animation and stall.  Its own <=4-action directional probe
            # replaces the 500-step novelty opening.
            slide = self._slide_policy(grid, latest_frame)
            if slide is not None:
                return slide
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
            # overlay-align head (re86 family): a SELECTED piece (centre hole
            # = 0) placed so its arm footprint covers a STATIC goal-overlay's
            # anchor centres.  Gated inside on the [1,2,3,4,5]-no-ACTION6
            # action set PLUS a static hollow-box overlay signature PLUS a
            # cursor that JUMPS (not translates) on the 5th verb — the
            # secondary gate is load-bearing: g50t shares the action set but
            # has a scrolling obstacle and no box overlay, so it cannot reach
            # the model.  Runs AFTER the port branch (re86 exposes no ACTION6,
            # so the port gate already declined) and before warmup.
            overlay = self._overlay_policy(grid, latest_frame)
            if overlay is not None:
                return overlay
            # remote-toggle switch planner (dc22 family), gated inside on
            # clicks + trusted movement rules + no register + no port
            # color + the readiness clause + SW_FRAMES diffed frames (so
            # cn04's port pre-vote always wins the race).  Runs AFTER the
            # port branch and before the warmup gate: every switch action
            # either probes a candidate button or executes a planned
            # (position x phase) route.
            switch = self._switch_policy(grid, latest_frame)
            if switch is not None:
                return switch
            # sequence-match assignment solver (sb26 family), gated inside on
            # clicks + NO movement rules + NO avatar + the structural target-
            # row / slots / matching-palette signature.  This [5,6,7]-style
            # no-avatar profile is structurally absent from every movement
            # floor game, so the gate cannot fire where an avatar or movement
            # rules exist; the strike/bench self-disable contains a mis-parse.
            # Runs after the switch branch and before warmup: a placement plan
            # is exact, so there is no value in random opening clicks.
            sort = self._sort_policy(grid, latest_frame)
            if sort is not None:
                return sort
            # attractor-herd solver (su15 family), gated on the EXCLUSIVE
            # {6,7} action set (click + secondary/undo — empirically unique to
            # this family across the battery: every other click game is {6}
            # only, and every ACTION7 game also exposes movement).  A click is
            # a black hole that pulls movable blobs within HERD_SEL toward it;
            # the head walks a mover into the static goal zone via successive
            # in-radius clicks, learning which look-alike blobs are inert
            # (decorations) from their non-response.  No movement rules, no
            # avatar, no register exist in this profile, so the gate cannot
            # collide with any floor/held game; a no-progress streak benches
            # the level so novelty takes over.
            herd = self._herd_policy(grid, latest_frame)
            if herd is not None:
                return herd
        # warmup ends early once the model is live (2+ trusted movement rules
        # and a located avatar): 500 random steps would burn through ls20's
        # 42-pip lives and cn04's 75-action budget before planning ever
        # starts.  The readiness clause mirrors the attr planner's gate: a
        # positional planner handed control at TWO rules monopolizes the
        # policy with two-direction plans and starves the remaining rules
        # of samples (measured on cn04: rules 3/4 untrusted until action
        # ~154 while the planner shuttled the piece vertically) — stay in
        # balanced novelty until every available simple action has a rule
        # or RULE_TRIES uses.  Click games get the analogous exit
        # (_click_warm: a confidently-productive affordance signature), and
        # a level-up onto a known level signature skips warmup outright
        # (_warmup_skip).
        if self.action_counter < NOVELTY_WARMUP \
                and not self._warmup_skip \
                and not self._click_warm():
            rules = self._movement_rules()
            avail_s = set(latest_frame.available_actions or []) \
                - {GameAction.RESET.value, GameAction.ACTION6.value}
            if not (len(rules) >= 2
                    and all(a in rules or self._act_uses[a] >= RULE_TRIES
                            for a in avail_s)
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

    def _emit_key(self, action_key: str) -> GameAction:
        """Re-issue a recorded action key verbatim (win-path replay):
        click coords included, so the repeat is exact."""
        if action_key.startswith("6:"):
            x, y = map(int, action_key[2:].split(","))
            action = GameAction.ACTION6
            action.set_data({"x": x, "y": y})
        else:
            action = GameAction.from_id(int(action_key))
        action.reasoning = {"why": "win-path replay"}
        self._prev_action = action_key
        return action

    def _click_warm(self) -> bool:
        """ADAPTIVE WARMUP EXIT for click games: a confidently-productive
        click signature (CLICK_WARM_TRIES+ tries, Laplace P(effect) above
        CLICK_WARM_P) is the click analog of a trusted movement rule —
        once the library holds one, and the lattice/attribute detectors
        have had CLICK_WARM_FRAMES diffed frames to fire first, novelty
        warmup has done its discovery job."""
        if self._frames_diffed < CLICK_WARM_FRAMES:
            return False
        return any(
            tries >= CLICK_WARM_TRIES
            and (changed + 1) / (tries + 2) > CLICK_WARM_P
            for changed, tries in self._click_effects.values())

    # ── switch model (dc22 family): remote cyclic toggles ───────────────
    def _sw_band_drop(self, a: Grid, b: Grid) -> int:
        """Changed cells inside the border band: a fall trap drains the
        budget strip there while restoring the board interior exactly."""
        n = 0
        for y in range(GRID):
            ra, rb = a[y], b[y]
            if ra == rb:
                continue
            band_row = y < HUD_BAND or y >= GRID - HUD_BAND
            for x in range(GRID):
                if ra[x] != rb[x] and (band_row or x < HUD_BAND
                                       or x >= GRID - HUD_BAND):
                    n += 1
        return n

    def _sw_same_inside(self, a: Grid, b: Grid) -> bool:
        """Grid equality over the board interior (border band and masked
        HUD cells exempt): the fall trap's exact-restore test."""
        for y in range(HUD_BAND, GRID - HUD_BAND):
            ra, rb = a[y], b[y]
            if ra == rb:
                continue
            for x in range(HUD_BAND, GRID - HUD_BAND):
                if ra[x] != rb[x] and (x, y) not in self._hud_mask:
                    return False
        return True

    def _learn_switch(
        self, grid: Grid, click: Cell,
        comps_prev: list[tuple[int, frozenset[Cell]]], sig: int,
    ) -> None:
        """Record a click's remote repaint for the switch model: the
        changed-cell set in the board interior, plus before/after pixel
        patches over the signature's running union mask.  An event that
        changes the clicked component itself (lp85: clicked pieces move at
        the click site) poisons the record for good; an event touching the
        avatar (it rode a toggled platform) is dropped — its patches would
        read avatar pixels as switch phases."""
        changed: set[Cell] = set()
        for y in range(HUD_BAND, GRID - HUD_BAND):
            prow, row = self._prev_grid[y], grid[y]
            if prow == row:
                continue
            for x in range(HUD_BAND, GRID - HUD_BAND):
                if prow[x] != row[x] and (x, y) not in self._hud_mask:
                    changed.add((x, y))
        if not changed:
            return
        rec = self._sw_recs.setdefault(
            sig, {"mask": set(), "events": [], "overlap": False})
        clicked = next((cells for _c, cells in comps_prev if click in cells),
                       frozenset())
        if changed & clicked:
            rec["overlap"] = True
            return
        av_prev = self._find_avatar(self._prev_grid)
        av_cur = self._find_avatar(grid)
        av_cells = set(av_prev[1]) if av_prev else set()
        if av_cur:
            av_cells |= set(av_cur[1])
        if (changed | rec["mask"]) & av_cells:
            # the avatar rode the toggle (dc22's paired-tile carry): the
            # patches are polluted, but the ride itself is the lesson —
            # clicking this switch with the avatar parked on a mask tile
            # teleports it to the paired tile
            if av_prev and av_cur and av_prev[0] != av_cur[0]:
                rec.setdefault("tele", {})[av_prev[0]] = av_cur[0]
                self._sw_nogoal = None
            return
        rec["mask"] |= changed
        cells = tuple(sorted(rec["mask"], key=lambda c: (c[1], c[0])))
        rec["events"].append((
            frozenset(changed), cells,
            tuple(self._prev_grid[cy][cx] for cx, cy in cells),
            tuple(grid[cy][cx] for cx, cy in cells)))
        del rec["events"][:-24]
        self._sw_nogoal = None  # new evidence: a dry planner may retry

    def _switch_cycle(
        self, sig: int
    ) -> Optional[tuple[tuple, list[tuple], Optional[list[int]], bool]]:
        """(mask cells raster-ordered, phase patches, successor indices,
        is_pingpong) of a confirmed toggle signature, or None.  Confirmed
        when (i) a recurring effect core exists — cells repainted in 2+
        uses (a multi-phase gate repaints a different subset per step, so
        the core is the union of everything that ever changed TWICE;
        cells that changed once are one-shot transients and stay out),
        (ii) no use ever changed the clicked component itself (rejects
        lp85-style translations), and (iii) the before→after core patches
        form either a DETERMINISTIC graph holding a cycle — a ring, or a
        transient tail leading into one (L1's hint boxes vanish on the
        first press of a life, then the bridges cycle: a rho shape) — or
        a simple PATH walked back and forth (dc22's gate crates render
        open→closing→bridge→opening with aliased frames — a ping-pong).
        Re-clicking a selected cn04 piece is a no-op, so selection clicks
        can never confirm either shape."""
        rec = self._sw_recs.get(sig)
        if rec is None or rec["overlap"] or len(rec["events"]) < 2:
            return None
        events = rec["events"]
        votes: Counter[Cell] = Counter()
        for chg, _c, _b, _a in events:
            votes.update(chg)
        cells = tuple(sorted((c for c, v in votes.items() if v >= 2),
                             key=lambda c: (c[1], c[0])))
        if not cells:
            return None
        succ: dict[tuple, set[tuple]] = {}
        full = 0
        for _chg, ecells, before, after in events:
            idx = {c: i for i, c in enumerate(ecells)}
            if any(c not in idx for c in cells):
                continue  # recorded before the union covered the core
            b = tuple(before[idx[c]] for c in cells)
            a = tuple(after[idx[c]] for c in cells)
            if b == a:
                continue  # this event only touched transient cells
            full += 1
            succ.setdefault(b, set()).add(a)
        if full < 2 or not succ:
            return None
        if all(len(v) == 1 for v in succ.values()):
            # deterministic successors: confirm when the graph holds a
            # cycle (clicking is eventually reversible — what makes this
            # a switch rather than lp85's one-way translations)
            trans = {b: next(iter(v)) for b, v in succ.items()}
            patches: list[tuple] = []
            for b, a in trans.items():
                if b not in patches:
                    patches.append(b)
                if a not in patches:
                    patches.append(a)
            if len(patches) <= SW_CYCLE_CAP:
                pidx = {p: i for i, p in enumerate(patches)}
                nexts = [pidx[trans[p]] if p in trans else -1
                         for p in patches]
                cyclic = False
                for s0 in range(len(patches)):
                    on_path: set[int] = set()
                    cur_i = s0
                    while cur_i >= 0 and cur_i not in on_path:
                        on_path.add(cur_i)
                        cur_i = nexts[cur_i]
                    if cur_i >= 0:
                        cyclic = True
                        break
                if cyclic:
                    return (cells, patches, nexts, False)
            return None
        # ping-pong: the undirected transition graph is one simple path —
        # only trusted once a BOUNCE was observed (some patch with two
        # distinct successors), or an under-sampled ring would alias
        if not any(len(v) == 2 for v in succ.values()):
            return None
        adj: dict[tuple, set[tuple]] = {}
        for b, outs in succ.items():
            if len(outs) > 2:
                return None
            for a in outs:
                adj.setdefault(b, set()).add(a)
                adj.setdefault(a, set()).add(b)
        ends = [n for n in adj if len(adj[n]) == 1]
        if len(ends) != 2 or any(len(v) > 2 for v in adj.values()):
            return None
        prev: Optional[tuple] = None
        cur = ends[0]
        path = [cur]
        while cur != ends[1]:
            nxts = [n for n in adj[cur] if n != prev]
            if len(nxts) != 1 or len(path) >= SW_CYCLE_CAP:
                return None
            prev, cur = cur, nxts[0]
            path.append(cur)
        if len(path) != len(adj):
            return None  # disconnected extras: polluted reading
        return (cells, path, None, True)

    @staticmethod
    def _sw_adv(
        k: int, nexts: Optional[list[int]], ph: tuple[int, int]
    ) -> Optional[tuple[int, int]]:
        """One click's effect on a (phase index, direction) state:
        deterministic graphs follow their successor table (None when the
        successor from here was never observed), ping-pong paths bounce
        off their endpoints."""
        idx, d = ph
        if d == 0:
            nx = -1 if nexts is None or idx >= len(nexts) else nexts[idx]
            return None if nx < 0 else (nx, 0)
        ni = idx + d
        nd = d
        if ni >= k - 1:
            nd = -1
        if ni <= 0:
            nd = 1
        return (ni, nd)

    @staticmethod
    def _sw_phase(
        grid: Grid, cells: tuple, ring: list[tuple],
        ignore: frozenset[Cell] | set[Cell] = frozenset(),
    ) -> Optional[int]:
        """Which phase patch the frame currently shows.  Cells under the
        avatar are wildcards (walking a toggled bridge occludes part of
        its own mask); None when no phase — or more than one — matches."""
        idxs = [i for i, c in enumerate(cells) if c not in ignore]
        if not idxs:
            return None
        patch = tuple(grid[cells[i][1]][cells[i][0]] for i in idxs)
        matches = [k for k, rp in enumerate(ring)
                   if tuple(rp[i] for i in idxs) == patch]
        return matches[0] if len(matches) == 1 else None

    def _sw_goal_anchors(
        self, grid: Grid, comps: list[tuple[int, frozenset[Cell]]],
        avatar: tuple[Cell, frozenset[Cell]],
    ) -> frozenset[Cell]:
        """Anchors of the rarest-color avatar-sized components — the
        exact-overlap goal hypothesis (dc22's win test is anchor equality
        with a 2x2 pad).  The avatar's own pixels, the border band and
        masked HUD cells don't vote for rarity; only components matching
        the avatar's pixel count and bbox qualify (exact overlap needs an
        avatar-shaped pad — dc22 L3's 2-pixel checker speckles tie the
        goal color on rarity but can never BE the goal)."""
        av_cells = set(avatar[1])
        counts: Counter[int] = Counter()
        for y in range(HUD_BAND, GRID - HUD_BAND):
            row = grid[y]
            for x in range(HUD_BAND, GRID - HUD_BAND):
                if (x, y) not in av_cells and (x, y) not in self._hud_mask:
                    counts[row[x]] += 1
        if not counts:
            return frozenset()
        background = counts.most_common(1)[0][0]
        axs = [x for x, _y in av_cells]
        ays = [y for _x, y in av_cells]
        av_w = max(axs) - min(axs) + 1
        av_h = max(ays) - min(ays) + 1
        for color, _n in sorted(counts.items(), key=lambda kv: (kv[1], kv[0])):
            if color == background:
                continue
            anchors: set[Cell] = set()
            for ccolor, cells in comps:
                if ccolor != color or cells & av_cells \
                        or len(cells) != len(av_cells):
                    continue
                xs = [x for x, _y in cells]
                ys = [y for _x, y in cells]
                if max(ys) < HUD_BAND or min(ys) >= GRID - HUD_BAND \
                        or max(xs) < HUD_BAND or min(xs) >= GRID - HUD_BAND:
                    continue  # band-confined: HUD chrome
                if max(xs) - min(xs) + 1 != av_w \
                        or max(ys) - min(ys) + 1 != av_h:
                    continue
                anchors.add(min(cells))
            if anchors:
                return frozenset(anchors)
        return frozenset()

    @staticmethod
    def _sw_click_cell(cells: frozenset[Cell]) -> Cell:
        """A pixel that hits this component: centroid, else first cell."""
        xs = [x for x, _y in cells]
        ys = [y for _x, y in cells]
        cx, cy = sum(xs) // len(xs), sum(ys) // len(ys)
        return (cx, cy) if (cx, cy) in cells else min(cells)

    def _sw_emit(self, key: str) -> GameAction:
        if key.startswith("6:"):
            x, y = map(int, key[2:].split(","))
            return self._pc_click((x, y), "switch plan click")
        return self._step(GameAction.from_id(int(key)))

    def _switch_policy(
        self, grid: Optional[Grid], latest_frame: FrameData
    ) -> Optional[GameAction]:
        """Remote-toggle switch policy, five-gated at engage time: clicks
        available + 2+ trusted movement rules + no attribute register + no
        voted port color + every available simple action has a rule or
        RULE_TRIES uses, all only after SW_FRAMES diffed frames.  Click-
        only games never form rules, ls20/tr87 expose no clicks, cn04
        votes a port color (and its branch runs first), so only avatar-
        with-buttons games (dc22 family) reach the model.  The probe phase
        clicks each unconfirmed compact signature a capped number of
        times; the plan phase BFS-es the (anchor x switch-phase) product
        graph with signature passability over the predicted grid."""
        if grid is None:
            return None
        avail = set(latest_frame.available_actions or [])
        if GameAction.ACTION6.value not in avail:
            return None
        rules = self._movement_rules()
        if len(rules) < 2:
            return None
        if self._attr_register() is not None \
                or self._port_color() is not None:
            return None
        simple = avail - {GameAction.RESET.value, GameAction.ACTION6.value}
        if not all(a in rules or self._act_uses[a] >= RULE_TRIES
                   for a in simple):
            return None
        if self._frames_diffed < SW_FRAMES:
            return None
        avatar = self._find_avatar(grid)
        if avatar is None:
            return None
        level = latest_frame.levels_completed
        if self._sw_benched == level:
            return None
        if self._sw_strikes >= SW_STRIKES:
            self._sw_benched = level
            return None
        # the budget strip rides along in the search state; observing it
        # from inside this gate keeps ls20/cn04's streams bit-identical
        if self._prev_grid is not None \
                and flood_color(self._prev_grid) == flood_color(grid):
            self._observe_budget(self._prev_grid, grid)
        comps = components(grid)
        confirmed: dict[int, tuple] = {}
        mask_cells: set[Cell] = set()
        for s in self._sw_recs:
            cyc = self._switch_cycle(s)
            if cyc is not None:
                confirmed[s] = cyc
                mask_cells |= set(cyc[0])
        av_cells = set(avatar[1])
        # phase-belief tracking: the phase INDEX is read off the frame
        # (avatar-occluded mask cells are wildcards), the ping-pong
        # DIRECTION is hidden state — forced at the path endpoints, and
        # carried frame-to-frame through observed single-step transitions
        # (riding a growing gate keeps the belief current even though
        # avatar-overlap events are never recorded)
        readable: dict[int, tuple] = {}
        idx_now: dict[int, Optional[int]] = {}
        for s, (cells, ring, nexts, pong) in confirmed.items():
            idx = self._sw_phase(grid, cells, ring, av_cells)
            idx_now[s] = idx
            if idx is None:
                continue  # occluded: keep the old belief untouched
            if not pong:
                self._sw_belief[s] = (idx, 0)
            elif idx == 0:
                self._sw_belief[s] = (idx, 1)
            elif idx == len(ring) - 1:
                self._sw_belief[s] = (idx, -1)
            else:
                old = self._sw_belief.get(s)
                if old is None or (idx != old[0] and abs(idx - old[0]) != 1):
                    self._sw_belief[s] = (idx, None)
                elif idx != old[0]:
                    self._sw_belief[s] = (idx, idx - old[0])
            st = self._sw_belief[s]
            if st[1] is not None:
                readable[s] = (cells, ring, nexts, st)
        swt = tuple(sorted(readable))
        phases = tuple(readable[s][3] for s in swt)
        # 1 — execute the standing plan while the world still matches it;
        # any surprise (a bounce, an unpredicted repaint, a lost phase)
        # voids the plan and replans from the live frame.  Only the
        # OBSERVABLE phase index is validated — the direction bit is the
        # model's own bookkeeping
        if self._sw_plan:
            key, exp_anchor, exp_swt, exp_ph = self._sw_plan[0]
            cur_idx: Optional[tuple] = None
            if all(s in confirmed for s in exp_swt):
                cur_idx = tuple(idx_now.get(s) for s in exp_swt)
            if avatar[0] == exp_anchor \
                    and cur_idx == tuple(p[0] for p in exp_ph):
                self._sw_plan.popleft()
                return self._sw_emit(key)
            self._sw_plan.clear()
            self._sw_strikes += 1
        # a confirmed ping-pong switch whose direction is unreadable (the
        # board sits mid-path with no fresh transition on record): one
        # click both re-reveals the direction and advances the puzzle —
        # but never blind-click a switch whose mask holds our own support
        for s in sorted(confirmed):
            if s in readable or self._sw_probe[s] >= 2 * SW_CYCLE_CAP:
                continue
            cells, ring, _nx, _pong = confirmed[s]
            if idx_now.get(s) is None or av_cells & set(cells):
                continue  # occluded, not direction-blind
            cand = next(
                (c for color, c in comps
                 if shape_signature(color, c) == s and not (c & mask_cells)),
                None)
            if cand is not None:
                self._sw_probe[s] += 1
                self._sw_plan.clear()
                return self._pc_click(self._sw_click_cell(cand),
                                      "re-read switch direction")
        support = self._tile_sig(grid, avatar[1])
        goal_anchors = self._sw_goal_anchors(grid, comps, avatar)
        # carry candidates: stand-on-a-mask-tile-and-click experiments —
        # dc22's paired tiles teleport the avatar across islands, and the
        # only way to find out is to ride one (capped per tile anchor;
        # the fall prune inside the planner keeps these clicks safe)
        carry: dict[Cell, int] = {}
        for s, (cells, _ring, _nx, _ph) in readable.items():
            tele = self._sw_recs.get(s, {}).get("tele", {})
            for blob in pixel_blobs(frozenset(cells)):
                a = min(blob)
                if a not in tele and self._sw_carry[(s, a)] < 2:
                    carry[a] = s
        # 2 — plan over confirmed switches; a route to the exact-overlap
        # goal preempts everything (first-discovery speed is the metric)
        plan: Optional[list] = None
        rank = 0
        if readable and self._sw_nogoal != (self._prev_key, swt, phases):
            plan, rank = self._plan_switch_route(
                grid, avatar, rules, readable, comps, goal_anchors, support,
                carry)
            if plan is None:
                self._sw_nogoal = (self._prev_key, swt, phases)
            elif rank == 4:
                a = plan[-1][1]  # the carry probe's stand-and-click anchor
                self._sw_carry[(carry[a], a)] += 1
        if plan is not None and rank == 1:
            key, _pos, _ph = plan[0]
            self._sw_plan.extend((k, p, swt, ph) for k, p, ph in plan[1:])
            return self._sw_emit(key)
        # 3 — probe: unconfirmed compact signatures, observed-remote-effect
        # candidates first (one more click may close their cycle), then
        # larger components first (buttons are chunky, goals are tiny)
        cands: list[tuple[int, int, int, Cell]] = []
        seen_sigs: set[int] = set(confirmed)
        for color, cells in comps:
            xs = [x for x, _y in cells]
            ys = [y for _x, y in cells]
            if max(xs) - min(xs) >= GROUP_BBOX \
                    or max(ys) - min(ys) >= GROUP_BBOX:
                continue
            if max(ys) < HUD_BAND or min(ys) >= GRID - HUD_BAND \
                    or max(xs) < HUD_BAND or min(xs) >= GRID - HUD_BAND:
                continue  # band-confined: HUD chrome
            if cells & av_cells or cells & mask_cells \
                    or min(cells) in goal_anchors:
                continue  # toggled tiles and goal pads are not causes
            sig = shape_signature(color, cells)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            if self._click_dead(sig):
                continue
            rec = self._sw_recs.get(sig)
            if rec is not None and rec["overlap"]:
                continue
            changed, tries = self._click_effects.get(sig, (0, 0))
            if tries >= 2 and changed == 0:
                continue  # twice tried, never reacted: conserve budget
            cap = SW_CYCLE_CAP if rec and rec["events"] else SW_PROBES
            if self._sw_probe[sig] >= cap:
                continue
            if (sig, support, avatar[0]) in self._fall_bans:
                continue  # this exact context dropped us once already
            has_fx = 1 if rec and rec["events"] else 0
            cands.append((-has_fx, -len(cells), sig,
                          self._sw_click_cell(cells)))
        if cands:
            cands.sort()
            _hf, _sz, sig, cell = cands[0]
            self._sw_probe[sig] += 1
            self._sw_plan.clear()
            return self._pc_click(cell, "probe switch signature")
        if plan is not None:
            key, _pos, _ph = plan[0]
            self._sw_plan.extend((k, p, swt, ph) for k, p, ph in plan[1:])
            return self._sw_emit(key)
        return None

    def _plan_switch_route(
        self, grid: Grid, avatar: tuple[Cell, frozenset[Cell]],
        rules: dict[int, Cell], readable: dict[int, tuple],
        comps: list[tuple[int, frozenset[Cell]]],
        goal_anchors: frozenset[Cell], support: tuple,
        carry: dict[Cell, int],
    ) -> tuple[Optional[list], int]:
        """BFS over the (anchor, switch-phase vector) product graph.
        Movement edges are validated against the PREDICTED grid — the
        current frame with every confirmed switch's mask rewritten to its
        phase patch: a destination footprint signature in _stepped_sigs is
        walkable, one bounced SW_BLOCK_VOTES+ times is blocked, and an
        unknown patterned one is itself a probe goal (entering it grows
        the model).  Click edges advance one switch's phase at
        SW_CLICK_COST budget units (dc22 charges a button press twice),
        pruned by the fall-trap ban and predictively: never click a switch
        whose mask holds the avatar's own support unless the next phase is
        known walkable.  A learned paired-tile carry (dc22's teleporters)
        turns a click into a movement edge.  Budget-dying branches are
        pruned outright.  Goals, best rank wins: 1 — exact anchor overlap
        with a rarest-color compact component (dc22's win test), 2 — an
        unknown footprint probe, 3 — an unvisited anchor, 4 — a stand-on-
        a-mask-tile-and-click carry probe.  Returns (steps, rank) where
        each step is (action key, anchor before, phases before)."""
        anchor0, cells0 = avatar
        av_cells = set(cells0)
        shape = sorted(((x - anchor0[0], y - anchor0[1]) for x, y in cells0),
                       key=lambda d: (d[1], d[0]))  # raster: matches _tile_sig
        self._visited_anchors.add(anchor0)
        counts: Counter[int] = Counter()
        for row in grid:
            counts.update(row)
        background = counts.most_common(1)[0][0]
        swt = tuple(sorted(readable))
        phases0 = tuple(readable[s][3] for s in swt)
        rings: list[list[tuple]] = []
        nexts_l: list[Optional[list[int]]] = []
        sw_cells: list[set[Cell]] = []
        teles: list[dict[Cell, Cell]] = []
        owner: dict[Cell, tuple[int, int]] = {}
        for i, s in enumerate(swt):
            cells, ring, nexts, _ph = readable[s]
            rings.append(ring)
            nexts_l.append(nexts)
            sw_cells.append(set(cells))
            teles.append(self._sw_recs.get(s, {}).get("tele", {}))
            for k, c in enumerate(cells):
                owner.setdefault(c, (i, k))
        all_mask = set(owner)
        click_at: dict[int, Cell] = {}
        for color, cells in comps:
            sig = shape_signature(color, cells)
            if sig in readable and sig not in click_at \
                    and not (cells & all_mask):
                click_at[sig] = self._sw_click_cell(cells)

        def pix(c: Cell, ph: tuple) -> int:
            own = owner.get(c)
            if own is None:
                return grid[c[1]][c[0]]
            i, k = own
            return rings[i][ph[i][0]][k]

        def sig_at(pos: Cell, ph: tuple) -> Optional[tuple]:
            px, py = pos
            out = []
            for dx, dy in shape:
                nx, ny = px + dx, py + dy
                if not (0 <= nx < GRID and 0 <= ny < GRID):
                    return None
                out.append(pix((nx, ny), ph))
            return tuple(out)

        def carry_click(pos: Cell, ph: tuple) -> Optional[tuple]:
            """The stand-here-and-click probe step, when predicted safe."""
            s = carry[pos]
            i = swt.index(s)
            cell = click_at.get(s)
            if cell is None or (s, support, pos) in self._fall_bans:
                return None
            adv = self._sw_adv(len(rings[i]), nexts_l[i], ph[i])
            if adv is None:
                return None  # unknown successor: not a safe experiment
            nph = tuple(adv if j == i else p for j, p in enumerate(ph))
            nsig = sig_at(pos, nph)
            if nsig is None or nsig not in self._stepped_sigs:
                return None  # the toggle would drop our own floor
            return (f"6:{cell[0]},{cell[1]}", pos, ph)

        INF = 1 << 30
        remaining = self._budget_remaining(grid)
        start_left = INF if remaining is None else remaining
        goals: dict[int, list] = {}
        start = (anchor0, phases0)
        if anchor0 in carry:
            probe = carry_click(anchor0, phases0)
            if probe is not None:
                goals[4] = [probe]
        best_left: dict[tuple, int] = {start: start_left}
        dq: deque[tuple[tuple, list, int]] = deque([(start, [], start_left)])
        popped = 0
        while dq and popped < SW_BFS_CAP and 1 not in goals:
            (pos, ph), path, left = dq.popleft()
            popped += 1
            fp = [(pos[0] + dx, pos[1] + dy) for dx, dy in shape]
            for i, s in enumerate(swt):
                cell = click_at.get(s)
                if cell is None:
                    continue
                nleft = left - SW_CLICK_COST
                if nleft < 0:
                    continue  # this branch dies before arriving
                if (s, support, pos) in self._fall_bans:
                    continue
                adv = self._sw_adv(len(rings[i]), nexts_l[i], ph[i])
                if adv is None:
                    continue  # unknown successor from this phase
                nph = tuple(adv if j == i else p for j, p in enumerate(ph))
                npos = teles[i].get(pos, pos)
                if npos != pos:
                    # a learned paired-tile carry: the click moves us
                    nsig = sig_at(npos, nph)
                    if nsig is None or nsig not in self._stepped_sigs:
                        continue  # the far pad isn't known to hold us
                elif any(c in sw_cells[i] for c in fp):
                    nsig = sig_at(pos, nph)
                    if nsig is None or nsig not in self._stepped_sigs:
                        continue  # the toggle would drop our own floor
                state = (npos, nph)
                if nleft <= best_left.get(state, -1):
                    continue
                best_left[state] = nleft
                dq.append((state,
                           path + [(f"6:{cell[0]},{cell[1]}", pos, ph)],
                           nleft))
            for act, (dx, dy) in rules.items():
                nxt = (pos[0] + dx, pos[1] + dy)
                nleft = left - 1
                if nleft < 0:
                    continue
                npath = path + [(str(act), pos, ph)]
                if nxt in goal_anchors and nxt != anchor0:
                    gsig = sig_at(nxt, ph)
                    if gsig is not None \
                            and self._block_sigs[gsig] < SW_BLOCK_VOTES:
                        goals.setdefault(1, npath)
                    continue
                nfp = [(nxt[0] + sx, nxt[1] + sy) for sx, sy in shape]
                if not any(c in av_cells for c in nfp):
                    # (footprints over our own pixels read the avatar, not
                    # the floor: walk them — we are standing there now)
                    sig = sig_at(nxt, ph)
                    if sig is None:
                        continue  # off-frame
                    if self._block_sigs[sig] >= SW_BLOCK_VOTES:
                        continue
                    if sig not in self._stepped_sigs:
                        if any(v != background for v in sig):
                            goals.setdefault(2, npath)
                        continue  # unknown: probe goal or virgin background
                state = (nxt, ph)
                if nleft <= best_left.get(state, -1):
                    continue
                best_left[state] = nleft
                if nxt not in self._visited_anchors:
                    goals.setdefault(3, npath)
                if nxt in carry and 4 not in goals:
                    probe = carry_click(nxt, ph)
                    if probe is not None:
                        goals[4] = npath + [probe]
                dq.append((state, npath, nleft))
        for r in (1, 2, 3, 4):
            if r in goals:
                return goals[r], r
        return None, 0

    # ── slide / node-maze head (tu93 family) ────────────────────────────
    def _sl_lattice(
        self, grid: Grid, av_center: Optional[Cell] = None
    ) -> Optional[tuple[int, int, int, set, int, int]]:
        """Detect a regular binary corridor/wall lattice over the frame.

        Returns (pitch, ox, oy, nodes, cA, cB) where nodes are the lattice
        node anchors on the pitch sublattice and cA/cB are the two board
        colors (one is the passable corridor, the other the wall — which is
        which is learned by the head).  None when the frame isn't a clean
        two-colour repeating lattice (keeps ls20's irregular tiles and
        tr87's glyph cycle out of the head)."""
        counts: Counter[int] = Counter()
        for row in grid:
            counts.update(row)
        if len(counts) < 3:
            return None
        bg = counts.most_common(1)[0][0]
        comps = components(grid)
        # the board: same-shaped square blocks of exactly two colours that
        # tile the frame on one pitch.  Group anchors by (color, shape).
        shape_anchors: dict[tuple, list[Cell]] = defaultdict(list)
        for color, cells in comps:
            if color == bg:
                continue
            mx = min(x for x, _y in cells)
            my = min(y for _x, y in cells)
            w = max(x for x, _y in cells) - mx + 1
            h = max(y for _x, y in cells) - my + 1
            shape_anchors[(color, w, h)].append((mx, my))
        # the two board colours are the two most-prolific same-square-shape
        # block classes that share one block size
        squares = [(len(a), c, w, a) for (c, w, h), a in shape_anchors.items()
                   if w == h and 2 <= w <= 6 and len(a) >= LATTICE_MIN_CELLS]
        if len(squares) < 2:
            return None
        squares.sort(reverse=True)
        # require the top two share the block size (the corridor/wall pair)
        if squares[0][2] != squares[1][2]:
            return None
        blk = squares[0][2]
        cA, cB = squares[0][1], squares[1][1]
        anchors = squares[0][3] + squares[1][3]
        xs = sorted({x for x, _y in anchors})
        ys = sorted({y for _x, y in anchors})
        if len(xs) < 3 or len(ys) < 3:
            return None
        # block pitch = the smallest consistent anchor gap; node pitch is two
        # blocks (corridor + wall straddle), the spacing the avatar steps by
        gaps = Counter(b - a for a, b in zip(xs, xs[1:]) if b > a)
        gaps.update(b - a for a, b in zip(ys, ys[1:]) if b > a)
        if not gaps:
            return None
        blk_pitch, _ = gaps.most_common(1)[0]
        if blk_pitch < blk:
            return None
        pitch = blk_pitch * 2
        # node phase: anchor on the avatar — it always sits EXACTLY on a node,
        # whereas a board-block origin can be off by a block (the avatar
        # itself hides the node block it occupies, and the very first level's
        # board can start mid-pitch).  Span nodes over the whole board extent.
        if av_center is not None:
            ox = av_center[0] % pitch
            oy = av_center[1] % pitch
        else:
            ox, oy = xs[0] % pitch, ys[0] % pitch
        bx0, bx1 = min(xs), max(xs) + blk
        by0, by1 = min(ys), max(ys) + blk
        if av_center is not None:
            bx0 = min(bx0, av_center[0]); bx1 = max(bx1, av_center[0])
            by0 = min(by0, av_center[1]); by1 = max(by1, av_center[1])
        nodes = {(x, y)
                 for x in range(ox, GRID, pitch)
                 for y in range(oy, GRID, pitch)
                 if bx0 - pitch <= x <= bx1 + pitch
                 and by0 - pitch <= y <= by1 + pitch}
        if len(nodes) < LATTICE_MIN_CELLS:
            return None
        return (pitch, ox, oy, nodes, cA, cB)

    def _sl_node(self, cell: Cell, lat: tuple) -> Optional[Cell]:
        """Snap a cell to the nearest ACTUAL lattice node (over the node
        set, by Chebyshev distance).  Rounding alone is fragile when the
        avatar's accent sits at an arbitrary sub-offset inside its body
        (the offset varies by level/sprite); nearest-over-the-set tolerates
        any offset up to a half pitch.  None if no node is within range."""
        pitch, ox, oy, nodes, _cA, _cB = lat
        cx, cy = cell
        best: Optional[tuple[int, Cell]] = None
        for nx, ny in nodes:
            d = max(abs(nx - cx), abs(ny - cy))
            if best is None or d < best[0]:
                best = (d, (nx, ny))
        if best is None or best[0] > pitch:
            return None
        return best[1]

    def _sl_exit_node(
        self, grid: Grid, lat: tuple, av_cells: set
    ) -> Optional[Cell]:
        """The rare-colour goal component (smallest non-board, non-avatar,
        non-budget-bar blob), snapped to its lattice node."""
        pitch, ox, oy, nodes, cA, cB = lat
        counts: Counter[int] = Counter()
        for row in grid:
            counts.update(row)
        bg = counts.most_common(1)[0][0]
        best: Optional[tuple[int, Cell]] = None
        for color, cells in components(grid):
            if color in (bg, cA, cB):
                continue
            cs = set(cells)
            if cs & av_cells:
                continue
            ys = {y for _x, y in cs}
            xs = {x for x, _y in cs}
            # skip wide thin border strips (budget / status bars)
            if len(cs) >= 20 and (len(ys) <= 2 or len(xs) <= 2):
                continue
            if best is None or len(cs) < best[0]:
                best = (len(cs), min(cs), cs)
        if best is None:
            return None
        # snap the exit blob's CENTRE (its min-corner can sit a half-block
        # off a node when the goal sprite isn't block-aligned)
        cs2 = best[2]
        cx = (min(x for x, _y in cs2) + max(x for x, _y in cs2)) // 2
        cy = (min(y for _x, y in cs2) + max(y for _x, y in cs2)) // 2
        node = self._sl_node((cx, cy), lat)
        return node if node is not None and node in nodes else None

    def _sl_avatar_cells(
        self, grid: Grid, anchor: Cell, lat: Optional[tuple] = None
    ) -> set:
        """The avatar's body+accent pixels: components touching the accent,
        EXCLUDING the two board colours when the lattice is known (so a body
        block abutting a wall block doesn't drag the wall in), or just the
        background colour while bootstrapping.  Tight blob → its centre snaps
        to the right node."""
        if lat is not None:
            skip = {lat[4], lat[5]}
        else:
            counts: Counter[int] = Counter()
            for row in grid:
                counts.update(row)
            skip = {counts.most_common(1)[0][0]}
        out: set = set()
        for c, cells in components(grid):
            if c in skip:
                continue
            cs = set(cells)
            if any(abs(x - anchor[0]) <= 1 and abs(y - anchor[1]) <= 1
                   for x, y in cs):
                out |= cs
        return out

    def _sl_body_center(
        self, grid: Grid, anchor: Cell, lat: Optional[tuple] = None
    ) -> Cell:
        """Bounding-box centre of the avatar body — a jitter-proof anchor for
        snapping (the accent's sub-offset varies, the body centre doesn't)."""
        cells = self._sl_avatar_cells(grid, anchor, lat)
        if not cells:
            return anchor
        xs = [x for x, _y in cells]
        ys = [y for _x, y in cells]
        return ((min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2)

    def _sl_route(
        self, grid: Grid, start: Cell, lat: tuple, corridor: int
    ) -> Optional[tuple[list[int], Cell]]:
        """BFS over the node graph (edge in dir d iff the straddle block
        between adjacent nodes is the corridor colour) to the exit node.
        Returns (action list, exit node) or None."""
        pitch, ox, oy, nodes, cA, cB = lat
        av_cells = self._sl_avatar_cells(grid, start, lat)
        # snap the avatar BODY CENTRE (jitter-proof) to its node
        snode = self._sl_node(self._sl_body_center(grid, start, lat), lat)
        exit_node = self._sl_exit_node(grid, lat, av_cells)
        if exit_node is None or snode is None or snode not in nodes:
            return None
        half = pitch // 2

        def passable(nx: int, ny: int, dx: int, dy: int) -> bool:
            sx, sy = nx + dx * half, ny + dy * half
            # sample the straddle block interior (a couple of probes guards
            # against the avatar's accent landing on the sample pixel)
            for ddx in (0, 1):
                for ddy in (0, 1):
                    px, py = sx + ddx, sy + ddy
                    if 0 <= px < GRID and 0 <= py < GRID \
                            and grid[py][px] == corridor:
                        return True
            return False

        # default cardinal mapping, OVERRIDDEN by any confidently-probed
        # action direction: a probe that hit a wall at the start node learns
        # nothing, so its default is retained rather than dropping the edge
        DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        DIRS.update(self._sl_dirmap)
        seen = {snode}
        dq: deque[tuple[Cell, list[int]]] = deque([(snode, [])])
        while dq:
            pos, path = dq.popleft()
            if pos == exit_node:
                return (path, exit_node)
            for a, (dx, dy) in DIRS.items():
                if dx == dy == 0:
                    continue
                if passable(pos[0], pos[1], dx, dy):
                    nxt = (pos[0] + dx * pitch, pos[1] + dy * pitch)
                    if nxt in nodes and nxt not in seen:
                        seen.add(nxt)
                        dq.append((nxt, path + [a]))
        return None

    def _slide_policy(
        self, grid: Grid, latest_frame: FrameData
    ) -> Optional[GameAction]:
        """Directional node-maze solver (tu93 family).  Gate (load-bearing
        beyond the [1,2,3,4] action set it shares with ls20/tr87): a clean
        binary corridor/wall lattice, a unique tiny accent avatar snapped to
        a node, and a rare-colour exit node.  A <=4-action directional probe
        learns each action's unit step (replacing the 500-step warmup); then
        BFS over the node graph routes to the exit.  The corridor colour (vs
        wall) is a self-correcting hypothesis: a planned step that the avatar
        DIDN'T take (zero slide-vote) flips it.  Strikes bench the level."""
        avail = set(latest_frame.available_actions or []) \
            - {GameAction.RESET.value}
        # action-set gate: movement-only, no clicks
        if not avail or not avail <= {1, 2, 3, 4}:
            return None
        level = latest_frame.levels_completed
        if self._sl_benched == level:
            return None
        # structural gate
        accent = self._sl_avatar_anchor(grid)
        if accent is None:
            return None
        av_anchor = accent[0]
        # anchor the node phase on the avatar (it always sits on a node)
        av_center = self._sl_body_center(grid, av_anchor)
        lat = self._sl_lattice(grid, av_center)
        if lat is None:
            return None
        pitch, ox, oy, nodes, cA, cB = lat
        av_cells = self._sl_avatar_cells(grid, av_anchor, lat)
        snode = self._sl_node(self._sl_body_center(grid, av_anchor, lat), lat)
        if snode is None or snode not in nodes:
            return None
        exit_node = self._sl_exit_node(grid, lat, av_cells)
        if exit_node is None:
            return None

        # learn the corridor colour: the board colour that, as a straddle,
        # the avatar has been observed to cross.  Bootstrap: the colour whose
        # straddles immediately surround the avatar's node is the corridor (a
        # corridor must touch the start, walls box it in).  Default to the
        # board colour with more straddle cells adjacent to the start node.
        if self._sl_corridor is None:
            self._sl_corridor = self._sl_guess_corridor(grid, snode, lat)
        corridor = self._sl_corridor

        # directional probe: learn each action's unit step before routing, so
        # the node-graph plan is expressed in the engine's real directions.
        # one probe per action; the probe NEVER strikes (it only spends the
        # <=4 opening actions and reads the result from _slide_votes).
        self._sl_sync_dirmap(lat)
        for a in sorted(avail):
            if a not in self._sl_dirmap and self._sl_probe[a] < SL_PROBE_CAP:
                self._sl_probe[a] += 1
                return self._step(GameAction.from_id(a))

        self._sl_engaged = True
        # RE-PLAN every step (a 36-node BFS is cheap): blind plan replay
        # desyncs the instant any edge reading is off, so route fresh from
        # the avatar's current node and take only the first action.  This is
        # self-correcting across jitter and mis-classified corridors.
        route = self._sl_route(grid, av_anchor, lat, corridor)
        if route is None:
            # the corridor-colour hypothesis may be inverted on this board:
            # try the alternate once and adopt it if it routes
            alt = cB if corridor == cA else cA
            route = self._sl_route(grid, av_anchor, lat, alt)
            if route is not None:
                self._sl_corridor = alt
        if route is None or not route[0]:
            self._sl_strikes += 1
            if self._sl_strikes >= SL_STRIKES:
                self._sl_benched = level
            return None
        return self._step(GameAction.from_id(route[0][0]))

    def _sl_guess_corridor(self, grid: Grid, snode: Cell, lat: tuple) -> int:
        """The board colour that appears as a straddle adjacent to the start
        node more often (a corridor must connect the avatar to the maze)."""
        pitch, ox, oy, nodes, cA, cB = lat
        half = pitch // 2
        tally: Counter[int] = Counter()
        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            sx, sy = snode[0] + dx * half, snode[1] + dy * half
            for ddx in (0, 1):
                for ddy in (0, 1):
                    px, py = sx + ddx, sy + ddy
                    if 0 <= px < GRID and 0 <= py < GRID:
                        v = grid[py][px]
                        if v in (cA, cB):
                            tally[v] += 1
        if not tally:
            return cA
        return tally.most_common(1)[0][0]

    def _sl_sync_dirmap(self, lat: tuple) -> None:
        """Resolve each action's unit step from its dominant non-zero slide
        vote.  The accent pixel jitters +/-1 across the minor axis as the
        body re-renders, so QUANTIZE the displacement to node cells (round by
        pitch) and keep only the dominant axis — that strips the jitter and
        yields a clean cardinal unit step (one node per action)."""
        pitch = lat[0]
        for a, votes in self._slide_votes.items():
            nz = [(n, d) for d, n in votes.items() if d != (0, 0)]
            if not nz:
                continue
            nz.sort(reverse=True)
            _n, (ddx, ddy) = nz[0]
            cx = round(ddx / pitch)
            cy = round(ddy / pitch)
            if abs(cx) >= abs(cy):
                cy = 0
            else:
                cx = 0
            sx = (cx > 0) - (cx < 0)
            sy = (cy > 0) - (cy < 0)
            if (sx, sy) != (0, 0):
                self._sl_dirmap[a] = (sx, sy)

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

    def _click_dead(self, sig: int) -> bool:
        """A signature proven non-reactive: CLICK_PROBES+ tries without a
        single effect, or twice that many tries with a Laplace P(effect)
        still under 2*CLICK_DEAD (absorbs effects faked by pre-HUD-mask
        chrome ticks in the first ~30 frames).  Appearance is physics, so
        deadness transfers across states, levels and lives — this is the
        cross-state pruning that conserves per-life click budgets."""
        changed, tries = self._click_effects.get(sig, (0, 0))
        if tries >= CLICK_PROBES and changed == 0:
            return True
        return tries >= 2 * CLICK_PROBES \
            and (changed + 1) / (tries + 2) < 2 * CLICK_DEAD

    def _afford_rank(
        self, grid: Grid,
        untried: list[tuple[str, GameAction, Optional[Cell]]],
    ) -> list[tuple[str, GameAction, Optional[Cell]]]:
        """Reorder untried clicks by the affordance library.  STABLE sort
        on descending optimistic score (changed+2)/(tries+3) — Laplace
        with one virtual success.  On a fresh game every target scores the
        2/3 prior, every score ties, and the sort is the identity, so the
        rarest-color enumeration that empirically cracks lp85 is preserved
        bit-for-bit.  Once the library has data the order is: reliably-
        productive classes (score > 2/3 — the repeated piece-clicks that
        drive translation puzzles), then unseen classes at the prior
        (optimism: a class that just APPEARED is often the thing our
        actions unlocked), then uncertain classes, and proven-dead
        signatures (_click_dead) drop out entirely — that's what conserves
        per-life click budgets.  Non-click options never move slots."""
        comps = components(grid)
        scored: list[tuple[float, tuple[str, GameAction, Optional[Cell]]]] = []
        for opt in untried:
            if opt[2] is None:
                continue
            sig = signature_under(comps, opt[2])
            if self._click_dead(sig):
                continue  # proven dead: stop spending budget on it
            changed, tries = self._click_effects.get(sig, (0, 0))
            score = (changed + 2) / (tries + 3)  # one virtual success
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

    def _use_balance(
        self, untried: list[tuple[str, GameAction, Optional[Cell]]],
    ) -> list[tuple[str, GameAction, Optional[Cell]]]:
        """Stable-reorder untried SIMPLE actions by ascending global use
        count; click options never move slots (the mirror image of
        _afford_rank).  Per-state enumeration tries ACTION1 first in every
        fresh state, which starves the later actions of the consistent
        observations their movement rules need (measured on ls20: uses
        {1:136, 2:102, 3:15, 4:7}, rule 4 untrusted until action 262 —
        and the attr/port planners gate on the full rule set).  Sorting by
        global use evens the sampling so all rules trust within ~30
        actions.  On a fresh game all counts are zero and the stable sort
        is the identity: untried[0] is unchanged."""
        simples = sorted(
            (o for o in untried if o[2] is None),
            key=lambda o: self._act_uses[int(o[0])])
        fill = iter(simples)
        return [o if o[2] is not None else next(fill) for o in untried]

    # ── graph explorer (CURIO_EXPLORER=graph) ───────────────────────────
    def _gx_options(
        self, grid: Optional[Grid], avail: set[int]
    ) -> list[tuple[str, GameAction, Optional[Cell]]]:
        """Candidate (action-key, action, coords) for a state — same
        primitives and key format as _novelty_policy so _tried/_transitions
        interoperate.  ACTION6 enumerates over the explorer's own tamed
        target set (component-centroid + signature-dedup)."""
        options: list[tuple[str, GameAction, Optional[Cell]]] = []
        for action in GameAction:
            if action is GameAction.RESET:
                continue
            if avail and action.value not in avail:
                continue
            if action.is_complex():
                for x, y in self._gx_click_targets(grid):
                    options.append((f"6:{x},{y}", action, (x, y)))
            else:
                options.append((str(action.value), action, None))
        return options

    def _gx_tier(
        self, grid: Grid, comps: list[tuple[int, frozenset[Cell]]],
        opt: tuple[str, GameAction, Optional[Cell]]
    ) -> int:
        """Salience tier for an untried option (higher = explore first).
        -1 dead (excluded); 0 known-rule moves / background clicks;
        1 simple actions still lacking a trusted rule (cheap, high info);
        2 clicks with positive Laplace evidence; 3 clicks whose signature
        is globally unseen and salient (a just-appeared novel sprite)."""
        akey, _action, coords = opt
        if coords is None:                       # simple action
            act = int(akey)
            rules = self._movement_rules()
            if act in rules:
                return 0                         # trusted rule: low info
            if self._act_uses[act] < RULE_TRIES:
                return 1                         # still learning: high info
            return 0
        sig = signature_under(comps, coords)
        if sig == 0:
            return 0                             # background click
        if self._click_dead(sig) or sig in self._gx_lethal_sig:
            return -1                            # proven dead / lethal: exclude
        changed, tries = self._click_effects.get(sig, (0, 0))
        if tries == 0 and sig not in self._gx_global_tried_sig:
            return 3                             # unseen salient class: top
        if tries > 0 and (changed + 1) / (tries + 2) > (1.0 / 2.0):
            return 2                             # known-productive class
        return 0

    def _gx_node(
        self, key: int, grid: Optional[Grid], avail: set[int]
    ) -> dict[str, Any]:
        """Build (once) and return the node record for a masked-frame hash.
        Idempotent within a level: a state's geography/candidate set is
        stable, so a re-visit reuses the cached record (only its untried set
        is pruned live against _tried)."""
        node = self._gx_nodes.get(key)
        if node is None:
            options = self._gx_options(grid, avail)
            comps = components(grid) if grid is not None else []
            salience: dict[str, int] = {}
            actions: list[str] = []
            optmap: dict[str, tuple[str, GameAction, Optional[Cell]]] = {}
            sigmap: dict[str, int] = {}
            for opt in options:
                tier = self._gx_tier(grid, comps, opt) if grid is not None else 0
                if tier < 0:
                    continue                     # dead signature: never offer
                actions.append(opt[0])
                salience[opt[0]] = tier
                optmap[opt[0]] = opt
                if opt[2] is not None:
                    sigmap[opt[0]] = signature_under(comps, opt[2])
            node = {"actions": actions, "salience": salience,
                    "optmap": optmap, "sig": sigmap}
            self._gx_nodes[key] = node
        return node

    def _gx_untried(self, key: int, node: dict[str, Any]) -> list[str]:
        """Action-keys at this node not yet tried (per-state) and not
        globally exhausted by signature; sorted highest-salience first,
        ties lexicographic (deterministic BFS/route reproduction)."""
        tried = self._tried[key]
        out: list[str] = []
        for akey in node["actions"]:
            if akey in tried:
                continue
            opt = node["optmap"][akey]
            if opt[2] is not None:
                sig = node.get("sig", {}).get(akey)
                # cross-state pruning: a click whose class is globally
                # exhausted (proven dead anywhere) or proven lethal is never
                # re-offered
                if sig is not None and (
                        sig in self._gx_global_tried_sig
                        or self._click_dead(sig)
                        or sig in self._gx_lethal_sig):
                    continue
            out.append(akey)
        out.sort(key=lambda a: (-node["salience"].get(a, 0), a))
        return out

    def _gx_bfs(self, cur: int) -> Optional[tuple[list[str], int]]:
        """Shortest path of action-keys over observed edges (_transitions)
        from cur to the nearest node owning an untried action, maximizing
        that node's best untried salience tier (a small per-tier bonus
        subtracts from path cost so a tier-3 target beats a tier-1 one at
        equal-ish distance).  Deterministic: edges iterated by lexicographic
        action-key.  Bounded by LATTICE_BFS_CAP expansions."""
        # adjacency restricted to current-level nodes (_gx_nodes is cleared
        # per level), built once: keeps BFS off stale cross-level geography
        # and turns the per-expansion edge scan into an O(1) dict lookup
        adj: dict[int, list[str]] = defaultdict(list)
        for (s, akey), dest in self._transitions.items():
            if s in self._gx_nodes:
                adj[s].append(akey)
        for s in adj:
            adj[s].sort()                        # deterministic expansion order
        best: Optional[tuple[float, list[str], int]] = None
        seen = {cur}
        dq: deque[tuple[int, list[str]]] = deque([(cur, [])])
        expansions = 0
        while dq and expansions < LATTICE_BFS_CAP:
            node_key, path = dq.popleft()
            expansions += 1
            if node_key != cur and node_key in self._gx_nodes:
                ut = self._gx_untried(node_key, self._gx_nodes[node_key])
                if ut:
                    tier = self._gx_nodes[node_key]["salience"].get(ut[0], 0)
                    cost = len(path) - 0.25 * tier  # salience bonus
                    if best is None or cost < best[0]:
                        best = (cost, path, node_key)
            for akey in adj.get(node_key, ()):
                dest = self._transitions.get((node_key, akey))
                if dest is None or dest in seen:
                    continue
                seen.add(dest)
                dq.append((dest, path + [akey]))
        if best is None:
            return None
        return (best[1], best[2])

    def _gx_emit(
        self, key: int, choice: tuple[str, GameAction, Optional[Cell]]
    ) -> GameAction:
        """Mark an option tried (per-state + global-signature) and emit it."""
        akey, action, coords = choice
        self._tried[key].add(akey)
        if coords is not None:
            action.set_data({"x": coords[0], "y": coords[1]})
            action.reasoning = {"why": f"graph-explore object at {coords}"}
        else:
            action.reasoning = {"why": f"graph-explore action {action.value}"}
        self._prev_action = akey
        return action

    def _graph_explore_policy(
        self, grid: Optional[Grid], latest_frame: FrameData
    ) -> GameAction:
        """Salience-tiered global state-graph explorer.  Drop-in for the
        novelty fallback when CURIO_EXPLORER=graph.  Local greedy choice is
        a strict superset of v7 (pick the highest-salience untried action);
        when the current state is exhausted, BFS the observed-edge graph to
        the nearest high-salience frontier and walk a cached route there;
        if the reachable graph has no frontier, backtrack via RESET to the
        level start; if even that is spent, defer to the v7 revisit-ranker."""
        key = self._masked_hash(grid)
        avail = set(latest_frame.available_actions or [])
        # the HUD mask must be frozen for node identities to be stable;
        # before that, defer to legacy novelty to avoid phantom edges
        if self._frames_diffed < HUD_FREEZE and not self._hud_mask:
            return self._novelty_body(grid, latest_frame)
        node = self._gx_node(key, grid, avail)
        if not node["actions"]:
            self._prev_action = "0"
            return GameAction.RESET

        # 1. LOCAL FRONTIER: highest-salience untried action in this state.
        untried = self._gx_untried(key, node)
        if untried:
            self._gx_route.clear()
            self._gx_route_dest = None
            if self._steps_since_novelty > STUCK_LIMIT:
                akey = self._rng.choice(untried)
            else:
                # tie-break within the top tier by affordance / use-balance
                akey = self._gx_pick_top(grid, node, untried)
            return self._gx_emit(key, node["optmap"][akey])

        # 2. CACHED ROUTE to a distant frontier (re-plan on drift/empty).
        if self._gx_route and self._gx_route_dest is not None:
            nxt = self._gx_route[0]
            exp_dest = self._transitions.get((key, nxt))
            dnode = self._gx_nodes.get(self._gx_route_dest)
            dest_live = dnode is not None and bool(
                self._gx_untried(self._gx_route_dest, dnode))
            if exp_dest is not None and dest_live:
                self._gx_route.popleft()
                return self._gx_emit(key, node["optmap"].get(
                    nxt, (nxt, *self._gx_key_to_action(nxt))))
            self._gx_route.clear()
            self._gx_route_dest = None

        # 3. RE-PLAN: BFS to nearest untried frontier.
        plan = self._gx_bfs(key)
        if plan is not None:
            path, dest = plan
            if path:
                self._gx_route = deque(path)
                self._gx_route_dest = dest
                nxt = self._gx_route.popleft()
                return self._gx_emit(key, node["optmap"].get(
                    nxt, (nxt, *self._gx_key_to_action(nxt))))

        # 4. No reachable frontier in the current graph: RESET-backtrack to
        # the level start (last resort, bounded) so exploration resumes from
        # a known node instead of wandering by 1-step lookahead.  RESET is
        # only worthwhile when we are NOT already at the start and the start
        # node still has frontier to reach.
        if avail and GameAction.RESET.value in avail \
                and self._gx_start is not None and key != self._gx_start \
                and self._gx_resets < GX_RESET_CAP:
            start_node = self._gx_nodes.get(self._gx_start)
            start_has_frontier = start_node is None or bool(
                self._gx_untried(self._gx_start, start_node))
            if start_has_frontier:
                self._gx_resets += 1
                self._gx_route.clear()
                self._gx_route_dest = None
                self._tried[key].add("0")        # this RESET edge is spent
                self._gx_pending_reset = (key, "0")
                action = GameAction.RESET
                action.reasoning = {"why": "graph-explore RESET-backtrack"}
                self._prev_action = "0"
                return action

        # 5. Graph genuinely exhausted (or RESET unavailable / capped):
        # defer to the v7 revisit-ranker so behavior degrades gracefully.
        return self._novelty_body(grid, latest_frame)

    def _gx_key_to_action(
        self, akey: str
    ) -> tuple[GameAction, Optional[Cell]]:
        """Reconstruct (action, coords) from an action-key for walking a
        route edge whose action isn't in the current node's optmap."""
        if akey.startswith("6:"):
            x, y = map(int, akey[2:].split(","))
            return (GameAction.ACTION6, (x, y))
        return (GameAction.from_id(int(akey)), None)

    def _gx_pick_top(
        self, grid: Optional[Grid], node: dict[str, Any], untried: list[str]
    ) -> str:
        """Pick within the highest-salience tier, tie-broken by the legacy
        affordance/use-balance ordering so intra-tier behavior matches the
        proven heuristics."""
        if grid is None:
            return untried[0]
        top_tier = node["salience"].get(untried[0], 0)
        top = [a for a in untried if node["salience"].get(a, 0) == top_tier]
        if len(top) == 1:
            return top[0]
        opts = [node["optmap"][a] for a in top]
        # any clicks in the tier: rank by affordance; else balance simples.
        # _afford_rank drops click options whose LIVE signature is dead (the
        # grid can differ from the node's first-visit frame under HUD masking),
        # so the reordered list may be shorter or empty — fall back to top[0].
        if any(o[2] is not None for o in opts):
            opts = self._afford_rank(grid, opts)
        opts = self._use_balance(opts)
        return opts[0][0] if opts else top[0]

    def _gx_click_targets(self, grid: Optional[Grid]) -> list[Cell]:
        """Tamed ACTION6 candidates for the graph explorer: ONE click per
        connected-component centroid (snapped onto the component when the raw
        centroid falls off-color), deduped to at most one representative per
        distinct shape_signature per state (we only need to learn what a
        CLASS does — that transfers), and pruned of dead / globally-exhausted
        signatures.  This collapses a grid of identical tiles from dozens of
        redundant clicks to a handful of class-reps, shrinking per-node
        branching so the BFS frontier stays tractable.  Ordering preserves
        rarest-color / smallest-component first (smallest comps first), so the
        salience tiering lines up with the legacy enumeration's intuition.

        Separate from _click_targets so legacy novelty / lp85's exact
        rarest-color enumeration is untouched."""
        if grid is None:
            return [(32, 32)]
        comps = components(grid)
        if not comps:
            return [(32, 32)]
        # color rarity: rarest colors first (matches _click_targets intent)
        color_count: Counter[int] = Counter(c for c, _cells in comps)
        ordered = sorted(
            comps, key=lambda c: (color_count[c[0]], len(c[1])))
        out: list[Cell] = []
        seen_sig: set[int] = set()
        for color, cells in ordered:
            sig = shape_signature(color, cells)
            if sig in seen_sig:
                continue                          # one rep per class per state
            seen_sig.add(sig)
            if sig in self._gx_global_tried_sig or self._click_dead(sig) \
                    or sig in self._gx_lethal_sig:
                continue                          # spent/dead/lethal: skip
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            cx, cy = sum(xs) // len(xs), sum(ys) // len(ys)
            if (cx, cy) not in cells:
                # centroid fell off the component (concave/ring shapes):
                # snap to the nearest on-component cell (deterministic min)
                cx, cy = min(cells, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
            out.append((cx, cy))
        # dedup coords, keep order
        s: set[Cell] = set()
        deduped = [t for t in out if not (t in s or s.add(t))]
        return deduped or [(32, 32)]

    def _novelty_body(
        self, grid: Optional[Grid], latest_frame: FrameData
    ) -> GameAction:
        """The verbatim v7 novelty fallback, callable when the graph
        explorer defers (graph not yet armed, or no reachable frontier)."""
        return self._novelty_policy_impl(grid, latest_frame)

    def _novelty_policy(
        self, grid: Optional[Grid], latest_frame: FrameData
    ) -> GameAction:
        if self._gx_on:
            # all four _policy fallthroughs route here; one early-return keeps
            # the call sites untouched, and with the toggle off this branch
            # never executes (the impl below is the verbatim v7 fallback)
            return self._graph_explore_policy(grid, latest_frame)
        return self._novelty_policy_impl(grid, latest_frame)

    def _novelty_policy_impl(
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
        if untried and grid is not None \
                and (self._game_overs
                     or any(t >= VOTE_THRESHOLD
                            for _c, t in self._click_effects.values())):
            # rank once the library holds REAL data: a death, or any
            # signature sampled VOTE_THRESHOLD+ times (the click analog of
            # trusting a movement rule).  Before that the ranking would be
            # the identity anyway — except for 1-2-try noise, and lp85's
            # empirical first-life enumeration is too valuable to shuffle
            untried = self._afford_rank(grid, untried)
        if untried:
            untried = self._use_balance(untried)
        if untried:
            choice = (self._rng.choice(untried)
                      if self._steps_since_novelty > STUCK_LIMIT else untried[0])
        else:
            # all options tried in this state: revisit, but never spend a
            # click on a signature the library has proven dead
            if grid is not None and any(o[2] is not None for o in options):
                comps = components(grid)
                live = [o for o in options if o[2] is None
                        or not self._click_dead(
                            signature_under(comps, o[2]))]
                options = live or options

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
