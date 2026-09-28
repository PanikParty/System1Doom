"""Grid navigation graph over a Doom map.

Nodes: grid points (GRID units apart) inside a sector with >= RADIUS clearance from walls.
Edges: 8-neighbour, DIRECTED (step up <= 24, any drop down), headroom >= 56, no blocking
line crossed. Doors and lifts are evaluated with best-case heights and marked as *gated*
so the runtime can check their current state. Teleport lines add long-range edges.
"""
from __future__ import annotations

import heapq
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from s1doom.mapmodel import specials as SP
from s1doom.mapmodel.wadparse import MapData

GRID = 32.0
RADIUS = 15.0
MAX_STEP = 24.0
MIN_HEADROOM = 56.0
BUCKET = 128.0
CACHE_DIR = Path(__file__).resolve().parents[3] / "cache"
VERSION = 5
SAMPLES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
# Doom geometry sits on integer (usually 8/16/32-aligned) coordinates; an off-integer grid
# offset keeps nodes and edges from lying exactly on linedefs (missed crossings).
GRID_OFFSET = 7.31


@dataclass
class Edge:
    to: int
    cost: float
    gate: int = -1          # sector index whose height must be checked at runtime
    teleport: int = -1      # line index for teleport edges
    use_line: int = -1      # manual door / lift line to press USE on


@dataclass
class NavGraph:
    map_name: str
    xs: np.ndarray
    ys: np.ndarray
    sector: np.ndarray
    adj: list[list[Edge]]
    cell: dict[tuple[int, int], int]
    origin: tuple[float, float]
    dynamic: dict[int, str] = field(default_factory=dict)   # sector -> "door"|"lift"|"remote_door"
    door_lines: dict[int, list[int]] = field(default_factory=dict)  # sector -> manual door lines
    damage: set[int] = field(default_factory=set)
    version: int = VERSION

    # -- lookup -----------------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.xs)

    def pos(self, i: int) -> tuple[float, float]:
        return float(self.xs[i]), float(self.ys[i])

    def nearest(self, x: float, y: float, max_cells: int = 4) -> int | None:
        ox, oy = self.origin
        cx, cy = round((x - ox) / GRID), round((y - oy) / GRID)
        best, bd = None, 1e18
        for r in range(max_cells + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    i = self.cell.get((cx + dx, cy + dy))
                    if i is not None and self.adj[i]:
                        d = (self.xs[i] - x) ** 2 + (self.ys[i] - y) ** 2
                        if d < bd:
                            best, bd = i, d
            if best is not None and r >= 1:
                return best
        return best

    # -- search -----------------------------------------------------------
    def astar(self, start: int, goal: int, edge_cost=None, max_expand: int = 200_000) -> list[int] | None:
        gx, gy = self.xs[goal], self.ys[goal]
        openh = [(0.0, start)]
        g = {start: 0.0}
        came: dict[int, int] = {}
        closed = set()
        while openh and len(closed) < max_expand:
            _, u = heapq.heappop(openh)
            if u == goal:
                path = [u]
                while u in came:
                    u = came[u]
                    path.append(u)
                return path[::-1]
            if u in closed:
                continue
            closed.add(u)
            for e in self.adj[u]:
                c = e.cost if edge_cost is None else edge_cost(u, e)
                if not math.isfinite(c):
                    continue
                ng = g[u] + c
                if ng < g.get(e.to, 1e18):
                    g[e.to] = ng
                    came[e.to] = u
                    h = math.hypot(self.xs[e.to] - gx, self.ys[e.to] - gy)
                    heapq.heappush(openh, (ng + h, e.to))
        return None

    def dijkstra(self, start: int, edge_cost=None, limit: float = 1e18) -> dict[int, float]:
        dist = {start: 0.0}
        h = [(0.0, start)]
        while h:
            d, u = heapq.heappop(h)
            if d > dist.get(u, 1e18) or d > limit:
                continue
            for e in self.adj[u]:
                c = e.cost if edge_cost is None else edge_cost(u, e)
                if not math.isfinite(c):
                    continue
                nd = d + c
                if nd < dist.get(e.to, 1e18):
                    dist[e.to] = nd
                    heapq.heappush(h, (nd, e.to))
        return dist

    def edge(self, u: int, v: int) -> Edge | None:
        return next((e for e in self.adj[u] if e.to == v), None)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------

def _point_sectors(m: MapData, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Sector index for each point (-1 outside), even-odd test per sector."""
    by_sec: dict[int, list[int]] = defaultdict(list)
    for l in m.lines:
        if l.front == l.back:
            continue
        if l.front >= 0:
            by_sec[l.front].append(l.idx)
        if l.back >= 0:
            by_sec[l.back].append(l.idx)
    out = np.full(len(px), -1, dtype=np.int32)
    best_area = np.full(len(px), np.inf)
    for s, idxs in by_sec.items():
        seg = m.seg[idxs]
        x0, x1 = min(seg[:, 0].min(), seg[:, 2].min()), max(seg[:, 0].max(), seg[:, 2].max())
        y0, y1 = min(seg[:, 1].min(), seg[:, 3].min()), max(seg[:, 1].max(), seg[:, 3].max())
        cand = np.nonzero((px >= x0) & (px <= x1) & (py >= y0) & (py <= y1))[0]
        if len(cand) == 0:
            continue
        cx, cy = px[cand][:, None], py[cand][:, None]
        ax, ay, bx, by = seg[:, 0][None], seg[:, 1][None], seg[:, 2][None], seg[:, 3][None]
        straddle = (ay > cy) != (by > cy)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = ax + (cy - ay) * (bx - ax) / (by - ay)
        inside = (np.sum(straddle & (cx < xint), axis=1) % 2) == 1
        area = (x1 - x0) * (y1 - y0)
        sel = cand[inside]
        better = area < best_area[sel]
        out[sel[better]] = s
        best_area[sel[better]] = area
    return out


def _min_wall_dist(m: MapData, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    seg = m.seg[m.blocking_mask]
    out = np.full(len(px), np.inf)
    for i in range(0, len(px), 2048):
        x = px[i:i + 2048][:, None]
        y = py[i:i + 2048][:, None]
        ax, ay, bx, by = seg[:, 0][None], seg[:, 1][None], seg[:, 2][None], seg[:, 3][None]
        dx, dy = bx - ax, by - ay
        L2 = np.maximum(dx * dx + dy * dy, 1e-9)
        t = np.clip(((x - ax) * dx + (y - ay) * dy) / L2, 0, 1)
        d = np.hypot(x - (ax + t * dx), y - (ay + t * dy))
        out[i:i + 2048] = d.min(axis=1)
    return out


def _effective_heights(m: MapData) -> tuple[np.ndarray, np.ndarray, dict[int, str], dict[int, list[int]]]:
    """Best-case floor/ceil for movable sectors, plus which sectors are dynamic."""
    floor = np.array([s.floor for s in m.sectors])
    ceil = np.array([s.ceil for s in m.sectors])
    neighbours: dict[int, set[int]] = defaultdict(set)
    for l in m.lines:
        if l.two_sided and l.front != l.back:
            neighbours[l.front].add(l.back)
            neighbours[l.back].add(l.front)
    dyn: dict[int, str] = {}
    door_lines: dict[int, list[int]] = defaultdict(list)
    for l in m.lines:
        if l.kind == "door" and l.back >= 0:
            dyn[l.back] = "door"
            door_lines[l.back].append(l.idx)
        elif l.kind == "remote_door":
            for s in m.tagged_sectors(l.tag):
                dyn.setdefault(s, "remote_door")
        elif l.kind == "lift":
            for s in m.tagged_sectors(l.tag):
                dyn[s] = "lift"
            if l.back >= 0 and SP.is_use(l.special) and l.tag == 0:
                dyn[l.back] = "lift"
    ef, ec = floor.copy(), ceil.copy()
    for s, kind in dyn.items():
        nb = neighbours.get(s) or set()
        if not nb:
            continue
        if kind in ("door", "remote_door"):
            ec[s] = max(ceil[s], min(ceil[n] for n in nb) - 4)
        elif kind == "lift":
            ef[s] = min(floor[s], min(floor[n] for n in nb))
    return ef, ec, dyn, dict(door_lines)


def _seg_cross(ax, ay, bx, by, seg):
    """Return (mask, t) for lines crossed by segment a->b (proper crossing)."""
    cx, cy, dx, dy = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    den = rx * sy - ry * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((cx - ax) * sy - (cy - ay) * sx) / den
        u = ((cx - ax) * ry - (cy - ay) * rx) / den
    mask = (np.abs(den) > 1e-9) & (t >= -1e-9) & (t <= 1 + 1e-9) & (u >= -1e-9) & (u <= 1 + 1e-9)
    return mask, t


def _near_lines(x: float, y: float, seg: np.ndarray, r: float) -> np.ndarray:
    ax, ay, bx, by = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / L2, 0, 1)
    return np.nonzero(np.hypot(x - (ax + t * dx), y - (ay + t * dy)) < r)[0]


def build_navgraph(m: MapData) -> NavGraph:
    x0, y0, x1, y1 = m.bounds
    ox, oy = math.floor(x0 / GRID) * GRID + GRID_OFFSET, math.floor(y0 / GRID) * GRID + GRID_OFFSET
    nx, ny = int((x1 - ox) / GRID) + 2, int((y1 - oy) / GRID) + 2
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    px, py = ox + gx * GRID, oy + gy * GRID
    sec = _point_sectors(m, px.astype(float), py.astype(float))
    ok = sec >= 0
    px, py, gx, gy, sec = px[ok], py[ok], gx[ok], gy[ok], sec[ok]
    clear = _min_wall_dist(m, px.astype(float), py.astype(float)) >= RADIUS
    ef, ec, dyn, door_lines = _effective_heights(m)
    # also drop nodes where (best-case) the sector can't hold the player
    fits = (ec[sec] - ef[sec]) >= MIN_HEADROOM
    keep = clear & fits
    px, py, gx, gy, sec = px[keep], py[keep], gx[keep], gy[keep], sec[keep]
    n = len(px)
    cell = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(gx, gy))}
    real_floor = np.array([s.floor for s in m.sectors])
    real_ceil = np.array([s.ceil for s in m.sectors])
    damage = {s.idx for s in m.sectors if s.special in SP.DAMAGE_SECTORS}

    # bucket index of lines
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for l in m.lines:
        bx0, bx1 = sorted((l.x1, l.x2))
        by0, by1 = sorted((l.y1, l.y2))
        for bi in range(int(math.floor((bx0 - GRID) / BUCKET)), int(math.floor((bx1 + GRID) / BUCKET)) + 1):
            for bj in range(int(math.floor((by0 - GRID) / BUCKET)), int(math.floor((by1 + GRID) / BUCKET)) + 1):
                buckets[(bi, bj)].append(l.idx)

    adj: list[list[Edge]] = [[] for _ in range(n)]
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for i in range(n):
        ax, ay, sa = float(px[i]), float(py[i]), int(sec[i])
        cand = buckets.get((int(math.floor(ax / BUCKET)), int(math.floor(ay / BUCKET))), [])
        cseg = m.seg[cand] if cand else np.zeros((0, 4))
        for dx, dy in dirs:
            j = cell.get((int(gx[i]) + dx, int(gy[i]) + dy))
            if j is None:
                continue
            bx, by = float(px[j]), float(py[j])
            mask, t = _seg_cross(ax, ay, bx, by, cseg)
            hit = sorted((t[k], cand[k]) for k in np.nonzero(mask)[0])
            if any(m.lines[li].blocking for _, li in hit):
                continue
            # sector sequence along the segment
            seq_t, seq_s, cur = [0.0], [sa], sa
            for tt, li in hit:
                L = m.lines[li]
                cur = L.back if cur == L.front else L.front
                seq_t.append(tt)
                seq_s.append(cur)
            # Doom physics: the player's 32-unit box stands on the HIGHEST floor and under the
            # LOWEST ceiling of every sector it touches, so thin gaps/cracks are walkable.
            valid, gate, use_line = True, -1, -1
            prev_fb = prev_fr = None
            for ts in SAMPLES:
                sx, sy = ax + (bx - ax) * ts, ay + (by - ay) * ts
                own = seq_s[max(k for k, tt in enumerate(seq_t) if tt <= ts + 1e-9)]
                touched = {own}
                if len(cseg):
                    near = _near_lines(sx, sy, cseg, RADIUS)
                    for k in near:
                        L = m.lines[cand[k]]
                        if L.two_sided:
                            touched.add(L.front)
                            touched.add(L.back)
                tl = list(touched)
                fb, cb = ef[tl].max(), ec[tl].min()
                fr, cr = real_floor[tl].max(), real_ceil[tl].min()
                if cb - fb < MIN_HEADROOM or (prev_fb is not None and fb - prev_fb > MAX_STEP):
                    valid = False
                    break
                if cr - fr < MIN_HEADROOM or (prev_fr is not None and fr - prev_fr > MAX_STEP):
                    g_s = next((x for x in tl if x in dyn), -1)
                    if g_s < 0:
                        valid = False
                        break
                    gate = g_s
                    if gate in door_lines:
                        use_line = door_lines[gate][0]
                elif gate < 0:
                    g_s = next((x for x in tl if x in dyn and dyn[x] == "lift"), -1)
                    if g_s >= 0:
                        gate = g_s
                prev_fb, prev_fr = fb, fr
            if not valid:
                continue
            cost = math.hypot(bx - ax, by - ay)
            if int(sec[j]) in damage:
                cost *= 6
            adj[i].append(Edge(j, cost, gate, -1, use_line))

    g = NavGraph(m.name, px.astype(float), py.astype(float), sec.astype(np.int32), adj, cell,
                 (ox, oy), dyn, door_lines, damage)
    _add_teleports(g, m)
    return g


def _add_teleports(g: NavGraph, m: MapData) -> None:
    for l in m.lines:
        if l.kind != "teleport" or l.special in (125, 126):
            continue
        targets = [s for s in m.tagged_sectors(l.tag)]
        dest_node = None
        for t in m.things:
            if t.type != 14:
                continue
            # find which sector the destination thing is in via nearest node's sector
            ni = g.nearest(t.x, t.y)
            if ni is not None and int(g.sector[ni]) in targets:
                dest_node = ni
                break
        if dest_node is None:
            continue
        mx, my = l.mid
        nxv, nyv = l.front_normal
        for i in range(g.n):
            dx, dy = g.xs[i] - mx, g.ys[i] - my
            if dx * dx + dy * dy < 56 ** 2 and dx * nxv + dy * nyv > 0:
                g.adj[i].append(Edge(dest_node, 64.0, -1, l.idx, -1))


def load_or_build(m: MapData, use_cache: bool = True) -> NavGraph:
    CACHE_DIR.mkdir(exist_ok=True)
    p = CACHE_DIR / f"{m.wad}_{m.name}.navgraph.pkl"
    if use_cache and p.exists():
        try:
            g = pickle.loads(p.read_bytes())
            if getattr(g, "version", 0) == VERSION:
                return g
        except Exception:
            pass
    g = build_navgraph(m)
    p.write_bytes(pickle.dumps(g))
    return g
