"""Level model at runtime: static WAD geometry + live sector heights + exploration."""
from __future__ import annotations

import math

import numpy as np

from s1doom.mapmodel import specials as SP
from s1doom.mapmodel.frontier import Exploration
from s1doom.mapmodel.navgraph import MAX_STEP, MIN_HEADROOM, Edge, NavGraph, load_or_build
from s1doom.mapmodel.visibility import Visibility
from s1doom.mapmodel.wadparse import Line, MapData, load_map

EXIT_SIGHT = 768.0
DOOR_RANGE = 512.0
TRIGGER_REACH = 192.0   # a switch this close to a closed gate is pressed on the way


class LevelModel:
    def __init__(self, wad_path: str, map_name: str, use_cache: bool = True) -> None:
        self.m: MapData = load_map(wad_path, map_name)
        self.nav: NavGraph = load_or_build(self.m, use_cache)
        self.vis = Visibility(self.m)
        self.explore = Exploration(self.nav, self.vis)
        self.floor = np.array([s.floor for s in self.m.sectors], dtype=float)
        self.ceil = np.array([s.ceil for s in self.m.sectors], dtype=float)
        self.exits: list[Line] = self.m.lines_of_kind("exit")
        self.exit_known = False
        self.keys_held: set[str] = set()
        self.blocked_edges: dict[tuple[int, int], int] = {}   # (u, v) -> expiry tic
        self.tic = 0
        self.used_lines: set[int] = set()
        # sector -> USE-activated lines that move it (remote door switches, lift switches)
        self.triggers: dict[int, list[int]] = {}
        for l in self.m.lines:
            if l.tag and l.kind in ("remote_door", "lift", "floor") and SP.is_use(l.special):
                for s in self.m.tagged_sectors(l.tag):
                    self.triggers.setdefault(s, []).append(l.idx)

    def trigger_near(self, sector: int, x: float, y: float, radius: float = TRIGGER_REACH) -> Line | None:
        best, bd = None, radius
        for li in self.triggers.get(sector, []):
            l = self.m.lines[li]
            d = math.hypot(l.mid[0] - x, l.mid[1] - y)
            nx, ny = l.front_normal
            # Doom only activates a line used from its FRONT side (P_UseLines)
            if (x - l.mid[0]) * nx + (y - l.mid[1]) * ny <= 0:
                continue
            if d < bd:
                best, bd = l, d
        return best

    def reset(self) -> None:
        self.explore.reset()
        self.exit_known = False
        self.keys_held.clear()
        self.blocked_edges.clear()
        self.used_lines.clear()

    # -- live state ---------------------------------------------------------
    def update(self, x: float, y: float, sectors, tic: int) -> None:
        self.tic = tic
        if sectors and len(sectors) == len(self.floor):
            for s in sectors:
                self.floor[s.id] = s.floor_height
                self.ceil[s.id] = s.ceiling_height
        closed = {s for s, k in self.nav.dynamic.items() if k != "lift" and self.ceil[s] - self.floor[s] < MIN_HEADROOM}
        self.vis.set_closed_sectors(closed)
        self.explore.update(x, y)
        if not self.exit_known:
            for l in self.exits:
                mx, my = l.mid
                if math.hypot(mx - x, my - y) < EXIT_SIGHT and self.vis.los(x, y, mx, my):
                    self.exit_known = True

    def sector_open(self, s: int) -> bool:
        return self.ceil[s] - self.floor[s] >= MIN_HEADROOM

    def edge_cost(self, u: int, e: Edge) -> float:
        if self.blocked_edges.get((u, e.to), -1) > self.tic:
            return math.inf
        if e.gate < 0:
            return e.cost
        s = e.gate
        kind = self.nav.dynamic.get(s)
        su = int(self.nav.sector[u])
        step = self.floor[int(self.nav.sector[e.to])] - self.floor[su]
        passable_now = self.sector_open(s) and step <= MAX_STEP
        if passable_now:
            return e.cost
        if kind == "door":
            need = self._door_key(s)
            if need and need not in self.keys_held:
                return math.inf
            return e.cost + 96
        if kind == "lift":
            return e.cost + 160
        # remote door: passable only if its switch is right there (pressed on arrival)
        if self.trigger_near(s, float(self.nav.xs[u]), float(self.nav.ys[u])) is not None:
            return e.cost + 128
        return math.inf

    def _door_key(self, s: int) -> str | None:
        for li in self.nav.door_lines.get(s, []):
            k = SP.door_key(self.m.lines[li].special)
            if k:
                return k
        return None

    def block_edge(self, u: int, v: int, tics: int = 35 * 20) -> None:
        self.blocked_edges[(u, v)] = self.tic + tics

    # -- queries ------------------------------------------------------------
    def node(self, x: float, y: float) -> int | None:
        return self.nav.nearest(x, y)

    def plan(self, x: float, y: float, gx: float, gy: float) -> list[int] | None:
        a, b = self.node(x, y), self.node(gx, gy)
        if a is None or b is None:
            return None
        return self.nav.astar(a, b, self.edge_cost)

    def exit_goal(self) -> tuple[float, float, Line] | None:
        if not self.exits:
            return None
        l = self.exits[0]
        mx, my = l.mid
        nx, ny = l.front_normal
        return mx + nx * 40, my + ny * 40, l

    def path_length(self, path: list[int] | None) -> float:
        if not path:
            return math.inf
        xs, ys = self.nav.xs[path], self.nav.ys[path]
        return float(np.hypot(np.diff(xs), np.diff(ys)).sum())

    def frontier_goal(self, x: float, y: float) -> int | None:
        start = self.node(x, y)
        if start is None:
            return None
        front = set(self.explore.frontier().tolist())
        front -= {f for f, n in self.explore.failed.items() if n >= 2}
        if not front:
            return None
        dist = self.nav.dijkstra(start, self.edge_cost)
        cands = [(d, i) for i, d in dist.items() if i in front and d > 48]
        if not cands:
            return None
        return min(cands)[1]

    def doors_near(self, x: float, y: float, exclude: set | None = None) -> list[dict]:
        out = []
        for s, lines in self.nav.door_lines.items():
            if self.sector_open(s):
                continue
            for li in lines:
                l = self.m.lines[li]
                mx, my = l.mid
                d = math.hypot(mx - x, my - y)
                if d < DOOR_RANGE and self.vis.los(x, y, mx, my):
                    # approach from the side the player is on
                    nx, ny = l.front_normal
                    sgn = 1 if (x - mx) * nx + (y - my) * ny >= 0 else -1
                    if exclude and f"d{li}" in exclude:
                        break
                    if self.plan(x, y, mx + sgn * nx * 40, my + sgn * ny * 40) is None:
                        break
                    out.append({"line": li, "sector": s, "x": mx, "y": my,
                                "ax": mx + sgn * nx * 40, "ay": my + sgn * ny * 40,
                                "distance": d, "needs_key": SP.door_key(l.special)})
                    break
        return sorted(out, key=lambda d: d["distance"])

    def use_lines_near(self, x: float, y: float, radius: float = 80.0) -> list[Line]:
        out = []
        for l in self.m.lines:
            if l.special and SP.is_use(l.special):
                mx, my = l.mid
                if math.hypot(mx - x, my - y) < radius:
                    out.append(l)
        return out
