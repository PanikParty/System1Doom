"""Waypoint following over the nav graph: doors, lifts, teleports, stuck recovery."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from s1doom.control.aim import MAX_MOVE
from s1doom.mapmodel.navgraph import Edge
from s1doom.mapmodel.runtime import LevelModel
from s1doom.perception.entities import Percept
from s1doom.perception.geometry import bearing_deg, turn_command

WAYPOINT_REACHED = 30.0
STUCK_TICS = 35
STUCK_DIST = 16.0
REPLAN_TICS = 105


@dataclass
class NavStatus:
    arrived: bool = False
    no_path: bool = False
    stuck: bool = False
    teleported: bool = False
    waiting: str | None = None      # "door" | "lift"


@dataclass
class Navigator:
    level: LevelModel
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    goal: tuple[float, float] | None = None
    goal_tag: str = ""
    path: list[int] | None = None
    idx: int = 0
    planned_at: int = -10 ** 9
    _hist: list[tuple[int, float, float]] = field(default_factory=list)
    _recover_until: int = -1
    _recover_dir: float = 1.0
    _last_pos: tuple[float, float] | None = None
    _wait_since: int | None = None
    stuck_count: int = 0

    def set_goal(self, x: float, y: float, tag: str = "") -> None:
        if self.goal is not None and math.hypot(self.goal[0] - x, self.goal[1] - y) < 48 and self.goal_tag == tag:
            return
        self.goal, self.goal_tag = (x, y), tag
        self.path = None

    def clear(self) -> None:
        self.goal, self.path = None, None

    def reset(self) -> None:
        """New episode: tics restart at 0, so every tic-stamped field must go."""
        self.clear()
        self.goal_tag = ""
        self.idx = 0
        self.planned_at = -10 ** 9
        self._hist.clear()
        self._recover_until = -1
        self._last_pos = None
        self._wait_since = None
        self.stuck_count = 0

    def _replan(self, p: Percept) -> bool:
        self.planned_at = p.tic
        if self.goal is None:
            return False
        self.path = self.level.plan(p.x, p.y, *self.goal)
        self.idx = 0
        return self.path is not None

    def _cur_edge(self) -> Edge | None:
        if not self.path or self.idx == 0 or self.idx >= len(self.path):
            return None
        return self.level.nav.edge(self.path[self.idx - 1], self.path[self.idx])

    def step(self, p: Percept, face: tuple[float, float] | None = None) -> tuple[dict, NavStatus]:
        """Action toward the goal. If `face` is given, keep facing that point and
        move with forward/strafe components (retreat / fight-while-moving)."""
        st = NavStatus()
        if self._last_pos and math.hypot(p.x - self._last_pos[0], p.y - self._last_pos[1]) > 160:
            st.teleported = True
            self.path = None
        self._last_pos = (p.x, p.y)
        if self.goal is None:
            return {}, st
        if math.hypot(p.x - self.goal[0], p.y - self.goal[1]) < WAYPOINT_REACHED:
            st.arrived = True
            return {}, st
        if self.path is None or p.tic - self.planned_at > REPLAN_TICS:
            if not self._replan(p):
                st.no_path = True
                return {}, st

        # stuck recovery in progress
        if p.tic < self._recover_until:
            return {"strafe": self._recover_dir * MAX_MOVE, "forward": 25.0, "use": float(p.tic % 2)}, st

        nav = self.level.nav
        # advance along path
        while self.idx < len(self.path):
            wx, wy = nav.pos(self.path[self.idx])
            if math.hypot(p.x - wx, p.y - wy) < WAYPOINT_REACHED and self.idx < len(self.path) - 1:
                self.idx += 1
            else:
                break
        # skip ahead: if a later waypoint is close and in LOS, go straight to it
        for j in range(min(len(self.path) - 1, self.idx + 4), self.idx, -1):
            e = nav.edge(self.path[j - 1], self.path[j])
            if e is None or e.gate >= 0 or e.teleport >= 0:
                break
            wx, wy = nav.pos(self.path[j])
            if math.hypot(p.x - wx, p.y - wy) < 140 and self.level.vis.los(p.x, p.y, wx, wy):
                self.idx = j
                break
        if self.idx >= len(self.path):
            tx, ty = self.goal
        else:
            tx, ty = nav.pos(self.path[self.idx])
        act: dict = {}

        # gated edge ahead (door / lift)?
        e = self._cur_edge()
        if e is not None and e.gate >= 0 and not self.level.sector_open(e.gate):
            kind = nav.dynamic.get(e.gate)
            st.waiting = kind
            door_ln = self.level.door_face(e.gate, p.x, p.y) if e.use_line >= 0 else None
            if door_ln is not None:
                l = door_ln
                mx, my = l.mid
                if math.hypot(p.x - mx, p.y - my) < 96:
                    tx, ty = mx, my
                    act["use"] = 1.0 if p.tic % 6 < 3 else 0.0
            else:
                trig = self.level.trigger_near(e.gate, p.x, p.y)
                if trig is not None:
                    # walk up to the switch that moves this sector and press it
                    mx, my = trig.mid
                    if math.hypot(p.x - mx, p.y - my) < 72:
                        tx, ty = mx, my
                        act["use"] = 1.0 if p.tic % 6 < 3 else 0.0
                    else:
                        nx, ny = trig.front_normal
                        tx, ty = mx + nx * 40, my + ny * 40
                else:
                    act["use"] = 1.0 if p.tic % 6 < 3 else 0.0
            if self._wait_since is None:
                self._wait_since = p.tic
            elif p.tic - self._wait_since > 35 * 6:
                self.level.block_edge(self.path[self.idx - 1], self.path[self.idx])
                self._wait_since = None
                self.path = None
                return act, st
        else:
            self._wait_since = None
        if e is not None and e.teleport >= 0:
            l = self.level.m.lines[e.teleport]
            mx, my = l.mid
            nx, ny = l.front_normal
            tx, ty = mx - nx * 24, my - ny * 24

        b = bearing_deg(p.x, p.y, p.angle, tx, ty)
        if face is None:
            act["turn"] = turn_command(b, 0.5, 10.0)
            ab = abs(b)
            act["forward"] = MAX_MOVE if ab < 30 else 25.0 if ab < 70 else 0.0
        else:
            fb = bearing_deg(p.x, p.y, p.angle, *face)
            act["turn"] = turn_command(fb, 0.6, 12.0)
            rel = math.radians(b)
            act["forward"] = MAX_MOVE * math.cos(rel)
            act["strafe"] = -MAX_MOVE * math.sin(rel)

        # stuck detection (only when we are trying to move and not waiting on a door)
        self._hist.append((p.tic, p.x, p.y))
        self._hist = [h for h in self._hist if p.tic - h[0] <= STUCK_TICS]
        if st.waiting is None and len(self._hist) > 1 and p.tic - self._hist[0][0] >= STUCK_TICS - 1:
            t0, x0, y0 = self._hist[0]
            if math.hypot(p.x - x0, p.y - y0) < STUCK_DIST:
                st.stuck = True
                self.stuck_count += 1
                self._recover_until = p.tic + 12
                self._recover_dir = self.rng.choice((-1.0, 1.0))
                self._hist.clear()
                if self.path and 0 < self.idx < len(self.path):
                    self.level.block_edge(self.path[self.idx - 1], self.path[self.idx], 35 * 10)
                self.path = None
        return act, st
