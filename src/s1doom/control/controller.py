"""Deterministic per-tic controller: turns the committed macro + target into buttons."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from s1doom.control import macros as M
from s1doom.control.navigator import Navigator
from s1doom.decision.policy import Decision
from s1doom.mapmodel.runtime import LevelModel
from s1doom.perception.entities import Entity, Percept

TARGET_LOST_TICS = 6


@dataclass
class Controller:
    buttons: list[str]
    level: LevelModel | None = None
    seed: int = 0
    decision: Decision | None = None
    refs: dict = field(default_factory=dict)
    target_id: int | None = None
    pickup_id: int | None = None
    door: dict | None = None
    spawn_angle: float = 0.0
    scan_dir: float = 1.0
    explore_goal: int | None = None
    wander_turn_until: int = -1
    _pos_hist: list[tuple[int, float, float]] = field(default_factory=list)
    _target_missing: int = 0
    nav: Navigator | None = None
    rng: random.Random = field(init=False)
    events: set[str] = field(default_factory=set)
    weapon_request: int | None = None
    opportunistic_fire: bool = True
    blacklist: dict = field(default_factory=dict)   # entity id | "d<line>" -> until tic
    _goal_started: int = 0

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.can_move = "forward" in self.buttons
        self.has_strafe = "strafe" in self.buttons
        if self.level is not None:
            self.nav = Navigator(self.level, random.Random(self.seed))

    def reset(self, spawn_angle: float) -> None:
        self.decision = None
        self.target_id = self.pickup_id = None
        self.door = None
        self.spawn_angle = spawn_angle
        self.explore_goal = None
        self._pos_hist.clear()
        self.blacklist.clear()
        self.wander_turn_until = -1
        self._goal_started = 0
        self._target_missing = 0
        self.weapon_request = None
        if self.nav is not None:
            self.nav.reset()

    # -- decisions ----------------------------------------------------------
    def commit(self, d: Decision, refs: dict, weapons: dict[str, int]) -> None:
        self.decision = d
        self.refs = refs
        t = refs.get(d.target) if d.target else None
        if isinstance(t, Entity):
            self.target_id = t.id
        elif d.mode in ("fight", "retreat"):
            ms = [v for k, v in refs.items() if k.startswith("m") and isinstance(v, Entity) and v.visible]
            self.target_id = ms[0].id if ms else None
        prev_pickup = self.pickup_id
        pk = refs.get(d.pickup) if d.pickup else None
        if isinstance(pk, Entity):
            self.pickup_id = pk.id
        elif d.mode in ("grab_pickup", "get_key"):
            ps = [v for k, v in refs.items() if k.startswith("p") and isinstance(v, Entity)]
            if d.mode == "get_key":
                ps = [e for e in ps if e.kind == "key"] or ps
            self.pickup_id = ps[0].id if ps else None
        if self.pickup_id != prev_pickup:
            self._goal_started = -1   # stamped on the next tic
        doors = [v for k, v in refs.items() if k.startswith("d")]
        new_door = doors[0] if doors else None
        if (new_door or {}).get("line") != (self.door or {}).get("line"):
            self._goal_started = -1
        self.door = new_door
        self.weapon_request = weapons.get(d.weapon) if d.weapon else None
        self._target_missing = 0

    # -- helpers ------------------------------------------------------------
    def moved_recently(self, p: Percept, tics: int) -> float | None:
        old = [h for h in self._pos_hist if p.tic - h[0] >= tics]
        if not old:
            return None
        _, x, y = old[-1]
        return math.hypot(p.x - x, p.y - y)

    def cover_point(self, p: Percept, threat: Entity) -> tuple[float, float] | None:
        lv = self.level
        if lv is None:
            return None
        nav = lv.nav
        d2 = (nav.xs - p.x) ** 2 + (nav.ys - p.y) ** 2
        near = np.nonzero((d2 < 480 ** 2) & (d2 > 64 ** 2))[0]
        if len(near) == 0:
            return None
        # farther from the threat than we are, and out of its line of sight
        tdist = np.hypot(nav.xs[near] - threat.x, nav.ys[near] - threat.y)
        near = near[tdist > threat.distance]
        if len(near) == 0:
            return None
        pts = np.stack([nav.xs[near], nav.ys[near]], axis=1)
        hidden = ~lv.vis.los_many(threat.x, threat.y, pts)
        cand = near[hidden]
        if len(cand) == 0:
            return None
        i = cand[np.argmin(d2[cand])]
        return nav.pos(int(i))

    # -- per tic ------------------------------------------------------------
    def act(self, p: Percept) -> dict:
        self.events = set()
        self._pos_hist.append((p.tic, p.x, p.y))
        if len(self._pos_hist) > 80:
            self._pos_hist = self._pos_hist[-80:]
        d = self.decision
        if d is None:
            return {}
        tgt = p.entity(self.target_id)
        if d.mode in ("fight", "retreat"):
            if tgt is None or not tgt.visible:
                self._target_missing += 1
                if self._target_missing == TARGET_LOST_TICS:
                    self.events.add("target_lost")
            else:
                self._target_missing = 0
        mode = d.mode
        if mode == "fight":
            act, done = M.fight(self, p, tgt)
        elif mode == "retreat":
            act, done = M.retreat(self, p, tgt)
        elif mode == "scan":
            act, done = M.scan(self, p)
        elif mode in ("grab_pickup", "get_key"):
            e = p.entity(self.pickup_id)
            if self._goal_started < 0:
                self._goal_started = p.tic
            act, done = M.goto_entity(self, p, e, mode)
            # give up on items we can't collect: reached in 2D but still there (ledge),
            # no path, or no success within 10 s
            if e is not None and (done or p.tic - self._goal_started > 350
                                  or (self.nav is not None and self.nav.goal_tag.startswith(mode) and self.nav.path is None
                                      and p.tic - self.nav.planned_at < 2)):
                if done or p.tic - self._goal_started > 350 or self.level.plan(p.x, p.y, e.x, e.y) is None:
                    self.blacklist[e.id] = p.tic + 35 * 60
                    self.events.add("macro_done")
                    done = True
        elif mode == "explore":
            act, done = M.explore(self, p)
        elif mode == "go_to_exit":
            act, done = M.go_to_exit(self, p)
        elif mode == "open_door":
            if self._goal_started < 0:
                self._goal_started = p.tic
            act, done = M.open_door(self, p, self.door)
            if self.door is not None and (p.tic - self._goal_started > 350 or
                                          (done and not self.level.sector_open(self.door["sector"]))):
                self.blacklist[f"d{self.door['line']}"] = p.tic + 35 * 60
                done = True
        elif mode == "advance":
            act, done = M.advance(self, p)
        else:
            act, done = {}, True
        if done:
            self.events.add("macro_done")
        if self.nav is not None and self.nav.stuck_count:
            self.events.add("stuck")
            self.nav.stuck_count = 0
        if self.weapon_request and self.weapon_request != p.weapon:
            act[f"weapon{self.weapon_request}"] = 1.0
        # opportunistic fire: never waste a free aligned shot at a visible monster
        if self.opportunistic_fire and "attack" not in act and d.mode not in ("retreat",):
            if any(e.aligned and M.worth_shot(p, e) for e in p.visible_monsters) and (p.ammo > 0 or p.weapon == 1):
                act["attack"] = 1.0
        # open doors we bump into while navigating
        if self.level is not None and self.can_move and "use" not in act and p.tic % 10 == 0:
            if self.level.use_lines_near(p.x, p.y, 72.0):
                act["use"] = 1.0
        return {k: v for k, v in act.items() if k in self.buttons or k.startswith("weapon")}
