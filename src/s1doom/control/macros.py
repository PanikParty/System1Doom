"""Per-macro controller behaviours. Each returns (action, done)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from s1doom.control.aim import MAX_MOVE, aim
from s1doom.perception.entities import Entity, Percept
from s1doom.perception.geometry import bearing_deg, norm_deg, turn_command

if TYPE_CHECKING:
    from s1doom.control.controller import Controller


def fight(c: "Controller", p: Percept, t: Entity | None) -> tuple[dict, bool]:
    if t is None:
        return scan(c, p)
    act = {"turn": aim(t)}
    if t.visible and t.aligned and (p.ammo > 0 or p.weapon == 1) and worth_shot(p, t):
        act["attack"] = 1.0
    if c.can_move and t.visible:
        agg = c.decision.aggression if c.decision else 2
        if agg >= 2 and t.threat >= 0.3 and t.range_bucket in ("melee", "close"):
            # circle-strafe: flip direction every second
            act["strafe"] = MAX_MOVE * (1 if (p.tic // 35) % 2 == 0 else -1)
        if agg >= 3 and t.range_bucket in ("mid", "far"):
            act["forward"] = MAX_MOVE * 0.6
        elif agg <= 1 and t.range_bucket in ("melee", "close"):
            act["forward"] = -MAX_MOVE
    return act, False


def barrel_risk(p: Percept, t: Entity) -> bool:
    """A barrel in the line of fire, or next to a close target, would blow up in our face."""
    for b in p.entities:
        if b.kind != "barrel" or not b.visible:
            continue
        if b.distance < t.distance + 32 and abs(b.bearing - t.bearing) < 8 and b.distance < 400:
            return True
        if math.hypot(b.x - t.x, b.y - t.y) < 96 and b.distance < 256:
            return True
    return False


def worth_shot(p: Percept, t: Entity) -> bool:
    """Hitscan spread makes far shots mostly misses; hold fire at far range when ammo is scarce."""
    if barrel_risk(p, t):
        return False
    if t.range_bucket != "far":
        return True
    return p.ammo >= 50 or p.weapon in (5, 6, 7)


def scan(c: "Controller", p: Percept) -> tuple[dict, bool]:
    # turn toward the most recent remembered monster, else keep sweeping
    mem = [e for e in p.monsters if not e.visible]
    if mem:
        return {"turn": turn_command(mem[0].bearing, 0.6, 10.0)}, False
    if p.unseen_attack:
        c.scan_dir = 1.0
        return {"turn": 12.0}, False
    return {"turn": 8.0 * c.scan_dir}, False


def retreat(c: "Controller", p: Percept, t: Entity | None) -> tuple[dict, bool]:
    if t is None:
        return {"forward": -MAX_MOVE}, False
    act = {"turn": aim(t)}
    if t.visible and t.aligned:
        act["attack"] = 1.0
    if c.nav is not None:
        cover = c.cover_point(p, t)
        if cover is not None:
            c.nav.set_goal(*cover, tag="cover")
            nav_act, st = c.nav.step(p, face=(t.x, t.y))
            if not st.no_path and not st.arrived:
                nav_act["turn"] = act["turn"]
                nav_act["attack"] = act.get("attack", 0.0)
                return nav_act, False
    if c.can_move:
        act["forward"] = -MAX_MOVE
        act["strafe"] = MAX_MOVE * 0.5 * (1 if (p.tic // 50) % 2 == 0 else -1)
    return act, False


def goto_entity(c: "Controller", p: Percept, e: Entity | None, tag: str) -> tuple[dict, bool]:
    if e is None:
        return {}, True
    if e.distance < 24:
        return {}, True
    if c.nav is not None and e.distance > 96:
        c.nav.set_goal(e.x, e.y, tag=f"{tag}:{e.id}")
        act, st = c.nav.step(p)
        if not st.no_path:
            return act, st.arrived
    return steer_to(c, p, e.x, e.y), False


def steer_to(c: "Controller", p: Percept, x: float, y: float) -> dict:
    b = bearing_deg(p.x, p.y, p.angle, x, y)
    act = {"turn": turn_command(b, 0.6, 12.0)}
    if c.can_move:
        ab = abs(b)
        act["forward"] = MAX_MOVE if ab < 25 else 20.0 if ab < 60 else 0.0
        # side-step poison vials in the way
        for h in p.hazards:
            if h.distance < 160 and abs(h.bearing) < 18 and h.distance < math.hypot(x - p.x, y - p.y):
                act["turn"] = -12.0 if h.bearing >= 0 else 12.0
                if c.has_strafe:
                    act["strafe"] = MAX_MOVE if h.bearing >= 0 else -MAX_MOVE
    return act


def explore(c: "Controller", p: Percept) -> tuple[dict, bool]:
    if c.nav is not None and c.level is not None:
        if c.explore_goal is None or c.nav.goal_tag != "explore" or c.level.explore.seen[c.explore_goal]:
            c.explore_goal = c.level.frontier_goal(p.x, p.y)
            if c.explore_goal is None:
                return wander(c, p), True
            c.nav.set_goal(*c.level.nav.pos(c.explore_goal), tag="explore")
        act, st = c.nav.step(p)
        if st.no_path or st.arrived:
            if st.no_path and c.explore_goal is not None:
                c.level.explore.failed[c.explore_goal] = c.level.explore.failed.get(c.explore_goal, 0) + 1
            c.explore_goal = None
            return act, True
        return act, False
    # no map model: head for remembered pickups, else wander
    worth = [e for e in p.pickups if e.kind == "pickup_health"]
    if worth:
        return steer_to(c, p, worth[0].x, worth[0].y), False
    return wander(c, p), False


def wander(c: "Controller", p: Percept) -> dict:
    if not c.can_move:
        return {"turn": 8.0}
    if c.wander_turn_until > p.tic:
        return {"turn": 10.0 * c.scan_dir, "forward": 10.0}
    moved = c.moved_recently(p, 20)
    if moved is not None and moved < 8:
        c.wander_turn_until = p.tic + c.rng.randint(8, 18)
        c.scan_dir = c.rng.choice((-1.0, 1.0))
    return {"forward": MAX_MOVE, "turn": 1.5 * c.scan_dir}


def advance(c: "Controller", p: Percept) -> tuple[dict, bool]:
    b = norm_deg(c.spawn_angle - p.angle)
    act = {"turn": turn_command(b, 0.5, 10.0), "forward": MAX_MOVE}
    vis = [e for e in p.visible_monsters if e.aligned]
    if vis and p.ammo > 0:
        act["attack"] = 1.0
    return act, False


def go_to_exit(c: "Controller", p: Percept) -> tuple[dict, bool]:
    lv = c.level
    if lv is None:
        return advance(c, p)
    goal = lv.exit_goal()
    if goal is None:
        return explore(c, p)
    gx, gy, line = goal
    mx, my = line.mid
    if math.hypot(p.x - gx, p.y - gy) < 56 or math.hypot(p.x - mx, p.y - my) < 64:
        b = bearing_deg(p.x, p.y, p.angle, mx, my)
        return {"turn": turn_command(b, 0.6, 12.0), "forward": 25.0 if abs(b) < 20 else 0.0,
                "use": 1.0 if p.tic % 4 < 2 else 0.0}, False
    c.nav.set_goal(gx, gy, tag="exit")
    act, st = c.nav.step(p)
    return act, st.no_path


def open_door(c: "Controller", p: Percept, door: dict | None) -> tuple[dict, bool]:
    if door is None or c.level is None:
        return {}, True
    if c.level.sector_open(door["sector"]):
        return {}, True
    mx, my = door["x"], door["y"]
    if math.hypot(p.x - mx, p.y - my) < 80:
        b = bearing_deg(p.x, p.y, p.angle, mx, my)
        return {"turn": turn_command(b, 0.6, 12.0), "forward": 15.0,
                "use": 1.0 if p.tic % 6 < 3 else 0.0}, False
    c.nav.set_goal(door["ax"], door["ay"], tag=f"door:{door['line']}")
    act, st = c.nav.step(p)
    if st.arrived:
        b = bearing_deg(p.x, p.y, p.angle, mx, my)
        return {"turn": turn_command(b, 0.6, 12.0), "forward": 20.0, "use": 1.0 if p.tic % 6 < 3 else 0.0}, False
    return act, st.no_path
