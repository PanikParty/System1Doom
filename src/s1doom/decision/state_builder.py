"""Compact, pre-digested JSON state with a token budget.

Rule: code computes the geometry, the model makes the judgment. Everything here is a
bucket or boolean except bearing_deg.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from s1doom.decision.tokens import count_tokens
from s1doom.mapmodel.runtime import LevelModel
from s1doom.perception import classes as C
from s1doom.perception.entities import Entity, Percept
from s1doom.perception.geometry import range_bucket

MAX_MONSTERS, MAX_PICKUPS, MAX_DOORS = 5, 3, 2
TOKEN_BUDGET = 600


@dataclass
class StateContext:
    objective: str = "survive"
    level: LevelModel | None = None
    mode: str | None = None
    recent: list[str] = field(default_factory=list)
    rules: str | None = None
    include_rules: bool = True
    budget: int = TOKEN_BUDGET
    exit_path_len: float = math.inf
    exclude: set = field(default_factory=set)      # blacklisted entity ids / "d<line>" doors


@dataclass
class BuiltState:
    state: dict
    refs: dict[str, object]           # "m1" -> Entity, "p1" -> Entity, "d1" -> door dict
    tokens: int
    weapons: dict[str, int]           # weapon option key -> slot


def threat_word(t: float) -> str:
    return "high" if t >= 0.4 else "medium" if t >= 0.2 else "low"


def owned_weapons(p: Percept) -> dict[str, tuple[int, int]]:
    out = {}
    for slot, name in C.WEAPON_SLOTS.items():
        if p.vars.get(f"WEAPON{slot}", 0) > 0 or slot == p.weapon:
            ai = C.WEAPON_AMMO[slot]
            ammo = int(p.vars.get(f"AMMO{ai}", 0)) if ai is not None else 999
            if ammo > 0:
                out[name] = (slot, ammo)
    return out


def pickup_worth(p: Percept, e: Entity) -> bool:
    if e.kind == "pickup_health":
        v = C.HEALTH_VALUE.get(e.name, 10)
        return p.health < 100 and (v >= 10 or p.health < 50)
    if e.kind == "pickup_armor":
        return p.armor < C.ARMOR_VALUE.get(e.name, 1) or e.name == "ArmorBonus" and p.armor < 200
    if e.kind == "pickup_ammo":
        return p.ammo < 40
    return e.kind in ("weapon", "key")


def _monster_row(ref: str, e: Entity) -> dict:
    return {"ref": ref, "kind": C.monster_info(e.name)[2], "visible": e.visible, "range": e.range_bucket,
            "bearing_deg": int(round(e.bearing)), "aligned": e.aligned, "side": e.side,
            "threat": threat_word(e.threat)}


def build_state(p: Percept, ctx: StateContext) -> BuiltState:
    refs: dict[str, object] = {}
    monsters = p.monsters[:MAX_MONSTERS]
    mrows = []
    for i, e in enumerate(monsters, 1):
        refs[f"m{i}"] = e
        mrows.append(_monster_row(f"m{i}", e))
    pickups = [e for e in p.pickups if e.distance < 1024 and e.id not in ctx.exclude][:MAX_PICKUPS]
    prows = []
    for i, e in enumerate(pickups, 1):
        refs[f"p{i}"] = e
        prows.append({"ref": f"p{i}", "kind": e.kind.replace("pickup_", ""), "item": e.name,
                      "range": e.range_bucket, "side": e.side, "worth_it": pickup_worth(p, e)})
    drows = []
    lv = ctx.level
    if lv is not None:
        for i, d in enumerate(lv.doors_near(p.x, p.y, ctx.exclude)[:MAX_DOORS], 1):
            refs[f"d{i}"] = d
            drows.append({"ref": f"d{i}", "needs_key": d["needs_key"], "range": range_bucket(d["distance"])})
    weapons = owned_weapons(p)
    cur_w = C.WEAPON_SLOTS.get(p.weapon, "unknown")
    vis = [e for e in p.monsters if e.visible]
    nearest = min(vis, key=lambda e: e.distance) if vis else None
    situation = {
        "monsters_visible": len(vis),
        "nearest_threat_range": nearest.range_bucket if nearest else "none",
        "aligned_target": any(e.aligned for e in vis),
        "hit_by_unseen": p.unseen_attack,
        "objective": ctx.objective,
        "current_mode": ctx.mode,
    }
    if lv is not None:
        situation["exit_known"] = lv.exit_known
        if lv.exit_known:
            situation["exit_reachable"] = math.isfinite(ctx.exit_path_len)
            if math.isfinite(ctx.exit_path_len):
                situation["exit_path_len"] = range_bucket(ctx.exit_path_len)
        situation["explored"] = f"{int(lv.explore.seen_fraction() * 100)}%"
    hazards = [e for e in p.hazards if e.distance < 400]
    if hazards:
        situation["poison_ahead"] = any(abs(e.bearing) < 20 for e in hazards)
    state = {
        "self": {"health": int(p.health), "armor": int(p.armor), "weapon": cur_w,
                 "ammo": int(p.ammo) if p.weapon != 1 else None,
                 "low_ammo": p.weapon != 1 and p.ammo < 5,
                 "taking_damage": p.damage_taken > 0},
        "situation": situation,
        "monsters": mrows,
        "pickups": prows,
    }
    if lv is not None:
        state["doors"] = drows
        state["keys_held"] = sorted(lv.keys_held)
    state["recent"] = ctx.recent[-3:]
    if ctx.include_rules and ctx.rules:
        state["rules"] = ctx.rules

    # enforce token budget: trim lowest-priority rows first
    tokens = count_tokens(state)
    while tokens > ctx.budget:
        if state["recent"]:
            state["recent"] = state["recent"][1:]
        elif len(state["pickups"]) > 1:
            state["pickups"].pop()
        elif len(state["monsters"]) > 2:
            state["monsters"].pop()
        elif state.get("doors"):
            state["doors"].pop()
        else:
            break
        tokens = count_tokens(state)
    kept = {r["ref"] for k in ("monsters", "pickups", "doors") for r in state.get(k, [])}
    refs = {k: v for k, v in refs.items() if k in kept}
    return BuiltState(state, refs, tokens, {k: v[0] for k, v in weapons.items()})
