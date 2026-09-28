"""Entity table from ViZDoom labels (visible-only) or objects (omniscient ablation)."""
from __future__ import annotations

from dataclasses import dataclass, field

from s1doom.env.game import Obs
from s1doom.perception import classes as C
from s1doom.perception.geometry import (aim_tol, bearing_deg, distance, range_bucket, side_of)

HFOV = 90.0


@dataclass
class Entity:
    id: int
    name: str
    kind: str
    x: float
    y: float
    z: float
    distance: float = 0.0
    bearing: float = 0.0
    screen_x_frac: float | None = None
    screen_w_frac: float | None = None
    aligned: bool = False
    range_bucket: str = "far"
    threat: float = 0.0
    hp_est: int = 0
    visible: bool = True
    last_seen_tic: int = 0
    first_seen_tic: int = 0
    vx: float = 0.0
    vy: float = 0.0

    @property
    def side(self) -> str:
        return side_of(self.bearing)

    def refresh(self, px: float, py: float, angle: float) -> None:
        self.distance = distance(px, py, self.x, self.y)
        self.bearing = bearing_deg(px, py, angle, self.x, self.y)
        self.range_bucket = range_bucket(self.distance)
        if self.kind == "monster":
            hp, danger, _ = C.monster_info(self.name)
            prox = 1.0 if self.distance < 256 else max(0.15, 256.0 / self.distance)
            self.threat = danger * prox
            self.hp_est = hp


def _aligned(bearing: float, dist: float, sx: float | None, sw: float | None) -> bool:
    if abs(bearing) < aim_tol(dist):
        return True
    if sx is not None and sw is not None:
        # label box spans the screen centre column, with a small margin
        left, right = sx - sw / 2, sx + sw / 2
        margin = min(0.02, sw * 0.25)
        return left + margin <= 0.5 <= right - margin
    return False


def entities_from_obs(obs: Obs, omniscient: bool = False) -> list[Entity]:
    px, py, ang = obs.v("POSITION_X"), obs.v("POSITION_Y"), obs.v("ANGLE")
    out: list[Entity] = []
    if omniscient:
        for o in obs.objects:
            kind = C.kind_of(o.name, getattr(o, "category", None))
            if kind == "other" and o.name not in C.HAZARDS:
                continue
            e = Entity(o.id, o.name, kind, o.position_x, o.position_y, o.position_z,
                       last_seen_tic=obs.tic, first_seen_tic=obs.tic,
                       vx=o.velocity_x, vy=o.velocity_y)
            e.refresh(px, py, ang)
            e.aligned = abs(e.bearing) < aim_tol(e.distance)
            out.append(e)
        return out
    for lab in obs.labels:
        cat = getattr(lab, "object_category", None)
        if cat == "Self":
            continue
        kind = C.kind_of(lab.object_name, cat)
        if kind == "other" and lab.object_name not in C.HAZARDS:
            continue
        sx = (lab.x + lab.width / 2) / obs.screen_w
        sw = lab.width / obs.screen_w
        e = Entity(lab.object_id, lab.object_name, kind, lab.object_position_x,
                   lab.object_position_y, lab.object_position_z, screen_x_frac=sx,
                   screen_w_frac=sw, last_seen_tic=obs.tic, first_seen_tic=obs.tic,
                   vx=getattr(lab, "object_velocity_x", 0.0), vy=getattr(lab, "object_velocity_y", 0.0))
        e.refresh(px, py, ang)
        e.aligned = _aligned(e.bearing, e.distance, sx, sw)
        out.append(e)
    return out


@dataclass
class Percept:
    tic: int
    x: float
    y: float
    z: float
    angle: float
    health: float
    armor: float
    weapon: int
    ammo: float
    vars: dict[str, float]
    entities: list[Entity] = field(default_factory=list)
    damage_taken: float = 0.0         # this decision period / tic delta
    unseen_attack: bool = False       # damage taken with no visible monster
    events: set[str] = field(default_factory=set)

    @property
    def monsters(self) -> list[Entity]:
        return sorted((e for e in self.entities if e.kind == "monster"),
                      key=lambda e: -e.threat)

    @property
    def visible_monsters(self) -> list[Entity]:
        return [e for e in self.monsters if e.visible]

    @property
    def pickups(self) -> list[Entity]:
        return sorted((e for e in self.entities if e.kind.startswith("pickup") or e.kind in ("weapon", "key")),
                      key=lambda e: e.distance)

    @property
    def hazards(self) -> list[Entity]:
        return [e for e in self.entities if e.name in C.HAZARDS]

    def entity(self, eid: int | None) -> Entity | None:
        if eid is None:
            return None
        return next((e for e in self.entities if e.id == eid), None)
