"""Short-term memory of recently seen entities, decayed over time, and event detection."""
from __future__ import annotations

from dataclasses import dataclass, field

from s1doom.env.game import Obs
from s1doom.perception.entities import HFOV, Entity, Percept, entities_from_obs

MONSTER_TTL = 70          # 2 s
PICKUP_TTL = 35 * 60      # 60 s
MONSTER_GONE_TICS = 4       # in view but not labelled -> dead or occluded
PICKUP_GONE_TICS = 8      # in clear view but not labelled for this long -> gone


@dataclass
class Perception:
    omniscient: bool = False
    health_thresholds: tuple[int, ...] = (60, 30)
    mem: dict[int, Entity] = field(default_factory=dict)
    _unseen_in_view: dict[int, int] = field(default_factory=dict)
    _prev_damage: float | None = None
    _prev_health: float | None = None
    _prev_kills: float = 0.0

    def reset(self) -> None:
        self.mem.clear()
        self._unseen_in_view.clear()
        self._prev_damage = None
        self._prev_health = None
        self._prev_kills = 0.0

    def update(self, obs: Obs) -> Percept:
        px, py, ang = obs.v("POSITION_X"), obs.v("POSITION_Y"), obs.v("ANGLE")
        seen = entities_from_obs(obs, self.omniscient)
        events: set[str] = set()
        seen_ids = set()
        for e in seen:
            seen_ids.add(e.id)
            old = self.mem.get(e.id)
            if old is None:
                if e.kind == "monster":
                    events.add("new_monster")
                elif e.kind == "key":
                    events.add("key_seen")
            else:
                e.first_seen_tic = old.first_seen_tic
            self.mem[e.id] = e
            self._unseen_in_view.pop(e.id, None)

        for eid, e in list(self.mem.items()):
            if eid in seen_ids:
                continue
            e.visible = False
            e.aligned = False
            e.screen_x_frac = e.screen_w_frac = None
            e.refresh(px, py, ang)
            age = obs.tic - e.last_seen_tic
            if e.kind == "monster":
                # dead monsters vanish from labels; so does one that stepped behind a wall.
                # Either way, one that should be in clear view but isn't is dropped fast.
                in_view = abs(e.bearing) < HFOV / 2 - 5 and e.distance < 1200
                n = self._unseen_in_view.get(eid, 0) + 1 if in_view else 0
                self._unseen_in_view[eid] = n
                if age > MONSTER_TTL or n >= MONSTER_GONE_TICS:
                    del self.mem[eid]
                    self._unseen_in_view.pop(eid, None)
                continue
            # pickups: gone when touched, when in clear view and not labelled, or on TTL
            if e.distance < 40 or age > PICKUP_TTL:
                del self.mem[eid]
                continue
            if abs(e.bearing) < HFOV / 2 - 8 and e.distance < 400:
                n = self._unseen_in_view.get(eid, 0) + 1
                self._unseen_in_view[eid] = n
                if n >= PICKUP_GONE_TICS:
                    del self.mem[eid]
            else:
                self._unseen_in_view.pop(eid, None)

        dmg_total = obs.v("DAMAGE_TAKEN")
        dmg = 0.0 if self._prev_damage is None else max(0.0, dmg_total - self._prev_damage)
        self._prev_damage = dmg_total
        health = obs.v("HEALTH")
        if dmg > 0:
            events.add("damage")
        if self._prev_health is not None:
            for th in self.health_thresholds:
                if self._prev_health >= th > health:
                    events.add("health_threshold")
        self._prev_health = health
        kills = obs.v("KILLCOUNT")
        if kills > self._prev_kills:
            events.add("kill")
        self._prev_kills = kills

        weapon = int(obs.v("SELECTED_WEAPON"))
        p = Percept(tic=obs.tic, x=px, y=py, z=obs.v("POSITION_Z"), angle=ang, health=health,
                    armor=obs.v("ARMOR"), weapon=weapon, ammo=obs.v("SELECTED_WEAPON_AMMO"),
                    vars=obs.vars, entities=list(self.mem.values()), damage_taken=dmg, events=events)
        p.unseen_attack = dmg > 0 and not p.visible_monsters
        return p
