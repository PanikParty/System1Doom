"""Parse Doom-format map lumps with omgifol into plain numpy-friendly structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import omg

from s1doom.mapmodel import specials as SP


@dataclass
class Line:
    idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    front: int            # sector index or -1
    back: int             # sector index or -1
    special: int
    tag: int
    impassable: bool
    kind: str | None      # specials.classify

    @property
    def mid(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def two_sided(self) -> bool:
        return self.front >= 0 and self.back >= 0

    @property
    def blocking(self) -> bool:
        return not self.two_sided or self.impassable

    @property
    def front_normal(self) -> tuple[float, float]:
        # Doom: front side is on the RIGHT of v1->v2
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        L = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        return dy / L, -dx / L


@dataclass
class Sector:
    idx: int
    floor: float
    ceil: float
    special: int
    tag: int


@dataclass
class Thing:
    type: int
    x: float
    y: float
    angle: int
    skills: tuple[bool, bool, bool]   # easy, medium, hard


@dataclass
class MapData:
    wad: str
    name: str
    lines: list[Line]
    sectors: list[Sector]
    things: list[Thing]
    seg: np.ndarray = field(init=False)          # (L, 4) x1 y1 x2 y2
    blocking_mask: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.seg = np.array([[l.x1, l.y1, l.x2, l.y2] for l in self.lines], dtype=np.float64)
        self.blocking_mask = np.array([l.blocking for l in self.lines])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = np.concatenate([self.seg[:, 0], self.seg[:, 2]])
        ys = np.concatenate([self.seg[:, 1], self.seg[:, 3]])
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    def lines_of_kind(self, *kinds: str) -> list[Line]:
        return [l for l in self.lines if l.kind in kinds]

    def tagged_sectors(self, tag: int) -> list[int]:
        return [s.idx for s in self.sectors if s.tag == tag and tag != 0]

    def player_start(self) -> Thing | None:
        return next((t for t in self.things if t.type == 1), None)


def load_map(wad_path: str, map_name: str) -> MapData:
    w = omg.WAD(str(wad_path))
    if map_name not in w.maps:
        raise KeyError(f"{map_name} not in {wad_path}")
    m = omg.MapEditor(w.maps[map_name])
    verts = m.vertexes
    sides = m.sidedefs
    lines = []
    for i, ld in enumerate(m.linedefs):
        a, b = verts[ld.vx_a], verts[ld.vx_b]
        front = sides[ld.front].sector if ld.front not in (-1, 0xFFFF) and ld.front < len(sides) else -1
        back = sides[ld.back].sector if ld.back not in (-1, 0xFFFF) and ld.back < len(sides) else -1
        lines.append(Line(i, a.x, a.y, b.x, b.y, front, back, ld.action, ld.tag,
                          bool(ld.impassable), SP.classify(ld.action)))
    sectors = [Sector(i, s.z_floor, s.z_ceil, s.type, s.tag) for i, s in enumerate(m.sectors)]
    things = [Thing(t.type, t.x, t.y, t.angle, (bool(t.easy), bool(t.medium), bool(t.hard)))
              for t in m.things]
    return MapData(Path(wad_path).stem, map_name, lines, sectors, things)
