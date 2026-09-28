"""Parse Doom-format map lumps with omgifol into plain numpy-friendly structures."""
from __future__ import annotations

import re
import struct
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


# UDMF (ZDoom/Hexen) special numbers -> vanilla-equivalent kind
UDMF_KIND = {10: "door", 11: "door", 12: "door", 13: "door", 202: "door", 243: "exit", 244: "secret_exit",
             70: "teleport", 71: "teleport", 60: "lift", 61: "lift", 62: "lift", 63: "lift", 64: "lift"}
UDMF_USE_KINDS = {"door", "exit", "secret_exit", "lift"}


def _udmf_blocks(text: str) -> list[tuple[str, dict]]:
    text = re.sub(r"//[^\n]*", "", text)
    out = []
    for m in re.finditer(r"(\w+)\s*\{([^}]*)\}", text):
        kv = {}
        for k, v in re.findall(r"(\w+)\s*=\s*([^;]+);", m.group(2)):
            v = v.strip()
            if v.startswith('"'):
                kv[k.lower()] = v.strip('"')
            elif v in ("true", "false"):
                kv[k.lower()] = v == "true"
            else:
                try:
                    kv[k.lower()] = float(v) if "." in v else int(v)
                except ValueError:
                    kv[k.lower()] = v
        out.append((m.group(1).lower(), kv))
    return out


def _load_udmf(wad_path: str, map_name: str, textmap: bytes) -> MapData:
    blocks = _udmf_blocks(textmap.decode("latin-1"))
    verts = [kv for t, kv in blocks if t == "vertex"]
    sides = [kv for t, kv in blocks if t == "sidedef"]
    secs = [kv for t, kv in blocks if t == "sector"]
    lines = []
    for i, kv in enumerate(kv for t, kv in blocks if t == "linedef"):
        a, b = verts[kv["v1"]], verts[kv["v2"]]
        fs, bs = kv.get("sidefront", -1), kv.get("sideback", -1)
        front = sides[fs]["sector"] if 0 <= fs < len(sides) else -1
        back = sides[bs]["sector"] if 0 <= bs < len(sides) else -1
        sp = kv.get("special", 0)
        kind = UDMF_KIND.get(sp, "other" if sp else None)
        tag = kv.get("arg0", 0) if kind in ("remote_door", "lift", "teleport") else 0
        l = Line(i, float(a["x"]), float(a["y"]), float(b["x"]), float(b["y"]), front, back,
                 sp, tag, bool(kv.get("blocking", False)), kind)
        l.udmf_use = bool(kv.get("playeruse", False))
        lines.append(l)
    sectors = [Sector(i, float(kv.get("heightfloor", 0)), float(kv.get("heightceiling", 0)),
                      int(kv.get("special", 0)), int(kv.get("id", 0))) for i, kv in enumerate(secs)]
    things = [Thing(int(kv.get("type", 0)), float(kv["x"]), float(kv["y"]), int(kv.get("angle", 0)),
                    (bool(kv.get("skill1", True)), bool(kv.get("skill3", True)), bool(kv.get("skill5", True))))
              for t, kv in blocks if t == "thing"]
    return MapData(Path(wad_path).stem, map_name, lines, sectors, things)


def _raw_lumps(wad_path: str) -> list[tuple[str, bytes]]:
    d = Path(wad_path).read_bytes()
    n, off = struct.unpack("<ii", d[4:12])
    out = []
    for i in range(n):
        pos, size = struct.unpack("<ii", d[off + 16 * i: off + 16 * i + 8])
        name = d[off + 16 * i + 8: off + 16 * i + 16].rstrip(b"\0").decode("latin-1")
        out.append((name, d[pos:pos + size]))
    return out


def load_map(wad_path: str, map_name: str) -> MapData:
    lumps = _raw_lumps(wad_path)
    names = [n for n, _ in lumps]
    if map_name.upper() in names:
        i = names.index(map_name.upper())
        if i + 1 < len(lumps) and lumps[i + 1][0] == "TEXTMAP":
            return _load_udmf(wad_path, map_name, lumps[i + 1][1])
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
