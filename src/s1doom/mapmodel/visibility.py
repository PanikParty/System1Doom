"""Line-of-sight tests against blocking geometry."""
from __future__ import annotations

import numpy as np

from s1doom.mapmodel.wadparse import MapData


class Visibility:
    def __init__(self, m: MapData) -> None:
        self.m = m
        self.seg = m.seg[m.blocking_mask]
        self._closed = np.zeros((0, 4))

    def set_closed_sectors(self, sectors: set[int]) -> None:
        """Two-sided lines bordering currently-closed sectors (doors) also block sight."""
        rows = [[l.x1, l.y1, l.x2, l.y2] for l in self.m.lines
                if l.two_sided and (l.front in sectors or l.back in sectors)]
        self._closed = np.array(rows) if rows else np.zeros((0, 4))

    def los(self, ax: float, ay: float, bx: float, by: float) -> bool:
        seg = np.vstack([self.seg, self._closed]) if len(self._closed) else self.seg
        return not _crosses_any(ax, ay, bx, by, seg)

    def los_many(self, ax: float, ay: float, pts: np.ndarray) -> np.ndarray:
        """Vectorized LOS from (ax, ay) to each point in pts (N, 2)."""
        seg = np.vstack([self.seg, self._closed]) if len(self._closed) else self.seg
        out = np.ones(len(pts), dtype=bool)
        cx, cy, dx, dy = seg[:, 0][None], seg[:, 1][None], seg[:, 2][None], seg[:, 3][None]
        for i in range(0, len(pts), 1024):
            bx = pts[i:i + 1024, 0][:, None]
            by = pts[i:i + 1024, 1][:, None]
            rx, ry = bx - ax, by - ay
            sx, sy = dx - cx, dy - cy
            den = rx * sy - ry * sx
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((cx - ax) * sy - (cy - ay) * sx) / den
                u = ((cx - ax) * ry - (cy - ay) * rx) / den
            hit = (np.abs(den) > 1e-9) & (t > 1e-6) & (t < 1 - 1e-6) & (u >= 0) & (u <= 1)
            out[i:i + 1024] = ~hit.any(axis=1)
        return out


def _crosses_any(ax, ay, bx, by, seg) -> bool:
    cx, cy, dx, dy = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    den = rx * sy - ry * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((cx - ax) * sy - (cy - ay) * sx) / den
        u = ((cx - ax) * ry - (cy - ay) * rx) / den
    return bool(((np.abs(den) > 1e-9) & (t > 1e-6) & (t < 1 - 1e-6) & (u >= 0) & (u <= 1)).any())
