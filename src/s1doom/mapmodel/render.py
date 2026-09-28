"""Render a map + nav graph + specials to an image (inspect-map, overlay inset)."""
from __future__ import annotations

import cv2
import numpy as np

from s1doom.mapmodel.navgraph import NavGraph
from s1doom.mapmodel.wadparse import MapData

COLORS = {"exit": (0, 0, 255), "secret_exit": (180, 0, 255), "door": (0, 200, 255),
          "remote_door": (0, 140, 255), "lift": (255, 128, 0), "teleport": (255, 0, 255),
          "floor": (128, 128, 0), "other": (90, 90, 90)}


class MapCanvas:
    def __init__(self, m: MapData, size: int = 1400, pad: int = 20) -> None:
        self.m = m
        x0, y0, x1, y1 = m.bounds
        self.x0, self.y1 = x0, y1
        self.scale = (size - 2 * pad) / max(x1 - x0, y1 - y0)
        self.pad = pad
        self.w = int((x1 - x0) * self.scale) + 2 * pad
        self.h = int((y1 - y0) * self.scale) + 2 * pad

    def px(self, x: float, y: float) -> tuple[int, int]:
        return (int(self.pad + (x - self.x0) * self.scale), int(self.pad + (self.y1 - y) * self.scale))

    def base(self, nav: NavGraph | None = None, show_edges: bool = False) -> np.ndarray:
        img = np.full((self.h, self.w, 3), 18, np.uint8)
        if nav is not None:
            if show_edges:
                for i in range(nav.n):
                    for e in nav.adj[i]:
                        col = (70, 70, 140) if e.gate >= 0 else (45, 60, 45)
                        if e.teleport >= 0:
                            col = (160, 0, 160)
                        cv2.line(img, self.px(*nav.pos(i)), self.px(*nav.pos(e.to)), col, 1)
            for i in range(nav.n):
                cv2.circle(img, self.px(*nav.pos(i)), 1, (60, 110, 60), -1)
        for l in self.m.lines:
            col = (200, 200, 200) if l.blocking else (90, 90, 90)
            th = 1
            if l.kind:
                col, th = COLORS.get(l.kind, (90, 90, 90)), 2
            cv2.line(img, self.px(l.x1, l.y1), self.px(l.x2, l.y2), col, th)
        st = self.m.player_start()
        if st:
            cv2.drawMarker(img, self.px(st.x, st.y), (0, 255, 0), cv2.MARKER_STAR, 14, 2)
        return img

    def path(self, img: np.ndarray, nav: NavGraph, path: list[int], col=(0, 255, 255)) -> None:
        for a, b in zip(path, path[1:]):
            cv2.line(img, self.px(*nav.pos(a)), self.px(*nav.pos(b)), col, 2)


def legend(img: np.ndarray) -> None:
    y = 20
    for k, c in COLORS.items():
        cv2.putText(img, k, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        y += 16
