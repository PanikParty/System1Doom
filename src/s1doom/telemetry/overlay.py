"""OpenCV overlay over the screen buffer: macro, target box, mode probs, confidence,
FALLBACK badge, latency sparkline, automap inset with nav path/frontier."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2
import numpy as np

from s1doom.decision.policy import Decision
from s1doom.perception.entities import Percept

SCALE = 3
FONT = cv2.FONT_HERSHEY_SIMPLEX


class Overlay:
    def __init__(self, show: bool = True, save_dir: Path | None = None, save_every: int = 0,
                 level=None) -> None:
        self.show = show
        self.save_dir = save_dir
        self.save_every = save_every
        self.lat: deque[float] = deque(maxlen=60)
        self.level = level
        self._canvas = None
        self._base = None
        if level is not None:
            from s1doom.mapmodel.render import MapCanvas

            self._canvas = MapCanvas(level.m, size=260, pad=6)
            self._base = self._canvas.base()

    def add_latency(self, ms: float) -> None:
        self.lat.append(ms)

    def render(self, screen: np.ndarray, p: Percept, d: Decision | None, backend: str, target_id: int | None,
               nav_path: list[int] | None = None) -> np.ndarray:
        img = cv2.resize(screen, (screen.shape[1] * SCALE, screen.shape[0] * SCALE), interpolation=cv2.INTER_NEAREST)
        for e in p.entities:
            if not e.visible or e.screen_x_frac is None:
                continue
            if e.kind == "monster":
                w = e.screen_w_frac * img.shape[1]
                x = e.screen_x_frac * img.shape[1]
                col = (0, 0, 255) if e.id == target_id else (0, 200, 255)
                cv2.rectangle(img, (int(x - w / 2), int(img.shape[0] * 0.25)), (int(x + w / 2), int(img.shape[0] * 0.75)), col, 2 if e.id == target_id else 1)
        # crosshair
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        cv2.line(img, (cx - 8, cy), (cx + 8, cy), (0, 255, 0), 1)
        cv2.line(img, (cx, cy - 8), (cx, cy + 8), (0, 255, 0), 1)
        # panel
        panel = img[:150, :420].copy()
        img[:150, :420] = (panel * 0.35).astype(np.uint8)
        y = 22
        mode = d.mode if d else "-"
        src = d.sources.get("mode", "-") if d else "-"
        cv2.putText(img, f"{backend}  mode: {mode}", (8, y), FONT, 0.6, (255, 255, 255), 1)
        if d and "fallback" in d.sources.values():
            cv2.putText(img, "FALLBACK", (img.shape[1] - 150, 30), FONT, 0.8, (0, 0, 255), 2)
        y += 20
        cv2.putText(img, f"src: {src}  danger: {d.in_danger if d else '-'}  aggr: {d.aggression if d else '-'}",
                    (8, y), FONT, 0.45, (200, 200, 200), 1)
        if d and "mode" in d.probs:
            top = sorted(d.probs["mode"].items(), key=lambda kv: -kv[1])[:3]
            for k, v in top:
                y += 18
                cv2.rectangle(img, (8, y - 11), (8 + int(v * 180), y + 2), (80, 180, 80), -1)
                cv2.putText(img, f"{k} {v:.2f}", (196, y), FONT, 0.45, (255, 255, 255), 1)
        if d:
            y += 20
            for k, c in list(d.conf.items())[:4]:
                cv2.putText(img, f"{k[:6]}", (8, y), FONT, 0.4, (200, 200, 200), 1)
                col = (80, 180, 80) if d.sources.get(k) == "model" else (60, 60, 230)
                cv2.rectangle(img, (70, y - 9), (70 + int(c * 100), y), col, -1)
                y += 13
        if self.lat:
            pts = np.array(self.lat)
            mx = max(pts.max(), 1.0)
            xs = np.linspace(430, 630, len(pts)).astype(int)
            ys = (60 - pts / mx * 50).astype(int)
            for i in range(1, len(pts)):
                cv2.line(img, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), (0, 255, 255), 1)
            cv2.putText(img, f"{pts[-1]:.0f} ms", (430, 75), FONT, 0.45, (0, 255, 255), 1)
        if self._canvas is not None:
            inset = self._base.copy()
            lv = self.level
            fr = lv.explore.frontier()
            for i in fr[::2]:
                cv2.circle(inset, self._canvas.px(*lv.nav.pos(int(i))), 1, (255, 0, 255), -1)
            if nav_path:
                self._canvas.path(inset, lv.nav, nav_path, (0, 255, 255))
            cv2.circle(inset, self._canvas.px(p.x, p.y), 3, (0, 255, 0), -1)
            h, w = inset.shape[:2]
            img[-h:, -w:] = inset
        return img

    def emit(self, img: np.ndarray, tic: int) -> bool:
        if self.save_dir is not None and self.save_every and tic % self.save_every == 0:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.save_dir / f"frame_{tic:06d}.png"), img)
        if self.show:
            cv2.imshow("s1doom", img)
            return (cv2.waitKey(1) & 0xFF) != ord("q")
        return True

    def close(self) -> None:
        if self.show:
            cv2.destroyAllWindows()
