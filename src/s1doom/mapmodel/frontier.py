"""Exploration memory: which nav nodes have been seen, and which are frontier.

"Seen" is derived from where the player has actually been: nodes within SEE_RADIUS of a
visited position AND in line of sight of it. Never from the omniscient map.
"""
from __future__ import annotations

import numpy as np

from s1doom.mapmodel.navgraph import NavGraph
from s1doom.mapmodel.visibility import Visibility

SEE_RADIUS = 384.0


class Exploration:
    def __init__(self, g: NavGraph, vis: Visibility) -> None:
        self.g = g
        self.vis = vis
        self.seen = np.zeros(g.n, dtype=bool)
        self.visited = np.zeros(g.n, dtype=bool)
        self._last = None
        self.failed: dict[int, int] = {}  # frontier node -> times planning to it failed

    def reset(self) -> None:
        self.seen[:] = False
        self.visited[:] = False
        self._last = None
        self.failed.clear()

    def update(self, x: float, y: float) -> None:
        if self._last is not None and (x - self._last[0]) ** 2 + (y - self._last[1]) ** 2 < 48 ** 2:
            return
        self._last = (x, y)
        d2 = (self.g.xs - x) ** 2 + (self.g.ys - y) ** 2
        near = np.nonzero(d2 < SEE_RADIUS ** 2)[0]
        if len(near):
            pts = np.stack([self.g.xs[near], self.g.ys[near]], axis=1)
            ok = self.vis.los_many(x, y, pts)
            self.seen[near[ok]] = True
        self.visited[np.nonzero(d2 < 64 ** 2)[0]] = True

    def frontier(self) -> np.ndarray:
        """Unseen nodes adjacent to seen nodes."""
        out = []
        for i in np.nonzero(~self.seen)[0]:
            for e in self.g.adj[i]:
                if self.seen[e.to]:
                    out.append(i)
                    break
        return np.array(out, dtype=np.int64)

    def seen_fraction(self) -> float:
        return float(self.seen.mean()) if len(self.seen) else 0.0

    def point_seen(self, x: float, y: float, radius: float = 96.0) -> bool:
        d2 = (self.g.xs - x) ** 2 + (self.g.ys - y) ** 2
        near = d2 < radius ** 2
        return bool(self.seen[near].any())
