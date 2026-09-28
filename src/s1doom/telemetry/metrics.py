"""Episode metrics and aggregate statistics (mean ± 95% CI)."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class EpisodeMetrics:
    scenario: str
    backend: str
    seed: int
    episode: int
    kills: int = 0
    damage_taken: float = 0.0
    hits: int = 0
    survival_tics: int = 0
    health_end: float = 0.0
    armor_end: float = 0.0
    items: int = 0
    secrets: int = 0
    dead: bool = False
    completed: bool = False
    timeout: bool = False
    time_to_exit_tics: int | None = None
    decisions: int = 0
    model_calls: int = 0
    backend_errors: int = 0
    fallback_rate: float = 0.0          # share of model-asked answers replaced by the heuristic
    agreement: dict[str, float] = field(default_factory=dict)   # per question vs heuristic
    mode_switches_per_min: float = 0.0
    latency_ms: list[float] = field(default_factory=list)
    staleness_ms: list[float] = field(default_factory=list)
    dropped_stale: int = 0
    input_tokens: int = 0
    state_tokens_max: int = 0
    explored: float | None = None
    stuck_events: int = 0
    mode_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        lat = np.array(self.latency_ms) if self.latency_ms else None
        d["latency_p50"] = float(np.percentile(lat, 50)) if lat is not None else None
        d["latency_p95"] = float(np.percentile(lat, 95)) if lat is not None else None
        d["latency_p99"] = float(np.percentile(lat, 99)) if lat is not None else None
        del d["latency_ms"]
        st = np.array(self.staleness_ms) if self.staleness_ms else None
        d["staleness_p50"] = float(np.percentile(st, 50)) if st is not None else None
        d["staleness_p95"] = float(np.percentile(st, 95)) if st is not None else None
        del d["staleness_ms"]
        return d


class DecisionStats:
    def __init__(self) -> None:
        self.asked = 0
        self.fallback = 0
        self.agree: Counter = Counter()
        self.total: Counter = Counter()
        self.modes: list[str] = []

    def add(self, sources: dict[str, str], asked: set[str], model_answers: dict, heur_answers: dict, mode: str) -> None:
        for k in asked:
            self.asked += 1
            if sources.get(k) == "fallback":
                self.fallback += 1
            if k in model_answers and k in heur_answers:
                self.total[k] += 1
                self.agree[k] += int(model_answers[k].argmax == heur_answers[k].argmax)
        self.modes.append(mode)

    def fallback_rate(self) -> float:
        return self.fallback / self.asked if self.asked else 0.0

    def agreement(self) -> dict[str, float]:
        return {k: self.agree[k] / n for k, n in self.total.items() if n}

    def switches_per_min(self, tics: int) -> float:
        sw = sum(1 for a, b in zip(self.modes, self.modes[1:]) if a != b)
        return sw / max(tics / 35 / 60, 1e-6)


def mean_ci(xs: list[float]) -> tuple[float, float]:
    """Mean and 95% CI half-width (t-approx 1.96 for n>=30, else a small-n table)."""
    a = np.array([x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan")
    if len(a) == 1:
        return float(a[0]), float("nan")
    tt = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26,
          15: 2.14, 20: 2.09, 25: 2.06, 30: 2.04}
    n = len(a)
    t = next((v for k, v in sorted(tt.items()) if n <= k), 1.96)
    return float(a.mean()), float(t * a.std(ddof=1) / math.sqrt(n))
