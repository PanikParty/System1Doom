"""runs/<id>/summary.json: per-episode metrics + aggregates."""
from __future__ import annotations

import json
from pathlib import Path

from s1doom.telemetry.metrics import EpisodeMetrics, mean_ci

AGG_FIELDS = ["kills", "damage_taken", "survival_tics", "health_end", "items", "fallback_rate",
              "mode_switches_per_min", "latency_p50", "latency_p95", "input_tokens", "explored",
              "state_tokens_max", "time_to_exit_tics"]


def aggregate(rows: list[dict]) -> dict:
    agg = {"n": len(rows)}
    for f in AGG_FIELDS:
        vals = [r.get(f) for r in rows if r.get(f) is not None]
        if vals:
            mu, ci = mean_ci(vals)
            agg[f] = {"mean": mu, "ci95": ci}
    agg["completed_rate"] = sum(bool(r["completed"]) for r in rows) / max(len(rows), 1)
    agg["death_rate"] = sum(bool(r["dead"]) for r in rows) / max(len(rows), 1)
    agree: dict[str, list[float]] = {}
    for r in rows:
        for k, v in (r.get("agreement") or {}).items():
            agree.setdefault(k, []).append(v)
    agg["agreement"] = {k: sum(v) / len(v) for k, v in agree.items()}
    return agg


def write_summary(run_dir: Path, eps: list[EpisodeMetrics]) -> dict:
    rows = [e.to_dict() for e in eps]
    s = {"episodes": rows, "aggregate": aggregate(rows)}
    (run_dir / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    return s
