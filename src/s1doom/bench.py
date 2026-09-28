"""Backend latency benchmark over real recorded states."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from s1doom.backends.base import BackendError
from s1doom.backends.factory import make_backend

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
ORDER = ["mode", "in_danger", "target", "aggression", "pickup", "weapon"]


def load_states(src: str | None, n: int) -> list[tuple[dict, dict]]:
    files = [Path(src)] if src and Path(src).is_file() else sorted((Path(src) if src else RUNS_DIR).glob("**/calls.jsonl"))
    out = []
    for f in files:
        for line in f.read_text().splitlines():
            row = json.loads(line)
            if row.get("kind") == "decision" and row.get("questions"):
                out.append((row["state"], row["questions"]))
            if len(out) >= n:
                return out
    return out


def _pad_questions(qs: dict, k: int) -> dict:
    """Exactly k questions: take what was asked, pad with generic ones from the same state."""
    out = {q: qs[q] for q in ORDER if q in qs}
    extra = [
        ("in_danger", {"type": "noul", "instructions": "Is the player in danger?"}),
        ("aggression", {"type": "score", "instructions": "How aggressive should the player be?",
                        "criteria": ["flee", "cautious", "engage", "all-in"]}),
        ("low_health", {"type": "noul", "instructions": "Is the player's health low?"}),
        ("should_reload", {"type": "noul", "instructions": "Is the player low on ammo?"}),
        ("urgency", {"type": "score", "instructions": "How urgent is the situation?",
                     "criteria": ["calm", "tense", "critical"]}),
        ("monster_ahead", {"type": "noul", "instructions": "Is there a monster straight ahead?"}),
    ]
    for name, q in extra:
        if len(out) >= k:
            break
        out.setdefault(name, q)
    return dict(list(out.items())[:k])


async def run_bench(backend: str, url: str | None, states_src: str | None, n: int, qcounts: list[int]) -> dict:
    states = load_states(states_src, n)
    if not states:
        raise SystemExit("no recorded states found; run `s1doom play` first")
    b = make_backend(backend, url=url)
    res = {"backend": backend, "url": url, "n_states": len(states), "by_qcount": {}}
    # warmup
    for s, q in states[:3]:
        try:
            await b.system_one(s, _pad_questions(q, 1))
        except BackendError:
            pass
    for k in qcounts:
        lat, errs = [], 0
        for s, q in states:
            qs = _pad_questions(q, k)
            t0 = time.perf_counter()
            try:
                await b.system_one(s, qs)
                lat.append((time.perf_counter() - t0) * 1000)
            except BackendError:
                errs += 1
        a = np.array(lat) if lat else np.array([np.nan])
        res["by_qcount"][k] = {"n": len(lat), "errors": errs, "p50": float(np.percentile(a, 50)),
                               "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
                               "mean": float(a.mean())}
    if hasattr(b, "aclose"):
        await b.aclose()
    return res
