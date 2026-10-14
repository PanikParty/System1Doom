"""Per-(backend, question type, n_options) temperature scaling, plus an optional bias for nouls.

Fit offline from logged runs: each decision row logs the model's raw answer and the
shadow heuristic's answer; the label is the heuristic answer (agreement labels) or, for
`in_danger`, whether damage was actually taken in the following period (outcome labels).
With `bias=True`, noul groups get Platt scaling (logit' = logit/T + b), which also fixes an
offset: Laya's in_danger is always 0.52-0.66 while damage follows only ~4% of the time.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from s1doom.backends.base import Answer

T_GRID = np.exp(np.linspace(math.log(0.25), math.log(12.0), 80))


def key(backend: str, qtype: str, k: int) -> str:
    return f"{backend}|{qtype}|{k}"


class Calibrator:
    def __init__(self, temps: dict[str, float] | None = None, biases: dict[str, float] | None = None) -> None:
        self.temps = temps or {}
        self.biases = biases or {}

    @classmethod
    def load(cls, path: str | None) -> "Calibrator":
        if not path or not Path(path).exists():
            return cls()
        cfg = yaml.safe_load(Path(path).read_text())
        return cls(cfg.get("temperatures", {}), cfg.get("biases", {}))

    def temperature(self, backend: str, a: Answer) -> float:
        k = 2 if a.type == "noul" else len(a.probabilities)
        return self.temps.get(key(backend, a.type, k), self.temps.get(key(backend, a.type, 0), 1.0))

    def apply(self, backend: str, a: Answer) -> Answer:
        T = self.temperature(backend, a)
        b = self.biases.get(key(backend, a.type, 2), 0.0) if a.type == "noul" else 0.0
        if T == 1.0 and b == 0.0:
            return a
        if a.type == "noul":
            p = min(max(float(a.value), 1e-6), 1 - 1e-6)
            z = math.log(p / (1 - p)) / T + b
            q = 1 / (1 + math.exp(-z))
            return Answer(type="noul", value=q, probabilities={"true": q, "false": 1 - q},
                          raw_confidence=a.raw_confidence)
        probs = temper(a.probabilities, T)
        val = a.value
        if a.type == "choice":
            val = max(probs, key=probs.get)
        else:
            val = sum(int(k) * v for k, v in probs.items())
        return Answer(type=a.type, value=val, probabilities=probs, raw_confidence=a.raw_confidence)


def temper(probs: dict[str, float], T: float) -> dict[str, float]:
    ks = list(probs)
    p = np.clip(np.array([probs[k] for k in ks], dtype=float), 1e-9, 1)
    z = np.log(p) / T
    z = np.exp(z - z.max())
    z /= z.sum()
    return dict(zip(ks, z.tolist()))


def _nll(p_label: np.ndarray) -> float:
    return float(-np.log(np.clip(p_label, 1e-9, 1)).mean())


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def collect(run_globs: list[str], labels: str = "outcome") -> dict[str, list[tuple[dict, str]]]:
    """group key -> list of (probabilities dict, label option).

    labels="outcome": only questions with a ground-truth outcome (in_danger vs damage actually
    taken next period). labels="agreement": also use the shadow heuristic's answer as the label
    for choice/score questions. That measures imitation of the heuristic, not correctness: Jev
    disagrees with the heuristic's MAP01 mode 80% of the time yet completes the level.
    """
    groups: dict[str, list[tuple[dict, str]]] = defaultdict(list)
    files: list[Path] = []
    for g in run_globs:
        p = Path(g)
        if p.is_file():
            files.append(p)
        else:
            files += sorted(Path().glob(f"{g.rstrip('/')}/**/calls.jsonl")) if not any(c in g for c in "*?[") \
                else [f for d in Path().glob(g) for f in Path(d).glob("**/calls.jsonl")]
    for f in files:
        for line in f.read_text().splitlines():
            row = json.loads(line)
            if row.get("kind") != "decision" or row.get("backend") in ("heuristic", "random", "replay"):
                continue
            raw = row.get("raw_answers") or {}
            heur = row.get("heuristic_answers") or {}
            outcome = row.get("outcome") or {}
            for q, a in raw.items():
                if q not in heur:
                    continue
                if labels == "outcome" and q not in outcome:
                    continue
                h = heur[q]
                if a["type"] == "noul":
                    label = "true" if (outcome.get(q, h["value"] >= 0.5)) else "false"
                    probs = {"true": a["value"], "false": 1 - a["value"]}
                    k = 2
                elif a["type"] == "choice":
                    label, probs, k = h["value"], a["probabilities"], len(a["probabilities"])
                else:
                    label = str(int(round(h["value"])))
                    probs, k = a["probabilities"], len(a["probabilities"])
                groups[key(row["backend"], a["type"], k)].append((probs, label))
    return groups


def platt(p_true: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """Logistic regression of y on logit(p): returns (T, b) with logit' = logit/T + b."""
    x = np.log(np.clip(p_true, 1e-6, 1 - 1e-6) / np.clip(1 - p_true, 1e-6, 1))
    X = np.stack([x, np.ones_like(x)], 1)
    w = np.array([1.0, 0.0])
    for _ in range(iters):  # Newton / IRLS, lightly ridged so separable data stays finite
        q = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (q * (1 - q))[:, None]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, X.T @ (y - q))
        w += step
        if np.abs(step).max() < 1e-8:
            break
    if w[0] <= 1e-3:  # no usable ranking signal: keep the probabilities' order, only shift them
        w[0] = 1e-3
    return float(1 / w[0]), float(w[1])


def fit(groups: dict[str, list[tuple[dict, str]]], min_n: int = 30,
        bias: bool = False) -> tuple[dict[str, float], dict]:
    """Fit temperatures (and, with bias=True, noul biases, returned in report[gk]["b"])."""
    temps, report = {}, {}
    for gk, rows in groups.items():
        if len(rows) < min_n:
            continue
        if bias and gk.split("|")[1] == "noul":
            pt = np.array([p["true"] for p, _ in rows], dtype=float)
            y = np.array([lbl == "true" for _, lbl in rows], dtype=float)
            T, b = platt(pt, y)
            p1 = np.array([Calibrator({gk: T}, {gk: b}).apply(gk.split("|")[0], Answer(type="noul", value=v)).value
                           for v in pt])
            corr0, corr1 = ((pt >= 0.5) == (y == 1)).astype(float), ((p1 >= 0.5) == (y == 1)).astype(float)
            temps[gk] = round(T, 4)
            report[gk] = {"n": len(rows), "T": round(T, 3), "b": round(b, 4), "acc": float(corr0.mean()),
                          "acc_after": float(corr1.mean()), "brier_before": float(((pt - y) ** 2).mean()),
                          "brier_after": float(((p1 - y) ** 2).mean()),
                          "ece_before": ece(np.maximum(pt, 1 - pt), corr0), "ece_after": ece(np.maximum(p1, 1 - p1), corr1),
                          "conf_before": np.maximum(pt, 1 - pt).tolist(), "conf_after": np.maximum(p1, 1 - p1).tolist(),
                          "correct": corr0.tolist()}
            continue
        best_T, best = 1.0, math.inf
        for T in T_GRID:
            pl = np.array([temper(p, T).get(lbl, 1e-9) for p, lbl in rows])
            n = _nll(pl)
            if n < best:
                best, best_T = n, float(T)
        conf0 = np.array([max(p.values()) for p, _ in rows])
        corr = np.array([max(p, key=p.get) == lbl for p, lbl in rows], dtype=float)
        conf1 = np.array([max(temper(p, best_T).values()) for p, _ in rows])
        temps[gk] = round(best_T, 4)
        report[gk] = {"n": len(rows), "T": round(best_T, 3), "acc": float(corr.mean()),
                      "ece_before": ece(conf0, corr), "ece_after": ece(conf1, corr),
                      "conf_before": conf0.tolist(), "conf_after": conf1.tolist(), "correct": corr.tolist()}
    return temps, report
