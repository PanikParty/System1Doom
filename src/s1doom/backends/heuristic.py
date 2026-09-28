"""Hand-written policy answering from the SAME serialized state the models see.

Serves as baseline, per-question fallback, and label source for calibration.
Probabilities are soft (not one-hot) so its answers look like a confident model.
"""
from __future__ import annotations

import time

from s1doom.backends.base import Answer, DecisionResult

WEAPON_PREF = ["shotgun", "chaingun", "super_shotgun", "plasma_rifle", "pistol",
               "rocket_launcher", "bfg", "fist"]


def _soft(options: list[str], pick: str, p: float = 0.85) -> dict[str, float]:
    if len(options) == 1:
        return {pick: 1.0}
    rest = (1 - p) / (len(options) - 1)
    return {o: (p if o == pick else rest) for o in options}


def heuristic_mode(state: dict, options: list[str]) -> str:
    s, sit = state["self"], state["situation"]
    vis = [m for m in state["monsters"] if m["visible"]]
    close_high = [m for m in vis if m["threat"] == "high" and m["range"] in ("melee", "close")]
    worth = [p for p in state["pickups"] if p["worth_it"]]

    def ok(m: str) -> bool:
        return m in options

    if vis:
        if ok("retreat") and s["health"] < 30 and close_high and len(vis) > 1:
            return "retreat"
        if ok("fight") and not (s.get("low_ammo") and s["weapon"] != "fist" and not any(m["aligned"] for m in vis)):
            return "fight"
    if ok("get_key") and any(p["kind"] == "key" for p in state["pickups"]):
        return "get_key"
    if ok("grab_pickup") and worth and (s["health"] < 60 or any(p["range"] in ("melee", "close") for p in worth)
                                        or "explore" not in options):
        return "grab_pickup"
    if ok("go_to_exit") and sit.get("exit_known") and sit.get("exit_reachable", True):
        return "go_to_exit"
    if ok("open_door") and state.get("doors"):
        return "open_door"
    for m in ("advance", "explore", "scan", "fight", "grab_pickup"):
        if ok(m):
            return m
    return options[0]


def heuristic_target(state: dict, options: list[str]) -> str:
    rng = {"melee": 0, "close": 1, "mid": 2, "far": 3}
    thr = {"high": 0, "medium": 1, "low": 2}
    rows = [m for m in state["monsters"] if m["ref"] in options]
    rows.sort(key=lambda m: (rng[m["range"]], not m["aligned"], thr[m["threat"]], abs(m["bearing_deg"])))
    return rows[0]["ref"] if rows else options[0]


def heuristic_pickup(state: dict, options: list[str]) -> str:
    rows = [p for p in state["pickups"] if p["ref"] in options and p["worth_it"]]
    return rows[0]["ref"] if rows else "none" if "none" in options else options[0]


def heuristic_in_danger(state: dict) -> float:
    s = state["self"]
    vis = [m for m in state["monsters"] if m["visible"]]
    close = [m for m in vis if m["range"] in ("melee", "close")]
    if s["taking_damage"] and (s["health"] < 40 or close):
        return 0.85
    if close and any(m["threat"] == "high" for m in close):
        return 0.7
    if s["taking_damage"]:
        return 0.6
    return 0.15


def heuristic_aggression(state: dict) -> int:
    h = state["self"]["health"]
    n = sum(1 for m in state["monsters"] if m["visible"])
    if h < 25 and n > 1:
        return 0
    if h < 50:
        return 1
    return 3 if n == 1 else 2


def heuristic_weapon(state: dict, options: list[str]) -> str:
    for w in WEAPON_PREF:
        if w in options:
            return w
    return options[0]


def answer_all(state: dict, questions: dict) -> dict[str, Answer]:
    out: dict[str, Answer] = {}
    for k, q in questions.items():
        if q["type"] == "choice":
            opts = list(q["criteria"])
            fn = {"mode": heuristic_mode, "target": heuristic_target, "pickup": heuristic_pickup,
                  "weapon": heuristic_weapon}.get(k)
            pick = fn(state, opts) if fn else opts[0]
            out[k] = Answer(type="choice", value=pick, probabilities=_soft(opts, pick))
        elif q["type"] == "noul":
            p = heuristic_in_danger(state) if k == "in_danger" else 0.5
            out[k] = Answer(type="noul", value=p, probabilities={"true": p, "false": 1 - p})
        else:
            n = len(q["criteria"])
            lvl = min(n - 1, heuristic_aggression(state)) if k == "aggression" else n // 2
            probs = _soft([str(i) for i in range(n)], str(lvl))
            out[k] = Answer(type="score", value=float(lvl), probabilities=probs)
    return out


class HeuristicBackend:
    name = "heuristic"

    async def system_one(self, state: dict, questions: dict) -> DecisionResult:
        t0 = time.perf_counter()
        ans = answer_all(state, questions)
        return DecisionResult(answers=ans, latency_ms=(time.perf_counter() - t0) * 1000, model="heuristic")
