"""Record raw /v1/systemone response bodies from Jev and Laya for contract tests.

usage: uv run --extra laya python scripts/record_fixtures.py [jev] [laya]
Writes tests/fixtures/<backend>_response.json (request + raw body; no credentials).
"""
import asyncio
import json
import sys
from pathlib import Path

from s1doom.backends.factory import make_backend

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
REQ_STATE = {
    "self": {"health": 62, "armor": 0, "weapon": "shotgun", "ammo": 11, "low_ammo": False, "taking_damage": True},
    "situation": {"monsters_visible": 2, "nearest_threat_range": "close", "aligned_target": True,
                  "objective": "reach_exit", "exit_known": False},
    "monsters": [
        {"ref": "m1", "kind": "imp", "visible": True, "range": "close", "bearing_deg": 2, "aligned": True, "side": "ahead", "threat": "high"},
        {"ref": "m2", "kind": "zombieman", "visible": True, "range": "far", "bearing_deg": -40, "aligned": False, "side": "right", "threat": "low"},
    ],
    "pickups": [{"ref": "p1", "kind": "health", "item": "Medikit", "range": "mid", "side": "left", "worth_it": True}],
    "rules": "bearing_deg: 0 = straight ahead, positive = LEFT. aligned=true means shooting now hits.",
}
REQ_Q = {
    "mode": {"type": "choice", "instructions": "What should the player do now?",
             "criteria": {"fight": "Attack a visible monster.", "retreat": "Back away.", "grab_pickup": "Get the pickup."}},
    "target": {"type": "choice", "instructions": "Which monster to attack?",
               "criteria": {"m1": "imp, close, aligned", "m2": "zombieman, far, right"}},
    "in_danger": {"type": "noul", "instructions": "Will the player take significant damage in the next second?"},
    "aggression": {"type": "score", "instructions": "How aggressively should the player fight?",
                   "criteria": ["flee", "cautious", "engage", "all-in"]},
}


async def main(which):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in which:
        b = make_backend({"jev": "jev", "laya": "laya-inproc", "semif": "semif",
                           "clef": "clef-flash"}[name])
        res = await b.system_one(REQ_STATE, REQ_Q)
        (OUT / f"{name}_response.json").write_text(json.dumps(
            {"request": {"state": REQ_STATE, "questions": REQ_Q}, "response": res.raw}, indent=2))
        print(name, "ok", res.latency_ms, {k: a.value for k, a in res.answers.items()})
        if hasattr(b, "aclose"):
            await b.aclose()


asyncio.run(main(sys.argv[1:] or ["jev", "laya"]))
