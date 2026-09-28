"""Build atomic typed questions whose options are only the actions legal right now."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from s1doom.decision.state_builder import BuiltState

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


@dataclass
class PromptSet:
    version: str
    rules: str
    spec: dict

    @classmethod
    def load(cls, version: str = "v1") -> "PromptSet":
        d = yaml.safe_load((PROMPTS_DIR / version / "questions.yaml").read_text())
        return cls(d["version"], d["rules"].strip(), d["questions"])


def legal_modes(bs: BuiltState, allowed: list[str], has_level: bool) -> list[str]:
    st = bs.state
    sit = st["situation"]
    vis = [m for m in st["monsters"] if m["visible"]]
    legal = []
    for m in allowed:
        if m == "fight" and vis:
            legal.append(m)
        elif m == "retreat" and vis:
            legal.append(m)
        elif m == "scan":
            legal.append(m)
        elif m == "grab_pickup" and st["pickups"]:
            legal.append(m)
        elif m == "explore":
            legal.append(m)
        elif m == "go_to_exit" and sit.get("exit_known") and sit.get("exit_reachable", True):
            legal.append(m)
        elif m == "open_door" and st.get("doors"):
            legal.append(m)
        elif m == "get_key" and any(p["kind"] == "key" for p in st["pickups"]):
            legal.append(m)
        elif m == "advance":
            legal.append(m)
    return legal or [allowed[0]]


def _fmt(tpl: str, row: dict) -> str:
    try:
        return tpl.format(**row)
    except KeyError:
        return str(row)


def build_questions(ps: PromptSet, bs: BuiltState, which: set[str], modes: list[str],
                    self_consistency: bool = False) -> tuple[dict, dict]:
    """Returns (questions to send, forced answers for single-option questions)."""
    q: dict = {}
    forced: dict = {}
    spec = ps.spec
    st = bs.state
    if "mode" in which:
        if len(modes) == 1:
            forced["mode"] = modes[0]
        else:
            s = spec["mode"]
            q["mode"] = {"type": "choice", "instructions": s["instructions"],
                         "criteria": {m: s["options"][m] for m in modes}}
            if self_consistency and s.get("alt_instructions"):
                q["mode_alt"] = {"type": "choice", "instructions": s["alt_instructions"],
                                 "criteria": dict(q["mode"]["criteria"])}
    if "target" in which:
        vis = [m for m in st["monsters"] if m["visible"]]
        if len(vis) == 1:
            forced["target"] = vis[0]["ref"]
        elif len(vis) > 1:
            s = spec["target"]
            q["target"] = {"type": "choice", "instructions": s["instructions"],
                           "criteria": {m["ref"]: _fmt(s["option_template"], m) for m in vis}}
    if "pickup" in which and st["pickups"]:
        s = spec["pickup"]
        crit = {p["ref"]: _fmt(s["option_template"], p) for p in st["pickups"]}
        crit["none"] = s["none_option"]
        q["pickup"] = {"type": "choice", "instructions": s["instructions"], "criteria": crit}
    if "in_danger" in which:
        s = spec["in_danger"]
        q["in_danger"] = {"type": "noul", "instructions": s["instructions"], "criteria": s["criteria"]}
    if "aggression" in which:
        s = spec["aggression"]
        q["aggression"] = {"type": "score", "instructions": s["instructions"], "criteria": s["criteria"]}
    if "weapon" in which and len(bs.weapons) > 1:
        s = spec["weapon"]
        crit = {}
        for w in bs.weapons:
            ammo = bs.weapon_ammo.get(w)
            row = {"name": w.replace("_", " "), "ammo": "melee, no" if w in ("fist", "chainsaw") else ammo}
            crit[w] = _fmt(s["option_template"], row)
        q["weapon"] = {"type": "choice", "instructions": s["instructions"], "criteria": crit}
    return q, forced
