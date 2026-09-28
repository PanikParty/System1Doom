"""Gating, commitment/hysteresis and fallback.

Every question's final answer carries a `source`: model | fallback | forced | held.
Fallbacks are never presented as model output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from s1doom.backends.base import Answer, DecisionResult
from s1doom.backends.heuristic import answer_all
from s1doom.decision.calibration import Calibrator


@dataclass
class PolicyConfig:
    confidence_floor: float = 0.55          # for noul (max(p, 1-p))
    choice_margin: float = 0.30             # choice/score floor = 1/k + margin * (1 - 1/k)
    min_commit_tics: int = 12
    switch_margin: float = 0.15
    fallback_enabled: bool = True


@dataclass
class Decision:
    mode: str
    target: str | None = None
    pickup: str | None = None
    in_danger: bool = False
    aggression: int = 2
    weapon: str | None = None
    sources: dict[str, str] = field(default_factory=dict)
    probs: dict[str, dict[str, float]] = field(default_factory=dict)
    conf: dict[str, float] = field(default_factory=dict)


def floor_for(a: Answer, cfg: PolicyConfig) -> float:
    if a.type == "noul":
        return cfg.confidence_floor
    k = max(2, len(a.probabilities))
    return 1 / k + cfg.choice_margin * (1 - 1 / k)


@dataclass
class Policy:
    backend_name: str
    cfg: PolicyConfig = field(default_factory=PolicyConfig)
    calibrator: Calibrator = field(default_factory=Calibrator)
    current: Decision | None = None
    committed_at: int = -10 ** 9

    def reset(self) -> None:
        self.current = None
        self.committed_at = -10 ** 9

    def decide(self, tic: int, state: dict, questions: dict, forced: dict, result: DecisionResult | None,
               triggered: bool, legal_modes: list[str]) -> tuple[Decision, dict[str, Answer], dict[str, Answer]]:
        heur = answer_all(state, questions)
        calibrated: dict[str, Answer] = {}
        final: dict[str, tuple[object, str]] = {}
        d = Decision(mode=legal_modes[0])
        for k, q in questions.items():
            a = result.answers.get(k) if result is not None else None
            if a is not None:
                a = self.calibrator.apply(self.backend_name, a)
                calibrated[k] = a
                d.probs[k] = a.probabilities
                d.conf[k] = a.max_prob
            if a is not None and (not self.cfg.fallback_enabled or a.max_prob >= floor_for(a, self.cfg)):
                final[k] = (a.argmax, "model")
            else:
                final[k] = (heur[k].argmax, "fallback")
        for k, v in forced.items():
            final[k] = (v, "forced")

        # self-consistency: two phrasings of `mode` must agree, else the model's answer is not used
        if "mode_alt" in final:
            alt, alt_src = final.pop("mode_alt")
            if final.get("mode", (None, ""))[1] == "model" and (alt_src != "model" or alt != final["mode"][0]):
                final["mode"] = (heur["mode"].argmax, "fallback")
                d.sources["mode_consistency"] = "disagree"
        # hysteresis on mode
        prev = self.current
        mode, src = final.get("mode", (legal_modes[0], "forced"))
        if (prev is not None and prev.mode != mode and prev.mode in legal_modes and not triggered
                and tic - self.committed_at < self.cfg.min_commit_tics):
            mode, src = prev.mode, "held"
        if prev is None or mode != prev.mode:
            self.committed_at = tic
        d.mode = mode
        d.sources["mode"] = src

        # hysteresis on target
        if "target" in final:
            tgt, tsrc = final["target"]
            if (prev is not None and prev.target and prev.target != tgt and "target" in calibrated
                    and prev.target in calibrated["target"].probabilities):
                p = calibrated["target"].probabilities
                if p.get(tgt, 0) - p.get(prev.target, 0) < self.cfg.switch_margin:
                    tgt, tsrc = prev.target, "held"
            d.target, d.sources["target"] = tgt, tsrc
        if "pickup" in final:
            v, s = final["pickup"]
            d.pickup, d.sources["pickup"] = (None if v == "none" else v), s
        if "in_danger" in final:
            v, s = final["in_danger"]
            d.in_danger, d.sources["in_danger"] = bool(v), s
        if "aggression" in final:
            v, s = final["aggression"]
            d.aggression, d.sources["aggression"] = int(v) if not isinstance(v, str) else int(v), s
        elif prev is not None:
            d.aggression = prev.aggression
        if "weapon" in final:
            d.weapon, d.sources["weapon"] = final["weapon"]
        self.current = d
        return d, calibrated, heur
