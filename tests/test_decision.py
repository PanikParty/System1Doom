"""State builder, question legality, heuristic, policy gating/hysteresis, calibration."""
import math

import numpy as np
import pytest

from s1doom.backends.base import Answer, DecisionResult
from s1doom.backends.heuristic import answer_all
from s1doom.decision.calibration import Calibrator, fit, temper
from s1doom.decision.policy import Policy, PolicyConfig
from s1doom.decision.questions import PromptSet, build_questions, legal_modes
from s1doom.decision.scheduler import QuestionScheduler
from s1doom.decision.state_builder import StateContext, build_state
from s1doom.perception.entities import Entity, Percept


def percept(monsters=(), pickups=(), health=100, ammo=50):
    ents = []
    for i, (name, dist, bearing, aligned) in enumerate(monsters):
        rad = math.radians(bearing)
        e = Entity(i + 1, name, "monster", dist * math.cos(rad), dist * math.sin(rad), 0)
        e.refresh(0, 0, 0)
        e.aligned = aligned
        ents.append(e)
    for j, (name, kind, dist) in enumerate(pickups):
        e = Entity(100 + j, name, kind, dist, 0, 0)
        e.refresh(0, 0, 0)
        ents.append(e)
    vars_ = {"WEAPON2": 1, "AMMO1": ammo, "WEAPON3": 0}
    return Percept(tic=10, x=0, y=0, z=0, angle=0, health=health, armor=0, weapon=2, ammo=ammo,
                   vars=vars_, entities=ents)


PS = PromptSet.load("v1")
ALL_MODES = ["fight", "retreat", "grab_pickup", "explore", "go_to_exit", "open_door", "get_key"]


def test_state_is_compact_and_bucketed():
    bs = build_state(percept([("DoomImp", 300, 20, False), ("Zombieman", 1500, -40, False)]),
                     StateContext(rules=PS.rules))
    st = bs.state
    assert st["monsters"][0]["kind"] == "imp" and st["monsters"][0]["range"] == "close"
    assert st["monsters"][0]["side"] == "left" and st["monsters"][1]["side"] == "right"
    assert st["monsters"][0]["threat"] in ("low", "medium", "high")
    assert "rules" in st and bs.tokens < 600
    assert set(bs.refs) == {"m1", "m2"}


def test_token_budget_trims():
    many = [("DoomImp", 100 + 50 * i, i * 10, False) for i in range(5)]
    items = [("Stimpack", "pickup_health", 200 + 40 * i) for i in range(3)]
    bs = build_state(percept(many, items), StateContext(rules=PS.rules, budget=250,
                                                         recent=["took 5 damage"] * 3))
    assert bs.tokens <= 250 or len(bs.state["monsters"]) == 2
    assert all(r["ref"] in bs.refs for r in bs.state["monsters"])


def test_options_are_only_legal_actions():
    bs = build_state(percept(), StateContext())
    modes = legal_modes(bs, ALL_MODES, has_level=True)
    assert "fight" not in modes and "retreat" not in modes and "open_door" not in modes
    assert "go_to_exit" not in modes      # exit not known
    q, forced = build_questions(PS, bs, {"mode", "target", "in_danger"}, modes)
    assert "target" not in q
    bs2 = build_state(percept([("DoomImp", 300, 0, True), ("Demon", 200, 30, False)]), StateContext())
    modes2 = legal_modes(bs2, ALL_MODES, True)
    q2, _ = build_questions(PS, bs2, {"mode", "target"}, modes2)
    assert set(q2["mode"]["criteria"]) >= {"fight", "retreat"}
    assert set(q2["target"]["criteria"]) == {"m1", "m2"}


def test_single_option_is_forced_not_asked():
    bs = build_state(percept([("DoomImp", 300, 0, True)]), StateContext())
    q, forced = build_questions(PS, bs, {"mode", "target"}, ["fight"])
    assert forced == {"mode": "fight", "target": "m1"} and q == {}


def test_scheduler_asks_mode_and_danger_always():
    s = QuestionScheduler()
    bs = build_state(percept(), StateContext())
    assert s.pick(bs, can_move=True) == {"mode", "in_danger"}


def test_heuristic_prefers_close_aligned_fight():
    bs = build_state(percept([("DoomImp", 300, 0, True), ("Zombieman", 1500, 40, False)]), StateContext())
    q, _ = build_questions(PS, bs, {"mode", "target", "in_danger"}, legal_modes(bs, ALL_MODES, True))
    ans = answer_all(bs.state, q)
    assert ans["mode"].value == "fight" and ans["target"].value == "m1"


def _res(**answers):
    return DecisionResult(answers=answers)


def test_policy_falls_back_below_floor_and_labels_it():
    bs = build_state(percept([("DoomImp", 300, 0, True)]), StateContext())
    q = {"mode": {"type": "choice", "instructions": "x", "criteria": {"fight": None, "explore": None}},
         "in_danger": {"type": "noul", "instructions": "x"}}
    pol = Policy("test")
    low = _res(mode=Answer(type="choice", value="explore", probabilities={"fight": 0.45, "explore": 0.55}),
               in_danger=Answer(type="noul", value=0.52, probabilities={"true": .52, "false": .48}))
    d, _, heur = pol.decide(0, bs.state, q, {}, low, True, ["fight", "explore"])
    assert d.sources["mode"] == "fallback" and d.mode == heur["mode"].value
    assert d.sources["in_danger"] == "fallback"
    hi = _res(mode=Answer(type="choice", value="explore", probabilities={"fight": 0.1, "explore": 0.9}),
              in_danger=Answer(type="noul", value=0.9, probabilities={"true": .9, "false": .1}))
    pol2 = Policy("test")
    d2, _, _ = pol2.decide(0, bs.state, q, {}, hi, True, ["fight", "explore"])
    assert d2.sources["mode"] == "model" and d2.mode == "explore" and d2.in_danger


def test_policy_hysteresis_holds_mode():
    bs = build_state(percept([("DoomImp", 300, 0, True)]), StateContext())
    q = {"mode": {"type": "choice", "instructions": "x", "criteria": {"fight": None, "explore": None}}}
    pol = Policy("t", PolicyConfig(min_commit_tics=12))
    r1 = _res(mode=Answer(type="choice", value="fight", probabilities={"fight": .9, "explore": .1}))
    r2 = _res(mode=Answer(type="choice", value="explore", probabilities={"fight": .1, "explore": .9}))
    pol.decide(0, bs.state, q, {}, r1, False, ["fight", "explore"])
    d, _, _ = pol.decide(8, bs.state, q, {}, r2, False, ["fight", "explore"])
    assert d.mode == "fight" and d.sources["mode"] == "held"
    d, _, _ = pol.decide(8, bs.state, q, {}, r2, True, ["fight", "explore"])  # trigger overrides
    assert d.mode == "explore"


def test_temper_and_fit_recover_overconfidence():
    assert temper({"a": .9, "b": .1}, 1.0)["a"] == pytest.approx(.9)
    assert temper({"a": .9, "b": .1}, 3.0)["a"] < .9
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(400):  # model says 0.95 but is right only 70% of the time
        label = "a" if rng.random() < 0.7 else "b"
        rows.append(({"a": 0.95, "b": 0.05}, label))
    temps, rep = fit({"x|choice|2": rows})
    assert temps["x|choice|2"] > 1.5
    assert rep["x|choice|2"]["ece_after"] < rep["x|choice|2"]["ece_before"]
    cal = Calibrator({"x|choice|2": temps["x|choice|2"]})
    a = cal.apply("x", Answer(type="choice", value="a", probabilities={"a": .95, "b": .05}))
    assert a.probabilities["a"] == pytest.approx(0.7, abs=0.08)
