"""Recorded Jev and Laya /v1/systemone bodies normalize to the same DecisionResult."""
import pytest

from s1doom.backends.base import BackendError, normalize_response, validate_questions


@pytest.mark.parametrize("name", ["jev_response.json", "laya_response.json", "semif_response.json"])
def test_fixture_normalizes(fixture_json, name):
    fx = fixture_json(name)
    qs = fx["request"]["questions"]
    res = normalize_response(qs, fx["response"], 100.0)
    assert set(res.answers) == set(qs)
    mode = res.answers["mode"]
    assert mode.type == "choice" and mode.value in qs["mode"]["criteria"]
    assert set(mode.probabilities) == set(qs["mode"]["criteria"])
    assert sum(mode.probabilities.values()) == pytest.approx(1.0, abs=0.02)
    noul = res.answers["in_danger"]
    assert noul.type == "noul" and 0 <= noul.value <= 1
    # Jev noul has no confidence -> derived as max(p, 1-p)
    assert noul.raw_confidence == pytest.approx(max(noul.value, 1 - noul.value), abs=1e-3)
    score = res.answers["aggression"]
    assert score.type == "score" and 0 <= score.value <= 3
    assert set(score.probabilities) == {"0", "1", "2", "3"}
    assert res.usage.get("input_tokens", 0) > 0
    # gate on max_prob, never on the backend's own `confidence` (different definitions)
    assert mode.max_prob == pytest.approx(max(mode.probabilities.values()))


def test_laya_distribution_list_is_mapped():
    qs = {"m": {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}}}
    res = normalize_response(qs, {"answers": {"m": {"type": "choice", "distribution": [0.2, 0.8]}}}, 1)
    assert res.answers["m"].value == "b"
    assert res.answers["m"].probabilities == {"a": 0.2, "b": 0.8}


def test_missing_answer_raises():
    qs = {"m": {"type": "noul", "instructions": "x"}}
    with pytest.raises(BackendError):
        normalize_response(qs, {"answers": {}}, 1)


def test_validate_limits():
    validate_questions({"s": {"type": "score", "instructions": "x", "criteria": ["a", "b"]}})
    with pytest.raises(ValueError):
        validate_questions({"s": {"type": "score", "instructions": "x", "criteria": ["a"]}})
    with pytest.raises(ValueError):
        validate_questions({"c": {"type": "choice", "instructions": "x",
                                  "criteria": {str(i): None for i in range(256)}}})


def test_semif_server_mapping_roundtrip():
    """scripts/semif_serve.py maps our question types to SemIf rows and back to Jev shape."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("semif_serve", Path(__file__).parents[1] / "scripts" / "semif_serve.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    qs = {"mode": {"type": "choice", "instructions": "x", "criteria": {"fight": "f", "explore": None}},
          "danger": {"type": "noul", "instructions": "y"},
          "aggr": {"type": "score", "instructions": "z", "criteria": ["lo", "mid", "hi"]}}
    rows = m.to_rows({"s": 1}, qs)
    assert [len(r["options"]) for r in rows] == [2, 2, 3]
    fake = {"mode": ([.2, .8], ["fight", "explore"]), "danger": ([.7, .3], ["true", "false"]),
            "aggr": ([.1, .1, .8], ["0", "1", "2"])}
    body = {"answers": {q: m.to_answer(qs[q], {"option_ids": fake[q][1], "probabilities": fake[q][0]}) for q in qs},
            "usage": {"input_tokens": 10}}
    res = normalize_response(qs, body, 1.0)
    assert res.answers["mode"].value == "explore" and res.answers["danger"].value == 0.7
    assert res.answers["aggr"].value == pytest.approx(1.7)
