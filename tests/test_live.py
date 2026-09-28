"""Live backend calls. Skipped by default; run `uv run pytest -m live`."""
import asyncio

import pytest

pytestmark = pytest.mark.live

Q = {"in_danger": {"type": "noul", "instructions": "Is the player in danger?"},
     "mode": {"type": "choice", "instructions": "What next?", "criteria": {"fight": None, "explore": None}}}
S = {"self": {"health": 20}, "monsters": [{"ref": "m1", "kind": "imp", "range": "melee", "visible": True}]}


def test_jev_live():
    from s1doom.backends.jev import make_jev

    r = asyncio.run(make_jev().system_one(S, Q))
    assert r.answers["mode"].value in ("fight", "explore") and r.usage["input_tokens"] > 0


def test_laya_inproc_live():
    from s1doom.backends.laya_inproc import LayaInproc

    r = asyncio.run(LayaInproc().system_one(S, Q))
    assert 0 <= r.answers["in_danger"].value <= 1
