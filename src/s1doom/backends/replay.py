"""Replay logged answers (calls.jsonl) in order, for deterministic regression tests."""
from __future__ import annotations

import json
from pathlib import Path

from s1doom.backends.base import Answer, BackendError, DecisionResult


class ReplayBackend:
    name = "replay"

    def __init__(self, calls_jsonl: str) -> None:
        self.rows = [json.loads(l) for l in Path(calls_jsonl).read_text().splitlines() if l.strip()]
        self.rows = [r for r in self.rows if r.get("kind", "decision") == "decision"]
        self.i = 0

    async def system_one(self, state: dict, questions: dict) -> DecisionResult:
        if self.i >= len(self.rows):
            raise BackendError("replay exhausted")
        row = self.rows[self.i]
        self.i += 1
        raw = row.get("raw_answers") or {}
        ans = {}
        for k in questions:
            if k not in raw:
                raise BackendError(f"replay row {self.i - 1} has no answer for {k!r} "
                                   f"(questions diverged: {sorted(raw)})")
            ans[k] = Answer(**raw[k])
        return DecisionResult(answers=ans, latency_ms=row.get("latency_ms", 0.0), model="replay")
