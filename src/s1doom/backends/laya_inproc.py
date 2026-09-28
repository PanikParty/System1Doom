"""Laya loaded in-process (no network hop). Requires the `laya` extra."""
from __future__ import annotations

import asyncio
import time

from s1doom.backends.base import DecisionResult, normalize_response, validate_questions


class LayaInproc:
    name = "laya-inproc"

    def __init__(self, repo: str = "convaiinnovations/laya", subfolder: str | None = "typed-decisions",
                 device: str | None = None, fast: bool = False) -> None:
        import laya  # heavy import deferred

        self.agent = laya.load(repo, subfolder=subfolder, device=device, fast=fast)
        self._lock = asyncio.Lock()
        self.calls = 0
        self.input_tokens = 0

    def predict_sync(self, state: dict, questions: dict) -> DecisionResult:
        validate_questions(questions)
        t0 = time.perf_counter()
        body = self.agent.predict(state, questions)
        dt = (time.perf_counter() - t0) * 1000
        res = normalize_response(questions, body, dt)
        self.calls += 1
        self.input_tokens += res.usage.get("input_tokens", 0)
        return res

    async def system_one(self, state: dict, questions: dict) -> DecisionResult:
        async with self._lock:
            return await asyncio.get_running_loop().run_in_executor(
                None, self.predict_sync, state, questions)

    def tokenizer(self):
        return getattr(self.agent, "tok", None)
