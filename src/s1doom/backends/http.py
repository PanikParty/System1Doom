"""Shared POST /v1/systemone client for Jev and laya-serve."""
from __future__ import annotations

import asyncio
import os
import random
import time
from pathlib import Path

import httpx

from s1doom.backends.base import BackendError, DecisionResult, normalize_response, validate_questions

RETRY_STATUS = {429, 503, 529}


class SystemOneHTTP:
    def __init__(self, name: str, url: str, model: str | None, api_key: str | None,
                 timeout_s: float = 10.0, max_retries: int = 4, extra: dict | None = None) -> None:
        self.name = name
        self.url = url.rstrip("/")
        if not self.url.endswith("/v1/systemone"):
            self.url += "/v1/systemone"
        self.model = model
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.extra = extra or {}
        self._client: httpx.AsyncClient | None = None
        self.calls = 0
        self.input_tokens = 0

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def system_one(self, state: dict, questions: dict) -> DecisionResult:
        validate_questions(questions)
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s, http2=False)
        body = {"state": state, "questions": questions, **self.extra}
        if self.model:
            body["model"] = self.model
        delay = 0.25
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = await self._client.post(self.url, json=body, headers=self._headers())
            except httpx.TransportError as e:
                if attempt == self.max_retries:
                    raise BackendError(f"{self.name}: transport error {e!r}") from e
            else:
                dt = (time.perf_counter() - t0) * 1000
                if r.status_code == 200:
                    res = normalize_response(questions, r.json(), dt)
                    self.calls += 1
                    self.input_tokens += res.usage.get("input_tokens", 0)
                    return res
                if r.status_code not in RETRY_STATUS or attempt == self.max_retries:
                    raise BackendError(f"{self.name}: HTTP {r.status_code}: {r.text[:300]}")
            await asyncio.sleep(delay * (1 + random.random()))
            delay *= 2
        raise BackendError(f"{self.name}: retries exhausted")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def read_key(env_var: str, file_env_var: str, default_file: str | None) -> str | None:
    """API key from env var, else from a key file (path from env or a default). Never logged."""
    if os.environ.get(env_var):
        return os.environ[env_var].strip()
    path = os.environ.get(file_env_var) or default_file
    if path and Path(path).exists():
        return Path(path).read_text().strip() or None
    return None
