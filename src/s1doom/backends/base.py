"""DecisionBackend protocol and the normalized answer/result types.

Jev and laya-serve both speak POST /v1/systemone. Differences normalized here:
  * Jev noul answers carry no `confidence`  -> derived as max(p, 1-p).
  * Laya's `confidence` for choice/score is normalized-entropy based, Jev's is
    unspecified; both are kept as `raw_confidence`. The gate uses `max_prob`
    (computed here, after calibration) so every backend is gated on one scale.
  * Laya may return `distribution` as a list in some versions -> mapped to a dict.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

QType = Literal["choice", "score", "noul"]


class Answer(BaseModel):
    type: QType
    value: Any                                # choice key, score float, or noul probability
    probabilities: dict[str, float] = Field(default_factory=dict)
    raw_confidence: float | None = None

    @property
    def max_prob(self) -> float:
        if self.type == "noul":
            p = float(self.value)
            return max(p, 1 - p)
        return max(self.probabilities.values()) if self.probabilities else 0.0

    @property
    def argmax(self) -> Any:
        if self.type == "noul":
            return float(self.value) >= 0.5
        if self.type == "choice":
            return self.value
        return max(self.probabilities, key=self.probabilities.get) if self.probabilities else round(self.value)


class DecisionResult(BaseModel):
    answers: dict[str, Answer]
    latency_ms: float = 0.0
    usage: dict[str, int] = Field(default_factory=dict)
    model: str | None = None
    raw: dict[str, Any] | None = None


@runtime_checkable
class DecisionBackend(Protocol):
    name: str

    async def system_one(self, state: dict, questions: dict) -> DecisionResult: ...


class BackendError(RuntimeError):
    pass


def normalize_answer(q: dict, a: dict) -> Answer:
    t = a.get("type") or q.get("type")
    conf = a.get("confidence")
    if t == "noul":
        p = float(a.get("noul", a.get("value", 0.0)))
        return Answer(type="noul", value=p, probabilities={"true": p, "false": 1 - p},
                      raw_confidence=conf if conf is not None else max(p, 1 - p))
    probs = a.get("probabilities")
    if probs is None and "distribution" in a:
        dist = a["distribution"]
        keys = list(q["criteria"].keys()) if t == "choice" else [str(i) for i in range(len(dist))]
        probs = dict(zip(keys, map(float, dist)))
    probs = {str(k): float(v) for k, v in (probs or {}).items()}
    if t == "choice":
        val = a.get("choice")
        if val is None and probs:
            val = max(probs, key=probs.get)
        return Answer(type="choice", value=val, probabilities=probs, raw_confidence=conf)
    if t == "score":
        val = a.get("score")
        if val is None and probs:
            val = sum(int(k) * v for k, v in probs.items())
        return Answer(type="score", value=float(val), probabilities=probs, raw_confidence=conf)
    raise BackendError(f"unknown answer type {t!r}")


def normalize_response(questions: dict, body: dict, latency_ms: float) -> DecisionResult:
    answers_raw = body.get("answers") or {}
    missing = set(questions) - set(answers_raw)
    if missing:
        raise BackendError(f"response missing answers for {sorted(missing)}")
    answers = {k: normalize_answer(questions[k], answers_raw[k]) for k in questions}
    usage = {k: int(v) for k, v in (body.get("usage") or {}).items() if isinstance(v, (int, float))}
    return DecisionResult(answers=answers, latency_ms=latency_ms, usage=usage,
                          model=body.get("model"), raw=body)


def validate_questions(questions: dict) -> None:
    """Enforce the documented Jev limits before sending."""
    for k, q in questions.items():
        t = q.get("type")
        if t == "choice":
            n = len(q.get("criteria") or {})
            if not 1 <= n <= 255:
                raise ValueError(f"choice {k!r}: {n} options (1..255 allowed)")
        elif t == "score":
            n = len(q.get("criteria") or [])
            if not 2 <= n <= 10:
                raise ValueError(f"score {k!r}: {n} levels (2..10 allowed)")
        elif t != "noul":
            raise ValueError(f"question {k!r}: bad type {t!r}")
