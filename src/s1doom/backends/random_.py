"""Uniform over legal options: the sanity floor."""
from __future__ import annotations

import random

from s1doom.backends.base import Answer, DecisionResult


class RandomBackend:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    async def system_one(self, state: dict, questions: dict) -> DecisionResult:
        out = {}
        for k, q in questions.items():
            if q["type"] == "choice":
                opts = list(q["criteria"])
                out[k] = Answer(type="choice", value=self.rng.choice(opts),
                                probabilities={o: 1 / len(opts) for o in opts})
            elif q["type"] == "noul":
                p = self.rng.random()
                out[k] = Answer(type="noul", value=p, probabilities={"true": p, "false": 1 - p})
            else:
                n = len(q["criteria"])
                lvl = self.rng.randrange(n)
                out[k] = Answer(type="score", value=float(lvl),
                                probabilities={str(i): 1 / n for i in range(n)})
        return DecisionResult(answers=out, model="random")
