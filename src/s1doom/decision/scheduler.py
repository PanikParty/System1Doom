"""Which questions to ask this decision: mode + in_danger always, the rest when relevant."""
from __future__ import annotations

from dataclasses import dataclass

from s1doom.decision.state_builder import BuiltState


@dataclass
class QuestionScheduler:
    aggression_every: int = 3
    weapon_every: int = 9          # ~2 s at 8 tics/decision
    n: int = 0

    def reset(self) -> None:
        self.n = 0

    def pick(self, bs: BuiltState, can_move: bool) -> set[str]:
        self.n += 1
        st = bs.state
        vis = [m for m in st["monsters"] if m["visible"]]
        which = {"mode", "in_danger"}
        if vis:
            which.add("target")
            if can_move and self.n % self.aggression_every == 1:
                which.add("aggression")
        if len(st["pickups"]) > 1:
            which.add("pickup")
        if len(bs.weapons) > 1 and self.n % self.weapon_every == 1:
            which.add("weapon")
        return which
