"""Aim and body-relative movement primitives."""
from __future__ import annotations

import math

from s1doom.perception.entities import Entity
from s1doom.perception.geometry import bearing_deg, turn_command

MAX_MOVE = 50.0  # MOVE_*_DELTA saturates at 50 (running speed); verified on ViZDoom 1.3.1


def aim(target: Entity, gain: float = 0.7, max_turn: float = 12.0) -> float:
    """Turn command toward the target. Screen position beats geometry when visible."""
    b = target.bearing
    if target.visible and target.screen_x_frac is not None:
        # screen x: 0 = left edge. HFOV 90: offset from centre -> degrees (positive = left)
        b = (0.5 - target.screen_x_frac) * 90.0
    return turn_command(b, gain, max_turn)


def move_toward(px: float, py: float, facing: float, tx: float, ty: float,
                speed: float = MAX_MOVE) -> tuple[float, float]:
    """(forward, strafe) that moves toward (tx, ty) without changing facing.
    strafe > 0 is RIGHT (verified)."""
    rel = math.radians(bearing_deg(px, py, facing, tx, ty))
    return speed * math.cos(rel), -speed * math.sin(rel)
