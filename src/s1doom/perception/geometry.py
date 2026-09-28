"""Angles and ranges.

Convention (matches Doom's ANGLE, counter-clockwise, 0 = east):
    bearing_deg = 0 dead ahead, POSITIVE = LEFT, NEGATIVE = RIGHT.
ViZDoom's TURN_LEFT_RIGHT_DELTA is positive = turn RIGHT, so turn_cmd = -bearing.
"""
from __future__ import annotations

import math

RANGE_BUCKETS = (("melee", 96.0), ("close", 384.0), ("mid", 1024.0))


def norm_deg(a: float) -> float:
    """Normalize to (-180, 180]."""
    a = math.fmod(a, 360.0)
    if a > 180.0:
        a -= 360.0
    elif a <= -180.0:
        a += 360.0
    return a


def bearing_deg(px: float, py: float, angle: float, tx: float, ty: float) -> float:
    world = math.degrees(math.atan2(ty - py, tx - px))
    return norm_deg(world - angle)


def turn_command(bearing: float, gain: float = 0.7, max_turn: float = 12.0) -> float:
    """TURN_LEFT_RIGHT_DELTA value (degrees/tic) that reduces |bearing|."""
    return max(-max_turn, min(max_turn, -bearing * gain))


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(bx - ax, by - ay)


def range_bucket(d: float) -> str:
    for name, lim in RANGE_BUCKETS:
        if d < lim:
            return name
    return "far"


def aim_tol(d: float, radius: float = 20.0) -> float:
    """Half-angle (deg) subtended by a target of `radius` at distance d, floored at 1.5 deg."""
    return max(1.5, math.degrees(math.atan2(radius * 0.8, max(d, 1.0))))


def side_of(bearing: float) -> str:
    if abs(bearing) < 10:
        return "ahead"
    if abs(bearing) > 135:
        return "behind"
    return "left" if bearing > 0 else "right"


def point_segment_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
