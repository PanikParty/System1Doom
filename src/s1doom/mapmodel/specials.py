"""Doom linedef special tables (vanilla/Doom 2 numbering).

Boom generalized specials (>= 0x2F80) are decoded coarsely by range. Freedoom2 MAP01
uses only classic specials (checked 2026-09-27), but other maps may not.
"""
from __future__ import annotations

EXIT = {11: "S1 exit", 52: "W1 exit", 197: "G1 exit"}
SECRET_EXIT = {51: "S1 secret exit", 124: "W1 secret exit", 198: "G1 secret exit"}

# Manual doors: pressed on the door line itself (no tag), open the back sector.
MANUAL_DOORS = {1: None, 117: None, 31: None, 118: None,
                26: "blue", 27: "yellow", 28: "red", 32: "blue", 33: "red", 34: "yellow"}
# Remote doors: switch/walk lines that open tagged sectors.
REMOTE_DOORS = {
    # switches
    29: None, 50: None, 103: None, 111: None, 112: None, 113: None, 42: None, 61: None,
    63: None, 114: None, 115: None, 116: None, 133: "blue", 134: "red", 135: "red",
    136: "yellow", 137: "yellow", 99: "blue",
    # walk triggers
    2: None, 3: None, 4: None, 16: None, 75: None, 76: None, 86: None, 90: None,
    105: None, 106: None, 107: None, 108: None, 109: None, 110: None,
    # gun
    46: None,
}
LIFTS = {62: "SR", 88: "WR", 120: "WR", 121: "W1", 123: "SR", 10: "W1", 21: "S1"}
TELEPORTS = {39: "W1", 97: "WR", 125: "W1 monster", 126: "WR monster"}
FLOOR_MOVERS = {18, 19, 20, 22, 23, 36, 37, 38, 45, 60, 64, 65, 69, 70, 71, 82, 83, 84,
                91, 92, 93, 94, 96, 98, 101, 102, 119, 128, 129, 130, 131, 132, 140}

# "Use"-activated specials (S* and D*) -- the ones pressed with USE
USE_ACTIVATED = {1, 7, 9, 11, 14, 15, 18, 20, 21, 26, 27, 28, 29, 31, 32, 33, 34, 41, 42,
                 43, 45, 49, 50, 51, 55, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
                 99, 101, 102, 103, 111, 112, 113, 114, 115, 116, 117, 118, 122, 123,
                 127, 131, 132, 133, 134, 135, 136, 137, 140}

# Sector specials that hurt the player
DAMAGE_SECTORS = {4: 20, 5: 10, 7: 5, 11: 20, 16: 20}
SECRET_SECTOR = 9

GENERALIZED_BASE = 0x2F80


def classify(special: int) -> str | None:
    if special == 0:
        return None
    if special in EXIT:
        return "exit"
    if special in SECRET_EXIT:
        return "secret_exit"
    if special in MANUAL_DOORS:
        return "door"
    if special in REMOTE_DOORS:
        return "remote_door"
    if special in LIFTS:
        return "lift"
    if special in TELEPORTS:
        return "teleport"
    if special in FLOOR_MOVERS:
        return "floor"
    if special >= GENERALIZED_BASE:
        # Boom generalized: 0x3C00-0x3FFF doors, 0x3800-0x3BFF locked doors, 0x3400 lifts
        if 0x3800 <= special < 0x4000:
            return "door"
        if 0x3400 <= special < 0x3800:
            return "lift"
        return "floor"
    return "other"


def door_key(special: int) -> str | None:
    return MANUAL_DOORS.get(special) or REMOTE_DOORS.get(special)


def is_use(special: int) -> bool:
    if special >= GENERALIZED_BASE:
        return (special & 0x6) in (0x2, 0x3)  # trigger bits: S1/SR/D1/DR approximation
    return special in USE_ACTIVATED
