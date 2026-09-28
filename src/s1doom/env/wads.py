"""Locate IWADs. Freedoom ships with ViZDoom; doom1.wad is optional and never committed."""
from __future__ import annotations

import os
from pathlib import Path

import vizdoom as vzd

VZD_DIR = Path(vzd.__file__).parent
REPO_WADS = Path(__file__).resolve().parents[3] / "wads"


class WadNotFound(FileNotFoundError):
    pass


def resolve_wad(name: str) -> str:
    p = Path(name)
    if p.exists():
        return str(p)
    fname = name if name.endswith(".wad") else f"{name}.wad"
    candidates = [VZD_DIR / fname, REPO_WADS / fname]
    if os.environ.get("DOOMWADDIR"):
        candidates.append(Path(os.environ["DOOMWADDIR"]) / fname)
    for c in candidates:
        if c.exists():
            return str(c)
    raise WadNotFound(f"{fname} not found (looked in {', '.join(map(str, candidates))})")


def wad_available(name: str) -> bool:
    try:
        resolve_wad(name)
        return True
    except WadNotFound:
        return False
