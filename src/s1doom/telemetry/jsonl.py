"""Append-only JSONL call log, one file per episode."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _default(o: Any):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if isinstance(o, set):
        return sorted(o)
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    return str(o)


class JsonlLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = path.open("w")

    def write(self, row: dict) -> None:
        self.f.write(json.dumps(row, default=_default, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self.f.close()
