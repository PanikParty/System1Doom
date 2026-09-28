"""Backend construction from a name (+ optional configs/backends/<name>.yaml)."""
from __future__ import annotations

from pathlib import Path

import yaml

from s1doom.backends.base import DecisionBackend

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "backends"
KINDS = ("jev", "laya-http", "laya-inproc", "systemone-http", "heuristic", "random", "replay")


def backend_config(name: str) -> dict:
    p = CONFIG_DIR / f"{name}.yaml"
    cfg = yaml.safe_load(p.read_text()) if p.exists() else {}
    cfg.setdefault("kind", name)
    return cfg


def make_backend(name: str, url: str | None = None, seed: int | None = None,
                 replay_path: str | None = None) -> DecisionBackend:
    cfg = backend_config(name)
    kind = cfg["kind"]
    if kind == "heuristic":
        from s1doom.backends.heuristic import HeuristicBackend

        return HeuristicBackend()
    if kind == "random":
        from s1doom.backends.random_ import RandomBackend

        return RandomBackend(seed)
    if kind == "jev":
        from s1doom.backends.jev import make_jev

        return make_jev(url or cfg.get("url"), cfg.get("model", "jev-latest"), cfg.get("timeout_s", 10.0))
    if kind == "laya-http":
        from s1doom.backends.laya_http import make_laya_http

        b = make_laya_http(url or cfg.get("url"), cfg.get("model", "typed-decisions"), cfg.get("timeout_s", 10.0))
        b.name = name if name != "laya-http" else b.name
        return b
    if kind == "systemone-http":
        # any Jev-compatible /v1/systemone server (e.g. scripts/semif_serve.py)
        from s1doom.backends.http import SystemOneHTTP, read_key

        return SystemOneHTTP(cfg.get("name", name), url or cfg["url"], cfg.get("model"),
                             read_key(cfg.get("key_env", "SYSTEMONE_API_KEY"), "SYSTEMONE_API_KEY_FILE", None),
                             timeout_s=cfg.get("timeout_s", 10.0))
    if kind == "laya-inproc":
        from s1doom.backends.laya_inproc import LayaInproc

        return LayaInproc(cfg.get("repo", "convaiinnovations/laya"), cfg.get("subfolder", "typed-decisions"),
                          cfg.get("device"), cfg.get("fast", False))
    if kind == "replay":
        from s1doom.backends.replay import ReplayBackend

        if not replay_path:
            raise ValueError("replay backend needs a calls.jsonl path")
        return ReplayBackend(replay_path)
    raise ValueError(f"unknown backend {name!r} (kinds: {KINDS})")
