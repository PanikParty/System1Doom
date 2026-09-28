"""laya-serve over HTTP (Jev-compatible /v1/systemone). Host is config."""
from __future__ import annotations

import os

from s1doom.backends.http import SystemOneHTTP, read_key


def make_laya_http(url: str | None = None, model: str = "typed-decisions",
                   timeout_s: float = 10.0) -> SystemOneHTTP:
    url = url or os.environ.get("LAYA_URL", "http://127.0.0.1:8000")
    key = read_key("LAYA_API_KEY", "LAYA_API_KEY_FILE", None)
    return SystemOneHTTP("laya-http", url, model, key, timeout_s=timeout_s)
