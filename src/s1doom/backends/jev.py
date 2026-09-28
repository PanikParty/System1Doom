"""TypeSafe Jev over raw HTTP (docs: https://docs.typesafe.ai/api.md)."""
from __future__ import annotations

from s1doom.backends.base import BackendError
from s1doom.backends.http import SystemOneHTTP, read_key

JEV_URL = "https://api.typesafe.ai/v1/systemone"


def make_jev(url: str | None = None, model: str = "jev-latest", timeout_s: float = 10.0) -> SystemOneHTTP:
    key = read_key("TYPESAFE_API_KEY", "TYPESAFE_API_KEY_FILE", "Jev.key")
    if not key:
        raise BackendError("Jev backend needs TYPESAFE_API_KEY (env) or a key file "
                           "(TYPESAFE_API_KEY_FILE, default ./Jev.key)")
    return SystemOneHTTP("jev", url or JEV_URL, model, key, timeout_s=timeout_s)
