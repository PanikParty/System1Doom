"""Local Jev/SystemOne-compatible server for Cloudflare Clef / Clef-Flash.

Mirrors `s1doom serve-laya`: loads the model in-process via transformers and the
model's own `joint_schema_model.py` (downloaded as part of the HF snapshot --
it carries record encoding, batching, and the `systemone()` request handler),
then exposes POST /v1/systemone so s1doom's existing SystemOneHTTP client
(shared with Jev and laya-serve) can talk to it unmodified.

Clef's API is documented as drop-in compatible with Jev/SystemOne
(https://huggingface.co/Cloudflare/clef, https://huggingface.co/Cloudflare/clef-flash):
same request/response shape, same answers{choice,score,noul} contract.

Usage:
    uv run --extra clef s1doom serve-clef --repo Cloudflare/clef-flash --port 8100
    uv run --extra clef s1doom serve-clef --repo Cloudflare/clef --port 8100 --device cuda

This host (Framework Desktop, Strix Halo / Radeon 8060S, 32GB unified RAM, no
discrete GPU) cannot fit Clef (27B, ~54GB bf16). Clef-Flash (9B, ~19GB bf16) is
the variant sized for this box; even so, CPU/iGPU inference here will be far
slower than Cloudflare's H200 benchmark numbers (209/39 ms median for Clef/Flash).
"""
from __future__ import annotations

import hmac
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serve_clef")

_STATE: dict = {}
# HF revisions checked 2026-10-12; pinned so runs are reproducible (override with --revision)
REVISIONS = {
    "Cloudflare/clef": "0b331204bb13fbd2ca93a64df1956f5b55478ce5",
    "Cloudflare/clef-flash": "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
}


def load(repo: str, device: str | None, dtype: str, revision: str | None = None) -> None:
    import torch
    from huggingface_hub import snapshot_download

    log.info("downloading/locating snapshot for %s ...", repo)
    revision = revision or REVISIONS.get(repo)
    path = snapshot_download(repo, revision=revision)
    if path not in sys.path:
        sys.path.insert(0, path)
    from joint_schema_model import load_release_model  # noqa: E402  (ships inside the HF snapshot)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = getattr(torch, dtype)
    log.info("loading %s on device=%s dtype=%s ...", repo, device, dtype)
    t0 = time.perf_counter()
    model, processor = load_release_model(path, device=device, dtype=torch_dtype)
    model.eval()
    log.info("model ready in %.1fs", time.perf_counter() - t0)
    _STATE["model"] = model
    _STATE["processor"] = processor
    _STATE["repo"] = repo
    _STATE["revision"] = revision
    _STATE["device"] = device


def make_app(api_key: str | None = None):
    import torch

    from joint_schema_model import systemone  # noqa: E402

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "model": _STATE.get("repo"), "revision": _STATE.get("revision"),
                "device": _STATE.get("device"), "torch": torch.__version__}

    @app.post("/v1/systemone")
    async def v1_systemone(req: Request):
        if api_key and not hmac.compare_digest(req.headers.get("authorization", ""), f"Bearer {api_key}"):
            return JSONResponse({"error": "missing or invalid bearer token"}, status_code=401)
        body = await req.json()
        t0 = time.perf_counter()
        try:
            with torch.inference_mode():
                resp = systemone(_STATE["model"], _STATE["processor"], body)
        except Exception as e:  # noqa: BLE001 -- surface as a normal Jev-style error body
            log.exception("systemone failed")
            return JSONResponse({"error": str(e)}, status_code=500)
        resp.setdefault("usage", {})
        resp["usage"].setdefault("latency_ms", round((time.perf_counter() - t0) * 1000, 1))
        return resp

    return app


def main(repo: str = "Cloudflare/clef-flash", port: int = 8100, device: str | None = None,
         dtype: str = "bfloat16", revision: str | None = None, host: str = "0.0.0.0") -> None:
    """Set CLEF_API_KEY to require `Authorization: Bearer <key>` (same scheme as laya-serve)."""
    import uvicorn

    api_key = os.environ.get("CLEF_API_KEY") or None
    if host not in ("127.0.0.1", "localhost") and not api_key:
        log.warning("listening on %s without CLEF_API_KEY set", host)
    load(repo, device, dtype, revision)
    app = make_app(api_key)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import typer

    typer.run(main)
