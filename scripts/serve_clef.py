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

import logging
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("serve_clef")

_STATE: dict = {}


def load(repo: str, device: str | None, dtype: str) -> None:
    import torch
    from huggingface_hub import snapshot_download

    log.info("downloading/locating snapshot for %s ...", repo)
    path = snapshot_download(repo)
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


def make_app():
    import torch

    from joint_schema_model import systemone  # noqa: E402

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "model": _STATE.get("repo")}

    @app.post("/v1/systemone")
    async def v1_systemone(req: Request):
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
         dtype: str = "bfloat16") -> None:
    import uvicorn

    load(repo, device, dtype)
    app = make_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    import typer

    typer.run(main)
