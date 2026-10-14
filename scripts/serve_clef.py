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


def load_direct(path: str, device: str, dtype):
    """Like joint_schema_model.load_release_model, but streams the backbone onto the device.

    from_pretrained stages the checkpoint in host RAM before moving it (peak about the model
    size), which the Framework's 31 GB of system RAM can't hold for the 27B (its 96 GiB are
    VRAM). Here the backbone is built on the device with init skipped (so the rotary buffers
    are computed as usual) and each checkpoint tensor is copied straight into its parameter.
    Checkpoint and model keys match 1:1 for both releases; scripts/check_clef_loader.py
    checks the result against the checkpoint and against a from_pretrained server.
    """
    import inspect
    import json

    import torch
    from joint_schema_model import ClefModel, JointSchemaHead  # noqa: E402
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
    from transformers.initialization import no_init_weights

    root = Path(path)
    config = AutoConfig.from_pretrained(root)
    with torch.device(device), no_init_weights():
        backbone = Qwen3_5ForConditionalGeneration._from_config(config, dtype=dtype)
    backbone.config.use_cache = False
    # Non-persistent buffers (rotary inv_freq) aren't in the checkpoint. Computed on the GPU they
    # differ from from_pretrained's CPU-computed ones by up to 3e-8, which moves probabilities by
    # up to 0.01, so rebuild each owning module on the CPU and copy its buffers over.
    persistent = set(backbone.state_dict())
    for name, mod in backbone.named_modules():
        own = [b for b, _ in mod.named_buffers(recurse=False) if f"{name}.{b}" not in persistent]
        if not own:
            continue
        if "config" not in inspect.signature(type(mod).__init__).parameters:
            raise ValueError(f"can't rebuild non-persistent buffers of {name}; use --loader pretrained")
        with torch.device("cpu"):
            ref = type(mod)(config=mod.config)
        with torch.no_grad():
            for b in own:
                getattr(mod, b).copy_(getattr(ref, b))
    params = dict(backbone.named_parameters())
    params.update(dict(backbone.named_buffers()))
    weight_map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    if set(weight_map) != set(backbone.state_dict()):
        raise ValueError("checkpoint keys don't match the model; use --loader pretrained")
    with torch.no_grad():
        for shard in sorted(set(weight_map.values())):
            with safe_open(str(root / shard), framework="pt", device=str(device)) as f:
                for key in f.keys():
                    params[key].copy_(f.get_tensor(key))
            log.info("loaded %s", shard)
    head = JointSchemaHead(**json.loads((root / "joint_head_config.json").read_text()))
    head.load_state_dict(load_file(root / "joint_head.safetensors"), strict=True)
    head = head.to(device=device, dtype=dtype)
    return ClefModel(backbone, head).eval(), AutoProcessor.from_pretrained(root)


def load(repo: str, device: str | None, dtype: str, revision: str | None = None,
         loader: str = "direct") -> None:
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
    log.info("loading %s on device=%s dtype=%s loader=%s ...", repo, device, dtype, loader)
    t0 = time.perf_counter()
    if loader == "direct" and device != "cpu":
        model, processor = load_direct(path, device, torch_dtype)
    else:
        model, processor = load_release_model(path, device=device, dtype=torch_dtype)
    model.eval()
    log.info("model ready in %.1fs", time.perf_counter() - t0)
    _STATE["model"] = model
    _STATE["processor"] = processor
    _STATE["repo"] = repo
    _STATE["revision"] = revision
    _STATE["device"] = device
    _STATE["loader"] = loader


def make_app(api_key: str | None = None):
    import torch

    from joint_schema_model import systemone  # noqa: E402

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "model": _STATE.get("repo"), "revision": _STATE.get("revision"),
                "device": _STATE.get("device"), "loader": _STATE.get("loader"), "torch": torch.__version__}

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
         dtype: str = "bfloat16", revision: str | None = None, host: str = "0.0.0.0",
         loader: str = "direct") -> None:
    """Set CLEF_API_KEY to require `Authorization: Bearer <key>` (same scheme as laya-serve)."""
    import uvicorn

    api_key = os.environ.get("CLEF_API_KEY") or None
    if host not in ("127.0.0.1", "localhost") and not api_key:
        log.warning("listening on %s without CLEF_API_KEY set", host)
    load(repo, device, dtype, revision, loader)
    app = make_app(api_key)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import typer

    typer.run(main)
