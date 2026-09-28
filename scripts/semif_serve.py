"""Jev-compatible POST /v1/systemone server over SemIf (github.com/TheoLeeCJ/SemIf-OpenJev).

SemIf scores typed options straight from a frozen open model's logits (Qwen3.5-4B by
default). It pins its own torch/mlx versions, so this server runs in SemIf's venv, not
ours, and the harness talks to it over HTTP like laya-serve:

    cd ~/Research/SemIf-OpenJev
    .venv/bin/python ~/Research/System1Doom/scripts/semif_serve.py --backend mlx --port 8010

Framework Desktop / Linux GPU box: see docs/semif-framework.md. Set SEMIF_API_KEY to require
`Authorization: Bearer <key>` (same scheme as laya-serve).

Mapping: choice -> options (criteria keys); noul -> yes/no options; score -> one option per
level (score = probability-weighted level index). All questions share the state, so they're
scored with SemIf's shared-prefix mode (one prefill, one suffix per question).
Stdlib only, so it adds no dependencies to SemIf's environment.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # pinned in SemIf's README


def _text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def to_rows(state, questions: dict) -> list[dict]:
    rows = []
    for qid, q in questions.items():
        t = q.get("type")
        crit = q.get("criteria")
        if t == "choice":
            opts = [{"id": k, "description": _text(v) if v is not None else k} for k, v in crit.items()]
        elif t == "noul":
            c = crit or {}
            opts = [{"id": "true", "description": _text(c.get("true", "Yes."))},
                    {"id": "false", "description": _text(c.get("false", "No."))}]
        elif t == "score":
            opts = [{"id": str(i), "description": _text(level)} for i, level in enumerate(crit)]
        else:
            raise ValueError(f"question {qid!r}: unknown type {t!r}")
        rows.append({"id": qid, "state": state, "question": _text(q["instructions"]), "options": opts})
    return rows


def to_answer(q: dict, res: dict) -> dict:
    probs = dict(zip(res["option_ids"], res["probabilities"]))
    t = q["type"]
    if t == "noul":
        return {"type": "noul", "noul": round(probs["true"], 4)}
    conf = round(max(probs.values()), 4)
    if t == "choice":
        return {"type": "choice", "choice": max(probs, key=probs.get),
                "probabilities": {k: round(v, 4) for k, v in probs.items()}, "confidence": conf}
    return {"type": "score", "score": round(sum(int(k) * v for k, v in probs.items()), 4),
            "legend": {str(i): _text(c) for i, c in enumerate(q["criteria"])},
            "probabilities": {k: round(v, 4) for k, v in probs.items()}, "confidence": conf}


class Scorer:
    def __init__(self, backend: str, model: str, revision: str, bits: int | None, device: str,
                 gguf: str | None = None, threads: int | None = None) -> None:
        self.backend = backend
        self.lock = Lock()
        if backend == "mlx":
            from semif_phase1 import mlx_backend as B

            self.model, self.tok, self.meta = B.load_model(model, revision, bits)
            self._shared, self._direct = B.score_shared, B.score
        elif backend == "llamacpp":
            # SemIf's llama.cpp path is CPU-only (n_gpu_layers=0 upstream); needs a local GGUF
            from semif_phase1 import llamacpp_backend as B

            self.model, self.tok, self.meta = B.load_model(model, revision, gguf, threads=threads,
                                                           context_tokens=4096)
            self._shared, self._direct = B.score_shared, B.score
        else:
            from semif_phase1.core import load_causal_model
            from semif_phase1.direct import score as direct
            from semif_phase1.shared import score_shared

            self.model, self.tok, self.meta = load_causal_model(model, revision, device, "bfloat16")
            self._shared, self._direct = score_shared, direct
        self.name = f"semif-{model.split('/')[-1]}-{backend}"

    def predict(self, state, questions: dict) -> dict:
        rows = to_rows(state, questions)
        with self.lock:
            if len(rows) == 1:
                results = [self._direct(self.model, self.tok, rows[0], self.meta)]
            else:
                results, _ = self._shared(self.model, self.tok, rows, self.meta)
        by_id = {r["id"]: r for r in results}
        answers = {qid: to_answer(q, by_id[qid]) for qid, q in questions.items()}
        return {"model": self.name, "answers": answers,
                "usage": {"input_tokens": sum(r["input_tokens"] for r in results), "output_tokens": 0}}


def make_handler(scorer: Scorer, api_key: str | None = None):
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict, extra: dict | None = None) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": scorer.name})
            else:
                self._send(404, {"detail": "not found"})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._send(404, {"detail": "not found"})
            if api_key and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {api_key}"):
                return self._send(401, {"detail": "missing or invalid bearer token"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                t0 = time.perf_counter()
                out = scorer.predict(body["state"], body["questions"])
                ms = (time.perf_counter() - t0) * 1000
                self._send(200, out, {"X-Inference-Time-Ms": f"{ms:.1f}"})
            except (ValueError, KeyError, TypeError) as e:
                self._send(422, {"detail": str(e)[:300]})

        def log_message(self, *args):  # quiet
            pass

    return H


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mlx", "torch", "llamacpp"], default="mlx")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--mlx-bits", type=int, default=None, help="quantize MLX weights (e.g. 4 or 8)")
    ap.add_argument("--device", default="auto", help="torch backend device: cuda | mps | cpu "
                    "(ROCm builds of torch expose AMD GPUs as cuda)")
    ap.add_argument("--gguf", default=None, help="llamacpp backend: path to a local GGUF checkpoint")
    ap.add_argument("--threads", type=int, default=None, help="llamacpp backend: CPU threads")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    a = ap.parse_args()
    api_key = os.environ.get("SEMIF_API_KEY") or None
    if a.host not in ("127.0.0.1", "localhost") and not api_key:
        print("warning: listening on a network interface without SEMIF_API_KEY set", flush=True)
    scorer = Scorer(a.backend, a.model, a.revision, a.mlx_bits, a.device, a.gguf, a.threads)
    print(f"{scorer.name} ready on http://{a.host}:{a.port}/v1/systemone"
          f" (auth {'on' if api_key else 'off'})", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(scorer, api_key)).serve_forever()


if __name__ == "__main__":
    main()
