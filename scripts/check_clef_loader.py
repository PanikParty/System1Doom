"""Check serve_clef.load_direct against the checkpoint and a from_pretrained-loaded server.

1. Every backbone tensor equals its checkpoint tensor bit for bit.
2. On real states (a calls.jsonl), answers match a running serve_clef server that was
   loaded with --loader pretrained (same repo and revision).

    LD_PRELOAD=/usr/lib64/libhsa-runtime64.so.1 ~/clef/.venv/bin/python scripts/check_clef_loader.py \\
        --states runs/_states/map01_heuristic_ep0.jsonl --url http://127.0.0.1:8100 --n 60
Needs CLEF_API_KEY for the server. Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serve_clef  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Cloudflare/clef-flash")
    ap.add_argument("--states", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(a.repo, revision=serve_clef.REVISIONS[a.repo])
    sys.path.insert(0, path)
    from joint_schema_model import systemone  # noqa: E402

    model, processor = serve_clef.load_direct(path, "cuda", torch.bfloat16)
    sd = model.language_model.state_dict()
    weight_map = json.loads((Path(path) / "model.safetensors.index.json").read_text())["weight_map"]
    bad = []
    for shard in sorted(set(weight_map.values())):
        with safe_open(str(Path(path) / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if not torch.equal(sd[key].cpu(), f.get_tensor(key)):
                    bad.append(key)
    print(f"tensors: {len(weight_map)} checked, {len(bad)} differ {bad[:5]}")
    for name, buf in model.language_model.named_buffers():
        if name not in weight_map:
            print(f"buffer {name}: {tuple(buf.shape)} {buf.dtype} first={buf.flatten()[:2].tolist()}")

    rows = [json.loads(line) for line in open(a.states)]
    rows = [r for r in rows if r.get("kind") == "decision" and r.get("questions")]
    step = max(1, len(rows) // a.n)
    key = os.environ["CLEF_API_KEY"]
    worst, mismatched, n = 0.0, 0, 0
    for r in rows[::step][: a.n]:
        body = {"model": "clef", "state": r["state"], "questions": r["questions"]}
        with torch.inference_mode():
            mine = systemone(model, processor, body)["answers"]
        req = urllib.request.Request(f"{a.url}/v1/systemone", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        theirs = json.loads(urllib.request.urlopen(req, timeout=60).read())["answers"]
        for q, ans in mine.items():
            pm = ans.get("probabilities") or {"true": ans["noul"]}
            pt = theirs[q].get("probabilities") or {"true": theirs[q]["noul"]}
            d = max(abs(pm[k] - pt[k]) for k in pm)
            worst = max(worst, d)
            mismatched += d > 0
            n += 1
    print(f"answers: {n} compared over {min(a.n, len(rows[::step]))} states, {mismatched} differ, "
          f"max |dp| = {worst:.4f} (answers are rounded to 4 decimals)")


if __name__ == "__main__":
    main()
