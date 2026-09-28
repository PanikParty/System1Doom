# Deploying SemIf on the Framework Desktop

This guide gets **SemIf** running on the Framework Desktop as a network service that the s1doom harness on the Mac calls over HTTP. SemIf is [TheoLeeCJ/SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev) (MIT): open-model typed decisions that read Qwen3.5-4B's option logits. It runs next to the Laya server that's already there (`http://framework.local:8000`).

**Why:** on the Mac (Apple M-series, MLX), SemIf takes 1.3–2.9 s per decision. That's fine for a few episodes, but far too slow for the full evaluation (one MAP01 seed can take 3+ hours). Laya went from 432 ms to 113 ms per 3-question decision when it moved to the Framework.

**What you're deploying:** SemIf's own Python package, plus one stdlib-only file from the s1doom repo, `scripts/semif_serve.py`. The file exposes SemIf through the same `POST /v1/systemone` protocol as Jev and laya-serve. Nothing else from s1doom is needed on the Framework.

| Item | Value |
|---|---|
| Port | **8010** (Laya uses 8000) |
| Endpoints | `GET /health`, `POST /v1/systemone` |
| Auth | `Authorization: Bearer <SEMIF_API_KEY>`, the same scheme as laya-serve |
| Model | `Qwen/Qwen3.5-4B`, pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` (SemIf's pin) |
| Memory | about 9 GB (BF16 weights plus working set) |
| Disk | about 9 GB in the Hugging Face cache (Path A), or about 3 GB of GGUF (Path B) |

> **Status of this guide:** Path A (GPU through ROCm PyTorch) is the goal, but **nobody has tested it on AMD**. SemIf's docs only cover CUDA, Apple MPS/MLX and CPU. Path B (CPU through llama.cpp) is the documented fallback. Please record which path worked and the benchmark numbers (step 7).

---

## 0. Prerequisites

- `git`, `uv` (or Python 3.10+ with `venv`/`pip`), and `curl`.
- The working **ROCm PyTorch** setup that laya-serve already uses. Its `/health` reports `"device": "cuda"`, which is how ROCm builds of torch present the Radeon GPU.
- Outbound access to huggingface.co for the first model download.
- Port 8010 reachable from the Mac on the LAN, with the same firewall treatment as 8000.
- `scripts/semif_serve.py` from the s1doom repo. That repo (`github.com/PanikParty/System1Doom`) is **private**, so either get read access or have the single file sent over. It's self-contained.

```bash
mkdir -p ~/semif && cd ~/semif
git clone https://github.com/TheoLeeCJ/SemIf-OpenJev
# put semif_serve.py at ~/semif/semif_serve.py  (from the s1doom repo: scripts/semif_serve.py)
```

## 1. Create the API key

```bash
cd ~/semif
printf 'SEMIF_API_KEY=%s\n' "$(openssl rand -hex 24)" > semif-serve.env
chmod 600 semif-serve.env
```

Send the same `semif-serve.env` file to the Mac user through a private channel, not chat or email in plain text. On the Mac it goes in the s1doom repo root, where `*.env` is gitignored.

## 2. Path A: GPU (ROCm PyTorch)

SemIf pins `torch==2.10.0`, and a plain `pip install` would pull the **CUDA** wheel. On the Framework, install a **ROCm** torch first, then SemIf without letting pip replace it.

```bash
cd ~/semif/SemIf-OpenJev
uv venv --python 3.11 .venv
. .venv/bin/activate

# (a) ROCm torch: use the SAME index/version that works for laya-serve on this machine.
#     Check what laya's env has:   <laya-venv>/bin/python -c "import torch; print(torch.__version__)"
#     e.g. (adjust rocmX.Y to your installed ROCm):
uv pip install torch --index-url https://download.pytorch.org/whl/rocmX.Y

# (b) SemIf's other pinned deps, then SemIf itself WITHOUT re-resolving torch
uv pip install transformers==5.17.0 accelerate==1.12.0 safetensors==0.8.0 \
  huggingface-hub==1.31.0 tokenizers==0.23.2 numpy==2.2.6 sentencepiece==0.2.1 protobuf==7.36.1
uv pip install --no-deps -e .

# (c) sanity check: must print True and the Radeon GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.__version__)"
```

Notes:
- **The torch version may differ from SemIf's pin (2.10.0).** That's acceptable for serving, but record it. SemIf's CUDA timing tables won't be comparable.
- **SemIf requires exactly one visible GPU.** It raises "Expose exactly one CUDA GPU" otherwise. If there's more than one device, set `HIP_VISIBLE_DEVICES=0` (and `CUDA_VISIBLE_DEVICES=0`).
- Qwen3.5's hybrid attention uses fast kernels (`causal_conv1d`, `flash-linear-attention`) that are **CUDA-only**. On ROCm, as on Apple MPS, torch falls back to reference kernels. It will work, but slower than a CUDA card of the same class.
- If the iGPU architecture isn't recognized (errors mentioning `gfx1151` / "invalid device function"), use the same ROCm environment variables that make laya-serve work. For example, some setups need `HSA_OVERRIDE_GFX_VERSION`. Copy whatever the Laya service uses; don't guess.

Start it:

```bash
cd ~/semif/SemIf-OpenJev && . .venv/bin/activate
set -a; . ~/semif/semif-serve.env; set +a
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
python ~/semif/semif_serve.py --backend torch --device cuda --host 0.0.0.0 --port 8010
# first start downloads ~9 GB; wait for:  "semif-Qwen3.5-4B-torch ready on http://0.0.0.0:8010/v1/systemone (auth on)"
```

## 3. Path B: CPU fallback (llama.cpp, GGUF)

Use this if Path A can't see the GPU. SemIf's llama.cpp backend is **CPU-only** by design (upstream sets `n_gpu_layers = 0`), but the Framework's 16-core CPU should still be usable. It keeps SemIf's prompt construction (same pinned tokenizer), so decisions are comparable up to quantization.

```bash
cd ~/semif/SemIf-OpenJev
uv venv --python 3.11 .venv-cpu && . .venv-cpu/bin/activate
uv pip install -e '.[llamacpp]'          # builds llama-cpp-python: needs cmake + a C/C++ compiler
# GGUF recommended by SemIf's README (~3 GB); huggingface-hub >= 1.0 names its CLI `hf`:
hf download bartowski/Qwen_Qwen3.5-4B-GGUF Qwen_Qwen3.5-4B-Q4_K_M.gguf --local-dir ~/semif/gguf

set -a; . ~/semif/semif-serve.env; set +a
python ~/semif/semif_serve.py --backend llamacpp --gguf ~/semif/gguf/Qwen_Qwen3.5-4B-Q4_K_M.gguf \
  --threads 16 --host 0.0.0.0 --port 8010
```

## 4. Smoke test (on the Framework)

```bash
set -a; . ~/semif/semif-serve.env; set +a
curl -s localhost:8010/health
# expect: {"status": "ok", "model": "semif-Qwen3.5-4B-torch"}   (or -llamacpp)

curl -s localhost:8010/v1/systemone -H "Authorization: Bearer $SEMIF_API_KEY" \
  -H 'Content-Type: application/json' -d '{
  "state": {"self": {"health": 30}, "monsters": [{"ref": "m1", "kind": "imp", "range": "close", "aligned": true}]},
  "questions": {
    "mode": {"type": "choice", "instructions": "What should the player do now?",
             "criteria": {"fight": "Shoot the aligned monster.", "retreat": "Back away."}},
    "in_danger": {"type": "noul", "instructions": "Will the player take damage in the next second?"}}}'
# expect: {"model": ..., "answers": {"mode": {"type": "choice", "choice": ..., "probabilities": {...}},
#          "in_danger": {"type": "noul", "noul": 0.x}}, "usage": {"input_tokens": N, ...}}

curl -s -o /dev/null -w '%{http_code}\n' localhost:8010/v1/systemone -d '{}'   # expect 401 (no key)
```

The first request after startup is slow while kernels compile or warm up. Time a few more requests after it.

## 5. Run as a service (systemd)

`/etc/systemd/system/semif.service`, adjusted for the user, paths, and the venv of the path that worked:

```ini
[Unit]
Description=SemIf typed-decision server (Jev-compatible /v1/systemone)
After=network-online.target
Wants=network-online.target

[Service]
User=duncan
WorkingDirectory=/home/duncan/semif/SemIf-OpenJev
EnvironmentFile=/home/duncan/semif/semif-serve.env
Environment=HF_HOME=/home/duncan/.cache/huggingface
Environment=HIP_VISIBLE_DEVICES=0
# plus any ROCm env vars the laya service uses
ExecStart=/home/duncan/semif/SemIf-OpenJev/.venv/bin/python /home/duncan/semif/semif_serve.py \
  --backend torch --device cuda --host 0.0.0.0 --port 8010
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now semif
journalctl -u semif -f        # wait for "ready on ... (auth on)"
```

**Memory:** Laya and SemIf together need roughly 2 + 9 GB of GPU-visible memory. That's fine on the Framework's unified memory, but both services compete for the same GPU. Benchmarks are cleanest when only one model is being evaluated at a time.

## 6. Hook it up from the Mac

The Mac side is already configured in `configs/backends/semif-framework.yaml` (`url: http://framework.local:8010`, key read from `SEMIF_API_KEY`). Put the shared `semif-serve.env` in the s1doom repo root, then:

```bash
cd ~/Research/System1Doom
set -a; . ./semif-serve.env; set +a
curl -s http://framework.local:8010/health
```

## 7. Verify and benchmark (from the Mac), then report back

```bash
set -a; . ./semif-serve.env; set +a
R=$(ls -td runs/*freedoom2_map01_heuristic_m3 | head -1)
uv run s1doom bench-backend --backend semif-framework --states $R/ep_0/calls.jsonl --n 100 --qcounts 1,3,6
uv run s1doom play --scenario defend_the_center --backend semif-framework --episodes 1   # quick end-to-end
```

Reference numbers (p50 per decision) for 1 / 3 / 6 questions:

| Backend | 1 q | 3 q | 6 q |
|---|---:|---:|---:|
| Jev (TypeSafe API) | 119 ms | 116 ms | 120 ms |
| Laya, Framework (CUDA/ROCm) | 56 ms | 113 ms | 181 ms |
| Laya, Mac MPS | 147 ms | 432 ms | 758 ms |
| SemIf, Mac MLX BF16 | 1264 ms | 1799 ms | 2672 ms |
| **SemIf, Framework** | ? | ? | ? |

Please report: which path worked (A or B), the torch and ROCm versions (or the GGUF file), any environment variables you needed, the `bench-backend` table, and anything that failed.

## 8. Full evaluation (from the Mac, once the benchmark looks sane)

```bash
set -a; . ./semif-serve.env; set +a
E="uv run s1doom eval --seed 1234 --backends semif-framework"
$E --scenario defend_the_center --episodes 20 --tag m1
$E --scenario deadly_corridor,health_gathering_supreme,my_way_home --episodes 20 --tag m2
$E --scenario freedoom2_map01 --episodes 10 --tag m3
$E --scenario freedoom2_map01 --episodes 5 --tag nofallback --no-fallback
uv run s1doom report
```

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `torch.cuda.is_available()` is False | A CUDA or CPU torch wheel got installed. Redo 2(a) from the ROCm index, and 2(b) with `--no-deps`. |
| "Expose exactly one CUDA GPU" | Set `HIP_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=0`. |
| "invalid device function" / gfx errors | The ROCm arch isn't recognized. Reuse laya-serve's ROCm environment variables, or use Path B. |
| Very slow first request | Normal warm-up. Benchmark from the second request on. |
| Slow on every request | Expected to some degree on ROCm (reference attention kernels). Compare against the CPU path. |
| `401` from the Mac | The key differs: both sides must use the same `semif-serve.env`. |
| `Connection refused` from the Mac | The server is bound to 127.0.0.1 (use `--host 0.0.0.0`), or the firewall blocks port 8010. |
| llama-cpp-python build fails | Install `cmake`, `gcc`/`g++` and `python3-dev`, then retry `uv pip install -e '.[llamacpp]'`. |
| `422` response | The request is malformed. The response body names the field. Report it with the request. |
