# s1doom: System One models play DOOM

A harness that lets **text-only System One decision models** play classic DOOM through ViZDoom. The supported models are TypeSafe **Jev** (API) and the open-weight **Laya**. Code turns the engine state into a compact JSON state and a few atomic typed questions whose options are only the actions legal right now. The model picks the tactic and target. A deterministic controller turns that into button presses every tic.

> **Rule: code computes the geometry, and the model makes the judgment.** (PLAN.md §0)

See `PLAN.md` for the design. Results live in `runs/report.md` (`uv run s1doom report`).

## Quick start
```bash
uv sync                      # core (ViZDoom ships Freedoom1/2)
uv sync --extra laya         # + Laya in-process / laya-serve (torch)
uv run pytest -q             # unit, contract (recorded Jev, Laya, SemIf fixtures), engine smoke

uv run s1doom play --scenario defend_the_center --backend heuristic --overlay --window
uv run s1doom play --scenario freedoom2_map01 --backend jev --overlay --window
uv run s1doom play --scenario freedoom2_map01 --backend laya-inproc --mode realtime --overlay --window
uv run s1doom eval --scenario deadly_corridor --backends heuristic,jev,laya-inproc --episodes 20 --seed 1234
uv run s1doom bench-backend --backend laya-http --url http://framework.local:8000
uv run s1doom calibrate --runs 'runs/*_m1' --out configs/calibration.yaml
uv run s1doom inspect-map --wad freedoom2 --map MAP01       # PNG of nav graph, exits, doors, path
uv run s1doom replay runs/<run_id>                          # re-run with logged answers
uv run s1doom serve-laya                                    # local Jev-compatible /v1/systemone
uv run s1doom report                                        # runs/report.md + charts
```
Keys: Jev reads `TYPESAFE_API_KEY`, or a key file at `TYPESAFE_API_KEY_FILE` (default `./Jev.key`). laya-http reads `LAYA_URL` and, optionally, `LAYA_API_KEY`. Key files are gitignored.

## Backends
| name | transport | notes |
|---|---|---|
| `heuristic` | Python | Answers from the **same serialized state** the models see. Baseline, fallback and label source. |
| `random` | Python | Uniform over legal options. Fallback is **disabled** so it stays a true floor. |
| `jev` | raw HTTP `POST api.typesafe.ai/v1/systemone` | Retries 429/503/529 with backoff and jitter. A circuit breaker falls back to the heuristic. `--max-calls` caps the budget. |
| `laya-http` | `laya-serve` `/v1/systemone` | The host is config (`configs/backends/laya-*.yaml`). |
| `laya-inproc` | `laya.load(..., subfolder="typed-decisions")` | MPS on Apple silicon. |
| `semif` | `scripts/semif_serve.py` → `/v1/systemone` | [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) (Qwen3.5-4B typed-option logits, MIT). Runs in SemIf's own venv (MLX on Apple Silicon, torch/CUDA elsewhere). |
| `clef-flash` / `clef` | `scripts/serve_clef.py` → `/v1/systemone` | [Cloudflare Clef-Flash (9B) / Clef (27B)](https://huggingface.co/Cloudflare/clef), joint-schema-head decision models, Apache-2.0. Runs in its own ROCm/CUDA torch venv (see "Clef"). |
| any `systemone-http` | Jev-compatible HTTP | Point a `configs/backends/<name>.yaml` with `kind: systemone-http` at any server that speaks the protocol. |
| `replay` | calls.jsonl | Deterministic regression runs. |

## Clef (Cloudflare)
[Clef-Flash / Clef](https://huggingface.co/Cloudflare/clef) speak the Jev/SystemOne protocol natively, so they use the generic `systemone-http` kind. `scripts/serve_clef.py` (also `s1doom serve-clef`) wraps the `systemone()` that ships in the model repo. It pins the HF revision, requires `Authorization: Bearer $CLEF_API_KEY` when that variable is set, and reports revision, device and torch on `/health`.

**Framework Desktop** (Strix Halo, 128 GB unified memory with 96 GiB carved out as VRAM, so Linux shows only 31 GB of RAM; Radeon 8060S / gfx1151; Fedora 44). Clef-Flash runs on the GPU from its own venv. Don't use `uv sync --extra clef` there: it swaps in a CUDA torch build, and in this repo's venv it also removes `laya`.
```bash
uv venv --python 3.11 ~/clef/.venv
uv pip install -p ~/clef/.venv/bin/python --index-url https://download.pytorch.org/whl/nightly/rocm7.1 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match torch==2.14.0.dev20260804+rocm7.1
uv pip install -p ~/clef/.venv/bin/python --no-deps --index-url https://download.pytorch.org/whl/nightly/rocm7.1 \
  torchvision==0.29.0.dev20260807+rocm7.1         # the Qwen3-VL processor imports it
uv pip install -p ~/clef/.venv/bin/python transformers==5.17.0 accelerate safetensors huggingface-hub pillow fastapi uvicorn typer
# the pip wheel's bundled libhsa segfaults on Fedora 44 -- preload the distro one (as laya-serve and semif-serve do)
LD_PRELOAD=/usr/lib64/libhsa-runtime64.so.1 CLEF_API_KEY=... ~/clef/.venv/bin/python scripts/serve_clef.py \
  --repo Cloudflare/clef-flash --port 8100 --device cuda       # user service: clef-flash-serve
```
On a CUDA box, SGLang serves `/v1/systemone` directly (`lmsysorg/sglang:dev-clef`, see the model card). Point `url` in `configs/backends/clef*.yaml` at it.

**Clef-Flash latency** (Framework GPU, MAP01 states, p50): 540 / 795 / 1051 ms for 1 / 3 / 6 questions. That is about 3× faster than on CPU, but still the slowest backend here. The joint head reads every question, so the cost grows with the question count. **Clef 27B** (`--repo Cloudflare/clef --port 8101`, user service `clef-serve`) takes 1260 / 1973 / 2741 ms. `from_pretrained` stages the whole checkpoint in host RAM, which the Framework's 31 GB can't hold, so `serve_clef.py` defaults to `--loader direct`. That loader streams each tensor onto the GPU; it is bit-identical to `from_pretrained` (checked with `scripts/check_clef_loader.py`) and loads the 27B in 34.5 s. It needs about 57 GB of VRAM, so stop `clef-flash-serve` while it runs alongside Laya and SemIf.

Configs for other Laya hosts (Framework Desktop, Jetson) are in `configs/backends/`. Point `url` at a running `laya-serve`.

## SemIf (open-model Jev-style decisions)
SemIf pins its own torch/mlx versions, so it runs in a separate venv and serves the Jev wire protocol:
```bash
git clone https://github.com/TheoLeeCJ/SemIf-OpenJev ~/Research/SemIf-OpenJev
cd ~/Research/SemIf-OpenJev && uv venv --python 3.11 .venv && uv pip install -p .venv/bin/python -e '.[mlx]'
.venv/bin/python ~/Research/System1Doom/scripts/semif_serve.py --backend mlx --port 8010   # first start downloads ~8 GB
# on a CUDA box (e.g. a 3090): --backend torch --device cuda --host 0.0.0.0, then set url in configs/backends/semif.yaml
uv run s1doom play --scenario freedoom2_map01 --backend semif      # from this repo
```
**Deploying on the Framework Desktop (or any Linux GPU box):** see [`docs/semif-framework.md`](docs/semif-framework.md). It covers the ROCm GPU path, a CPU llama.cpp fallback, the API key, a systemd unit, a smoke test, the benchmark and the eval commands. The Mac side uses `--backend semif-framework`.

Mapping: choice → options, noul → yes/no options, score → one option per level (score = probability-weighted level). All questions share one state, so they're scored with SemIf's shared-prefix mode. On an M-series Mac with MLX BF16 it takes ~1.3 s for one question and ~1.8 s for three (p50), so use turn mode.

## What's in a decision
Every decision logs one row to `runs/<id>/ep_<n>/calls.jsonl`. The row holds the state, the questions, the raw answers, the calibrated probabilities, the shadow heuristic's answers, the final decision with a per-question `source` (model | fallback | forced | held), latency, usage, `prompt_version`, `fairness`, position, and an outcome label for `in_danger`. Each episode also records a `.lmp` demo.

The **fairness** default is visible-only: monsters and items come from `state.labels` plus a decaying memory. `--omniscient` is an ablation flag and is recorded in the logs.

## Layout
```
configs/scenarios/*.yaml  configs/backends/*.yaml  prompts/v1/questions.yaml
src/s1doom/env         ViZDoom setup, button map, WAD lookup
src/s1doom/perception  labels -> entities (bearing, aligned, range, threat), memory, events
src/s1doom/mapmodel    WAD parse (Doom + UDMF), directed nav graph, visibility, frontier, runtime level model
src/s1doom/decision    state builder (token budget), questions, scheduler, policy (gate/hysteresis/fallback), calibration
src/s1doom/control     controller, macros, aim, navigator (doors, lifts, teleports, switches, stuck recovery)
src/s1doom/backends    base protocol + normalization, heuristic, random, replay, jev, laya_http, laya_inproc
src/s1doom/telemetry   JSONL, metrics, overlay, summary, report
src/s1doom/agent.py    episode loop (turn + realtime)      src/s1doom/cli.py  typer CLI
```

## Results (verification campaign v2, 2026-10-13)
Every backend ran on one code version (`ae293c3`, prompts v1.2; Clef 27B: see §), the same seeds (1234+i) and turn mode, on the Framework Desktop. The full method, CIs, provenance checks and a verdict for every earlier claim are in [`docs/verification-v2.md`](docs/verification-v2.md). Δ is the paired difference against the heuristic, with a bootstrap 95% CI; **bold** marks a CI that excludes 0.

| outcome (n) | heuristic | random | Jev | Laya | SemIf | Clef-Flash | Clef 27B § |
|---|---|---|---|---|---|---|---|
| defend_the_center kills (20) | 20.60 | 21.25 | 22.35 (Δ **+1.75**) | 22.50 (Δ **+1.90**) | 22.55 (Δ **+1.95**) | 22.45 (Δ **+1.85**) | 22.90 (Δ **+2.80**) |
| deadly_corridor vest (20) | 80% | 30% | 90% | 95% | 55% | 55% (Δ **−25**) | **90%** |
| health_gathering_supreme tics (20) | 2020 | 1236 | 1969 | 2020 † | 1811 (−10.3%) | 1953 | 2017 |
| my_way_home success (20) | 100% | 95% | 100% | 100% | 100% | 100% | 100% |
| Freedoom2 MAP01 exit (10) | 100% | 90% | 80% | 100% † | 100% | 90% | 100% (5/5) |
| **no fallback**: deadly_corridor / health_gathering / my_way_home / MAP01 | – | – | 95% / 1870 / 100% / 80% | 95% / **585** / **50%** / **40%** | 60% / 1935 / 100% / 90% | 70% / 1866 / 100% / 90% | **95%** / 1890 / 100% / 100% |
| fallback rate (dtc / dc / hgs / mwh / MAP01) | – | disabled | 2 / 7 / 9 / 1 / 2% | 13 / 59 / 66 / 19 / 66% | 2 / 5 / 8 / 0 / 2% | 2 / 7 / 26 / 0 / 7% | 4 / 6 / 23 / 0 / 3% |
| latency p50, 1 / 3 / 6 questions (solo bench) | – | – | 214 ‡ / 128 / 142 ms | 44 / 99 / 169 ms | 275 / 458 / 643 ms | 540 / 795 / 1051 ms | 1260 / 1973 / 2741 ms |
| in_danger AUC (accuracy; always-false scores 89–96%) ¶ | 0.70 (79%) | 0.52 | 0.82 (72%) | **0.88** (4%) | 0.76 (67%) | 0.78 (77%) | 0.77 (66%) |

† Laya's results with fallback are mostly the heuristic's: health_gathering is identical to the heuristic in 20/20 episodes. Alone, Laya fails health_gathering and my_way_home, and finishes MAP01 only 4/10 times (six timeouts).  
¶ Each backend is scored on the states it visited, and the scenario mixes differ, so read these as indicative.  
§ Clef 27B ran afterwards on `6187e8e` (game code unchanged) with 10 episodes per scenario and 5 on MAP01; deadly_corridor has the full 20. Its Δs are paired on its own seeds. Like-for-like tables are in `docs/verification-v2.md`, "Clef 27B".  
‡ Includes connection warm-up (p95 733 ms). Laya and SemIf are measured on the Framework's GPU; Jev is called from the Framework.

**Reading the table:**
- **deadly_corridor is the only scenario that separates the models.** Jev, Laya and Clef 27B reach the vest 90–95% of the time, and SemIf and Clef-Flash 55%. Paired on the same 20 seeds, the 27B beats Clef-Flash by +35 points [+15, +55], winning all 7 seeds where they disagree.
- **On defend_the_center, every model beats the heuristic by about 1.9 kills and is indistinguishable from the others.**
- **On MAP01, the navigator does the work.** Random finishes 9/10.
- **Laya's `in_danger` ranks danger best (AUC 0.88), but it is offset.** Every probability falls between 0.52 and 0.66, so it is "wrong" at a 0.5 threshold. That is a calibration problem, not an absence of signal.
- **Jev's `in_danger`, like every backend's, is less accurate than always answering "false".**

Jev isn't deterministic between runs. Its earlier v1.2 MAP01 result of 10/10 and this campaign's 8/10 are both consistent with roughly 80–90% at n=10.

Realtime mode (35 Hz, ASYNC_PLAYER) runs without frame stalls (0 overruns, p99 per-tic work 8 ms). With Laya on Apple MPS, answers arrive with 230–480 ms staleness, and those older than 400 ms are dropped.

Earlier v1/v1.1 results (Mac, 2026-09-27/28) were audited against `runs/`. The milestone table matched, and the summary cards didn't. See `docs/verification-v2.md`.

**Model notes:**
- **Laya**: `typed-decisions` has 0.42B parameters. It encodes the state once per question: on the Framework GPU it is the fastest backend at 1–3 questions, and Jev is faster at 6.
- **SemIf**: Qwen3.5-4B option logits with shared-prefix scoring.
- **Clef-Flash**: 9B, with one joint pass over all questions.
- **Clef 27B**: the same joint head on Qwen3.8-27B. It is the best of the open models on deadly_corridor and the slowest backend, at 1.3–2.7 s per decision on the iGPU.
- **Jev**: hosted API, flat at ~130 ms for 3–6 questions.

## Findings
- **The bearing and turn sign conventions were verified on the engine and are unit-tested.** The positive turn delta turns right, so `turn = -bearing`.
- **defend_the_center is dominated by the controller.** With aligned-only firing, random roughly matches the heuristic (~20 kills out of 26 pistol rounds). With `--no-opportunistic-fire`, random scores *highest* (23.55 against 22.75–23.10 for the models; v2 campaign). Neither variant is a test of decision quality. Use deadly_corridor, and the no-fallback runs, to tell backends apart.
- **Full-level autoplay needed Doom physics in the nav graph.** Three fixes were required: nodes offset off the linedef grid; floor and ceiling taken over every sector the 32-unit box touches (Freedoom2 MAP01's exit room is behind an 8-unit crack with floor −480); and remote-door switches planned as part of the path.
- **Latency depends on the host.** Jev is about 130 ms and flat across 3 and 6 questions. Laya encodes the state once per question: 147/432/758 ms on Apple MPS, and 44/99/169 ms on the Framework GPU. On the Framework, SemIf (275–643 ms) and Clef-Flash (540–1051 ms) are slower than both.
- **Confidence isn't comparable across backends.** Laya's choice `confidence` is entropy-based (0.01 for a 56/44 split), and Jev's noul answers have none. The gate uses our own max-probability. No eval so far has passed `--calibration`, so all gating used raw probabilities. Laya's in_danger (AUC 0.88, but always above 0.5) needs a fitted bias as well as a temperature before its fallback rate means anything.

## Deviations from PLAN.md (and why)
- **Repo layout:** the package lives at the repo root (`src/s1doom`), not in an `s1doom/` subfolder.
- **Jev transport:** raw HTTP instead of `typesafe-sdk`. The wire format is small and documented, and this avoids SDK drift.
- **Extra modes:** `scan` (turn in place) for the no-movement arena and `advance` for the corridor. Question options are scenario-dependent (`configs/scenarios/*.yaml: modes`).
- **Confidence gate:** it uses our own max-probability, and the floor for choice/score scales with the option count: `1/k + 0.3·(1−1/k)`. Nouls use 0.55. A flat 0.55 would be near-impossible for an 8-option `mode` and trivial for a noul.
- **Calibration labels:** outcome labels by default (`in_danger` vs damage actually taken). Heuristic-agreement labels are opt-in (`--labels agreement`) because they reward imitating the heuristic.
- **Map model:** also parses **UDMF** (ViZDoom's scenario WADs). The nav graph uses Doom box-touch physics, directed edges and front-side-only switch and door use.
- **Controller extras:** opportunistic fire at aligned monsters (`--no-opportunistic-fire` ablation), hold fire at far range when ammo is scarce, hold fire near barrels, and a blacklist for unreachable pickups and doors.
- **Demo recording:** demos are recorded to a short temp path and moved into `runs/`, because ViZDoom 1.3.1 hangs in `new_episode()` on demo paths longer than about 112 characters.

## Not done / needs outside input
- `doom1.wad` (E1M1) isn't shipped. Drop it in `./wads/` and run `eval --scenario doom1_e1m1`; the run is skipped cleanly if the file is absent.
- **Clef 27B at full n:** its lane ran 10 episodes per scenario (5 on MAP01). Only deadly_corridor has the full 20.
- **Calibration with bias on outcome labels**, then a rerun with `--calibration`, to test Laya without its offset.
- M5 (fine-tuning Laya, a VLM front end) wasn't started.
