# s1doom: System One models play DOOM

A harness that lets **text-only System One decision models** play classic DOOM through ViZDoom. The supported models are TypeSafe **Jev** (API) and the open-weight **Laya**. Code turns the engine state into a compact JSON state and a few atomic typed questions whose options are only the actions legal right now. The model picks the tactic and target. A deterministic controller turns that into button presses every tic.

> **Rule: code computes the geometry, and the model makes the judgment.** (PLAN.md §0)

See `PLAN.md` for the design. Results live in `runs/report.md` (`uv run s1doom report`).

## Quick start
```bash
uv sync                      # core (ViZDoom ships Freedoom1/2)
uv sync --extra laya         # + Laya in-process / laya-serve (torch)
uv run pytest -q             # 36 tests: unit, contract (recorded Jev+Laya fixtures), engine smoke

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
| any `systemone-http` | Jev-compatible HTTP | Point a `configs/backends/<name>.yaml` with `kind: systemone-http` at any server that speaks the protocol. |
| `replay` | calls.jsonl | Deterministic regression runs. |

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

## Results (turn mode, seed 1234, full tables and charts in `runs/report.md`)

| milestone gate (PLAN §4) | heuristic | random | **Jev** | **Laya** (typed-decisions, MPS) |
|---|---|---|---|---|
| M1 defend_the_center kills (20 ep), gate: ≥90% of heuristic, fallback <25% | 20.6 ± 1.1 | 20.6 ± 0.7 | **22.45 ± 1.2** ✅ (fallback 1%) | **22.5 ± 0.8** ✅ (fallback 12%) |
| M2 deadly_corridor vest reached (20 ep, skill 5), gate: ≥50% | 70% | 15% | **90%** ✅ | 70% ✅ (fallback 58%) |
| M2 health_gathering_supreme survival tics (20 ep), gate: ≥ heuristic −10% | 2020 | 1236 | **1859** ✅ (−8%) | 2020 ✅ (fallback 66%) |
| M2 my_way_home success (20 ep) | 100% | 95% | 100% | 100% (fallback 19%) |
| M3 Freedoom2 MAP01 exit (10 seeds, skill 2), gates: heuristic ≥60%, Jev ≥40% | **100%** ✅ | – | **70%** ✅ | 100% (fallback 71%) |
| MAP01 with **no fallback** (5 seeds): the model alone | – | – | **80%** | 0/1 (timed out at 32% explored; stopped: ~5 s/decision on MPS) |
| Latency p50 / p95 per decision | – | – | 116–130 / ~170 ms | 232–449 ms |

**With the v1.2 unseen-attacker fix** (`scan` becomes legal when damage comes from nothing visible), Jev completes **10/10** MAP01 seeds (up from 7/10), with less damage taken. Laya wasn't re-run on v1.2.

**How to read the Laya rows:** Laya's probabilities are flat, so 58–66% of its answers fall below the confidence floor and the heuristic answers instead. Its M2 numbers are therefore largely the heuristic's. With no fallback, Laya still scores 22.8 kills on defend_the_center (5 ep), but didn't finish MAP01. Laya's `in_danger` is anti-informative on outcome labels (4% accuracy); Jev's is 90%.

Realtime mode (35 Hz, ASYNC_PLAYER) runs without frame stalls (0 overruns, p99 per-tic work 8 ms). With Laya on MPS, answers arrive with 230–480 ms staleness, and those older than 400 ms are dropped.

## Findings
- **The bearing and turn sign conventions were verified on the engine and are unit-tested.** The positive turn delta turns right, so `turn = -bearing`.
- **defend_the_center is dominated by the controller.** With aligned-only firing, even `random` roughly matches the heuristic (~20 kills out of 26 pistol rounds), so it's a weak test of decision quality. Use `--no-opportunistic-fire` and the harder scenarios to tell backends apart.
- **Full-level autoplay needed Doom physics in the nav graph.** Three fixes were required: nodes offset off the linedef grid; floor and ceiling taken over every sector the 32-unit box touches (Freedoom2 MAP01's exit room is behind an 8-unit crack with floor −480); and remote-door switches planned as part of the path.
- **Jev versus Laya latency:** Jev is about 120 ms and flat across 1, 3 and 6 questions. Laya typed-decisions on Apple MPS takes about 145 ms per question (147/432/758 ms). Laya encodes the state once per question, so in realtime mode Laya answers are often stale on this host. A CUDA host is the fix; benchmark it with `bench-backend`.
- **Confidence isn't comparable across backends.** Laya's choice `confidence` is entropy-based (0.01 for a 56/44 split), and Jev's noul answers have none. The gate uses our own max-probability after temperature calibration.

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
- **Laya on a faster host (e.g. the Framework Desktop):** not run yet. On this Mac (MPS) Laya costs ~145 ms per question, and overnight it degraded to ~5 s per decision. Steps:
  1. On the Framework: `git clone` this repo, `uv sync --extra laya` (use a ROCm or CUDA torch build for its GPU), then `LAYA_MODELS=typed-decisions uv run s1doom serve-laya` (listens on :8000; set `LAYA_API_KEY` to require a bearer token).
  2. Here: set `url` in `configs/backends/laya-framework.yaml` (default `http://framework.local:8000`), then run `uv run s1doom bench-backend --backend laya-framework` to measure latency over the network.
  3. Evals: `uv run s1doom eval --scenario defend_the_center,freedoom2_map01 --backends laya-framework --episodes 10 --tag m3` (add `--no-fallback --tag nofallback` for Laya alone).
  4. Or run everything on the Framework with `--backends laya-inproc`, which avoids the network hop.
- M5 (fine-tuning Laya, a VLM front end) wasn't started.
