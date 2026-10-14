# Verification campaign v2 (2026-10-13)

We re-checked every quantitative claim in the README. There were two passes:

1. **Audit.** Every number was recomputed from the recorded `runs/`, using the definitions in `telemetry/metrics.py` and `report.py`.
2. **Rerun.** All backends were run again on the Framework Desktop with one code version (`ae293c3`, prompts v1.2), on the same seeds (1234+i).

The campaign script is `scripts/campaign_v2.sh`, and the runs are tagged `v2-*`.

| Item | Value |
|---|---|
| Host | Framework Desktop: Strix Halo, 128 GB unified memory with **96 GiB carved out as VRAM** (Linux shows 31 GB of RAM), Radeon 8060S (gfx1151), Fedora 44 |
| GPU stack | ROCm nightly torch `2.14.0.dev20260804+rocm7.1`, plus `LD_PRELOAD=/usr/lib64/libhsa-runtime64.so.1` |
| Backends | heuristic, random, Jev (hosted API), Laya `typed-decisions` 0.3.21 (`laya-framework`), SemIf Qwen3.5-4B (`semif-framework`), Clef-Flash 9B @ `8b2e5fd1` (`clef-flash`) |
| Episodes | 20 per scenario and 10 on MAP01. Each model also ran the same suite with `--no-fallback`. defend_the_center was also run with `--no-opportunistic-fire`. |
| Statistics | Bootstrap 95% CIs (10k resamples). "Δ" is the paired difference against the heuristic on the same seeds. |
| Not run | **Clef 27B**: see "Open items". |

## Provenance checks
- Every v2 run records `git: ae293c3` (clean tree) and `prompt_version: v1.2`. `calibration` is null in every run, old and new, so all gating used **raw** probabilities. The README's "after calibration" wording doesn't describe the evals.
- **The heuristic reproduces across hosts.** The Mac's 09-27/28 runs and the Framework's v2 runs are identical episode for episode on defend_the_center (20/20), health_gathering (20/20), my_way_home (20/20) and MAP01 (10/10). deadly_corridor matches on only 10/20, because v1.2's unseen-attacker `scan` changes play there: the heuristic reaches the vest 80% of the time on v1.2, up from 70% on v1.1.
- **Jev is not deterministic between runs.** On the same seeds and code, its MAP01 decision counts differ from run to run (seed 1234: 600 decisions on 09-28, 364 on 10-13).
- **No state was trimmed.** The largest state was 549 tokens against a budget of 600.
- **The Laya no-fallback MAP01 lane hit the `--max-calls 15000` cap after 5 episodes.** The cap is per run, and four timeouts of 2,650 decisions each used it up. Seeds 1239–1243 were rerun with no cap (`v2-nofb-m3b`).

## Results (outcome, mean [95% CI]; Δ vs heuristic, paired)

Scenarios and outcome measured:
- **dtc** = defend_the_center, scored on kills.
- **dc** = deadly_corridor, scored on whether the player reached the vest.
- **hgs** = health_gathering_supreme, scored on survival tics.
- **mwh** = my_way_home, scored on success.
- **MAP01** = Freedoom2 MAP01, scored on reaching the exit.

| | heuristic | random | Jev | Laya | SemIf | Clef-Flash |
|---|---|---|---|---|---|---|
| dtc kills | 20.60 [19.6, 21.6] | 21.25, Δ +0.65 [−0.65, +1.95] | 22.35, Δ **+1.75** [+1.00, +2.45] | 22.50, Δ **+1.90** [+1.00, +2.80] | 22.55, Δ **+1.95** [+1.20, +2.65] | 22.45, Δ **+1.85** [+1.10, +2.55] |
| dtc kills, no opportunistic fire | 20.85 | **23.55**, Δ +2.70 [+0.15, +4.55] | 22.75 | 23.10 | 23.05 | 23.00 |
| dc vest | 80% | 30%, Δ **−50** | 90% | 95% | 55%, Δ −25 [−50, 0] | 55%, Δ **−25** [−45, −10] |
| hgs tics | 2020 | 1236, Δ **−784** | 1969 (−2.5%) | 2020 (equals the heuristic in 20/20 episodes) | 1811 (−10.3%) | 1953 (−3.3%) |
| mwh success | 100% | 95% | 100% | 100% | 100% | 100% |
| MAP01 exit (n=10) | 100% | **90%** | 80% [50, 100] | 100% | 100% | 90% |
| fallback rate (dtc / dc / hgs / mwh / MAP01) | – | disabled | 1.6 / 6.8 / 9.4 / 0.6 / 2.1% | 12.5 / 58.8 / 65.9 / 18.8 / 65.9% | 2.0 / 4.9 / 7.7 / 0 / 1.8% | 2.2 / 7.0 / 26.0 / 0 / 6.9% |

**Model alone (`--no-fallback`):**

| | Jev | Laya | SemIf | Clef-Flash |
|---|---|---|---|---|
| dtc kills | 22.45 | 22.70 | 22.65 | 22.50 |
| dc vest | 95% | 95% | 60% | 70% |
| hgs tics | 1870 (−7.5%) | **585** (−71%) | 1935 (−4.2%) | 1866 (−7.6%) |
| mwh success | 100% | **50%** | 100% | 100% |
| MAP01 exit (n=10) | 80% | **40%** [10, 70] (6 timeouts) | 90% | 90% |

**Latency.** Each backend was benchmarked alone on the Framework, using MAP01 states, n=100 per question count. Numbers are p50 / p95 in ms.

| backend | 1 question | 3 questions | 6 questions |
|---|---|---|---|
| Jev (from the Framework) | 214 / 733 (includes connection warm-up) | 128 / 237 | 142 / 305 |
| Laya | 44 / 47 | 99 / 106 | 169 / 186 |
| SemIf | 275 / 324 | 458 / 537 | 643 / 809 |
| Clef-Flash | 540 / 590 | 795 / 836 | 1051 / 1285 |

During the campaign, four model lanes shared the GPU, so the per-decision latencies in `runs/` are inflated (Clef-Flash p50 was about 1.1–1.5 s). Use the table above for latency.

**in_danger** (noul) scored against outcome labels (damage taken in the next period). Columns: number of answers, base rate of damage, accuracy at a 0.5 threshold, the accuracy of always answering "false", Brier score, AUC, and how often the model's `mode` matches the heuristic's.

Each backend is scored on the states *it* visited, and the scenario mixes differ: Laya's pool includes six 21,000-tic MAP01 timeouts with a low damage rate. Treat the AUCs as indicative, not as a strict ranking.

| backend | n | base rate | acc@0.5 | always-false | Brier | AUC | mode = heuristic |
|---|---|---|---|---|---|---|---|
| heuristic | 15,214 | 10.7% | 79.1% | 89.3% | 0.137 | 0.70 | 100% |
| random | 15,797 | 7.3% | 49.6% | 92.7% | 0.333 | 0.52 | 44% |
| Jev | 37,550 | 8.4% | 71.8% | 91.6% | 0.160 | 0.82 | 80% |
| Laya | 59,345 | 3.6% | 3.6% | 96.4% | 0.327 | **0.88** | 42% |
| SemIf | 32,475 | 9.6% | 67.1% | 90.4% | 0.219 | 0.76 | 83% |
| Clef-Flash | 31,317 | 10.0% | 77.2% | 90.0% | 0.143 | 0.78 | 72% |

## README claims: verdicts

| Claim (README) | Verdict | Evidence |
|---|---|---|
| M1 numbers: heuristic 20.6, random 20.6, Jev 22.45 (fallback 1%), Laya 22.5 (fallback 12%) | **Confirmed** (random 21.25 on v1.2) | The audit matches; v2 matches within CIs |
| "defend_the_center is dominated by the controller" | **Confirmed, and stronger** | All four models sit at +1.75 to +1.95 and can't be told apart |
| "Use `--no-opportunistic-fire` … to tell backends apart" | **Refuted** | With it, random scores highest (23.55); all models fall within each other's CIs |
| deadly_corridor: heuristic 70%, random 15%, Jev 90%, Laya 70% | **Partly** | v1.2: heuristic 80%, random 30%, Jev 90%, Laya 95% (59% fallback). v1.1 numbers are confirmed by the audit |
| health_gathering: Jev 1859 (−8%), Laya 2020 | **Confirmed in kind** | Jev 1969 (−2.5%). Laya's 2020 *is* the heuristic's run (identical in 20/20 episodes). Laya alone scores 585 |
| MAP01: Jev 70% → **10/10 after v1.2** | **Not reproduced** | v2 got 8/10 [50, 100] (one timeout, one death). Jev isn't deterministic, so 10/10 and 8/10 are both consistent with roughly 80–90% |
| MAP01 with no fallback: Jev 80% | **Confirmed** | 8/10 |
| Laya MAP01 with no fallback: 0/1 | **Superseded** | n=1. v2 got 4/10 [10, 70]: six timeouts, no deaths |
| Random has "poor MAP01 completion" | **Refuted** (not measured before) | Random finishes 9/10. The navigator, not the decisions, completes the level |
| Laya: "58–66% fallback" on defend_the_center (summary card) | **Wrong** | 12.5%. 58–66% is the deadly_corridor/health_gathering range; MAP01 is 66% |
| Laya's probabilities are "near-uniform"; `in_danger` is "anti-informative" (4%) | **Refuted** | The ~4% accuracy reproduces (3.6%), but Laya has the best AUC (0.88). All 59,345 of its in_danger probabilities fall between 0.517 and 0.658 (median 0.58): it ranks danger well but is badly offset. That is a calibration problem, not a lack of signal |
| Jev in_danger 90%, "informative enough to serve as an outcome label" | **Refuted** | 71.8% accuracy, below the 91.6% of always answering "false". AUC is 0.82, and its Brier score is worse than the heuristic's on the same states |
| Laya is 7B | **Wrong** | `typed-decisions` has 0.42B parameters (counted from its safetensors) |
| Latency: Jev ~120 ms flat; Laya on the Framework 56/113/181 | **Confirmed** | Jev 128/142 ms (3/6 questions); Laya 44/99/169 |
| Jev p95 ~170 ms | **Wrong** | 183–209 ms on the Mac eval runs; 237–305 ms in the Framework bench |
| SemIf "no eval run yet" | **Out of date** | 6 runs existed already; v2 reproduces SemIf's 09-28 numbers exactly (dtc 22.55, hgs 1811) |
| Laya "degrades to ~5 s/decision on MPS" | **Unsupported** | The logged model latency was flat at ~337 ms; the slowdown was outside the model call (audit). Not re-tested |
| Framework: "32 GB RAM, no discrete GPU, no ROCm userspace" | **Wrong** | 96 GiB VRAM carve-out; ROCm torch works for Laya, SemIf and Clef-Flash |
| "`uv sync --extra clef && uv pip uninstall triton`" | **Harmful** | It replaced the ROCm torch with a CUDA build and removed `laya`, which would have broken laya-serve on its next restart |
| Clef-Flash spot check: 7 kills, 0 damage, 0% fallback, 14 decisions, 100% agreement, 1.4–2.3 s | **Refuted** | On the GPU, n=20: 22.45 kills, about 2% fallback, a median of 134 decisions per episode, 72% mode agreement. Every v2 defend_the_center episode, for every backend, ends in death after at least 100 damage taken, so a 14-decision episode with 0 damage wasn't a normal run. Latency on the GPU is 540–1051 ms |
| Clef-Flash "0% fallback is the headline" | **Refuted** | 0–26% depending on the scenario (health_gathering 26%) |
| Clef 27B "does not fit" | **Open** | The weights fit in VRAM, but transformers' loader stages them in the 31 GB of system RAM and the process is killed for running out of memory (twice, at ~20 GB). See below |

## What the data supports
- **Only deadly_corridor separates the backends.** Jev (90%) and Laya (95%) beat SemIf and Clef-Flash (55%). Clef-Flash is the only model significantly worse than the heuristic there (Δ −25 points, CI excludes 0). With no fallback, Jev and Laya stay at 95%, and SemIf and Clef-Flash reach 60–70%.
- **Everywhere else, n=20 can't tell models apart, or tell them from the heuristic**, except that every model is about 1.9 kills better on defend_the_center, and random comes close.
- **Laya with fallback measures the heuristic, not Laya.** On its own it fails health_gathering (585 tics) and my_way_home (50%). It finishes MAP01 only 4/10 times (six timeouts). Jev, SemIf and Clef-Flash hold up without fallback (80–90% on MAP01).
- **Clef-Flash is a usable backend.** It plays like SemIf and slightly worse than Jev. It is the slowest of the four (0.5–1.1 s on this iGPU).

## Open items
- **Clef 27B.** The 52 GB download is complete. Loading it fails because transformers builds a CPU-side copy of the weights in the 31 GB of system RAM. Options:
  1. Load the shards straight into GPU memory from a meta-device model, validated by matching Clef-Flash's logits.
  2. Reduce the BIOS carve-out (for example, 64 GB VRAM / 64 GB RAM).
  3. Add a temporary large swapfile (needs sudo).
- **Calibration.** Laya's in_danger has the best ranking (AUC 0.88) but sits above 0.5, so it mostly falls back. Fitting a temperature *and bias* on outcome labels, and evaluating with `--calibration`, would test whether Laya is better than its fallback rate suggests.
- **Sample size.** MAP01 at n=10 can't resolve differences smaller than about 30 points. deadly_corridor is the scenario worth scaling to more seeds.
