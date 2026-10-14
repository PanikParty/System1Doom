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
| Clef 27B | Run afterwards on commit `6187e8e`, which differs from `ae293c3` only in `serve_clef.py`. It used a reduced lane: 10 episodes per scenario and 5 on MAP01, with deadly_corridor at the full 20, with and without fallback. See "Clef 27B". |

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
| Clef 27B | 1260 / 1480 | 1973 / 2173 | 2741 / 3320 |

During the campaign, four model lanes shared the GPU, so the per-decision latencies in `runs/` are inflated (Clef-Flash p50 was about 1.1–1.5 s). Use the table above for latency.

**in_danger** (noul) scored against outcome labels (damage taken in the next period). Columns: number of answers, base rate of damage, accuracy at a 0.5 threshold, the accuracy of always answering "false", Brier score, AUC, and how often the model's `mode` matches the heuristic's.

**Don't rank backends by the pooled AUC below.** Each backend is scored on the states *it* visited, and the scenario mixes differ, so pooling mixes scenarios with very different damage rates (Simpson's paradox). Laya's pooled 0.88 is the clearest case: 34k of its 59k answers come from MAP01 (almost no damage, lower p), while health_gathering has 24% damage and higher p. Within each scenario Laya ranks danger *worst* of the models. Use the per-scenario table after this one.

| backend | n | base rate | acc@0.5 | always-false | Brier | AUC | mode = heuristic |
|---|---|---|---|---|---|---|---|
| heuristic | 15,214 | 10.7% | 79.1% | 89.3% | 0.137 | 0.70 | 100% |
| random | 15,797 | 7.3% | 49.6% | 92.7% | 0.333 | 0.52 | 44% |
| Jev | 37,550 | 8.4% | 71.8% | 91.6% | 0.160 | 0.82 | 80% |
| Laya | 59,345 | 3.6% | 3.6% | 96.4% | 0.327 | **0.88** | 42% |
| SemIf | 32,475 | 9.6% | 67.1% | 90.4% | 0.219 | 0.76 | 83% |
| Clef-Flash | 31,317 | 10.0% | 77.2% | 90.0% | 0.143 | 0.78 | 72% |
| Clef 27B | 15,124 | 10.6% | 65.7% | 89.4% | 0.195 | 0.77 | 80% |

**in_danger AUC within each scenario** (damage base rate in brackets). my_way_home never produces damage. On health_gathering, damage comes from the acid floor and every source is near or below chance. defend_the_center and deadly_corridor are the informative rows.

| scenario | heuristic | Jev | Laya | SemIf | Clef-Flash | Clef 27B |
|---|---|---|---|---|---|---|
| defend_the_center (≈3.9%) | 0.68 | 0.89 | **0.72** | 0.91 | 0.89 | 0.91 |
| deadly_corridor (5–9%) | 0.62 | 0.86 | **0.74** | 0.83 | 0.79 | 0.86 |
| health_gathering_supreme (≈24%) | 0.03 | 0.43 | 0.54 | 0.43 | 0.47 | 0.45 |
| freedoom2_map01 (0–0.9%) | 0.00 | 0.81 | 0.57 | 0.83 | 0.74 | 0.85 |

**in_danger has no effect on play.** `Policy` sets `Decision.in_danger`, but nothing in `control/` reads it; it is only logged and drawn on the overlay. It is an evaluation probe, not a decision.

## Laya calibration (temperature + bias)
`s1doom calibrate --bias` fits Platt scaling for noul questions (logit' = logit/T + b) on outcome labels. Fitted on all of Laya's v2 in_danger answers (n = 59,345): **T = 0.050, b = −10.69** (`configs/calibration-laya-v2.yaml`).

Leave-one-scenario-out (fit on four scenarios, score the fifth), Brier score:

| held-out scenario | raw | calibrated | heuristic | constant base rate |
|---|---|---|---|---|
| defend_the_center | 0.343 | 0.059 | 0.106 | 0.038 |
| deadly_corridor | 0.321 | 0.047 | 0.116 | 0.047 |
| health_gathering_supreme | 0.315 | 0.228 | 0.248 | 0.238 |
| my_way_home | 0.352 | 0.004 | 0.022 | 0.002 |
| freedoom2_map01 | 0.320 | 0.001 | 0.023 | 0.008 |

**What the calibration shows:**
- **It removes Laya's offset.** Brier drops by about 6× and accuracy rises from 4% to 93–100%.
- **But calibrated Laya is about as good as predicting the base rate, no better.** It beats the base rate on MAP01 (0.001 vs 0.008) and health_gathering (0.228 vs 0.238), ties on deadly_corridor, and loses on defend_the_center (0.059 vs 0.038) and my_way_home. On health_gathering its held-out log-loss is also worse than the raw probabilities' (0.900 vs 0.825), a sign of overconfidence when the fit is transferred.
- **Laya's in_danger carries little information beyond the base rate.** That is consistent with its weak within-scenario AUC.

**The rerun changes nothing, as predicted.** Laya with `--calibration` on defend_the_center, seeds 1234–1236, flips in_danger from "true" on 138/138 decisions to 0/138, and the games are **identical** to the uncalibrated runs: same kills, damage, tics, decision counts and modes. Because in_danger isn't used for control, a full calibrated rerun would only replay the v2 episodes, so none was run.

Laya's high fallback rate comes from its multi-option questions, not in_danger:

| question | Laya fallback rate (v2 runs with fallback) |
|---|---|
| mode | 81% |
| target | 100% |
| pickup | 100% |
| aggression | 100% |
| in_danger | 2% |

Those questions have no outcome labels, so outcome calibration can't help. The no-fallback runs already show how Laya's own choices play.

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
| Laya's probabilities are "near-uniform"; `in_danger` is "anti-informative" (4%) | **Partly** | The ~4% accuracy reproduces (3.6%): all 59,345 of its in_danger probabilities fall between 0.517 and 0.658, so it says "true" every time. It isn't anti-informative: within scenarios its AUC is 0.72–0.74, above the heuristic's. But it is the weakest of the models, and once calibrated it is no better than the base rate. (An earlier version of this report called its pooled AUC of 0.88 "the best ranking"; that was a pooling artifact.) Laya's *choice* probabilities are near-uniform: 81–100% of mode, target, pickup and aggression answers fall back |
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
| Clef 27B "does not fit" | **Wrong** | The weights fit in VRAM. Only transformers' loader failed: it stages the whole checkpoint in the 31 GB of system RAM. `serve_clef.py --loader direct` loads it in 34.5 s |

## Clef 27B
**Loading.** `from_pretrained` copies the whole checkpoint into host RAM before moving it to the GPU. On Clef-Flash the peak was 19.7 GB, about the model's size. For the 27B that copy was killed twice at ~20 GB of the 31 GB of system RAM.

`serve_clef.load_direct` (`--loader direct`, now the default) avoids the copy:
1. It builds the backbone on the GPU with weight initialisation skipped.
2. It rebuilds the rotary buffers on the CPU. Computed on the GPU, `inv_freq` differs by up to 3e-8, and that moved probabilities by up to 0.01.
3. It copies each checkpoint tensor straight into its parameter. Checkpoint and model keys match one to one for both releases.

`scripts/check_clef_loader.py` validated it on Clef-Flash:
- 760/760 tensors are bit-identical to the checkpoint.
- 190/190 answers on 60 real states are identical to a `from_pretrained` server.
- Peak RSS is 9.8 GB, down from 19.7 GB.

The 27B then loads in 34.5 s and takes about 57 GB of VRAM (19.3 → 76.8 GB in use). Clef-Flash was stopped during its runs to leave headroom.

**Results, on the same seeds as the other backends.** Each column is restricted to the 27B's seeds:

| outcome (n) | heuristic | random | Jev | Laya | SemIf | Clef-Flash | **Clef 27B** |
|---|---|---|---|---|---|---|---|
| dtc kills (10) | 20.10 | 20.90 | 22.30 | 21.90 | 23.00 | 22.80 | 22.90 |
| dtc kills, no opportunistic fire (10) | 20.80 | 24.80 | 23.60 | 22.90 | 24.00 | 23.80 | 24.00 |
| dc vest (20) | 80% | 30% | 90% | 95% | 55% | 55% | **90%** |
| hgs tics (10) | 2080 | 1202 | 2072 | 2080 | 1890 | 1963 | 2017 |
| mwh success (10) | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| MAP01 exit (5) | 100% | 100% | 80% | 100% | 100% | 100% | 100% |
| *no fallback:* dc / hgs / mwh / MAP01 | – | – | 95% / 1932 / 100% / 60% | 95% / 593 / 50% / 40% | 60% / 1999 / 100% / 100% | 70% / 1897 / 100% / 80% | **95%** / 1890 / 100% / 100% |
| fallback (dtc / dc / hgs / mwh / MAP01) | | | | | | | 4 / 6 / 23 / 0 / 3% |

**27B vs Clef-Flash on deadly_corridor**, paired on the same 20 seeds:
- With fallback: **+35 points [+15, +55]**. Of the 7 seeds where only one of them reached the vest, all 7 went to the 27B.
- No fallback: **+25 [+5, +45]**, 5 to 0.

That is the only significant difference between the two sizes. Elsewhere the 27B is indistinguishable from the other models. It is the slowest backend: 1.3–2.7 s per decision on this iGPU.

## What the data supports
- **Only deadly_corridor separates the backends.** Jev (90%), Laya (95%) and Clef 27B (90%) beat SemIf and Clef-Flash (55%). Clef-Flash is the only model significantly worse than the heuristic there (Δ −25 points, CI excludes 0). With no fallback, Jev, Laya and Clef 27B reach 95%, and SemIf and Clef-Flash 60–70%.
- **Everywhere else, n=20 can't tell models apart, or tell them from the heuristic**, except that every model is about 1.9 kills better on defend_the_center, and random comes close.
- **Laya with fallback measures the heuristic, not Laya.** On its own it fails health_gathering (585 tics) and my_way_home (50%). It finishes MAP01 only 4/10 times (six timeouts). Jev, SemIf and Clef-Flash hold up without fallback (80–90% on MAP01).
- **Clef-Flash is a usable backend.** It plays like SemIf and slightly worse than Jev, and takes 0.5–1.1 s per decision on this iGPU.
- **Clef 27B is clearly better than Clef-Flash on deadly_corridor** (+35 points, paired) and matches Jev there, but it costs 1.3–2.7 s per decision. On every other scenario it is indistinguishable from the rest.

## Open items
- **Labels for choice questions.** Laya's fallback comes from mode, target, pickup and aggression. Calibrating those needs outcome labels per option (for example, whether the chosen target died, or whether the mode led to damage). Heuristic-agreement labels only reward imitation.
- **Clef 27B at full n.** Its lane ran 10 episodes per scenario (5 on MAP01), except for deadly_corridor.
- **Sample size.** MAP01 at n=10 can't resolve differences smaller than about 30 points. deadly_corridor is the scenario worth scaling to more seeds.
