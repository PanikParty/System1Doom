# System 1 DOOM Player — Implementation Plan

Handoff doc for Claude Code. Build a harness that lets a **System One model** (TypeSafe **Jev**, or the open-weight **Laya**) play classic DOOM via ViZDoom. Status: planned, not started. Date: 2026-09-27.

---

## 0. The one fact that shapes everything

Jev and Laya are **not** vision or control policies. They are *typed decision functions*:

```
state (text / JSON)  +  questions {choice | score | noul}  →  typed answers + probabilities + confidence
```

- Jev: "Jev accepts text only… Images, audio, and video are not supported." Latency 70–500 ms (a community DOOM run measured ~212 ms median).
- Laya: bidirectional encoder (ModernBERT-large 421M / mmBERT-base 322M) with decision heads. ~33–40 ms for 1 question on a T4, 72–159 ms for 10 questions; 193–464 ms on CPU. Context **512 tokens** (`laya`) or **1,024** (`laya-typed-decisions`, `laya-multilingual`).
- `laya-serve` exposes **`POST /v1/systemone`, which is designed to be Jev-compatible**. That lets us write **one wire protocol and two backends**.

So the harness has three jobs:

1. **Perception to text:** turn ViZDoom engine state into a compact, *pre-digested* JSON state.
2. **Ask good questions:** pick a small set of atomic, well-scoped typed questions whose options are only the actions that are legal right now.
3. **Motor control in code:** a deterministic controller turns the chosen *macro/target* into button presses every tic.

**Lesson from prior art** ([jev-plays-doom](https://github.com/tirukovelamanoj/jev-plays-doom)): when the state held raw geometry and the model had to infer "is the enemy lined up?", it **never fired** and confidence collapsed. When the state or rule gave the threshold explicitly, it matched the heuristic. **Rule: code computes the geometry, and the model makes the judgment.** Put derived booleans and bucketed values in state (`aligned: true`, `range: "close"`), not raw angles alone.

---

## 1. Goals and non-goals

**Goals (v1, staged)**
- M1: survive and kill in the arena scenario (`defend_the_center`).
- M2: corridor combat and pickups (`deadly_corridor`, `health_gathering_supreme`, `my_way_home`).
- M3: **autoplay a full level**: Freedoom2 `MAP01` (bundled with ViZDoom), and `doom1.wad` `E1M1` if the user drops it in. Covers exploring, doors, keys, combat, pickups and reaching the exit.
- Pluggable backends: `jev`, `laya-http`, `laya-inproc`, `heuristic`, `random`, `replay`.
- Reproducible evals and a head-to-head comparison: Jev vs Laya vs heuristic.
- Full observability: a JSONL log of every call, demo recordings (`.lmp`), and a live overlay.

**Non-goals (v1)**
- Pixel-based policies and RL training of a control policy.
- Multi-level campaigns and multiplayer. The design should leave room for them.
- Shipping commercial WADs.

---

## 2. Architecture

```
┌──────────────┐  tic   ┌───────────────┐ state JSON ┌────────────────┐  HTTP / in-proc  ┌─────────────┐
│  ViZDoom     │──────▶ │ Perception    │──────────▶ │ Decision Layer │ ───────────────▶ │  Backend    │
│  (engine)    │        │  + Map Model  │            │ (questions,    │ ◀─────────────── │ Jev / Laya /│
└──────▲───────┘        └───────────────┘            │  gating,       │   typed answers  │ heuristic   │
       │ buttons                                     │  commitment)   │                  └─────────────┘
┌──────┴───────┐   macro + target                    └───────┬────────┘
│  Controller  │ ◀───────────────────────────────────────────┘
│ (aim, move,  │        ▲ nav path
│  use, fire)  │ ───────┘ Navigator (A* over static map graph)
└──────────────┘
            Telemetry: JSONL calls · episode metrics · .lmp demos · overlay HUD
```

### 2.1 Loop timing (two modes, both required)

| Mode | ViZDoom mode | Behavior | Use |
|---|---|---|---|
| `turn` (default) | `Mode.PLAYER` (sync) | Game **pauses** while the model thinks. A decision fires every `decision_tics` (default 8 ≈ 4.4 Hz) **or** on an event trigger. | Deterministic evals, fair model comparison |
| `realtime` | `Mode.ASYNC_PLAYER` at 35 Hz | Decisions run in a background worker. The controller keeps executing the **current macro** until a fresher decision lands. Stale answers (older than `max_staleness_ms`) are dropped. | Demos, and measuring how latency hurts play |

**Event triggers** force an immediate decision in either mode:
- a new monster becomes visible
- the target dies or is lost
- damage is taken
- health crosses a threshold
- the current macro completes
- the agent is stuck (no movement for N tics)
- a key or door is encountered

### 2.2 Perception: `perception/`

Enable in ViZDoom: `labels_buffer`, `objects_info`, `sectors_info`, `automap_buffer` (for the overlay only), `depth_buffer` (optional), plus these game variables:

`HEALTH, ARMOR, SELECTED_WEAPON, SELECTED_WEAPON_AMMO, AMMO0-9, WEAPON0-9, KILLCOUNT, ITEMCOUNT, SECRETCOUNT, POSITION_X/Y/Z, ANGLE, DAMAGECOUNT, HITCOUNT, DEAD, ON_GROUND`

**Fairness policy:** monsters and items in state come from **`state.labels` (currently visible objects) only**, plus short-term memory of things seen recently, decayed over time. `state.objects` is omniscient. Use it only for (a) debug and (b) an explicit `--omniscient` ablation. Make this a config flag and record it in every log.

Build an entity table for each visible entity:
- `id`
- `kind`: the class name, normalized into `monster | pickup_health | pickup_armor | pickup_ammo | weapon | key | barrel | projectile | other` via a lookup table in `perception/classes.py`
- `distance`
- `bearing_deg`: **0 = dead ahead, positive = LEFT, negative = RIGHT**, matching Doom's counter-clockwise ANGLE. Write a unit test for this, because prior art got it wrong and the bot spun in place.
- `screen_x_frac`, `screen_width_frac` from the label bounding box
- `aligned`: `abs(bearing) < aim_tol(distance)`
- `range_bucket`: `melee <96 | close <384 | mid <1024 | far`
- `threat`: a scalar from a per-monster danger table × proximity
- `hp_est`: from a per-class table

Items that aren't visible come from the map model (below).

### 2.3 Static map model: `mapmodel/`

ViZDoom's `Line` has only `x1, y1, x2, y2, is_blocking`, with **no linedef specials**. To find exits, doors, keys and switches we parse the WAD ourselves with **`omgifol`**, reading the map lumps (`LINEDEFS, SIDEDEFS, SECTORS, VERTEXES, THINGS`). Doom and Doom 2 format are enough for Freedoom and doom1.

- **Nav graph:** build nodes as a grid (32 map units) inside sectors, or as sector portals. Edges connect nodes where the step height is ≤ 24, headroom is ≥ 56, and no blocking linedef is crossed. Precompute once per map and cache it to `cache/<wad>_<map>.navgraph.pkl`.
- **Tag these specials:**

  | Tag | Specials |
  |---|---|
  | Exit | 11, 51, 52, 124 (switch/walk exit, secret exit) |
  | Doors | 1, 26–28 (keyed), 31–34, 117, 118 |
  | Lifts | 62, 88, 120, 121 |
  | Switches | generic S1/SR list |

  Keep a table in `mapmodel/specials.py`.
- **Dynamic updates:** door and lift sectors change height at runtime. Refresh the heights of those sectors from `state.sectors` each decision and re-mark their edges.
- **Exploration memory:** keep a visited grid plus **frontier cells** (reachable, never seen). "Seen" comes from visited cells, not from the omniscient map.
- **Navigator:** A* on the graph to produce a waypoint list. It also handles doors: approach, face the door, press `USE`, then wait for the sector to open. It re-plans when progress stalls.

### 2.4 State serialization: `decision/state_builder.py`

Build a compact JSON object with a **token budget**. Laya `laya-typed-decisions` has a 1,024-token context. Target **≤ 600 tokens** to leave room for questions. Measure with the Laya tokenizer, and apply the same budget to Jev for a fair comparison.

```json
{
  "self": {"health": 62, "armor": 0, "weapon": "shotgun", "ammo": 11, "low_ammo": false, "taking_damage": true},
  "situation": {"monsters_visible": 2, "nearest_threat_range": "close", "cornered": false, "objective": "reach_exit", "exit_known": true, "exit_path_len": "mid"},
  "monsters": [
    {"ref": "m1", "kind": "imp", "range": "close", "bearing_deg": 12, "aligned": false, "side": "left", "threat": "high"},
    {"ref": "m2", "kind": "zombieman", "range": "far", "bearing_deg": -40, "aligned": false, "side": "right", "threat": "low"}
  ],
  "pickups": [{"ref": "p1", "kind": "health", "range": "close", "worth_it": true}],
  "doors": [{"ref": "d1", "needs_key": null, "range": "mid"}],
  "keys_held": [],
  "recent": ["took 8 damage from m1", "killed zombieman"],
  "rules": "bearing 0 = straight ahead; positive = left. Only ATTACK when target aligned=true."
}
```

Guidelines:
- Cap the lists: monsters ≤ 5, sorted by threat; pickups ≤ 3; doors ≤ 2.
- Stable short refs (`m1`, `p1`) are used as choice option keys.
- Prefer **buckets and booleans** over raw floats. Keep one raw number where it helps, such as `bearing_deg`.
- The `rules` string is part of the state. Treat it as a tunable prompt and version it.

### 2.5 Questions: `decision/questions.py`

Keep each question **atomic** (TypeSafe guidance), and send all of them in **one call**. Both Jev and Laya score multiple questions in parallel. Options are **built dynamically**: include only what's legal right now. For example, drop `fight` if no monster is visible and `use_door` if no door is in range.

| Key | Type | Options / criteria | Consumed by |
|---|---|---|---|
| `mode` | choice | `fight`, `retreat`, `take_cover`, `grab_pickup`, `explore`, `go_to_exit`, `open_door`, `get_key` | Macro selector |
| `target` | choice | `m1..mk` with a one-line description each | Aim controller (when mode ∈ fight/retreat) |
| `pickup` | choice | `p1..pk`, `none` | Navigator goal |
| `in_danger` | noul | "Will you take significant damage in the next second if you keep doing what you are doing?" | Gating and retreat bias |
| `aggression` | score | 4 levels: `flee`, `cautious`, `engage`, `all-in` | Strafe/advance parameters |
| `weapon` | choice | owned weapons with ammo | Weapon switch (asked only when owned > 1, every ~2 s) |

**Question scheduler:** asking every question every tick wastes the context budget and adds latency. Ask `mode` and `in_danger` every decision. Ask `target` only in combat, `pickup` only when one is visible, and `weapon`/`aggression` at a lower rate.

Keep all question text and criteria in `prompts/*.yaml`. Give them versions, and record the prompt version in every log line.

### 2.6 Gating, commitment, fallback: `decision/policy.py`

- **Confidence floor** (per question, configurable, default 0.55): below it, use the **heuristic policy's** answer for that question. Mark it `source: "fallback"` in logs and on the overlay. Never pass a fallback off as model output.
- **Commitment/hysteresis:** a macro holds for at least `min_commit_tics` (default 12) unless a trigger fires. Switch targets only if the new target's probability beats the current one by more than `switch_margin`.
- **Self-consistency (optional):** in realtime mode with spare budget, ask two phrasings of `mode` and act only when they agree. See the TypeSafe self-consistency cookbook.
- **Calibration layer:** apply per-(backend, question, n_options) temperature scaling to probabilities before gating. Laya's docs say the shipped checkpoints are **over-confident** (ECE 0.466 → 0.081 after fitting). Fit temperatures offline from logged runs (§5).

### 2.7 Controller: `control/`

The controller is deterministic and runs every tic. It uses continuous delta buttons:
- `TURN_LEFT_RIGHT_DELTA`, `MOVE_FORWARD_BACKWARD_DELTA`, `MOVE_LEFT_RIGHT_DELTA`
- `ATTACK`, `USE`, `SPEED`, `SELECT_NEXT_WEAPON` (or direct weapon slots)

Each macro has a controller behavior:

| Macro | Behavior |
|---|---|
| fight | P-controller on the target's bearing. Fire when `aligned` and the target is visible. Doom's vertical autoaim handles pitch. Circle-strafe when `aggression ≥ engage` and threat is high. |
| retreat / take_cover | Navigator goal = the nearest graph node that breaks line of sight to the threats (visibility test against blocking lines), backpedaling while facing the target. |
| grab_pickup / get_key | Navigator to the item's position. |
| explore | Navigator to the best frontier cell (nearest first, tie-broken by unexplored area). |
| go_to_exit | Navigator to the exit linedef, then face it and press `USE` (switch exits) or walk over it. |
| open_door | Approach, face, `USE`, then wait. |

**Stuck recovery:** if there's no progress for 35 tics, try a short random strafe plus `USE`, then re-plan.

### 2.8 Backends: `backends/`

```python
class DecisionBackend(Protocol):
    name: str
    async def system_one(self, state: dict, questions: dict) -> DecisionResult: ...
# DecisionResult: answers{key: Answer(type, value, probabilities, confidence)}, latency_ms, usage, raw
```

| Backend | Transport | Notes |
|---|---|---|
| `jev` | `typesafe-sdk` (`TypeSafeClient` / async client) or raw `POST https://api.typesafe.ai/v1/systemone`, `model="jev-latest"` | Reads `TYPESAFE_API_KEY` from env. Retry 429/529 with exponential backoff and jitter. Choice options ≤ 255 and score levels 2–10; validate before sending. Track `usage.input_tokens` for cost (~$0.042/M input tokens per launch post). |
| `laya-http` | `POST {LAYA_URL}/v1/systemone` from `laya-serve` | **Host is config** (Mac MPS, Framework Desktop, Jetson). Optional `LAYA_API_KEY` bearer. |
| `laya-inproc` | `laya.load("convaiinnovations/laya", subfolder="typed-decisions", fast=True)` | Default checkpoint: **typed-decisions**. Base `laya` scored 0.36 vs 0.77 on typed decisions. Avoids the network hop. |
| `heuristic` | pure Python | Hand-written policy with the same answer shape. Serves as baseline, fallback and label source. |
| `random` | — | Uniform over legal options (sanity floor). |
| `replay` | JSONL | Replays logged answers for deterministic regression tests. |

**Normalization:** Jev noul answers have no `confidence`, but Laya's do. Derive a confidence for nouls as `max(p, 1-p)` when it's missing. Map Laya's `distribution` (a list) onto a probabilities dict. Write contract tests that run recorded Jev and Laya fixtures through the same `DecisionResult`.

**Latency benchmark:** the command `s1doom bench-backend --backend laya-http --url ...` sends 200 real recorded states, varies the question count between 1, 3 and 6, and reports p50/p95/p99. Use it to choose a Laya host.

### 2.9 Telemetry and UX: `telemetry/`

- **JSONL** per episode (`runs/<run_id>/calls.jsonl`) with these fields per line: `tic`, `state`, `questions`, `answers`, `calibrated_probs`, `source` (model/fallback), `latency_ms`, `backend`, `prompt_version`, `fairness`.
- **Demos:** `game.new_episode("runs/<id>/ep_<n>.lmp")` for exact replay.
- **Overlay:** an OpenCV window over `screen_buffer` showing the current macro, the target box (from the label), the top-3 probabilities for `mode`, confidence bars, a FALLBACK badge, and a latency sparkline. An automap inset shows the nav path and frontier. Toggle with `--overlay`.
- **Run summary:** `runs/<id>/summary.json`, with the metrics from §5.

---

## 3. Repo layout

```
s1doom/
  pyproject.toml            # uv; python 3.11+; deps: vizdoom>=1.3, omgifol, numpy, opencv-python, pydantic, httpx, typesafe-sdk, laya[serve] (extra), pyyaml, typer, rich, pytest
  CLAUDE.md                 # conventions for Claude Code (see §7)
  configs/
    scenarios/*.yaml        # defend_the_center, deadly_corridor, health_gathering_supreme, my_way_home, freedoom2_map01, doom1_e1m1
    backends/*.yaml         # jev, laya-local, laya-framework, laya-jetson, heuristic
  prompts/v1/*.yaml         # rules text + question definitions
  src/s1doom/
    env/            game.py (ViZDoom setup, modes, recording), wads.py (locate freedoom/doom1)
    perception/     entities.py, classes.py, geometry.py (bearing/aim), memory.py
    mapmodel/       wadparse.py (omgifol), specials.py, navgraph.py, frontier.py, visibility.py
    decision/       state_builder.py, questions.py, scheduler.py, policy.py, calibration.py, tokens.py
    control/        controller.py, aim.py, navigator.py, macros.py, stuck.py
    backends/       base.py, jev.py, laya_http.py, laya_inproc.py, heuristic.py, random_.py, replay.py
    telemetry/      jsonl.py, overlay.py, metrics.py
    cli.py          # typer app: play, eval, bench-backend, calibrate, replay, inspect-map
  tests/            unit + contract + smoke
  runs/  cache/     # gitignored
```

CLI examples:

```bash
s1doom play --scenario defend_the_center --backend heuristic --overlay
s1doom play --scenario freedoom2_map01 --backend jev --mode realtime --overlay
s1doom eval --scenario deadly_corridor --backends heuristic,jev,laya-http --episodes 20 --seed 1234
s1doom bench-backend --backend laya-http --url http://framework.local:8000
s1doom calibrate --runs runs/2026-10-* --out configs/calibration.yaml
s1doom inspect-map --wad freedoom2.wad --map MAP01   # renders nav graph, exits, doors to PNG
```

---

## 4. Milestones (each ends with passing acceptance criteria)

**M0: Skeleton and engine (0.5–1 day)**
- uv project, ViZDoom boots headless and windowed, and the scenario configs load.
- `heuristic` and `random` backends plus the `DecisionBackend` protocol.
- ✅ `s1doom play --scenario defend_the_center --backend random` runs one episode and writes JSONL and an `.lmp`.
- ✅ Unit test: bearing sign convention, with synthetic monsters left and right.

**M1: Arena combat with a System One model (1–2 days)**
- Perception from labels, state builder with token budgeting, `mode`/`target`/`in_danger` questions, aim controller.
- Jev backend and Laya (http + inproc) backends, response normalization, contract tests.
- Confidence gating and fallback.
- ✅ Over 20 episodes with seed 1234 on `defend_the_center`, Jev and Laya each reach **≥ 90% of heuristic kills** with fallback rate < 25%. Reference: the heuristic averaged about 6.55 kills in prior art. Measure our own baseline first.
- ✅ `bench-backend` report for at least one Laya host.

**M2: Movement, pickups, corridors (2–3 days)**
- Strafing and retreat macros, pickups, weapon choice, commitment/hysteresis, event triggers, realtime mode.
- ✅ `deadly_corridor`: model backend reaches the vest in ≥ 50% of 20 episodes (heuristic as reference).
- ✅ `health_gathering_supreme`: mean survival ≥ heuristic − 10%.
- ✅ Realtime mode runs at 35 Hz with no frame stalls. Logged staleness distribution.

**M3: Full level autoplay (4–6 days)**
- omgifol WAD parsing, nav graph, specials, dynamic door/lift updates, frontier exploration, keys/doors, exit seeking, stuck recovery. Add the `objective` field to state and the `explore/go_to_exit/open_door/get_key` modes.
- `inspect-map` tool for debugging graphs.
- ✅ Heuristic backend completes Freedoom2 `MAP01` on skill 2 (HNTR) in ≥ 60% of 10 seeds within 10 min of game time.
- ✅ The Jev backend completes it in ≥ 40%. The Laya backend is reported, with no gate on it.
- ✅ If `doom1.wad` is present, run the same eval on `E1M1`. It is skipped cleanly if absent.

**M4: Evaluation and calibration (1–2 days)**
- `eval` matrix (scenarios × backends × seeds), `calibrate` command (temperature scaling per question type and option count, fit on logged runs labeled by outcome and heuristic agreement), with a reliability diagram PNG.
- Ablations: `omniscient` vs visible-only, `rules` string on vs off, decision rate (4/8/16 tics), and with vs without calibration.
- ✅ `runs/report.md` auto-generated with tables and charts.

**M5 (stretch): Improve Laya**
- Distill: gather (state, questions → answer) labels from Jev and/or a hindsight-labeled heuristic. Fine-tune `laya-typed-decisions` using the upstream Kaggle notebook recipe. Re-evaluate.
- Try a vision-to-text front end (a small VLM captioner) as an alternative perception source, to compare against engine state.

---

## 5. Evaluation metrics

- **Game outcome:** kills, damage dealt and taken, survival tics, health/armor at end, items %, secrets, level completed (bool), time-to-exit, deaths.
- **Decision quality:** agreement with the heuristic per question, rate of flip-flopping (switches per minute), fallback rate, confidence distribution, ECE after calibration.
- **Systems:** latency p50/p95/p99 per backend and question count, staleness in realtime mode, Jev input tokens and $/hour.
- **Protocol:** fixed seeds (`game.set_seed`), N ≥ 20 episodes for scenarios and ≥ 10 for full levels, skill level recorded. Report mean ± 95% CI.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Models can't do geometry and stop committing (seen in prior art) | Precompute `aligned`, range buckets and threat; include explicit rules; confidence fallback to the heuristic |
| Laya's 512/1,024-token context | Token-budgeted state builder, list caps, question scheduler; default to the `typed-decisions` checkpoint (1,024) |
| Laya over-confident / weaker on novel tasks | Calibration layer; M5 fine-tune; always report fallback rate |
| Jev latency (≈ 70–500 ms) in realtime play | Default sync `turn` mode for evals. In realtime, the controller keeps executing the committed macro and drops stale answers. |
| Jev rate limits (429) / overload (529) | Backoff with jitter, a circuit breaker to the heuristic, and a budget cap per run |
| ViZDoom lines lack specials | omgifol WAD parsing for exits, doors and keys |
| Navigation is the long pole in M3 | Build and validate the navigator with the heuristic backend first. The model only selects objectives. |
| Hidden omniscience inflates results | Visible-only by default, fairness flag logged, omniscient only as an ablation |
| Jev early-access API/SDK drift | Thin raw-HTTP fallback path; contract tests against recorded fixtures |

---

## 7. Notes for Claude Code (put these in `CLAUDE.md`)

- Python 3.11+, `uv`, type hints everywhere, `pydantic` models for state, questions and answers, `pytest` with no network access in unit tests.
- Build order: **heuristic first, model second.** Every macro and navigator feature must work under `--backend heuristic` before a model backend is wired to it.
- Never hardcode API keys: `TYPESAFE_API_KEY` and `LAYA_URL`/`LAYA_API_KEY` come from env or `.env` (gitignored).
- Integration tests that call Jev or Laya are marked `@pytest.mark.live` and skipped by default.
- Every model decision must be logged. Fallbacks must be labeled `fallback` in logs and on the overlay, never presented as model output.
- Keep prompts and question text in `prompts/` (versioned). Don't inline them in Python.
- Freedoom2 ships with ViZDoom. Look for `doom1.wad` in `./wads/` or `$DOOMWADDIR`, and skip cleanly if it's not found. Never commit WADs.
- Commit per milestone with a short `runs/`-derived result note in the commit message body.
- Before writing code, spot-check these assumptions:
  1. The `laya-serve` `/v1/systemone` response shape vs Jev's. Record fixtures from both.
  2. Current `typesafe-sdk` class and async client names (docs: `https://docs.typesafe.ai/sdk/python/usage.md`).
  3. ViZDoom 1.3.x button and variable names.
  4. Whether TypeSafe has published its promised in-depth DOOM walkthrough, and adopt anything useful.

---

## 8. References

- Jev launch post (DOOM demo: structured state not images, ~10 queries/s, ~$7/hr): https://typesafe.ai/blog/introducing-system-one-models-and-jev
- TypeSafe docs index: https://docs.typesafe.ai/llms.txt · API: https://docs.typesafe.ai/api.md · State: https://docs.typesafe.ai/concepts/state.md
- Laya overview and eval guide: https://huggingface.co/blog/sora-2/laya-ai-model-how-it-works-run-it-locally-and-eval
- Laya repo (API, laya-serve, latency, calibration): https://github.com/NandhaKishorM/laya · Model: https://huggingface.co/convaiinnovations/laya
- ViZDoom 1.3.1: https://pypi.org/project/vizdoom/ · GameState API: https://vizdoom.farama.org/api/python/gameState/
- Prior art: https://github.com/tirukovelamanoj/jev-plays-doom (bearing convention, rule vs intent results) · https://github.com/ethanhinson/jevdoom (tactical choices, JSONL logs) · https://github.com/lukaske/jev-doom-agent (macros + local controller, FALLBACK labeling)
