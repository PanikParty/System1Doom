# s1doom: conventions for Claude Code

System One models (TypeSafe Jev, open-weight Laya) play DOOM via ViZDoom. The design is in PLAN.md, and the status and results are in README.md.

## Rules
- Python 3.11+ managed with `uv`, type hints everywhere, pydantic for answers and results. `pytest` unit tests never touch the network.
- **Build the heuristic first, the model second.** Every macro and navigator feature must work under `--backend heuristic` before a model backend is wired to it.
- Never hardcode API keys. `TYPESAFE_API_KEY` (or the key file `./Jev.key`, overridable with `TYPESAFE_API_KEY_FILE`) and `LAYA_URL`/`LAYA_API_KEY` come from env or files that are gitignored (`*.key`, `.env`). Never print or log a key.
- Tests that call Jev or Laya are marked `@pytest.mark.live` and skipped by default. Run them with `uv run --extra laya pytest -m live`.
- Every model decision is logged to `runs/<id>/ep_<n>/calls.jsonl`. Fallbacks are labeled `fallback` in the logs and on the overlay, and are never presented as model output.
- Prompts and question text live in `prompts/<version>/questions.yaml` and are never inlined in Python. Every log row records `prompt_version`.
- WADs are never committed. `doom1.wad` is looked up in `./wads/` or `$DOOMWADDIR`; if it's missing the run is skipped cleanly.
- Commit once per milestone, with a short result note derived from `runs/` in the commit body.

## Verified engine facts (ViZDoom 1.3.1, checked 2026-09-27; don't re-derive)
- `TURN_LEFT_RIGHT_DELTA` > 0 turns **right** (ANGLE decreases). Its unit is degrees per tic. Our bearing convention is 0 = ahead, **positive = left**, so `turn_cmd = -bearing`.
- `MOVE_FORWARD_BACKWARD_DELTA` saturates at 50 (running speed), and positive is forward. `MOVE_LEFT_RIGHT_DELTA` > 0 strafes right.
- Input is ignored for the first few tics of a full-map episode.
- Labels have `object_category` ('Monster', 'Self', …). The player's own label must be filtered out. Killed monsters vanish from the labels.
- `state.sectors[i]` index equals the WAD sector index, so live door and lift heights map straight onto the parsed map.
- ViZDoom's scenario WADs are **UDMF** (TEXTMAP), which omgifol can't read. `wadparse._load_udmf` handles them.
- `make_action` blocks holding the GIL. In realtime mode the loop sleeps until the tic is due so the decision thread can run.

## Nav graph gotchas (mapmodel/navgraph.py)
- The grid is offset off-integer (`GRID_OFFSET`) so nodes never sit on linedefs; before this fix, missed crossings let the graph route through ledges.
- Floor and ceiling are computed Doom-style: the player's 32-unit box stands on the **highest** floor and under the **lowest** ceiling of every sector it touches. Thin cracks are walkable, as in Freedoom2 MAP01's exit room.
- Edges are directed: step up ≤ 24, drop down any height.
- Doors and lifts are planned with best-case heights and gated at runtime. A remote door is passable only when its switch is within `TRIGGER_REACH`.
- Bump `VERSION` whenever graph semantics change; the cache lives in `cache/*.navgraph.pkl`.

## Backend facts
- Jev noul answers carry no `confidence`. Laya's choice/score `confidence` is entropy-based. The gate always uses our own `max_prob` after calibration.
- Laya encodes the state once per question (input_tokens ≈ n_questions × (state + question)). On Apple MPS that costs ~145 ms per question. Jev is ~120 ms flat.

## Common commands
```bash
uv run pytest -q                                       # unit + contract + engine smoke
uv run s1doom play --scenario freedoom2_map01 --backend heuristic --overlay --window
uv run s1doom eval --scenario defend_the_center --backends heuristic,jev,laya-inproc --episodes 20 --tag m1
uv run s1doom inspect-map --wad freedoom2 --map MAP01
uv run s1doom report
```
