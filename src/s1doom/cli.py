"""s1doom CLI: play, eval, bench-backend, calibrate, replay, inspect-map, report."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from s1doom.agent import RUNS_DIR, Agent, RunConfig
from s1doom.backends.factory import make_backend
from s1doom.decision.policy import PolicyConfig

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _episode_line(m) -> None:
    console.print(f"  ep {m.episode} seed {m.seed}: kills={m.kills} dmg={m.damage_taken:.0f} "
                  f"tics={m.survival_tics} dead={m.dead} completed={m.completed} "
                  f"fallback={m.fallback_rate:.0%} decisions={m.decisions}"
                  + (f" explored={m.explored:.0%}" if m.explored is not None else ""))


def _run(cfg: RunConfig) -> tuple[str, list]:
    from s1doom.env.wads import WadNotFound

    backend = make_backend(cfg.backend, url=cfg.url, seed=cfg.seed, replay_path=cfg.replay_path)
    try:
        agent = Agent(cfg, backend)
    except WadNotFound as e:
        console.print(f"[yellow]skip {cfg.scenario}: {e}[/]")
        return "", []
    console.print(f"[bold]{cfg.scenario}[/] backend={backend.name} mode={cfg.mode} run={agent.run_id}")
    try:
        eps = asyncio.run(agent.run(progress=_episode_line))
    finally:
        agent.close()
    return agent.run_id, eps


@app.command()
def play(scenario: str = "defend_the_center", backend: str = "heuristic", mode: str = "turn",
         episodes: int = 1, seed: int = 1234, overlay: bool = False, window: bool = False,
         omniscient: bool = False, no_rules: bool = False, decision_tics: int | None = None,
         calibration: str | None = None, url: str | None = None, skill: int | None = None,
         save_frames: int = 0, max_tics: int | None = None, floor: float = 0.55,
         no_fallback: bool = False, max_calls: int | None = None, no_opportunistic_fire: bool = False):
    """Play episodes of a scenario with a backend."""
    cfg = RunConfig(scenario=scenario, backend=backend, mode=mode, episodes=episodes, seed=seed,
                    overlay=overlay, visible=window, omniscient=omniscient, include_rules=not no_rules,
                    decision_tics=decision_tics, calibration=calibration, url=url, skill=skill,
                    save_frames_every=save_frames, max_tics=max_tics, max_calls=max_calls,
                    opportunistic_fire=not no_opportunistic_fire,
                    policy=PolicyConfig(confidence_floor=floor, fallback_enabled=not no_fallback))
    _run(cfg)


@app.command("eval")
def eval_(scenario: str = typer.Option(..., help="comma list of scenarios"),
          backends: str = "heuristic", episodes: int = 20, seed: int = 1234, mode: str = "turn",
          omniscient: bool = False, no_rules: bool = False, decision_tics: int | None = None,
          calibration: str | None = None, tag: str = "", max_calls: int | None = None,
          no_fallback: bool = False, no_opportunistic_fire: bool = False, skill: int | None = None):
    """Evaluation matrix: scenarios x backends x seeds. Writes runs/eval_<ts>/matrix.json."""
    rows = []
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    for sc in scenario.split(","):
        for b in backends.split(","):
            cfg = RunConfig(scenario=sc, backend=b, mode=mode, episodes=episodes, seed=seed,
                            omniscient=omniscient, include_rules=not no_rules, decision_tics=decision_tics,
                            calibration=calibration, max_calls=max_calls, skill=skill,
                            opportunistic_fire=not no_opportunistic_fire,
                            policy=PolicyConfig(fallback_enabled=not no_fallback),
                            run_id=f"{stamp}_{sc}_{b}{('_' + tag) if tag else ''}")
            try:
                run_id, eps = _run(cfg)
            except Exception as e:  # noqa: BLE001 -- one failing backend must not kill the matrix
                console.print(f"[red]{sc}/{b} failed: {e}[/]")
                rows.append({"scenario": sc, "backend": b, "error": str(e)})
                continue
            if run_id:
                s = json.loads((RUNS_DIR / run_id / "summary.json").read_text())
                rows.append({"scenario": sc, "backend": b, "run_id": run_id, "tag": tag, **s["aggregate"]})
    out = RUNS_DIR / f"eval_{stamp}{('_' + tag) if tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix.json").write_text(json.dumps(rows, indent=2, default=str))
    _print_matrix(rows)
    console.print(f"matrix -> {out / 'matrix.json'}")


def _fmt(v) -> str:
    if isinstance(v, dict) and "mean" in v:
        ci = v["ci95"]
        return f"{v['mean']:.2f}" + (f" ± {ci:.2f}" if ci == ci else "")
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _print_matrix(rows: list[dict]) -> None:
    t = Table("scenario", "backend", "n", "kills", "survival_tics", "completed", "fallback", "p50 ms")
    for r in rows:
        if "error" in r:
            t.add_row(r["scenario"], r["backend"], "-", "error", "", "", "", "")
            continue
        t.add_row(r["scenario"], r["backend"], str(r["n"]), _fmt(r.get("kills")), _fmt(r.get("survival_tics")),
                  f"{r['completed_rate']:.0%}", _fmt(r.get("fallback_rate")), _fmt(r.get("latency_p50")))
    console.print(t)


@app.command("bench-backend")
def bench_backend(backend: str = "laya-inproc", url: str | None = None, states: str | None = None,
                  n: int = 200, qcounts: str = "1,3,6", out: str | None = None):
    """Latency benchmark over recorded states (from runs/**/calls.jsonl) at several question counts."""
    from s1doom.bench import run_bench

    res = asyncio.run(run_bench(backend, url, states, n, [int(x) for x in qcounts.split(",")]))
    t = Table("questions", "n", "p50 ms", "p95 ms", "p99 ms", "errors")
    for q, r in res["by_qcount"].items():
        t.add_row(str(q), str(r["n"]), f"{r['p50']:.1f}", f"{r['p95']:.1f}", f"{r['p99']:.1f}", str(r["errors"]))
    console.print(t)
    path = Path(out) if out else RUNS_DIR / f"bench_{backend}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2))
    console.print(f"-> {path}")


@app.command()
def calibrate(runs: list[str] = typer.Option(..., help="run dirs or globs"),
              out: str = "configs/calibration.yaml", min_n: int = 30):
    """Fit temperature scaling per (backend, qtype, n_options) from logged runs."""
    import yaml

    from s1doom.decision.calibration import collect, fit
    from s1doom.telemetry.report import reliability_png

    groups = collect(runs)
    temps, report = fit(groups, min_n)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(yaml.safe_dump({"temperatures": temps, "fitted_at": time.strftime("%Y-%m-%d"),
                                         "runs": runs}))
    t = Table("group", "n", "T", "acc", "ECE before", "ECE after")
    for k, r in report.items():
        t.add_row(k, str(r["n"]), f"{r['T']:.2f}", f"{r['acc']:.2f}", f"{r['ece_before']:.3f}", f"{r['ece_after']:.3f}")
    console.print(t)
    if report:
        png = Path(out).with_suffix(".reliability.png")
        reliability_png(report, png)
        console.print(f"reliability diagram -> {png}")
    console.print(f"temperatures -> {out}")


@app.command()
def replay(run_dir: str, episode: int = 0):
    """Re-run an episode with its logged answers (deterministic regression check)."""
    cfgj = json.loads((Path(run_dir) / "config.json").read_text())
    calls = Path(run_dir) / f"ep_{episode}" / "calls.jsonl"
    cfg = RunConfig(scenario=cfgj["scenario"]["name"], backend="replay", replay_path=str(calls),
                    seed=cfgj["seed"] + episode, episodes=1, omniscient=cfgj["fairness"] == "omniscient",
                    include_rules=cfgj["include_rules"], decision_tics=cfgj["decision_tics"],
                    skill=cfgj["scenario"].get("skill"),
                    policy=PolicyConfig(**cfgj["policy"]), calibration=cfgj.get("calibration"))
    _, eps = _run(cfg)
    orig = [json.loads(l) for l in calls.read_text().splitlines() if '"episode_end"' in l]
    if orig and eps:
        o, n = orig[-1], eps[0].to_dict()
        same = all(o[k] == n[k] for k in ("kills", "survival_tics", "damage_taken"))
        console.print(f"replay {'MATCHES' if same else 'DIVERGES from'} original: "
                      f"kills {o['kills']}->{n['kills']} tics {o['survival_tics']}->{n['survival_tics']}")


@app.command("inspect-map")
def inspect_map(wad: str = "freedoom2", map: str = "MAP01", out: str | None = None, edges: bool = True):
    """Render nav graph, exits, doors, lifts, teleports and the start->exit path to PNG."""
    import cv2

    from s1doom.env.wads import resolve_wad
    from s1doom.mapmodel.render import MapCanvas, legend
    from s1doom.mapmodel.runtime import LevelModel

    lv = LevelModel(resolve_wad(wad), map, use_cache=False)
    c = MapCanvas(lv.m)
    img = c.base(lv.nav, show_edges=edges)
    legend(img)
    st = lv.m.player_start()
    info = {"nodes": lv.nav.n, "edges": sum(len(a) for a in lv.nav.adj), "dynamic_sectors": len(lv.nav.dynamic),
            "exits": [l.idx for l in lv.exits]}
    g = lv.exit_goal()
    if st and g:
        path = lv.plan(st.x, st.y, g[0], g[1])
        info["start_to_exit"] = f"{lv.path_length(path):.0f} units" if path else "NO PATH"
        if path:
            c.path(img, lv.nav, path)
    reach = lv.nav.dijkstra(lv.node(st.x, st.y)) if st else {}
    info["reachable_nodes"] = len(reach)
    p = Path(out or f"cache/{Path(wad).stem}_{map}.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)
    console.print(info)
    console.print(f"-> {p}")


@app.command()
def report(runs: list[str] = typer.Option(None, help="eval dirs (default: all runs/eval_*)"),
           out: str = "runs/report.md"):
    """Auto-generate runs/report.md with tables and charts from eval matrices."""
    from s1doom.telemetry.report import build_report

    p = build_report(runs, Path(out))
    console.print(f"-> {p}")


@app.command()
def serve_laya(port: int = 8000):
    """Run laya-serve locally (Jev-compatible /v1/systemone)."""
    import os

    os.environ.setdefault("LAYA_PORT", str(port))
    os.environ.setdefault("LAYA_MODELS", "typed-decisions")
    from laya.serve import main

    main()


if __name__ == "__main__":
    app()
