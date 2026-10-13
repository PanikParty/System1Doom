"""runs/report.md: tables + charts from eval matrices, benches and calibration."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"

# Fixed entity -> colour (validated categorical slots 1-4; colour follows the backend, never rank).
BACKEND_COLORS = {"heuristic": "#2a78d6", "random": "#eb6834", "jev": "#1baf7a", "laya-inproc": "#eda100",
                  "laya-http": "#eda100", "semif": "#e87ba4", "clef-flash": "#4a3aa7",
                  "clef": "#9a8ae0"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _bar_chart(rows: list[dict], metric: str, title: str, ylabel: str, out: Path, pct: bool = False) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scen = sorted({r["scenario"] for r in rows if metric in r or metric == "completed_rate"})
    backs = [b for b in BACKEND_COLORS if any(r["backend"] == b for r in rows)]
    backs += sorted({r["backend"] for r in rows} - set(backs))
    if not scen or not backs:
        return False
    fig, ax = plt.subplots(figsize=(min(10, 2.2 + 1.4 * len(scen) * max(1, len(backs)) / 2), 3.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    w = 0.8 / len(backs)
    for bi, b in enumerate(backs):
        xs, ys, es = [], [], []
        for si, s in enumerate(scen):
            r = next((r for r in rows if r["scenario"] == s and r["backend"] == b and "error" not in r), None)
            if r is None:
                continue
            v = r.get(metric)
            mu, ci = (v["mean"], v["ci95"]) if isinstance(v, dict) else (v, float("nan"))
            if mu is None:
                continue
            xs.append(si + (bi - (len(backs) - 1) / 2) * w)
            ys.append(mu * (100 if pct else 1))
            es.append(0 if (ci is None or math.isnan(ci)) else ci * (100 if pct else 1))
        col = BACKEND_COLORS.get(b, "#4a3aa7")
        bars = ax.bar(xs, ys, width=w - 0.02, color=col, edgecolor=SURFACE, linewidth=2, label=b)
        if any(e > 0 for e in es):
            ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        for x, y, e in zip(xs, ys, es):
            ax.annotate(f"{y:.0f}%" if pct else f"{y:.1f}", (x, y + e), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=INK)
        del bars
    if len(backs) == 1:
        title = f"{title}: {backs[0]}"
    ax.margins(y=0.12)
    ax.set_xticks(range(len(scen)))
    ax.set_xticklabels(scen)
    _style(ax, title, ylabel)
    if len(backs) >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=len(backs), loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return True


def reliability_png(report: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = list(report.items())[:6]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3.2), dpi=150, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (k, r) in zip(axes[0], groups):
        corr = np.array(r["correct"])
        edges = np.linspace(0, 1, 11)
        for key, col, lab in (("conf_before", "#eb6834", "raw"), ("conf_after", "#2a78d6", f"T={r['T']:.2f}")):
            conf = np.array(r[key])
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf > lo) & (conf <= hi)
                if m.sum() >= 3:
                    xs.append(conf[m].mean())
                    ys.append(corr[m].mean())
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=2, color=col, label=lab)
        ax.plot([0, 1], [0, 1], linestyle="--", color=GRID, linewidth=1)
        _style(ax, k.replace("|", " · "), "accuracy")
        ax.set_xlabel("confidence", color=INK2, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=7, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _fmt(v, pct=False) -> str:
    if v is None:
        return "–"
    if isinstance(v, dict):
        mu, ci = v.get("mean"), v.get("ci95")
        if mu is None:
            return "–"
        s = f"{mu * 100:.0f}%" if pct else f"{mu:.2f}"
        if ci is not None and not math.isnan(ci):
            s += f" ± {ci * 100:.0f}" if pct else f" ± {ci:.2f}"
        return s
    if isinstance(v, float):
        return f"{v * 100:.0f}%" if pct else f"{v:.2f}"
    return str(v)


def build_report(dirs: list[str] | None, out: Path) -> Path:
    evals = [Path(d) for d in dirs] if dirs else sorted(RUNS_DIR.glob("eval_*"))
    rows: list[dict] = []
    for d in evals:
        f = d / "matrix.json"
        if f.exists():
            for r in json.loads(f.read_text()):
                r["_eval"] = d.name
                rows.append(r)
    # latest result per (scenario, backend, tag)
    latest: dict = {}
    for r in rows:
        latest[(r["scenario"], r["backend"], r.get("tag", ""))] = r
    rows = list(latest.values())
    out.parent.mkdir(parents=True, exist_ok=True)
    fig_dir = out.parent / "report_figs"
    fig_dir.mkdir(exist_ok=True)
    L = [f"# s1doom evaluation report", "", f"Generated {time.strftime('%Y-%m-%d %H:%M')} from "
         f"{len(evals)} eval matrices. Mean ± 95% CI over episodes. Turn mode unless tagged.", ""]
    def tag_order(t: str) -> tuple:
        for i, pre in enumerate(("m1", "m2", "m3", "nofallback", "ablation")):
            if t.startswith(pre):
                return (i, t)
        return (9, t)

    tags = sorted({r.get("tag", "") for r in rows}, key=tag_order)
    for tag in tags:
        tr = [r for r in rows if r.get("tag", "") == tag]
        L += [f"## {tag or 'untagged'}", ""]
        L += ["| scenario | backend | n | kills | survival tics | completed | fallback | mode agree | p50 ms | p95 ms | Jev tokens/ep |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(tr, key=lambda r: (r["scenario"], r["backend"])):
            if "error" in r:
                L.append(f"| {r['scenario']} | {r['backend']} | – | error: {r['error'][:60]} |||||||||")
                continue
            ag = r.get("agreement", {}).get("mode")
            L.append(f"| {r['scenario']} | {r['backend']} | {r['n']} | {_fmt(r.get('kills'))} | "
                     f"{_fmt(r.get('survival_tics'))} | {r['completed_rate'] * 100:.0f}% | "
                     f"{_fmt(r.get('fallback_rate'), pct=True)} | {'–' if ag is None else f'{ag * 100:.0f}%'} | "
                     f"{_fmt(r.get('latency_p50'))} | {_fmt(r.get('latency_p95'))} | {_fmt(r.get('input_tokens'))} |")
        L.append("")
        safe = (tag or "untagged").replace("/", "_")
        for metric, title, ylabel, pct in (("kills", "Kills per episode", "kills", False),
                                           ("completed_rate", "Episodes completed", "% of episodes", True)):
            png = fig_dir / f"{safe}_{metric}.png"
            if _bar_chart(tr, metric, title, ylabel, png, pct):
                L += [f"![{title} ({tag})](report_figs/{png.name})", ""]
    benches = sorted(RUNS_DIR.glob("bench_*.json"))
    if benches:
        L += ["## Backend latency (bench-backend)", "", "| bench | backend | questions | n | p50 ms | p95 ms | p99 ms |",
              "|---|---|---|---|---|---|---|"]
        for b in benches:
            d = json.loads(b.read_text())
            for q, r in d["by_qcount"].items():
                L.append(f"| {b.stem} | {d['backend']} | {q} | {r['n']} | {r['p50']:.0f} | {r['p95']:.0f} | {r['p99']:.0f} |")
        L.append("")
    cal = Path("configs/calibration.yaml")
    if cal.exists():
        L += ["## Calibration", "", f"Temperatures from `{cal}`:", "", "```yaml", cal.read_text().strip(), "```", ""]
        rp = cal.with_suffix(".reliability.png")
        if rp.exists():
            import shutil
            shutil.copy(rp, fig_dir / rp.name)
            L += [f"![reliability](report_figs/{rp.name})", ""]
    out.write_text("\n".join(L))
    return out
