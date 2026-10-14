"""Campaign v2 analysis (docs/verification-v2.md): paired same-seed comparisons vs the heuristic,
bootstrap CIs, fallback, latency, and in_danger scored on outcome labels.

    uv run python scripts/analyze_v2.py [out.json]
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
rng = np.random.default_rng(0)
B = 10000
OUTCOME = {"defend_the_center": "kills", "deadly_corridor": "completed", "health_gathering_supreme": "survival_tics",
           "my_way_home": "completed", "freedoom2_map01": "completed"}


def boot_mean(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


runs = {}  # (tag, scen, backend) -> {seed: episode}
cfgs = {}
for d in sorted(glob.glob("runs/2026-10-1*_v2-*")):
    s = json.load(open(f"{d}/summary.json"))
    c = json.load(open(f"{d}/config.json"))
    tag = d.rsplit("_", 1)[1]
    sc, b = c["scenario"]["name"], c["backend"]
    runs[(tag, sc, b)] = {e["seed"]: e for e in s["episodes"]}
    cfgs[(tag, sc, b)] = (c.get("git"), c.get("prompt_version"), c["policy"]["fallback_enabled"], c["opportunistic_fire"])

# v2-nofb-m3b reruns the Laya seeds that hit the --max-calls cap in v2-nofb-m3: they replace them
for (tag, sc, b) in [k for k in runs if k[0] == "v2-nofb-m3b"]:
    runs[("v2-nofb-m3", sc, b)].update(runs.pop((tag, sc, b)))

print("provenance:", sorted({v[:2] for v in cfgs.values()}))

out = []
print(f"\n{'tag':12} {'scenario':25} {'backend':16} {'n':>3} {'outcome mean [95% CI]':>28} {'Δ vs heur (paired) [CI]':>28} "
      f"{'same-as-heur':>12} {'fallbk':>6} {'err':>4} {'p50ms':>6}")
for (tag, sc, b), eps in sorted(runs.items()):
    key = OUTCOME[sc]
    seeds = sorted(eps)
    y = [float(eps[s][key]) for s in seeds]
    m, lo, hi = boot_mean(y)
    htag = tag.replace("nofb-", "")
    h = runs.get((htag, sc, "heuristic"), {})
    paired = [s for s in seeds if s in h]
    if b != "heuristic" and paired:
        dlt = [float(eps[s][key]) - float(h[s][key]) for s in paired]
        dm, dlo, dhi = boot_mean(dlt)
        dstr = f"{dm:+8.2f} [{dlo:+.2f}, {dhi:+.2f}]"
        same = np.mean([all(eps[s][k] == h[s][k] for k in ("kills", "damage_taken", "survival_tics", "completed"))
                        for s in paired])
    else:
        dstr, same = "", np.nan
    fb = np.mean([eps[s]["fallback_rate"] for s in seeds])
    err = sum(eps[s].get("backend_errors", 0) for s in seeds)
    p50 = np.median([eps[s]["latency_p50"] for s in seeds if eps[s].get("latency_p50") is not None] or [np.nan])
    print(f"{tag:12} {sc:25} {b:16} {len(y):3} {m:10.2f} [{lo:7.2f}, {hi:7.2f}] {dstr:>28} {same:12.0%} {fb:6.1%} {err:4} {p50:6.0f}")
    out.append(dict(tag=tag, scenario=sc, backend=b, n=len(y), metric=key, mean=m, ci=[lo, hi], delta=dstr,
                    same_as_heuristic=same, fallback=fb, errors=err, latency_p50=p50,
                    model_calls=sum(eps[s].get("model_calls", 0) for s in seeds)))

# in_danger vs outcome labels, and how closely model probabilities track the heuristic's
print(f"\n{'backend':16} {'n':>7} {'base P(dmg)':>11} {'acc@0.5':>8} {'always-F':>8} {'Brier':>6} {'Brier-heur':>10} {'AUC':>5} "
      f"{'|p-p_heur|':>10} {'mode=heur':>9}")
idg = defaultdict(lambda: {"p": [], "y": [], "ph": [], "mode": []})
for d in glob.glob("runs/2026-10-1*_v2-*"):
    for f in glob.glob(f"{d}/ep_*/calls.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") != "decision":
                continue
            raw, heur, oc = r.get("raw_answers") or {}, r.get("heuristic_answers") or {}, r.get("outcome") or {}
            g = idg[r["backend"]]
            if "mode" in raw and "mode" in heur:
                g["mode"].append(raw["mode"]["value"] == heur["mode"]["value"])
            if "in_danger" in raw and "in_danger" in oc and "in_danger" in heur:
                g["p"].append(float(raw["in_danger"]["value"]))
                g["ph"].append(float(heur["in_danger"]["value"]))
                g["y"].append(bool(oc["in_danger"]))


def auc(p, y):
    p, y = np.asarray(p), np.asarray(y)
    pos, neg = p[y], p[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


idanger = {}
for b, g in sorted(idg.items()):
    p, y, ph = np.array(g["p"]), np.array(g["y"]), np.array(g["ph"])
    if len(p) == 0:
        continue
    row = dict(n=len(p), base=y.mean(), acc=((p >= 0.5) == y).mean(), always_false=1 - y.mean(),
               brier=((p - y) ** 2).mean(), brier_heur=((ph - y) ** 2).mean(), auc=auc(p, y),
               dist_heur=np.abs(p - ph).mean(), mode_agree=np.mean(g["mode"]) if g["mode"] else np.nan)
    idanger[b] = row
    print(f"{b:16} {row['n']:7} {row['base']:11.1%} {row['acc']:8.1%} {row['always_false']:8.1%} {row['brier']:6.3f} "
          f"{row['brier_heur']:10.3f} {row['auc']:5.2f} {row['dist_heur']:10.3f} {row['mode_agree']:9.1%}")

json.dump({"runs": out, "in_danger": idanger}, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"),
          indent=1, default=float)
