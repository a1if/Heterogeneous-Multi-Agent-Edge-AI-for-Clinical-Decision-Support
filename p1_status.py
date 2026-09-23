"""One-screen interim results for the running Phase 1 GPU queue, read from the
checkpoint files (never from the running process). Used by the 10-minute monitor.

    python p1_status.py
"""
import json
from pathlib import Path

import numpy as np


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def e3():
    s = load("results/p1_e3_multi_event.json")
    if not s:
        return None
    from p1_e3_multi_event import summarize
    S = summarize(s)
    wins = len({r["window"] for r in s["rows"]})
    parts = []
    for n, g in S["gate"].items():
        parts.append(f"N={n}: {g['b4_vs_acompact_prefill_rel_pct']:+.0f}% [{g['ci95'][0]:+.0f},{g['ci95'][1]:+.0f}]"
                     f"{' PASS' if g['passes'] else ''}")
    return f"E3 {wins}/20 windows | B-4 vs A-compact prefill: " + "; ".join(parts)


def e1():
    s = load("results/p1_e1_cached_scaffold.json")
    if not s or not s.get("rows"):
        return None
    arms = {}
    for r in s["rows"]:
        arms.setdefault(r["arm"], []).append(r["incremental_ms"])
    n_events = len({r["idx"] for r in s["rows"]})
    return (f"E1 {n_events}/80 events | cached-scaffold incremental prefill (median ms): "
            + ", ".join(f"{a} {np.median(v):.1f}" for a, v in arms.items()))


def e2():
    s = load("results/p1_e2_batched.json")
    if not s or not s.get("rows"):
        return None
    out = {}
    for r in s["rows"]:
        out.setdefault(r["batch"], {}).setdefault(r["arm"], []).append("OOM" if r["oom"] else r["events_per_s"])
    fmt = lambda v: "OOM" if "OOM" in v else f"{np.median(v):.2f}"
    return "E2 events/s by batch: " + "; ".join(
        f"b={b}: " + ", ".join(f"{a} {fmt(v)}" for a, v in arms.items()) for b, arms in sorted(out.items()))


def task5():
    s = load("results/p1_task5_tier_balanced.json")
    if not s:
        return None
    trained = list(s.get("train", {}))
    done = {k: v["summary"]["correct"] for k, v in s.get("eval", {}).items()}
    partial = {k: f"{sum(r['correct'] for r in v)}/{len(v)}" for k, v in s.get("eval_partial", {}).items()}
    return (f"Task 5 trained seeds {trained or 'none yet'} | finished evals (correct/80): {done or '-'}"
            f" | in-progress eval: {partial or '-'} | true-class baseline: 65, 63 /80")


if __name__ == "__main__":
    for fn in (task5, e2, e1, e3):
        line = fn()
        if line:
            print(line)
            break
