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
        # Nothing saved until the first seed finishes training: report the live epoch instead.
        log = Path("logs/p1_task5.log")
        if not log.exists():
            return None
        tail = log.read_text(encoding="utf-8", errors="replace")[-4000:].replace("\r", "\n")
        steps = [ln.strip() for ln in tail.splitlines() if ln.strip().startswith("Epoch")]
        last = steps[-1].encode("ascii", "ignore").decode() if steps else "starting"  # drop tqdm bar glyphs
        return f"Task 5 training seed 101 (first of 3): {last[:70]}"
    trained = list(s.get("train", {}))
    done = {k: v["summary"]["correct"] for k, v in s.get("eval", {}).items()}
    partial = {k: f"{sum(r['correct'] for r in v)}/{len(v)}" for k, v in s.get("eval_partial", {}).items()}
    live = ""
    if len(trained) < 3:  # a seed is still training: show its live step from the log
        tail = Path("logs/p1_task5.log").read_text(encoding="utf-8", errors="replace")[-4000:].replace("\r", "\n")
        steps = [ln.strip() for ln in tail.splitlines() if ln.strip().startswith("Epoch")]
        if steps:
            live = f" | now training seed #{len(trained) + 1}: " + steps[-1].encode("ascii", "ignore").decode()[:60]
    return (f"Task 5 trained seeds {trained or 'none yet'}{live} | finished evals (correct/80): {done or '-'}"
            f" | in-progress eval: {partial or '-'} | true-class baseline: 65, 63 /80")


def pilot():
    s = load("results/p1_pilot_multi_event.json")
    if not s or not s.get("rows"):
        return None
    parts = []
    for n in sorted({r["n"] for r in s["rows"]}):
        cell = []
        for arm in ("A-compact", "B-4"):
            a = [r for r in s["rows"] if r["n"] == n and r["arm"] == arm]
            if a:
                cell.append(f"{arm} {sum(r['correct'] for r in a)}/{len(a)} (parsed {sum(r['parsed'] for r in a)})")
        refs = [r["reference"] for r in s["rows"] if r["n"] == n and r["arm"] == "A-compact"]
        parts.append(f"N={n}: " + ", ".join(cell) + f", always-routine {sum(x == 'routine' for x in refs)}/{len(refs)}")
    return f"Pilot {len({r['window'] for r in s['rows']})}/20 windows | correct: " + "; ".join(parts)


def pilot2():
    s = load("results/p1_pilot2_stratified.json")
    if not s or not s.get("rows"):
        return "Pilot 2: replaying DS2 / loading model" if s else None
    parts = []
    for n in sorted({r["n"] for r in s["rows"]}):
        cells = []
        for arm in ("A-compact", "B-4"):
            a = [r for r in s["rows"] if r["n"] == n and r["arm"] == arm]
            if a:
                per = ", ".join(f"{t[0].upper()} {sum(r['correct'] for r in a if r['reference'] == t)}/"
                                f"{sum(r['reference'] == t for r in a)}" for t in ("routine", "priority", "urgent")
                                if any(r["reference"] == t for r in a))
                cells.append(f"{arm} [{per}] parsed {sum(r['parsed'] for r in a)}/{len(a)}")
        parts.append(f"N={n}: " + "; ".join(cells))
    return f"Pilot 2 {len(s['rows'])}/360 generations | " + " || ".join(parts)


def item7_baseline():
    s = load("results/p1_item7_baseline.json")
    if not s or not s.get("rows"):
        return "Item 7 stage 0: RR replay / loading model" if s else None
    rows, total = s["rows"], len(s.get("windows", []))
    parts = []
    for n in (1, 5, 10, 20):
        a = [r for r in rows if r["set"] == "stratified" and r["n"] == n]
        if a:
            per = ", ".join(f"{t[0].upper()} {sum(r['correct'] for r in a if r['reference'] == t)}/"
                            f"{sum(r['reference'] == t for r in a)}" for t in ("routine", "priority", "urgent")
                            if any(r["reference"] == t for r in a))
            parts.append(f"N={n} [{per}]")
    nat = [r for r in rows if r["set"] == "natural"]
    return (f"Item 7 stage 0 (A-compact, RR encoder) {len(rows)}/{total} | " + "; ".join(parts)
            + (f" | natural windows done {len(nat)}" if nat else ""))


def item7_train(tag="seed101"):
    s = load(f"results/p1_item7_train_{tag}.json")
    if not s:
        return f"Item 7 {tag}: loading / replaying DS1" if Path(f"logs/p1_item7_{tag}.log").exists() else None
    hist = s.get("history", [])
    vals = [h for h in hist if "val" in h]
    last = hist[-1] if hist else {}
    v = (f" | last val @update {vals[-1]['update']}{' (64-token, truncated)' if vals[-1]['val'].get('truncated_max_new_tokens_64') else ''}: bal-acc {vals[-1]['val']['balanced_accuracy']:.2f}, "
         f"parse {vals[-1]['val']['parse_rate']:.2f}, by N {vals[-1]['val']['by_n']}") if vals else ""
    best = f" | best bal-acc {s['best_metric']:.2f} @update {s['best_update']}" if s.get("best_update") else ""
    return (f"Item 7 {tag}: epoch {s.get('epoch', 0) + 1}, update {s.get('updates', 0)}, "
            f"loss {last.get('loss', float('nan')):.3f}{v}{best}")


RUNNING = {"p1_item7_train": item7_train, "p1_item7_baseline": item7_baseline, "p1_pilot2": pilot2, "p1_task5": task5, "p1_pilot": pilot, "p1_e2": e2, "p1_e1": e1, "p1_e3": e3}

if __name__ == "__main__":
    import sys
    # With a job name (the running script's prefix) print that job; otherwise the latest with data.
    fns = [RUNNING[sys.argv[1]]] if len(sys.argv) > 1 and sys.argv[1] in RUNNING else (item7_train, item7_baseline, pilot2, task5, pilot, e2, e1, e3)
    for fn in fns:
        line = fn()
        if line:
            print(line)
            break
