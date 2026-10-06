"""Pilot 2 (Deviation 8): the multi-event pilot with TIER-STRATIFIED windows.

The first pilot drew windows at random positions; ~89% of DS2 beats are normal, so
short windows were almost all "routine" (N = 1: 20/20) and plain accuracy mostly
measured saying "routine". Here, for each N in {5, 10, 20}, 20 windows are drawn for
each reference tier (routine / priority / urgent; window tier = most urgent beat's
reference tier), from DS2 only, at most MAX_PER_RECORD per (N, tier) cell, seed 0.
Same scaffold, task, arms and parsing as p1_pilot_multi_event.py. The first pilot's
random windows remain the natural-prevalence view.

Reported per N and arm: recall per tier, balanced accuracy, parse rate, and the false
alarm rate (routine windows answered priority or urgent). Atomic checkpoint per
generation; resumable.

Run (from repo root):
    python p1_pilot2_stratified.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from p1_io import save_json_atomic
from p1_step1_seeded_headline import provenance, sha256
from p1_step4_baseline_family import B4_CHECKPOINT
from perception.perception_agent import PerceptionAgent, replay_selected
from project_config import DS2_PATH, PERCEPTION_CHECKPOINT
from reasoning.training_targets import urgency_tier_from_event

NS = (5, 10, 20)
TIERS = ("routine", "priority", "urgent")
RANK = {t: i for i, t in enumerate(TIERS)}
PER_CELL = 20
MAX_PER_RECORD = 3
ARMS = ("A-compact", "B-4")
RESULTS_PATH = Path("results/p1_pilot2_stratified.json")


def stratified_windows(tiers, record_ids, rng, ns=NS, per_cell=PER_CELL, max_per_record=MAX_PER_RECORD,
                       classes=None, records=None):
    """{(n, tier): [start index, ...]} with per_cell windows per cell, each inside
    one record, at most max_per_record per record per cell. Defaults reproduce
    pilot 2 exactly. With ``classes`` (the predicted class of every beat), each
    cell is filled round-robin over the class of the window's most urgent beat,
    so a cell is not dominated by one class (used for item 7 training data).
    With ``records``, only windows inside those record ids are candidates (item 7's
    train / validation split by whole records)."""
    rank = np.array([RANK[t] for t in tiers])
    starts = np.flatnonzero(np.r_[True, np.diff(record_ids) != 0])
    ends = np.r_[starts[1:], len(record_ids)]
    cells = {}
    for n in ns:
        by_tier = {t: [] for t in TIERS}
        for s0, e0 in zip(starts, ends):
            if records is not None and int(record_ids[s0]) not in records:
                continue
            for s in range(s0, e0 - n + 1, n):  # non-overlapping candidate windows
                by_tier[TIERS[rank[s:s + n].max()]].append(s)
        for t in TIERS:
            cand = rng.permutation(by_tier[t])
            if classes is not None:
                cand = _round_robin_by_top_class(cand, n, rank, classes)
            chosen, per_rec = [], {}
            for s in cand:
                r = int(record_ids[s])
                if per_rec.get(r, 0) < max_per_record:
                    chosen.append(int(s))
                    per_rec[r] = per_rec.get(r, 0) + 1
                if len(chosen) == per_cell:
                    break
            cells[(n, t)] = chosen
    return cells


def _round_robin_by_top_class(cand, n, rank, classes):
    """Reorder candidate windows so the class of each window's most urgent beat
    cycles (A, B, C, A, B, ...), keeping the random order within each class."""
    groups = {}
    for s in cand:
        top = int(s) + int(np.argmax(rank[s:s + n]))
        groups.setdefault(classes[top], []).append(s)
    queues = [list(g) for _, g in sorted(groups.items())]
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def main():
    # Model-side pieces are imported here so the sampler above stays importable on CPU.
    import p1_pilot_multi_event as p1
    from reasoning.adapter_arm import load_trained_adapter
    from reasoning.baseline_arm import _extract_last_json_object
    from reasoning.model_loader import load_model
    from reasoning.output_schema import ReasoningOutput

    state = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {
        "design": __doc__, "provenance": provenance(),
        "b4_checkpoint": {"path": str(B4_CHECKPOINT), "sha256": sha256(B4_CHECKPOINT)}, "rows": []}
    done = {(r["n"], r["tier_cell"], r["start"], r["arm"]) for r in state["rows"]}

    with np.load(DS2_PATH) as z:
        data = {k: z[k] for k in ("features", "labels", "rr_interval_ms", "record_ids")}
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    replayed = replay_selected(agent, data["features"], data["rr_interval_ms"], data["record_ids"],
                               list(range(len(data["labels"]))))  # every beat: its reference tier is needed
    del agent
    tiers = [urgency_tier_from_event(replayed[i][0]) for i in range(len(data["labels"]))]
    cells = stratified_windows(tiers, data["record_ids"], np.random.default_rng(0))
    state["cells"] = {f"{n}|{t}": v for (n, t), v in cells.items()}
    save_json_atomic(RESULTS_PATH, state)

    model, processor = load_model()
    adapter = load_trained_adapter(str(B4_CHECKPOINT), model)
    generate = p1.make_generator(model, processor, adapter, replayed)

    t0 = time.time()
    with torch.no_grad():
        for (n, tier), window_starts in cells.items():
            for s in window_starts:
                idx = list(range(s, s + n))
                ref = max((tiers[i] for i in idx), key=RANK.get)
                assert ref == tier
                for arm in ARMS:
                    if (n, tier, s, arm) in done:
                        continue
                    text = generate(arm, idx)
                    try:
                        answer, parsed = ReasoningOutput(**_extract_last_json_object(text)).urgency_tier, True
                    except (ValueError, TypeError):
                        answer, parsed = None, False
                    state["rows"].append({"n": n, "tier_cell": tier, "start": s, "arm": arm, "reference": ref,
                                          "tier": answer, "parsed": parsed, "correct": answer == ref,
                                          "text": text[:600]})
                    save_json_atomic(RESULTS_PATH, state)
            print(f"[N={n} {tier}] done, elapsed={time.time() - t0:.0f}s", flush=True)

    state["summary"] = summarize(state["rows"])
    save_json_atomic(RESULTS_PATH, state)
    print(json.dumps(state["summary"], indent=2))


def summarize(rows):
    out = {}
    for n in NS:
        out[str(n)] = {}
        for arm in ARMS:
            a = [r for r in rows if r["n"] == n and r["arm"] == arm]
            recall = {t: float(np.mean([r["correct"] for r in a if r["reference"] == t])) for t in TIERS
                      if any(r["reference"] == t for r in a)}
            routine = [r for r in a if r["reference"] == "routine"]
            out[str(n)][arm] = {"recall": recall, "balanced_accuracy": float(np.mean(list(recall.values()))),
                                "parse_rate": float(np.mean([r["parsed"] for r in a])),
                                "false_alarm_rate": float(np.mean([r["tier"] in ("priority", "urgent")
                                                                   for r in routine])) if routine else None}
    return out


if __name__ == "__main__":
    main()
