"""Item 7 stage 0 (Deviation 9): the A-compact baseline on RR-encoder events.

Reference tiers come from the rule applied to the RR encoder's own events, so pilot
2's windows are not reused. On DS2:
  stratified  N in {1, 5, 10, 20} x tier, 20 windows per cell, <= 3 per record per
              cell, seed 0 (p1_pilot2_stratified.stratified_windows)
  natural     pilot-1-style random windows (p1_e3_multi_event.windows, seed 0), first
              N beats for N in {5, 10, 20}: the false-alarm view
A-compact only, with pilots 1-2's scaffold, task and parsing
(p1_pilot_multi_event.make_generator). The window lists are saved so the MEA
evaluation (p1_item7_eval.py) runs on exactly the same windows. Atomic checkpoint
per generation; resumable.

Run (from repo root):
    python p1_item7_baseline.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

import p1_pilot_multi_event as p1
from p1_e3_multi_event import windows as natural_windows
from p1_io import save_json_atomic
from p1_item7_common import RR_ENCODER, replay_split
from p1_pilot2_stratified import RANK, TIERS, stratified_windows
from p1_step1_seeded_headline import provenance, sha256
from reasoning.baseline_arm import _extract_last_json_object
from reasoning.model_loader import load_model
from reasoning.output_schema import ReasoningOutput

NS_STRAT = (1, 5, 10, 20)
NS_NATURAL = (5, 10, 20)
RESULTS_PATH = Path("results/p1_item7_baseline.json")


class Replayed:
    """Adapter for p1.make_generator, which indexes replayed[i] -> (event, vector)."""

    def __init__(self, r):
        self.r = r

    def __getitem__(self, i):
        return self.r["events"][i], self.r["vectors"][i]


def build_windows(r):
    cells = stratified_windows(r["tiers"], r["record_ids"], np.random.default_rng(0), ns=NS_STRAT, per_cell=20)
    nat = natural_windows({"record_ids": r["record_ids"]}, np.random.default_rng(0))
    windows = [{"set": "stratified", "n": n, "tier_cell": t, "start": s} for (n, t), ss in cells.items() for s in ss]
    windows += [{"set": "natural", "n": n, "tier_cell": None, "start": w[0]} for w in nat for n in NS_NATURAL]
    for w in windows:
        w["reference"] = max((r["tiers"][i] for i in range(w["start"], w["start"] + w["n"])), key=RANK.get)
    return windows


def main():
    r = replay_split("ds2")
    state = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {
        "analysis_plan": "docs/analysis_plan.md (Deviation 9, stage 0)", "provenance": provenance(),
        "encoder": {"path": RR_ENCODER, "sha256": sha256(Path(RR_ENCODER))}, "rows": []}
    state["windows"] = build_windows(r)
    save_json_atomic(RESULTS_PATH, state)
    done = {(w["set"], w["n"], w["start"]) for w in state["rows"]}

    model, processor = load_model()
    generate = p1.make_generator(model, processor, None, Replayed(r))
    t0 = time.time()
    with torch.no_grad():
        for k, w in enumerate(state["windows"]):
            if (w["set"], w["n"], w["start"]) in done:
                continue
            text = generate("A-compact", list(range(w["start"], w["start"] + w["n"])))
            try:
                answer, parsed = ReasoningOutput(**_extract_last_json_object(text)).urgency_tier, True
            except (ValueError, TypeError):
                answer, parsed = None, False
            state["rows"].append({**w, "arm": "A-compact", "tier": answer, "parsed": parsed,
                                  "correct": answer == w["reference"], "text": text[:600]})
            save_json_atomic(RESULTS_PATH, state)
            if (k + 1) % 20 == 0:
                print(f"[{k+1}/{len(state['windows'])}] elapsed={time.time() - t0:.0f}s", flush=True)

    state["summary"] = summarize(state["rows"])
    save_json_atomic(RESULTS_PATH, state)
    print(json.dumps(state["summary"], indent=2))


def summarize(rows, arm="A-compact"):
    out = {"stratified": {}, "natural": {}}
    for n in NS_STRAT:
        a = [x for x in rows if x["set"] == "stratified" and x["n"] == n and x["arm"] == arm]
        recall = {t: float(np.mean([x["correct"] for x in a if x["reference"] == t])) for t in TIERS
                  if any(x["reference"] == t for x in a)}
        out["stratified"][str(n)] = {"recall": recall, "balanced_accuracy": float(np.mean(list(recall.values()))),
                                     "parse_rate": float(np.mean([x["parsed"] for x in a])), "n_windows": len(a)}
    for n in NS_NATURAL:
        a = [x for x in rows if x["set"] == "natural" and x["n"] == n and x["arm"] == arm]
        routine = [x for x in a if x["reference"] == "routine"]
        out["natural"][str(n)] = {"accuracy": float(np.mean([x["correct"] for x in a])),
                                  "false_alarm_rate": float(np.mean([x["tier"] in ("priority", "urgent")
                                                                     for x in routine])) if routine else None,
                                  "reference_counts": {t: sum(x["reference"] == t for x in a) for t in TIERS}}
    return out


if __name__ == "__main__":
    main()
