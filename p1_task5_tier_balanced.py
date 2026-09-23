"""Deviation 5 task 5 / Deviation 6: does tier-balanced adapter training close the
accuracy gap? (docs/analysis_plan.md)

Trains the k = 4 adapter with the headline config except balance_by = "tier"
(64 examples, 22 routine / 21 priority / 21 urgent, first qualifying DS1 beats in
chronological order), for seeds 101, 202, 303, and evaluates each on E80 with the
dissertation's Arm B protocol. Compared against the true-class-balanced seeds on the
same events: E2 seeds 101 / 202 (65/80, 63/80) and step 4's B-4 (65/80).

Checkpointed: training resumes mid-run (train_adapter), evaluation after every event.

Run (from repo root):
    python p1_task5_tier_balanced.py
"""
import dataclasses
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ablation_common import prepare_events, run_arm_b_eval, summarize_arm_b
from p1_io import save_json_atomic
from p1_step1_seeded_headline import headline_config, provenance, sha256
from reasoning.adapter_arm import load_trained_adapter
from reasoning.adapter_training import train_adapter
from reasoning.model_loader import load_model

SEEDS = (101, 202, 303)
RESULTS_PATH = Path("results/p1_task5_tier_balanced.json")
BASELINE = {"true_class seed101 (E2)": 65, "true_class seed202 (E2)": 63, "true_class seed101 (step 4 B-4)": 65}


def checkpoint_path(seed):
    return Path(f"reasoning/checkpoints/p1_tier_k4_seed{seed}.pt")


def main():
    state = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {
        "analysis_plan": "docs/analysis_plan.md (Deviations 5, 6)", "provenance": provenance(),
        "train": {}, "eval": {}, "eval_partial": {}}

    for seed in SEEDS:
        path = checkpoint_path(seed)
        if str(seed) in state["train"] and path.exists():
            continue
        cfg = dataclasses.replace(headline_config(seed), balance_by="tier")
        summary = train_adapter(cfg, output_path=path)
        state["train"][str(seed)] = {"checkpoint": str(path), "sha256": sha256(path), "config": dataclasses.asdict(cfg),
                                     "losses": summary["losses"]}
        save_json_atomic(RESULTS_PATH, state)
        torch.cuda.empty_cache()

    events = prepare_events(note=" (E80, task 5)")
    model, processor = load_model()
    for seed in SEEDS:
        if str(seed) in state["eval"]:
            continue
        adapter = load_trained_adapter(str(checkpoint_path(seed)), model)

        def checkpoint(rows, seed=seed):
            state["eval_partial"][str(seed)] = rows
            save_json_atomic(RESULTS_PATH, state)
        rows = run_arm_b_eval(events, model, processor, adapter, label=f"tier seed={seed}",
                              resume_from=state["eval_partial"].get(str(seed)), on_event=checkpoint)
        state["eval"][str(seed)] = {"summary": summarize_arm_b(rows), "per_event": rows,
                                    "by_reference_tier": {t: f"{sum(r['correct'] for r in rows if r['reference_tier'] == t)}"
                                                             f"/{sum(1 for r in rows if r['reference_tier'] == t)}"
                                                          for t in ("routine", "priority", "urgent")}}
        state["eval_partial"].pop(str(seed), None)
        save_json_atomic(RESULTS_PATH, state)
        del adapter
        torch.cuda.empty_cache()

    correct = [state["eval"][str(s)]["summary"]["correct"] for s in SEEDS]
    state["summary"] = {"tier_balanced_correct_by_seed": dict(zip(map(str, SEEDS), correct)),
                        "tier_balanced_accuracy_mean": float(np.mean(correct)) / 80,
                        "true_class_baseline_correct": BASELINE,
                        "by_reference_tier": {str(s): state["eval"][str(s)]["by_reference_tier"] for s in SEEDS}}
    save_json_atomic(RESULTS_PATH, state)
    print(json.dumps(state["summary"], indent=2))


if __name__ == "__main__":
    main()
