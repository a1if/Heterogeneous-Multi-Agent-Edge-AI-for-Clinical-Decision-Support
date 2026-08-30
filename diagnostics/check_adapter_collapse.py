"""
Quick representation-collapse check for Arm B — NOT the full Day 6 harness.

Question this answers: does Arm B's output vary AT ALL with the input event,
or does it produce the same (or near-same) urgency_tier/justification
regardless of what the Perception Agent actually saw? This is a fast,
small-N diagnostic to catch strong collapse before committing GPU time to
the full 60-event formal comparison.

Uses 3 held-out DS2 events per class (12 total) -- all DS2, all untouched by
Day 4/5 training (which only ever used DS1). Deterministic first-occurrence
selection, same pattern used throughout this project.

Run (from repo root):
    python diagnostics/check_adapter_collapse.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for perception./reasoning.

from perception.perception_agent import PerceptionAgent, replay_selected
from reasoning.adapter_arm import load_trained_adapter, run_adapter_arm_timed
from reasoning.model_loader import load_model
from reasoning.training_targets import urgency_tier_from_event
from project_config import DS2_PATH, PERCEPTION_CHECKPOINT
from perception.model import AAMI_CLASSES  # single source of truth

DEFAULT_CHECKPOINT = "reasoning/checkpoints/virtual_adapter_day5_larger.pt"
PER_CLASS = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint

    print(f"Using checkpoint: {checkpoint_path}")

    print("Loading Perception Agent + DS2...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    data = np.load(DS2_PATH)
    X, y, rr, record_ids = (
        data["features"], data["labels"], data["rr_interval_ms"], data["record_ids"]
    )

    # Pick PER_CLASS events per class (N/S/V/F -- Q absent in DS2)
    selected = []
    for class_id in range(4):
        indices = np.flatnonzero(y == class_id)[:PER_CLASS]
        selected.extend(int(i) for i in indices)
    print(f"Selected {len(selected)} DS2 events ({PER_CLASS} per class, N/S/V/F)")

    # Build all Perception outputs BEFORE loading Gemma (same pattern as
    # build_real_training_examples -- avoid VRAM contention).
    #
    # Chronological per-record replay, NOT a direct predict() on the scattered
    # selected indices. consecutive_abnormal_beats is stateful and resets only at
    # record boundaries, so predicting straight down a hand-picked index list makes
    # the counter accumulate across unrelated records -- exactly what
    # day6_run_comparison.py's docstring warns gives "arbitrary, incorrect state".
    # This script used to do that, and it showed: idx=1905 reported
    # consecutive_abnormal_beats=2 inherited from idx=257, a different record ~1,650
    # beats earlier. That corrupted requires_urgent_review, hence reference_tier --
    # the very thing the agreement count below is scored against.
    prepared = []
    replayed = replay_selected(agent, X, rr, record_ids, selected)
    for idx in selected:
        health_event, context_vector = replayed[idx]
        prepared.append({
            "idx": idx,
            "true_class": AAMI_CLASSES[y[idx]],
            "predicted_class": health_event["classification"]["label"],
            "health_event": health_event,
            "context_vector": context_vector,
            "reference_tier": urgency_tier_from_event(health_event),
        })

    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading Gemma 4 E4B + trained adapter...")
    model, processor = load_model()
    adapter = load_trained_adapter(checkpoint_path, model)

    print(f"\nRunning {len(prepared)} events through Arm B...\n")
    results = []
    for item in prepared:
        result = run_adapter_arm_timed(
            item["health_event"], item["context_vector"], model, processor, adapter
        )
        results.append({**item, "output": result["result"]})
        print(f"DS2 idx={item['idx']:>6} true={item['true_class']} "
              f"perception_pred={item['predicted_class']} "
              f"reference_tier={item['reference_tier']:>8} "
              f"-> adapter_tier={result['result']['urgency_tier']:>8}")
        print(f"   justification: {result['result']['justification']}")

    print(f"\n{'='*70}")
    print("COLLAPSE CHECK")
    print(f"{'='*70}")
    tiers_output = [r["output"]["urgency_tier"] for r in results]
    justifications = [r["output"]["justification"] for r in results]
    unique_tiers = set(tiers_output)
    unique_justifications = set(justifications)

    print(f"Unique urgency_tier values produced: {unique_tiers} "
          f"({len(unique_tiers)} distinct out of {len(results)} events)")
    print(f"Unique justification strings produced: {len(unique_justifications)} "
          f"distinct out of {len(results)} events")

    agreement_with_reference = sum(
        1 for r in results if r["output"]["urgency_tier"] == r["reference_tier"]
    )
    print(f"Agreement with deterministic reference tier: {agreement_with_reference}/{len(results)}")

    if len(unique_tiers) == 1:
        print("\n*** STRONG COLLAPSE SIGNAL: every event produced the SAME urgency_tier, "
              "regardless of true class. The adapter is very likely ignoring the virtual "
              "tokens' content entirely (or the tokens carry no separable signal the LLM "
              "can use), consistent with the memorization-not-generalization concern "
              "flagged after the Day 5 training-subset audit. ***")
    elif len(unique_tiers) < 3:
        print(f"\n*** PARTIAL SIGNAL: only {len(unique_tiers)}/3 possible tiers appear across "
              "held-out events. Not full collapse, but limited discrimination -- worth "
              "reporting honestly rather than rounding up. ***")
    else:
        print("\nAll three tiers appear across held-out events -- not collapsed, though "
              "this says nothing yet about whether tier assignments are CORRECT. "
              "The full 60-event Day 6 run will measure actual accuracy.")

    results_path = f"reasoning/checkpoints/collapse_check_results{args.output_suffix}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to {results_path}")


if __name__ == "__main__":
    main()
