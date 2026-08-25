"""
Diagnostic, not a fix. Re-runs Arm A ONLY on the specific DS2 events where
it disagreed with the reference tier in the (corrected, chronologically-
replayed) Day 6 run, capturing full justification text -- day6_results.json
only saved the tier label, not the reasoning behind it.

Question this answers: does Arm A ignore the numeric consecutive_abnormal_beats
count entirely, or does it use the count but map it to the wrong label name
(no rule in OUTPUT_INSTRUCTIONS ever states which count/confidence threshold
corresponds to which of the three label words)?

Targets the V-class consecutive run (idx 15504-15756) specifically -- the
clearest, most repeated disagreement pattern in the Day 6 run.

Run (from repo root):
    python diagnostics/diagnose_arm_a_disagreements.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for perception./reasoning.

from perception.perception_agent import PerceptionAgent, replay_selected
from reasoning.baseline_arm import run_baseline_arm_timed
from reasoning.training_targets import urgency_tier_from_event

# Paths below are relative to CWD (repo root), per the "run from repo root" convention
# used throughout this project -- NOT relative to this script's own location.
PERCEPTION_CHECKPOINT = "perception/checkpoints/cnn_lstm.pt"
DS2_PATH = "data/processed/ds2_test.npz"
RESULTS_PATH = "results/arm_a_disagreement_diagnosis.json"

# The known V-class disagreement run from the Day 6 log, plus two S-class
# disagreements for comparison (different class, same "priority vs urgent"
# pattern) -- confirm this list matches your own day6 log if it differs.
TARGET_INDICES = [15504, 15630, 15750, 15752, 15754, 15756, 9535, 10289]


def main():
    print("Loading Perception Agent + DS2, chronological replay for correct state...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    data = np.load(DS2_PATH)
    X, rr, record_ids = data["features"], data["rr_interval_ms"], data["record_ids"]

    replayed = replay_selected(agent, X, rr, record_ids, TARGET_INDICES)
    prepared = {i: event for i, (event, _vector) in replayed.items()}

    del agent
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nRunning Arm A on {len(prepared)} disagreement-case events, capturing full justification...\n")
    results = []
    for idx in TARGET_INDICES:
        health_event = prepared[idx]
        result = run_baseline_arm_timed(health_event, perception_agent=None)
        ref_tier = urgency_tier_from_event(health_event)
        consecutive = health_event["clinical_flags"]["consecutive_abnormal_beats"]
        confidence = health_event["classification"]["confidence"]
        label = health_event["classification"]["label"]

        print(f"{'='*70}")
        print(f"DS2 idx={idx} | true_class={label} confidence={confidence:.3f} "
              f"consecutive_abnormal_beats={consecutive}")
        print(f"Reference tier: {ref_tier} | Arm A said: {result['result']['urgency_tier']}")
        print(f"Arm A's justification: {result['result']['justification']}")
        print(f"Arm A's referenced_guideline_fact: {result['result']['referenced_guideline_fact']}")
        print()

        results.append({
            "idx": idx, "true_class": label, "confidence": confidence,
            "consecutive_abnormal_beats": consecutive, "reference_tier": ref_tier,
            "arm_a_tier": result["result"]["urgency_tier"],
            "arm_a_justification": result["result"]["justification"],
            "arm_a_referenced_fact": result["result"]["referenced_guideline_fact"],
        })

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
