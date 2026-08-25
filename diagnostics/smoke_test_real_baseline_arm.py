"""
Real end-to-end smoke test — NOT pytest. Uses the actual trained
PerceptionAgent checkpoint and real MIT-BIH DS2 events (not dummy/synthetic
data) through the full Arm A pipeline: raw ECG -> Perception -> JSON ->
prompt -> Gemma -> ReasoningOutput.

This is the real Day 2 gate: "valid structured output on a handful of real
events; prompt/weights locked and not touched again" (design doc Section 6).

Run (from repo root):
    python diagnostics/smoke_test_real_baseline_arm.py

Picks one event per available AAMI class from DS2 (deterministic, documented
selection — not data[:5]), so you see how the pipeline behaves across the
different clinical categories, not just one lucky/unlucky case.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for perception./reasoning.

from perception.perception_agent import PerceptionAgent
from reasoning.baseline_arm import run_baseline_arm_timed

CHECKPOINT_PATH = "perception/checkpoints/cnn_lstm.pt"
DS2_PATH = "data/processed/ds2_test.npz"
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]


def pick_one_event_per_class(X, y, rr, max_classes=5):
    """Deterministic selection: first available example of each class
    present in DS2. Documented here rather than buried in a fixture —
    this is also the exact selection used to resolve the
    'small_real_event_sample' fixture in tests/test_day2_baseline_arm.py."""
    picked = {}
    for i in range(len(y)):
        label_int = y[i]
        if label_int not in picked:
            picked[label_int] = i
        if len(picked) == max_classes:
            break
    return picked


def main():
    print("Loading trained PerceptionAgent checkpoint...")
    agent = PerceptionAgent(checkpoint_path=CHECKPOINT_PATH)

    print("Loading DS2 (real, held-out test set)...")
    data = np.load(DS2_PATH)
    X, y, rr = data["features"], data["labels"], data["rr_interval_ms"]

    picked = pick_one_event_per_class(X, y, rr)
    print(f"Selected {len(picked)} events, one per available AAMI class: "
          f"{[AAMI_CLASSES[c] for c in picked]}\n")

    for label_int, idx in picked.items():
        true_label = AAMI_CLASSES[label_int]
        print(f"{'='*60}")
        print(f"Event index {idx}, TRUE label: {true_label}")
        print(f"{'='*60}")

        # Call predict() directly first (with the real rr value) so we can print
        # the Perception classification for comparison against the true label,
        # then feed the resulting HealthEventJSON into run_baseline_arm_timed
        # via its perception_agent=None path (accepts an already-built event).
        health_event = agent.predict(X[idx], rr_interval_ms=float(rr[idx]), event_seq=idx)
        print(f"Perception output: label={health_event['classification']['label']} "
              f"(true={true_label}) confidence={health_event['classification']['confidence']:.3f} "
              f"SQI={health_event['segment_metadata']['signal_quality_index']:.3f}")

        result = run_baseline_arm_timed(health_event, perception_agent=None)

        print(f"\nReasoning output:")
        print(json.dumps(result["result"], indent=2))
        print(f"prompt_tokens={result['prompt_tokens']} "
              f"output_tokens={result['output_tokens']} "
              f"gen_duration={result['generation_duration_ms']:.0f}ms "
              f"parse_attempts={result['parse_attempts']}")
        print()

    print("If all events above produced valid structured output with no "
          "exceptions, Day 2's gate is met. Next: run test_baseline_is_frozen "
          "to lock the baseline before starting Day 3 (adapter).")


if __name__ == "__main__":
    main()
