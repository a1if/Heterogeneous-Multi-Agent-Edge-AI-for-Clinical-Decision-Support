"""Matched-population check: does the probe-vs-classifier S-class gap survive
when both numbers come from the SAME evaluation events?

Two things this answers:
  1. What is the actual S-class N inside the probe's evaluation set?
     (If it's small, e.g. under ~15, the 60% figure needs raw counts, not
     a percentage, the same way the McNemar result did.)
  2. What is the classification head's S-class recall on that SAME subset
     (not the full DS2 test set)? This isolates "representation vs.
     classification head" from "different evaluation population."

Adjust PERCEPTION_CHECKPOINT / DS2_PATH / import paths to match your actual
file layout -- this reuses the same select_events() function already used
in day7_auditability_probe.py and day3_norm_check.py, specifically so
the event indices are guaranteed identical, not a fresh re-sample.

Run:
    python s_class_matched_check.py
"""
import json
import math
import numpy as np
from collections import Counter

from perception.perception_agent import PerceptionAgent, compute_sqi, SQI_LOW_THRESHOLD  # noqa: F401 -- SQI_LOW_THRESHOLD used for diagnostic below
from day7_auditability_probe import select_events  # same selection fn as the probe itself

PERCEPTION_CHECKPOINT = "perception/checkpoints/cnn_lstm.pt"
DS2_PATH = "data/processed/ds2_test.npz"
RESULTS_PATH = "results/s_class_matched_check_results.json"

# AAMI class labels -- adjust if your label encoding differs
CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}  # confirm against your actual mapping

# Known from day7_auditability_results_v2.json: probe recovers S-class at 12/20 (60.0%)
PROBE_S_CORRECT = 12
PROBE_S_TOTAL = 20


def wilson_ci(correct, total, z=1.96):
    """Wilson score 95% CI for a binomial proportion. Returns (lower, upper) as percentages."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = correct / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    return (100 * (center - margin), 100 * (center + margin))


def main():
    data = np.load(DS2_PATH)
    X, y, record_ids = data["features"], data["labels"], data["record_ids"]

    selected = select_events(y, record_ids)
    y_selected = y[selected]

    print(f"Probe evaluation set: N={len(selected)}")
    class_counts = Counter(y_selected)
    print("\nPer-class N in the probe's evaluation set:")
    for cls_idx, name in CLASS_NAMES.items():
        n = class_counts.get(cls_idx, 0)
        print(f"  {name}: {n}")
        if n > 0 and n < 15:
            print(f"    ^ WARNING: N={n} is small -- report raw counts (e.g. '{{correct}}/{n}'), not a bare percentage")

    s_class_idx = [k for k, v in CLASS_NAMES.items() if v == "S"][0]
    s_n = class_counts.get(s_class_idx, 0)

    if s_n == 0:
        print("\nS-class N is ZERO in the probe's evaluation set.")
        print("The '60% probe recall' figure cannot be an S-class recall computed on this")
        print("selection -- check whether it actually comes from a different evaluation set,")
        print("or whether 'S-class' in that figure means something else than assumed here.")
        return

    # Run the classifier on the SAME selected events, same checkpoint
    print(f"\nRunning classifier on the same {len(selected)} events (checkpoint: {PERCEPTION_CHECKPOINT})...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    predictions = []       # predict()'s actual output label (string, e.g. "S"), possibly SQI-overridden
    sqi_values = []        # raw signal-quality index per event, independent of predict()'s branching
    for idx in selected:
        event = agent.predict(X[idx])
        predictions.append(event["classification"]["label"])       # string AAMI letter, not an int
        sqi_values.append(event["segment_metadata"]["signal_quality_index"])
    predictions = np.array(predictions)
    sqi_values = np.array(sqi_values)

    # S-class recall on THIS subset only -- compare strings, not ints
    s_mask = (y_selected == s_class_idx)
    s_label = CLASS_NAMES[s_class_idx]  # "S"
    s_correct = (predictions[s_mask] == s_label).sum()
    s_total = s_mask.sum()

    # SQI-override diagnostic: for the true-S events, how many fell below the
    # threshold that forces label="Q" regardless of the model's real argmax?
    # This is the confound predict() introduces that the probe's context
    # vector is NOT subject to -- if this count is nonzero, predict()'s label
    # is not a clean proxy for "what the classification head actually decided"
    # for those specific events.
    s_sqi = sqi_values[s_mask]
    n_sqi_overridden = int((s_sqi < SQI_LOW_THRESHOLD).sum())
    print(f"\nSQI-override diagnostic (true-S events only): "
          f"{n_sqi_overridden}/{s_total} fell below SQI_LOW_THRESHOLD and were forced to 'Q' "
          f"regardless of the model's real argmax.")
    if n_sqi_overridden > 0:
        print("  ^ WARNING: predict()'s label is NOT a clean proxy for classifier recall on "
              "these events -- the probe's context vector is extracted before this override "
              "and is unaffected by it. Consider bypassing predict()'s SQI branch (recompute "
              "argmax(probs) directly) for a comparison that matches what the probe measures.")
    else:
        print("  No S-class events hit the SQI override -- predict()'s label is a clean proxy "
              "for the raw classifier decision on this subset.")
    matched_recall = s_correct / s_total if s_total > 0 else float("nan")
    clf_ci_low, clf_ci_high = wilson_ci(s_correct, s_total)
    probe_ci_low, probe_ci_high = wilson_ci(PROBE_S_CORRECT, PROBE_S_TOTAL)

    print(f"\nMatched-subset S-class recall (classifier): {s_correct}/{s_total} = "
          f"{matched_recall:.1%}  [95% CI: {clf_ci_low:.1f}-{clf_ci_high:.1f}%]")
    print(f"Probe S-class recall (from day7_auditability_results_v2.json): "
          f"{PROBE_S_CORRECT}/{PROBE_S_TOTAL} = {PROBE_S_CORRECT/PROBE_S_TOTAL:.1%}  "
          f"[95% CI: {probe_ci_low:.1f}-{probe_ci_high:.1f}%]")

    overlap = not (clf_ci_high < probe_ci_low or probe_ci_high < clf_ci_low)
    print(f"\nCIs {'OVERLAP' if overlap else 'DO NOT OVERLAP'} -- "
          f"{'gap is NOT clearly established at this N; report cautiously' if overlap else 'gap holds even accounting for sampling uncertainty at both N'}")

    results = {
        "probe_eval_set_n": int(len(selected)),
        "per_class_n": {name: int(class_counts.get(idx, 0)) for idx, name in CLASS_NAMES.items()},
        "s_class": {
            "n_in_probe_subset": int(s_total),
            "classifier_correct": int(s_correct),
            "classifier_recall_matched_subset": float(matched_recall),
            "classifier_ci_wilson95": [round(clf_ci_low, 1), round(clf_ci_high, 1)],
            "probe_correct": PROBE_S_CORRECT,
            "probe_total": PROBE_S_TOTAL,
            "probe_recall": PROBE_S_CORRECT / PROBE_S_TOTAL,
            "probe_ci_wilson95": [round(probe_ci_low, 1), round(probe_ci_high, 1)],
            "cis_overlap": overlap,
            "n_sqi_overridden": n_sqi_overridden,
            "note": "Both figures now on the SAME 20-event S-class population with matching "
                    "Wilson 95% CIs. Do not report either as a bare point estimate in Discussion. "
                    "If n_sqi_overridden > 0, classifier_recall_matched_subset reflects predict()'s "
                    "SQI-override behavior, not a pure argmax(probs) comparison -- see console warning.",
        },
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
