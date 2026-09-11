"""Probe-target comparison: true AAMI class vs transmitted (predicted) class.

The headline class-recoverability probe (day7_auditability_probe.py) trains
against ``true_classes`` -- the ground-truth AAMI label from the dataset array.
Arm A's "100% by construction" refers to a different quantity: the class named
in the transmitted JSON, which is the ECG model's PREDICTION, not the label.

Those are two different recovery targets, and the headline comparison pairs one
of each. This script measures both targets under the identical probe pipeline so
the two arms can be compared like for like:

    target = true AAMI class        -> what the headline probe reports
    target = predicted class        -> what Arm A's readable field actually carries

Arm A's score against each target is not probed but read off directly: 100% for
the transmitted class (it is a named field) and the ECG model's own agreement
with ground truth for the true class.

Scope. This runs the CONTEXT-32D probe, i.e. the adapter's input, not its
output. That is deliberate: the attribution result of Section 4.5 found no
significant difference between probing the adapter's input and its output
(paired Wilcoxon p = 0.530), so context-32d stands as a validated proxy and this
script needs no GPU, no Gemma load, and no adapter checkpoint. The proxy was
established for the true-class target; extending it to the predicted-class
target is an assumption, flagged as such where the figure is reported.

Validation: the true-class arm of this script must reproduce the recorded
context-32d headline of 69.6% +/- 9.9pp exactly. It does.

Run:
    python day7_probe_target_comparison.py
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from perception.model import AAMI_CLASSES
from perception.perception_agent import PerceptionAgent
from project_config import DS2_PATH, PERCEPTION_CHECKPOINT, select_events

# Identical to day7_auditability_probe.py -- same folds, same repeats, same seed,
# same preprocessing. Only the probe TARGET differs between the two runs below.
N_FOLDS = 5
N_REPEATS = 3
CV_RANDOM_STATE = 42

RESULTS_PATH = "results/day7_probe_target_comparison.json"


def probe(features: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pipeline = Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=5000))])
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                 random_state=CV_RANDOM_STATE)
    scores = cross_val_score(pipeline, features, target, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def main():
    print("Loading Perception Agent + DS2 (same 80-event selection as Day 6/7)...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT, device="cpu")
    data = np.load(DS2_PATH)
    X, y, record_ids = data["features"], data["labels"], data["record_ids"]

    context_vectors, true_classes, predicted_classes = [], [], []
    for idx in select_events(y, record_ids):
        event = agent.predict(X[idx])
        context_vectors.append(agent.get_last_context_vector().copy())
        true_classes.append(int(y[idx]))
        predicted_classes.append(AAMI_CLASSES.index(event["classification"]["label"]))

    context_vectors = np.stack(context_vectors)
    true_classes = np.array(true_classes)
    predicted_classes = np.array(predicted_classes)
    n = len(true_classes)

    # Arm A's readable field IS the prediction, so its agreement with the true
    # label is exactly the accuracy of that field as a recovery of true class.
    agreement = float((true_classes == predicted_classes).mean())

    true_mean, true_sd = probe(context_vectors, true_classes)
    pred_mean, pred_sd = probe(context_vectors, predicted_classes)

    print(f"\nn = {n} events; ECG model agreement with ground truth: {agreement:.1%}\n")
    print("=" * 78)
    print(f"{'Recovery target':<34}{'Arm A (JSON field)':>20}{'Arm B (probe)':>24}")
    print("=" * 78)
    print(f"{'Transmitted (predicted) class':<34}{'100.0% (named field)':>20}"
          f"{pred_mean:>17.1%} +/- {pred_sd*100:.1f}pp")
    print(f"{'True AAMI class':<34}{agreement:>19.1%}"
          f"{true_mean:>17.1%} +/- {true_sd*100:.1f}pp")
    print("=" * 78)
    print("\nArm B figures are the context-32d probe (the adapter's input), used as a\n"
          "proxy for the adapter's output per the attribution result (p = 0.530).")

    expected = 0.696
    status = "MATCH" if abs(true_mean - expected) < 0.001 else "DIVERGED"
    print(f"\nValidation vs recorded context-32d headline ({expected:.1%}): "
          f"{true_mean:.1%} -- {status}")

    results = {
        "n_events": n,
        "probe": "context-32d, StandardScaler + LogisticRegression, "
                 f"{N_FOLDS}-fold x {N_REPEATS} repeats, random_state={CV_RANDOM_STATE}",
        "ecg_model_agreement_with_ground_truth": agreement,
        "arm_a_transmitted_class": 1.0,
        "arm_a_true_class": agreement,
        "arm_b_transmitted_class": {"mean": pred_mean, "sd": pred_sd},
        "arm_b_true_class": {"mean": true_mean, "sd": true_sd},
        "note": "Arm B is the adapter's INPUT (context-32d), a validated proxy for its "
                "output under the Section 4.5 attribution result (paired Wilcoxon "
                "p = 0.530). That equivalence was established for the true-class "
                "target; its extension to the predicted-class target is an assumption.",
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
