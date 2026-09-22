"""Phase 1 run-order item 3: class recoverability at the 32-d context vector,
reference CNN-LSTM vs RR-branch encoder, on the E80 events.

Protocol as dissertation §4.5 / day7_auditability_probe_aux.py: the same event
selection (select_events imported from day7_auditability_probe), StandardScaler +
LogisticRegression, RepeatedStratifiedKFold 5x3, random_state=42, so the two
encoders are scored on identical folds and can be compared per fold (paired
Wilcoxon). One change: vectors are collected by chronological replay
(replay_selected), because the RR encoder's features depend on the preceding
beats. The reference encoder's context vector does not depend on agent state,
so the reference probe must reproduce the dissertation's 69.6% (gate below).
CPU only.

Run (from repo root):
    python p1_rr_probe.py
"""
import json

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from day7_auditability_probe import DS2_PATH, N_FOLDS, N_REPEATS, select_events
from perception.model import AAMI_CLASSES
from perception.perception_agent import PerceptionAgent, replay_selected

ENCODERS = {"reference_cnn_lstm": "perception/checkpoints/cnn_lstm.pt",
            "rr_encoder_seed0": "perception/checkpoints/cnn_lstm_rr_seed0.pt"}
DISSERTATION_CONTEXT_PROBE = 0.696
RESULTS_PATH = "results/p1_rr_probe_results.json"


def pipeline():
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))])


def main():
    with np.load(DS2_PATH) as npz:
        X, y, rr, record_ids = (npz[k] for k in ("features", "labels", "rr_interval_ms", "record_ids"))
    selected = [int(i) for i in select_events(y, record_ids)]
    labels = y[selected]
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=42)

    results, fold_scores = {"event_indices": selected, "encoders": {}}, {}
    for name, path in ENCODERS.items():
        agent = PerceptionAgent(checkpoint_path=path, device="cpu")
        replayed = replay_selected(agent, X, rr, record_ids, selected)
        vectors = np.stack([replayed[i][1] for i in selected])
        predicted = np.array([AAMI_CLASSES.index(replayed[i][0]["classification"]["label"]) for i in selected])

        scores = cross_val_score(pipeline(), vectors, labels, cv=cv, scoring="accuracy")
        fold_scores[name] = scores
        # One 5-fold pass for the confusion matrix (the scores above use all 3 repeats).
        cm_pred = cross_val_predict(pipeline(), vectors, labels,
                                    cv=StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42))
        present = sorted(set(labels))
        cm = confusion_matrix(labels, cm_pred, labels=present)
        results["encoders"][name] = {
            "checkpoint": path,
            "probe_accuracy_mean": float(scores.mean()), "probe_accuracy_sd": float(scores.std()),
            "probe_per_class_recall": {AAMI_CLASSES[c]: float(cm[k, k] / cm[k].sum()) for k, c in enumerate(present)},
            "probe_confusion_matrix": {"classes": [AAMI_CLASSES[c] for c in present], "matrix": cm.tolist()},
            "encoder_accuracy_on_e80": float((predicted == labels).mean()),
        }
        print(f"{name:20s} probe {scores.mean():.3f} ± {scores.std():.3f}  "
              f"encoder acc on E80 {results['encoders'][name]['encoder_accuracy_on_e80']:.3f}  "
              f"probe recall {results['encoders'][name]['probe_per_class_recall']}")

    ref = results["encoders"]["reference_cnn_lstm"]["probe_accuracy_mean"]
    results["reference_reproduces_dissertation"] = abs(ref - DISSERTATION_CONTEXT_PROBE) < 0.005
    if not results["reference_reproduces_dissertation"]:
        print(f"WARNING: reference probe {ref:.3f} != dissertation {DISSERTATION_CONTEXT_PROBE}")
    stat = wilcoxon(fold_scores["rr_encoder_seed0"], fold_scores["reference_cnn_lstm"])
    results["paired_wilcoxon_rr_vs_reference"] = {"statistic": float(stat.statistic), "p": float(stat.pvalue),
                                                  "n_folds": len(fold_scores["reference_cnn_lstm"])}
    print(f"paired Wilcoxon over {len(fold_scores['reference_cnn_lstm'])} matched folds: p = {stat.pvalue:.3g}")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
