"""
Attribution probe for the auxiliary-reconstruction-objective experiment.

Same protocol and same 80 selected DS2 events as day7_auditability_probe.py's
context-32d attribution probe (KEY_CONTEXT_LIN: StandardScaler + Logistic
Regression, RepeatedStratifiedKFold 5-fold x 3-repeat, random_state=42) --
imports select_events() from that script directly so the event selection is
identical, not re-derived. Skips loading Gemma + the adapter entirely: this
experiment only needs the raw context vector, not the adapter comparison, so
loading the full reasoning-side model would be unnecessary weight.

Run:
    python day7_auditability_probe_aux.py --checkpoint perception/checkpoints/cnn_lstm_aux_seed101.pt
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from day7_auditability_probe import DS2_PATH, N_FOLDS, N_REPEATS, select_events
from perception.perception_agent import PerceptionAgent


def probe_context_vector(checkpoint_path: str) -> dict:
    agent = PerceptionAgent(checkpoint_path=checkpoint_path)
    data = np.load(DS2_PATH)
    X, y, record_ids = data["features"], data["labels"], data["record_ids"]
    selected = select_events(y, record_ids)

    context_vectors, true_classes = [], []
    for idx in selected:
        agent.predict(X[idx])
        context_vectors.append(agent.get_last_context_vector().copy())
        true_classes.append(int(y[idx]))
    context_vectors = np.stack(context_vectors)  # (80, 32)
    true_classes = np.array(true_classes)

    pipeline = Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000)),
    ])
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=42)
    scores = cross_val_score(pipeline, context_vectors, true_classes, cv=cv, scoring="accuracy")

    return {
        "checkpoint": checkpoint_path,
        "n_events": len(selected),
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
        "fold_scores": scores.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    result = probe_context_vector(args.checkpoint)
    print(json.dumps(result, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.out}")
