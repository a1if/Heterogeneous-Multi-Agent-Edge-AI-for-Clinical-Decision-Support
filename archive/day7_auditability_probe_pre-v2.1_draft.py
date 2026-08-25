"""
Day 7 (v2): strengthened linear-probe auditability test.

Upgrades over the original day7_auditability_probe.py:

  1. N scaled to 80 (20/class) to match the Day 6 run exactly -- the original
     used 60 events / 20 test samples, which was too small to be credible.

  2. Repeated stratified k-fold CV (5-fold x 3 repeats) replaces the single
     40/20 split. Reports mean +/- std instead of one fragile point estimate.

  3. PCA dimensionality reduction. KEY INSIGHT: the adapter is a single LINEAR
     projection 32 -> 4x2560 (=10,240). Its output therefore lives in an
     affine subspace of rank <= 32 (verified: 337,920 params = 32*10240 + bias,
     design doc S12.1). The original probe ran logistic regression on all
     10,240 dims against 40 samples -- a pathological high-dim-low-N regime.
     PCA to 30 components removes ~10,210 noise dimensions. (Pipeline ensures
     PCA is fit per-fold, so no data leakage.)

  4. ATTRIBUTION DIAGNOSTIC: also probes the RAW 32-dim context vector (the
     adapter's INPUT). Because the adapter is a linear expansion 32 -> 10,240,
     it cannot create information absent from the 32-dim input. So:
        - if adapter-output probe ~= context-32d probe  -> the auditability
          ceiling is set by the upstream classifier's 32-dim representation
          (shaped only by classification loss, design doc S10); the adapter
          preserves information faithfully.
        - if adapter-output probe << context-32d probe  -> the adapter is
          destroying information. A different, important finding.

  5. NON-LINEAR UPPER BOUND: an RBF-SVM probe establishes whether class
     information is present but non-linearly encoded (linear low, RBF high)
     versus genuinely absent (both low).

Baseline for comparison: Arm A's JSON is 100% auditable BY CONSTRUCTION (the
class is a named field). Arm B is, at best, indirectly reconstructable -- a
categorically weaker property (design doc S14).

No chronological replay needed: the probe targets are (a) ground-truth AAMI
class from the dataset array, and (b) the 32-dim context vector, which is a
per-window CNN-LSTM feature, not the stateful consecutive_abnormal_beats
counter. (Same assumption as the original Day 7 script.)

Run:
    python day7_auditability_probe.py
"""
import json
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RepeatedStratifiedKFold, cross_val_score, cross_val_predict,
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from perception.perception_agent import PerceptionAgent
from reasoning.adapter_arm import load_trained_adapter
from reasoning.model_loader import load_model

# --- Configuration -----------------------------------------------------------
ADAPTER_CHECKPOINT   = "reasoning/checkpoints/virtual_adapter_day5_larger.pt"
PERCEPTION_CHECKPOINT = "perception/checkpoints/cnn_lstm.pt"
DS2_PATH             = "data/processed/ds2_test.npz"

PER_CLASS      = 20     # matches Day 6's 80-event run (was 15 -> 60 events)
MAX_PER_RECORD = 5
N_FOLDS        = 5
N_REPEATS      = 3
PCA_COMPONENTS = 30     # just under the theoretical rank-32 of the adapter output

AAMI_CLASSES   = ["N", "S", "V", "F", "Q"]
RESULTS_PATH   = "day7_auditability_results_v2.json"


def select_events(y: np.ndarray, record_ids: np.ndarray) -> list[int]:
    """Identical logic to day6_run_comparison.py so the probe evaluates the
    exact same 80 events Day 6 scored -- direct comparability."""
    from collections import defaultdict
    selected = []
    for class_id in range(4):                      # N/S/V/F (Q absent in DS2)
        per_record_count = defaultdict(int)
        class_indices = np.flatnonzero(y == class_id)
        picked = []
        for idx in class_indices:
            rec = int(record_ids[idx])
            if per_record_count[rec] >= MAX_PER_RECORD:
                continue
            picked.append(int(idx))
            per_record_count[rec] += 1
            if len(picked) == PER_CLASS:
                break
        if len(picked) < PER_CLASS:
            raise RuntimeError(
                f"Class {AAMI_CLASSES[class_id]}: only found "
                f"{len(picked)}/{PER_CLASS} events under the "
                f"{MAX_PER_RECORD}-per-record cap."
            )
        selected.extend(picked)
    return selected


def build_probe_suite() -> list[tuple[str, str, Pipeline]]:
    """(display name, feature key, sklearn Pipeline). Pipelines keep PCA/scaling
    inside the CV loop so nothing leaks from test fold into fitting."""
    return [
        # --- Adapter output (10,240-dim) ---
        ("Adapter | Linear | No PCA (original method)", "adapter",
         Pipeline([("clf", LogisticRegression(max_iter=2000))])),

        ("Adapter | Linear | PCA-30   [HEADLINE]", "adapter",
         Pipeline([("pca", PCA(n_components=PCA_COMPONENTS)),
                   ("clf", LogisticRegression(max_iter=2000))])),

        ("Adapter | RBF-SVM | PCA-30  [non-linear UB]", "adapter",
         Pipeline([("pca", PCA(n_components=PCA_COMPONENTS)),
                   ("sc", StandardScaler()),
                   ("clf", SVC(kernel="rbf"))])),

        # --- Raw 32-dim context vector (attribution) ---
        ("Context-32d | Linear        [attribution]", "context",
         Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000))])),

        ("Context-32d | RBF-SVM       [attribution UB]", "context",
         Pipeline([("sc", StandardScaler()),
                   ("clf", SVC(kernel="rbf"))])),
    ]


def main():
    print("Loading Perception Agent + DS2...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    data = np.load(DS2_PATH)
    X, y, record_ids = data["features"], data["labels"], data["record_ids"]
    selected = select_events(y, record_ids)
    print(f"Selected {len(selected)} events ({PER_CLASS}/class, "
          f"{MAX_PER_RECORD}/record cap) -- same selection as Day 6.")

    # Collect 32-dim context vectors + ground-truth classes.
    context_vectors, true_classes = [], []
    for idx in selected:
        agent.predict(X[idx])
        context_vectors.append(agent.get_last_context_vector().copy())
        true_classes.append(int(y[idx]))
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    context_vectors = np.stack(context_vectors)      # (80, 32)
    true_classes = np.array(true_classes)

    print("Loading Gemma (embedding width only) + trained adapter...")
    model, _ = load_model()
    adapter = load_trained_adapter(ADAPTER_CHECKPOINT, model)

    print("Projecting context vectors through the trained adapter...")
    with torch.no_grad():
        ctx = torch.from_numpy(context_vectors).float().to(
            next(adapter.parameters()).device)
        adapter_vectors = adapter(ctx).flatten(start_dim=1).cpu().numpy()
    print(f"Adapter output shape: {adapter_vectors.shape} "
          f"(rank <= 32 by construction).")

    feature_sets = {"adapter": adapter_vectors, "context": context_vectors}

    # --- Repeated stratified k-fold evaluation -------------------------------
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS,
                                 random_state=42)
    chance = 1.0 / 4.0

    print(f"\nEvaluating probes ({N_FOLDS}-fold x {N_REPEATS} repeats, "
          f"n={len(true_classes)})... chance baseline = {chance:.1%}\n")
    print("=" * 74)
    print(f"{'Probe':<48} {'Accuracy':>14}")
    print("=" * 74)

    results = []
    for name, feat_key, pipeline in build_probe_suite():
        scores = cross_val_score(pipeline, feature_sets[feat_key],
                                 true_classes, cv=cv, scoring="accuracy")
        mean, std = scores.mean(), scores.std()
        results.append({"probe": name, "mean_accuracy": float(mean),
                        "std_accuracy": float(std), "n_folds": len(scores)})
        print(f"{name:<48} {mean:>7.1%} +/- {std:<5.1%}")
    print("=" * 74)

    # --- Per-class breakdown for the HEADLINE probe (out-of-fold predictions)
    headline_pipeline = Pipeline([("pca", PCA(n_components=PCA_COMPONENTS)),
                                  ("clf", LogisticRegression(max_iter=2000))])
    oof_pred = cross_val_predict(headline_pipeline, adapter_vectors,
                                 true_classes, cv=StratifiedKFoldHelper())
    present = sorted(set(true_classes.tolist()) | set(oof_pred.tolist()))
    print("\nHEADLINE probe (Adapter | Linear | PCA-30), out-of-fold predictions:")
    print(classification_report(true_classes, oof_pred, labels=present,
          target_names=[AAMI_CLASSES[i] for i in present], zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"  Classes: {[AAMI_CLASSES[i] for i in present]}")
    print(confusion_matrix(true_classes, oof_pred, labels=present))

    # --- Interpretation guide -------------------------------------------------
    by_name = {r["probe"]: r["mean_accuracy"] for r in results}
    head  = by_name["Adapter | Linear | PCA-30   [HEADLINE]"]
    nopca = by_name["Adapter | Linear | No PCA (original method)"]
    rbf   = by_name["Adapter | RBF-SVM | PCA-30  [non-linear UB]"]
    ctx_l = by_name["Context-32d | Linear        [attribution]"]

    print("\n" + "=" * 74)
    print("INTERPRETATION")
    print("=" * 74)
    print(f"[PCA benefit]  No-PCA {nopca:.1%} -> PCA-30 {head:.1%}: "
          f"{'PCA stabilises the estimate.' if head >= nopca else 'check regime.'}")
    gap = head - ctx_l
    print(f"[Attribution]  Adapter {head:.1%} vs Context-32d {ctx_l:.1%} "
          f"(delta {gap:+.1%}):")
    if abs(gap) <= 0.05:
        print("    ~equal -> auditability ceiling is set by the upstream 32-dim")
        print("    representation (classification-loss-shaped); adapter preserves info.")
    elif gap < 0:
        print("    adapter LOWER -> adapter is destroying information (investigate).")
    else:
        print("    adapter HIGHER -> adapter adds reconstructable structure.")
    print(f"[Linearity]    Linear {head:.1%} vs RBF-SVM {rbf:.1%}:")
    if rbf - head > 0.08:
        print("    RBF notably better -> info present but NON-LINEARLY encoded.")
    else:
        print("    similar -> information is genuinely limited, not just non-linear.")
    print("\nArm A (JSON) remains 100% auditable by construction; Arm B is at best")
    print("indirect, calibration-dependent reconstruction -- a categorical gap (S14).")

    # --- Persist ----------------------------------------------------------------
    out = {"config": {"per_class": PER_CLASS, "max_per_record": MAX_PER_RECORD,
                      "n_events": len(selected), "n_folds": N_FOLDS,
                      "n_repeats": N_REPEATS, "pca_components": PCA_COMPONENTS,
                      "chance_baseline": chance},
           "results": results,
           "headline_oof_predictions": oof_pred.tolist(),
           "true_classes": true_classes.tolist(),
           "note": "v2: 80-event parity with Day 6, repeated stratified k-fold CV, "
                   "PCA rank-32 correction, 32-dim attribution + non-linear UB probes."}
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


def StratifiedKFoldHelper():
    """5-fold for the out-of-fold confusion matrix (single pass, no repeats)."""
    from sklearn.model_selection import StratifiedKFold
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)


if __name__ == "__main__":
    main()