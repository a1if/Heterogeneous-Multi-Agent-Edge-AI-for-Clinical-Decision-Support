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

     v2.1 fix: StandardScaler now runs BEFORE PCA for every pipeline,
     including the adapter probes (v2 originally scaled only the context-32d
     pipelines, an inconsistency flagged on review -- PCA is scale-sensitive,
     and the two feature sets need identical preprocessing for the
     attribution comparison in S4 to be trustworthy).

  4. ATTRIBUTION DIAGNOSTIC: also probes the RAW 32-dim context vector (the
     adapter's INPUT). Because the adapter is a linear expansion 32 -> 10,240,
     it cannot create information absent from the 32-dim input. So:
        - if adapter-output probe ~= context-32d probe  -> the auditability
          ceiling is set by the upstream classifier's 32-dim representation
          (shaped only by classification loss, design doc S10); the adapter
          preserves information faithfully.
        - if adapter-output probe << context-32d probe  -> the adapter is
          destroying information. A different, important finding.

     v2.1 fix: "within noise" is now a paired Wilcoxon signed-rank test on
     the 15 matched fold-level scores (same CV splits, both feature sets),
     not an eyeball threshold against the reported std. Reports a p-value.

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
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_score, cross_val_predict,
)
from sklearn.metrics import confusion_matrix, classification_report

from perception.perception_agent import PerceptionAgent
from reasoning.adapter_arm import load_trained_adapter
from reasoning.model_loader import load_model
from project_config import ADAPTER_CHECKPOINT, DS2_PATH, MAX_PER_RECORD, PERCEPTION_CHECKPOINT, PER_CLASS, select_events
from perception.model import AAMI_CLASSES  # single source of truth

# --- Configuration -----------------------------------------------------------

N_FOLDS        = 5
N_REPEATS      = 3
PCA_COMPONENTS = 30     # just under the theoretical rank-32 of the adapter output

# PCA MUST be seeded. At this shape (64 training samples x 10,240 adapter dims)
# sklearn's svd_solver="auto" selects the RANDOMIZED solver, which draws a fresh
# random projection every call when random_state is unset. A bare
# PCA(n_components=30) therefore made the headline non-reproducible: re-running
# the unmodified script gave 67.5% where 67.9% had been recorded, and across 10
# PCA seeds the figure spans [67.1%, 67.9%]. Diagnosed by elimination -- the two
# probes that use no PCA (adapter no-PCA, context-32d) reproduced bit-exactly
# across runs while both PCA probes did not, and seeding removes the difference.
# Reported alongside a seed-sensitivity sweep (see PCA_SEED_SWEEP) so the
# headline is quoted as a central estimate, not one lucky draw.
PCA_RANDOM_STATE = 42
PCA_SEED_SWEEP = list(range(10))

RESULTS_PATH   = "results/day7_auditability_results_v2.json"

# Stable internal keys, decoupled from display strings -- v1 used the display
# name itself as a dict key (fragile: any whitespace edit to a label silently
# breaks the lookup with a KeyError). Named keys are used throughout v2.1.
KEY_ADAPTER_NOPCA = "adapter_linear_nopca"
KEY_ADAPTER_PCA   = "adapter_linear_pca"
KEY_ADAPTER_RBF   = "adapter_rbf_pca"
KEY_CONTEXT_LIN   = "context_linear"
KEY_CONTEXT_RBF   = "context_rbf"


def build_probe_suite() -> list[tuple[str, str, str, Pipeline]]:
    """(internal key, display name, feature key, sklearn Pipeline).

    v2.1: EVERY pipeline now scales before any dimensionality reduction or
    classification -- consistent preprocessing across adapter and context
    feature sets, required for the attribution comparison (S4) to be a fair
    like-for-like test rather than comparing differently-preprocessed inputs.
    """
    return [
        (KEY_ADAPTER_NOPCA, "Adapter | Linear | No PCA (original method)", "adapter",
         Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=5000))])),

        (KEY_ADAPTER_PCA, "Adapter | Linear | PCA-30   [HEADLINE]", "adapter",
         Pipeline([("sc", StandardScaler()),
                   ("pca", PCA(n_components=PCA_COMPONENTS, random_state=PCA_RANDOM_STATE)),
                   ("clf", LogisticRegression(max_iter=5000))])),

        (KEY_ADAPTER_RBF, "Adapter | RBF-SVM | PCA-30  [non-linear UB]", "adapter",
         Pipeline([("sc", StandardScaler()),
                   ("pca", PCA(n_components=PCA_COMPONENTS, random_state=PCA_RANDOM_STATE)),
                   ("clf", SVC(kernel="rbf"))])),

        (KEY_CONTEXT_LIN, "Context-32d | Linear        [attribution]", "context",
         Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=5000))])),

        (KEY_CONTEXT_RBF, "Context-32d | RBF-SVM       [attribution UB]", "context",
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
    fold_scores_by_key = {}  # for the paired significance test below
    for key, name, feat_key, pipeline in build_probe_suite():
        scores = cross_val_score(pipeline, feature_sets[feat_key],
                                  true_classes, cv=cv, scoring="accuracy")
        fold_scores_by_key[key] = scores
        mean, std = scores.mean(), scores.std()
        results.append({"key": key, "probe": name, "mean_accuracy": float(mean),
                         "std_accuracy": float(std), "n_folds": len(scores)})
        print(f"{name:<48} {mean:>7.1%} +/- {std:<5.1%}")
    print("=" * 74)

    # --- PCA seed sensitivity ------------------------------------------------
    # The randomized SVD solver means the headline depends on the PCA seed even
    # with everything else fixed. Seeding makes any ONE run reproducible, but the
    # honest figure to quote is the centre of this distribution, not whichever
    # draw a given seed lands on -- so sweep it and report mean/sd/range.
    seed_means = []
    for pca_seed in PCA_SEED_SWEEP:
        sweep_pipeline = Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=PCA_COMPONENTS, random_state=pca_seed)),
            ("clf", LogisticRegression(max_iter=5000)),
        ])
        seed_means.append(float(cross_val_score(
            sweep_pipeline, feature_sets["adapter"], true_classes,
            cv=cv, scoring="accuracy").mean()))
    seed_means = np.array(seed_means)
    headline_central = float(seed_means.mean())
    print(f"\nHEADLINE PCA-seed sensitivity ({len(PCA_SEED_SWEEP)} seeds): "
          f"mean {headline_central:.2%}, sd {seed_means.std(ddof=1):.2%}, "
          f"range [{seed_means.min():.1%}, {seed_means.max():.1%}]")
    print(f"  seed={PCA_RANDOM_STATE} (the reproducible default) gives "
          f"{fold_scores_by_key[KEY_ADAPTER_PCA].mean():.2%}")
    print("  Quote the seed-swept mean as the headline; a single seed is one draw.")

    # --- Per-class breakdown for the HEADLINE probe (out-of-fold predictions)
    headline_pipeline = Pipeline([("sc", StandardScaler()),
                                   ("pca", PCA(n_components=PCA_COMPONENTS, random_state=PCA_RANDOM_STATE)),
                                   ("clf", LogisticRegression(max_iter=5000))])
    single_pass_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_pred = cross_val_predict(headline_pipeline, adapter_vectors,
                                  true_classes, cv=single_pass_cv)
    present = sorted(set(true_classes.tolist()) | set(oof_pred.tolist()))
    print("\nHEADLINE probe (Adapter | Linear | PCA-30), out-of-fold predictions:")
    print(classification_report(true_classes, oof_pred, labels=present,
          target_names=[AAMI_CLASSES[i] for i in present], zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(f"  Classes: {[AAMI_CLASSES[i] for i in present]}")
    print(confusion_matrix(true_classes, oof_pred, labels=present))

    # --- Interpretation guide, using named keys (not fragile string lookups) --
    by_mean = {r["key"]: r["mean_accuracy"] for r in results}
    head  = by_mean[KEY_ADAPTER_PCA]
    nopca = by_mean[KEY_ADAPTER_NOPCA]
    rbf   = by_mean[KEY_ADAPTER_RBF]
    ctx_l = by_mean[KEY_CONTEXT_LIN]

    print("\n" + "=" * 74)
    print("INTERPRETATION")
    print("=" * 74)
    print(f"[PCA benefit]  No-PCA {nopca:.1%} -> PCA-30 {head:.1%}: "
          f"{'PCA stabilises the estimate.' if head >= nopca else 'check regime.'}")

    # v2.1: paired Wilcoxon signed-rank test on matched fold-level scores,
    # replacing the v2 eyeball "within +/-5pp of std" heuristic. Both arrays
    # come from the SAME cv splits (same `cv` object, same random_state), so
    # fold i in one array and fold i in the other score the same held-out
    # patients -- a valid paired comparison.
    adapter_fold_scores = fold_scores_by_key[KEY_ADAPTER_PCA]
    context_fold_scores = fold_scores_by_key[KEY_CONTEXT_LIN]
    gap = head - ctx_l
    # NOTE: scipy's wilcoxon() does not always raise on a fully-degenerate
    # input (all paired differences == 0) -- in some versions it instead
    # returns nan with a RuntimeWarning. Check for nan explicitly after the
    # call, in addition to catching ValueError, so this case is always
    # reported clearly rather than silently skipped by the branches below.
    try:
        wilcoxon_stat, p_value = scipy_stats.wilcoxon(
            adapter_fold_scores, context_fold_scores
        )
    except ValueError as e:
        wilcoxon_stat, p_value = float("nan"), float("nan")
        print(f"[Attribution]  Wilcoxon test degenerate ({e}); falling back to "
              f"mean comparison only.")

    print(f"[Attribution]  Adapter {head:.1%} vs Context-32d {ctx_l:.1%} "
          f"(delta {gap:+.1%}); paired Wilcoxon p={p_value:.3f}:")
    if np.isnan(p_value):
        print("    Wilcoxon test returned nan (degenerate: fold scores likely "
              "identical between feature sets). Falling back to the raw mean "
              f"gap ({gap:+.1%}) -- treat as inconclusive, not evidence either way.")
    elif p_value >= 0.05:
        print(f"    p={p_value:.3f} >= 0.05 -> NOT statistically distinguishable.")
        print("    Auditability ceiling is set by the upstream 32-dim representation")
        print("    (classification-loss-shaped); adapter preserves info faithfully.")
    elif gap < 0:
        print(f"    p={p_value:.3f} < 0.05, adapter LOWER -> adapter measurably")
        print("    destroys recoverable information relative to its own input.")
    else:
        print(f"    p={p_value:.3f} < 0.05, adapter HIGHER -> adapter measurably")
        print("    adds reconstructable structure beyond its raw input.")

    print(f"[Linearity]    Linear {head:.1%} vs RBF-SVM {rbf:.1%}:")
    if rbf - head > 0.08:
        print("    RBF notably better -> info present but NON-LINEARLY encoded.")
    else:
        print("    similar -> information is genuinely limited, not just non-linear")
        print("    (caveat: RBF-SVM is also more prone to overfitting at this")
        print("    sample size regardless of true structure -- both explanations")
        print("    should be stated, not just the more favourable one).")
    print("\nArm A (JSON) remains 100% auditable by construction; Arm B is at best")
    print("indirect, calibration-dependent reconstruction -- a categorical gap (S14).")

    # --- Persist ----------------------------------------------------------------
    out = {"config": {"per_class": PER_CLASS, "max_per_record": MAX_PER_RECORD,
                       "n_events": len(selected), "n_folds": N_FOLDS,
                       "n_repeats": N_REPEATS, "pca_components": PCA_COMPONENTS,
                       "chance_baseline": chance},
           "results": results,
           "attribution_wilcoxon": {
               "statistic": float(wilcoxon_stat) if not np.isnan(wilcoxon_stat) else None,
               "p_value": float(p_value) if not np.isnan(p_value) else None,
               "adapter_fold_scores": adapter_fold_scores.tolist(),
               "context_fold_scores": context_fold_scores.tolist(),
           },
           "headline_oof_predictions": oof_pred.tolist(),
           "true_classes": true_classes.tolist(),
           "pca_seed_sensitivity": {
               "seeds": PCA_SEED_SWEEP,
               "per_seed_mean_accuracy": seed_means.tolist(),
               "mean": headline_central,
               "sd": float(seed_means.std(ddof=1)),
               "min": float(seed_means.min()),
               "max": float(seed_means.max()),
               "default_seed": PCA_RANDOM_STATE,
               "note": ("The headline to quote is `mean` here, not the single-seed value: "
                        "sklearn picks the randomized SVD solver at this shape, so an "
                        "unseeded PCA made the figure irreproducible (a previously recorded "
                        "67.9% re-ran as 67.5%). random_state now fixes any one run; this "
                        "sweep gives the central estimate and its spread."),
           },
           "note": "v2.1: 80-event parity with Day 6, repeated stratified k-fold CV, "
                   "PCA rank-32 correction (now with consistent pre-PCA scaling across "
                   "ALL probes), 32-dim attribution + non-linear UB probes, paired "
                   "Wilcoxon significance test replacing the v2 eyeball threshold."}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
