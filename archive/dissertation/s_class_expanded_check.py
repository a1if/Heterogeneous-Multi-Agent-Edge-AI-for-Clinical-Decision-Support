"""Expand the S-class matched comparison (classifier recall vs. probe recall)
from N=20 to N=63, using the 43 additional S-class events identified by
find_additional_s_events.py, without touching the main 80-event balanced
auditability headline or its CV design.

Two different but equally valid "held out" mechanisms are combined, and this
script keeps them separately attributable rather than silently blending them:

  - PROBE, original 20: already-valid out-of-fold predictions from the
    existing 5-fold CV run (day7_auditability_probe.py's saved
    headline_oof_predictions), where each event was scored by a model that
    never saw it during that fold's training. Loaded from the existing
    results file, NOT recomputed -- guarantees consistency with everything
    already reported using these 20.
  - PROBE, new 43: predicted by a single probe (identical pipeline:
    StandardScaler -> PCA-30 -> LogisticRegression) fit ONCE on the original
    80-event balanced set. These 43 were never in that training set, so this
    is a genuine, unbiased holdout -- just a different mechanism than CV.
  - CLASSIFIER, both original 20 and new 43: the frozen Perception Agent's
    predict(), same as s_class_matched_check.py, extended to all 63. The
    original 20's classifier correctness was PREVIOUSLY only a hardcoded
    aggregate (6/20, copied from s_class_matched_check.py's own aggregate --
    itself never saved per-event either); this run recomputes it fresh,
    per-event, and cross-checks the total still matches 6/20.

Per-event correctness (classifier and probe, both groups) plus each event's
DS2 record_id is saved via two np.save calls -- s_class_per_event_original20.npy
and s_class_per_event_new43.npy, structured arrays with fields (idx, record_id,
classifier_correct, probe_correct). Neither existing results file (this
script's own aggregate JSON, or s_class_matched_check.py's) had ever retained
this; check_s_class_per_record.py needs it to test for a record-level pattern
in the N=20 vs N=63 recall drop, which an aggregate correct/total count cannot
distinguish from uniform per-event noise.

Needs the GPU: load_trained_adapter() requires the loaded Gemma model object
to size itself (see day7_auditability_probe.py L178-180), even though only
the adapter's own forward pass is used below -- there is no cheaper path
found in the reviewed code. Budget the same load time as day3_norm_check.py.
Adapter-projected vectors for both groups are cached to disk after the first
run (s_class_adapter_vectors_80_cache.npy / _new43_cache.npy) -- a rerun that
only needs to change per-event bookkeeping, not the projections themselves,
skips the Gemma load entirely. Delete the cache files to force a real rerun
of the adapter projection step (e.g. if the headline checkpoint changes).

Run:
    python s_class_expanded_check.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from perception.perception_agent import PerceptionAgent, SQI_LOW_THRESHOLD
from reasoning.adapter_arm import load_trained_adapter
from reasoning.model_loader import load_model
from day7_auditability_probe import select_events, PCA_COMPONENTS  # exact same pipeline params
from project_config import ADAPTER_CHECKPOINT, DS2_PATH, PERCEPTION_CHECKPOINT, wilson_ci

EXISTING_PROBE_RESULTS = "results/day7_auditability_results_v2.json"
ADDITIONAL_S_INDICES = "cache/additional_s_class_indices.npy"
RESULTS_PATH = "results/s_class_expanded_results.json"
PER_EVENT_ORIGINAL20_PATH = "cache/s_class_per_event_original20.npy"
PER_EVENT_NEW43_PATH = "cache/s_class_per_event_new43.npy"
ADAPTER_VECTORS_CACHE_80 = "cache/s_class_adapter_vectors_80_cache.npy"
ADAPTER_VECTORS_CACHE_NEW43 = "cache/s_class_adapter_vectors_new43_cache.npy"

S_CLASS_IDX = 1  # confirm against your actual encoding
PER_EVENT_DTYPE = [("idx", "i8"), ("record_id", "i8"), ("classifier_correct", "?"), ("probe_correct", "?")]


def main():
    # --- Load original 20's already-valid probe OOF predictions -------------
    with open(EXISTING_PROBE_RESULTS, encoding="utf-8") as f:
        existing = json.load(f)
    oof_pred = np.array(existing["headline_oof_predictions"])
    oof_true = np.array(existing["true_classes"])
    # Positions 20:40 are the S-class block -- select_events() iterates
    # class_id 0..3 (N/S/V/F) and extends sequentially, so this slice is
    # exact, not inferred (verified against oof_true[20:40] all == S_CLASS_IDX).
    assert (oof_true[20:40] == S_CLASS_IDX).all(), \
        "Position assumption broken -- oof array ordering has changed, do not proceed"
    original20_probe_correct_mask = (oof_pred[20:40] == S_CLASS_IDX)
    original_s_probe_correct = int(original20_probe_correct_mask.sum())
    print(f"Original 20 (probe, via existing CV OOF): {original_s_probe_correct}/20 correct")

    # --- Load data, both original 80 (for probe fit) and new 43 -------------
    data = np.load(DS2_PATH)
    X, y, record_ids = data["features"], data["labels"], data["record_ids"]
    original_80 = select_events(y, record_ids)
    original_20_s_indices = original_80[20:40]  # same S-class 20, same order as oof_pred[20:40]
    additional_s = np.load(ADDITIONAL_S_INDICES).tolist()
    print(f"New S-class events to evaluate: {len(additional_s)}")

    print("Loading Perception Agent...")
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)

    def get_context_and_predictions(indices):
        contexts, class_preds, sqis = [], [], []
        for idx in indices:
            event = agent.predict(X[idx])
            contexts.append(agent.get_last_context_vector().copy())
            class_preds.append(event["classification"]["label"])
            sqis.append(event["segment_metadata"]["signal_quality_index"])
        return np.stack(contexts), np.array(class_preds), np.array(sqis)

    print("Running classifier + collecting context vectors for original 80 (probe fit set)...")
    ctx_80, _, _ = get_context_and_predictions(original_80)
    true_80 = np.array([int(y[idx]) for idx in original_80])

    print("Running classifier on the original 20 S-class events (per-event -- "
          "previously only a hardcoded aggregate, never computed per-event by any script)...")
    _, clf_pred_original20, _ = get_context_and_predictions(original_20_s_indices)
    original20_clf_correct_mask = (clf_pred_original20 == "S")
    original_s_clf_correct = int(original20_clf_correct_mask.sum())
    print(f"Original 20 (classifier, freshly computed): {original_s_clf_correct}/20 correct")
    if original_s_clf_correct != 6:
        print(f"  WARNING: mismatch against the 6/20 previously hardcoded from "
              f"s_class_matched_check.py -- investigate before trusting downstream N=63 numbers")

    print("Running classifier + collecting context vectors for new 43...")
    ctx_new43, clf_pred_new43, sqi_new43 = get_context_and_predictions(additional_s)
    del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- SQI-override diagnostic, same check as s_class_matched_check.py ----
    n_sqi_overridden = int((sqi_new43 < SQI_LOW_THRESHOLD).sum())
    print(f"\nSQI-override diagnostic (new 43): {n_sqi_overridden}/{len(additional_s)} "
          f"forced to 'Q' regardless of real argmax.")
    if n_sqi_overridden > 0:
        print("  WARNING: classifier recall on the new 43 is not a clean proxy for these events.")

    # --- Classifier recall, new 43 -------------------------------------------
    new43_clf_correct_mask = (clf_pred_new43 == "S")
    new43_clf_correct = int(new43_clf_correct_mask.sum())
    print(f"New 43 (classifier): {new43_clf_correct}/{len(additional_s)} correct")

    # --- Adapter projection, both sets (cached after first run) --------------
    if Path(ADAPTER_VECTORS_CACHE_80).exists() and Path(ADAPTER_VECTORS_CACHE_NEW43).exists():
        print("\nUsing cached adapter-projected vectors, skipping Gemma load entirely...")
        adapter_vectors_80 = np.load(ADAPTER_VECTORS_CACHE_80)
        adapter_vectors_new43 = np.load(ADAPTER_VECTORS_CACHE_NEW43)
    else:
        print("\nLoading Gemma (for adapter sizing) + trained adapter...")
        model, _ = load_model()
        adapter = load_trained_adapter(ADAPTER_CHECKPOINT, model)
        device = next(adapter.parameters()).device
        with torch.no_grad():
            adapter_vectors_80 = adapter(
                torch.from_numpy(ctx_80).float().to(device)
            ).flatten(start_dim=1).cpu().numpy()
            adapter_vectors_new43 = adapter(
                torch.from_numpy(ctx_new43).float().to(device)
            ).flatten(start_dim=1).cpu().numpy()
        np.save(ADAPTER_VECTORS_CACHE_80, adapter_vectors_80)
        np.save(ADAPTER_VECTORS_CACHE_NEW43, adapter_vectors_new43)

    # --- Fit ONE probe on the original 80, predict on the new 43 ------------
    # Identical pipeline to day7_auditability_probe.py's headline probe.
    # Fitting on all 80 (not just S) matches the real headline protocol's
    # feature distribution; predicting only on the 43 keeps this a genuine
    # holdout, since none of them were in the fit.
    print("Fitting single probe on original 80, predicting on new 43...")
    probe = Pipeline([("sc", StandardScaler()),
                       ("pca", PCA(n_components=PCA_COMPONENTS)),
                       ("clf", LogisticRegression(max_iter=5000))])
    probe.fit(adapter_vectors_80, true_80)
    probe_pred_new43 = probe.predict(adapter_vectors_new43)
    new43_probe_correct_mask = (probe_pred_new43 == S_CLASS_IDX)
    new43_probe_correct = int(new43_probe_correct_mask.sum())
    print(f"New 43 (probe, single-fit holdout): {new43_probe_correct}/{len(additional_s)} correct")

    # --- Save per-event correctness + record_id, for check_s_class_per_record.py
    original20_per_event = np.array(
        list(zip(
            original_20_s_indices,
            record_ids[original_20_s_indices].tolist(),
            original20_clf_correct_mask.tolist(),
            original20_probe_correct_mask.tolist(),
        )),
        dtype=PER_EVENT_DTYPE,
    )
    new43_per_event = np.array(
        list(zip(
            additional_s,
            record_ids[additional_s].tolist(),
            new43_clf_correct_mask.tolist(),
            new43_probe_correct_mask.tolist(),
        )),
        dtype=PER_EVENT_DTYPE,
    )
    np.save(PER_EVENT_ORIGINAL20_PATH, original20_per_event)
    np.save(PER_EVENT_NEW43_PATH, new43_per_event)
    print(f"\nSaved per-event correctness+record_id: {PER_EVENT_ORIGINAL20_PATH}, {PER_EVENT_NEW43_PATH}")

    # --- Combine: N=63 for both classifier and probe ------------------------
    total_n = 20 + len(additional_s)
    clf_total_correct = original_s_clf_correct + new43_clf_correct
    probe_total_correct = original_s_probe_correct + new43_probe_correct

    clf_recall = 100 * clf_total_correct / total_n
    probe_recall = 100 * probe_total_correct / total_n
    clf_ci = wilson_ci(clf_total_correct, total_n)
    probe_ci = wilson_ci(probe_total_correct, total_n)

    print(f"\n{'='*70}")
    print(f"EXPANDED S-CLASS COMPARISON (N={total_n})")
    print(f"{'='*70}")
    print(f"Classifier: {clf_total_correct}/{total_n} = {clf_recall:.1f}% "
          f"[95% CI: {clf_ci[0]:.1f}-{clf_ci[1]:.1f}%]")
    print(f"Probe:      {probe_total_correct}/{total_n} = {probe_recall:.1f}% "
          f"[95% CI: {probe_ci[0]:.1f}-{probe_ci[1]:.1f}%]")
    overlap = not (clf_ci[1] < probe_ci[0] or probe_ci[1] < clf_ci[0])
    print(f"CIs {'OVERLAP' if overlap else 'DO NOT OVERLAP'} at N={total_n} "
          f"(compare to N=20's substantial overlap)")

    results = {
        "n": total_n,
        "classifier": {"correct": clf_total_correct, "recall": clf_recall,
                        "ci_wilson95": [round(clf_ci[0], 1), round(clf_ci[1], 1)],
                        "n_sqi_overridden_new43": n_sqi_overridden},
        "probe": {"correct": probe_total_correct, "recall": probe_recall,
                   "ci_wilson95": [round(probe_ci[0], 1), round(probe_ci[1], 1)],
                   "note": "combines original-20 CV out-of-fold predictions (existing, "
                           "unmodified) with new-43 predictions from a single probe fit "
                           "once on the original 80 -- two distinct but both genuinely "
                           "held-out mechanisms, disclosed rather than blended silently."},
        "cis_overlap": overlap,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
