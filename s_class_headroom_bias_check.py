"""S-class headroom-bias check: was the ORIGINAL 20-event S draw unrepresentative?

This is a different question from the existing N=63 supplementary check, and the
writeup must keep them apart:

  * N=63 (s_class_expanded_check.py) asks "does MORE S-class data change the
    picture" -- an expanded supplementary sample alongside the headline.
  * THIS script asks "was the headline's own original 20-event S draw itself
    drawn from favourable records" -- a redraw of the headline set's S quarter.

Motivation (results/s_class_per_record_results.json): the headline's S quarter
comes from select_events(), which walks S-class indices in first-occurrence order
under a 5-per-record cap. That happened to land on only 7 of the 16 S-containing
records, and three of those contributed exactly ONE event each (117, 121, 202) --
each scoring probe_acc 1.0 -- plus record 113 (n=5) also at 1.0. Per-record probe
accuracy averaged 0.686 on the original 20 versus 0.245 on the 43 additional
events, i.e. the original draw sits at the favourable end of a real spread.

Method: hold the N/V/F quarters EXACTLY as the headline has them, replace only
the S quarter with a round-robin draw across all 16 S-containing records (1st
event from each record, then 2nd, ...), keeping the same 5-per-record cap. This
caps any single record at 2 events instead of 5 and gives 16 records
representation instead of 7. Everything else -- 5-fold x 3-repeat stratified CV,
PCA-30, StandardScaler, LogisticRegression, random_state=42 -- is unchanged.

Both selections are scored in ONE process so the comparison cannot be confounded
by environment drift; the original is recomputed rather than read from the
ledger, which doubles as a reproduction check on the published 67.9% figure.

Runs entirely on CPU and does NOT load Gemma: the probe only needs the adapter's
weights, and the checkpoint stores embedding_dim/num_tokens/input_dim itself.
(day7_auditability_probe.py loads the full model purely to read a device off it.)
That keeps this safe to run alongside a GPU job.

Run (from repo root):
    python s_class_headroom_bias_check.py
"""
import collections
import json

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from day7_auditability_probe import (
    AAMI_CLASSES, MAX_PER_RECORD, N_FOLDS, N_REPEATS, PCA_COMPONENTS, PER_CLASS,
    select_events,
)
from perception.perception_agent import PerceptionAgent
from reasoning.virtual_adapter import VirtualTokenAdapter

ADAPTER_CHECKPOINT = "reasoning/checkpoints/virtual_adapter_day5_larger.pt"
PERCEPTION_CHECKPOINT = "perception/checkpoints/cnn_lstm.pt"
DS2_PATH = "data/processed/ds2_test.npz"
RESULTS_PATH = "results/s_class_headroom_bias_results.json"
S_CLASS_ID = 1


def stratified_s_draw(y, record_ids, per_class=PER_CLASS, max_per_record=MAX_PER_RECORD):
    """Round-robin S-class draw across every S-containing record.

    Takes the 1st event from each record, then the 2nd from each, and so on,
    stopping at per_class events or max_per_record rounds. Deterministic: records
    are visited in sorted id order and events within a record stay in
    first-occurrence order, matching the existing selection convention.
    """
    by_record = collections.defaultdict(list)
    for i in np.flatnonzero(y == S_CLASS_ID):
        by_record[int(record_ids[i])].append(int(i))

    picked = []
    for round_no in range(max_per_record):
        for record in sorted(by_record):
            if len(picked) >= per_class:
                break
            if round_no < len(by_record[record]):
                picked.append(by_record[record][round_no])
        if len(picked) >= per_class:
            break
    if len(picked) < per_class:
        raise RuntimeError(f"stratified draw found only {len(picked)}/{per_class} S events")
    return sorted(picked)


PCA_SEEDS = list(range(10))


def headline_pipeline(pca_seed):
    """PCA gets an explicit random_state.

    day7_auditability_probe.py uses a bare PCA(n_components=30). At this shape
    (64 training samples x 10,240 adapter dims) sklearn's svd_solver='auto'
    selects the RANDOMIZED solver, which is stochastic without a random_state --
    so the published headline is one draw from a distribution, not a fixed value.
    Confirmed directly: two identical runs of the unmodified probe differ, and
    only the PCA-using probes differ (the no-PCA and context-32d probes
    reproduce bit-exactly). Comparing two selections on single unseeded draws
    would therefore measure PCA noise as much as the redraw effect, so both
    selections are scored across the same PCA_SEEDS and compared as
    distributions.
    """
    return Pipeline([
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=PCA_COMPONENTS, random_state=pca_seed)),
        ("clf", LogisticRegression(max_iter=5000)),
    ])


def score_selection(label, selected, X, y, agent, adapter):
    """Full headline-probe protocol on one 80-event selection."""
    context_vectors = []
    for idx in selected:
        agent.predict(X[idx])
        context_vectors.append(agent.get_last_context_vector().copy())
    context_vectors = np.stack(context_vectors)
    true_classes = np.array([int(y[i]) for i in selected])

    with torch.no_grad():
        ctx = torch.from_numpy(context_vectors).float().to(next(adapter.parameters()).device)
        adapter_vectors = adapter(ctx).flatten(start_dim=1).cpu().numpy()

    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=42)
    per_seed = {}
    for seed in PCA_SEEDS:
        per_seed[seed] = float(cross_val_score(headline_pipeline(seed), adapter_vectors,
                                               true_classes, cv=cv, scoring="accuracy").mean())
    seed_means = np.array(list(per_seed.values()))
    scores = cross_val_score(headline_pipeline(PCA_SEEDS[0]), adapter_vectors,
                             true_classes, cv=cv, scoring="accuracy")

    single_pass_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof = cross_val_predict(headline_pipeline(PCA_SEEDS[0]), adapter_vectors, true_classes,
                            cv=single_pass_cv)
    present = sorted(set(true_classes.tolist()) | set(oof.tolist()))
    report = classification_report(true_classes, oof, labels=present,
                                   target_names=[AAMI_CLASSES[i] for i in present],
                                   zero_division=0, output_dict=True)

    s_mask = true_classes == S_CLASS_ID
    s_recall = float((oof[s_mask] == S_CLASS_ID).mean())

    print(f"\n{'='*72}\n{label}\n{'='*72}")
    print(classification_report(true_classes, oof, labels=present,
                                target_names=[AAMI_CLASSES[i] for i in present],
                                zero_division=0))
    print("Confusion (rows=true, cols=pred), classes %s:" % [AAMI_CLASSES[i] for i in present])
    print(confusion_matrix(true_classes, oof, labels=present))
    print(f"HEADLINE over {len(PCA_SEEDS)} PCA seeds: mean {seed_means.mean():.2%} "
          f"sd {seed_means.std(ddof=1):.2%} range [{seed_means.min():.1%}, {seed_means.max():.1%}]")
    print(f"  (single seed={PCA_SEEDS[0]}: {scores.mean():.1%} +/- {scores.std():.1%} across folds)")
    print(f"pooled out-of-fold accuracy: {(oof == true_classes).mean():.1%}")
    print(f"S-class recall: {s_recall:.1%} ({int((oof[s_mask]==S_CLASS_ID).sum())}/{int(s_mask.sum())})")

    return {
        "headline_seed_mean": float(seed_means.mean()),
        "headline_seed_sd": float(seed_means.std(ddof=1)),
        "headline_seed_min": float(seed_means.min()),
        "headline_seed_max": float(seed_means.max()),
        "headline_per_pca_seed": per_seed,
        "headline_mean": float(scores.mean()),
        "headline_sd": float(scores.std()),
        "fold_scores": scores.tolist(),
        "pooled_oof_accuracy": float((oof == true_classes).mean()),
        "s_recall": s_recall,
        "per_class": {AAMI_CLASSES[i]: report[AAMI_CLASSES[i]] for i in present},
        "s_records": dict(sorted(collections.Counter(
            int(r) for r in [RECORD_IDS[i] for i in selected if y[i] == S_CLASS_ID]).items())),
        "selected": selected,
    }


def main(device: str = "cpu"):
    global RECORD_IDS
    data = np.load(DS2_PATH)
    X, y, RECORD_IDS = data["features"], data["labels"], data["record_ids"]

    original = select_events(y, RECORD_IDS)
    nvf = [i for i in original if y[i] != S_CLASS_ID]           # N/V/F quarters untouched
    redrawn = sorted(nvf + stratified_s_draw(y, RECORD_IDS))

    orig_s = sorted(collections.Counter(int(RECORD_IDS[i]) for i in original if y[i] == S_CLASS_ID).items())
    new_s = sorted(collections.Counter(int(RECORD_IDS[i]) for i in redrawn if y[i] == S_CLASS_ID).items())
    print(f"original S quarter: {len(orig_s)} records, max {max(c for _, c in orig_s)}/record -> {orig_s}")
    print(f"redrawn  S quarter: {len(new_s)} records, max {max(c for _, c in new_s)}/record -> {new_s}")
    print(f"N/V/F quarters identical: {sorted(nvf) == sorted(i for i in redrawn if y[i] != S_CLASS_ID)}")

    # CPU-only: no Gemma, adapter rebuilt straight from its own saved dims.
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT, device=device)
    ckpt = torch.load(ADAPTER_CHECKPOINT, map_location="cpu", weights_only=True)
    adapter = VirtualTokenAdapter(ckpt["embedding_dim"], num_tokens=ckpt["num_tokens"],
                                  input_dim=ckpt["input_dim"])
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    adapter.eval()
    adapter.to(device)

    before = score_selection("ORIGINAL selection (reproduction check vs published 67.9%)",
                             original, X, y, agent, adapter)
    after = score_selection("REDRAWN S quarter (stratified across all 16 S records)",
                            redrawn, X, y, agent, adapter)

    delta = after["headline_mean"] - before["headline_mean"]
    print(f"\n{'='*72}\nVERDICT\n{'='*72}")
    print(f"headline {before['headline_mean']:.1%} -> {after['headline_mean']:.1%} ({delta*100:+.1f} pp)")
    print(f"S recall {before['s_recall']:.1%} -> {after['s_recall']:.1%} "
          f"({(after['s_recall']-before['s_recall'])*100:+.1f} pp)")

    out = {"original": before, "redrawn": after,
           "delta_headline_pp": delta * 100,
           "delta_s_recall_pp": (after["s_recall"] - before["s_recall"]) * 100,
           "config": {"per_class": PER_CLASS, "max_per_record": MAX_PER_RECORD,
                      "n_folds": N_FOLDS, "n_repeats": N_REPEATS,
                      "pca_components": PCA_COMPONENTS, "device": device}}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, safe alongside a GPU job) or cuda (matches the "
                         "hardware the published figure was computed on).")
    main(ap.parse_args().device)
