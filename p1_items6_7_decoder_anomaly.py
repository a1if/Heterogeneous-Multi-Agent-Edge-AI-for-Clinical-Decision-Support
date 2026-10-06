"""Phase 1 run-order items 6 and 7 (plan steps 7 and 8). CPU only: the adapter is
a single affine map, so its virtual tokens are computed from checkpoint weights
without loading Gemma.

Item 6, inverse decoder: which HealthEventJSON fields can be recovered from
what Arm B transmits? Decoders are trained on a DS1 sample and tested on a DS2
sample (inter-patient), for
  - the reference encoder's 32-d context vector (the adapter's input)
  - the adapter's virtual tokens for the three k=4 checkpoints (headline, E2 seeds
    101/202), reduced by PCA-32 fitted on DS1 (the adapter output has rank <= 32)
  - the RR encoder's context vector (upper bound for a future RR adapter)
Fields: predicted label, urgency tier, urgent flag, beat morphology (accuracy /
balanced accuracy) and confidence, top-2 margin, heart rate, RR, SQI, QRS, run
length (R^2). Decoders: linear (logistic / ridge) and an MLP.

Item 7, pre-generation anomaly check: scores computed from the 32-d vector alone,
before any generation:
  mahalanobis  min class-conditional Mahalanobis distance to DS1 (shared covariance)
  head_margin  1 - top-2 softmax margin of the perception head (the head is a
               function of the same vector)
  tier_decoder 1 - max probability of the MLP tier decoder from item 6
Evaluated (a) as corrupted-input detectors (AUROC, clean vs corrupted DS2 windows)
and (b) as predictors of Arm B's wrong answers on E80 (AUROC, per checkpoint, from
results/p1_step2_contested_events.json).

Run (from repo root):
    python p1_items6_7_decoder_anomaly.py
"""
import json

import numpy as np
import torch
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from perception.model import AAMI_CLASSES, CNNLSTM
from perception.perception_agent import PerceptionAgent, replay_selected
from project_config import select_events
from reasoning.training_targets import urgency_tier_from_event
from reasoning.virtual_adapter import VirtualTokenAdapter

DS1, DS2 = "data/processed/ds1_train.npz", "data/processed/ds2_test.npz"
REFERENCE, RR_ENCODER = "perception/checkpoints/cnn_lstm.pt", "perception/checkpoints/cnn_lstm_rr_seed0.pt"
ADAPTERS = {"headline": "reasoning/checkpoints/virtual_adapter_day5_larger.pt",
            "seed101": "reasoning/checkpoints/virtual_adapter_e2_seed101.pt",
            "seed202": "reasoning/checkpoints/virtual_adapter_e2_seed202.pt"}
STEP2 = "results/p1_step2_contested_events.json"
RESULTS_PATH = "results/p1_items6_7_decoder_anomaly.json"
PER_CLASS_DS1, PER_CLASS_DS2 = 1500, 500
CLASSIFY = ("label", "tier", "urgent", "morphology")
REGRESS = ("confidence", "margin", "heart_rate", "rr", "sqi", "qrs", "run")


RR_GAP_CAP_MS = 11999.0  # HealthEventJSON rejects heart rates < 5 bpm (RR > 12 s)


def load(path):
    """DS1 holds annotation gaps of up to ~100 s stored as RR intervals, which make
    PerceptionAgent's schema validation fail (heart rate < 5 bpm). They are gaps,
    not beats; they are capped just under the schema limit for this analysis only,
    and the count is reported."""
    with np.load(path) as z:
        data = {k: z[k] for k in ("features", "labels", "rr_interval_ms", "record_ids")}
    data["n_rr_capped"] = int((data["rr_interval_ms"] > RR_GAP_CAP_MS).sum())
    data["rr_interval_ms"] = np.minimum(data["rr_interval_ms"], RR_GAP_CAP_MS)
    return data


def stratified(labels, per_class, rng):
    idx = []
    for c in (0, 1, 2, 3):  # N, S, V, F (Q is absent from DS2)
        pool = np.flatnonzero(labels == c)
        idx.extend(rng.choice(pool, min(per_class, len(pool)), replace=False).tolist())
    return sorted(int(i) for i in idx)


def collect(checkpoint, data, selected):
    agent = PerceptionAgent(checkpoint_path=checkpoint, device="cpu")
    out = replay_selected(agent, data["features"], data["rr_interval_ms"], data["record_ids"], selected)
    vectors = np.stack([out[i][1] for i in selected])
    fields = {k: [] for k in CLASSIFY + REGRESS}
    for i in selected:
        ev = out[i][0]
        c, sf = ev["classification"], ev["signal_features"]
        fields["label"].append(c["label"])
        fields["tier"].append(urgency_tier_from_event(ev))
        fields["urgent"].append(ev["clinical_flags"]["requires_urgent_review"])
        fields["morphology"].append(sf["beat_morphology"])
        fields["confidence"].append(c["confidence"])
        fields["margin"].append(c["top_3"][0]["confidence"] - c["top_3"][1]["confidence"])
        fields["heart_rate"].append(sf["heart_rate_bpm"])
        fields["rr"].append(sf["rr_interval_ms"])
        fields["sqi"].append(ev["segment_metadata"]["signal_quality_index"])
        fields["qrs"].append(sf["qrs_duration_ms"])
        fields["run"].append(ev["clinical_flags"]["consecutive_abnormal_beats"])
    return vectors, {k: np.asarray(v) for k, v in fields.items()}


def adapter_tokens(path, vectors):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    adapter = VirtualTokenAdapter(ckpt["embedding_dim"], num_tokens=ckpt["num_tokens"], input_dim=ckpt["input_dim"])
    adapter.load_state_dict(ckpt["adapter_state_dict"])
    with torch.no_grad():
        tokens = adapter(torch.from_numpy(vectors).float()).reshape(len(vectors), -1).numpy()
    s = np.linalg.svd(adapter.projection.weight.detach().numpy(), compute_uv=False)
    return tokens, {"rank": int((s > s[0] * 1e-6).sum()), "condition_number": float(s[0] / s[-1])}


def decode(train_x, train_f, test_x, test_f, reduce=False):
    pre = [StandardScaler()] + ([PCA(n_components=32, random_state=0)] if reduce else [])
    res, tier_models = {}, {}
    for field in CLASSIFY:
        ytr, yte = train_f[field].astype(str), test_f[field].astype(str)
        res[field] = {}
        if len(set(ytr)) < 2:
            res[field] = {"skipped": "single class in DS1 sample"}
            continue
        for kind, clf in (("linear", LogisticRegression(max_iter=5000)),
                          ("mlp", MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=0))):
            m = make_pipeline(*[clone(p) for p in pre], clf).fit(train_x, ytr)
            pred = m.predict(test_x)
            res[field][kind] = {"accuracy": float((pred == yte).mean()),
                                "balanced_accuracy": float(balanced_accuracy_score(yte, pred))}
            if field == "tier":
                tier_models[kind] = m
    for field in REGRESS:
        ytr, yte = train_f[field].astype(float), test_f[field].astype(float)
        res[field] = {}
        for kind, reg in (("linear", Ridge(alpha=1.0)),
                          ("mlp", MLPRegressor(hidden_layer_sizes=(64,), max_iter=2000, random_state=0))):
            m = make_pipeline(*[clone(p) for p in pre], reg).fit(train_x, (ytr - ytr.mean()) / (ytr.std() + 1e-9))
            pred = m.predict(test_x) * (ytr.std() + 1e-9) + ytr.mean()
            res[field][kind] = {"r2": float(r2_score(yte, pred))}
    return res, tier_models


def mahalanobis_scorer(train_x, train_labels):
    classes = sorted(set(train_labels))
    means = {c: train_x[train_labels == c].mean(0) for c in classes}
    centered = np.concatenate([train_x[train_labels == c] - means[c] for c in classes])
    precision = np.linalg.pinv(np.cov(centered, rowvar=False))

    def score(x):
        return np.min([np.einsum("ij,jk,ik->i", x - means[c], precision, x - means[c]) for c in classes], axis=0)
    return score


def head_margin_score(vectors):
    model = CNNLSTM()
    model.load_state_dict(torch.load(REFERENCE, map_location="cpu"))
    with torch.no_grad():
        p = torch.softmax(model.head.eval()(torch.from_numpy(vectors).float()), dim=-1).numpy()
    top2 = np.sort(p, axis=1)[:, -2:]
    return 1.0 - (top2[:, 1] - top2[:, 0])


def corrupt(windows, kind, rng):
    w = windows.copy()
    n, L = w.shape
    t = np.arange(L) / 360.0
    if kind == "gaussian_noise_0dB":
        w += rng.normal(0, w.std(axis=1, keepdims=True), w.shape)
    elif kind == "baseline_wander":
        w += 3.0 * np.sin(2 * np.pi * 0.5 * t + rng.uniform(0, 2 * np.pi, (n, 1)))
    elif kind == "clipping":
        lim = np.percentile(np.abs(w), 60, axis=1, keepdims=True)
        w = np.clip(w, -lim, lim)
    elif kind == "segment_dropout":
        start = rng.integers(0, L // 2, n)
        for i, s in enumerate(start):
            w[i, s:s + L // 3] = w[i, s]
    elif kind == "powerline_60Hz":
        w += 1.0 * np.sin(2 * np.pi * 60 * t)
    return w


def raw_context(windows):
    """Context vectors straight from the reference model (its context vector does
    not depend on agent state), z-scoring each window as PerceptionAgent does."""
    model = CNNLSTM()
    model.load_state_dict(torch.load(REFERENCE, map_location="cpu"))
    model.eval()
    z = (windows - windows.mean(1, keepdims=True)) / (windows.std(1, keepdims=True) + 1e-8)
    with torch.no_grad():
        _, ctx = model(torch.from_numpy(z).float().unsqueeze(1))
    return ctx.numpy()


def main():
    rng = np.random.default_rng(0)
    ds1, ds2 = load(DS1), load(DS2)
    tr_idx = stratified(ds1["labels"], PER_CLASS_DS1, rng)
    e80 = [int(i) for i in select_events(ds2["labels"], ds2["record_ids"])]
    te_idx = sorted(set(stratified(ds2["labels"], PER_CLASS_DS2, rng)) | set(e80))
    results = {"n_train_ds1": len(tr_idx), "n_test_ds2": len(te_idx),
               "rr_gaps_capped": {"ds1": ds1["n_rr_capped"], "ds2": ds2["n_rr_capped"]},
               "item6": {}, "item7": {}}
    print(f"RR gaps > 12 s capped: DS1 {ds1['n_rr_capped']}, DS2 {ds2['n_rr_capped']}")

    print(f"Replaying DS1 ({len(tr_idx)}) and DS2 ({len(te_idx)}) with the reference encoder...")
    xtr, ftr = collect(REFERENCE, ds1, tr_idx)
    xte, fte = collect(REFERENCE, ds2, te_idx)
    results["item6"]["reference_context32"], tier_models = decode(xtr, ftr, xte, fte)

    for name, path in ADAPTERS.items():
        ttr, info = adapter_tokens(path, xtr)
        tte, _ = adapter_tokens(path, xte)
        res, _ = decode(ttr, ftr, tte, fte, reduce=True)
        results["item6"][f"adapter_{name}"] = {"projection": info, "decoding": res}
        print(f"adapter {name}: rank {info['rank']}, cond {info['condition_number']:.1f}")

    print("Replaying with the RR encoder...")
    rtr, rftr = collect(RR_ENCODER, ds1, tr_idx)
    rte, rfte = collect(RR_ENCODER, ds2, te_idx)
    results["item6"]["rr_encoder_context32"], _ = decode(rtr, rftr, rte, rfte)

    # ---- item 7 ----
    maha = mahalanobis_scorer(xtr, ftr["label"])
    tier_mlp = tier_models["mlp"]
    scores = {
        "mahalanobis": maha,
        "head_margin": head_margin_score,
        "tier_decoder": lambda x: 1.0 - tier_mlp.predict_proba(x).max(axis=1),
    }
    clean = ds2["features"][te_idx]
    clean_ctx = raw_context(clean)
    corr = {}
    for kind in ("gaussian_noise_0dB", "baseline_wander", "clipping", "segment_dropout", "powerline_60Hz"):
        bad_ctx = raw_context(corrupt(clean, kind, rng))
        corr[kind] = {name: float(roc_auc_score(np.r_[np.zeros(len(clean_ctx)), np.ones(len(bad_ctx))],
                                                np.r_[fn(clean_ctx), fn(bad_ctx)]))
                      for name, fn in scores.items()}
    results["item7"]["corruption_auroc"] = corr

    with open(STEP2, encoding="utf-8") as f:
        step2 = json.load(f)
    pos = {e: i for i, e in enumerate(te_idx)}
    e80_ctx = xte[[pos[i] for i in e80]]
    err = {}
    for ck, c in step2["checkpoints"].items():
        wrong = {w["idx"] for w in c["wrong_events"]}
        y = np.array([i in wrong for i in e80])
        err[ck] = {"n_wrong": int(y.sum())}
        for name, fn in scores.items():
            err[ck][name] = float(roc_auc_score(y, fn(e80_ctx))) if 0 < y.sum() < len(y) else None
    results["item7"]["arm_b_error_auroc_e80"] = err

    # Affine-invariance check: Mahalanobis on the headline adapter's tokens (PCA-32) vs on the input.
    ttr, _ = adapter_tokens(ADAPTERS["headline"], xtr)
    tte, _ = adapter_tokens(ADAPTERS["headline"], xte)
    pca = PCA(n_components=32, random_state=0).fit(ttr)
    tok_score = mahalanobis_scorer(pca.transform(ttr), ftr["label"])(pca.transform(tte))
    results["item7"]["mahalanobis_tokens_vs_input_spearman"] = float(
        np.corrcoef(np.argsort(np.argsort(tok_score)), np.argsort(np.argsort(maha(xte))))[0, 1])

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    def show(title, res):
        print(f"\n{title}")
        for field in CLASSIFY:
            r = res[field]
            print(f"  {field:11s} " + ("skipped" if "skipped" in r else
                  f"lin acc {r['linear']['accuracy']:.3f} (bal {r['linear']['balanced_accuracy']:.3f})  "
                  f"mlp acc {r['mlp']['accuracy']:.3f} (bal {r['mlp']['balanced_accuracy']:.3f})"))
        for field in REGRESS:
            r = res[field]
            print(f"  {field:11s} lin R2 {r['linear']['r2']:+.3f}  mlp R2 {r['mlp']['r2']:+.3f}")
    show("Reference context-32 (adapter input)", results["item6"]["reference_context32"])
    show("Headline adapter tokens (PCA-32)", results["item6"]["adapter_headline"]["decoding"])
    show("RR encoder context-32", results["item6"]["rr_encoder_context32"])
    print("\nCorruption AUROC:", json.dumps(corr, indent=1))
    print("Arm B error AUROC (E80):", json.dumps(err, indent=1))
    print("Mahalanobis tokens vs input rank corr:", results["item7"]["mahalanobis_tokens_vs_input_spearman"])


if __name__ == "__main__":
    main()
