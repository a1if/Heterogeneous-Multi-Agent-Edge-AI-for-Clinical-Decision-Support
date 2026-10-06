"""Item 7 secondary endpoint (Deviation 9): per-event auditability of the multi-event
adapter (MEA), by the item 6 inverse-decoder method. CPU only.

The MEA maps event i of a window to slot i's k = 4 tokens:
    tokens_i = normalize(W v_i + b + position_i) * scale
so slot i depends only on event i's 32-d context vector v_i and the slot's fixed
position embedding (independence is by construction and unit-tested in
tests/test_multi_event_adapter.py). The question here is whether what an auditor
needs per event survives that map, at any slot of a 50-event window.

For the three recipe-r3 checkpoints (50 slots), a sample of beats is placed at slots
0, 9, 19 and 49; decoders are trained on DS1 and tested on DS2 (inter-patient,
item 6's sample sizes: 1,500 / 500 beats per true class N, S, V, F), with item 6's
decode() (standardise, PCA-32 for tokens, linear and MLP). Fields: predicted label,
urgency tier, urgent flag, confidence, top-2 margin, heart rate, RR, run length.
Reference: decoding from the 32-d input vector itself (the upper bound).
Cross-slot: a decoder trained on slot-0 tokens (DS1) applied to slot-49 tokens (DS2)
tests whether one auditor decoder works at every position.

Events and vectors come from the cached RR-encoder replays (p1_item7_common).

Run (from repo root):
    python p1_item7_slot_decoder.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import p1_items6_7_decoder_anomaly as item6
from p1_io import save_json_atomic
from reasoning.training_targets import urgency_tier_from_event

CHECKPOINTS = {s: f"reasoning/checkpoints/p1_item7_mea_r3_seed{s}.pt" for s in (101, 202, 303)}
SLOTS = (0, 9, 19, 49)
CLASSIFY = ("label", "tier", "urgent")
REGRESS = ("confidence", "margin", "heart_rate", "rr", "run")
RESULTS = Path("results/p1_item7_slot_decoder.json")


def fields_from_events(events):
    f = {k: [] for k in CLASSIFY + REGRESS}
    for ev in events:
        c, sf = ev["classification"], ev["signal_features"]
        f["label"].append(c["label"])
        f["tier"].append(urgency_tier_from_event(ev))
        f["urgent"].append(ev["clinical_flags"]["requires_urgent_review"])
        f["confidence"].append(c["confidence"])
        f["margin"].append(c["top_3"][0]["confidence"] - c["top_3"][1]["confidence"])
        f["heart_rate"].append(sf["heart_rate_bpm"])
        f["rr"].append(sf["rr_interval_ms"])
        f["run"].append(ev["clinical_flags"]["consecutive_abnormal_beats"])
    return {k: np.asarray(v) for k, v in f.items()}


def slot_tokens(state, vectors, slot):
    """(n, 32) -> (n, k * E) tokens for events placed at ``slot`` (MultiEventVirtualAdapter math)."""
    W, b, pos = state["projection.weight"], state["projection.bias"], state["position"]
    k, e = pos.shape[1], pos.shape[2]
    with torch.no_grad():
        x = (torch.from_numpy(vectors).float() @ W.T + b).view(len(vectors), k, e) + pos[slot]
        x = F.normalize(x, dim=-1) * state["log_scale"].exp()
    return x.reshape(len(vectors), -1).numpy()


def decode(train_x, train_f, test_x, test_f, reduce):
    item6.CLASSIFY, item6.REGRESS = CLASSIFY, REGRESS  # item 6's decode() over this field list
    return item6.decode(train_x, train_f, test_x, test_f, reduce=reduce)[0]


def cross_slot(tr_x, tr_f, te_x, te_f):
    out = {}
    for field in ("label", "tier"):
        m = make_pipeline(StandardScaler(), PCA(n_components=32, random_state=0),
                          MLPClassifier(hidden_layer_sizes=(64,), max_iter=2000, random_state=0))
        m.fit(tr_x, tr_f[field].astype(str))
        pred, y = m.predict(te_x), te_f[field].astype(str)
        out[field] = {"accuracy": float((pred == y).mean()), "balanced_accuracy": float(balanced_accuracy_score(y, pred))}
    return out


def main():
    from p1_item7_common import replay_split
    t0 = time.time()
    r1, r2 = replay_split("ds1"), replay_split("ds2")
    rng = np.random.default_rng(0)
    s1 = item6.stratified(np.asarray(r1["labels"]), item6.PER_CLASS_DS1, rng)
    s2 = item6.stratified(np.asarray(r2["labels"]), item6.PER_CLASS_DS2, rng)
    v1, v2 = np.asarray(r1["vectors"])[s1], np.asarray(r2["vectors"])[s2]
    f1, f2 = fields_from_events([r1["events"][i] for i in s1]), fields_from_events([r2["events"][i] for i in s2])
    res = {"design": __doc__, "n_train": len(s1), "n_test": len(s2),
           "input_vector": decode(v1, f1, v2, f2, reduce=False), "checkpoints": {}}
    print(f"input vector decoded ({time.time() - t0:.0f}s)", flush=True)
    for seed, path in CHECKPOINTS.items():
        state = torch.load(path, map_location="cpu", weights_only=False)["adapter_state_dict"]
        ck = {"slots": {}}
        for slot in SLOTS:
            ck["slots"][str(slot)] = decode(slot_tokens(state, v1, slot), f1, slot_tokens(state, v2, slot), f2, reduce=True)
            print(f"seed {seed} slot {slot} ({time.time() - t0:.0f}s)", flush=True)
        ck["cross_slot_0_to_49"] = cross_slot(slot_tokens(state, v1, 0), f1, slot_tokens(state, v2, 49), f2)
        res["checkpoints"][str(seed)] = ck
        save_json_atomic(RESULTS, res)
    res["seconds"] = time.time() - t0
    save_json_atomic(RESULTS, res)
    show(res)


def show(res):
    def row(d):
        c = " ".join(f"{f} {d[f]['mlp']['balanced_accuracy']:.2f}" for f in CLASSIFY)
        r = " ".join(f"{f} {d[f]['mlp']['r2']:.2f}" for f in REGRESS)
        return f"bal-acc [{c}] | R2 mlp [{r}]"
    print("input vector      ", row(res["input_vector"]))
    for seed, ck in res["checkpoints"].items():
        for slot, d in ck["slots"].items():
            print(f"seed {seed} slot {slot:>2}", row(d))
        print(f"seed {seed} cross 0->49", ck["cross_slot_0_to_49"])


if __name__ == "__main__":
    main()
