"""Deviation 19: the RR-branch encoder across 10 seeds (0-9). CPU.

Collects each seed's DS2 metrics (written by train_perception_agent_rr.py) and scores
each seed's context vectors with the p1_rr_probe protocol (E80 events, chronological
replay, StandardScaler + logistic regression, RepeatedStratifiedKFold 5x3,
random_state 42). Applies the decision rules fixed in docs/analysis_plan.md
(Deviation 19). Resumable: probe scores are saved per seed.

Run (from repo root), after training seeds 1-9:
    python p1_rr_seeds.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from day7_auditability_probe import DS2_PATH, N_FOLDS, N_REPEATS, select_events
from p1_io import save_json_atomic
from p1_rr_probe import pipeline
from perception.perception_agent import PerceptionAgent, replay_selected

SEEDS = range(10)
OUT = Path("results/p1_rr_seeds.json")
CLASSES = ("N", "S", "V", "F")


def ds2_metrics(seed):
    path = Path("results/p1_rr_encoder_results.json" if seed == 0 else f"results/p1_rr_encoder_results_seed{seed}.json")
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["seed"] == seed, (path, d["seed"])
    return d["ds2"]


def probe(checkpoint, X, rr, record_ids, selected, labels):
    agent = PerceptionAgent(checkpoint_path=checkpoint, device="cpu")
    replayed = replay_selected(agent, X, rr, record_ids, selected)
    vectors = np.stack([replayed[i][1] for i in selected])
    cv = RepeatedStratifiedKFold(n_splits=N_FOLDS, n_repeats=N_REPEATS, random_state=42)
    scores = cross_val_score(pipeline(), vectors, labels, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def summary(values):
    v = np.asarray(values, dtype=float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "min": float(v.min()), "max": float(v.max())}


def main():
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"design": __doc__, "seeds": {}}
    with np.load(DS2_PATH) as npz:
        X, y, rr, record_ids = (npz[k] for k in ("features", "labels", "rr_interval_ms", "record_ids"))
    selected = [int(i) for i in select_events(y, record_ids)]
    labels = y[selected]
    ref_probe = json.loads(Path("results/p1_rr_probe_results.json").read_text(encoding="utf-8"))["encoders"]["reference_cnn_lstm"]
    for seed in SEEDS:
        if str(seed) in out["seeds"]:
            continue
        m = ds2_metrics(seed)
        p_mean, p_sd = probe(f"perception/checkpoints/cnn_lstm_rr_seed{seed}.pt", X, rr, record_ids, selected, labels)
        out["seeds"][str(seed)] = {"ds2": m["rr_encoder"], "probe_mean": p_mean, "probe_sd": p_sd}
        out["reference"] = {"ds2": m["reference_cnn_lstm"], "probe_mean": ref_probe["probe_accuracy_mean"]}
        save_json_atomic(OUT, out)
        print(f"seed {seed}: DS2 acc {m['rr_encoder']['accuracy']:.4f}  S Se {m['rr_encoder']['per_class']['S']['se']:.3f}  "
              f"probe {p_mean:.3f}", flush=True)

    S, ref = out["seeds"], out["reference"]
    metrics = {"ds2_accuracy": (lambda s: s["ds2"]["accuracy"], ref["ds2"]["accuracy"]),
               "probe_accuracy": (lambda s: s["probe_mean"], ref["probe_mean"])}
    for c in CLASSES:
        metrics[f"se_{c}"] = (lambda s, c=c: s["ds2"]["per_class"][c]["se"], ref["ds2"]["per_class"][c]["se"])
        metrics[f"ppv_{c}"] = (lambda s, c=c: s["ds2"]["per_class"][c]["ppv"] or 0.0, ref["ds2"]["per_class"][c]["ppv"] or 0.0)
    agg = {}
    for name, (get, ref_value) in metrics.items():
        vals = {k: get(v) for k, v in S.items()}
        rank = 1 + sum(v < vals["0"] for v in vals.values())  # 1 = lowest
        agg[name] = {**summary(list(vals.values())), "reference": ref_value, "seed0": vals["0"], "seed0_rank_of_10": rank,
                     "seeds_above_reference": int(sum(v > ref_value for v in vals.values())), "per_seed": vals}
    out["aggregate"] = agg
    out["decisions"] = {
        "robust_improvement": {k: agg[k]["seeds_above_reference"] >= 9 for k in agg},
        "seed0_representative": {k: 2 <= agg[k]["seed0_rank_of_10"] <= 9 for k in ("ds2_accuracy", "se_S", "probe_accuracy")},
    }
    save_json_atomic(OUT, out)
    for k, a in agg.items():
        print(f"{k:15s} mean {a['mean']:.3f} ± {a['sd']:.3f} [{a['min']:.3f}, {a['max']:.3f}]  ref {a['reference']:.3f}  "
              f"seeds>ref {a['seeds_above_reference']}/10  seed0 {a['seed0']:.3f} (rank {a['seed0_rank_of_10']})")
    print("decisions:", json.dumps(out["decisions"]))


if __name__ == "__main__":
    main()
