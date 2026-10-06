"""Deviations 26 and 27: three-seed group comparisons on DS2 v2 (CPU).

  --dev 27  three-seed compression comparison: k = 1 and k = 2 (recipes r4k1, r4k2; seeds 101/202/303)
            against k = 4 (r4 seeds 101/202/303). Primary: group-mean balanced accuracy difference at
            N = 5, 10, 20, 50 with record-cluster bootstrap 95% CI; non-inferior if the lower bound > -0.05.
  --dev 26  r4 ablation: r4hn (hard negatives, no side inputs) against r4 (side-input contribution) and
            r3 against r4hn (hard-negative contribution; approximate, see Deviation 26).
Both report, per group: balanced accuracy by N (mean and seed range), false alarms (all / natural routine
windows), parse rate, and per-slot decodability (slots 0 and 49; Deviation 18 / 18b methods) per seed.
Rows: results/p1_item7_eval_ds2v2.json (r3), results/p1_item7_eval_ds2v2_r4.json (r4),
results/p1_item7_eval_ds2v2_sweep.json (k = 1, 2), results/p1_item7_eval_ds2v2_ablation.json (r4hn).

Run (from repo root):
    python p1_seed_group_analysis.py --dev 27      # -> results/p1_dev27_kseeds.json
    python p1_seed_group_analysis.py --dev 26      # -> results/p1_dev26_ablation.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from p1_io import save_json_atomic

SEEDS = (101, 202, 303)
FILES = ("results/p1_item7_eval_ds2v2.json", "results/p1_item7_eval_ds2v2_r4.json",
         "results/p1_item7_eval_ds2v2_sweep.json", "results/p1_item7_eval_ds2v2_ablation.json",
         "results/p1_item7_eval_ds2v2_scaling.json")
NS = (5, 10, 20, 50)
MARGIN = -0.05


def group(recipe):
    return [f"MEA:{recipe}_seed{s}" for s in SEEDS]


def load_rows():
    rows, seen = [], set()
    for f in FILES:
        if not Path(f).exists():
            continue
        for x in json.loads(Path(f).read_text(encoding="utf-8"))["rows"]:
            key = (x["arm"], x["set"], x["n"], x["start"])
            if x["arm"].startswith("MEA:") and key not in seen:  # each adapter arm once
                seen.add(key)
                rows.append(x)
    return rows


def group_ci(wins, a_arms, b_arms, record_of, n_boot=20000, seed=0):
    """Mean balanced accuracy of group a minus group b on windows every arm answered; record-cluster bootstrap."""
    import p1_item7_tier1 as t1
    rng = np.random.default_rng(seed)
    starts = [s for s, w in wins.items() if all(a in w["arms"] for a in a_arms + b_arms)]
    recs = sorted({record_of[s] for s in starts})
    by_rec = {r: [s for s in starts if record_of[s] == r] for r in recs}

    def diff(sel):
        ba = lambda arm: t1.bal([(wins[s]["reference"], wins[s]["arms"][arm]) for s in sel])
        return np.mean([ba(m) for m in a_arms]) - np.mean([ba(m) for m in b_arms])

    point = float(diff(starts))
    boots = [d for d in (diff([s for r in rng.choice(recs, len(recs)) for s in by_rec[r]]) for _ in range(n_boot))
             if np.isfinite(d)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"difference": point, "cluster_ci95": [float(lo), float(hi)], "n_windows": len(starts),
            "n_records": len(recs), "non_inferior": bool(lo > MARGIN), "superior": bool(lo > 0)}


def describe(rows, wins, arms):
    import p1_item7_tier1 as t1
    out = {"balanced_accuracy": {}, "false_alarm": {}, "parse_rate": None}
    for n, w in sorted(wins.items()):
        per = [t1.bal([(v["reference"], v["arms"][a]) for v in w.values() if a in v["arms"]]) for a in arms]
        out["balanced_accuracy"][str(n)] = {"mean": float(np.mean(per)), "range": [float(min(per)), float(max(per))]}
    for a in arms:
        rt = [x for x in rows if x["arm"] == a and x["reference"] == "routine"]
        nat = [x for x in rt if x["set"] == "natural"]
        out["false_alarm"][a] = {"all": float(np.mean([x["tier"] != "routine" for x in rt])),
                                 "natural": float(np.mean([x["tier"] != "routine" for x in nat])) if nat else None}
    out["parse_rate"] = float(np.mean([x["parsed"] for x in rows if x["arm"] in arms]))
    return out


def decodability(arms):
    """Deviation 18 / 18b per-slot decoders (slots 0 and 49) for each adapter checkpoint."""
    import torch
    import p1_item7_slot_decoder as sd
    import p1_items6_7_decoder_anomaly as item6
    from p1_item7_common import replay_split, window_vectors
    from p1_item7_runlen import run_fields
    r1, r2 = replay_split("ds1"), replay_split("ds2")
    rng = np.random.default_rng(0)
    s1 = item6.stratified(np.asarray(r1["labels"]), item6.PER_CLASS_DS1, rng)
    s2 = item6.stratified(np.asarray(r2["labels"]), item6.PER_CLASS_DS2, rng)
    f1 = {**sd.fields_from_events([r1["events"][i] for i in s1]), **run_fields([r1["events"][i] for i in s1])}
    f2 = {**sd.fields_from_events([r2["events"][i] for i in s2]), **run_fields([r2["events"][i] for i in s2])}
    item6.CLASSIFY, item6.REGRESS = ("label", "tier", "run3"), ("heart_rate", "rr")
    out = {}
    for arm in arms:
        state = torch.load(f"reasoning/checkpoints/p1_item7_mea_{arm.split(':')[1]}.pt", map_location="cpu",
                           weights_only=False)["adapter_state_dict"]
        dim = state["projection.weight"].shape[1]
        x1 = np.concatenate([window_vectors(r1, i, 1, dim) for i in s1])
        x2 = np.concatenate([window_vectors(r2, i, 1, dim) for i in s2])
        res = {}
        for slot in (0, 49):
            d = item6.decode(sd.slot_tokens(state, x1, slot), f1, sd.slot_tokens(state, x2, slot), f2, reduce=True)[0]
            res[f"slot{slot}"] = {"label": d["label"]["mlp"]["balanced_accuracy"], "tier": d["tier"]["mlp"]["balanced_accuracy"],
                                  "run3": d["run3"]["mlp"]["balanced_accuracy"], "heart_rate_r2": d["heart_rate"]["mlp"]["r2"],
                                  "rr_r2": d["rr"]["mlp"]["r2"]}
        out[arm] = res
    return out


def main(dev):
    import p1_item7_tier1 as t1
    from p1_item7_common import replay_split
    rec = np.asarray(replay_split("ds2")["record_ids"])
    rows = load_rows()
    wins = t1.windows_by_n(rows)
    record_of = {s: int(rec[s]) for w in wins.values() for s in w}
    if dev == 29:
        groups = {"d25": group("r4d25"), "d50": group("r4d50"), "r4": group("r4"), "dmax": group("r4dmax")}
        contrasts = {"d25_minus_r4": ("d25", "r4"), "d50_minus_r4": ("d50", "r4"), "dmax_minus_r4": ("dmax", "r4")}
        out_path = Path("results/p1_dev29_scaling.json")
    elif dev == 27:
        groups = {"k1": group("r4k1"), "k2": group("r4k2"), "k4": group("r4")}
        contrasts = {"k1_minus_k4": ("k1", "k4"), "k2_minus_k4": ("k2", "k4")}
        out_path = Path("results/p1_dev27_kseeds.json")
    else:
        groups = {"r3": group("r3"), "r4hn": group("r4hn"), "r4": group("r4")}
        contrasts = {"side_inputs_r4_minus_r4hn": ("r4", "r4hn"), "hard_negatives_r4hn_minus_r3": ("r4hn", "r3")}
        out_path = Path("results/p1_dev26_ablation.json")
    present = {a for x in rows for a in [x["arm"]]}
    missing = [a for g in groups.values() for a in g if a not in present]
    if missing:
        raise SystemExit(f"missing arms: {missing}")
    out = {"design": __doc__, "deviation": dev, "groups": groups,
           "describe": {g: describe(rows, wins, arms) for g, arms in groups.items()}, "contrasts": {}}
    for name, (a, b) in contrasts.items():
        out["contrasts"][name] = {str(n): group_ci(wins[n], groups[a], groups[b], record_of) for n in NS if n in wins}
    if dev == 27:
        out["no_accuracy_cost"] = {name: all(v["non_inferior"] for v in c.values()) for name, c in out["contrasts"].items()}
    if dev != 29:
        out["decodability"] = decodability([a for g in groups.values() for a in g if not (dev == 26 and g == groups["r3"])])
    save_json_atomic(out_path, out)
    for name, c in out["contrasts"].items():
        print(name, {n: f"{v['difference']:+.3f} [{v['cluster_ci95'][0]:+.3f}, {v['cluster_ci95'][1]:+.3f}]" for n, v in c.items()})
    for g, d in out["describe"].items():
        fa = [v["all"] for v in d["false_alarm"].values()]
        print(g, {n: round(v["mean"], 3) for n, v in d["balanced_accuracy"].items()}, f"FA {np.mean(fa):.3f} [{min(fa):.3f}, {max(fa):.3f}]")
    if dev == 27:
        print("no accuracy cost (non-inferior at N = 5, 10, 20, 50):", out["no_accuracy_cost"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, choices=(26, 27, 29), required=True)
    main(ap.parse_args().dev)
