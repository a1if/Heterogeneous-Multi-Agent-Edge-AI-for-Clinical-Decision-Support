"""Deviation 24 analysis: compression sweep, k virtual tokens per event (1, 2, 4, 8), seed 101. CPU.

Inputs: results/p1_item7_eval_ds2v2_sweep.json (k = 1, 2, 8), results/p1_item7_eval_ds2v2_r4.json (k = 4 =
r4 seed 101, plus A-compact / A-filtered references), the reasoning/checkpoints/p1_item7_mea_r4k*_seed101.pt
adapters and r4 seed 101 for per-slot decodability. Output: results/p1_sweep_analysis.json.
Per k: balanced accuracy by N, parse rate, false alarms, prompt tokens (scaffold + kN, from the MEA token
count at k = 4), and decodability at slots 0 and 49 of label, tier, run >= 3 (balanced accuracy) and heart
rate, RR (R^2). The r4 seed spread at k = 4 is reported as the noise band. Descriptive (no tests).

Run (from repo root):
    python p1_sweep_analysis.py
"""
import json
from pathlib import Path

import numpy as np

from p1_io import save_json_atomic

OUT = Path("results/p1_sweep_analysis.json")
ARMS = {1: "MEA:r4k1_seed101", 2: "MEA:r4k2_seed101", 4: "MEA:r4_seed101", 8: "MEA:r4k8_seed101"}


def main():
    import torch
    import p1_item7_slot_decoder as sd
    import p1_item7_tier1 as t1
    import p1_items6_7_decoder_anomaly as item6
    from p1_item7_common import replay_split, window_vectors
    from p1_item7_runlen import run_fields
    rows = [x for x in json.loads(Path("results/p1_item7_eval_ds2v2_sweep.json").read_text(encoding="utf-8"))["rows"]
            if x["arm"].startswith("MEA")]
    base = json.loads(Path("results/p1_item7_eval_ds2v2_r4.json").read_text(encoding="utf-8"))["rows"]
    rows += [x for x in base if x["arm"] in ("MEA:r4_seed101", "A-compact", "A-filtered")]
    seeds_r4 = [x for x in base if x["arm"] in ("MEA:r4_seed101", "MEA:r4_seed202", "MEA:r4_seed303")]
    tk = json.loads(Path("results/p1_item7_filtered_tokens.json").read_text(encoding="utf-8"))["rows"]
    out = {"design": __doc__, "by_k": {}, "references": {}, "r4_seed_band": {}}
    wins = t1.windows_by_n(rows)
    wins_r4 = t1.windows_by_n(seeds_r4)
    for n, w in sorted(wins.items()):
        for ref in ("A-compact", "A-filtered"):
            out["references"].setdefault(ref, {})[str(n)] = t1.bal([(v["reference"], v["arms"][ref]) for v in w.values() if ref in v["arms"]])
        band = [t1.bal([(v["reference"], v["arms"][a]) for v in wins_r4[n].values() if a in v["arms"]])
                for a in ("MEA:r4_seed101", "MEA:r4_seed202", "MEA:r4_seed303")]
        out["r4_seed_band"][str(n)] = [float(min(band)), float(max(band))]
    for k, arm in ARMS.items():
        xs = [x for x in rows if x["arm"] == arm]
        if not xs:
            continue
        rt = [x for x in xs if x["reference"] == "routine"]
        d = {"balanced_accuracy": {str(n): t1.bal([(v["reference"], v["arms"][arm]) for v in w.values() if arm in v["arms"]])
                                   for n, w in sorted(wins.items())},
             "parse_rate": float(np.mean([x["parsed"] for x in xs])),
             "false_alarm": float(np.mean([x["tier"] != "routine" for x in rt])),
             "prompt_tokens": {str(n): float(np.median([x["MEA"] - 4 * n + k * n for x in tk if x["n"] == n]))
                               for n in sorted({x["n"] for x in tk})}}
        out["by_k"][str(k)] = d
    # per-slot decodability (Deviation 18 / 18b methods)
    r1, r2 = replay_split("ds1"), replay_split("ds2")
    rng = np.random.default_rng(0)
    s1 = item6.stratified(np.asarray(r1["labels"]), item6.PER_CLASS_DS1, rng)
    s2 = item6.stratified(np.asarray(r2["labels"]), item6.PER_CLASS_DS2, rng)
    f1 = {**sd.fields_from_events([r1["events"][i] for i in s1]), **run_fields([r1["events"][i] for i in s1])}
    f2 = {**sd.fields_from_events([r2["events"][i] for i in s2]), **run_fields([r2["events"][i] for i in s2])}
    x1 = np.concatenate([window_vectors(r1, i, 1, 35) for i in s1])
    x2 = np.concatenate([window_vectors(r2, i, 1, 35) for i in s2])
    item6.CLASSIFY, item6.REGRESS = ("label", "tier", "run3"), ("heart_rate", "rr")
    for k, arm in ARMS.items():
        path = Path(f"reasoning/checkpoints/p1_item7_mea_{arm.split(':')[1]}.pt")
        if not path.exists() or str(k) not in out["by_k"]:
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)["adapter_state_dict"]
        out["by_k"][str(k)]["decodability"] = {
            f"slot{slot}": item6.decode(sd.slot_tokens(state, x1, slot), f1, sd.slot_tokens(state, x2, slot), f2,
                                        reduce=True)[0] for slot in (0, 49)}
    save_json_atomic(OUT, out)
    for k, d in out["by_k"].items():
        dec = d.get("decodability", {}).get("slot0", {})
        print(f"k={k} bal-acc {({n: round(v, 3) for n, v in d['balanced_accuracy'].items()})} FA {d['false_alarm']:.3f} "
              f"tokens@50 {d['prompt_tokens'].get('50')}"
              + (f" | slot0 label {dec['label']['mlp']['balanced_accuracy']:.2f} tier {dec['tier']['mlp']['balanced_accuracy']:.2f} "
                 f"run3 {dec['run3']['mlp']['balanced_accuracy']:.2f} hr {dec['heart_rate']['mlp']['r2']:.2f} rr {dec['rr']['mlp']['r2']:.2f}"
                 if dec else ""))
    print("r4 seed band:", out["r4_seed_band"])


if __name__ == "__main__":
    main()
