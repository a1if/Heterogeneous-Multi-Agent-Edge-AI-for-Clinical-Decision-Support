"""Deviation 22 analysis: the r4 protocol on the second sender (ResNet1D-RR), RQ1-RQ3. CPU.

Must run with the sender selected:  P1_ENCODER=perception/checkpoints/resnet1d_rr_seed0.pt P1_ENCODER_TAG=_res
Inputs (all produced by scripts/run_second_sender_eval.sh):
  results/p1_item7_eval_ds2v2_r4_res.json      MEA r4_res seeds 101/202/303, A-compact, A-filtered
  results/p1_item7_compact_logits_res.json     tier logits, val + ds2v2 (Deviation 17 calibration)
  results/p1_item7_filtered_logits_res.json
  results/p1_item7_filtered_tokens_res.json    prompt tokens per window
  results/p1_item7_filtered_timing_res.json    TTFT / time to decision
Output: results/p1_second_sender_analysis.json. Sections whose inputs are missing are skipped.

  RQ2  balanced accuracy by N; MEA 3-seed mean minus each text arm (default and calibrated by the
       Deviation 17 primary rule), record-level 95% CIs; false alarms (all / natural routine windows)
  RQ1  prompt tokens (median) and timing (median paired relative difference, MEA vs each text arm)
  RQ3  probe recoverability of the sender's context vector (p1_rr_probe protocol) and per-slot
       decodability (slots 0 and 49) of label, tier, heart rate, RR (R^2) and run >= 3 (balanced
       accuracy) from each seed's adapter tokens, Deviation 18 / 18b methods

Run (from repo root, with the environment above):
    python p1_second_sender_analysis.py
"""
import json
import os
from pathlib import Path

import numpy as np

from p1_io import save_json_atomic

TAG = os.environ.get("P1_ENCODER_TAG", "")
assert TAG == "_res", "run with P1_ENCODER and P1_ENCODER_TAG=_res set (see docstring)"
EVAL = Path(f"results/p1_item7_eval_ds2v2_r4{TAG}.json")
LOGITS = {"A-compact": Path(f"results/p1_item7_compact_logits{TAG}.json"),
          "A-filtered": Path(f"results/p1_item7_filtered_logits{TAG}.json")}
TOKENS = Path(f"results/p1_item7_filtered_tokens{TAG}.json")
TIMING = Path(f"results/p1_item7_filtered_timing{TAG}.json")
OUT = Path("results/p1_second_sender_analysis.json")
MEA = tuple(f"MEA:r4{TAG}_seed{s}" for s in (101, 202, 303))


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def calibrate(rows, arm, logits_rows):
    """Deviation 17 primary rule on the validation logits; returns (delta, rows with arm '<arm>-cal')."""
    import p1_item7_tier1 as t1
    from p1_item7_calib import GRID, answer
    val = [x for x in logits_rows if x["split"] == "val"]
    curve = [(float(d), t1.bal([(x["reference"], answer(x["logits"], d)) for x in val])) for d in GRID]
    delta = max(curve, key=lambda c: (round(c[1], 6), c[0]))[0]
    ds2 = {(x["set"], x["n"], x["start"]): x["logits"] for x in logits_rows if x["split"] == "ds2v2"}
    out = []
    for x in rows:
        if x["arm"] == arm and (x["set"], x["n"], x["start"]) in ds2:
            t = answer(ds2[(x["set"], x["n"], x["start"])], delta)
            out.append({**x, "arm": f"{arm}-cal", "tier": t, "correct": t == x["reference"]})
    return delta, dict(curve)[delta], out


def rq2(rows, rec):
    import p1_item7_tier1 as t1
    res = {"by_n": {}, "comparisons": {}, "false_alarm": {}}
    bases = sorted({x["arm"] for x in rows if not x["arm"].startswith("MEA")})
    for n, wins in sorted(t1.windows_by_n(rows).items()):
        d = {}
        for a in MEA + tuple(bases):
            pairs = [(w["reference"], w["arms"][a]) for w in wins.values() if a in w["arms"]]
            d[a] = t1.bal(pairs) if pairs else None
        d["MEA_mean"] = float(np.mean([d[m] for m in MEA]))
        res["by_n"][str(n)] = d
        if n >= 5:
            for b in bases:
                starts = [s for s, w in wins.items() if all(a in w["arms"] for a in MEA + (b,))]
                if starts:
                    sub = {s: wins[s] for s in starts}
                    res["comparisons"].setdefault(b, {})[str(n)] = t1.cluster_ci(sub, list(MEA), {s: int(rec[s]) for s in starts}, base=b)
    for a in MEA + tuple(bases):
        rt = [x for x in rows if x["arm"] == a and x["reference"] == "routine"]
        nat = [x for x in rt if x["set"] == "natural"]
        res["false_alarm"][a] = {"all": float(np.mean([x["tier"] != "routine" for x in rt])) if rt else None,
                                 "natural": float(np.mean([x["tier"] != "routine" for x in nat])) if nat else None}
    res["parse_rate"] = {a: float(np.mean([x["parsed"] for x in rows if x["arm"] == a])) for a in MEA + tuple(bases)
                         if not a.endswith("-cal")}
    return res


def rq1():
    out = {}
    tk = load(TOKENS)
    if tk:
        out["prompt_tokens_median"] = {str(n): {a: float(np.median([x[a] for x in tk["rows"] if x["n"] == n]))
                                                for a in ("A-compact", "A-filtered", "MEA")}
                                       for n in sorted({x["n"] for x in tk["rows"]})}
    tm = load(TIMING)
    if tm:
        rng = np.random.default_rng(0)
        out["timing"] = {}
        for n in sorted({x["n"] for x in tm["rows"]}):
            by = {}
            for x in tm["rows"]:
                if x["n"] == n:
                    by.setdefault(x["start"], {})[x["arm"]] = x
            trip = [v for v in by.values() if len(v) == 3]
            d = {"n_windows": len(trip)}
            for m in ("ttft_ms", "ttd_ms"):
                d[m] = {a: float(np.median([v[a][m] for v in trip])) for a in ("A-compact", "A-filtered", "MEA")}
                for b in ("A-compact", "A-filtered"):
                    rel = np.array([v["MEA"][m] / v[b][m] - 1 for v in trip])
                    boots = [np.median(rel[rng.integers(0, len(rel), len(rel))]) for _ in range(20000)]
                    d[m][f"MEA_vs_{b}"] = {"median_rel": float(np.median(rel)),
                                           "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
            out["timing"][str(n)] = d
    return out


def rq3():
    import torch
    import p1_item7_slot_decoder as sd
    import p1_items6_7_decoder_anomaly as item6
    from day7_auditability_probe import DS2_PATH, select_events
    from p1_item7_common import RR_ENCODER, replay_split, window_vectors
    from p1_item7_runlen import run_fields
    from p1_rr_seeds import probe
    out = {}
    with np.load(DS2_PATH) as z:
        X, y, rr, record_ids = (z[k] for k in ("features", "labels", "rr_interval_ms", "record_ids"))
    selected = [int(i) for i in select_events(y, record_ids)]
    out["probe_recoverability"] = dict(zip(("mean", "sd"), probe(RR_ENCODER, X, rr, record_ids, selected, y[selected])))
    r1, r2 = replay_split("ds1"), replay_split("ds2")
    rng = np.random.default_rng(0)
    s1 = item6.stratified(np.asarray(r1["labels"]), item6.PER_CLASS_DS1, rng)
    s2 = item6.stratified(np.asarray(r2["labels"]), item6.PER_CLASS_DS2, rng)
    f1 = {**sd.fields_from_events([r1["events"][i] for i in s1]), **run_fields([r1["events"][i] for i in s1])}
    f2 = {**sd.fields_from_events([r2["events"][i] for i in s2]), **run_fields([r2["events"][i] for i in s2])}
    x1 = np.concatenate([window_vectors(r1, i, 1, 35) for i in s1])
    x2 = np.concatenate([window_vectors(r2, i, 1, 35) for i in s2])
    item6.CLASSIFY, item6.REGRESS = ("label", "tier", "run3"), ("heart_rate", "rr")
    out["input_35d"] = item6.decode(x1, f1, x2, f2, reduce=False)[0]
    for m in MEA:
        state = torch.load(f"reasoning/checkpoints/p1_item7_mea_{m.split(':')[1]}.pt", map_location="cpu",
                           weights_only=False)["adapter_state_dict"]
        for slot in (0, 49):
            out[f"{m.split(':')[1]}_slot{slot}"] = item6.decode(sd.slot_tokens(state, x1, slot), f1,
                                                                sd.slot_tokens(state, x2, slot), f2, reduce=True)[0]
    return out


def main():
    from p1_item7_common import replay_split
    rec = np.asarray(replay_split("ds2")["record_ids"])
    out = {"design": __doc__}
    ev = load(EVAL)
    rows = ev["rows"]
    out["calibration"] = {}
    for arm, path in LOGITS.items():
        lg = load(path)
        if lg and any(x["arm"] == arm for x in rows):
            delta, val_ba, cal = calibrate(rows, arm, lg["rows"])
            out["calibration"][arm] = {"delta": delta, "val_bal_acc": val_ba}
            rows = rows + cal
    out["RQ2"] = rq2(rows, rec)
    out["RQ1"] = rq1()
    save_json_atomic(OUT, out)
    out["RQ3"] = rq3()
    save_json_atomic(OUT, out)
    show(out)


def show(out):
    f = lambda d: f"{d['difference']:+.3f} [{d['cluster_ci95'][0]:+.3f}, {d['cluster_ci95'][1]:+.3f}]"
    print("calibration:", out["calibration"])
    for n, d in out["RQ2"]["by_n"].items():
        line = f"N={n:>2} MEA {d['MEA_mean']:.3f} " + " ".join(f"{a} {v:.3f}" for a, v in d.items()
                                                               if not a.startswith("MEA") and v is not None)
        cmp = {b: c[n] for b, c in out["RQ2"]["comparisons"].items() if n in c}
        print(line + ("  || " + "  ".join(f"vs {b} {f(c)}" for b, c in cmp.items()) if cmp else ""))
    print("false alarms:", {a: {k: (round(v * 100, 1) if v is not None else None) for k, v in d.items()}
                            for a, d in out["RQ2"]["false_alarm"].items()})
    print("RQ1:", json.dumps(out["RQ1"].get("prompt_tokens_median")))
    for n, d in out["RQ1"].get("timing", {}).items():
        print(f"  N={n} TTFT {d['ttft_ms']}")
    r3 = out.get("RQ3", {})
    if r3:
        print("probe recoverability:", r3["probe_recoverability"])
        for k, d in r3.items():
            if isinstance(d, dict) and "label" in d:
                print(f"  {k:22s} label {d['label']['mlp']['balanced_accuracy']:.2f} tier {d['tier']['mlp']['balanced_accuracy']:.2f} "
                      f"run>=3 {d['run3']['mlp']['balanced_accuracy']:.2f} | R2 hr {d['heart_rate']['mlp']['r2']:.2f} rr {d['rr']['mlp']['r2']:.2f}")


if __name__ == "__main__":
    main()
