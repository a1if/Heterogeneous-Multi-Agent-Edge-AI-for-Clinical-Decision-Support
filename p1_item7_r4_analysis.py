"""Item 7, Deviation 18 analysis: recipe r4 (side inputs + hard negatives) vs text and vs r3. CPU.

Inputs: results/p1_item7_eval_ds2v2_r4.json (r4 seeds + A-compact, reused rows),
results/p1_item7_eval_ds2v2.json (r3 seeds), results/p1_item7_calib*.json (text deltas and
tier logits). Outputs results/p1_item7_r4_analysis.json:
  primary      r4 3-seed mean minus A-compact (default and calibrated), record-level CIs
  vs_r3        balanced accuracy per N, r4 vs r3
  false_alarm  on all DS2 routine windows and on natural windows, r3 vs r4 vs text
  urgent       recall on run-based vs confidence-based urgent windows (N >= 10)
  slots        per-slot decodability of heart rate, RR and run length from r4 tokens
               (slots 0 and 49, item 6 method), vs r3 tokens and vs the 35-d input

Run (from repo root):
    python p1_item7_r4_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import torch

import p1_item7_tier1 as t1
from p1_io import save_json_atomic

R4 = Path("results/p1_item7_eval_ds2v2_r4.json")
R3 = Path("results/p1_item7_eval_ds2v2.json")
OUT = Path("results/p1_item7_r4_analysis.json")
TIERS = ("routine", "priority", "urgent")


def arms_of(rows, prefix):
    return sorted({x["arm"] for x in rows if x["arm"].startswith(prefix)})


def primary(rows, mea, rec):
    out = {}
    for n, wins in sorted(t1.windows_by_n(rows).items()):
        starts = [s for s, w in wins.items() if all(a in w["arms"] for a in mea + ["A-compact"])]
        ba = {a: t1.bal([(wins[s]["reference"], wins[s]["arms"][a]) for s in starts]) for a in mea + ["A-compact"]}
        d = {"text": ba["A-compact"], "mea_mean": float(np.mean([ba[m] for m in mea])), "per_seed": {m: ba[m] for m in mea}}
        if n >= 5:
            d["test"] = t1.cluster_ci(wins, mea, {s: int(rec[s]) for s in starts})
        out[str(n)] = d
    return out


def calibrated_text(rows):
    cal = json.loads(Path("results/p1_item7_calib.json").read_text(encoding="utf-8"))
    delta = cal["deltas"]["A-compact"]["primary_rule"]
    logits = {(x["set"], x["n"], x["start"]): x["logits"]
              for x in json.loads(Path("results/p1_item7_calib_logits.json").read_text(encoding="utf-8"))["rows"]
              if x["split"] == "ds2v2" and x["arm"] == "A-compact"}
    out = []
    for x in rows:
        if x["arm"] == "A-compact":
            z = np.asarray(logits[(x["set"], x["n"], x["start"])], dtype=float)
            z[0] += delta
            t = TIERS[int(np.argmax(z))]
            x = {**x, "tier": t, "correct": t == x["reference"]}
        out.append(x)
    return out, delta


def false_alarms(rows, arms):
    res = {}
    for a in arms:
        xs = [x for x in rows if x["arm"] == a and x["reference"] == "routine"]
        nat = [x for x in xs if x["set"] == "natural"]
        res[a] = {"all_routine": float(np.mean([x["tier"] != "routine" for x in xs])),
                  "natural_routine": float(np.mean([x["tier"] != "routine" for x in nat])), "n": len(xs)}
    return res


def urgent_by_reason(rows, arms, events):
    from reasoning.training_targets import most_urgent_index
    res = {a: {"run>=3": [], "high-conf V/F": []} for a in arms}
    for x in rows:
        if x["arm"] not in res or x["set"] != "stratified" or x["n"] < 10 or x["reference"] != "urgent":
            continue
        w = events[x["start"]:x["start"] + x["n"]]
        reason = "run>=3" if w[most_urgent_index(w)]["clinical_flags"]["consecutive_abnormal_beats"] >= 3 else "high-conf V/F"
        res[x["arm"]][reason].append(x["correct"])
    return {a: {k: {"recall": float(np.mean(v)), "n": len(v)} for k, v in d.items() if v} for a, d in res.items()}


def slot_decoding():
    import p1_item7_slot_decoder as sd
    import p1_items6_7_decoder_anomaly as item6
    from p1_item7_common import replay_split, window_vectors
    r1, r2 = replay_split("ds1"), replay_split("ds2")
    rng = np.random.default_rng(0)
    s1 = item6.stratified(np.asarray(r1["labels"]), item6.PER_CLASS_DS1, rng)
    s2 = item6.stratified(np.asarray(r2["labels"]), item6.PER_CLASS_DS2, rng)
    f1 = sd.fields_from_events([r1["events"][i] for i in s1])
    f2 = sd.fields_from_events([r2["events"][i] for i in s2])
    x1 = np.concatenate([window_vectors(r1, i, 1, 35) for i in s1])
    x2 = np.concatenate([window_vectors(r2, i, 1, 35) for i in s2])
    out = {"input_35d": sd.decode(x1, f1, x2, f2, reduce=False)}
    for tag in ("r3", "r4"):
        for seed in (101, 202, 303):
            state = torch.load(f"reasoning/checkpoints/p1_item7_mea_{tag}_seed{seed}.pt", map_location="cpu",
                               weights_only=False)["adapter_state_dict"]
            dim = state["projection.weight"].shape[1]
            a1, a2 = (x1, x2) if dim == 35 else (x1[:, :32], x2[:, :32])
            for slot in (0, 49):
                out[f"{tag}_seed{seed}_slot{slot}"] = sd.decode(sd.slot_tokens(state, a1, slot), f1,
                                                                sd.slot_tokens(state, a2, slot), f2, reduce=True)
    return out


def main():
    from p1_item7_common import replay_split
    r2 = replay_split("ds2")
    rec = np.asarray(r2["record_ids"])
    rows4 = json.loads(R4.read_text(encoding="utf-8"))["rows"]
    rows3 = [x for x in json.loads(R3.read_text(encoding="utf-8"))["rows"] if x["arm"].startswith("MEA:")]
    m4, m3 = arms_of(rows4, "MEA:"), arms_of(rows3, "MEA:")
    out = {"design": __doc__, "r4_arms": m4, "r3_arms": m3}
    out["primary_default"] = primary(rows4, m4, rec)
    cal_rows, delta = calibrated_text(rows4)
    out["primary_calibrated_text"] = {"text_delta": delta, "by_n": primary(cal_rows, m4, rec)}
    out["r3_default"] = primary(rows3 + [x for x in rows4 if x["arm"] == "A-compact"], m3, rec)
    out["false_alarm"] = false_alarms(rows3 + rows4, m3 + m4 + ["A-compact"])
    out["urgent_by_reason"] = urgent_by_reason(rows3 + rows4, m3 + m4 + ["A-compact"], r2["events"])
    save_json_atomic(OUT, out)
    out["slots"] = slot_decoding()
    save_json_atomic(OUT, out)
    show(out)


def show(out):
    f = lambda d: f"{d['difference']:+.3f} [{d['cluster_ci95'][0]:+.3f}, {d['cluster_ci95'][1]:+.3f}] NI={d['non_inferior']} sup={d['superior']}"
    for n, d in out["primary_default"].items():
        r3 = out["r3_default"][n]["mea_mean"]
        c = out["primary_calibrated_text"]["by_n"][n]
        line = f"N={n:>2} text {d['text']:.3f} (cal {c['text']:.3f}) | r3 {r3:.3f} | r4 {d['mea_mean']:.3f}"
        if "test" in d:
            line += f" || vs default {f(d['test'])} || vs calibrated {f(c['test'])}"
        print(line)
    fa = out["false_alarm"]
    print("false alarms (all routine / natural):", {a.replace("MEA:", ""): (round(v["all_routine"], 3), round(v["natural_routine"], 3)) for a, v in fa.items()})
    print("urgent recall by reason:", {a.replace("MEA:", ""): {k: round(v["recall"], 2) for k, v in d.items()} for a, d in out["urgent_by_reason"].items()})
    if "slots" in out:
        for k, d in out["slots"].items():
            print(f"{k:22s} label {d['label']['mlp']['balanced_accuracy']:.2f} tier {d['tier']['mlp']['balanced_accuracy']:.2f} | "
                  f"R2 hr {d['heart_rate']['mlp']['r2']:.2f} rr {d['rr']['mlp']['r2']:.2f} run {d['run']['mlp']['r2']:.2f}")


if __name__ == "__main__":
    main()
