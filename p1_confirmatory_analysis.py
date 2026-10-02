"""Deviation 21: confirmatory analysis of r4-confirmatory on INCART (docs/confirmatory_plan.md). CPU.

Inputs: results/p1_item7_eval_incart.json (r4 seeds, A-compact, A-filtered), results/p1_incart_compact_logits.json
and results/p1_incart_filtered_logits.json (tier logits; calibration biases FROZEN in
results/p1_freeze_manifest.json, never re-tuned), results/p1_incart_timing.json, and the patient ids in
data/processed/incart_test.npz.

Hypotheses in fixed sequence (alpha 0.05; stop at the first failure; later ones are reported descriptively
as 'not tested in sequence'):
  H1  r4 (3-seed mean) non-inferior to calibrated A-compact at N = 10, 20 and 50 (all must pass; margin -0.05)
  H2  r4 superior to calibrated A-compact at N = 10 and 20
  H3  r4 non-inferior to calibrated A-filtered at N = 10, 20 and 50
  H4  r4 median time to first token lower than A-compact's at N = 10, 20 and 50 (95% CI upper bound < 0)
Patient-cluster bootstrap, 20,000 resamples, seed 0. A failed generation (no tier) counts as incorrect and
as an alarm. Everything else is descriptive.

Run (from repo root):
    python p1_confirmatory_analysis.py
"""
import json
from pathlib import Path

import numpy as np

from p1_io import save_json_atomic

EVAL = Path("results/p1_item7_eval_incart.json")
LOGITS = {"A-compact": Path("results/p1_incart_compact_logits.json"),
          "A-filtered": Path("results/p1_incart_filtered_logits.json")}
TIMING = Path("results/p1_incart_timing.json")
MANIFEST = Path("results/p1_freeze_manifest.json")
OUT = Path("results/p1_confirmatory_incart.json")
R4 = ("MEA:r4_seed101", "MEA:r4_seed202", "MEA:r4_seed303")
TIERS = ("routine", "priority", "urgent")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main():
    import p1_item7_tier1 as t1
    from p1_item7_calib import answer
    manifest = load(MANIFEST)
    deltas = manifest["calibration_deltas"]
    with np.load("data/processed/incart_test.npz") as z:
        patient = z["patient_ids"]
    rows = load(EVAL)["rows"]
    for arm, path in LOGITS.items():  # calibrated text with the frozen bias
        lg = {(x["set"], x["n"], x["start"]): x["logits"] for x in load(path)["rows"]}
        for x in [x for x in rows if x["arm"] == arm]:
            t = answer(lg[(x["set"], x["n"], x["start"])], deltas[arm])
            rows.append({**x, "arm": f"{arm}-cal", "tier": t, "correct": t == x["reference"]})
    wins = t1.windows_by_n(rows)

    def test(base, n):
        w = wins[n]
        starts = [s for s, v in w.items() if all(a in v["arms"] for a in R4 + (base,))]
        return t1.cluster_ci({s: w[s] for s in starts}, list(R4), {s: int(patient[s]) for s in starts}, base=base)

    out = {"design": __doc__, "frozen_manifest_sha": manifest.get("manifest_sha256"), "deltas": deltas, "hypotheses": {}}
    tm = load(TIMING)["rows"]

    def ttft(n):
        by = {}
        for x in tm:
            if x["n"] == n:
                by.setdefault(x["start"], {})[x["arm"]] = x
        pair = [v for v in by.values() if "MEA" in v and "A-compact" in v]
        rel = np.array([v["MEA"]["ttft_ms"] / v["A-compact"]["ttft_ms"] - 1 for v in pair])
        rng = np.random.default_rng(0)
        boots = [np.median(rel[rng.integers(0, len(rel), len(rel))]) for _ in range(20000)]
        return {"median_rel": float(np.median(rel)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "n_windows": len(pair)}

    plan = [("H1", [(n, "A-compact-cal", "non_inferior") for n in (10, 20, 50)]),
            ("H2", [(n, "A-compact-cal", "superior") for n in (10, 20)]),
            ("H3", [(n, "A-filtered-cal", "non_inferior") for n in (10, 20, 50)]),
            ("H4", [(n, "timing", "faster") for n in (10, 20, 50)])]
    still_testing = True
    for name, parts in plan:
        res = {}
        for n, base, kind in parts:
            if kind == "faster":
                d = ttft(n)
                res[str(n)] = {**d, "pass": d["ci95"][1] < 0}
            else:
                d = test(base, n)
                res[str(n)] = {**d, "pass": d[kind]}
        passed = all(v["pass"] for v in res.values())
        out["hypotheses"][name] = {"tests": res, "passed": passed,
                                   "status": ("confirmed" if passed else "not confirmed") if still_testing else "not tested in sequence"}
        if still_testing and not passed:
            still_testing = False
    # descriptive
    arms = sorted({x["arm"] for x in rows})
    out["balanced_accuracy"] = {str(n): {a: t1.bal([(v["reference"], v["arms"][a]) for v in w.values() if a in v["arms"]])
                                         for a in arms} for n, w in sorted(wins.items())}
    out["false_alarm"] = {}
    for a in arms:
        rt = [x for x in rows if x["arm"] == a and x["reference"] == "routine"]
        nat = [x for x in rt if x["set"] == "natural"]
        out["false_alarm"][a] = {"all": float(np.mean([x["tier"] != "routine" for x in rt])) if rt else None,
                                 "natural": float(np.mean([x["tier"] != "routine" for x in nat])) if nat else None}
    out["parse_rate"] = {a: float(np.mean([x["parsed"] for x in rows if x["arm"] == a])) for a in arms if not a.endswith("-cal")}
    out["field_cap_rate"] = {a: float(np.mean([x["hit_field_cap"] for x in rows if x["arm"] == a])) for a in arms
                             if not a.endswith("-cal")}
    out["timing"] = {str(n): ttft(n) for n in (1, 5, 10, 20, 50) if any(x["n"] == n for x in tm)}
    save_json_atomic(OUT, out)
    for h, d in out["hypotheses"].items():
        print(h, d["status"], {n: (round(v.get("difference", v.get("median_rel")), 3), v["pass"]) for n, v in d["tests"].items()})
    for n, d in out["balanced_accuracy"].items():
        print(f"N={n:>2}", {a.replace("MEA:", ""): round(v, 3) for a, v in d.items()})


if __name__ == "__main__":
    main()
