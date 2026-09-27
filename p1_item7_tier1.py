"""Item 7, Deviation 15 (EXPLORATORY): robustness analyses on the saved outputs. CPU only.

Sets (all constrained decoding, DS2):
  r2      results/p1_item7_eval_ds2.json      MEA r2 seeds 101/202/303 + A-compact, N 1-20
  r3      results/p1_item7_eval_ds2_r3.json   MEA r3 seeds + A-compact (reused), N 1-20
  r3_n50  results/p1_item7_eval_ds2_n50.json  MEA r3 seeds + A-compact, N 50
Analyses:
  1 cluster  paired balanced-accuracy difference (MEA seed mean - A-compact), 95% CI from
             resampling DS2 records (all windows of a record together), vs the window bootstrap
  2 truth    escalation (answer != routine) scored against "the window holds >= 1 truly
             abnormal beat (annotation S/V/F/Q)": sensitivity / specificity, per arm and for
             the encoder-rule reference itself (the ceiling from these events)
  3 ensemble majority vote of the 3 MEA seeds (median tier rank) per window
  4 routing  MEA majority when all 3 seeds agree, else A-compact; share routed to text

Run (from repo root):
    python p1_item7_tier1.py
"""
import json
from pathlib import Path

import numpy as np

from p1_io import save_json_atomic

SETS = {"r2": "results/p1_item7_eval_ds2.json", "r3": "results/p1_item7_eval_ds2_r3.json",
        "r3_n50": "results/p1_item7_eval_ds2_n50.json"}
TIERS = ("routine", "priority", "urgent")
RANK = {t: i for i, t in enumerate(TIERS)}
LABELS = "NSVFQ"
RESULTS = Path("results/p1_item7_tier1.json")


def bal(recs):
    """recs: list of (reference, answer) -> balanced accuracy over the tiers present."""
    per = [np.mean([a == t for r, a in recs if r == t]) for t in TIERS if any(r == t for r, _ in recs)]
    return float(np.mean(per)) if per else float("nan")


def windows_by_n(rows):
    """{n: {start: {"reference", "arms": {arm: tier}}}} for the stratified set."""
    out = {}
    for x in rows:
        if x["set"] != "stratified":
            continue
        w = out.setdefault(x["n"], {}).setdefault(x["start"], {"reference": x["reference"], "arms": {}})
        w["arms"][x["arm"]] = x["tier"]
    return out


def cluster_ci(wins, mea, record_of, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    starts = [s for s, w in wins.items() if all(a in w["arms"] for a in mea + ["A-compact"])]
    recs = sorted({record_of[s] for s in starts})
    by_rec = {r: [s for s in starts if record_of[s] == r] for r in recs}

    def diff(sel):
        ba = lambda arm: bal([(wins[s]["reference"], wins[s]["arms"][arm]) for s in sel])
        return np.mean([ba(m) for m in mea]) - ba("A-compact")

    point = float(diff(starts))
    boots = []
    for _ in range(n_boot):
        sel = [s for r in rng.choice(recs, len(recs)) for s in by_rec[r]]
        d = diff(sel)
        if np.isfinite(d):
            boots.append(d)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"difference": point, "cluster_ci95": [float(lo), float(hi)], "n_windows": len(starts), "n_records": len(recs),
            "non_inferior": bool(lo > -0.05), "superior": bool(lo > 0)}


def majority(tiers):
    ranks = sorted(RANK[t] for t in tiers if t is not None)
    return TIERS[ranks[len(ranks) // 2]] if ranks else None


def main():
    from p1_item7_common import replay_split
    r = replay_split("ds2")
    rec_ids, labels = np.asarray(r["record_ids"]), np.asarray(r["labels"])
    out = {"design": __doc__, "sets": {}}
    for name, path in SETS.items():
        rows = json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
        mea = sorted({x["arm"] for x in rows if x["arm"].startswith("MEA:")})
        wbn = windows_by_n(rows)
        res = {"mea_arms": mea, "by_n": {}}
        for n, wins in sorted(wbn.items()):
            starts = [s for s, w in wins.items() if all(a in w["arms"] for a in mea + ["A-compact"])]
            record_of = {s: int(rec_ids[s]) for s in starts}
            d = {}
            if n >= 5:
                d["cluster"] = cluster_ci(wins, mea, record_of)
            # 2 truth: escalation vs "any truly abnormal beat"
            truth = {s: bool(any(LABELS[int(labels[i])] != "N" for i in range(s, s + n))) for s in starts}
            def esc_scores(get):
                pos = [s for s in starts if truth[s]]
                neg = [s for s in starts if not truth[s]]
                sens = float(np.mean([get(s) != "routine" for s in pos])) if pos else None
                spec = float(np.mean([get(s) == "routine" for s in neg])) if neg else None
                return {"sensitivity": sens, "specificity": spec, "n_truly_abnormal": len(pos), "n_truly_normal": len(neg)}
            d["truth"] = {"encoder_rule_reference": esc_scores(lambda s: wins[s]["reference"]),
                          "A-compact": esc_scores(lambda s: wins[s]["arms"]["A-compact"])}
            for m in mea:
                d["truth"][m] = esc_scores(lambda s, m=m: wins[s]["arms"][m])
            # 3 ensemble, 4 routing
            ens = {s: majority([wins[s]["arms"][m] for m in mea]) for s in starts}
            agree = {s: len({wins[s]["arms"][m] for m in mea}) == 1 for s in starts}
            routed = {s: (ens[s] if agree[s] else wins[s]["arms"]["A-compact"]) for s in starts}
            ref = lambda s: wins[s]["reference"]
            d["balanced_accuracy"] = {
                "A-compact": bal([(ref(s), wins[s]["arms"]["A-compact"]) for s in starts]),
                "MEA_seed_mean": float(np.mean([bal([(ref(s), wins[s]["arms"][m]) for s in starts]) for m in mea])),
                "MEA_majority_vote": bal([(ref(s), ens[s]) for s in starts]),
                "router_agree_else_text": bal([(ref(s), routed[s]) for s in starts])}
            d["router_share_to_text"] = float(np.mean([not agree[s] for s in starts]))
            d["truth"]["MEA_majority_vote"] = esc_scores(lambda s: ens[s])
            d["truth"]["router"] = esc_scores(lambda s: routed[s])
            res["by_n"][str(n)] = d
        out["sets"][name] = res
    save_json_atomic(RESULTS, out)
    show(out)


def show(out):
    f = lambda v: "  -  " if v is None else f"{v:.2f}"
    for name, res in out["sets"].items():
        print(f"== {name} ({', '.join(res['mea_arms'])})")
        for n, d in res["by_n"].items():
            ba = d["balanced_accuracy"]
            line = (f"N={n:>2} bal-acc text {ba['A-compact']:.2f} | MEA mean {ba['MEA_seed_mean']:.2f} | vote {ba['MEA_majority_vote']:.2f}"
                    f" | router {ba['router_agree_else_text']:.2f} ({d['router_share_to_text']*100:.0f}% to text)")
            if "cluster" in d:
                c = d["cluster"]
                line += f" || diff {c['difference']:+.2f} cluster CI [{c['cluster_ci95'][0]:+.2f}, {c['cluster_ci95'][1]:+.2f}] ({c['n_records']} records)"
            print(line)
            t = d["truth"]
            print("       truth sens/spec: " + " ; ".join(
                f"{k.replace('encoder_rule_reference', 'ref').replace('MEA:', '')} {f(v['sensitivity'])}/{f(v['specificity'])}"
                for k, v in t.items()))


if __name__ == "__main__":
    main()
