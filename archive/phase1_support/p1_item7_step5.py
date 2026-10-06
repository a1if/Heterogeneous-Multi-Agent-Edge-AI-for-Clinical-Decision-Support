"""Item 7, Deviation 16 analysis (run-order step 5). CPU only.

On results/p1_item7_eval_ds2v2.json (enlarged DS2 test set, corrected decoder):
  consistency  tiers on the 380 earlier windows vs the earlier runs, per arm
  primary      MEA r3 3-seed mean minus A-compact balanced accuracy at N = 5, 10, 20, 50;
               95% CI from the record-level (cluster) bootstrap (primary) and the window
               bootstrap (secondary); non-inferior if lower bound > -0.05, superior if > 0
  secondary    N = 1; confusion matrices; recall per tier and per predicted class of the
               most urgent beat; true-label escalation (sensitivity / specificity);
               3-seed majority vote; natural windows: accuracy and false-alarm rate with a
               record-level 95% CI; field-cap rate and output length
Writes results/p1_item7_step5.json.

Run (from repo root):
    python p1_item7_step5.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

import p1_item7_tier1 as t1
from p1_io import save_json_atomic

RESULTS = Path("results/p1_item7_eval_ds2v2.json")
OUT = Path("results/p1_item7_step5.json")
TIERS = t1.TIERS
OLD = {"MEA": ["results/p1_item7_eval_ds2_r3.json", "results/p1_item7_eval_ds2_n50.json"],
       "A-compact": ["results/p1_item7_eval_ds2.json", "results/p1_item7_eval_ds2_n50.json"]}


def window_ci(wins, mea, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    starts = [s for s, w in wins.items() if all(a in w["arms"] for a in mea + ["A-compact"])]
    by_tier = {t: [s for s in starts if wins[s]["reference"] == t] for t in TIERS}

    def diff(sel):
        ba = lambda arm: t1.bal([(wins[s]["reference"], wins[s]["arms"][arm]) for s in sel])
        return np.mean([ba(m) for m in mea]) - ba("A-compact")

    boots = [diff([s for t, v in by_tier.items() if v for s in rng.choice(v, len(v))]) for _ in range(n_boot)]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def rate_ci(flags_by_record, n_boot=20000, seed=0):
    """Mean of per-window flags with a record-level bootstrap CI."""
    rng = np.random.default_rng(seed)
    recs = [r for r in flags_by_record if flags_by_record[r]]
    allf = [f for r in recs for f in flags_by_record[r]]
    boots = []
    for _ in range(n_boot):
        sel = [f for r in rng.choice(recs, len(recs)) for f in flags_by_record[r]]
        boots.append(np.mean(sel))
    return {"rate": float(np.mean(allf)), "n": len(allf), "ci95": [float(np.percentile(boots, 2.5)),
                                                                   float(np.percentile(boots, 97.5))]}


def main():
    from p1_item7_common import replay_split
    r = replay_split("ds2")
    rec, labels = np.asarray(r["record_ids"]), np.asarray(r["labels"])
    rows = json.loads(RESULTS.read_text(encoding="utf-8"))["rows"]
    mea = sorted({x["arm"] for x in rows if x["arm"].startswith("MEA:")})
    arms = mea + ["A-compact"]
    out = {"design": __doc__, "arms": arms, "n_rows": len(rows)}

    # consistency with the earlier runs on the 380 shared windows
    key = lambda x: (x["set"], x["n"], x["start"])
    cons = {}
    for arm in arms:
        old = {}
        for f in OLD["MEA" if arm.startswith("MEA") else "A-compact"]:
            for x in json.loads(Path(f).read_text(encoding="utf-8"))["rows"]:
                if x["arm"] == arm:
                    old[key(x)] = x["tier"]
        new = {key(x): x["tier"] for x in rows if x["arm"] == arm}
        both = [k for k in old if k in new]
        cons[arm] = {"n": len(both), "same_tier": sum(old[k] == new[k] for k in both)}
    out["consistency_on_earlier_windows"] = cons

    wbn = t1.windows_by_n(rows)
    out["by_n"] = {}
    for n, wins in sorted(wbn.items()):
        starts = [s for s, w in wins.items() if all(a in w["arms"] for a in arms)]
        ref = lambda s: wins[s]["reference"]
        d = {"n_windows": len(starts), "n_records": len({int(rec[s]) for s in starts})}
        d["balanced_accuracy"] = {a: t1.bal([(ref(s), wins[s]["arms"][a]) for s in starts]) for a in arms}
        d["balanced_accuracy"]["MEA_mean"] = float(np.mean([d["balanced_accuracy"][m] for m in mea]))
        ens = {s: t1.majority([wins[s]["arms"][m] for m in mea]) for s in starts}
        d["balanced_accuracy"]["MEA_vote"] = t1.bal([(ref(s), ens[s]) for s in starts])
        if n >= 5:
            c = t1.cluster_ci(wins, mea, {s: int(rec[s]) for s in starts})
            c["window_ci95"] = window_ci(wins, mea)
            d["primary"] = c
        d["confusion"] = {a: {t: dict(Counter(wins[s]["arms"][a] for s in starts if ref(s) == t)) for t in TIERS}
                          for a in ["A-compact"] + mea}
        top = {x["start"]: x.get("top_class_predicted") for x in rows if x["set"] == "stratified" and x["n"] == n}
        d["recall_by_class"] = {a: {f"{t}|{c}": float(np.mean([wins[s]["arms"][a] == t for s in starts
                                                                if ref(s) == t and top[s] == c]))
                                    for t in ("priority", "urgent") for c in ("F", "S", "V")
                                    if any(ref(s) == t and top[s] == c for s in starts)}
                                for a in ["A-compact"] + mea}
        truth = {s: bool(any("NSVFQ"[int(labels[i])] != "N" for i in range(s, s + n))) for s in starts}

        def esc(get):
            pos, neg = [s for s in starts if truth[s]], [s for s in starts if not truth[s]]
            return {"sensitivity": float(np.mean([get(s) != "routine" for s in pos])) if pos else None,
                    "specificity": float(np.mean([get(s) == "routine" for s in neg])) if neg else None}
        d["truth"] = {"reference": esc(ref), **{a: esc(lambda s, a=a: wins[s]["arms"][a]) for a in arms},
                      "MEA_vote": esc(lambda s: ens[s])}
        out["by_n"][str(n)] = d

    # natural windows: accuracy and false alarms (routine reference answered priority / urgent)
    nat = {}
    for n in sorted({x["n"] for x in rows if x["set"] == "natural"}):
        a_rows = [x for x in rows if x["set"] == "natural" and x["n"] == n]
        res = {}
        for a in arms:
            xs = [x for x in a_rows if x["arm"] == a]
            fa = {}
            for x in xs:
                if x["reference"] == "routine":
                    fa.setdefault(int(rec[x["start"]]), []).append(x["tier"] != "routine")
            res[a] = {"accuracy": float(np.mean([x["correct"] for x in xs])), "false_alarm": rate_ci(fa)}
        nat[str(n)] = res
    out["natural"] = nat
    out["text"] = {a: {"field_cap_rate": float(np.mean([x["hit_field_cap"] for x in rows if x["arm"] == a])),
                       "median_output_tokens": float(np.median([x["n_tokens"] for x in rows if x["arm"] == a])),
                       "parse_rate": float(np.mean([x["parsed"] for x in rows if x["arm"] == a]))} for a in arms}
    save_json_atomic(OUT, out)
    show(out)


def show(out):
    print("consistency:", {a: f"{v['same_tier']}/{v['n']}" for a, v in out["consistency_on_earlier_windows"].items()})
    for n, d in out["by_n"].items():
        ba = d["balanced_accuracy"]
        line = (f"N={n:>2} ({d['n_windows']} win, {d['n_records']} rec) text {ba['A-compact']:.3f} | MEA mean "
                f"{ba['MEA_mean']:.3f} | vote {ba['MEA_vote']:.3f}")
        if "primary" in d:
            p = d["primary"]
            line += (f" || diff {p['difference']:+.3f} record CI [{p['cluster_ci95'][0]:+.3f}, {p['cluster_ci95'][1]:+.3f}]"
                     f" window CI [{p['window_ci95'][0]:+.3f}, {p['window_ci95'][1]:+.3f}] NI={p['non_inferior']} sup={p['superior']}")
        print(line)
    for n, res in out["natural"].items():
        print(f"natural N={n}: " + " ; ".join(f"{a.replace('MEA:r3_', '')} acc {v['accuracy']:.2f} FA {v['false_alarm']['rate']:.3f} "
                                             f"[{v['false_alarm']['ci95'][0]:.3f},{v['false_alarm']['ci95'][1]:.3f}] n={v['false_alarm']['n']}"
                                             for a, v in res.items()))
    print("text:", out["text"])


if __name__ == "__main__":
    main()
