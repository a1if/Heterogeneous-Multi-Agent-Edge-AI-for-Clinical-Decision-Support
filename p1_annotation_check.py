"""Escalation scored against the database annotations instead of the sender's own rule (primary DS2 test set). CPU.

Same definition as p1_item7_tier1.py, applied to the 1,195 primary windows: a window is truly abnormal if any of its
beats carries a non-normal annotation (S, V, F or Q). An answer escalates if its tier is not routine. Reported per
number of events N, for the reference rule (the tier the rule gives on the sender's own outputs), compact text under
default decoding, filtered text, the final adapter (three seeds) and the earlier adapter (three seeds).

Run (from repo root):  python p1_annotation_check.py   -> results/p1_annotation_check.json
"""
import json
from pathlib import Path

import numpy as np

from p1_io import save_json_atomic

LABELS = "NSVFQ"
FILES = ["results/p1_item7_eval_ds2v2_r4.json", "results/p1_item7_eval_ds2v2.json"]
OUT = Path("results/p1_annotation_check.json")


def main():
    from p1_item7_common import replay_split
    labels = np.asarray(replay_split("ds2")["labels"])
    wins = {}  # (n, start) -> {"reference": tier, "arms": {arm: tier}}
    for f in FILES:
        for r in json.loads(Path(f).read_text(encoding="utf-8"))["rows"]:
            w = wins.setdefault((r["n"], r["start"]), {"reference": r["reference"], "arms": {}})
            w["arms"][r["arm"]] = r["tier"]
    out = {"design": __doc__, "by_n": {}}
    for n in sorted({n for n, _ in wins}):
        keys = [k for k in wins if k[0] == n]
        truth = {k: any(LABELS[int(labels[i])] != "N" for i in range(k[1], k[1] + n)) for k in keys}
        def score(get):
            pos = [k for k in keys if truth[k] and get(k) is not None]
            neg = [k for k in keys if not truth[k] and get(k) is not None]
            return {"sensitivity": float(np.mean([get(k) != "routine" for k in pos])),
                    "specificity": float(np.mean([get(k) == "routine" for k in neg])),
                    "n_truly_abnormal": len(pos), "n_truly_normal": len(neg)}
        arms = sorted({a for k in keys for a in wins[k]["arms"]})
        d = {"reference_rule": score(lambda k: wins[k]["reference"])}
        for a in arms:
            d[a] = score(lambda k, a=a: wins[k]["arms"].get(a))
        out["by_n"][str(n)] = d
    save_json_atomic(OUT, out)
    for n, d in out["by_n"].items():
        print(f"N={n:>2} " + " | ".join(f"{a.replace('MEA:', '')} {v['sensitivity']:.2f}/{v['specificity']:.2f}" for a, v in d.items()))


if __name__ == "__main__":
    main()
