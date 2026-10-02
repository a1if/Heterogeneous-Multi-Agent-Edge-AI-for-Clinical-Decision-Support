"""Deviation 25: does stopping text generation at the tier token change any answer? CPU.

Compares the stop-at-tier generations (results/p1_item7_eval_ds2v2_stoptier.json; p1_item7_eval.py
--stop-at-tier, main sender) with the saved full generations (results/p1_item7_eval_ds2v2_r4.json) for
A-compact and A-filtered on every Deviation 16 DS2 window. Decision rule (fixed before the data): if the
tier is identical on every window of both arms, the INCART text arms are generated with --stop-at-tier
(amendment to docs/confirmatory_plan.md, made before the freeze); otherwise full generation is kept.

Run (from repo root):
    python p1_stop_at_tier_check.py
"""
import json
from pathlib import Path

from p1_io import save_json_atomic

FULL = Path("results/p1_item7_eval_ds2v2_r4.json")
STOP = Path("results/p1_item7_eval_ds2v2_stoptier.json")
OUT = Path("results/p1_stop_at_tier_check.json")
ARMS = ("A-compact", "A-filtered")


def main():
    key = lambda x: (x["arm"], x["set"], x["n"], x["start"])
    full = {key(x): x["tier"] for x in json.loads(FULL.read_text(encoding="utf-8"))["rows"] if x["arm"] in ARMS}
    stop = {key(x): x["tier"] for x in json.loads(STOP.read_text(encoding="utf-8"))["rows"] if x["arm"] in ARMS}
    out = {"design": __doc__, "by_arm": {}}
    for arm in ARMS:
        ks = [k for k in full if k[0] == arm]
        missing = [k for k in ks if k not in stop]
        diff = [{"set": k[1], "n": k[2], "start": k[3], "full": full[k], "stop": stop[k]}
                for k in ks if k in stop and stop[k] != full[k]]
        out["by_arm"][arm] = {"n_windows": len(ks), "missing": len(missing), "n_different": len(diff),
                              "agreement": 1 - len(diff) / max(1, len(ks) - len(missing)), "differences": diff[:50]}
    out["identical"] = all(d["missing"] == 0 and d["n_different"] == 0 for d in out["by_arm"].values())
    out["decision"] = ("use --stop-at-tier for the INCART text arms (amend the confirmatory plan before the freeze)"
                       if out["identical"] else "keep full generation for the INCART text arms")
    save_json_atomic(OUT, out)
    for arm, d in out["by_arm"].items():
        print(f"{arm}: {d['n_windows']} windows, missing {d['missing']}, different {d['n_different']}")
    print("DECISION:", out["decision"])


if __name__ == "__main__":
    main()
