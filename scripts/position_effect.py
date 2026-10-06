"""Accuracy by position of the deciding beat in the window (the 'lost in the middle' check).

Same logic as fig_why_text_fails in reports/make_figures.py, which computed this on the fly and never saved it.
Reads results/p1_item7_eval_ds2v2_r4.json (A-compact and the three r4 seeds), replays the DS2 events, and
writes results/p1_position_effect.json. Exploratory (not pre-registered). CPU only.

Run from the repository root:
    python scripts/position_effect.py
"""
import json
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1_item7_common import replay_split  # noqa: E402
from reasoning.training_targets import most_urgent_index  # noqa: E402

R4_ARMS = ("MEA:r4_seed101", "MEA:r4_seed202", "MEA:r4_seed303")
ev = replay_split("ds2")["events"]
rows = [x for x in json.loads((ROOT / "results" / "p1_item7_eval_ds2v2_r4.json").read_text(encoding="utf-8"))["rows"]
        if x["arm"] in ("A-compact",) + R4_ARMS]

thirds = ("First third", "Middle third", "Last third")
acc = defaultdict(lambda: defaultdict(list))        # arm -> third -> [correct]
per_seed = defaultdict(lambda: defaultdict(list))   # r4 seed -> third -> [correct]
for x in rows:
    if x["set"] != "stratified" or x["n"] < 10 or x["reference"] == "routine":
        continue
    w = ev[x["start"]:x["start"] + x["n"]]
    i = most_urgent_index(w)
    third = thirds[0] if i < x["n"] / 3 else (thirds[1] if i < 2 * x["n"] / 3 else thirds[2])
    if x["arm"] == "A-compact":
        acc["text"][third].append(bool(x["correct"]))
    else:
        acc["adapter"][third].append(bool(x["correct"]))
        per_seed[x["arm"]][third].append(bool(x["correct"]))

out = {
    "design": "Exploratory: accuracy by position of the window's most urgent beat (first/middle/last third), "
              "stratified DS2 windows with N >= 10 and a non-routine reference tier. Logic copied from "
              "reports/make_figures.py fig_why_text_fails.",
    "by_arm": {arm: {t: {"n": len(v), "accuracy": sum(v) / len(v)} for t, v in d.items()} for arm, d in acc.items()},
    "by_adapter_seed": {arm: {t: {"n": len(v), "accuracy": sum(v) / len(v)} for t, v in d.items()} for arm, d in per_seed.items()},
}
(ROOT / "results" / "p1_position_effect.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for arm, d in out["by_arm"].items():
    print(arm, {t: (v["n"], round(v["accuracy"], 3)) for t, v in d.items()})
