"""Exploratory (post hoc, Deviation 22 note): why does A-filtered drop at N = 50 on the second sender? CPU.
Urgent-window breakdown (stratified, N = 10/20/50): urgency reason, urgent beats, listed beats, deciding V/F
confidence, position of the first urgent beat; filtered-text vs adapter recall.

Run (from repo root):  python p1_filtered_n50_diag.py                      # main sender
    P1_ENCODER=perception/checkpoints/resnet1d_rr_seed0.pt P1_ENCODER_TAG=_res python p1_filtered_n50_diag.py
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.getcwd())
from p1_item7_common import replay_split  # noqa: E402

tag = os.environ.get("P1_ENCODER_TAG", "")
r = replay_split("ds2")
ev = r["events"]
rows = [x for x in json.load(open(f"results/p1_item7_eval_ds2v2_r4{tag}.json", encoding="utf-8"))["rows"]
        if x["set"] == "stratified" and x["reference"] == "urgent" and x["n"] in (10, 20, 50)]
mea = {}
for x in rows:
    if x["arm"].startswith("MEA"):
        mea.setdefault((x["n"], x["start"]), []).append(x["correct"])
out = []
for x in rows:
    if x["arm"] != "A-filtered":
        continue
    s, n = x["start"], x["n"]
    win = ev[s:s + n]
    urg = [k for k, e in enumerate(win) if e["clinical_flags"]["requires_urgent_review"]]
    reasons = set()
    for k in urg:
        e = win[k]
        c = e["classification"]
        if e["clinical_flags"]["consecutive_abnormal_beats"] >= 3:
            reasons.add("run")
        if c["label"] in ("V", "F") and c["confidence"] > 0.85:
            reasons.add("conf")
    listed = sum(e["classification"]["label"] != "N" for e in win)
    vf = [e["classification"]["confidence"] for e in win if e["classification"]["label"] in ("V", "F")]
    out.append({"n": n, "ok": x["correct"], "answer": x["tier"], "reason": "+".join(sorted(reasons)) or "?",
                "n_urgent": len(urg), "listed": listed, "first_urgent_pos": (urg[0] + 1) if urg else None,
                "max_vf_conf": max(vf) if vf else None, "mea_ok": float(np.mean(mea.get((n, s), [np.nan])))})
for n in (10, 20, 50):
    o = [d for d in out if d["n"] == n]
    print(f"\n== {tag or 'main'} N={n}: urgent windows {len(o)}, filtered correct {sum(d['ok'] for d in o)}")
    for reason in sorted({d["reason"] for d in o}):
        g = [d for d in o if d["reason"] == reason]
        print(f"  reason {reason:9s} windows {len(g):3d}  filtered recall {np.mean([d['ok'] for d in g]):.2f}  "
              f"adapter recall {np.nanmean([d['mea_ok'] for d in g]):.2f}  "
              f"urgent beats median {np.median([d['n_urgent'] for d in g]):.0f}  listed median {np.median([d['listed'] for d in g]):.0f}")
    bad = [d for d in o if not d["ok"]]
    if bad:
        print("  misses: n_urgent", Counter(min(d["n_urgent"], 3) for d in bad), " max V/F conf",
              [round(d["max_vf_conf"], 3) if d["max_vf_conf"] else None for d in bad][:25])
        print("  misses: first urgent position", sorted(d["first_urgent_pos"] for d in bad))
    good = [d for d in o if d["ok"]]
    if good:
        print("  hits:   n_urgent", Counter(min(d["n_urgent"], 3) for d in good))
