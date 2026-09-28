"""Item 7, Deviation 16: the enlarged DS2 test set (frozen before any generation). CPU only.

Same sampler, seeds and rules as the Deviation 10 / 13 sets, with larger quotas, so every
previously evaluated window is included:
  stratified  N in {1, 5, 10, 20}: 60 per tier (was 20); N = 50: 40 per tier (was 20),
              drawn in a separate call exactly as before. Class round-robin over the
              predicted class of the most urgent beat, at most 3 per record per cell,
              non-overlapping, seed 0 (p1_pilot2_stratified.stratified_windows)
  natural     100 random 50-beat windows (was 20), same generator and seed as
              p1_e3_multi_event.windows, first N beats for N in {5, 10, 20, 50}
Writes results/p1_item7_testset_v2.json with the windows, the achieved class mix per
cell, the subset check against the old sets, and a SHA-256 of the window list.

Run (from repo root):
    python p1_item7_testset_v2.py
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from p1_io import save_json_atomic

PER_TIER = {1: 60, 5: 60, 10: 60, 20: 60, 50: 40}
N_NATURAL = 100
NS_NATURAL = (5, 10, 20, 50)
OUT = Path("results/p1_item7_testset_v2.json")
LABELS = "NSVFQ"


def natural_starts(record_ids, rng, count, length=50):
    """p1_e3_multi_event.windows with a count parameter (identical draws, so the first 20 match)."""
    starts = np.flatnonzero(np.r_[True, np.diff(record_ids) != 0])
    ends = np.r_[starts[1:], len(record_ids)]
    out = []
    while len(out) < count:
        r = rng.integers(len(starts))
        if ends[r] - starts[r] < length + 1:
            continue
        out.append(int(rng.integers(starts[r], ends[r] - length)))
    return out


def build(r):
    from p1_pilot2_stratified import RANK, stratified_windows
    small = stratified_windows(r["tiers"], r["record_ids"], np.random.default_rng(0), ns=(1, 5, 10, 20),
                               per_cell=60, classes=r["classes"])
    big = stratified_windows(r["tiers"], r["record_ids"], np.random.default_rng(0), ns=(50,), per_cell=40,
                             classes=r["classes"])
    windows = [{"set": "stratified", "n": n, "start": s} for (n, t), ss in {**small, **big}.items() for s in ss]
    for s in natural_starts(np.asarray(r["record_ids"]), np.random.default_rng(0), N_NATURAL):
        windows += [{"set": "natural", "n": n, "start": s} for n in NS_NATURAL]
    for w in windows:
        idx = range(w["start"], w["start"] + w["n"])
        w["reference"] = max((r["tiers"][i] for i in idx), key=RANK.get)
        top = max(idx, key=lambda i: RANK[r["tiers"][i]])
        w["top_class_predicted"], w["top_class_true"] = r["classes"][top], LABELS[int(r["labels"][top])]
        w["record"] = int(r["record_ids"][w["start"]])
    return windows


def main():
    from p1_item7_common import replay_split
    from p1_item7_eval import load_windows
    r = replay_split("ds2")
    windows = build(r)
    key = lambda w: (w["set"], w["n"], w["start"])
    old = {key(w) for w in load_windows("ds2", r) + load_windows("ds2_n50", r)}
    new = {key(w) for w in windows}
    mix = {}
    for w in windows:
        if w["set"] == "stratified":
            mix.setdefault(f"{w['n']}|{w['reference']}", Counter())[w["top_class_predicted"]] += 1
    nat = Counter((w["n"], w["reference"]) for w in windows if w["set"] == "natural")
    digest = hashlib.sha256(json.dumps(sorted(new)).encode()).hexdigest()
    out = {"analysis_plan": "docs/analysis_plan.md (Deviation 16)", "design": __doc__,
           "n_windows": len(windows), "sha256_window_keys": digest,
           "old_windows_all_included": old <= new, "n_old": len(old), "n_new_only": len(new - old),
           "stratified_class_mix": {k: dict(v) for k, v in sorted(mix.items())},
           "natural_reference_counts": {f"{n}|{t}": c for (n, t), c in sorted(nat.items())},
           "records_per_n": {str(n): len({w["record"] for w in windows if w["n"] == n and w["set"] == "stratified"})
                             for n in PER_TIER},
           "windows": windows}
    save_json_atomic(OUT, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("windows", "design")}, indent=1))


if __name__ == "__main__":
    main()
