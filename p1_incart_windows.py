"""Deviation 21: INCART confirmatory test windows (docs/confirmatory_plan.md, section 2). CPU.

The p1_item7_testset_v2.build sampler, unchanged (seed 0; 60 windows per tier at N = 1 / 5 / 10 / 20,
40 at N = 50, class round-robin, at most 3 per recording per cell; 100 natural 50-beat windows at
N = 5 / 10 / 20 / 50), applied to the frozen perception agent's replay of INCART. Short cells stay
short. Writes the window list, achieved counts per cell, and a SHA-256 of the window keys. Nothing
about model output is computed.

Run (from repo root, after incart_prep.py):
    python p1_incart_windows.py
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from p1_io import save_json_atomic

OUT = Path("results/p1_item7_testset_incart.json")


def main():
    from p1_item7_common import replay_split
    from p1_item7_testset_v2 import build
    r = replay_split("incart")
    windows = build(r)
    with np.load("data/processed/incart_test.npz") as z:
        patients = z["patient_ids"]
    for w in windows:
        w["patient"] = int(patients[w["start"]])
    keys = sorted((w["set"], w["n"], w["start"]) for w in windows)
    counts = Counter(f"{w['set']}|{w['n']}|{w['reference']}" for w in windows)
    out = {"analysis_plan": "docs/confirmatory_plan.md (Deviation 21)", "design": __doc__, "n_windows": len(windows),
           "sha256_window_keys": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
           "cells": dict(sorted(counts.items())), "patients_covered": len({w["patient"] for w in windows}),
           "windows": windows}
    save_json_atomic(OUT, out)
    print(out["n_windows"], "windows;", out["patients_covered"], "patients; sha256", out["sha256_window_keys"][:16])
    print(json.dumps(out["cells"], indent=1))


if __name__ == "__main__":
    main()
