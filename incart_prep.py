"""INCART conversion for the confirmatory evaluation (Deviation 21, docs/confirmatory_plan.md).

Converts the St Petersburg INCART 12-lead Arrhythmia Database (PhysioNet incartdb 1.0.0,
already downloaded to data/incartdb/ and checked against SHA256SUMS.txt) into the same
array format as data_prep.py, so the frozen perception agent and every downstream script
can read it unchanged. The procedure is fixed by the plan, before any inspection:

  - lead II (asserted present); 257 Hz -> 360 Hz polyphase resampling (up 360, down 257)
  - annotation positions mapped to the 360 Hz grid: round(sample * 360 / 257)
  - beats whose symbol is in data_prep.AAMI_MAP (unchanged AAMI EC57 mapping); 360-sample
    window centred on the mapped position, inside the recording, z-scored (SD < 1e-8 skipped)
  - RR interval in ms from the ORIGINAL 257 Hz positions to the previous accepted beat;
    the first beat of a recording gets the split median (as data_prep.build_split);
    gaps > 12 s are capped later by p1_item7_common.load_split, as for MIT-BIH
  - record_ids = recording number (I01 -> 1 ... I75 -> 75), used for chronological replay
    and the sampler's per-record cap; patient_ids from files-patients-diagnoses.txt, used
    for the patient-cluster bootstrap

Only conversion checks are written (beat counts per class and recording, exclusions);
nothing about model performance.

Run (from repo root):
    python incart_prep.py
Produces:
    data/processed/incart_test.npz   (features, labels, rr_interval_ms, record_ids, patient_ids)
    results/p1_incart_conversion.json
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from data_prep import AAMI_MAP, LABEL_TO_INT, SAMPLE_RATE_HZ, WINDOW_HALF

SRC_HZ = 257
DL_DIR = Path("data/incartdb")
OUT = Path("data/processed/incart_test.npz")
CHECKS = Path("results/p1_incart_conversion.json")
LEAD = "II"


def to_target_index(sample):
    """257 Hz annotation position -> index on the 360 Hz grid."""
    return int(round(sample * SAMPLE_RATE_HZ / SRC_HZ))


def convert_record(signal_src, ann_samples, ann_symbols):
    """One recording. signal_src: lead II at 257 Hz. Returns windows (n, 360) float32, labels
    (list of AAMI letters), rr_ms (list, None for the first accepted beat), exclusion counts."""
    sig = resample_poly(np.asarray(signal_src, dtype=np.float64), SAMPLE_RATE_HZ, SRC_HZ)
    windows, labels, rr = [], [], []
    skipped = Counter()
    prev = None
    for s, sym in zip(ann_samples, ann_symbols):
        if sym not in AAMI_MAP:
            skipped["symbol_not_mapped"] += 1
            continue
        c = to_target_index(s)
        start, end = c - WINDOW_HALF, c + WINDOW_HALF
        if start < 0 or end > len(sig):
            skipped["window_outside_recording"] += 1
            continue
        w = sig[start:end]
        sd = w.std()
        if sd < 1e-8:
            skipped["flat_window"] += 1
            continue
        rr.append((s - prev) / SRC_HZ * 1000.0 if prev is not None else None)
        prev = s
        windows.append(((w - w.mean()) / sd).astype(np.float32))
        labels.append(AAMI_MAP[sym])
    return np.asarray(windows, dtype=np.float32).reshape(-1, 2 * WINDOW_HALF), labels, rr, skipped


def patient_map(path=DL_DIR / "files-patients-diagnoses.txt"):
    out, patient = {}, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        g = re.match(r"\s*patient\s+(\d+)", line)
        if g:
            patient = int(g.group(1))
            continue
        for rec in re.findall(r"I\d\d", line):
            out[rec] = patient
    return out


def main():
    import wfdb
    records = (DL_DIR / "RECORDS").read_text().split()
    patients = patient_map()
    assert sorted(patients) == sorted(records), "every recording must map to a patient"
    X, y, rr, rec_ids, pat_ids = [], [], [], [], []
    checks = {"design": __doc__, "records": {}, "skipped": Counter()}
    for rec in records:
        signal, fields = wfdb.rdsamp(str(DL_DIR / rec))
        assert fields["fs"] == SRC_HZ, (rec, fields["fs"])
        assert LEAD in fields["sig_name"], (rec, fields["sig_name"])
        ann = wfdb.rdann(str(DL_DIR / rec), "atr")
        w, labels, r, skipped = convert_record(signal[:, fields["sig_name"].index(LEAD)], ann.sample, ann.symbol)
        n = len(labels)
        X.append(w)
        y += labels
        rr += r
        rec_ids += [int(rec[1:])] * n
        pat_ids += [patients[rec]] * n
        checks["records"][rec] = {"patient": patients[rec], "beats": n, "by_class": dict(Counter(labels)),
                                  "skipped": dict(skipped)}
        checks["skipped"].update(skipped)
    known = [v for v in rr if v is not None]
    median_rr = float(np.median(known))
    rr_arr = np.array([v if v is not None else median_rr for v in rr], dtype=np.float32)
    X = np.concatenate(X, axis=0)
    y_arr = np.array([LABEL_TO_INT[l] for l in y], dtype=np.int64)
    assert X.shape[1] == 2 * WINDOW_HALF and len(X) == len(y_arr) == len(rr_arr) == len(rec_ids)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, features=X, labels=y_arr, rr_interval_ms=rr_arr, record_ids=np.array(rec_ids, dtype=np.int64),
             patient_ids=np.array(pat_ids, dtype=np.int64))
    checks["skipped"] = dict(checks["skipped"])
    checks["totals"] = {"beats": int(len(y_arr)), "recordings": len(records), "patients": len(set(pat_ids)),
                        "by_class": dict(Counter(y)), "median_rr_ms_fill": median_rr,
                        "rr_gaps_over_12s": int((rr_arr > 12000).sum())}
    from p1_io import save_json_atomic
    save_json_atomic(CHECKS, checks)
    print(json.dumps(checks["totals"], indent=1))
    print("skipped:", checks["skipped"])


if __name__ == "__main__":
    main()
