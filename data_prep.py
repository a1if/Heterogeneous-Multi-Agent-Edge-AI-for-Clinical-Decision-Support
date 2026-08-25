"""
data_prep.py

Downloads the MIT-BIH Arrhythmia Database (PhysioNet, Open Access / ODC-By,
no credentialing required) and produces windowed, labeled, normalized
training/test arrays for the Perception Agent (CNN-LSTM).

Reconstructed from spec (KB Section 3.2, 11.2) after environment reset —
recovery procedure per KB Section 20.1.

MUST be run locally with network access to PhysioNet — not reachable from
sandboxed/CI environments.

Run:
    python data_prep.py

Produces:
    data/processed/ds1_train.npz  (features, labels)
    data/processed/ds2_test.npz   (features, labels)
"""
import os
import numpy as np
import wfdb

# ---------------------------------------------------------------------------
# de Chazal et al. (2004) inter-patient DS1/DS2 split.
# 44 records total (4 paced-beat records — 102, 104, 107, 217 — excluded,
# consistent with the "44 records (~100K heartbeats)" figure used elsewhere
# in this project's own scoping discussion).
# ---------------------------------------------------------------------------
DS1_RECORDS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
               201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_RECORDS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
               213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# AAMI EC57 5-class mapping from MIT-BIH annotation symbols.
# Symbols not in this map are skipped (not silently mapped to a default class).
AAMI_MAP = {
    # N — Normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S — Supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # V — Ventricular ectopic
    "V": "V", "E": "V",
    # F — Fusion
    "F": "F",
    # Q — Unknown/paced/unclassifiable
    "P": "Q", "/": "Q", "f": "Q", "u": "Q",
}
LABEL_TO_INT = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}

WINDOW_HALF = 180          # samples either side of R-peak -> 360-sample window
SAMPLE_RATE_HZ = 360
DL_DIR = "data/mitdb"
OUT_DIR = "data/processed"


def download_database():
    os.makedirs(DL_DIR, exist_ok=True)
    print(f"Downloading MIT-BIH Arrhythmia Database to {DL_DIR} ...")
    wfdb.dl_database("mitdb", dl_dir=DL_DIR)
    print("Download complete.")


def extract_windows_for_record(record_id):
    """Returns (windows, labels, rr_intervals_ms) for one record.
    rr_interval_ms is computed from the true annotation sample gap to the
    PREVIOUS accepted beat in this record — real timing, not a placeholder.
    First accepted beat in each record has no previous beat; uses None,
    filtered out by the caller (or defaulted downstream for synthetic use)."""
    record_path = os.path.join(DL_DIR, str(record_id))
    signal, fields = wfdb.rdsamp(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    lead_names = fields["sig_name"]
    lead_idx = lead_names.index("MLII") if "MLII" in lead_names else 0
    lead_signal = signal[:, lead_idx]

    windows, labels, rr_intervals = [], [], []
    prev_accepted_sample_idx = None
    for sample_idx, symbol in zip(annotation.sample, annotation.symbol):
        if symbol not in AAMI_MAP:
            continue

        start = sample_idx - WINDOW_HALF
        end = sample_idx + WINDOW_HALF
        if start < 0 or end > len(lead_signal):
            continue

        window = lead_signal[start:end]
        if len(window) != 2 * WINDOW_HALF:
            continue

        mean, std = window.mean(), window.std()
        if std < 1e-8:
            continue
        normalized = (window - mean) / std

        rr_ms = (
            (sample_idx - prev_accepted_sample_idx) / SAMPLE_RATE_HZ * 1000.0
            if prev_accepted_sample_idx is not None else None
        )
        prev_accepted_sample_idx = sample_idx

        windows.append(normalized)
        labels.append(AAMI_MAP[symbol])
        rr_intervals.append(rr_ms)

    return np.array(windows, dtype=np.float32), labels, rr_intervals


def build_split(record_ids, split_name):
    all_windows, all_labels, all_rr, all_record_ids = [], [], [], []
    for record_id in record_ids:
        print(f"  [{split_name}] processing record {record_id} ...")
        windows, labels, rr_intervals = extract_windows_for_record(record_id)
        all_windows.append(windows)
        all_labels.extend(labels)
        all_rr.extend(rr_intervals)
        all_record_ids.extend([record_id] * len(labels))
    X = np.concatenate(all_windows, axis=0)
    y = np.array([LABEL_TO_INT[l] for l in all_labels], dtype=np.int64)
    record_ids_arr = np.array(all_record_ids, dtype=np.int64)

    known_rr = [r for r in all_rr if r is not None]
    median_rr = float(np.median(known_rr)) if known_rr else 800.0
    rr = np.array([r if r is not None else median_rr for r in all_rr], dtype=np.float32)

    return X, y, rr, record_ids_arr


def main():
    if not os.path.exists(DL_DIR) or not os.listdir(DL_DIR):
        download_database()
    else:
        print(f"{DL_DIR} already populated — skipping download.")

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Building DS1 (train) ...")
    X_train, y_train, rr_train, rec_train = build_split(DS1_RECORDS, "DS1/train")
    print(f"DS1: {X_train.shape[0]} windows")
    np.savez(os.path.join(OUT_DIR, "ds1_train.npz"), features=X_train, labels=y_train,
              rr_interval_ms=rr_train, record_ids=rec_train)

    print("Building DS2 (test) ...")
    X_test, y_test, rr_test, rec_test = build_split(DS2_RECORDS, "DS2/test")
    print(f"DS2: {X_test.shape[0]} windows")
    np.savez(os.path.join(OUT_DIR, "ds2_test.npz"), features=X_test, labels=y_test,
              rr_interval_ms=rr_test, record_ids=rec_test)

    # Sanity checks matching KB 11.2's verified expectations
    assert X_train.shape[1] == 360, "Window length must be exactly 360 samples."
    assert abs(X_train[0].mean()) < 0.5, "Normalization sanity check: mean should be near 0."

    print("Done. Class distribution (train):")
    for label, idx in LABEL_TO_INT.items():
        count = int((y_train == idx).sum())
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
