"""Quick check: how many S-class events actually exist in DS2, unconstrained,
and how are they distributed across records? Determines whether a targeted
S-only oversample is feasible for the auditability comparison, or whether S
hits the same per-record ceiling that already caps F at 20.

No GPU needed, no model loading. Run:
    python check_s_class_headroom.py
"""
import numpy as np
from collections import Counter
from project_config import DS2_PATH, MAX_PER_RECORD

CLASS_NAMES = {0: "N", 1: "S", 2: "V", 3: "F", 4: "Q"}  # confirm against your actual encoding


def main():
    data = np.load(DS2_PATH)
    y, record_ids = data["labels"], data["record_ids"]

    print("Total available events per class (unconstrained by any cap):")
    class_totals = Counter(y)
    for idx, name in CLASS_NAMES.items():
        print(f"  {name}: {class_totals.get(idx, 0)}")

    print(f"\nPer-record distribution under the existing {MAX_PER_RECORD}-per-record cap:")
    for idx, name in CLASS_NAMES.items():
        mask = (y == idx)
        recs = record_ids[mask]
        rec_counts = Counter(recs)
        n_records_with_class = len(rec_counts)
        capped_total = sum(min(c, MAX_PER_RECORD) for c in rec_counts.values())
        print(f"  {name}: {n_records_with_class} records contain this class | "
              f"raw total={class_totals.get(idx, 0)} | "
              f"capped total (max {MAX_PER_RECORD}/record)={capped_total}")

    print("\nInterpretation: compare S's 'capped total' against F's. If S's capped")
    print("total is meaningfully above 20, there's real headroom for a targeted")
    print("S-only oversample without changing the N/V/F=80 anchor. If S's capped")
    print("total is close to 20 (like F's), it's already near its own ceiling and")
    print("the current N=20 comparison is close to the best available at this")
    print("record-level design -- report as-is, no further action needed.")


if __name__ == "__main__":
    main()
