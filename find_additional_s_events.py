"""Identify the ~43 additional S-class events available under the existing
5-per-record cap, beyond the 20 already used everywhere else in this project.

Does NOT reselect S-class events from scratch -- that risks swapping which
specific 20 are used, breaking consistency with the main auditability
headline (day7_auditability_results_v2.json) and the existing S-class
matched comparison. Instead: get the original 20 via the normal
select_events() call (unchanged), separately enumerate all S-class events
available under the same cap, and take the set difference.

No GPU needed. Run:
    python find_additional_s_events.py
"""
import numpy as np
from collections import Counter

from day7_auditability_probe import select_events  # unchanged -- gives the original 80

DS2_PATH = "data/processed/ds2_test.npz"
S_CLASS_IDX = 1  # confirm against your actual encoding
MAX_PER_RECORD = 5  # matches select_events()'s own cap


def all_s_events_under_cap(y, record_ids):
    """Same per-record cap logic as select_events(), applied to S-class only,
    to enumerate the full available pool (not just the balanced-selection 20)."""
    s_indices = np.where(y == S_CLASS_IDX)[0]
    per_record = {}
    for idx in s_indices:
        rec = record_ids[idx]
        per_record.setdefault(rec, []).append(idx)
    kept = []
    for rec, idxs in per_record.items():
        kept.extend(idxs[:MAX_PER_RECORD])  # confirm this matches select_events()'s
                                              # own tie-breaking rule (e.g. chronological
                                              # order vs arbitrary) -- adjust if it differs
    return sorted(kept)


def main():
    data = np.load(DS2_PATH)
    y, record_ids = data["labels"], data["record_ids"]

    original_80 = select_events(y, record_ids)
    original_s = sorted([i for i in original_80 if y[i] == S_CLASS_IDX])
    print(f"Original S-class events already in use: {len(original_s)}")

    full_s_pool = all_s_events_under_cap(y, record_ids)
    print(f"Full S-class pool under the {MAX_PER_RECORD}-per-record cap: {len(full_s_pool)}")

    additional_s = sorted(set(full_s_pool) - set(original_s))
    print(f"Additional S-class events available (not already used): {len(additional_s)}")

    # Sanity check: original 20 should be a strict subset of the full pool.
    # If this fails, select_events()'s tie-breaking rule differs from the one
    # used above -- do not proceed until reconciled, or you risk silently
    # using a different "original 20" than everything else in the project.
    if not set(original_s).issubset(set(full_s_pool)):
        missing = set(original_s) - set(full_s_pool)
        print(f"\nWARNING: {len(missing)} of the original 20 S-class events are NOT in "
              f"the recomputed full pool. Tie-breaking rule mismatch -- STOP and reconcile "
              f"against select_events()'s actual per-record selection logic before using "
              f"'additional_s' for anything.")
        return

    print(f"\nExpanded S-class set: {len(original_s)} original + {len(additional_s)} new "
          f"= {len(original_s) + len(additional_s)} total")
    np.save("cache/additional_s_class_indices.npy", np.array(additional_s))
    print("Saved additional event indices to cache/additional_s_class_indices.npy")


if __name__ == "__main__":
    main()
