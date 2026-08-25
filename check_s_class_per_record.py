"""Tests whether the N=20 -> N=63 S-class recall drop (s_class_expanded_check.py)
is explained by WHICH DS2 RECORDS the events come from, rather than uniform
per-event noise spread evenly across records.

Needs s_class_per_event_original20.npy and s_class_per_event_new43.npy
(both written by s_class_expanded_check.py -- run that first if missing).

No GPU needed: this is pure post-hoc analysis of already-saved per-event
correctness, not a new evaluation.

"Clean record-level split" here means: grouping each S-class event by its DS2
record_id and averaging correctness within each record, do the original-20's
records and the new-43's records separate into two non-overlapping accuracy
bands, or do individual records interleave across the full range regardless
of which group they belong to? The former names a specific, checkable
mechanism (some patient records are harder than others; the original 20
happened to sample easier ones); the latter would mean the recall drop is
just aggregate noise with no record-level structure worth naming.

Run:
    python check_s_class_per_record.py
"""
import json

import numpy as np

ORIGINAL20_PATH = "cache/s_class_per_event_original20.npy"
NEW43_PATH = "cache/s_class_per_event_new43.npy"
RESULTS_PATH = "results/s_class_per_record_results.json"


def per_record_summary(events: np.ndarray) -> dict:
    """record_id -> {n, classifier_acc, probe_acc}."""
    summary = {}
    for record_id in sorted(set(events["record_id"].tolist())):
        mask = events["record_id"] == record_id
        n = int(mask.sum())
        summary[int(record_id)] = {
            "n": n,
            "classifier_acc": float(events["classifier_correct"][mask].mean()),
            "probe_acc": float(events["probe_correct"][mask].mean()),
        }
    return summary


def main():
    original20 = np.load(ORIGINAL20_PATH)
    new43 = np.load(NEW43_PATH)

    orig_records = per_record_summary(original20)
    new_records = per_record_summary(new43)

    shared = set(orig_records) & set(new_records)
    print(f"Original 20: {len(original20)} events across {len(orig_records)} records")
    print(f"New 43:      {len(new43)} events across {len(new_records)} records")
    print(f"Records appearing in BOTH groups: {len(shared)} {sorted(shared) if shared else ''}")

    def report_group(name, records):
        print(f"\n{name} per-record:")
        for rid, s in sorted(records.items()):
            print(f"  record {rid}: n={s['n']} classifier_acc={s['classifier_acc']:.2f} "
                  f"probe_acc={s['probe_acc']:.2f}")

    report_group("Original 20", orig_records)
    report_group("New 43", new_records)

    for metric in ("classifier_acc", "probe_acc"):
        orig_vals = [s[metric] for s in orig_records.values()]
        new_vals = [s[metric] for s in new_records.values()]
        orig_min, orig_max = min(orig_vals), max(orig_vals)
        new_min, new_max = min(new_vals), max(new_vals)
        clean_split = orig_min >= new_max  # every original-20 record >= every new-43 record
        print(f"\n{metric}: original-20 records range [{orig_min:.2f}, {orig_max:.2f}] "
              f"(mean {np.mean(orig_vals):.2f}), new-43 records range [{new_min:.2f}, {new_max:.2f}] "
              f"(mean {np.mean(new_vals):.2f})")
        print(f"  Clean split (worst original-20 record >= best new-43 record)? {clean_split}")

    results = {
        "original20_per_record": orig_records,
        "new43_per_record": new_records,
        "shared_records": sorted(shared),
        "clean_split": {
            metric: {
                "original20_range": [min(s[metric] for s in orig_records.values()),
                                       max(s[metric] for s in orig_records.values())],
                "new43_range": [min(s[metric] for s in new_records.values()),
                                  max(s[metric] for s in new_records.values())],
                "original20_mean": float(np.mean([s[metric] for s in orig_records.values()])),
                "new43_mean": float(np.mean([s[metric] for s in new_records.values()])),
                "clean_split": bool(min(s[metric] for s in orig_records.values())
                                     >= max(s[metric] for s in new_records.values())),
            }
            for metric in ("classifier_acc", "probe_acc")
        },
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
