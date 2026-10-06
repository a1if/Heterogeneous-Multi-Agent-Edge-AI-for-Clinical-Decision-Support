"""Phase 1 step 2 (dissertation §5.3 / §5.9): the ECG model's confidence and
top-3 for the events Arm B gets wrong. CPU only, no generation.

Hypothesis from §5.3: Arm A's JSON carries the ECG model's confidence and top-3
explicitly, while the adapter's input (the pre-classification context vector)
carries them only implicitly, so Arm B should fail on borderline events. The
reference tier makes one borderline case concrete: a predicted V/F beat is
"urgent" iff confidence > 0.85 (or a run of >= 3 abnormal beats), so Arm B must
recover confidence against a hard threshold. Tested on three k=4 checkpoints
over the same E80 events:
  - headline (unseeded), from the reported Day 6 pass
  - E2 seeds 101 and 202, parsed from logs/e1_e2_run.log (no per-event file was
    saved); parsed counts must match results/e2_seed_variance_results.json

Run (from repo root):
    python p1_step2_contested_events.py
"""
import json
import re
from collections import Counter

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu

from ablation_common import prepare_events
from reasoning.adapter_training import DEFAULT_DATASET, build_real_training_examples
from reasoning.training_targets import urgency_tier_from_event

DAY6_REPORTED_PASS = "results/day6_results.json.bak_pre_rerun_20260816"
E2_LOG = "logs/e1_e2_run.log"
E2_SUMMARY = "results/e2_seed_variance_results.json"
THRESHOLD = 0.85
NEAR = 0.10  # "near threshold": predicted V/F with |confidence - 0.85| <= NEAR
RESULTS_PATH = "results/p1_step2_contested_events.json"
LINE = re.compile(r"^\[seed=(\d+)\]\[\d+/80\] idx=(\d+) true=\w (?:B=(\w+) ref=(\w+)|GENERATION FAILED)")


def rule_path(ev: dict) -> str:
    label = ev["classification"]["label"]
    if ev["clinical_flags"]["requires_urgent_review"]:
        return "urgent_run" if ev["clinical_flags"]["consecutive_abnormal_beats"] >= 3 else "urgent_confidence"
    return "routine_N" if label == "N" else "priority"


def outcomes() -> dict:
    """checkpoint -> {idx: (arm_b_tier or None if generation failed, correct)}"""
    out = {"headline": {}}
    with open(DAY6_REPORTED_PASS, encoding="utf-8") as f:
        for r in json.load(f):
            if r["arm"] == "B":
                out["headline"][r["idx"]] = (r["urgency_tier"], r["correct"])
    with open(E2_LOG, encoding="utf-8", errors="replace") as f:
        text = f.read().replace("\r", "\n")
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            seed, idx, b, ref = m.groups()
            out.setdefault(f"seed{seed}", {})[int(idx)] = (b, b is not None and b == ref)
    with open(E2_SUMMARY, encoding="utf-8") as f:
        summary = json.load(f)["by_seed"]
    for seed in ("101", "202"):
        got = out[f"seed{seed}"]
        correct = sum(c for _, c in got.values())
        if len(got) != 80 or correct != summary[seed]["correct"]:
            raise RuntimeError(f"seed {seed}: parsed {len(got)} events / {correct} correct, "
                               f"summary says 80 / {summary[seed]['correct']}")
    return out


def main():
    events = {e["idx"]: e for e in prepare_events(note=" (E80, step 2)")}
    rows = {}
    for idx, e in events.items():
        ev = e["health_event"]
        c = ev["classification"]
        top3 = c["top_3"]
        rows[idx] = {
            "idx": idx, "record_id": e["record_id"], "true_class": e["true_class"],
            "predicted_class": c["label"], "upstream_correct": c["label"] == e["true_class"],
            "confidence": c["confidence"], "top_3": top3,
            "top3_margin": top3[0]["confidence"] - top3[1]["confidence"],
            "consecutive_abnormal_beats": ev["clinical_flags"]["consecutive_abnormal_beats"],
            "reference_tier": e["reference_tier"], "rule_path": rule_path(ev),
            "near_threshold": c["label"] in ("V", "F") and abs(c["confidence"] - THRESHOLD) <= NEAR,
        }

    results = {"threshold": THRESHOLD, "near_window": NEAR, "checkpoints": {}}
    for name, outs in outcomes().items():
        failed = [i for i, (b, _) in outs.items() if b is None]
        wrong = [i for i, (b, c) in outs.items() if b is not None and not c]
        right = [i for i, (b, c) in outs.items() if c]
        conf_w = [rows[i]["confidence"] for i in wrong]
        conf_r = [rows[i]["confidence"] for i in right]
        near = [i for i in rows if rows[i]["near_threshold"] and i not in failed]
        table = [[sum(i in wrong for i in near), sum(i in right for i in near)],
                 [sum(i in wrong for i in rows if i not in near and i not in failed),
                  sum(i in right for i in rows if i not in near)]]
        results["checkpoints"][name] = {
            "n_wrong": len(wrong), "n_generation_failed": len(failed),
            "wrong_events": [dict(rows[i], arm_b_tier=outs[i][0]) for i in sorted(wrong)],
            "wrong_by_rule_path": dict(Counter(rows[i]["rule_path"] for i in wrong)),
            "right_by_rule_path": dict(Counter(rows[i]["rule_path"] for i in right)),
            "wrong_upstream_correct": sum(rows[i]["upstream_correct"] for i in wrong),
            "confidence_median_wrong_vs_right": [float(np.median(conf_w)) if conf_w else None,
                                                 float(np.median(conf_r))],
            "mannwhitney_confidence_p": float(mannwhitneyu(conf_w, conf_r).pvalue) if len(conf_w) > 1 else None,
            "near_threshold_2x2_[[near_wrong,near_right],[other_wrong,other_right]]": table,
            "fisher_near_threshold_p": float(fisher_exact(table).pvalue),
            "tier_confusions": dict(Counter(f"ref={rows[i]['reference_tier']}->B={outs[i][0]}" for i in wrong)),
        }
        r = results["checkpoints"][name]
        print(f"\n{name}: {len(wrong)} wrong, {len(failed)} failed | upstream class correct on "
              f"{r['wrong_upstream_correct']}/{len(wrong)} wrong")
        print(f"  confidence median wrong {r['confidence_median_wrong_vs_right'][0]} vs right "
              f"{r['confidence_median_wrong_vs_right'][1]:.3f} (MWU p={r['mannwhitney_confidence_p']})")
        print(f"  near-threshold V/F: {table[0][0]}/{sum(table[0])} wrong vs other {table[1][0]}/{sum(table[1])} "
              f"(Fisher p={r['fisher_near_threshold_p']:.3g})")
        print(f"  wrong by rule path {r['wrong_by_rule_path']} | right by rule path {r['right_by_rule_path']}")
        print(f"  tier confusions {r['tier_confusions']}")

    # Where the errors concentrate (rule path) vs how often the adapter saw that
    # tier in training: the 64 examples are balanced on TRUE class, not on tier.
    train = build_real_training_examples(DEFAULT_DATASET, per_class=16)
    results["adapter_training_set"] = {
        "tiers": dict(Counter(urgency_tier_from_event(e["health_event"]) for e in train)),
        "predicted_labels": dict(Counter(e["health_event"]["classification"]["label"] for e in train)),
    }
    pri = [i for i in rows if rows[i]["rule_path"] == "priority"]
    results["priority_path"] = {
        "n_events": len(pri),
        "wrong_by_checkpoint": {name: sum(i in {w["idx"] for w in c["wrong_events"]} for i in pri)
                                for name, c in results["checkpoints"].items()},
    }
    print(f"\npriority-path events wrong per checkpoint: {results['priority_path']['wrong_by_checkpoint']} "
          f"of {len(pri)}; adapter training tiers {results['adapter_training_set']['tiers']}")
    results["events"] = [rows[i] for i in sorted(rows)]
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
