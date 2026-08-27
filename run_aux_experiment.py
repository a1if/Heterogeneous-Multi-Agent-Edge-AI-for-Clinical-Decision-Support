"""
Full multi-seed pipeline for the auxiliary reconstruction objective
experiment, per aux_training_precommitted_analysis_plan.md: trains one
CNNLSTM + ReconstructionDecoder per seed, evaluates DS2 accuracy, re-runs the
seeded context-32d attribution probe on each resulting checkpoint, then
applies the plan's two-step decision criteria (accuracy gate, then probe
recall comparison per seed) to pick a branch.

Only meant to be run AFTER the single-seed sanity check (loss_recon
decreasing, accuracy within ~3pp of 85.4%) has already passed -- this script
does not repeat that check, it commits to the full multi-seed run.

Run:
    python run_aux_experiment.py --seeds 101 202 303 --lam 0.1
"""
import argparse
import json

import numpy as np

from day7_auditability_probe_aux import probe_context_vector
from train_perception_agent_aux import train_one_seed

BASELINE_ACCURACY = 85.4  # %, perception.accuracy.ds2.overall in results_ledger.json
BASELINE_PROBE_RECALL = 69.6  # %, context_linear in day7_auditability_results_v2.json
ACCURACY_GATE_DROP_PP = 3.0
RESULTS_PATH = "results/aux_reconstruction_results.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--lam", type=float, default=0.1)
    args = parser.parse_args()

    per_seed = []
    for seed in args.seeds:
        print(f"\n{'=' * 74}\nSeed {seed}: training\n{'=' * 74}")
        train_result = train_one_seed(seed, args.lam)

        print(f"\n{'=' * 74}\nSeed {seed}: attribution probe\n{'=' * 74}")
        probe_result = probe_context_vector(train_result["checkpoint"])

        per_seed.append({
            "seed": seed,
            "ds2_accuracy_pct": train_result["ds2_accuracy"] * 100.0,
            "final_train_loss_recon": train_result["final_train_loss_recon"],
            "first_train_loss_recon": train_result["first_train_loss_recon"],
            "probe_recall_mean_pct": probe_result["mean_accuracy"] * 100.0,
            "probe_recall_std_pct": probe_result["std_accuracy"] * 100.0,
            "checkpoint": train_result["checkpoint"],
        })

    accuracies = np.array([s["ds2_accuracy_pct"] for s in per_seed])
    recalls = np.array([s["probe_recall_mean_pct"] for s in per_seed])

    aux = {
        "n_seeds": len(args.seeds),
        "lam": args.lam,
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
        "probe_recall_mean": float(recalls.mean()),
        "probe_recall_std": float(recalls.std(ddof=1)) if len(recalls) > 1 else 0.0,
        "probe_recall_min": float(recalls.min()),
        "probe_recall_max": float(recalls.max()),
        "per_seed": per_seed,
    }

    # --- Step 1: accuracy gate --------------------------------------------
    accuracy_drop = BASELINE_ACCURACY - aux["accuracy_mean"]
    if accuracy_drop > ACCURACY_GATE_DROP_PP:
        branch = "C"
        branch_reason = (
            f"accuracy dropped {accuracy_drop:.1f}pp (> {ACCURACY_GATE_DROP_PP}pp gate): "
            f"{aux['accuracy_mean']:.1f}% vs baseline {BASELINE_ACCURACY}%"
        )
    else:
        # --- Step 2: probe recall comparison, per seed, not just average --
        # Checked in the plan's own order: A, then D (either of its two
        # triggering conditions), then B. D's conditions are checked with
        # "or", not "and" -- either one alone is enough, so within-one-SD
        # and mixed-across-seeds are not mutually exclusive routes to B.
        seed_sd = aux["probe_recall_std"] if aux["probe_recall_std"] > 0 else 1e-9
        all_above_one_sd = all(
            s["probe_recall_mean_pct"] > BASELINE_PROBE_RECALL + seed_sd for s in per_seed
        )
        all_at_or_below = all(
            s["probe_recall_mean_pct"] <= BASELINE_PROBE_RECALL for s in per_seed
        )
        within_one_sd = abs(aux["probe_recall_mean"] - BASELINE_PROBE_RECALL) <= seed_sd
        mixed = not all_above_one_sd and not all_at_or_below

        if all_above_one_sd:
            branch = "A"
            branch_reason = (
                f"every seed's probe recall exceeds baseline {BASELINE_PROBE_RECALL}% "
                f"by more than one seed-SD ({seed_sd:.1f}pp), consistently"
            )
        elif within_one_sd or mixed:
            branch = "D"
            reasons = []
            if within_one_sd:
                reasons.append(
                    f"mean {aux['probe_recall_mean']:.1f}% is within one seed-SD "
                    f"({seed_sd:.1f}pp) of baseline {BASELINE_PROBE_RECALL}%"
                )
            if mixed:
                reasons.append(
                    "improvement appears in some seeds but not others "
                    f"(range [{aux['probe_recall_min']:.1f}%, {aux['probe_recall_max']:.1f}%])"
                )
            branch_reason = "; ".join(reasons)
        else:
            branch = "B"
            branch_reason = (
                f"mean {aux['probe_recall_mean']:.1f}% at or below baseline "
                f"{BASELINE_PROBE_RECALL}%, consistently"
            )

    aux["branch"] = branch
    aux["branch_reason"] = branch_reason
    aux["baseline_accuracy"] = BASELINE_ACCURACY
    aux["baseline_probe_recall"] = BASELINE_PROBE_RECALL

    print(f"\n{'=' * 74}\nDECISION\n{'=' * 74}")
    print(f"Accuracy: {aux['accuracy_mean']:.2f}% +/- {aux['accuracy_std']:.2f}% "
          f"(baseline {BASELINE_ACCURACY}%, drop {accuracy_drop:.2f}pp)")
    print(f"Probe recall: {aux['probe_recall_mean']:.2f}% +/- {aux['probe_recall_std']:.2f}% "
          f"(baseline {BASELINE_PROBE_RECALL}%), "
          f"range [{aux['probe_recall_min']:.1f}%, {aux['probe_recall_max']:.1f}%]")
    print(f"Per-seed recall: "
          f"{[round(s['probe_recall_mean_pct'], 1) for s in per_seed]}")
    print(f"\n=> Branch {branch}: {branch_reason}")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(aux, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
