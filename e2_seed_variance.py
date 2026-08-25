"""E2: adapter training seed variance (2 new seeds + the existing checkpoint, all k=4).

Sprint plan §3, E2. Puts an error bar on the 95% accuracy / 24.7% token-reduction
headline. Only the adapter's random weight initialization is seed-dependent
(build_real_training_examples selects examples deterministically, no shuffling
during training -- see reasoning/adapter_training.py), so this isolates
sensitivity to initialization, not to data order.

The existing checkpoint (virtual_adapter_day5_larger.pt) was trained before
explicit seed control existed and stands as one data point ("seed: existing,
unrecorded") -- NOT retrained, so this experiment cannot perturb the k=4
headline (additive only). Two NEW seeds (101, 202) are trained fresh, same
config as the existing checkpoint (read back from its own saved config) except
for the seed.

Home: Results §4.x, as +/- on the headline table.
Fallback: "Results are reported from a single training seed; seed sensitivity
is not characterised (§6.x)."
Kill time: end of Day 5.

Run (from repo root):
    python e2_seed_variance.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from ablation_common import prepare_events, run_arm_b_eval, summarize_arm_b
from reasoning.adapter_arm import load_trained_adapter
from reasoning.adapter_training import TrainingConfig, train_adapter
from reasoning.model_loader import load_model

EXISTING_CHECKPOINT = Path("reasoning/checkpoints/virtual_adapter_day5_larger.pt")
DAY6_RESULTS = Path("results/day6_results.json")
NEW_SEEDS = [101, 202]
RESULTS_PATH = Path("results/e2_seed_variance_results.json")


def base_config_from_existing_checkpoint() -> TrainingConfig:
    ckpt = torch.load(EXISTING_CHECKPOINT, map_location="cpu", weights_only=True)
    saved = ckpt["config"]
    return TrainingConfig(
        per_class=saved["per_class"],
        epochs=saved["epochs"],
        learning_rate=saved["learning_rate"],
        loss_mode=saved["loss_mode"],
        tier_weight=saved["tier_weight"],
        max_examples=saved["max_examples"],
    )


def existing_summary_from_day6() -> dict:
    with open(DAY6_RESULTS, encoding="utf-8") as f:
        results = json.load(f)
    arm_b = [r for r in results if r["arm"] == "B"]
    return summarize_arm_b(arm_b)


def checkpoint_path_for_seed(seed: int) -> Path:
    return Path(f"reasoning/checkpoints/virtual_adapter_e2_seed{seed}.pt")


def aggregate_over(by_seed: dict) -> dict:
    # ddof=1 (sample std, Bessel's correction): these 3 seeds are a SAMPLE of
    # possible initializations, not the full population -- ddof=0 (NumPy's
    # default) understates the uncertainty, materially so at n=3 (~22% low here).
    accuracies = [v["accuracy"] for v in by_seed.values()]
    tokens = [v["tokens_mean"] for v in by_seed.values() if v["tokens_mean"] is not None]
    gen_ms = [v["gen_ms_mean"] for v in by_seed.values() if v["gen_ms_mean"] is not None]
    return {
        "n_seeds": len(by_seed),
        "accuracy_mean": float(np.mean(accuracies)), "accuracy_std": float(np.std(accuracies, ddof=1)),
        "tokens_mean": float(np.mean(tokens)) if tokens else None,
        "tokens_std": float(np.std(tokens, ddof=1)) if len(tokens) > 1 else None,
        "gen_ms_mean": float(np.mean(gen_ms)) if gen_ms else None,
        "gen_ms_std": float(np.std(gen_ms, ddof=1)) if len(gen_ms) > 1 else None,
    }


def save(base_config: TrainingConfig, by_seed: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "base_config": base_config.__dict__, "by_seed": by_seed,
            "aggregate": aggregate_over(by_seed),
        }, f, indent=2)


def main():
    base_config = base_config_from_existing_checkpoint()
    print(f"Base config (matched to the existing k=4 checkpoint): {base_config}")

    # Resume support: if a prior run saved partial results (e.g. crashed mid-seed),
    # keep everything it already has rather than re-evaluating from scratch.
    by_seed = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            by_seed = json.load(f)["by_seed"]
        print(f"Resuming: found existing results for seeds={list(by_seed)} in {RESULTS_PATH}")

    if "existing" not in by_seed:
        by_seed["existing"] = {
            "seed": "existing (unrecorded, pre-dates seed control)",
            "checkpoint": str(EXISTING_CHECKPOINT), "retrained": False,
            **existing_summary_from_day6(),
        }
        save(base_config, by_seed)

    prepared = prepare_events()
    model, processor = load_model()

    for seed in NEW_SEEDS:
        if str(seed) in by_seed:
            print(f"\nseed={seed}: already evaluated (in {RESULTS_PATH}), skipping "
                  f"(delete its entry there to force re-evaluation).")
            continue

        checkpoint_path = checkpoint_path_for_seed(seed)
        if checkpoint_path.exists():
            print(f"\nseed={seed}: checkpoint already exists at {checkpoint_path}, skipping training "
                  f"(delete it to force retraining).")
        else:
            print(f"\nTraining seed={seed} adapter (same config as the existing checkpoint, k=4)...")
            config = TrainingConfig(
                per_class=base_config.per_class, epochs=base_config.epochs,
                learning_rate=base_config.learning_rate, loss_mode=base_config.loss_mode,
                tier_weight=base_config.tier_weight, max_examples=base_config.max_examples,
                seed=seed,
            )
            train_adapter(config, output_path=checkpoint_path)

        adapter = load_trained_adapter(str(checkpoint_path), model)
        eval_results = run_arm_b_eval(prepared, model, processor, adapter, label=f"seed={seed}")
        by_seed[str(seed)] = {
            "seed": seed, "checkpoint": str(checkpoint_path), "retrained": True,
            **summarize_arm_b(eval_results),
        }
        save(base_config, by_seed)  # incremental: survives a later seed's crash
        print(f"seed={seed} done, saved to {RESULTS_PATH}")

    aggregate = aggregate_over(by_seed)
    print(f"\n{'='*70}\nE2 SEED VARIANCE SUMMARY\n{'='*70}")
    for key, r in by_seed.items():
        tokens = f"{r['tokens_mean']:.1f}" if r["tokens_mean"] is not None else "n/a"
        gen_ms = f"{r['gen_ms_mean']:.1f}" if r["gen_ms_mean"] is not None else "n/a"
        print(f"seed={key}: accuracy={r['accuracy']*100:.1f}% (n_failed={r.get('n_failed', 0)}) "
              f"tokens={tokens} gen_ms={gen_ms}")
    if aggregate["tokens_mean"] is not None:
        print(f"\nAggregate (n={aggregate['n_seeds']}): "
              f"accuracy={aggregate['accuracy_mean']*100:.1f}% +/- {aggregate['accuracy_std']*100:.1f}pp, "
              f"tokens={aggregate['tokens_mean']:.1f} +/- {aggregate['tokens_std']:.1f}")
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
