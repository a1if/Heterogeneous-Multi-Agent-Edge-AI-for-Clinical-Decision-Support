"""E1: virtual-token-count ablation (k in {1, 2, 8} vs. the existing k=4).

Sprint plan §3, E1. Converts the single k=4 trade-off point into a compression-
ratio-vs-accuracy curve. All controls held identical to the existing headline
checkpoint (reasoning/checkpoints/virtual_adapter_day5_larger.pt) except k:
per_class=16, epochs=3, learning_rate=0.002, loss_mode='full', tier_weight=4.0
(read back from that checkpoint's own saved config, not retyped by hand).

k=4 is NOT retrained -- the existing checkpoint and its results/day6_results.json
Arm B numbers are reused as-is, so this experiment cannot perturb the k=4
headline (additive only, per the sprint plan's experiment contract).

TWO MODES
---------
Unseeded (default, `python e1_ablation.py`): the original run. Each k trained
with NO explicit seed, i.e. whatever global RNG state happened to be current --
three independent, unrecorded initializations. Already complete; results in
results/e1_ablation_results.json. Left runnable only for provenance.

Seeded (`python e1_ablation.py --seed 101`): all k values share ONE training
seed, so initialization stops being a free variable across the curve. Writes to
SEPARATE checkpoint and results paths so it can never collide with, or be
mistaken for, the unseeded series. In this mode the k=4 point is taken from
E2's seed=101 run (reasoning/checkpoints/virtual_adapter_e2_seed101.pt, k=4,
same per_class/epochs/lr) rather than the headline checkpoint -- because the
headline checkpoint's own seed is unrecorded and would reintroduce exactly the
confound this mode exists to remove.

WHAT THE SEEDED CURVE DOES AND DOES NOT ESTABLISH -- read before writing prose.
It controls ONE confound (uncontrolled initialization across k). It does NOT
give multiple seeds per k, so it cannot separate "k=4 is genuinely better" from
"k=4 happens to be good under seed 101 specifically." E2 measured pure seed
noise at fixed k=4 as std 8.7pp (ddof=1, correcting an earlier ddof=0 undercount
of 7.1pp) / range 16.2pp, which is the same order as
E1's adjacent-k deltas. The supportable claim is therefore "directionally
consistent under a matched-seed check", NOT "accuracy depends on k". Do not let
a cleaner-looking curve silently upgrade that claim.

Home: Results §4.x "Compression ratio ablation"; new figure.
Fallback: "The number of virtual tokens was fixed at four throughout;
systematically varying this parameter to trace the compression-accuracy
frontier is identified as future work (§6.x)."
Kill time: end of Day 3.

Run (from repo root):
    python e1_ablation.py --seed 101
"""
import argparse
import json
from pathlib import Path

import torch

from ablation_common import prepare_events, run_arm_b_eval, summarize_arm_b
from reasoning.adapter_arm import load_trained_adapter
from reasoning.adapter_training import TrainingConfig, train_adapter
from reasoning.model_loader import load_model
from project_config import DAY6_RESULTS as DAY6_RESULTS_STR

EXISTING_K4_CHECKPOINT = Path("reasoning/checkpoints/virtual_adapter_day5_larger.pt")
DAY6_RESULTS = Path(DAY6_RESULTS_STR)
E2_RESULTS = Path("results/e2_seed_variance_results.json")
ABLATION_K_VALUES = [1, 2, 8]


def base_config_from_existing_checkpoint() -> TrainingConfig:
    """Reads the exact hyperparameters used for the k=4 checkpoint, so the
    ablation varies ONLY num_tokens -- everything else stays identical."""
    ckpt = torch.load(EXISTING_K4_CHECKPOINT, map_location="cpu", weights_only=True)
    saved = ckpt["config"]
    return TrainingConfig(
        per_class=saved["per_class"],
        epochs=saved["epochs"],
        learning_rate=saved["learning_rate"],
        loss_mode=saved["loss_mode"],
        tier_weight=saved["tier_weight"],
        max_examples=saved["max_examples"],
    )


def k4_anchor(seed: int | None) -> dict:
    """The k=4 point of the curve, never retrained by this script.

    Unseeded mode: the headline checkpoint's Arm B numbers from Day 6.
    Seeded mode: E2's run at the SAME seed (also k=4, same per_class/epochs/lr),
    because the headline checkpoint's own initialization seed is unrecorded --
    anchoring a matched-seed curve to it would reintroduce the confound.
    """
    if seed is None:
        with open(DAY6_RESULTS, encoding="utf-8") as f:
            results = json.load(f)
        summary = summarize_arm_b([r for r in results if r["arm"] == "B"])
        return {"k": 4, "checkpoint": str(EXISTING_K4_CHECKPOINT), "retrained": False,
                "seed": None, "source": "day6_run_comparison.py Arm B", **summary}

    with open(E2_RESULTS, encoding="utf-8") as f:
        by_seed = json.load(f)["by_seed"]
    if str(seed) not in by_seed:
        raise RuntimeError(
            f"Seeded mode needs a k=4 point at seed={seed}, but results/e2_seed_variance_results.json "
            f"has only seeds {list(by_seed)}. Run e2_seed_variance.py for this seed first, or pick a "
            f"seed it already covers -- do NOT silently fall back to the headline checkpoint, whose "
            f"seed is unrecorded."
        )
    r = dict(by_seed[str(seed)])
    r.update({"k": 4, "retrained": False, "seed": seed,
              "source": "e2_seed_variance.py (reused; NOT the headline checkpoint)"})
    return r


def checkpoint_path_for_k(k: int, seed: int | None) -> Path:
    suffix = "" if seed is None else f"_seed{seed}"
    return Path(f"reasoning/checkpoints/virtual_adapter_e1_k{k}{suffix}.pt")


def results_path_for(seed: int | None, suffix: str = "") -> Path:
    if seed is None:
        return Path(f"results/e1_ablation_results{suffix}.json")
    return Path(f"results/e1_seeded_ablation_results_seed{seed}{suffix}.json")


def save(base_config: TrainingConfig, ablation_results: dict, seed: int | None,
         suffix: str = "") -> None:
    path = results_path_for(seed, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"base_config": base_config.__dict__, "shared_seed": seed,
                    "by_k": ablation_results}, f, indent=2)


def main(seed: int | None, suffix: str = ""):
    base_config = base_config_from_existing_checkpoint()
    results_path = results_path_for(seed, suffix)
    mode = "UNSEEDED (legacy)" if seed is None else f"SEEDED, shared seed={seed}"
    print(f"Mode: {mode}")
    print(f"Base config (matched to the k=4 checkpoint): {base_config}")
    print(f"Writing to: {results_path}")

    # Resume support: if a prior run saved partial results (e.g. crashed mid-k),
    # keep everything it already has rather than re-evaluating from scratch.
    ablation_results = {}
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            ablation_results = {int(k): v for k, v in json.load(f)["by_k"].items()}
        print(f"Resuming: found existing results for k={sorted(ablation_results)} in {results_path}")

    if 4 not in ablation_results:
        ablation_results[4] = k4_anchor(seed)
        save(base_config, ablation_results, seed, suffix)
        print(f"k=4 anchor: {ablation_results[4]['source']} "
              f"(accuracy={ablation_results[4]['accuracy']*100:.1f}%)")

    prepared = prepare_events()
    model, processor = load_model()

    for k in ABLATION_K_VALUES:
        if k in ablation_results:
            print(f"\nk={k}: already evaluated (in {results_path}), skipping "
                  f"(delete its entry there to force re-evaluation).")
            continue

        checkpoint_path = checkpoint_path_for_k(k, seed)
        if checkpoint_path.exists():
            print(f"\nk={k}: checkpoint already exists at {checkpoint_path}, skipping training "
                  f"(delete it to force retraining).")
        else:
            print(f"\nTraining k={k} adapter (num_tokens={k}, seed={seed}, all else matched to k=4)...")
            config = TrainingConfig(
                per_class=base_config.per_class, epochs=base_config.epochs,
                learning_rate=base_config.learning_rate, loss_mode=base_config.loss_mode,
                tier_weight=base_config.tier_weight, max_examples=base_config.max_examples,
                num_tokens=k, seed=seed,
            )
            train_adapter(config, output_path=checkpoint_path)

        adapter = load_trained_adapter(str(checkpoint_path), model)
        eval_results = run_arm_b_eval(prepared, model, processor, adapter, label=f"k={k}")
        ablation_results[k] = {
            "k": k, "checkpoint": str(checkpoint_path), "retrained": True, "seed": seed,
            "source": "e1_ablation.py", **summarize_arm_b(eval_results),
            # Keep the raw per-event records. Previously only the summary survived,
            # which is why the k=1 latency anomaly could not be diagnosed after the
            # fact without a full re-run -- parse_attempts/output_tokens existed at
            # this point and were thrown away here.
            "events": eval_results,
        }
        save(base_config, ablation_results, seed, suffix)  # incremental: survives a later k's crash
        print(f"k={k} done, saved to {results_path}")

        # Multiple k's train sequentially in this one process; each training run's
        # allocator state (variable-length examples -> variable tensor shapes) can
        # fragment free VRAM enough to OOM a LATER k even with total free memory to
        # spare. Confirmed happening in practice (k=1 OOM'd on step 51/64 of its own
        # first epoch, not even a later k) -- clear the cache between iterations.
        del adapter
        torch.cuda.empty_cache()

    print(f"\n{'='*70}\nE1 ABLATION SUMMARY ({mode})\n{'='*70}")
    for k in sorted(ablation_results):
        r = ablation_results[k]
        tokens = f"{r['tokens_mean']:.1f}" if r["tokens_mean"] is not None else "n/a"
        gen_ms = f"{r['gen_ms_mean']:.1f}" if r["gen_ms_mean"] is not None else "n/a"
        print(f"k={k}: accuracy={r['accuracy']*100:.1f}% (n_failed={r.get('n_failed', 0)}) "
              f"tokens={tokens} gen_ms={gen_ms}")
    if seed is not None:
        print("\nREMINDER: one shared seed controls initialization across k, but gives n=1 per k. "
              "Supportable claim = 'directionally consistent under a matched-seed check', "
              "NOT 'accuracy depends on k'. E2's seed noise at fixed k=4 was std 8.7pp / range 16.2pp.")
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None,
                        help="Shared training seed for ALL k values. Omit for the legacy unseeded run.")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix for the results filename, so a diagnostic re-run cannot "
                             "overwrite the ledger-cited results/e1_ablation_results.json.")
    args = parser.parse_args()
    main(args.seed, args.output_suffix)
