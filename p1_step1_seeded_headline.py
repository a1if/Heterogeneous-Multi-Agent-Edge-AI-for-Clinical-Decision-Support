"""Phase 1, step 1: seeded headline adapter, training-determinism check, and the
E60 invariance check (offline). Pre-specified in docs/analysis_plan.md (sections 2, 4, 6).

Stages (saved atomically as they progress; rerunning resumes from the results file.
Evaluation is checkpointed after every event; a seed's ~30-min training restarts
from the beginning if interrupted):
  train     -- train k=4 adapters for seeds 101..505 with the headline config
               (read back from virtual_adapter_day5_larger.pt's own saved config)
               into reasoning/checkpoints/p1_k4_seed{seed}.pt. Existing
               virtual_adapter_* checkpoints are never overwritten.
  determinism -- max |dw| between p1 seeds 101/202 and the E2 checkpoints of the
               same seed (gate 6a: <= 1e-4).
  eval_e80   -- Arm B on the standard 80-event set for every seed (gate 6b).
  e60        -- offline: E60 (15/class) is a strict subset of E80, so its
               figures are computed from E80 per-event results, no GPU
               (Deviation 1 in the analysis plan; gate 6c).

Run (from repo root):
    python p1_step1_seeded_headline.py
"""
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from ablation_common import prepare_events, run_arm_b_eval, summarize_arm_b
from p1_io import save_json_atomic
from project_config import DS2_PATH, select_events
from reasoning.adapter_arm import load_trained_adapter
from reasoning.adapter_training import TrainingConfig, train_adapter
from reasoning.model_loader import load_model

SEEDS = [101, 202, 303, 404, 505]
REFERENCE_SEED = 101
HEADLINE_CHECKPOINT = Path("reasoning/checkpoints/virtual_adapter_day5_larger.pt")
E2_CHECKPOINTS = {101: Path("reasoning/checkpoints/virtual_adapter_e2_seed101.pt"),
                  202: Path("reasoning/checkpoints/virtual_adapter_e2_seed202.pt")}
RESULTS_PATH = Path("results/p1_step1_seeded_headline.json")
DAY6_REPORTED_PASS = Path("results/day6_results.json.bak_pre_rerun_20260816")  # the pass the dissertation cites
DETERMINISM_TOL = 1e-4


def checkpoint_path(seed: int) -> Path:
    return Path(f"reasoning/checkpoints/p1_k4_seed{seed}.pt")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    import transformers
    import bitsandbytes
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    return {
        "git_commit": commit, "git_dirty": dirty,
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def headline_config(seed: int) -> TrainingConfig:
    cfg = torch.load(HEADLINE_CHECKPOINT, map_location="cpu", weights_only=False)["config"]
    return TrainingConfig(
        per_class=cfg["per_class"], epochs=cfg["epochs"], learning_rate=cfg["learning_rate"],
        loss_mode=cfg["loss_mode"], tier_weight=cfg["tier_weight"],
        max_examples=cfg["max_examples"], num_tokens=4, seed=seed,
    )


def load_state() -> dict:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"analysis_plan": "docs/analysis_plan.md", "provenance": provenance(),
            "train": {}, "determinism": {}, "eval_e80": {}, "e60": None}


def save(state: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json_atomic(RESULTS_PATH, state)


def max_abs_diff(a: Path, b: Path) -> float:
    sa = torch.load(a, map_location="cpu", weights_only=False)["adapter_state_dict"]
    sb = torch.load(b, map_location="cpu", weights_only=False)["adapter_state_dict"]
    return max(float((sa[k].float() - sb[k].float()).abs().max()) for k in sa)


def stage_train(state: dict) -> None:
    for seed in SEEDS:
        path = checkpoint_path(seed)
        if str(seed) in state["train"] and path.exists():
            continue
        print(f"\n[train] seed={seed} -> {path}")
        t0 = time.time()
        summary = train_adapter(headline_config(seed), output_path=path)
        # Guard on the example cache: same 64 DS1 examples as the headline checkpoint.
        trained = torch.load(path, map_location="cpu", weights_only=False)["source_indices"]
        headline = torch.load(HEADLINE_CHECKPOINT, map_location="cpu", weights_only=False)["source_indices"]
        if trained != headline:
            raise RuntimeError(f"seed={seed}: training examples differ from the headline checkpoint's")
        state["train"][str(seed)] = {
            "checkpoint": str(path), "sha256": sha256(path),
            "losses": summary["losses"], "train_seconds": round(time.time() - t0, 1),
        }
        save(state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def stage_determinism(state: dict) -> None:
    for seed, e2_path in E2_CHECKPOINTS.items():
        diff = max_abs_diff(checkpoint_path(seed), e2_path)
        state["determinism"][str(seed)] = {
            "compared_to": str(e2_path), "max_abs_weight_diff": diff,
            "within_tolerance": diff <= DETERMINISM_TOL, "tolerance": DETERMINISM_TOL,
        }
        print(f"[determinism] seed={seed} max|dw|={diff:.3e} "
              f"{'OK' if diff <= DETERMINISM_TOL else 'EXCEEDS TOLERANCE'}")
    save(state)


def stage_eval_e80(state: dict, model, processor) -> None:
    prepared = None
    for seed in SEEDS:
        if str(seed) in state["eval_e80"]:
            continue
        if prepared is None:
            prepared = prepare_events(note=" (E80)")
        adapter = load_trained_adapter(str(checkpoint_path(seed)), model)
        partial = state.setdefault("eval_e80_partial", {})

        def checkpoint(rows, seed=seed):
            partial[str(seed)] = rows
            save(state)
        results = run_arm_b_eval(prepared, model, processor, adapter, label=f"p1 seed={seed}",
                                 resume_from=partial.get(str(seed)), on_event=checkpoint)
        summary = summarize_arm_b(results)
        if summary.get("tokens_mean") not in (None, 486.0):
            raise RuntimeError(f"seed={seed}: token mean {summary['tokens_mean']} != 486 "
                               "(gate 6b: construction fixes this; pipeline bug, stopping)")
        state["eval_e80"][str(seed)] = {"summary": summary, "per_event": results}
        partial.pop(str(seed), None)
        save(state)
        del adapter
        torch.cuda.empty_cache()

    accs = [state["eval_e80"][str(s)]["summary"]["accuracy"] for s in SEEDS]
    mean, sd = float(np.mean(accs)), float(np.std(accs, ddof=1))
    from scipy.stats import t
    half = t.ppf(0.975, len(accs) - 1) * sd / np.sqrt(len(accs))
    state["eval_e80_aggregate"] = {"accuracy_by_seed": dict(zip(map(str, SEEDS), accs)),
                                   "accuracy_mean": mean, "accuracy_sd": sd,
                                   "accuracy_t95": [mean - half, mean + half]}
    save(state)


def stage_e60(state: dict) -> None:
    """Offline, no GPU (Deviation 1, docs/analysis_plan.md): E60 is a strict subset
    of E80, so a GPU rerun adds no independent evidence. Arm A comes from the
    dissertation's reported Day 6 pass, Arm B from this run's seed-101 E80 events.
    Only accuracy and prompt tokens are reported -- the arms come from different
    sessions, so latency is not a paired, interleaved comparison here."""
    if state.get("e60"):
        return
    data = np.load(DS2_PATH)
    e60 = {int(i) for i in select_events(data["labels"], data["record_ids"], per_class=15)}
    e80 = {int(i) for i in select_events(data["labels"], data["record_ids"])}
    if not e60 <= e80:
        raise RuntimeError("E60 is not a subset of E80; Deviation 1's premise fails")
    with open(DAY6_REPORTED_PASS, encoding="utf-8") as f:
        arm_a = [r for r in json.load(f) if r["arm"] == "A"]
    arm_b = state["eval_e80"][str(REFERENCE_SEED)]["per_event"]

    def stats(rows, idx):
        r = [x for x in rows if int(x["idx"]) in idx]
        ok = [x for x in r if not x.get("generation_failed", False)]
        return {"n": len(r), "n_failed": len(r) - len(ok),
                "accuracy": float(np.mean([x["correct"] for x in r])),
                "prompt_tokens_mean": float(np.mean([x["prompt_tokens"] for x in ok]))}

    out = {"method": "offline subset of E80 (Deviation 1)", "reference_seed": REFERENCE_SEED,
           "arm_a_source": str(DAY6_REPORTED_PASS), "event_indices": sorted(e60)}
    for name, idx in (("e60", e60), ("e80", e80)):
        a, b = stats(arm_a, idx), stats(arm_b, idx)
        out[name] = {"armA": a, "armB": b, "token_reduction_pct":
                     100 * (1 - b["prompt_tokens_mean"] / a["prompt_tokens_mean"])}
    state["e60"] = out
    save(state)


def main():
    state = load_state()
    stage_train(state)
    stage_determinism(state)
    model, processor = load_model()
    stage_eval_e80(state, model, processor)
    stage_e60(state)
    print(json.dumps({k: state[k] for k in ("determinism", "eval_e80_aggregate")}, indent=2))
    print(json.dumps({k: v for k, v in state["e60"].items() if k != "event_indices"}, indent=2))


if __name__ == "__main__":
    main()
