"""Phase 1 step 4: the interface family on E80, and the P1 gate
(docs/analysis_plan.md §3, §6; Deviation 3 for the B-4 checkpoint).

Generating arms, all in one session, interleaved per event with the arm order
rotated each event so no arm always runs first:
  A-full     frozen baseline prompt (run_baseline_arm_timed, unchanged)
  A-compact  same scaffold, payload = label, confidence, run length, SQI
  A-label    same scaffold, payload = label only
  B-4        k=4 adapter, seed 101 (virtual_adapter_e2_seed101.pt)
  B-shuffle  B-4 fed another event's context vector (fixed derangement, seed 0)
  B-null     k=4 adapter trained on zero context vectors (a fixed learned prefix)
Offline arm: A-rule (no LLM), urgency_tier_from_event on the predicted event.

Interface-attributable tokens: text arms = prompt tokens minus the same prompt
with an empty payload (scaffold); adapter arms = k.

Rows are saved atomically after every arm (p1_io.save_json_atomic); rerunning resumes.

Run (from repo root):
    python p1_step4_baseline_family.py
"""
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from ablation_common import prepare_events
from p1_io import save_json_atomic
from p1_step1_seeded_headline import headline_config, provenance, sha256
from project_config import measure_vram
from reasoning.adapter_arm import load_trained_adapter, run_adapter_arm_timed
from reasoning.adapter_training import train_adapter
from reasoning.baseline_arm import _generate_and_parse, run_baseline_arm_timed
from reasoning.model_loader import load_model
from reasoning.prompt_template_family import build_family_prompt
from reasoning.training_targets import urgency_tier_from_event

B4_CHECKPOINT = Path("reasoning/checkpoints/virtual_adapter_e2_seed101.pt")
BNULL_CHECKPOINT = Path("reasoning/checkpoints/p1_bnull_k4_seed101.pt")
ARMS = ("A-full", "A-compact", "A-label", "B-4", "B-shuffle", "B-null")
K = 4
RESULTS_PATH = Path("results/p1_step4_baseline_family.json")


def chat_inputs(processor, model, prompt):
    return processor.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)


def run_text_arm_timed(health_event, payload, model, processor):
    """Same timer boundaries and return shape as run_baseline_arm_timed."""
    perception_complete_ts = time.time()
    inputs = chat_inputs(processor, model, build_family_prompt(health_event, payload))
    with torch.no_grad():
        result, timing = _generate_and_parse(model, processor, inputs["input_ids"], inputs["attention_mask"])
    return {
        "perception_complete_ts": perception_complete_ts,
        "first_reasoning_token_ts": timing["first_token_ts"],
        "result": result,
        "prompt_tokens": inputs["input_ids"].shape[1],
        "output_tokens": timing["output_tokens"],
        "generation_duration_ms": (timing["gen_end_ts"] - timing["gen_start_ts"]) * 1000,
        "time_to_first_token_ms": (timing["first_token_ts"] - timing["gen_start_ts"]) * 1000,
        "parse_attempts": timing["attempt"],
    }


def derangement(n, seed=0):
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p


def ensure_bnull():
    if BNULL_CHECKPOINT.exists():
        return
    cfg = dataclasses.replace(headline_config(101), zero_context=True)
    print(f"Training B-null ({cfg}) -> {BNULL_CHECKPOINT}")
    train_adapter(cfg, output_path=BNULL_CHECKPOINT)
    torch.cuda.empty_cache()


def main():
    ensure_bnull()
    state = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {
        "analysis_plan": "docs/analysis_plan.md (§3, §6; Deviation 3)", "provenance": provenance(),
        "checkpoints": {"B-4": {"path": str(B4_CHECKPOINT), "sha256": sha256(B4_CHECKPOINT)},
                        "B-null": {"path": str(BNULL_CHECKPOINT), "sha256": sha256(BNULL_CHECKPOINT)}},
        "rows": []}
    done = {(r["idx"], r["arm"]) for r in state["rows"]}

    events = prepare_events(note=" (E80, step 4)")
    perm = derangement(len(events))
    state["shuffle_source_idx"] = {str(e["idx"]): events[perm[i]]["idx"] for i, e in enumerate(events)}

    model, processor = load_model()
    b4 = load_trained_adapter(str(B4_CHECKPOINT), model)
    bnull = load_trained_adapter(str(BNULL_CHECKPOINT), model)
    zero = np.zeros_like(events[0]["context_vector"])

    runners = {
        "A-full": lambda e, i: run_baseline_arm_timed(e["health_event"], perception_agent=None),
        "A-compact": lambda e, i: run_text_arm_timed(e["health_event"], "compact", model, processor),
        "A-label": lambda e, i: run_text_arm_timed(e["health_event"], "label", model, processor),
        "B-4": lambda e, i: run_adapter_arm_timed(e["health_event"], e["context_vector"], model, processor, b4),
        "B-shuffle": lambda e, i: run_adapter_arm_timed(e["health_event"], events[perm[i]]["context_vector"],
                                                         model, processor, b4),
        "B-null": lambda e, i: run_adapter_arm_timed(e["health_event"], zero, model, processor, bnull),
    }

    t0 = time.time()
    for i, e in enumerate(events):
        scaffold = chat_inputs(processor, model, build_family_prompt(e["health_event"], "none"))["input_ids"].shape[1]
        order = ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]
        for arm in order:
            if (e["idx"], arm) in done:
                continue
            row = {"idx": e["idx"], "arm": arm, "true_class": e["true_class"],
                   "predicted_class": e["predicted_class"], "record_id": e["record_id"],
                   "reference_tier": e["reference_tier"], "position_in_order": order.index(arm)}
            try:
                out, vram = measure_vram(runners[arm], e, i)
                row.update({
                    "generation_failed": False, "urgency_tier": out["result"]["urgency_tier"],
                    "correct": out["result"]["urgency_tier"] == e["reference_tier"],
                    "prompt_tokens": out["prompt_tokens"],
                    "interface_tokens": K if arm.startswith("B") else out["prompt_tokens"] - scaffold,
                    "scaffold_tokens": scaffold if arm.startswith("A") else None,
                    "output_tokens": out["output_tokens"],
                    "time_to_first_token_ms": out["time_to_first_token_ms"],
                    "generation_duration_ms": out["generation_duration_ms"],
                    "peak_vram_mb": vram, "parse_attempts": out["parse_attempts"],
                    "justification": out["result"]["justification"],
                    "referenced_guideline_fact": out["result"]["referenced_guideline_fact"],
                })
            except RuntimeError as err:
                row.update({"generation_failed": True, "urgency_tier": None, "correct": False, "error": str(err)})
            state["rows"].append(row)
            save_json_atomic(RESULTS_PATH, state)  # checkpoint after every arm
        tiers = {r["arm"]: r["urgency_tier"] for r in state["rows"] if r["idx"] == e["idx"]}
        print(f"[{i+1}/{len(events)}] idx={e['idx']} ref={e['reference_tier']} {tiers} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)

    state["summary"] = summarize(state["rows"], events)
    save_json_atomic(RESULTS_PATH, state)
    print(json.dumps(state["summary"], indent=2))


def paired_diff_ci(a, b, n_boot=20000, seed=0):
    """95% paired-bootstrap CI for mean(a) - mean(b) over the same events."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def summarize(rows, events):
    by_arm = {arm: {r["idx"]: r for r in rows if r["arm"] == arm} for arm in ARMS}
    order = [e["idx"] for e in events]
    out = {"A-rule": {"accuracy": float(np.mean([urgency_tier_from_event(e["health_event"]) == e["reference_tier"]
                                                 for e in events])),
                      "note": "identical to the reference tier by construction; no LLM"}}
    for arm, rs in by_arm.items():
        ok = [r for r in rs.values() if not r["generation_failed"]]
        mean = lambda k: float(np.mean([r[k] for r in ok])) if ok else None
        out[arm] = {"n": len(rs), "n_failed": len(rs) - len(ok),
                    "accuracy": float(np.mean([r["correct"] for r in rs.values()])),
                    "interface_tokens_mean": mean("interface_tokens"), "prompt_tokens_mean": mean("prompt_tokens"),
                    "output_tokens_mean": mean("output_tokens"), "ttft_ms_mean": mean("time_to_first_token_ms"),
                    "generation_ms_mean": mean("generation_duration_ms")}
    acc = {arm: [by_arm[arm][i]["correct"] for i in order] for arm in ARMS}
    diff, ci = paired_diff_ci(acc["B-4"], acc["A-label"])
    tokens_ok = out["B-4"]["interface_tokens_mean"] < out["A-label"]["interface_tokens_mean"]
    out["p1_gate"] = {
        "b4_minus_alabel_accuracy": diff, "ci95": ci,
        "b4_fewer_interface_tokens_than_alabel": tokens_ok,
        "non_inferior_(lower_bound_>_-0.10)": ci[0] > -0.10,
        "framing": ("latent interface dominates on cost" if tokens_ok and ci[0] > -0.10
                    else "trade-off characterisation"),
    }
    out["paired_accuracy_vs_A-full"] = {arm: dict(zip(("diff", "ci95"), paired_diff_ci(acc[arm], acc["A-full"])))
                                        for arm in ARMS if arm != "A-full"}
    return out


if __name__ == "__main__":
    main()
