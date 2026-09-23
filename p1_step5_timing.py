"""Phase 1 step 5: timing and energy with output length removed as a variable
(docs/analysis_plan.md §5 S2; Deviation 4).

Every arm generates exactly N new tokens (min_new_tokens = max_new_tokens = N,
greedy), so the output-length confound in the dissertation's latency and energy
figures cannot arise: any difference left is the input side, i.e. the interface.
  N = 1   prefill only (time to first token): the interface's own cost
  N = 64  a fixed decode between the arms' natural lengths (B ~50, A ~85)
  decode cost per token = (t64 - t1) / 63
Energy: NVML power sampled at 100 Hz during each call (measure_comm_cost.sample_power_during),
reported gross and net of an idle baseline measured before and after the run.
Content is not scored here (step 4 covers accuracy).

Arms: A-full, A-compact, A-label (text family, step 4 prompts) and B-4 (seed 101).
B-null / B-shuffle have B-4's input cost by construction and are not repeated.
Arm order rotates per event; the two lengths alternate. Resumable.

Run (from repo root):
    python p1_step5_timing.py
"""
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
from transformers import TextIteratorStreamer

from ablation_common import prepare_events
from measure_comm_cost import _HAS_NVML, idle_baseline, sample_power_during
from p1_step1_seeded_headline import provenance, sha256
from p1_step4_baseline_family import B4_CHECKPOINT, chat_inputs
from reasoning.adapter_arm import load_trained_adapter
from reasoning.model_loader import load_model
from reasoning.prompt_template_family import build_family_prompt
from reasoning.virtual_adapter import prepare_adapter_inputs

ARMS = ("A-full", "A-compact", "A-label", "B-4")
LENGTHS = (1, 64)
RESULTS_PATH = Path("results/p1_step5_timing.json")


def timed_generate(model, processor, n_tokens, **inputs):
    """Greedy generation of exactly n_tokens; returns prompt length, TTFT and total ms."""
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=False)
    kwargs = dict(inputs, do_sample=False, min_new_tokens=n_tokens, max_new_tokens=n_tokens, streamer=streamer)
    torch.cuda.synchronize()
    start = time.perf_counter()
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()
    first = None
    for _ in streamer:
        if first is None:
            first = time.perf_counter()
    thread.join()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return {"ttft_ms": ((first or end) - start) * 1000, "total_ms": (end - start) * 1000}


def main():
    if not _HAS_NVML:
        raise RuntimeError("pynvml unavailable; step 5 needs energy readings")
    state = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {
        "analysis_plan": "docs/analysis_plan.md (§5 S2; Deviation 4)", "provenance": provenance(),
        "b4_checkpoint": {"path": str(B4_CHECKPOINT), "sha256": sha256(B4_CHECKPOINT)},
        "lengths": list(LENGTHS), "rows": []}
    done = {(r["idx"], r["arm"], r["n_tokens"]) for r in state["rows"]}

    events = prepare_events(note=" (E80, step 5)")
    model, processor = load_model()
    adapter = load_trained_adapter(str(B4_CHECKPOINT), model)
    state.setdefault("idle_w", {})["start"] = idle_baseline()

    def inputs_for(arm, e):
        if arm == "B-4":
            ctx = torch.from_numpy(e["context_vector"]).unsqueeze(0)
            ai = prepare_adapter_inputs(model, processor, adapter, e["health_event"], ctx)
            return (dict(inputs_embeds=ai.inputs_embeds, attention_mask=ai.attention_mask,
                         per_layer_inputs=ai.per_layer_inputs),
                    ai.prefix_token_count + adapter.num_tokens + ai.suffix_token_count)
        payload = {"A-full": "full", "A-compact": "compact", "A-label": "label"}[arm]
        ci = chat_inputs(processor, model, build_family_prompt(e["health_event"], payload))
        return dict(input_ids=ci["input_ids"], attention_mask=ci["attention_mask"]), ci["input_ids"].shape[1]

    # One untimed warm-up per arm so first-call CUDA / allocator effects hit no measured event.
    for arm in ARMS:
        kw, _ = inputs_for(arm, events[0])
        timed_generate(model, processor, 4, **kw)

    t0 = time.time()
    for i, e in enumerate(events):
        order = ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]
        lengths = LENGTHS if i % 2 == 0 else LENGTHS[::-1]
        for arm in order:
            kw, prompt_tokens = inputs_for(arm, e)
            for n in lengths:
                if (e["idx"], arm, n) in done:
                    continue
                timing, joules, mean_w, elapsed_s, n_samples = sample_power_during(
                    timed_generate, model, processor, n, **kw)
                state["rows"].append({"idx": e["idx"], "arm": arm, "n_tokens": n, "prompt_tokens": prompt_tokens,
                                      **timing, "energy_j": joules, "mean_power_w": mean_w,
                                      "power_samples": n_samples, "position_in_order": order.index(arm)})
        RESULTS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"[{i+1}/{len(events)}] idx={e['idx']} elapsed={time.time() - t0:.0f}s", flush=True)

    state["idle_w"]["end"] = idle_baseline()
    state["summary"] = summarize(state)
    RESULTS_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state["summary"], indent=2))


def boot_ci(x, n_boot=20000, seed=0):
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def summarize(state):
    from scipy.stats import wilcoxon
    idle = np.mean([state["idle_w"]["start"], state["idle_w"]["end"]])
    # Short N=1 calls can catch < 2 NVML samples, in which case energy is None -> NaN (excluded).
    rows = [dict(r, energy_j=np.nan if r["energy_j"] is None else r["energy_j"]) for r in state["rows"]]
    get = {(r["idx"], r["arm"], r["n_tokens"]): r for r in rows}
    idxs = sorted({r["idx"] for r in rows})
    per_event = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for i in idxs:
            r1, r64 = get[(i, arm, 1)], get[(i, arm, 64)]
            per_event[arm][i] = {
                "prompt_tokens": r1["prompt_tokens"],
                "prefill_ms": r1["total_ms"],
                "decode_ms_per_token": (r64["total_ms"] - r1["total_ms"]) / 63,
                "total_ms_64": r64["total_ms"],
                "energy_j_1": r1["energy_j"], "energy_j_64": r64["energy_j"],
                "net_energy_j_64": r64["energy_j"] - idle * r64["total_ms"] / 1000,
                "energy_j_per_decode_token": (r64["energy_j"] - r1["energy_j"]) / 63,
            }
    out = {"idle_w_mean": float(idle), "arms": {}, "b4_vs": {}}
    metrics = ("prompt_tokens", "prefill_ms", "decode_ms_per_token", "total_ms_64",
               "energy_j_1", "energy_j_64", "net_energy_j_64", "energy_j_per_decode_token")
    for arm in ARMS:
        out["arms"][arm] = {m: float(np.nanmean([per_event[arm][i][m] for i in idxs])) for m in metrics}
        out["arms"][arm]["n_energy_missing_n1"] = int(np.isnan([per_event[arm][i]["energy_j_1"] for i in idxs]).sum())
        out["arms"][arm]["prefill_ms_sd"] = float(np.std([per_event[arm][i]["prefill_ms"] for i in idxs]))
    for other in ARMS[:3]:
        out["b4_vs"][other] = {}
        for m in ("prefill_ms", "decode_ms_per_token", "total_ms_64", "energy_j_64", "net_energy_j_64"):
            a = np.array([per_event["B-4"][i][m] for i in idxs])
            b = np.array([per_event[other][i][m] for i in idxs])
            keep = ~(np.isnan(a) | np.isnan(b))
            a, b = a[keep], b[keep]
            rel = 100 * (a - b) / b
            out["b4_vs"][other][m] = {"mean_rel_diff_pct": float(rel.mean()), "ci95_pct": boot_ci(rel),
                                      "wilcoxon_p": float(wilcoxon(a, b).pvalue)}
    return out


if __name__ == "__main__":
    main()
