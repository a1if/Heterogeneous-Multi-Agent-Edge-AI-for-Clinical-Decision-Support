"""E1 (docs/analysis_plan.md, Deviation 5): per-event cost once the static scaffold
is cached, which is how a deployed service would serve a fixed system prompt.

Both arms get the IDENTICAL static scaffold (system prompt, class-neutral context,
output instructions, event-data heading), rendered with the chat template and run
once into a KV cache. This removes the scaffold confound of steps 4-5 (Arm B's
scaffold was ~80 tokens longer). Each event then pays only for an incremental
forward pass over its own payload plus the closing template tokens:
  A-full     full JSON payload         A-compact  compact JSON payload
  A-label    label-only payload        B-4        k = 4 virtual tokens
Measured: the incremental forward (the step that yields the first output token),
logits for the last position only, median of REPS repeats per event, each on a
fresh copy of the cached scaffold (copied outside the timer). Cost only; content
is not scored. E80 events; arm order rotates; atomic checkpoint per event.

Run (from repo root):
    python p1_e1_cached_scaffold.py
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from ablation_common import prepare_events
from p1_io import save_json_atomic
from p1_step1_seeded_headline import provenance, sha256
from p1_step4_baseline_family import B4_CHECKPOINT
from reasoning.adapter_arm import load_trained_adapter
from reasoning.fixed_context_neutral import NEUTRAL_CONTEXT
from reasoning.model_loader import load_model
from reasoning.prompt_template import OUTPUT_INSTRUCTIONS, SYSTEM_PROMPT, _extract_prompt_fields
from reasoning.prompt_template_family import _payload
from reasoning.virtual_adapter import _render_prompt_with_placeholder, _tokenize_and_remove_placeholder

ARMS = ("A-full", "A-compact", "A-label", "B-4")
REPS = 5
RESULTS_PATH = Path("results/p1_e1_cached_scaffold.json")
STATIC = (f"{SYSTEM_PROMPT}\n\n--- Background context ---\n{NEUTRAL_CONTEXT}\n"
          f"--- Instructions ---\n{OUTPUT_INSTRUCTIONS}\n\n--- Event data ---\n")


def language_model(model):
    lm = getattr(model, "language_model", None)
    return lm if lm is not None else model.model.language_model


def main():
    state = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {
        "analysis_plan": "docs/analysis_plan.md (Deviation 5, E1)", "provenance": provenance(),
        "b4_checkpoint": {"path": str(B4_CHECKPOINT), "sha256": sha256(B4_CHECKPOINT)}, "rows": []}
    done = {(r["idx"], r["arm"]) for r in state["rows"]}

    events = prepare_events(note=" (E80, E1)")
    model, processor = load_model()
    adapter = load_trained_adapter(str(B4_CHECKPOINT), model)
    device = model.get_input_embeddings().weight.device
    embed = model.get_input_embeddings()
    tok = processor.tokenizer
    pad_id = model.config.text_config.pad_token_id

    rendered, s0, s1 = _render_prompt_with_placeholder(processor, STATIC, "")
    static_ids, tail_ids = (t.to(device) for t in _tokenize_and_remove_placeholder(processor, rendered, s0, s1))
    with torch.no_grad():
        torch.cuda.synchronize()
        t = time.perf_counter()
        cache = model(input_ids=static_ids, use_cache=True, logits_to_keep=1).past_key_values
        torch.cuda.synchronize()
        state["static_scaffold"] = {"tokens": int(static_ids.shape[1]), "tail_tokens": int(tail_ids.shape[1]),
                                    "prefill_ms_once": (time.perf_counter() - t) * 1000}

    def incremental(arm, e):
        if arm == "B-4":
            ctx = torch.from_numpy(e["context_vector"]).unsqueeze(0).to(device)
            virt = adapter(ctx).to(dtype=embed.weight.dtype)
            embeds = torch.cat((virt, embed(tail_ids)), dim=1)
            surrogate = torch.cat((torch.full((1, adapter.num_tokens), pad_id, device=device), tail_ids), dim=1)
            return {"inputs_embeds": embeds,
                    "per_layer_inputs": language_model(model).get_per_layer_inputs(surrogate, None)}, adapter.num_tokens
        ev = e["health_event"]
        text = json.dumps(_extract_prompt_fields(ev) if arm == "A-full" else
                          _payload(ev, "compact" if arm == "A-compact" else "label"), indent=2)
        payload_ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        return {"input_ids": torch.cat((payload_ids, tail_ids), dim=1)}, int(payload_ids.shape[1])

    def timed(kw):
        n_new = next(v for k, v in kw.items() if k in ("input_ids", "inputs_embeds")).shape[1]
        mask = torch.ones((1, static_ids.shape[1] + n_new), dtype=torch.long, device=device)
        c = copy.deepcopy(cache)
        torch.cuda.synchronize()
        t = time.perf_counter()
        model(**kw, attention_mask=mask, past_key_values=c, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000

    with torch.no_grad():
        for arm in ARMS:  # untimed warm-up
            timed(incremental(arm, events[0])[0])
        t0 = time.time()
        for i, e in enumerate(events):
            order = ARMS[i % len(ARMS):] + ARMS[:i % len(ARMS)]
            for arm in order:
                if (e["idx"], arm) in done:
                    continue
                kw, payload_tokens = incremental(arm, e)
                reps = [timed(kw) for _ in range(REPS)]
                state["rows"].append({"idx": e["idx"], "arm": arm, "payload_tokens": payload_tokens,
                                      "incremental_tokens": payload_tokens + int(tail_ids.shape[1]),
                                      "incremental_ms": float(np.median(reps)), "reps_ms": reps})
            save_json_atomic(RESULTS_PATH, state)
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(events)}] elapsed={time.time() - t0:.0f}s", flush=True)

    state["summary"] = summarize(state)
    save_json_atomic(RESULTS_PATH, state)
    print(json.dumps(state["summary"], indent=2))


def summarize(state):
    get = {(r["idx"], r["arm"]): r for r in state["rows"]}
    idxs = sorted({r["idx"] for r in state["rows"]})
    out = {"static_scaffold": state["static_scaffold"], "arms": {}}
    for arm in ARMS:
        ms = [get[(i, arm)]["incremental_ms"] for i in idxs]
        out["arms"][arm] = {"payload_tokens_mean": float(np.mean([get[(i, arm)]["payload_tokens"] for i in idxs])),
                            "incremental_ms_median": float(np.median(ms)),
                            "incremental_ms_iqr": [float(np.percentile(ms, 25)), float(np.percentile(ms, 75))]}
    rel = np.array([100 * (get[(i, "B-4")]["incremental_ms"] - get[(i, "A-compact")]["incremental_ms"])
                    / get[(i, "A-compact")]["incremental_ms"] for i in idxs])
    boot = np.random.default_rng(0).integers(0, len(rel), (20000, len(rel)))
    ci = np.percentile(rel[boot].mean(1), [2.5, 97.5])
    out["gate"] = {"b4_vs_acompact_incremental_rel_pct": float(rel.mean()), "ci95": [float(ci[0]), float(ci[1])],
                   "passes": bool(ci[1] < 0 and rel.mean() <= -10)}
    return out


if __name__ == "__main__":
    main()
