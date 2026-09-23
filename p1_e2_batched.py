"""E2 (docs/analysis_plan.md, Deviation 5): throughput under batched serving.

Every arm uses the identical static scaffold of E1 (system prompt, class-neutral
context, instructions, event-data heading) followed by its own payload, so the
only per-request difference is the interface: A-full / A-compact / A-label JSON
text vs B-4's k = 4 virtual tokens. No prefix caching here: each request carries
the whole prompt, as in a simple server.

For batch sizes 1, 4, 8, 16, 32: REPS batches of E80 events (a rotating slice),
forced 16-token greedy generation. Text batches are left-padded; B-4 batches need
no padding (every B-4 prompt has the same length). Measured per batch: wall time,
events per second, peak allocated memory, NVML energy per event. Out-of-memory
is recorded, not raised. Atomic checkpoint after every batch; resumable.

Run (from repo root):
    python p1_e2_batched.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from ablation_common import prepare_events
from measure_comm_cost import _HAS_NVML, sample_power_during
from p1_e1_cached_scaffold import STATIC, language_model
from p1_io import save_json_atomic
from p1_step1_seeded_headline import provenance, sha256
from p1_step4_baseline_family import B4_CHECKPOINT
from reasoning.adapter_arm import load_trained_adapter
from reasoning.model_loader import load_model
from reasoning.prompt_template import _extract_prompt_fields
from reasoning.prompt_template_family import _payload
from reasoning.virtual_adapter import _render_prompt_with_placeholder, _tokenize_and_remove_placeholder

ARMS = ("A-full", "A-compact", "A-label", "B-4")
BATCHES = (1, 4, 8, 16, 32)
REPS = 3
NEW_TOKENS = 16
RESULTS_PATH = Path("results/p1_e2_batched.json")


def main():
    if not _HAS_NVML:
        raise RuntimeError("pynvml unavailable")
    state = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {
        "analysis_plan": "docs/analysis_plan.md (Deviation 5, E2)", "provenance": provenance(),
        "b4_checkpoint": {"path": str(B4_CHECKPOINT), "sha256": sha256(B4_CHECKPOINT)}, "rows": []}
    done = {(r["batch"], r["rep"], r["arm"]) for r in state["rows"]}

    events = prepare_events(note=" (E80, E2)")
    model, processor = load_model()
    adapter = load_trained_adapter(str(B4_CHECKPOINT), model)
    device = model.get_input_embeddings().weight.device
    embed = model.get_input_embeddings()
    tok = processor.tokenizer
    tok.padding_side = "left"
    pad_id = model.config.text_config.pad_token_id
    rendered, s0, s1 = _render_prompt_with_placeholder(processor, STATIC, "")
    static_ids, tail_ids = (t.to(device) for t in _tokenize_and_remove_placeholder(processor, rendered, s0, s1))

    def batch_inputs(arm, evs):
        if arm == "B-4":
            ctx = torch.from_numpy(np.stack([e["context_vector"] for e in evs])).to(device)
            b = len(evs)
            virt = adapter(ctx).to(dtype=embed.weight.dtype)                      # (b, k, E)
            embeds = torch.cat((embed(static_ids).expand(b, -1, -1), virt, embed(tail_ids).expand(b, -1, -1)), 1)
            surrogate = torch.cat((static_ids.expand(b, -1),
                                   torch.full((b, adapter.num_tokens), pad_id, device=device),
                                   tail_ids.expand(b, -1)), 1)
            return dict(inputs_embeds=embeds, per_layer_inputs=language_model(model).get_per_layer_inputs(surrogate, None),
                        attention_mask=torch.ones(surrogate.shape, dtype=torch.long, device=device)), surrogate.shape[1]
        texts = [json.dumps(_extract_prompt_fields(e["health_event"]) if arm == "A-full" else
                            _payload(e["health_event"], "compact" if arm == "A-compact" else "label"), indent=2)
                 for e in evs]
        ids = [torch.cat((static_ids[0], tok(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device),
                          tail_ids[0])) for t in texts]
        width = max(len(x) for x in ids)
        input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(input_ids)
        for r, x in enumerate(ids):  # left padding
            input_ids[r, width - len(x):] = x
            mask[r, width - len(x):] = 1
        return dict(input_ids=input_ids, attention_mask=mask), float(np.mean([len(x) for x in ids]))

    def generate(kw):
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.generate(**kw, do_sample=False, min_new_tokens=NEW_TOKENS, max_new_tokens=NEW_TOKENS)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) * 1000

    with torch.no_grad():
        for arm in ARMS:  # untimed warm-up
            generate(batch_inputs(arm, events[:2])[0])
        j = 0
        for b in BATCHES:
            for rep in range(REPS):
                evs = [events[(rep * b + i) % len(events)] for i in range(b)]
                order = ARMS[j % len(ARMS):] + ARMS[:j % len(ARMS)]
                j += 1
                for arm in order:
                    if (b, rep, arm) in done:
                        continue
                    row = {"batch": b, "rep": rep, "arm": arm}
                    try:
                        kw, mean_prompt = batch_inputs(arm, evs)
                        torch.cuda.reset_peak_memory_stats()
                        ms, joules, _, _, n_samples = sample_power_during(generate, kw)
                        row.update({"oom": False, "prompt_tokens_mean": mean_prompt, "wall_ms": ms,
                                    "events_per_s": b / (ms / 1000), "energy_j_per_event": joules / b if joules else None,
                                    "peak_mem_mb": torch.cuda.max_memory_allocated() / 2 ** 20})
                    except torch.cuda.OutOfMemoryError:
                        row["oom"] = True
                        torch.cuda.empty_cache()
                    state["rows"].append(row)
                    save_json_atomic(RESULTS_PATH, state)
            print(f"[batch {b}] done", flush=True)

    state["summary"] = summarize(state)
    save_json_atomic(RESULTS_PATH, state)
    print(json.dumps(state["summary"], indent=2))


def summarize(state):
    out = {"by_batch": {}, "gate": {}}
    for b in BATCHES:
        res = {}
        for arm in ARMS:
            rs = [r for r in state["rows"] if r["batch"] == b and r["arm"] == arm]
            ok = [r for r in rs if not r["oom"]]
            res[arm] = {"n_oom": len(rs) - len(ok),
                        **({"events_per_s_median": float(np.median([r["events_per_s"] for r in ok])),
                            "prompt_tokens_mean": float(np.mean([r["prompt_tokens_mean"] for r in ok])),
                            "peak_mem_mb_median": float(np.median([r["peak_mem_mb"] for r in ok])),
                            "energy_j_per_event_median": float(np.median([r["energy_j_per_event"] for r in ok
                                                                          if r["energy_j_per_event"]]))} if ok else {})}
        out["by_batch"][str(b)] = res
        a, bb = res["A-compact"], res["B-4"]
        if "events_per_s_median" in a and "events_per_s_median" in bb:
            pa = {r["rep"]: r for r in state["rows"] if r["batch"] == b and r["arm"] == "A-compact" and not r["oom"]}
            pb = {r["rep"]: r for r in state["rows"] if r["batch"] == b and r["arm"] == "B-4" and not r["oom"]}
            rel = [100 * (pb[k]["events_per_s"] - pa[k]["events_per_s"]) / pa[k]["events_per_s"] for k in pa if k in pb]
            out["gate"][str(b)] = {"b4_vs_acompact_throughput_rel_pct_mean": float(np.mean(rel)),
                                   "per_rep": rel, "n_reps": len(rel)}
    return out


if __name__ == "__main__":
    main()
