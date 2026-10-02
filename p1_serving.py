"""Deviation 23: serving realism for RQ1 (batching, prefix caching, throughput, memory, energy). GPU.

Arms: A-compact, A-filtered (Deviation 20), MEA = recipe r4 seed 101 (main sender). Windows: the
Deviation 14 timing windows (21 per N, first 7 per tier), N in {10, 20, 50}. Batch sizes B in {1, 4, 8}:
consecutive chunks of B windows of the same N (a short last chunk is dropped); 3 repeats; medians.

  prefill        one forward pass over the whole prompt (no cache) for a batch: the time to first token
                 without caching, and peak allocated memory. Text batches are right-padded; the pads are
                 masked, so the cost is that of serving padded batches (timing only, logits unused).
  prefill_cached the shared scaffold prefix (everything before the event payload: system prompt, neutral
                 context, multi-event note, "--- Event data ---") is computed once per (arm, N, B) and its
                 KV cache reused, as a serving stack with prefix caching does; only the remainder (event
                 payload or virtual tokens, instructions, chat-template tail) is timed.
  decision       constrained greedy generation up to the tier token (token 7) for the batch, uncached, text
                 batches left-padded: decisions per second and energy per decision (NVML power, trapezoid).

CUDA-synchronised wall clock; the cache copy for each cached call is made before its timer starts.

Run (from repo root):
    python p1_serving.py            # resumable; results/p1_serving.json
    python p1_serving.py --summary
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from p1_io import save_json_atomic

OUT = Path("results/p1_serving.json")
NS = (10, 20, 50)
BATCHES = (1, 4, 8)
REPEATS = 3
ARMS = ("A-compact", "A-filtered", "MEA")
MEA_CKPT = "reasoning/checkpoints/p1_item7_mea_r4_seed101.pt"
EVENT_MARK = "--- Event data ---\n"


def run():
    import torch
    from measure_comm_cost import sample_power_during
    from p1_item7_common import replay_split, window_vectors
    from p1_item7_eval import load_windows
    from p1_item7_filtered import compact_content, filtered_content
    from p1_item7_ttd import TIER_TOKEN_INDEX, select
    from p1_pilot_multi_event import scaffold_parts
    from reasoning.constrained_json import SchemaJsonProcessor, TokenTable
    from reasoning.model_loader import load_model
    from reasoning.multi_event_adapter import MultiEventVirtualAdapter, compose_multi_event_inputs
    from reasoning.virtual_adapter import _render_prompt_with_placeholder, _tokenize_and_remove_placeholder

    r = replay_split("ds2")
    ws = [w for w in select(load_windows("ds2", r) + load_windows("ds2_n50", r)) if w["n"] in NS]
    state = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"design": __doc__, "rows": []}
    done = {(x["arm"], x["n"], x["batch"], x["repeat"], x["chunk"]) for x in state["rows"]}
    model, processor = load_model()
    tok = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    table = TokenTable(tok, device=device)
    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    ck = torch.load(MEA_CKPT, map_location="cpu", weights_only=False)["adapter_state_dict"]
    adapter = MultiEventVirtualAdapter.for_model(model, max_events=ck["position"].shape[0],
                                                 input_dim=ck["projection.weight"].shape[1]).to(device)
    adapter.load_state_dict(ck)
    adapter.eval()
    pad_id = tok.pad_token_id

    def text_ids(content):
        rendered = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                 tokenize=False)
        enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        cut = rendered.index(EVENT_MARK) + len(EVENT_MARK)
        k = sum(1 for a, b in enc["offset_mapping"] if b <= cut)  # tokens entirely inside the shared prefix
        return enc["input_ids"], k

    def window_inputs(arm, w):
        """-> dict(kind, ids or embeds/pli, prefix_len) for one window."""
        n, s = w["n"], w["start"]
        if arm == "MEA":
            pre, suf = scaffold_parts(n)
            rendered, s0, s1 = _render_prompt_with_placeholder(processor, pre, suf)
            pre_ids, suf_ids = _tokenize_and_remove_placeholder(processor, rendered, s0, s1)
            ai = compose_multi_event_inputs(model, adapter, window_vectors(r, s, n, adapter.input_dim), pre_ids, suf_ids)
            return {"embeds": ai.inputs_embeds, "pli": ai.per_layer_inputs, "k": int(pre_ids.shape[1])}
        ids, k = text_ids((compact_content if arm == "A-compact" else filtered_content)(r["events"], s, n))
        return {"ids": ids, "k": k}

    def sync():
        torch.cuda.synchronize()

    def batch_tensors(items, side):
        """Stack a batch. Text: pad to the longest (side = 'left' or 'right'). MEA windows of one N are equal length."""
        if "embeds" in items[0]:
            e = torch.cat([x["embeds"] for x in items])
            p = torch.cat([x["pli"] for x in items])
            return {"inputs_embeds": e, "per_layer_inputs": p,
                    "attention_mask": torch.ones(e.shape[:2], dtype=torch.long, device=device)}
        L = max(len(x["ids"]) for x in items)
        rows, mask = [], []
        for x in items:
            padn = L - len(x["ids"])
            rows.append(([pad_id] * padn + x["ids"]) if side == "left" else (x["ids"] + [pad_id] * padn))
            mask.append(([0] * padn + [1] * len(x["ids"])) if side == "left" else ([1] * len(x["ids"]) + [0] * padn))
        return {"input_ids": torch.tensor(rows, device=device), "attention_mask": torch.tensor(mask, device=device)}

    def timed(fn):
        sync()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        sync()
        return out, (time.perf_counter() - t0) * 1000, torch.cuda.max_memory_allocated() / 2 ** 20

    def prefill(items):
        kw = batch_tensors(items, "right")
        with torch.no_grad():
            return timed(lambda: model(**kw, use_cache=False, logits_to_keep=1))[1:]

    def prefill_cached(items):
        k = items[0]["k"]
        assert all(x["k"] == k for x in items), "shared prefix length differs within a batch"
        full = batch_tensors(items, "right")
        if "input_ids" in full:
            pre = {"input_ids": full["input_ids"][:, :k], "attention_mask": full["attention_mask"][:, :k]}
            rest = {"input_ids": full["input_ids"][:, k:]}
        else:
            pre = {"inputs_embeds": full["inputs_embeds"][:, :k], "per_layer_inputs": full["per_layer_inputs"][:, :k],
                   "attention_mask": full["attention_mask"][:, :k]}
            rest = {"inputs_embeds": full["inputs_embeds"][:, k:], "per_layer_inputs": full["per_layer_inputs"][:, k:]}
        with torch.no_grad():
            cache = model(**pre, use_cache=True).past_key_values
            c = copy.deepcopy(cache)
            return timed(lambda: model(**rest, attention_mask=full["attention_mask"], past_key_values=c,
                                       use_cache=True, logits_to_keep=1))[1:]

    def decision(items):
        kw = batch_tensors(items, "left")
        proc = SchemaJsonProcessor(table, eos, 40)

        def go():
            with torch.no_grad():
                return model.generate(**kw, do_sample=False, max_new_tokens=TIER_TOKEN_INDEX, logits_processor=[proc],
                                      pad_token_id=pad_id)
        sync()
        (_, joules, watts, secs, n_samples) = sample_power_during(go)
        return secs * 1000, joules, n_samples

    for arm in ARMS:  # warm-up
        for w in ws[:2]:
            prefill([window_inputs(arm, w)])
    t_start = time.time()
    for rep in range(REPEATS):
        for n in NS:
            group = [w for w in ws if w["n"] == n]
            for b in BATCHES:
                chunks = [group[i:i + b] for i in range(0, len(group) - b + 1, b)]
                for ci, chunk in enumerate(chunks):
                    for arm in (ARMS[(rep + ci) % 3:] + ARMS[:(rep + ci) % 3]):  # rotate arm order
                        if (arm, n, b, rep, ci) in done:
                            continue
                        items = [window_inputs(arm, w) for w in chunk]
                        p_ms, p_mb = prefill(items)
                        c_ms, c_mb = prefill_cached(items)
                        d_ms, d_j, d_samples = decision(items)
                        lengths = [len(x["ids"]) if "ids" in x else int(x["embeds"].shape[1]) for x in items]
                        state["rows"].append({"arm": arm, "n": n, "batch": b, "repeat": rep, "chunk": ci,
                                              "prompt_tokens": lengths, "prefix_tokens": items[0]["k"],
                                              "prefill_ms": p_ms, "prefill_peak_mb": p_mb,
                                              "prefill_cached_ms": c_ms, "prefill_cached_peak_mb": c_mb,
                                              "decision_ms": d_ms, "decision_joules": d_j, "power_samples": d_samples,
                                              "decisions_per_s": b / (d_ms / 1000), "joules_per_decision": d_j / b})
                        save_json_atomic(OUT, state)
                        torch.cuda.empty_cache()
        print(f"repeat {rep} done, {time.time() - t_start:.0f}s", flush=True)
    state["summary"] = summarize(state["rows"])
    save_json_atomic(OUT, state)
    print(json.dumps(state["summary"], indent=1))


def summarize(rows):
    out = {}
    for n in NS:
        for b in BATCHES:
            cell = {}
            for arm in ARMS:
                xs = [x for x in rows if x["arm"] == arm and x["n"] == n and x["batch"] == b]
                if not xs:
                    continue
                med = lambda k: float(np.median([x[k] for x in xs]))
                cell[arm] = {k: med(k) for k in ("prefill_ms", "prefill_cached_ms", "decision_ms", "decisions_per_s",
                                                 "joules_per_decision", "prefill_peak_mb", "prefill_cached_peak_mb")}
                cell[arm]["prompt_tokens_median"] = float(np.median([t for x in xs for t in x["prompt_tokens"]]))
                cell[arm]["n_measurements"] = len(xs)
            if "MEA" in cell:
                for base in ("A-compact", "A-filtered"):
                    if base in cell:
                        cell[f"MEA_vs_{base}"] = {k: cell["MEA"][k] / cell[base][k] - 1
                                                  for k in ("prefill_ms", "prefill_cached_ms", "decision_ms",
                                                            "joules_per_decision", "prefill_peak_mb")}
            out[f"N{n}_B{b}"] = cell
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    if ap.parse_args().summary:
        print(json.dumps(summarize(json.loads(OUT.read_text(encoding="utf-8"))["rows"]), indent=1))
    else:
        run()
