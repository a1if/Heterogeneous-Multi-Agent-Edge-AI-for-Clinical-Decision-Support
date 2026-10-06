"""Item 7 diagnostic: are the recipe-r2 parse failures truncation or malformed output?

Validation (Gate A) generates at most 96 new tokens; the DS2 evaluation protocol
allows 256 (the pilots' cap). For each r2 best checkpoint, generate on the same 120
validation windows with 256 tokens (greedy, batches of 8 same-N windows as in
training), then score each output twice: in full (256) and cut to its first 96
tokens (greedy decoding makes that prefix the 96-token output). Categories as in
p1_item7_diagnose. Diagnostic only: Gate A stays as pre-registered (96 tokens).

Run (from repo root):
    python p1_item7_parsecheck.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from p1_io import save_json_atomic
from p1_item7_diagnose import categorise
from p1_item7_train import RECIPES, TIERS, balanced_metrics, build_windows

RESULTS = Path("results/p1_item7_parsecheck.json")
CHECKPOINTS = {"r2_seed101": "reasoning/checkpoints/p1_item7_mea_r2_seed101.pt",
               "r2_seed202": "reasoning/checkpoints/p1_item7_mea_r2_seed202.pt"}
MAX_NEW, CUT = 256, 96


def main():
    from p1_item7_common import replay_split
    from p1_pilot_multi_event import scaffold_parts
    from reasoning.model_loader import load_model
    from reasoning.multi_event_adapter import MultiEventVirtualAdapter, compose_multi_event_inputs
    from reasoning.training_targets import most_urgent_index
    from reasoning.virtual_adapter import _render_prompt_with_placeholder, _tokenize_and_remove_placeholder

    rc = RECIPES["r2"]
    r = replay_split("ds1")
    _, val = build_windows(r, rc["val_per_cell"], rc["val_max_per_record"])
    state = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {"design": __doc__, "rows": []}
    done = {(x["ckpt"], x["n"], x["start"]) for x in state["rows"]}
    model, processor = load_model()
    tok = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    scaffold = {}

    def parts(n):
        if n not in scaffold:
            pre, suf = scaffold_parts(n)
            rendered, s0, s1 = _render_prompt_with_placeholder(processor, pre, suf)
            scaffold[n] = _tokenize_and_remove_placeholder(processor, rendered, s0, s1)
        return scaffold[n]

    t0 = time.time()
    for name, path in CHECKPOINTS.items():
        adapter = MultiEventVirtualAdapter.for_model(model).to(device)
        adapter.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["adapter_state_dict"])
        adapter.eval()
        todo = [w for w in val if (name, w["n"], w["start"]) not in done]
        groups = [[w for w in todo if w["n"] == n] for n in sorted({w["n"] for w in todo})]
        for batch in [g[i:i + rc["val_batch"]] for g in groups for i in range(0, len(g), rc["val_batch"])]:
            with torch.no_grad():
                ais = [compose_multi_event_inputs(model, adapter, r["vectors"][w["start"]:w["start"] + w["n"]],
                                                  *parts(w["n"])) for w in batch]
                out = model.generate(inputs_embeds=torch.cat([a.inputs_embeds for a in ais]),
                                     attention_mask=torch.cat([a.attention_mask for a in ais]),
                                     per_layer_inputs=torch.cat([a.per_layer_inputs for a in ais]),
                                     do_sample=False, max_new_tokens=MAX_NEW)
            for w, o in zip(batch, out):
                ids = [int(t) for t in o if int(t) != tok.pad_token_id]
                full, cut = tok.decode(ids, skip_special_tokens=True), tok.decode(ids[:CUT], skip_special_tokens=True)
                target_beat = most_urgent_index(r["events"][w["start"]:w["start"] + w["n"]])
                state["rows"].append({"ckpt": name, "n": w["n"], "start": w["start"], "tier": w["tier"],
                                      "n_tokens": len(ids), "text": full,
                                      "at256": categorise(full, w["tier"], w["n"], target_beat),
                                      "at96": categorise(cut, w["tier"], w["n"], target_beat)})
            save_json_atomic(RESULTS, state)
        print(f"[{name}] done, elapsed {time.time() - t0:.0f}s", flush=True)
        del adapter
        torch.cuda.empty_cache()
    state["summary"] = summarize(state["rows"])
    save_json_atomic(RESULTS, state)
    print(json.dumps(state["summary"], indent=1))


def summarize(rows):
    out = {}
    for ck in sorted({x["ckpt"] for x in rows}):
        a = [x for x in rows if x["ckpt"] == ck]
        for lim in ("at96", "at256"):
            m = balanced_metrics([{"n": x["n"], "tier": x["tier"], "answer": x[lim]["answer"],
                                   "parsed": x[lim]["answer"] is not None} for x in a])
            cats = {}
            for x in a:
                cats[x[lim]["category"]] = cats.get(x[lim]["category"], 0) + 1
            out[f"{ck}|{lim}"] = {**m, "categories": cats}
        out[f"{ck}|tokens"] = {"median": float(np.median([x["n_tokens"] for x in a])),
                               "over_96": int(sum(x["n_tokens"] > CUT for x in a)),
                               "hit_256": int(sum(x["n_tokens"] >= MAX_NEW for x in a))}
    return out


if __name__ == "__main__":
    main()
