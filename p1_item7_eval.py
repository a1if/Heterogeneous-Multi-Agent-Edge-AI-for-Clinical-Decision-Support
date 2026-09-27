"""Item 7 evaluation with schema-constrained decoding (Deviation 12).

Arms (identical scaffold, task and decoding):
  A-compact          text JSON payload of the N events (p1_pilot_multi_event scaffold)
  MEA:<tag>          multi-event adapter checkpoint reasoning/checkpoints/p1_item7_mea_<tag>.pt
Window sets:
  --split val        the 120 recipe-r2 validation windows (DS1 held-out records): Gate A re-check
  --split ds2        the Deviation 10 DS2 windows saved by p1_item7_baseline.py (class-balanced
                     stratified + natural): the pre-registered test set
  --split ds2_n50    Deviation 13: DS2 N = 50 windows, class-balanced by the same rule and seed
                     (20 per tier) + 20 natural windows (first 50 beats of the E3 windows)
Greedy decoding constrained to the ReasoningOutput schema (reasoning/constrained_json.py),
128 new tokens, field cap 40. Per generation: the tier, whether a field hit the cap, the
raw text. MEA windows are batched 8 per same-N group (no padding); A-compact runs one at a
time (prompt lengths differ). Atomic save per generation; resumable.

Run (from repo root):
    python p1_item7_eval.py --split val --arms MEA:r2_seed101 MEA:r2_seed202
    python p1_item7_eval.py --split ds2 --arms A-compact MEA:r2_seed101 MEA:r2_seed202 MEA:r2_seed303
    python p1_item7_eval.py --split ds2 --summary
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from p1_io import save_json_atomic

TIERS = ("routine", "priority", "urgent")
MAX_NEW, FIELD_CAP, MEA_BATCH = 128, 40, 8
NS_TEST = (1, 5, 10, 20)
MARGIN = -0.05


def results_path(split, suffix=""):
    return Path(f"results/p1_item7_eval_{split}{'_' + suffix if suffix else ''}.json")


def load_windows(split, r):
    if split == "val":
        from p1_item7_train import RECIPES, build_windows
        rc = RECIPES["r2"]
        _, val = build_windows(r, rc["val_per_cell"], rc["val_max_per_record"])
        return [{"set": "val", "n": w["n"], "start": w["start"], "reference": w["tier"]} for w in val]
    if split == "ds2_n50":
        return n50_windows(r)
    base = json.loads(Path("results/p1_item7_baseline.json").read_text(encoding="utf-8"))
    return [{"set": w["set"], "n": w["n"], "start": w["start"], "reference": w["reference"],
             "top_class_predicted": w.get("top_class_predicted"), "top_class_true": w.get("top_class_true")}
            for w in base["windows"]]


def n50_windows(r):
    from p1_e3_multi_event import windows as natural_windows
    from p1_pilot2_stratified import RANK, stratified_windows
    labels = "NSVFQ"
    cells = stratified_windows(r["tiers"], r["record_ids"], np.random.default_rng(0), ns=(50,), per_cell=20,
                               classes=r["classes"])
    out = [{"set": "stratified", "n": 50, "start": s} for (n, t), ss in cells.items() for s in ss]
    out += [{"set": "natural", "n": 50, "start": w[0]}
            for w in natural_windows({"record_ids": r["record_ids"]}, np.random.default_rng(0))]
    for w in out:
        idx = range(w["start"], w["start"] + 50)
        w["reference"] = max((r["tiers"][i] for i in idx), key=RANK.get)
        top = max(idx, key=lambda i: RANK[r["tiers"][i]])
        w["top_class_predicted"], w["top_class_true"] = r["classes"][top], labels[int(r["labels"][top])]
    return out


def run(split, arms, suffix=""):
    from p1_item7_common import replay_split
    from p1_pilot_multi_event import scaffold_parts
    from reasoning.baseline_arm import _extract_last_json_object
    from reasoning.constrained_json import SchemaJsonProcessor, TokenTable
    from reasoning.model_loader import load_model
    from reasoning.multi_event_adapter import MultiEventVirtualAdapter, compose_multi_event_inputs
    from reasoning.output_schema import ReasoningOutput
    from reasoning.prompt_template_family import _payload
    from reasoning.virtual_adapter import _render_prompt_with_placeholder, _tokenize_and_remove_placeholder

    r = replay_split("ds1" if split == "val" else "ds2")
    windows = load_windows(split, r)
    path = results_path(split, suffix)
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "analysis_plan": "docs/analysis_plan.md (Deviations 9-13)", "design": __doc__, "rows": []}
    base = results_path(split)
    if suffix and not state["rows"] and base.exists():
        # Separate results file (e.g. recipe r3 at N = 5-20): reuse the A-compact generations for the
        # same windows (identical prompts, constrained greedy decoding), flagged as reused.
        state["rows"] = [{**x, "reused_from": str(base)} for x in json.loads(base.read_text(encoding="utf-8"))["rows"]
                         if x["arm"] == "A-compact"]
    done = {(x["arm"], x["set"], x["n"], x["start"]) for x in state["rows"]}

    model, processor = load_model()
    tok = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    table = TokenTable(tok, device=device)
    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
    scaffold = {}

    def parts(n):
        if n not in scaffold:
            pre, suf = scaffold_parts(n)
            rendered, s0, s1 = _render_prompt_with_placeholder(processor, pre, suf)
            scaffold[n] = _tokenize_and_remove_placeholder(processor, rendered, s0, s1)
        return scaffold[n]

    def record(arm, w, text, info, n_tokens):
        try:
            answer = ReasoningOutput(**_extract_last_json_object(text)).urgency_tier
        except (ValueError, TypeError):
            answer = None  # must not happen under constrained decoding; counted if it does
        state["rows"].append({**w, "arm": arm, "tier": answer, "parsed": answer is not None,
                              "correct": answer == w["reference"], "hit_field_cap": info["hit_field_cap"],
                              "n_tokens": n_tokens, "text": text})

    t0 = time.time()
    for arm in arms:
        todo = [w for w in windows if (arm, w["set"], w["n"], w["start"]) not in done]
        if arm == "A-compact":
            for k, w in enumerate(todo):
                pre, suf = scaffold_parts(w["n"])
                payload = json.dumps([_payload(r["events"][i], "compact") for i in range(w["start"], w["start"] + w["n"])],
                                     indent=2)
                ids = processor.apply_chat_template([{"role": "user", "content": pre + payload + suf}],
                                                    add_generation_prompt=True, tokenize=True, return_dict=True,
                                                    return_tensors="pt")
                proc = SchemaJsonProcessor(table, eos, FIELD_CAP)
                with torch.no_grad():
                    out = model.generate(input_ids=ids["input_ids"].to(device),
                                         attention_mask=ids["attention_mask"].to(device), do_sample=False,
                                         max_new_tokens=MAX_NEW, logits_processor=[proc])
                gen = out[0][ids["input_ids"].shape[1]:]
                record(arm, w, tok.decode(gen, skip_special_tokens=True), proc.summary()[0],
                       int((gen != tok.pad_token_id).sum()))
                save_json_atomic(path, state)
                if (k + 1) % 20 == 0:
                    print(f"[{arm} {k + 1}/{len(todo)}] elapsed {time.time() - t0:.0f}s", flush=True)
        else:
            tag = arm.split(":", 1)[1]
            ck = torch.load(f"reasoning/checkpoints/p1_item7_mea_{tag}.pt", map_location="cpu", weights_only=False)
            slots = ck["adapter_state_dict"]["position"].shape[0]  # 20 (r2) or 50 (r3)
            adapter = MultiEventVirtualAdapter.for_model(model, max_events=slots).to(device)
            adapter.load_state_dict(ck["adapter_state_dict"])
            adapter.eval()
            state.setdefault("checkpoints", {})[arm] = {"update": ck.get("update"), "val": ck.get("val")}
            groups = [[w for w in todo if w["n"] == n] for n in sorted({w["n"] for w in todo})]
            for batch in [g[i:i + MEA_BATCH] for g in groups for i in range(0, len(g), MEA_BATCH)]:
                proc = SchemaJsonProcessor(table, eos, FIELD_CAP)
                with torch.no_grad():
                    ais = [compose_multi_event_inputs(model, adapter, r["vectors"][w["start"]:w["start"] + w["n"]],
                                                      *parts(w["n"])) for w in batch]
                    out = model.generate(inputs_embeds=torch.cat([a.inputs_embeds for a in ais]),
                                         attention_mask=torch.cat([a.attention_mask for a in ais]),
                                         per_layer_inputs=torch.cat([a.per_layer_inputs for a in ais]),
                                         do_sample=False, max_new_tokens=MAX_NEW, logits_processor=[proc])
                for w, o, info in zip(batch, out, proc.summary()):
                    record(arm, w, tok.decode(o, skip_special_tokens=True), info, int((o != tok.pad_token_id).sum()))
                save_json_atomic(path, state)
            del adapter
            torch.cuda.empty_cache()
        print(f"[{arm}] done, elapsed {time.time() - t0:.0f}s", flush=True)
    state["summary"] = summarize(state["rows"], split)
    save_json_atomic(path, state)
    print(json.dumps(state["summary"], indent=1))


def _bal(rows):
    rec = [np.mean([x["correct"] for x in rows if x["reference"] == t]) for t in TIERS
           if any(x["reference"] == t for x in rows)]
    return float(np.mean(rec)) if rec else None


def summarize(rows, split, n_boot=10000, seed=0):
    strat = "val" if split == "val" else "stratified"
    arms = sorted({x["arm"] for x in rows})
    out = {"by_arm": {}}
    for arm in arms:
        a = [x for x in rows if x["arm"] == arm and x["set"] == strat]
        out["by_arm"][arm] = {
            "n_windows": len(a), "balanced_accuracy": _bal(a),
            "by_n": {str(n): {"balanced_accuracy": _bal([x for x in a if x["n"] == n]),
                              "recall": {t: float(np.mean([x["correct"] for x in a if x["n"] == n and x["reference"] == t]))
                                         for t in TIERS if any(x["n"] == n and x["reference"] == t for x in a)}}
                      for n in sorted({x["n"] for x in a})},
            "parse_rate": float(np.mean([x["parsed"] for x in a])) if a else None,
            "field_cap_rate": float(np.mean([x["hit_field_cap"] for x in a])) if a else None}
        nat = [x for x in rows if x["arm"] == arm and x["set"] == "natural"]
        if nat:
            out["by_arm"][arm]["natural"] = {
                str(n): {"accuracy": float(np.mean([x["correct"] for x in nat if x["n"] == n])),
                         "false_alarm_rate": float(np.mean([x["tier"] in ("priority", "urgent") for x in nat
                                                            if x["n"] == n and x["reference"] == "routine"]))}
                for n in sorted({x["n"] for x in nat})}
    mea = [a for a in arms if a.startswith("MEA:")]
    if split.startswith("ds2") and "A-compact" in arms and mea:
        out["primary"] = primary(rows, mea, n_boot, seed)
    return out


def primary(rows, mea, n_boot, seed):
    """MEA (mean balanced accuracy over seeds) minus A-compact, per N in {5, 10, 20};
    95% bootstrap CI resampling windows within each (N, tier) cell; non-inferior if the
    lower bound > MARGIN."""
    rng = np.random.default_rng(seed)
    idx = {}
    for x in rows:
        if x["set"] == "stratified":
            idx.setdefault((x["n"], x["reference"], x["start"]), {})[x["arm"]] = x["correct"]
    need = set(mea) | {"A-compact"}
    idx = {k: v for k, v in idx.items() if need <= set(v)}  # paired: windows every arm has answered
    out = {}
    for n in sorted({k[0] for k in idx} & {5, 10, 20, 50}):
        cells = {t: [v for (nn, tt, _), v in idx.items() if nn == n and tt == t] for t in TIERS}
        cells = {t: v for t, v in cells.items() if v}

        def diff(sample):
            ba = lambda arm: np.mean([np.mean([w[arm] for w in sample[t]]) for t in sample])
            return float(np.mean([ba(m) for m in mea]) - ba("A-compact"))

        point = diff(cells)
        boots = []
        for _ in range(n_boot):
            boots.append(diff({t: [v[i] for i in rng.integers(0, len(v), len(v))] for t, v in cells.items()}))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[str(n)] = {"difference": point, "ci95": [float(lo), float(hi)], "non_inferior": bool(lo > MARGIN),
                       "n_mea_seeds": len(mea), "n_windows": sum(len(v) for v in cells.values())}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("val", "ds2", "ds2_n50"), required=True)
    ap.add_argument("--arms", nargs="*", default=[])
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--suffix", default="", help="separate results file, e.g. r3 -> p1_item7_eval_ds2_r3.json")
    a = ap.parse_args()
    if a.summary:
        p = results_path(a.split, a.suffix)
        print(json.dumps(summarize(json.loads(p.read_text(encoding="utf-8"))["rows"], a.split), indent=1))
    else:
        run(a.split, a.arms, a.suffix)
