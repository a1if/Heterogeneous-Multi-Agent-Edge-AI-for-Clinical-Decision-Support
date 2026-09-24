"""Item 7 (Deviation 9): train the multi-event adapter (MEA) on RR-encoder windows.

Data: DS1 windows stratified by window tier over N in {1, 5, 10, 20}, the class of
the most urgent beat round-robined within each cell. Train = 25 per (N, tier) cell
(300 windows) from all DS1 records except the held-out validation records
{109, 205, 223}. Validation = 5 per cell (60 windows) from those records only.
Window selection is fixed (seed 0 / 1); the training seed changes only the
adapter's initialisation and the example order.

Recipe: teacher forcing on canonical_window_target (tier tokens x4), AdamW lr 1e-3,
gradient accumulation 8 (batch-1 memory), clip 1.0. Every EVAL_EVERY updates,
greedy-generate on the validation windows and keep the checkpoint with the best
pooled balanced accuracy (not loss); stop after PATIENCE evaluations without
improvement or MAX_EPOCHS epochs. Resumable: a resume checkpoint every
RESUME_EVERY updates; rerunning continues from it.

Gate A (reported, applied by the user / next stage): the best checkpoint's
validation balanced accuracy at N = 10 >= 0.60 and parse rate >= 95%.

Run (from repo root):
    python p1_item7_train.py --seed 101
    python p1_item7_train.py --smoke        # 16 windows, 2 updates: memory and norm check
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from p1_io import save_json_atomic

NS = (1, 5, 10, 20)
TRAIN_PER_CELL, VAL_PER_CELL = 25, 5
VAL_RECORDS = {109, 205, 223}  # = train_perception_agent.VAL_RECORDS (held out from encoder training too)
LR, ACCUM, CLIP, TIER_WEIGHT = 1e-3, 8, 1.0, 4.0
EVAL_EVERY, PATIENCE, MAX_EPOCHS, RESUME_EVERY = 20, 3, 3, 8
VAL_MAX_NEW_TOKENS = 64
TIERS = ("routine", "priority", "urgent")


def build_windows(r):
    """Deterministic train / validation windows from the DS1 replay dict r."""
    from p1_pilot2_stratified import stratified_windows
    all_records = {int(x) for x in np.unique(r["record_ids"])}
    def cells_to_list(cells):
        return [{"n": n, "tier": t, "start": s} for (n, t), ss in sorted(cells.items()) for s in ss]
    train = stratified_windows(r["tiers"], r["record_ids"], np.random.default_rng(0), ns=NS,
                               per_cell=TRAIN_PER_CELL, classes=r["classes"], records=all_records - VAL_RECORDS)
    val = stratified_windows(r["tiers"], r["record_ids"], np.random.default_rng(1), ns=NS,
                             per_cell=VAL_PER_CELL, classes=r["classes"], records=VAL_RECORDS)
    return cells_to_list(train), cells_to_list(val)


def balanced_metrics(rows):
    """rows: dicts with n, tier (reference), answer, parsed -> pooled and per-N metrics."""
    def bal(rs):
        rec = [np.mean([x["answer"] == t for x in rs if x["tier"] == t]) for t in TIERS if any(x["tier"] == t for x in rs)]
        return float(np.mean(rec)) if rec else None
    return {"balanced_accuracy": bal(rows), "parse_rate": float(np.mean([x["parsed"] for x in rows])),
            "by_n": {str(n): bal([x for x in rows if x["n"] == n]) for n in sorted({x["n"] for x in rows})}}


def train_loop(*, adapter, optimizer, train, val, loss_fn, validate_fn, seed, paths, log=print,
               accum=ACCUM, eval_every=EVAL_EVERY, patience=PATIENCE, max_epochs=MAX_EPOCHS,
               resume_every=RESUME_EVERY, max_updates=None):
    """Model-agnostic training loop. loss_fn(window) -> scalar loss tensor for one
    window; validate_fn(val) -> metrics dict with 'balanced_accuracy'. paths: dict
    with 'resume', 'best', 'state' (JSON). Returns the final state dict."""
    from reasoning.adapter_training import _atomic_torch_save

    state = {"history": [], "best_metric": -1.0, "best_update": None, "evals_since_best": 0,
             "epoch": 0, "pos": 0, "updates": 0, "done": False}
    if paths["resume"].exists():
        ck = torch.load(paths["resume"], map_location="cpu", weights_only=False)
        adapter.load_state_dict(ck["adapter_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        torch.set_rng_state(ck["torch_rng_state"])
        state = ck["state"]
        log(f"resuming at epoch {state['epoch'] + 1}, position {state['pos']}, update {state['updates']}")

    def save_resume():
        _atomic_torch_save({"adapter_state_dict": adapter.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                            "torch_rng_state": torch.get_rng_state(), "state": state}, paths["resume"])

    adapter.train()
    while not state["done"] and state["epoch"] < max_epochs:
        order = np.random.default_rng(seed * 1000 + state["epoch"]).permutation(len(train))
        while state["pos"] + accum <= len(order):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for k in order[state["pos"]:state["pos"] + accum]:
                loss = loss_fn(train[int(k)]) / accum
                loss.backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), CLIP)
            optimizer.step()
            state["pos"] += accum
            state["updates"] += 1
            state["history"].append({"update": state["updates"], "loss": total})
            if state["updates"] % eval_every == 0:
                m = validate_fn(val)
                adapter.train()
                state["history"][-1]["val"] = m
                log(f"update {state['updates']}: loss {total:.4f} val bal-acc {m['balanced_accuracy']:.3f} "
                    f"parse {m['parse_rate']:.2f} by N {m['by_n']}")
                if m["balanced_accuracy"] > state["best_metric"]:
                    state.update(best_metric=m["balanced_accuracy"], best_update=state["updates"], best_val=m,
                                 evals_since_best=0)
                    _atomic_torch_save({"adapter_state_dict": adapter.state_dict(), "val": m,
                                        "update": state["updates"], "seed": seed}, paths["best"])
                else:
                    state["evals_since_best"] += 1
                    if state["evals_since_best"] >= patience:
                        state["done"] = True
            if state["updates"] % resume_every == 0 or state["done"]:
                save_resume()
                save_json_atomic(paths["state"], state)
            if state["done"] or (max_updates and state["updates"] >= max_updates):
                state["done"] = True
                break
        if not state["done"]:
            state["epoch"] += 1
            state["pos"] = 0
            save_resume()
            save_json_atomic(paths["state"], state)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from p1_item7_common import RR_ENCODER, replay_split
    from p1_pilot_multi_event import scaffold_parts
    from p1_step1_seeded_headline import provenance, sha256
    from reasoning.adapter_training import (_append_target_for_teacher_forcing, _labels_for_target,
                                            _target_ids_and_weights, _weighted_teacher_forcing_loss)
    from reasoning.baseline_arm import _extract_last_json_object
    from reasoning.model_loader import load_model
    from reasoning.multi_event_adapter import MultiEventVirtualAdapter, compose_multi_event_inputs
    from reasoning.output_schema import ReasoningOutput
    from reasoning.training_targets import canonical_window_target
    from reasoning.virtual_adapter import (_render_prompt_with_placeholder, _tokenize_and_remove_placeholder,
                                           freeze_language_model)

    r = replay_split("ds1")
    train, val = build_windows(r)
    tag = "smoke" if args.smoke else f"seed{args.seed}"
    paths = {"resume": Path(f"reasoning/checkpoints/p1_item7_mea_{tag}.resume.pt"),
             "best": Path(f"reasoning/checkpoints/p1_item7_mea_{tag}.pt"),
             "state": Path(f"results/p1_item7_train_{tag}.json")}
    if args.smoke:
        train = [w for w in train if w["n"] == 20][:8] + [w for w in train if w["n"] != 20][:8]
        val = val[:3]

    model, processor = load_model()
    freeze_language_model(model)
    torch.manual_seed(args.seed)
    device = model.get_input_embeddings().weight.device
    adapter = MultiEventVirtualAdapter.for_model(model).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR, weight_decay=0.0)
    tok = processor.tokenizer

    scaffold = {}
    def window_inputs(w):
        n = w["n"]
        if n not in scaffold:
            pre, suf = scaffold_parts(n)
            rendered, s0, s1 = _render_prompt_with_placeholder(processor, pre, suf)
            scaffold[n] = _tokenize_and_remove_placeholder(processor, rendered, s0, s1)
        return compose_multi_event_inputs(model, adapter, r["vectors"][w["start"]:w["start"] + n], *scaffold[n])

    step = {"i": 0}
    def loss_fn(w):
        ai = window_inputs(w)
        text = canonical_window_target(r["events"][w["start"]:w["start"] + w["n"]])
        target_ids, weights = _target_ids_and_weights(processor, text, w["tier"], TIER_WEIGHT)
        embeds, mask, pli = _append_target_for_teacher_forcing(model, ai, target_ids)
        labels, lw = _labels_for_target(ai.sequence_length, target_ids.to(embeds.device), weights.to(embeds.device))
        out = model(inputs_embeds=embeds, attention_mask=mask, per_layer_inputs=pli, use_cache=False)
        step["i"] += 1
        if step["i"] % 10 == 0:
            torch.cuda.empty_cache()  # variable-length windows fragment the allocator (see train_adapter)
        return _weighted_teacher_forcing_loss(out.logits, labels, lw)

    def validate_fn(windows):
        adapter.eval()
        rows = []
        with torch.no_grad():
            for w in windows:
                ai = window_inputs(w)
                out = model.generate(inputs_embeds=ai.inputs_embeds, attention_mask=ai.attention_mask,
                                     per_layer_inputs=ai.per_layer_inputs, do_sample=False,
                                     max_new_tokens=VAL_MAX_NEW_TOKENS)
                text = tok.decode(out[0], skip_special_tokens=True)
                try:
                    answer, parsed = ReasoningOutput(**_extract_last_json_object(text)).urgency_tier, True
                except (ValueError, TypeError):
                    answer, parsed = None, False
                rows.append({"n": w["n"], "tier": w["tier"], "answer": answer, "parsed": parsed})
        return balanced_metrics(rows)

    meta = {"analysis_plan": "docs/analysis_plan.md (Deviation 9)", "provenance": provenance(),
            "encoder": {"path": RR_ENCODER, "sha256": sha256(Path(RR_ENCODER))}, "seed": args.seed,
            "n_train": len(train), "n_val": len(val), "init_scale": float(adapter.scale)}
    t0 = time.time()
    if args.smoke:
        torch.cuda.reset_peak_memory_stats()
        n20 = [w for w in train if w["n"] == 20][0]
        loss_fn(n20).backward()
        peak_n20 = torch.cuda.max_memory_allocated() / 2 ** 20
        adapter.zero_grad(set_to_none=True)
        state = train_loop(adapter=adapter, optimizer=optimizer, train=train, val=val, loss_fn=loss_fn,
                           validate_fn=validate_fn, seed=args.seed, paths=paths, eval_every=2, max_updates=2)
        with torch.no_grad():
            norms = window_inputs(n20).inputs_embeds.float().norm(dim=-1)[0]
        smoke = {**meta, "peak_mem_mb_n20_train_step": peak_n20, "memory_ok": peak_n20 < 11500,
                 "scale_after": float(adapter.scale), "seconds": time.time() - t0, "state": state}
        save_json_atomic(Path("results/p1_item7_smoke.json"), smoke)
        print(json.dumps({k: v for k, v in smoke.items() if k != "state"}, indent=2))
        print("virtual-token norm range at N=20 (text tokens included):", float(norms.min()), float(norms.max()))
        return

    state = train_loop(adapter=adapter, optimizer=optimizer, train=train, val=val, loss_fn=loss_fn,
                       validate_fn=validate_fn, seed=args.seed, paths=paths)
    best = state.get("best_val") or {}
    gate = {"n10_balanced_accuracy": (best.get("by_n") or {}).get("10"), "parse_rate": best.get("parse_rate")}
    gate["passes"] = bool(gate["n10_balanced_accuracy"] is not None and gate["n10_balanced_accuracy"] >= 0.60
                          and gate["parse_rate"] is not None and gate["parse_rate"] >= 0.95)
    save_json_atomic(paths["state"], {**meta, **state, "gate_a": gate, "seconds": time.time() - t0})
    print(json.dumps({"best_update": state["best_update"], "best_val": best, "gate_a": gate}, indent=2))


if __name__ == "__main__":
    main()
