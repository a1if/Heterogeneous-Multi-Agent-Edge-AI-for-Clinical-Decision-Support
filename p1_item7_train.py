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

Recipes (--recipe): r1 = the above (Deviation 9; seeds 101, 202 on record). r2 =
Deviation 11 (default): batch 1 (no accumulation, ~900 steps), lr 5e-4 with linear
warm-up (5% of max steps) and cosine decay to 10%, 120 validation windows (10 per
cell, <= 5 per record per cell) evaluated every 75 steps with batched generation.
Diagnostics showed r1 under-trains (seed 202: train = val balanced accuracy ~0.38).

Run (from repo root):
    python p1_item7_train.py --seed 101                 # recipe r2
    python p1_item7_train.py --seed 101 --recipe r1     # Deviation 9 recipe
    python p1_item7_train.py --smoke                    # memory and norm check
"""
import math
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
VAL_MAX_NEW_TOKENS = 96  # longest window target is 66 tokens; 64 truncated priority/urgent answers (seed 101, updates 20-40)
TIERS = ("routine", "priority", "urgent")
RECIPES = {
    "r1": dict(lr=1e-3, accum=8, warmup_frac=0.0, min_lr_frac=1.0, val_per_cell=5, val_max_per_record=3,
               eval_every=20, patience=3, max_epochs=3, resume_every=8, val_batch=1),
    "r2": dict(lr=5e-4, accum=1, warmup_frac=0.05, min_lr_frac=0.1, val_per_cell=10, val_max_per_record=5,
               eval_every=75, patience=3, max_epochs=3, resume_every=25, val_batch=8,
               min_updates=600),  # Deviation 11 amendment: no early stop before 2 epochs
    # Deviation 13: r2 + N = 50 windows (375 training windows) and 50 position slots
    "r3": dict(lr=5e-4, accum=1, warmup_frac=0.05, min_lr_frac=0.1, val_per_cell=10, val_max_per_record=5,
               eval_every=93, patience=3, max_epochs=3, resume_every=25, val_batch=8, min_updates=750,
               ns=(1, 5, 10, 20, 50), max_events=50),
    # Deviation 18: r3 + side inputs (heart rate, RR, run length -> 35-d) + hard-negative routine windows
    "r4": dict(lr=5e-4, accum=1, warmup_frac=0.05, min_lr_frac=0.1, val_per_cell=10, val_max_per_record=5,
               eval_every=111, patience=3, max_epochs=3, resume_every=25, val_batch=8, min_updates=894,
               ns=(1, 5, 10, 20, 50), max_events=50, input_dim=35,
               hard_neg_train=15, hard_neg_val=5, hard_neg_conf=0.8),
}
# Deviation 24: compression sweep, recipe r4 with k virtual tokens per event (k = 4 is r4 itself)
for _k in (1, 2, 8):
    RECIPES[f"r4k{_k}"] = dict(RECIPES["r4"], tokens=_k)
# Deviation 26: ablation, r4 with hard negatives but without the side inputs (32-d input)
RECIPES["r4hn"] = dict(RECIPES["r4"], input_dim=32)
# Deviation 29: training-data scaling (schedule scaled with the training-set size as r4 relates to its own)
RECIPES["r4d25"] = dict(RECIPES["r4"], train_frac=0.25, scale_schedule=True)
RECIPES["r4d50"] = dict(RECIPES["r4"], train_frac=0.50, scale_schedule=True)
RECIPES["r4dmax"] = dict(RECIPES["r4"], train_per_cell=50, hard_neg_train=30, scale_schedule=True)


def hard_negative_windows(r, records, per_n, rng, ns, conf_thr=0.8, max_per_record=3, exclude=()):
    """Routine windows (every beat's reference tier routine) whose lowest beat confidence
    is below conf_thr: the windows behind the r3 false alarms (Deviation 18).
    Non-overlapping candidates inside one record, shuffled, at most max_per_record per
    record per N; windows already in `exclude` ((n, start) pairs) are skipped."""
    rid = np.asarray(r["record_ids"])
    conf = np.array([e["classification"]["confidence"] for e in r["events"]])
    routine = np.array([t == "routine" for t in r["tiers"]])
    starts = np.flatnonzero(np.r_[True, np.diff(rid) != 0])
    ends = np.r_[starts[1:], len(rid)]
    out = []
    for n in ns:
        cand = [s for s0, e0 in zip(starts, ends) if int(rid[s0]) in records
                for s in range(s0, e0 - n + 1, n)
                if routine[s:s + n].all() and conf[s:s + n].min() < conf_thr and (n, s) not in exclude]
        chosen, per_rec = [], {}
        for s in rng.permutation(cand) if cand else []:
            k = int(rid[s])
            if per_rec.get(k, 0) < max_per_record:
                chosen.append({"n": n, "tier": "routine", "start": int(s), "hard_negative": True})
                per_rec[k] = per_rec.get(k, 0) + 1
            if len(chosen) == per_n:
                break
        out += chosen
    return out


def build_windows(r, val_per_cell=VAL_PER_CELL, val_max_per_record=3, ns=NS, train_per_cell=TRAIN_PER_CELL):
    """Deterministic train / validation windows from the DS1 replay dict r. Cells are
    drawn in N order from one generator, so appending a larger N (recipe r3) leaves
    the smaller-N windows unchanged."""
    from p1_pilot2_stratified import stratified_windows
    all_records = {int(x) for x in np.unique(r["record_ids"])}
    def cells_to_list(cells):
        return [{"n": n, "tier": t, "start": s} for (n, t), ss in sorted(cells.items()) for s in ss]
    train = stratified_windows(r["tiers"], r["record_ids"], np.random.default_rng(0), ns=ns,
                               per_cell=train_per_cell, classes=r["classes"], records=all_records - VAL_RECORDS)
    val = stratified_windows(r["tiers"], r["record_ids"], np.random.default_rng(1), ns=ns,
                             per_cell=val_per_cell, max_per_record=val_max_per_record,
                             classes=r["classes"], records=VAL_RECORDS)
    return cells_to_list(train), cells_to_list(val)


def balanced_metrics(rows):
    """rows: dicts with n, tier (reference), answer, parsed -> pooled and per-N metrics."""
    def bal(rs):
        rec = [np.mean([x["answer"] == t for x in rs if x["tier"] == t]) for t in TIERS if any(x["tier"] == t for x in rs)]
        return float(np.mean(rec)) if rec else None
    return {"balanced_accuracy": bal(rows), "parse_rate": float(np.mean([x["parsed"] for x in rows])),
            "by_n": {str(n): bal([x for x in rows if x["n"] == n]) for n in sorted({x["n"] for x in rows})}}


def warmup_cosine(total_steps, warmup_frac, min_lr_frac):
    """LambdaLR factor: linear warm-up, then cosine decay to min_lr_frac of the peak.
    warmup_frac 0 and min_lr_frac 1 give a constant learning rate (recipe r1)."""
    warm = int(round(total_steps * warmup_frac))

    def f(step):
        if step < warm:
            return (step + 1) / warm
        t = min(1.0, (step - warm) / max(1, total_steps - warm))
        return min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t))
    return f


def train_loop(*, adapter, optimizer, train, val, loss_fn, validate_fn, seed, paths, log=print,
               accum=ACCUM, eval_every=EVAL_EVERY, patience=PATIENCE, max_epochs=MAX_EPOCHS,
               resume_every=RESUME_EVERY, max_updates=None, scheduler=None, min_updates=0):
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
        if scheduler is not None:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        state = ck["state"]
        log(f"resuming at epoch {state['epoch'] + 1}, position {state['pos']}, update {state['updates']}")

    def save_resume():
        _atomic_torch_save({"adapter_state_dict": adapter.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                            "torch_rng_state": torch.get_rng_state(), "state": state,
                            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None},
                           paths["resume"])

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
            if scheduler is not None:
                scheduler.step()
            state["pos"] += accum
            state["updates"] += 1
            state["history"].append({"update": state["updates"], "loss": total,
                                     "lr": optimizer.param_groups[0]["lr"]})
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
                    if state["evals_since_best"] >= patience and state["updates"] >= min_updates:
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
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="r2")
    args = ap.parse_args()

    from p1_item7_common import RR_ENCODER, replay_split, window_vectors
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

    rc = RECIPES[args.recipe]
    r = replay_split("ds1")
    train, val = build_windows(r, rc["val_per_cell"], rc["val_max_per_record"], rc.get("ns", NS),
                               rc.get("train_per_cell", TRAIN_PER_CELL))
    if rc.get("hard_neg_train"):
        allrec = {int(x) for x in np.unique(r["record_ids"])}
        seen = {(w["n"], w["start"]) for w in train + val}
        train += hard_negative_windows(r, allrec - VAL_RECORDS, rc["hard_neg_train"], np.random.default_rng(2),
                                       rc["ns"], rc["hard_neg_conf"], exclude=seen)
        val += hard_negative_windows(r, VAL_RECORDS, rc["hard_neg_val"], np.random.default_rng(3),
                                     rc["ns"], rc["hard_neg_conf"], exclude=seen)
    if rc.get("train_frac"):  # Deviation 29: nested subset, same fraction of every cell, in sampled order
        import math
        cells = {}
        for w in train:
            cells.setdefault((w["n"], w.get("tier"), bool(w.get("hard_negative"))), []).append(w)
        train = [w for ws in cells.values() for w in ws[:math.ceil(rc["train_frac"] * len(ws))]]
    if rc.get("scale_schedule"):  # Deviation 29: as r4 relates to its 447 windows (accum 1)
        epoch = len(train) // rc["accum"]
        rc = dict(rc, eval_every=max(1, epoch // 4), min_updates=2 * epoch)
    from p1_item7_common import ENCODER_TAG
    prefix = "" if args.recipe == "r1" else f"{args.recipe}{ENCODER_TAG}_"  # r1 keeps its original file names; tag = Deviation 22 sender
    tag = f"{prefix}smoke" if args.smoke else f"{prefix}seed{args.seed}"
    paths = {"resume": Path(f"reasoning/checkpoints/p1_item7_mea_{tag}.resume.pt"),
             "best": Path(f"reasoning/checkpoints/p1_item7_mea_{tag}.pt"),
             "state": Path(f"results/p1_item7_train_{tag}.json")}
    if args.smoke:
        big = max(w["n"] for w in train)
        train = [w for w in train if w["n"] == big][:8] + [w for w in train if w["n"] != big][:8]
        val = [w for w in val if w["n"] == big][:rc["val_batch"]] + [w for w in val if w["n"] == 1][:3]

    model, processor = load_model()
    freeze_language_model(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    torch.manual_seed(args.seed)
    device = model.get_input_embeddings().weight.device
    adapter = MultiEventVirtualAdapter.for_model(model, max_events=rc.get("max_events", 20), num_tokens=rc.get("tokens", 4),
                                                 input_dim=rc.get("input_dim", 32)).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=rc["lr"], weight_decay=0.0)
    total_steps = rc["max_epochs"] * (len(train) // rc["accum"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total_steps, rc["warmup_frac"], rc["min_lr_frac"]))
    tok = processor.tokenizer

    scaffold = {}
    def window_inputs(w):
        n = w["n"]
        if n not in scaffold:
            pre, suf = scaffold_parts(n)
            rendered, s0, s1 = _render_prompt_with_placeholder(processor, pre, suf)
            scaffold[n] = _tokenize_and_remove_placeholder(processor, rendered, s0, s1)
        return compose_multi_event_inputs(model, adapter, window_vectors(r, w["start"], n, adapter.input_dim), *scaffold[n])

    step = {"i": 0}
    def loss_fn(w):
        ai = window_inputs(w)
        text = canonical_window_target(r["events"][w["start"]:w["start"] + w["n"]])
        target_ids, weights = _target_ids_and_weights(processor, text, w["tier"], TIER_WEIGHT)
        embeds, mask, pli = _append_target_for_teacher_forcing(model, ai, target_ids)
        labels, lw = _labels_for_target(ai.sequence_length, target_ids.to(embeds.device), weights.to(embeds.device))
        # Memory (smoke test: 15.3 GB at N=20 on a 12 GB GPU). Neither change alters the loss:
        # logits only for the last T+1 positions (the prompt's are masked out anyway; with a
        # 262k vocabulary the full logits and their float copy cost GBs), and gradient
        # checkpointing, which recomputes the frozen LM's activations in the backward pass.
        keep = target_ids.shape[1] + 1
        model.train()  # checkpointing is active only in train mode; the LM has no dropout
        out = model(inputs_embeds=embeds, attention_mask=mask, per_layer_inputs=pli, use_cache=False,
                    logits_to_keep=keep)
        step["i"] += 1
        if step["i"] % 10 == 0:
            torch.cuda.empty_cache()  # variable-length windows fragment the allocator (see train_adapter)
        return _weighted_teacher_forcing_loss(out.logits, labels[:, -keep:], lw[:, -keep:])

    def validate_fn(windows, batch_size=None, return_rows=False):
        batch_size = batch_size or rc["val_batch"]
        adapter.eval()
        model.eval()
        rows = []
        # Batches of windows with the same N have identical prompt lengths: no padding.
        groups = [[w for w in windows if w["n"] == n] for n in sorted({w["n"] for w in windows})]
        batches = [g[i:i + batch_size] for g in groups for i in range(0, len(g), batch_size)]
        with torch.no_grad():
            for batch in batches:
                ais = [window_inputs(w) for w in batch]
                out = model.generate(inputs_embeds=torch.cat([a.inputs_embeds for a in ais]),
                                     attention_mask=torch.cat([a.attention_mask for a in ais]),
                                     per_layer_inputs=torch.cat([a.per_layer_inputs for a in ais]),
                                     do_sample=False, max_new_tokens=VAL_MAX_NEW_TOKENS)
                for w, o in zip(batch, out):
                    text = tok.decode(o, skip_special_tokens=True)
                    try:
                        answer, parsed = ReasoningOutput(**_extract_last_json_object(text)).urgency_tier, True
                    except (ValueError, TypeError):
                        answer, parsed = None, False
                    rows.append({"n": w["n"], "tier": w["tier"], "answer": answer, "parsed": parsed, "text": text})
        return rows if return_rows else balanced_metrics(rows)

    meta = {"analysis_plan": "docs/analysis_plan.md (Deviation 9" + (")" if args.recipe == "r1" else ", Deviation 11)"),
            "recipe": args.recipe, "recipe_params": rc, "provenance": provenance(),
            "encoder": {"path": RR_ENCODER, "sha256": sha256(Path(RR_ENCODER))}, "seed": args.seed,
            "n_train": len(train), "n_val": len(val), "init_scale": float(adapter.scale)}
    t0 = time.time()
    if args.smoke:
        torch.cuda.reset_peak_memory_stats()
        n20 = [w for w in train if w["n"] == big][0]  # the largest N (20 for r1/r2, 50 for r3)
        with torch.no_grad():  # logits_to_keep must not change the loss: compare with full logits
            ai = window_inputs(n20)
            tid, tw = _target_ids_and_weights(processor, canonical_window_target(
                r["events"][n20["start"]:n20["start"] + n20["n"]]), n20["tier"], TIER_WEIGHT)
            e, m, p = _append_target_for_teacher_forcing(model, ai, tid)
            lab, lw = _labels_for_target(ai.sequence_length, tid.to(e.device), tw.to(e.device))
            full = float(_weighted_teacher_forcing_loss(model(inputs_embeds=e, attention_mask=m, per_layer_inputs=p,
                                                              use_cache=False).logits, lab, lw))
            del ai, e, m, p
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        kept = loss_fn(n20)
        loss_check = {"full_logits": full, "logits_to_keep": float(kept.detach())}
        kept.backward()
        peak_n20 = torch.cuda.max_memory_allocated() / 2 ** 20
        adapter.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        state = train_loop(adapter=adapter, optimizer=optimizer, train=train, val=val, loss_fn=loss_fn,
                           validate_fn=validate_fn, seed=args.seed, paths=paths, accum=rc["accum"],
                           eval_every=2, max_updates=2, scheduler=scheduler)
        peak_loop = torch.cuda.max_memory_allocated() / 2 ** 20  # includes batched N=20 validation
        # Batched vs one-at-a-time greedy generation on the same windows (numerics may differ slightly)
        rb = validate_fn(val, return_rows=True)
        r1 = validate_fn(val, batch_size=1, return_rows=True)
        batch_check = {"n": len(rb), "same_answer": sum(a["answer"] == b["answer"] for a, b in zip(rb, r1)),
                       "same_text": sum(a["text"] == b["text"] for a, b in zip(rb, r1))}
        with torch.no_grad():
            norms = window_inputs(n20).inputs_embeds.float().norm(dim=-1)[0]
        smoke = {**meta, "largest_n": big, "peak_mem_mb_n20_train_step": peak_n20, "peak_mem_mb_loop_incl_validation": peak_loop,
                 "memory_ok": max(peak_n20, peak_loop) < 11500, "batch_check": batch_check,
                 "loss_check": loss_check, "scale_after": float(adapter.scale), "seconds": time.time() - t0, "state": state}
        save_json_atomic(Path("results/p1_item7_smoke.json" if args.recipe == "r1"
                              else f"results/p1_item7_smoke_{args.recipe}{ENCODER_TAG}.json"), smoke)
        print(json.dumps({k: v for k, v in smoke.items() if k != "state"}, indent=2))
        print("virtual-token norm range at N=20 (text tokens included):", float(norms.min()), float(norms.max()))
        return

    state = train_loop(adapter=adapter, optimizer=optimizer, train=train, val=val, loss_fn=loss_fn,
                       validate_fn=validate_fn, seed=args.seed, paths=paths, accum=rc["accum"],
                       eval_every=rc["eval_every"], patience=rc["patience"], max_epochs=rc["max_epochs"],
                       resume_every=rc["resume_every"], scheduler=scheduler, min_updates=rc.get("min_updates", 0))
    best = state.get("best_val") or {}
    gate = {"n10_balanced_accuracy": (best.get("by_n") or {}).get("10"), "parse_rate": best.get("parse_rate")}
    gate["passes"] = bool(gate["n10_balanced_accuracy"] is not None and gate["n10_balanced_accuracy"] >= 0.60
                          and gate["parse_rate"] is not None and gate["parse_rate"] >= 0.95)
    save_json_atomic(paths["state"], {**meta, **state, "gate_a": gate, "seconds": time.time() - t0})
    print(json.dumps({"best_update": state["best_update"], "best_val": best, "gate_a": gate}, indent=2))


if __name__ == "__main__":
    main()
