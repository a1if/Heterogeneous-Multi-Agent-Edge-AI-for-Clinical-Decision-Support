"""Interactive demo: send one heartbeat window to the frozen Gemma receiver through each channel and compare.

Picks a window of N consecutive beats from a test set, shows what the sender saw, what each channel sends
(compact text, filtered text, or virtual tokens from the adapter), and, unless --no-llm is given, the receiver's
answer, its tier and the time each channel took. Answers use the same schema-constrained greedy decoding as the
paper (default decoding, no calibration).

Examples (from the repository root):
    python demo.py                                  # a random urgent window of 10 beats from the MIT-BIH test set
    python demo.py --n 50 --tier priority           # a 50-beat window whose correct answer is "priority"
    python demo.py --n 20 --pick 7                  # a different random window
    python demo.py --record 208 --n 10              # a window from MIT-BIH record 208
    python demo.py --split incart --n 20            # the external INCART test set
    python demo.py --no-llm                         # CPU only: show the messages, skip Gemma

Needs: the replay cache is built on first use (about 4 minutes per split). Without --no-llm: a CUDA GPU with about
10 GB free, and access to google/gemma-4-E4B-it on Hugging Face (`huggingface-cli login`, licence accepted).
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

TESTSETS = {"ds2": "results/p1_item7_testset_v2.json", "incart": "results/p1_item7_testset_incart.json"}
LABELS = "NSVFQ"
NAMES = {"N": "normal", "S": "supraventricular", "V": "ventricular", "F": "fusion", "Q": "unclassifiable"}
MAX_NEW, FIELD_CAP = 128, 40


def pick_window(args):
    windows = json.loads(Path(TESTSETS[args.split]).read_text(encoding="utf-8"))["windows"]
    pool = [w for w in windows if w["n"] == args.n and w["set"] == "stratified"
            and (args.tier is None or w["reference"] == args.tier)
            and (args.record is None or int(w["record"]) == args.record)]
    if not pool:
        raise SystemExit(f"no {args.split} window with n={args.n}, tier={args.tier}, record={args.record}; "
                         f"n must be one of {sorted({w['n'] for w in windows})}")
    return random.Random(args.pick).choice(pool)


def show_window(r, w):
    print(f"\nWindow: {w['n']} beats from record {w['record']} (start index {w['start']})")
    print(f"Correct answer (the fixed rule on the sender's outputs): {w['reference'].upper()}\n")
    print("  #  sender says        confidence  abnormal run  tier      annotation")
    for k, i in enumerate(range(w["start"], w["start"] + w["n"])):
        e = r["events"][i]
        cls = r["classes"][i]
        conf = e.get("confidence") or e.get("classification", {}).get("confidence")
        run = e.get("consecutive_abnormal_beats", e.get("clinical_flags", {}).get("consecutive_abnormal_beats", ""))
        mark = "  <-" if r["tiers"][i] != "routine" else ""
        conf_s = f"{conf:.3f}" if isinstance(conf, float) else str(conf)
        print(f" {k:2d}  {NAMES.get(cls, cls):17s}  {conf_s:>10}  {str(run):>12}  {r['tiers'][i]:8s}  "
              f"{NAMES[LABELS[int(r['labels'][i])]]}{mark}")


def messages(r, w):
    from p1_item7_filtered import filtered_content
    from p1_pilot_multi_event import scaffold_parts
    from reasoning.prompt_template_family import _payload
    pre, suf = scaffold_parts(w["n"])
    compact = pre + json.dumps([_payload(r["events"][i], "compact") for i in range(w["start"], w["start"] + w["n"])],
                               indent=2) + suf
    return {"A-compact": compact, "A-filtered": filtered_content(r["events"], w["start"], w["n"]), "scaffold": (pre, suf)}


def event_data(text, lines=16):
    """Only the part of the prompt that differs between channels; the instructions around it are shared."""
    body = text.split("--- Event data ---", 1)[1].split("--- Instructions ---", 1)[0].strip().splitlines()
    shown = body if len(body) <= lines else body[:lines // 2] + [f"    ... ({len(body) - lines} more lines) ..."] + body[-lines // 2:]
    return "\n".join(shown) + f"\n({len(text):,} characters in the full prompt, shared instructions included)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=sorted(TESTSETS), default="ds2", help="MIT-BIH test set (ds2) or INCART")
    ap.add_argument("--n", type=int, default=10, help="beats per window: 1, 5, 10, 20 or 50")
    ap.add_argument("--tier", choices=("routine", "priority", "urgent"), default="urgent", help="correct answer to pick")
    ap.add_argument("--any-tier", action="store_true", help="ignore --tier")
    ap.add_argument("--record", type=int, help="only windows from this record")
    ap.add_argument("--pick", type=int, default=0, help="which random window to take")
    ap.add_argument("--seed", type=int, choices=(101, 202, 303), default=101, help="which final-adapter run")
    ap.add_argument("--no-llm", action="store_true", help="show the messages only; no GPU, no Gemma")
    args = ap.parse_args()
    if args.any_tier:
        args.tier = None

    from p1_item7_common import replay_split, window_vectors
    r = replay_split(args.split)
    w = pick_window(args)
    show_window(r, w)
    msg = messages(r, w)
    print("\nEvery channel shares the same instructions; only the event data differs.")
    print("\n--- Compact text (every event) ---\n" + event_data(msg["A-compact"]))
    print("\n--- Filtered text (abnormal events only) ---\n" + event_data(msg["A-filtered"]))
    print(f"\n--- Adapter (latent) ---\n{w['n']} events x 4 virtual tokens = {4 * w['n']} tokens in place of the event text"
          f" (checkpoint reasoning/checkpoints/p1_item7_mea_r4_seed{args.seed}.pt)")
    if args.no_llm:
        return

    from reasoning.baseline_arm import _extract_last_json_object
    from reasoning.constrained_json import SchemaJsonProcessor, TokenTable
    from reasoning.model_loader import load_model
    from reasoning.multi_event_adapter import MultiEventVirtualAdapter, compose_multi_event_inputs
    from reasoning.output_schema import ReasoningOutput
    from reasoning.virtual_adapter import _render_prompt_with_placeholder, _tokenize_and_remove_placeholder

    model, processor = load_model()
    tok = processor.tokenizer
    device = model.get_input_embeddings().weight.device
    table = TokenTable(tok, device=device)
    eos = model.generation_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]

    def generate(**inputs):
        proc = SchemaJsonProcessor(table, eos, FIELD_CAP)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW, logits_processor=[proc])
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    warm = processor.apply_chat_template([{"role": "user", "content": "Hello"}], add_generation_prompt=True,
                                         tokenize=True, return_dict=True, return_tensors="pt")
    with torch.no_grad():  # the first call on a GPU includes one-off set-up; keep it out of the timings
        model.generate(input_ids=warm["input_ids"].to(device), attention_mask=warm["attention_mask"].to(device),
                       do_sample=False, max_new_tokens=4)

    results = {}
    for arm in ("A-compact", "A-filtered"):
        ids = processor.apply_chat_template([{"role": "user", "content": msg[arm]}], add_generation_prompt=True,
                                            tokenize=True, return_dict=True, return_tensors="pt")
        out, secs = generate(input_ids=ids["input_ids"].to(device), attention_mask=ids["attention_mask"].to(device))
        gen = out[0][ids["input_ids"].shape[1]:]
        results[arm] = (tok.decode(gen, skip_special_tokens=True), ids["input_ids"].shape[1], secs,
                        int((gen != tok.pad_token_id).sum()))

    ck = torch.load(f"reasoning/checkpoints/p1_item7_mea_r4_seed{args.seed}.pt", map_location="cpu", weights_only=False)
    sd = ck["adapter_state_dict"]
    adapter = MultiEventVirtualAdapter.for_model(model, max_events=sd["position"].shape[0], num_tokens=sd["position"].shape[1],
                                                 input_dim=sd["projection.weight"].shape[1]).to(device)
    adapter.load_state_dict(sd)
    adapter.eval()
    pre, suf = msg["scaffold"]
    rendered, s0, s1 = _render_prompt_with_placeholder(processor, pre, suf)
    with torch.no_grad():
        ai = compose_multi_event_inputs(model, adapter, window_vectors(r, w["start"], w["n"], adapter.input_dim),
                                        *_tokenize_and_remove_placeholder(processor, rendered, s0, s1))
    out, secs = generate(inputs_embeds=ai.inputs_embeds, attention_mask=ai.attention_mask,
                         per_layer_inputs=ai.per_layer_inputs)
    results["Adapter"] = (tok.decode(out[0], skip_special_tokens=True), int(ai.attention_mask.shape[1]), secs,
                          int((out[0] != tok.pad_token_id).sum()))

    print(f"\n=== Receiver's answers (correct: {w['reference'].upper()}) ===")
    print("Time grows with answer length (about 0.16 s per generated token), so the paper compares channels by the\n"
          "time to the decision instead. Single windows vary: try several (--pick) and see the paper for averages.")
    for arm, (text, n_prompt, secs, n_gen) in results.items():
        try:
            ans = ReasoningOutput(**_extract_last_json_object(text))
            tier, why = ans.urgency_tier, ans.model_dump()
        except (ValueError, TypeError):
            tier, why = None, text
        ok = "correct" if tier == w["reference"] else "WRONG"
        print(f"\n[{arm}] tier: {tier} ({ok}) | prompt {n_prompt} tokens | answer {n_gen} tokens in {secs:.1f} s")
        print(json.dumps(why, indent=2)[:700] if isinstance(why, dict) else why[:700])


if __name__ == "__main__":
    main()
