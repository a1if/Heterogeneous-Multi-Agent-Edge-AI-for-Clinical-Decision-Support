"""Pilot (no training): can today's single-event adapter handle N events at once?

Task: N consecutive beats (N = 1, 5, 10, 20; E3's 20 windows). The model must report
the most urgent tier among the N events. Reference = max over the events' own
reference tiers (routine < priority < urgent). Output schema, parsing and the
256-token cap are the same as every other Arm A / B run; one attempt per prompt
(greedy retries are identical).

Arms, with an IDENTICAL scaffold (system prompt, class-neutral context, a one-line
multi-event note, output instructions), so only the payload differs:
  A-compact  JSON list of N compact payloads
  B-4        N x 4 virtual tokens from the single-event k=4 adapter (never trained
             on N > 1), via p1_e3_multi_event.MultiEventAdapter
Reported: parse rate, accuracy by N and arm, and the accuracy of always answering
"routine" (what a no-information model scores). Atomic checkpoint per generation.

Run (from repo root):
    python p1_pilot_multi_event.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from p1_e3_multi_event import MultiEventAdapter, windows
from p1_io import save_json_atomic
from p1_step1_seeded_headline import provenance, sha256
from p1_step4_baseline_family import B4_CHECKPOINT
from perception.perception_agent import PerceptionAgent, replay_selected
from project_config import DS2_PATH, PERCEPTION_CHECKPOINT
from reasoning.adapter_arm import load_trained_adapter
from reasoning.baseline_arm import _extract_last_json_object
from reasoning.fixed_context_neutral import NEUTRAL_CONTEXT
from reasoning.model_loader import load_model
from reasoning.output_schema import ReasoningOutput
from reasoning.prompt_template import OUTPUT_INSTRUCTIONS, SYSTEM_PROMPT
from reasoning.prompt_template_family import _payload
from reasoning.training_targets import urgency_tier_from_event
from reasoning.virtual_adapter import (_render_prompt_with_placeholder, _tokenize_and_remove_placeholder,
                                       compose_adapter_inputs)

NS = (1, 5, 10, 20)
ARMS = ("A-compact", "B-4")
RANK = {"routine": 0, "priority": 1, "urgent": 2}
RESULTS_PATH = Path("results/p1_pilot_multi_event.json")


def note(n):
    return ("The event data below contains {n} consecutive beats from one recording. Apply the rule to each "
            "beat and report the single most urgent tier among them.\n").format(n=n)


def make_generator(model, processor, adapter, replayed):
    """generate(arm, idx) -> raw model text for the events at indices idx (shared by both pilots)."""
    device = model.get_input_embeddings().weight.device
    tok = processor.tokenizer

    def parts(n):
        prefix = (f"{SYSTEM_PROMPT}\n\n--- Background context ---\n{NEUTRAL_CONTEXT}\n"
                  f"{note(n)}--- Event data ---\n")
        return prefix, f"\n\n--- Instructions ---\n{OUTPUT_INSTRUCTIONS}"

    def generate(arm, idx):
        prefix, suffix = parts(len(idx))
        if arm == "B-4":
            rendered, s0, s1 = _render_prompt_with_placeholder(processor, prefix, suffix)
            pre, suf = _tokenize_and_remove_placeholder(processor, rendered, s0, s1)
            ctx = torch.from_numpy(np.stack([replayed[i][1] for i in idx]).reshape(1, -1))
            ai = compose_adapter_inputs(model, MultiEventAdapter(adapter, len(idx)), ctx, pre, suf)
            kw = dict(inputs_embeds=ai.inputs_embeds, attention_mask=ai.attention_mask,
                      per_layer_inputs=ai.per_layer_inputs)
        else:
            payload = json.dumps([_payload(replayed[i][0], "compact") for i in idx], indent=2)
            ids = processor.apply_chat_template([{"role": "user", "content": prefix + payload + suffix}],
                                                add_generation_prompt=True, tokenize=True, return_dict=True,
                                                return_tensors="pt")
            kw = dict(input_ids=ids["input_ids"].to(device), attention_mask=ids["attention_mask"].to(device))
        start = kw["input_ids"].shape[1] if "input_ids" in kw else 0
        out = model.generate(**kw, do_sample=False, max_new_tokens=256)
        return tok.decode(out[0][start:], skip_special_tokens=True)
    return generate


def main():
    state = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else {
        "design": __doc__, "provenance": provenance(),
        "b4_checkpoint": {"path": str(B4_CHECKPOINT), "sha256": sha256(B4_CHECKPOINT)}, "rows": []}
    done = {(r["window"], r["n"], r["arm"]) for r in state["rows"]}

    with np.load(DS2_PATH) as z:
        data = {k: z[k] for k in ("features", "labels", "rr_interval_ms", "record_ids")}
    wins = windows(data, np.random.default_rng(0))
    agent = PerceptionAgent(checkpoint_path=PERCEPTION_CHECKPOINT)
    replayed = replay_selected(agent, data["features"], data["rr_interval_ms"], data["record_ids"],
                               sorted({i for w in wins for i in w[:max(NS)]}))
    del agent

    model, processor = load_model()
    adapter = load_trained_adapter(str(B4_CHECKPOINT), model)
    generate = make_generator(model, processor, adapter, replayed)

    t0 = time.time()
    with torch.no_grad():
        for w, idx_all in enumerate(wins):
            for n in NS:
                idx = idx_all[:n]
                ref = max((urgency_tier_from_event(replayed[i][0]) for i in idx), key=RANK.get)
                for arm in ARMS:
                    if (w, n, arm) in done:
                        continue
                    text = generate(arm, idx)
                    try:
                        tier, parsed = ReasoningOutput(**_extract_last_json_object(text)).urgency_tier, True
                    except (ValueError, TypeError):
                        tier, parsed = None, False
                    state["rows"].append({"window": w, "n": n, "arm": arm, "reference": ref, "tier": tier,
                                          "parsed": parsed, "correct": tier == ref, "text": text[:600]})
                    save_json_atomic(RESULTS_PATH, state)
            print(f"[window {w+1}/{len(wins)}] elapsed={time.time() - t0:.0f}s", flush=True)

    state["summary"] = summarize(state["rows"])
    save_json_atomic(RESULTS_PATH, state)
    print(json.dumps(state["summary"], indent=2))


def summarize(rows):
    out = {}
    for n in NS:
        rs = [r for r in rows if r["n"] == n]
        refs = [r["reference"] for r in rs if r["arm"] == ARMS[0]]
        out[str(n)] = {"always_routine_accuracy": float(np.mean([x == "routine" for x in refs])),
                       "reference_counts": {t: refs.count(t) for t in RANK}}
        for arm in ARMS:
            a = [r for r in rs if r["arm"] == arm]
            out[str(n)][arm] = {"accuracy": float(np.mean([r["correct"] for r in a])),
                                "parse_rate": float(np.mean([r["parsed"] for r in a])),
                                "predicted": {t: sum(r["tier"] == t for r in a) for t in RANK}}
    return out


if __name__ == "__main__":
    main()
