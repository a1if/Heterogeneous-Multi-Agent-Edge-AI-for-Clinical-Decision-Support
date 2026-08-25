"""
Baseline arm (Arm A), rewritten for HuggingFace transformers — REPLACES the
Ollama-based version. Ollama could not support Arm B's inputs_embeds
injection, so both arms now share this exact model_loader.load_model()
call, satisfying the "identical checkpoint and quantisation, both arms"
control (design doc Section 3). Only the input construction differs:
Arm A tokenizes text to input_ids; Arm B (Day 3+) will build inputs_embeds.

Once test_baseline_is_frozen passes, nothing in this file or
prompt_template.py should change again without deliberately re-freezing.
"""
import json
import threading
import time

import torch
from transformers import TextIteratorStreamer

from reasoning.model_loader import load_model
from reasoning.prompt_template import build_baseline_prompt
from reasoning.output_schema import ReasoningOutput

# Greedy decoding, per the design doc's controlled-variable table (Section 3).
# do_sample=False makes temperature irrelevant to HF's generate(), but we keep
# it explicit here for audit clarity against the original Ollama config.
GENERATION_CONFIG = {"do_sample": False, "temperature": None, "max_new_tokens": 1024}

MAX_PARSE_RETRIES = 3


def _extract_last_json_object(text: str) -> dict:
    """Finds the last balanced {...} block in the text and parses it.
    Robust to a preceding thinking trace of unknown exact tag format —
    scans from the end for a balanced brace region rather than assuming
    a specific <think>...</think> style tag.
    """
    depth = 0
    end_idx = None
    start_idx = None
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if ch == "}":
            if depth == 0:
                end_idx = i
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0 and end_idx is not None:
                start_idx = i
                break
    if start_idx is None or end_idx is None:
        raise ValueError(f"No balanced JSON object found in output: {text!r}")
    return json.loads(text[start_idx:end_idx + 1])


def _generate_and_parse(model, processor, input_ids, attention_mask):
    """Runs generation with streaming (for first-token timing), retries
    parsing up to MAX_PARSE_RETRIES times on malformed JSON. Returns
    (parsed_result_dict, timing_dict)."""
    last_error = None
    full_text = ""
    for attempt in range(MAX_PARSE_RETRIES):
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=GENERATION_CONFIG["do_sample"],
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            streamer=streamer,
        )

        torch.cuda.synchronize()
        gen_start = time.time()
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        first_token_ts = None
        chunks = []
        for chunk in streamer:
            if first_token_ts is None:
                first_token_ts = time.time()
            chunks.append(chunk)
        thread.join()
        torch.cuda.synchronize()
        gen_end = time.time()

        full_text = "".join(chunks)
        try:
            raw = _extract_last_json_object(full_text)
            parsed = ReasoningOutput(**raw)
            timing = {
                "gen_start_ts": gen_start,
                "first_token_ts": first_token_ts,
                "gen_end_ts": gen_end,
                "output_tokens": len(processor.tokenizer(full_text)["input_ids"]),
                "attempt": attempt + 1,
            }
            return parsed.model_dump(), timing
        except (ValueError, TypeError) as e:
            last_error = e
            continue  # retry: same prompt, same greedy config -> will likely repeat,
            # but kept as a real retry path per the design decision (agreed: retry-parse
            # over adding a constrained-decoding dependency)

    raise RuntimeError(
        f"Failed to parse valid ReasoningOutput JSON after {MAX_PARSE_RETRIES} attempts. "
        f"Last error: {last_error}. Last raw output: {full_text!r}"
    )


def run_baseline_arm(raw_ecg_segment, perception_agent) -> dict:
    """Full Arm A path on one real event. Returns a dict matching ReasoningOutput."""
    health_event = perception_agent.predict(raw_ecg_segment)
    prompt = build_baseline_prompt(health_event)

    model, processor = load_model()
    messages = [{"role": "user", "content": prompt}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        result, _timing = _generate_and_parse(model, processor, inputs["input_ids"], inputs["attention_mask"])
    return result


def run_baseline_arm_timed(input_data, perception_agent=None, rr_interval_ms=None) -> dict:
    """Instrumented Arm A run. Per the design doc's timer-boundary control:
    clock starts when Perception's forward pass completes, stops at
    Reasoning's first output token.

    If perception_agent is provided, input_data is a raw ECG segment and
    rr_interval_ms is passed through to PerceptionAgent.predict() (use the
    real value from data_prep.py's output for genuine evaluation runs —
    omitting it silently falls back to PerceptionAgent's synthetic default).
    If perception_agent is None, input_data is treated as an already-complete
    HealthEventJSON dict (for testing the Reasoning side in isolation).
    """
    if perception_agent is not None:
        health_event = perception_agent.predict(input_data, rr_interval_ms=rr_interval_ms)
    else:
        health_event = input_data
    perception_complete_ts = time.time()

    prompt = build_baseline_prompt(health_event)
    model, processor = load_model()
    messages = [{"role": "user", "content": prompt}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        result, timing = _generate_and_parse(model, processor, inputs["input_ids"], inputs["attention_mask"])

    return {
        "perception_complete_ts": perception_complete_ts,
        "first_reasoning_token_ts": timing["first_token_ts"],
        "result": result,
        "prompt_tokens": prompt_tokens,
        "output_tokens": timing["output_tokens"],
        "generation_duration_ms": (timing["gen_end_ts"] - timing["gen_start_ts"]) * 1000,
        "time_to_first_token_ms": (timing["first_token_ts"] - timing["gen_start_ts"]) * 1000,
        "parse_attempts": timing["attempt"],
    }
