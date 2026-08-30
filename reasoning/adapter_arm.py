"""
Arm B (adapter) inference — REAL generate(), not teacher-forced training.

This is the untested path: adapter_training.py only ever calls model()
(forward, one pass, teacher-forced against a known target) to compute loss.
Real evaluation needs model.generate() to autoregressively produce output
from the adapter's virtual tokens, which is a different code path that has
never been exercised. Verify with smoke_test_adapter_generation.py before
trusting this for the full Day 6 comparison.
"""
import threading
import time

import torch
from transformers import TextIteratorStreamer

from reasoning.baseline_arm import _extract_last_json_object, GENERATION_CONFIG
from reasoning.output_schema import ReasoningOutput
from reasoning.virtual_adapter import VirtualTokenAdapter, prepare_adapter_inputs

MAX_PARSE_RETRIES = 3


def load_trained_adapter(checkpoint_path: str, model) -> VirtualTokenAdapter:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    adapter = VirtualTokenAdapter(
        checkpoint["embedding_dim"],
        num_tokens=checkpoint["num_tokens"],
        input_dim=checkpoint["input_dim"],
    )
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    return adapter.to(model.get_input_embeddings().weight.device)


def run_adapter_arm_timed(health_event: dict, context_vector, model, processor, adapter) -> dict:
    """Real generate() through the adapter's virtual tokens. Mirrors
    run_baseline_arm_timed's return shape for direct Arm A/B comparison."""
    context_tensor = torch.from_numpy(context_vector).unsqueeze(0) if hasattr(context_vector, "shape") else context_vector

    perception_complete_ts = time.time()
    adapter_inputs = prepare_adapter_inputs(model, processor, adapter, health_event, context_tensor)

    last_error = None
    full_text = ""
    for attempt in range(MAX_PARSE_RETRIES):
        streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            inputs_embeds=adapter_inputs.inputs_embeds,
            attention_mask=adapter_inputs.attention_mask,
            per_layer_inputs=adapter_inputs.per_layer_inputs,
            do_sample=GENERATION_CONFIG["do_sample"],
            max_new_tokens=GENERATION_CONFIG["max_new_tokens"],
            streamer=streamer,
        )
        # per_layer_inputs MUST always be passed explicitly (surrogate-PAD-ID
        # method, same as training's _append_target_for_teacher_forcing). If
        # omitted, Gemma 4 tries to derive it by comparing the continuous
        # adapter output against the full 262k-token vocab embedding table --
        # a ~197GiB allocation that OOMs immediately. Confirmed by a real
        # crash during smoke testing; this is not a hypothetical risk.

        gen_start = time.time()
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()

        first_token_ts = None
        chunks = []
        try:
            for chunk in streamer:
                if first_token_ts is None:
                    first_token_ts = time.time()
                chunks.append(chunk)
            thread.join()
        except Exception as e:
            last_error = e
            continue
        gen_end = time.time()

        full_text = "".join(chunks)
        try:
            raw = _extract_last_json_object(full_text)
            parsed = ReasoningOutput(**raw)
            output_token_count = len(processor.tokenizer(full_text)["input_ids"])
            return {
                "perception_complete_ts": perception_complete_ts,
                "first_reasoning_token_ts": first_token_ts,
                "result": parsed.model_dump(),
                "prompt_tokens": (
                    adapter_inputs.prefix_token_count + adapter.num_tokens + adapter_inputs.suffix_token_count
                ),
                "output_tokens": output_token_count,
                "generation_duration_ms": (gen_end - gen_start) * 1000,
                "time_to_first_token_ms": (first_token_ts - gen_start) * 1000 if first_token_ts else None,
                "parse_attempts": attempt + 1,
            }
        except (ValueError, TypeError) as e:
            last_error = e
            continue

    raise RuntimeError(
        f"Arm B generation/parsing failed after {MAX_PARSE_RETRIES} attempts. "
        f"Last error: {last_error}. Last raw output: {full_text!r}"
    )
