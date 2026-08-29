"""
Ablation copy of run_baseline_arm_timed (baseline_arm.py): identical in every
respect -- still generates a JSON prompt, still decodes text via
_generate_and_parse (imported, not duplicated), still uses the full Arm A
text pipeline -- except the prompt is built with the class-neutral heading
(prompt_template_class_heading_ablation.build_baseline_prompt_neutral)
instead of the class-specific one.

Not an edit to baseline_arm.py: the original stays untouched so the headline
100% Arm A result stays reproducible from the unmodified script.
"""
import time

import torch

from reasoning.baseline_arm import _generate_and_parse
from reasoning.model_loader import load_model
from reasoning.prompt_template_class_heading_ablation import build_baseline_prompt_neutral


def run_baseline_arm_neutral_timed(input_data, perception_agent=None, rr_interval_ms=None) -> dict:
    """Same signature and return shape as run_baseline_arm_timed. Only the
    prompt builder differs -- see module docstring."""
    if perception_agent is not None:
        health_event = perception_agent.predict(input_data, rr_interval_ms=rr_interval_ms)
    else:
        health_event = input_data
    perception_complete_ts = time.time()

    prompt = build_baseline_prompt_neutral(health_event)
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
