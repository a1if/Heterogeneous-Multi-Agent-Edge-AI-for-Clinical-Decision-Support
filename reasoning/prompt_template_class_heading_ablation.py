"""
Ablation: Arm A's context-injection function (build_baseline_prompt in
prompt_template.py) with the class-specific heading swapped for Arm B's
already-fixed class-neutral block (reasoning/fixed_context_neutral.py) --
NEUTRAL_CONTEXT is imported, not re-paraphrased, so this ablation changes
exactly one variable relative to the original.

Not an edit to prompt_template.py: the original stays untouched so the
headline 100% Arm A result stays reproducible from the unmodified script.
SYSTEM_PROMPT, OUTPUT_INSTRUCTIONS, and _extract_prompt_fields are imported
from prompt_template.py unchanged -- only the background-context content and
its heading differ, and only by exactly as much as Arm B's fix already
differs from Arm A (same heading text Arm B uses: "--- Background context
---", no class name).
"""
import json

from reasoning.fixed_context_neutral import NEUTRAL_CONTEXT
from reasoning.prompt_template import (
    OUTPUT_INSTRUCTIONS,
    SYSTEM_PROMPT,
    _extract_prompt_fields,
)


def build_baseline_prompt_neutral(health_event: dict) -> str:
    """Byte-identical to build_baseline_prompt() except the background-context
    block: class-neutral (Arm B's block, copied verbatim) instead of
    class-specific. Same purity guarantee: same input -> byte-identical
    output, every call."""
    fields = _extract_prompt_fields(health_event)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Background context ---\n"
        f"{NEUTRAL_CONTEXT}\n\n"
        f"--- Event data ---\n"
        f"{json.dumps(fields, indent=2)}\n\n"
        f"--- Instructions ---\n"
        f"{OUTPUT_INSTRUCTIONS}"
    )
    return prompt
