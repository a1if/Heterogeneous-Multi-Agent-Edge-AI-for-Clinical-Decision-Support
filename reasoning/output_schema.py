"""
Reasoning Agent output schema — strict, per design decision (7-day prototype).

Both Arm A (JSON baseline) and Arm B (adapter) must produce output validating
against this exact schema, so the two arms are comparable on accuracy without
any parsing ambiguity. This is the "downstream task, defined precisely"
target from the research gap brief, Section 5.
"""
from typing import Literal
from pydantic import BaseModel, field_validator


class ReasoningOutput(BaseModel):
    urgency_tier: Literal["routine", "priority", "urgent"]
    justification: str
    # The specific fixed-context fact the model drew on, echoed explicitly —
    # not attributed to any named guideline body (design decision: content is
    # generic paraphrased clinical reasoning, see reasoning/fixed_context.py).
    referenced_guideline_fact: str

    @field_validator("justification")
    @classmethod
    def justification_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("justification must not be empty")
        return v

    @field_validator("referenced_guideline_fact")
    @classmethod
    def referenced_fact_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "referenced_guideline_fact must not be empty — the downstream task "
                "requires the output to reference at least one injected context fact "
                "(research gap brief, Section 5)."
            )
        return v
