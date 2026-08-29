"""
Loads the SAME Gemma 4 E4B checkpoint as model_loader.py, but 8-bit
quantized instead of 4-bit NF4 -- for the quantization-coupling pilot only
(see quantization_coupling_pilot.py). Deliberately a separate module with
its own cache, not an edit to model_loader.py: the study's headline results
all load through load_model()'s 4-bit path, and that must stay untouched and
reproducible. This tests whether the trained adapter, calibrated to the
4-bit embedding geometry, still works when that geometry shifts under a
different precision of the identical weights -- not a claim about which
quantization is "correct" or preferred.

Requires: same `huggingface-cli login` / Gemma license acceptance as
model_loader.py.
"""
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma4ForConditionalGeneration

from reasoning.model_loader import MODEL_ID

_model_8bit = None
_processor_8bit = None


def load_model_8bit():
    """Returns (model, processor), 8-bit quantized. Loads once, cached for
    the process lifetime -- separate cache from load_model()'s 4-bit model,
    so both could in principle be loaded in the same process (VRAM
    permitting), though this pilot only ever loads one at a time."""
    global _model_8bit, _processor_8bit
    if _model_8bit is None:
        print(f"Loading {MODEL_ID} (8-bit, first call only)...")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        _processor_8bit = AutoProcessor.from_pretrained(MODEL_ID)
        _model_8bit = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="cuda",
            attn_implementation="sdpa",
        )
        _model_8bit.eval()
        print("8-bit model loaded.")
    return _model_8bit, _processor_8bit
