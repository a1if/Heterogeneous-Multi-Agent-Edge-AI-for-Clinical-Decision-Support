"""
Loads Gemma 4 E4B once via HuggingFace transformers, bitsandbytes 4-bit
quantized. Both Arm A (baseline, input_ids) and Arm B (adapter, inputs_embeds)
must load through this SAME function to satisfy the "identical checkpoint and
quantisation, both arms" control (design doc Section 3).

Requires: `huggingface-cli login` already run, Gemma license accepted at
https://huggingface.co/google/gemma-4-E4B-it
"""
import torch
from transformers import AutoProcessor, Gemma4ForConditionalGeneration, BitsAndBytesConfig

MODEL_ID = "google/gemma-4-E4B-it"

_model = None
_processor = None


def load_model():
    """Returns (model, processor). Loads once, cached for the process lifetime."""
    global _model, _processor
    if _model is None:
        print(f"Loading {MODEL_ID} (4-bit, first call only)...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quant_config,
            device_map="cuda",
            attn_implementation="sdpa",  # avoid the slow "eager" default
        )
        _model.eval()
        print("Model loaded.")
    return _model, _processor
