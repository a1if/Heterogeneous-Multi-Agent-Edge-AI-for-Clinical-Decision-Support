# Feasibility: Qwen3.5-4B as a second language model

Written 2026-10-01. Based on the model card, config and file list, the local software versions, and a code audit. No weights have been downloaded and no GPU test has been run.

## Verdict
**Feasible, with two conditions:**
1. Gemma-specific input code needs a small model-agnostic layer. This is additive: frozen Gemma paths stay byte-identical.
2. Qwen3.5's linear-attention layers have no fast kernel installed here, so speed and timing must be measured with that caveat, or the kernel installed in a separate environment.

The real cost is GPU time: about 25–30 h for a full replication on DS2 and INCART.

## Model facts
| | Gemma 4 E4B (current) | Qwen3.5-4B |
|---|---|---|
| Hidden size | 2,560 | 2,560 |
| Layers | 42, transformer | 32, hybrid: 3 linear-attention (Gated DeltaNet) : 1 full-attention, repeating |
| Vocabulary | about 262k | 248,320 |
| Per-layer embeddings (PLE) | yes, which our input code must build | no |
| Default mode | direct answer | **thinking by default**; disabled with `enable_thinking=False` |
| Modality | multimodal | multimodal (vision encoder); a text-only `Qwen3_5ForCausalLM` class exists locally |
| Weights | already local | **9.3 GB** (2 safetensors files) plus about 23 MB of tokenizer files; Apache-2.0 |
| Library support | – | `qwen3_5` is in the installed transformers 5.14.1; no upgrade needed |

## What has to change (code)
- **Loader:** load `Qwen/Qwen3.5-4B` in 4-bit with the text-only class, and apply the chat template with thinking disabled.
- **Input composition:** Gemma needs `per_layer_inputs` built from PAD surrogates (`virtual_adapter.py`, `adapter_training.py`, and the eval, calibration and timing scripts). Qwen needs only `inputs_embeds` and `attention_mask`. Add a backend switch that returns no extra inputs for Qwen, leaving Gemma's path unchanged.
- **Constrained decoder:** `TokenTable` derives its forced tokens from the tokenizer, so it should carry over, but its unit tests must pass on Qwen's tokenizer: tier words, closers, and merged quote tokens.
- **Adapter:** same architecture (2,560-d output, 4 tokens per event, 50 slots). The scale is initialised from Qwen's own embedding norms. It must be **retrained** on Qwen, because the embedding spaces differ.

## Risks
1. **Speed of the linear-attention layers.** `flash-linear-attention` and `causal-conv1d` are not installed, so transformers uses its pure-PyTorch Gated DeltaNet fallback. Training and generation may be several times slower, and Windows support for the fast kernels (via Triton) is uncertain. Mitigation:
   - run a speed smoke test first;
   - if it's too slow, install the kernels in a **separate venv**, so the frozen `r4-confirmatory` environment is untouched.
2. **The efficiency claim may weaken on Qwen.** With 24 of 32 layers in linear attention, the cost of a longer text prompt grows more slowly than on Gemma, so fewer tokens buy less time. That is a legitimate scientific outcome, which is why a second model is worth testing.
3. **Thinking mode leaking.** The template must have thinking switched off; the constrained decoder forces JSON from the first token anyway. Smoke-check the template text.
4. **Memory:** about 3–4 GB in 4-bit, so training at about 700-token sequences should fit the 12 GB card. It must be measured by a smoke test, as for r4.

## Proposed design (to pre-register as Deviation 22 before any Qwen training)
- **Transfer the r4 recipe unchanged:** the same data, hard negatives, side inputs, schedule, early-stopping rule, and seeds 101, 202 and 303. **No tuning on Qwen.**
  - Because nothing is tuned on Qwen, Qwen's DS2 result is a fair test of whether the method transfers.
- **Arms:** Qwen-r4 (3 seeds), Qwen A-compact, and Qwen A-filtered (zero-shot, with calibration by the Deviation 17 rule on DS1 validation).
- **Evaluation:** first on DS2 (the 1,195 windows). Then add Qwen to the INCART confirmatory run as a **second hypothesis family**, using the same H1–H4 structure. Gemma stays the primary family.
  - This amendment to `docs/confirmatory_plan.md` must be made before any INCART data exists.

## Cost estimate (GPU, RTX 5070)
| Step | Estimate | Note |
|---|---|---|
| Download | 9.3 GB | Hugging Face is usually much faster than PhysioNet |
| Code + tests | about 3–4 h of my work | no GPU |
| Smoke test (memory, speed, template) | about 20 min | after the current DS2 run ends (about 16:45) |
| Qwen-r4 training, 3 seeds | about 7–8 h if speed is close to Gemma's | could be 2–4× more on the fallback kernel |
| DS2 evaluation (3 adapter seeds + 2 text arms + calibration + timing) | about 8 h | text arms generate full answers |
| INCART confirmatory, Qwen family | about 9 h | added to the overnight run |
| **Total** | **about 25–30 h** (more if the fallback is slow) | about 3 overnight runs |

## Order
1. Freeze and tag `r4-confirmatory` (Gemma) first. Qwen code then goes on a branch from the tag, and the Gemma INCART run uses the tag.
2. Download Qwen, add the backend switch and tests, run the smoke test, and decide from it whether a fast-kernel venv is needed.
3. Pre-register Deviation 22 and amend the confirmatory plan (both before any INCART data). Then train, evaluate on DS2, and run INCART.
