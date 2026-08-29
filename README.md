# Heterogeneous Multi-Agent Edge AI for Clinical Decision Support

Research code for a dissertation investigating whether a learned adapter that
projects a non-transformer perception model's internal feature representation
directly into a language model's input embedding space can replace a
text-based agent-to-agent interface, reducing inter-agent communication cost
(tokens, latency, memory) while preserving task accuracy  and at what
auditability cost.

## Research Question

> Can a small, learned adapter that projects a non-transformer perception
> model's internal feature representation directly into a language model's
> input embedding space,  replacing a text-based structured interface ,
> reduce inter-agent communication cost (tokens, latency, memory) compared to
> the text-based baseline, while preserving task accuracy, and what
> auditability cost does this trade-off impose relative to the structured
> text interface?

## System Overview

The system is a two-agent pipeline evaluated under two competing interfaces
for passing information between agents.

**Perception Agent** (`perception/`) — a CNN-LSTM classifier trained on the
five-way AAMI EC57 arrhythmia classification task from single-lead ECG
windows. Three 1D convolutional blocks feed a bidirectional LSTM, reduced by
a unidirectional context LSTM to a 32-dimensional context vector immediately
before the classification head. This context vector is the sole quantity
passed downstream to the adapter.

**Reasoning Agent** (`reasoning/`) — Gemma 4 E4B, loaded via HuggingFace
`transformers` with 4-bit (NF4) quantisation through `bitsandbytes`, frozen
in both experimental arms (no fine-tuning of the LLM itself).

**Two arms, one controlled variable** — whether an event crosses the
Perception-to-Reasoning boundary as:

- **Arm A (baseline):** generated JSON text, following a structured
  text-based interface.
- **Arm B (treatment):** adapter-projected virtual tokens. `VirtualTokenAdapter`
  is a single linear layer mapping the Perception Agent's 32-dimensional
  context vector directly into Gemma 4 E4B's 2,560-dimensional input
  embedding space, bypassing text generation entirely.

Every other variable — model weights, input events, prompt scaffolding, test
set, decoding strategy, run order, timer boundaries — is held identical
across both arms. Four dependent variables are measured directly (token
count, generation latency, VRAM footprint, task accuracy), plus auditability,
assessed separately via a held-out linear-probe protocol that recovers the
true class label from each arm's transmitted representation.

## Dataset

The MIT-BIH Arrhythmia Database (Moody and Mark, 2001), accessed via the
`wfdb` Python library under its Open Data Commons Attribution licence. Paced
recordings are excluded following standard practice, leaving 44 recordings.
Records are split by patient (inter-patient split, following de Chazal,
O'Dwyer and Reilly, 2004) into DS1 (training) and DS2 (testing), with no
patient appearing in both sets, to avoid morphology leakage inflating
reported accuracy.

The raw dataset is not included in this repository. See
`data_prep.py` for the preprocessing pipeline that extracts, windows, and
splits the data once acquired.

## Repository Structure

```
perception/           CNN-LSTM Perception Agent (model, schema, checkpoint dir)
reasoning/             Arm A / Arm B reasoning pipeline: baseline arm, adapter
                       arm, virtual-token adapter, prompt templates, training
tests/                 pytest suite
diagnostics/           One-off manual verification / smoke-test scripts
archive/               Superseded scripts kept for reference, not run
results/               Raw JSON outputs from the scripts below
results_ledger.json    Single source of numeric truth, consumed by render_ledger.py
```

Active pipeline scripts are deliberately kept flat at the repository root
(`data_prep.py`, `train_perception_agent.py`, `day6_run_comparison.py`,
`day7_auditability_probe.py`, `e0_statistics.py`, `e1_ablation.py`,
`e2_seed_variance.py`, `render_ledger.py`, and related checks) — they rely on
Python adding an invoked script's own directory to `sys.path[0]` to resolve
`perception.*` / `reasoning.*` imports, so they must be run from the
repository root. See `PROJECT_LAYOUT.md` for the full rationale and import
dependency graph between scripts.

## Requirements

- Python 3.13
- PyTorch with CUDA support, `transformers`, `bitsandbytes`
- `wfdb`, `numpy`, `scipy`, `pandas`, `scikit-learn`
- Access to `google/gemma-4-E4B-it` (gated model; requires licence acceptance
  and `huggingface-cli login`)
- An NVIDIA GPU with at least 12GB VRAM (development and evaluation were
  performed on a single RTX 5070, 12GB GDDR7)

## Reproducing the Experiments

1. Acquire the MIT-BIH Arrhythmia Database and run `data_prep.py` to produce
   the DS1/DS2 splits under `data/`.
2. Train the Perception Agent: `python train_perception_agent.py`.
3. Train the adapter (Arm B): see `reasoning/adapter_training.py`.
4. Run the comparison and statistics scripts from the repository root, e.g.
   `python day6_run_comparison.py`, `python e0_statistics.py`,
   `python e1_ablation.py`.
5. Regenerate the results ledger with `render_ledger.py`.

Each script's own docstring documents its exact invocation and outputs.

## Notes on Scope

This repository contains the experimental code and results supporting the
dissertation. It does not include the dissertation manuscript itself, raw
dataset files, trained model checkpoints, or the Python virtual environment;
see `.gitignore`.

## Author

Alif Tasbir
