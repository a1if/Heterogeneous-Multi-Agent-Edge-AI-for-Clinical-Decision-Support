# Reproducing the results

Back to the [README](../README.md). Install the environment first:

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 1. Rebuild the paper's tables and figures (CPU, about a minute)

```bash
python scripts/make_paper_tables.py
python scripts/make_paper_figures.py
cd docs/paper && latexmk -pdf main.tex
```

Both scripts read only `results/`, re-check the numbers quoted in the text, and write `docs/paper/tables/` and
`docs/paper/figures/`.

| Paper item | Result file(s) | Produced by |
|---|---|---|
| Cost table | `p1_e3b_context_costs.json`, `p1_item7_ttd.json` | `p1_e3b_context_costs.py`, `p1_item7_ttd.py` |
| Accuracy table, accuracy figure | `p1_item7_r4_analysis.json` | `p1_item7_r4_analysis.py` |
| Filtered text | `p1_item7_filtered.json` | `p1_item7_filtered.py` |
| Serving table and figure | `p1_serving.json` | `p1_serving.py` |
| Recoverability | `p1_item7_r4_analysis.json`, `p1_item7_runlen.json` | `p1_item7_slot_decoder.py`, `p1_item7_runlen.py` |
| Second sender | `p1_second_sender_analysis.json` | `p1_second_sender_analysis.py` |
| Compression sweep, ablation | `p1_sweep_analysis.json`, `p1_dev27_kseeds.json`, `p1_dev26_ablation.json` | `p1_sweep_analysis.py`, `p1_seed_group_analysis.py --dev 27`, `--dev 26` |
| INCART tables | `p1_confirmatory_incart.json`, `p1_incart_timing_n50_capped.json` | `p1_confirmatory_analysis.py`, `p1_incart_timing_recheck.py` |
| Bandwidth | `p1_bandwidth.json` | `p1_bandwidth.py` |
| Learning curves | `p1_learning_curves.json` | `p1_learning_curves.py` |
| Position of the deciding beat | `p1_position_effect.json` | `scripts/position_effect.py` |

## 2. Re-run the analyses from the raw outputs (CPU)

Every analysis re-derives its numbers from the committed per-window outputs in `results/`. Each one was re-run from a
clean checkout and reproduced the committed result exactly. They run on the CPU but load the sender checkpoint, which
was saved on a GPU, so run them on a machine with CUDA. The first one also rebuilds the sender's replay cache
(about 4 minutes per split).

```bash
python p1_item7_r4_analysis.py                 # accuracy, cost, recoverability (about 16 min)
python p1_seed_group_analysis.py --dev 26      # side-input ablation (about 8 min)
python p1_seed_group_analysis.py --dev 27      # tokens per event, three runs (about 8 min)
python p1_sweep_analysis.py                    # compression sweep (about 4 min)
python p1_item7_runlen.py                      # abnormal-run recoverability (about 3 min)
python p1_confirmatory_analysis.py             # INCART hypotheses H1-H4 (about 1.5 min)
python p1_e3b_context_costs.py
python p1_bandwidth.py
python p1_learning_curves.py
python scripts/position_effect.py
P1_ENCODER=perception/checkpoints/resnet1d_rr_seed0.pt P1_ENCODER_TAG=_res python p1_second_sender_analysis.py   # needs the second sender's checkpoint (about 13 min)
```

## 3. Re-run everything (GPU)

Requirements: an NVIDIA GPU with 12 GB, a Hugging Face account with the
[Gemma licence](https://huggingface.co/google/gemma-4-E4B-it) accepted (`huggingface-cli login`), and PhysioNet access
for the data. Each step is resumable and writes to `results/`.

```bash
# Data: MIT-BIH (downloaded and split into DS1/DS2) and INCART (download incartdb 1.0.0 to data/incartdb/ first)
python data_prep.py

# Sender (checkpoint perception/checkpoints/cnn_lstm_rr_seed0.pt is included), then replay both splits through it
python train_perception_agent_rr.py --seed 0
python -c "from p1_item7_common import replay_split; replay_split('ds1'); replay_split('ds2')"

# DS2 test windows
python p1_item7_testset_v2.py

# Final adapter, three runs (checkpoints reasoning/checkpoints/p1_item7_mea_r4_seed{101,202,303}.pt are included)
python p1_item7_train.py --recipe r4 --seed 101     # and 202, 303

# DS2 evaluation: adapter runs and text arms, calibration logits, token counts, timing, serving
python p1_item7_eval.py --split ds2v2 --suffix r4 --arms MEA:r4_seed101 MEA:r4_seed202 MEA:r4_seed303 A-compact A-filtered
python p1_item7_calib.py collect && python p1_item7_calib.py analyse
python p1_item7_filtered.py collect --text-arm A-compact
python p1_item7_filtered.py collect --text-arm A-filtered
python p1_item7_filtered.py tokens && python p1_item7_filtered.py timing
python p1_item7_ttd.py
python p1_serving.py

# External test on INCART with the frozen system
python p1_freeze.py
python incart_prep.py
python -c "from p1_item7_common import replay_split; replay_split('incart')"
python p1_incart_windows.py
python p1_item7_eval.py --split incart --arms MEA:r4_seed101 MEA:r4_seed202 MEA:r4_seed303
python p1_item7_eval.py --split incart --stop-at-tier --arms A-compact A-filtered
python p1_item7_filtered.py collect --text-arm A-compact --split incart
python p1_item7_filtered.py collect --text-arm A-filtered --split incart
python p1_item7_filtered.py timing --split incart
python p1_incart_timing_recheck.py
python p1_confirmatory_analysis.py
```

The second sender uses the same commands with `P1_ENCODER=perception/checkpoints/resnet1d_rr_seed0.pt` and
`P1_ENCODER_TAG=_res` set, the evaluation suffix `r4_res` and arms `MEA:r4_res_seed101` and so on (train the sender
with `python train_perception_agent_rr.py --arch resnet1d_rr --seed 0`; its checkpoints are not included).
Ablation and compression variants use `p1_item7_train.py --recipe r4hn | r4k1 | r4k2 | r4k8`.

Reference timings on an RTX 5070 (12 GB): one DS2 evaluation of all arms about 10 hours; INCART about 10 hours;
serving benchmark 3 to 4 hours.

## Tests

```bash
pytest                                                               # all tests
pytest --ignore=tests/test_day2_baseline_arm.py --ignore=tests/test_day3_adapter.py   # CPU only (49 tests)
```

