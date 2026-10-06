# Run order to 19 October 2026

Every GPU step runs one at a time. Scripts marked "to write" are written and pre-registered during the day before their run.
`res` means the Deviation 22 sender: set `P1_ENCODER=perception/checkpoints/resnet1d_rr_seed0.pt` and `P1_ENCODER_TAG=_res`.

## Blocker before any GPU step
- `llama-server` (not part of this project) holds about 2.5 GB of RAM and about 3 GB of VRAM. Gemma's GPU tests now crash with a Windows access violation (they passed at 15:43, before it was running). Close it, then run `pytest tests/test_day2_baseline_arm.py` (about 3 min) to confirm.

## Thu 1 Oct, evening and night: second sender (Deviation 22)
| # | Step | Command | Time | Gate / output |
|---|---|---|---|---|
| 1 | Train ResNet1D-RR sender, seed 0 | `python train_perception_agent_rr.py --arch resnet1d_rr --seed 0` | about 2 min | **gate:** best DS1 validation accuracy >= 0.85 |
| 2 | Replay DS1 and DS2 through it | `res` + `python -c "from p1_item7_common import replay_split; replay_split('ds1'); replay_split('ds2')"` | about 20 min | cached events and vectors |
| 3 | Build its DS2 test windows | `res` + `python p1_item7_testset_v2.py` | under 1 min | `results/p1_item7_testset_v2_res.json` |
| 4 | r4 smoke test | `res` + `python p1_item7_train.py --smoke --recipe r4` | about 5 min | **gate:** memory_ok |
| 5 | Adapter training, seeds 101 / 202 / 303 | `res` + `python p1_item7_train.py --recipe r4 --seed S` | about 8 h | `p1_item7_mea_r4_res_seedS.pt` |

## Fri 2 Oct
| # | Step | Time | Note |
|---|---|---|---|
| 6 | **Serving benchmark (Deviation 23, to write):** batch 1/4/8, prefix caching of the scaffold, throughput, peak memory, energy; arms MEA r4, A-compact, A-filtered; N = 10 / 20 / 50 | about 3–4 h (day) | after step 5 finishes |
| 7 | Second-sender DS2 generation: MEA ×3, A-compact, A-filtered | about 10 h (night) | `res` + `p1_item7_eval.py --split ds2v2 --suffix r4_res --arms ...` |

## Sat 3 Oct
| # | Step | Time | Note |
|---|---|---|---|
| 8 | Second sender: calibration logits (A-compact and A-filtered), timing, token counts; analysis for RQ1–RQ3 (to write: deltas, CIs, probe and slot decoding) | about 1 h GPU + CPU | `res` + `p1_item7_filtered.py collect [--text-arm A-compact]`, `timing`, `tokens` |
| 9 | **Compression sweep (Deviation 24, to write):** r4 recipe at k = 1, 2, 8 tokens per event, seed 101 | about 7.5 h (night) | k = 4 is r4 seed 101 |

## Sun 4 Oct
| # | Step | Time |
|---|---|---|
| 10 | Sweep DS2 evaluation (MEA k = 1, 2, 8) + per-slot decodability (CPU) + rate–accuracy–recoverability analysis | about 2 h GPU + CPU |
| 11 | Protocol-selection guide (analysis only) | CPU |

## Mon 5 Oct: freeze and confirm
| # | Step | Time | Gate |
|---|---|---|---|
| 12 | **Freeze:** manifest (hashes of checkpoints, code, biases, versions), git tag `r4-confirmatory` (to write) | under 1 h | nothing changes after this |
| 13 | INCART conversion: `python incart_prep.py` | about 5 min | inspect conversion checks only |
| 14 | INCART replay + windows: `replay_split('incart')` + test-set build for `incart` (to write: an `incart` option in the builder) | about 15 min | window list hash recorded |
| 15 | INCART single run: MEA ×3, A-compact, A-filtered, calibration logits (frozen biases), timing | about 10–11 h (night) | no reruns except technical retries |

## Tue 6 – Wed 7 Oct
| # | Step |
|---|---|
| 16 | INCART analysis: H1 -> H4 in fixed sequence (to write: confirmatory analysis script with the patient bootstrap) |
| 17 | Buffer nights for technical reruns only |

## Writing, in parallel from Fri 2
- Outline and related work: Fri 2 – Sat 3.
- Methods: Sun 4 – Tue 6.
- Results and discussion: Wed 7 – Sat 10.
- Artifact package: Thu 8 – Sat 10.
- Full draft: Sun 11. Supervisor: Mon 12 – Wed 14. Revise: Thu 15 – Sat 17. Final check: Sun 18. **Submission-ready: Mon 19.**
