# Phase 1 Analysis Plan

Status: **pre-specified**. Written and committed before any Phase 1 run. The commit timestamp is the pre-specification record. Any later change goes under "Deviations" at the end, with date and reason. Earlier text is never edited silently.

Scope: Phase 1 steps 1–10 of `docs/publication_plan.md` §9. The dissertation's results (N=80, `virtual_adapter_day5_larger.pt`) are the *prior* study. Phase 1 is a replication and extension of it, not a re-analysis.

---

## 1. Corrections to the prior record found while writing this plan

1. **Number of weight updates.** Dissertation §3.5.1 says the adapter was trained "as a single full batch across 3 epochs, giving three weight updates in total". `reasoning/adapter_training.py` calls `optimizer.step()` once per example: 64 examples × 3 epochs = **192 updates**, batch size 1, fixed example order. The code is authoritative. The paper will describe the regime as 192 per-example AdamW updates. Publication-plan blocker B5 wording is corrected accordingly.
2. **The prior headline is one favourable draw.** Same config, three initialisations: unseeded 95.0% (76/80), seed 101 81.3% (65/80, 2 generation failures), seed 202 78.8% (63/80) (`results/e2_seed_variance_results.json`). The 95.0% is therefore not treated as the expected accuracy of the method.

## 2. Headline reporting rule (fixed now, before new results exist)

- The Phase 1 headline for every Arm B quantity is the **mean over 5 training seeds** {101, 202, 303, 404, 505}, with a 95% t-interval over seeds. It is also reported with the pooled per-event interval (events × seeds, clustered by record).
- **No single checkpoint is selected as "the" headline after seeing results.** The unseeded `virtual_adapter_day5_larger.pt` is reported as a sixth, unrecorded-seed draw, labelled as such, and is never in the mean.
- **Reference checkpoint** for single-checkpoint analyses (per-event plots, the N=60 invariance run, the probe confusion matrix) is **seed 101**, chosen now because it is the lowest pre-existing seed, not because of its score. Every probe/attribution number is also reported across all 5 seeds.
- Text arms (A-*) use greedy decoding, have no trained component, and are run once. Their run-to-run variability is quantified by step 3 (nondeterminism).

## 3. Arms

| Arm | Payload | First run in |
|---|---|---|
| A-full | current prompt-subset JSON (`reasoning/prompt_template.py`, few-shot, frozen per `BASELINE_FROZEN.json`) | step 1 (N=60), step 4 |
| A-compact | class, confidence, consecutive_abnormal_beats, SQI | step 4 |
| A-label | class label only | step 4 |
| A-rule | no LLM; `urgency_tier_from_event` applied to the *predicted* event | step 4 |
| B-k | k virtual tokens, k ∈ {1,2,4,8}; k=4 is the primary | steps 1, 6 |
| B-null | learned k=4 prefix, zero context vector at train and test | step 4 |
| B-shuffle | trained B-4, context vector from a random other event (fixed permutation, seed 0) | step 4 |

Everything else is held fixed as in the dissertation (Table 3.1): frozen Gemma 4 E4B NF4, frozen CNN-LSTM `perception/checkpoints/cnn_lstm.pt`, greedy decoding, interleaved run order per event, same timer boundaries.

## 4. Evaluation sets

- **E80:** the existing `select_events` set (20/class N/S/V/F, ≤ 5 per record). Primary set for steps 1–8.
- **E60:** `select_events(per_class=15)`, same cap. Used *only* for the step 1 invariance check that replaces the lost 60-event raw output. It is not a copy of the original file, which is unrecoverable. It is a fresh, fully logged run under the stated reference checkpoint.
- **E400 (step 10):** stratified DS2 draw, 100/class where available, per-record cap removed. Fusion takes every available event if fewer than 100 exist. Drawn with seed 0 and frozen as an index file before any arm is run on it.

## 5. Endpoints

**Primary (efficiency, interface-attributable):**
- P1: interface-attributable prompt tokens (payload tokens only, scaffold excluded), per event.
- P2: prefill time (time-to-first-token), per event.

**Secondary:**
- S1: end-to-end prompt tokens, total generation time, generation energy, peak VRAM (dissertation measures, for continuity).
- S2: decode time and output tokens, reported separately (step 5), plus the same under fixed `max_new_tokens`.
- S3: urgency-tier accuracy against the reference tier (constraint check, as before).
- S4 (step 9): multi-flag decision accuracy and unsupported-claim rate.
- S5: class recoverability (linear probe, dissertation protocol), inverse-decoder field accuracy, confidence R² (step 7).
- S6: generation-failure rate, and anomaly-check detection rate (step 8).

## 6. Pre-specified gates

**Step 1 gate (reproducibility):**
- (a) *Training determinism.* Retraining seed 101 with identical config reproduces `virtual_adapter_e2_seed101.pt`, with max |Δw| ≤ 1e-4. If it does not, training nondeterminism is reported as a finding, and all seeds are retrained in one session so the checkpoints share a software state.
- (b) *Replication.* The 5-seed mean accuracy on E80 is reported whatever its value. Token count must equal 486 for every seed, because construction fixes it. Any deviation is a pipeline bug and stops Phase 1.
- (c) *Invariance.* On E60 vs E80 with the reference checkpoint: A-full accuracy and the end-to-end token reduction must fall within the E80 95% CIs.

**P1 gate (after step 4, decides framing, not whether to publish):**
- If B-4 has fewer interface-attributable tokens than A-label **and** non-inferior S3 (lower 95% bound of the difference > −10 pp), then the framing is "latent interface dominates on cost".
- Otherwise the framing is "trade-off characterisation: when a latent interface pays off and what it costs in recoverability". The abstract is written for this case by default.

## 7. Statistics

- Paired arm contrasts on continuous measures: Wilcoxon signed-rank, plus a paired bootstrap 95% CI on the relative reduction (20,000 same-event resamples, as in the dissertation). Effect size: Cliff's δ.
- Accuracy contrasts: exact McNemar (one-sided only where one discordant cell is structurally empty, as in dissertation §3.8), with Wilson or exact intervals.
- Multiple comparisons: Holm across the arm family, applied separately within each endpoint.
- Seeds: mean ± 95% t-interval over 5 seeds. The seed SD is reported alongside the between-arm difference.
- E400 (step 10): mixed-effects logistic regression `correct ~ arm + (1 | record)`, and a linear mixed model for tokens and time with the same random effect.
- α = 0.05, two-sided unless stated. Report exact p-values and CIs, never significance stars alone.

## 8. Provenance

Every run writes a `results/p1_*.json` file containing the git commit, checkpoint SHA-256, seed, library versions, and the event index list. Numbers reach the paper only through `results_ledger.json`. The existing `virtual_adapter_*` checkpoints are never overwritten. Phase 1 checkpoints use the `p1_` prefix.

## 9. Deviations

**Deviation 1 (2026-09-22, before any E60 data was produced): E60 GPU rerun dropped.** `select_events(per_class=15)` is a strict subset of the E80 set (60/60 events overlap), so a GPU rerun on E60 adds no independent evidence. E60 figures for gate 6c are now computed offline. Arm A comes from the dissertation's reported Day 6 pass (`results/day6_results.json.bak_pre_rerun_20260816`), and Arm B from the seed-101 E80 per-event results of step 1. Only accuracy and prompt tokens are reported, because the two arms come from different sessions and their latencies are not paired. Consequence for the paper: dissertation §4.8's "stability across sample sizes" compared a set with its own superset. It is not a robustness test, and it will not be presented as one. Step 10 (E400) is the stability test.

**Deviation 2 (2026-09-22): RR-branch encoder added as a new, unplanned arm of the encoder sweep (plan §2.6.2).** The reference CNN-LSTM sees a single 1-s window and no RR intervals, yet the JSON arm passes `rr_interval_ms` to the LLM. Part of the measured "upstream bottleneck" is therefore information the encoder was never given. A check made before training (no model outputs involved): 85% of DS2 S beats have pre-RR / local-average-RR < 0.85, against 4% of N beats. The new encoder (`perception/model_rr.py`, features in `perception/rr_features.py`) matches the reference in every respect except an RR embedding fed into the context LSTM. The 32-d context vector keeps its extraction point. The reference CNN-LSTM stays the primary encoder for every pre-specified endpoint. RR-encoder results are reported as the encoder-sweep contrast: DS2 accuracy, per-class Se/+P against the reference under identical metric code, then probe recoverability at the context vector under the dissertation protocol. Post-RR needs the next beat, so an online agent using this encoder emits each beat one beat late. This is reported as a latency cost.

**Deviation 3 (2026-09-23, before any step 4 data): step 4 runs before step 1 finishes.** The run order changed to "least GPU first", and step 1's seeded retrain has not run yet. So B-4 and B-shuffle use `virtual_adapter_e2_seed101.pt`, which has the same seed (101) and the same config as the pre-specified reference checkpoint. B-null is trained with seed 101 on all-zero context vectors (`reasoning/checkpoints/p1_bnull_k4_seed101.pt`). If step 1's determinism gate (6a) shows that the retrained seed-101 checkpoint differs from the E2 one, the B arms of step 4 are rerun with the retrained checkpoint, and both results are reported. A-compact's payload is exactly the §3 list: label, confidence, consecutive_abnormal_beats, SQI. Every text arm reuses A-full's scaffold byte for byte, and only the event-data block changes. A-rule is scored offline. Arm order is rotated per event.
