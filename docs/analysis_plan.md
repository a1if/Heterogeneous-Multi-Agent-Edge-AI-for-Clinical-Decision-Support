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

**Deviation 4 (2026-09-23, before any step 5 data): the length-controlled condition forces output length instead of changing the output schema.** Publication plan §2.2 proposed a schema with no free-text justification. That would need the adapter retrained on new targets. Instead, every arm generates exactly N tokens (min_new_tokens = max_new_tokens = N, greedy), with N = 1 (prefill / time to first token) and N = 64. Per-token decode cost = (t64 - t1) / 63. Output length is therefore equal across arms by construction, and content is not scored (step 4 covers accuracy). Arms: A-full, A-compact, A-label, B-4 (B-null and B-shuffle have B-4's input cost by construction). Energy is NVML power at 100 Hz, integrated per call, gross and net of an idle baseline taken before and after. Step 4 already showed equal decode speed across arms under natural lengths (~169 ms per output token), so step 5 is the controlled confirmation.

**Deviation 5 (2026-09-23, after steps 4-5, before any of the experiments below): reprioritisation toward the efficiency claim, and the efficiency gate.** Steps 4-5 showed that at one event, batch 1, the latent interface has no efficiency advantage over a compact text interface (A-compact: 100% accurate, 458 vs 486 prompt tokens; with output length forced equal, time and energy within 3%). The dissertation's efficiency claim is re-tested in the regimes where per-event payload is multiplied. Run order, cheapest compute first:
1. E3, multi-event context: N = 1, 5, 10, 20, 50 consecutive events in one prompt, for A-full, A-compact and B-4. Prefill time (forced 1 token), peak allocated memory, and energy (forced 16 tokens). Text arms carry N payload blocks. B-4 carries N x k virtual tokens from the existing single-event adapter. Cost only; accuracy is not claimed for N > 1 (that is item 7).
2. E1, cached-scaffold per-event cost: the static scaffold's KV cache is computed once, and per-event cost is the incremental prefill of the payload only (text: payload tokens; B: k).
3. E2, batched serving: batch 1-32 with forced short output; events per second, peak memory, and the largest batch that fits in 12 GB.
4-6. Accuracy: tier-balanced adapter training (3 seeds, then 5), and the step 1 5-seed headline.
7. Architecture v2 (temporal compressor: N events -> k tokens, trained on a multi-event task), only if E3 passes the gate.

**Efficiency gate, fixed before any E1-E3 data exists:** an efficiency advantage is claimed for a regime only if B-4 beats A-compact (the best text interface that matches its accuracy) on that regime's primary metric, with the 95% CI of the relative difference excluding zero and a point estimate of at least 10%. Primary metrics: E3, prefill time at each N; E1, incremental per-event prefill time; E2, events per second at the largest batch both arms fit. Every regime and every N is reported, pass or fail.

**Generation cap.** `max_new_tokens` goes from 1024 to 256 for all runs from now on. The longest valid output observed in 476 step 4 outputs and the dissertation's Day 6 pass is 140 tokens, so the cap cannot truncate a valid answer. It only shortens degenerate loops. The retry rule (3 attempts) is unchanged, so failure counting is unchanged.

**Timing statistic.** Step 4's A-full mean time to first token (396 ms) was inflated by outliers. Its median is 189 ms at every order position, consistent with step 5 (185 ms). Timing endpoints are reported as medians with IQR, plus means with bootstrap CIs.

**Deviation 6 (2026-09-23, before any data from it): tier-balanced adapter training.** Step 2 found that Arm B's errors concentrate in the "priority" tier, and that the 64 true-class-balanced training examples contain only 5 priority examples. New arm: the headline config with `balance_by="tier"`. It keeps the same total (64: 22 routine / 21 priority / 21 urgent), the same selection rule (first qualifying DS1 beats in chronological order), the same 192 updates and the same learning rate. Seeds 101, 202, 303. Evaluated on E80 with the Arm B protocol. Primary contrast: accuracy against the true-class-balanced seeds on the same events (E2 seeds 101/202: 65/80, 63/80). Secondary: accuracy per reference tier, especially priority (true-class seeds: 3/18 unseeded, 6/18 seed 101, 1/18 seed 202). If the tier-balanced mean exceeds the true-class mean by at least 10 pp, it becomes the training regime for all later Arm B work (task 6 extends it to 5 seeds). Otherwise the true-class regime stays. Both are reported either way.

**Deviation 7 (2026-09-23, during E3, after 8 of 20 windows): A-full at N = 50 is measured on windows 0-7 only.** Each A-full N = 50 prefill takes ~80 s (12.6k prompt tokens), more than half of each window's GPU time. A-full is not part of the gate (B-4 vs A-compact), and its N = 50 cost was unambiguous across the 8 complete windows. All other conditions keep all 20 windows. The gate's inputs are unaffected.

**Deviation 8 (2026-09-23, before any data from it): tier-stratified multi-event pilot (pilot 2).** The first multi-event pilot drew windows at random positions. About 89% of DS2 beats are normal, so short windows were almost all routine (N = 1: 20/20 routine), and plain accuracy mostly measured whether an arm answers "routine". Pilot 2: for N in {5, 10, 20}, 20 DS2 windows per reference tier (routine / priority / urgent; window tier = its most urgent beat's reference tier), non-overlapping candidates, at most 3 windows per record per (N, tier) cell, seed 0. Same scaffold, task, arms (A-compact, B-4) and parsing as pilot 1. Endpoints: recall per tier, balanced accuracy, parse rate, and the false-alarm rate (routine windows answered priority or urgent). Pilot 1's random windows remain the natural-prevalence view. Both views will be required for item 7's multi-event adapter. Not queued automatically; run on the user's go-ahead.

**Deviation 9 (2026-09-24, before any data from it): item 7, a trained multi-event adapter (MEA) on the RR encoder.** Motivation: E3 and E3b show the latent interface's cost advantage exists only when many events share one context. Pilots 1-2 show today's single-event adapter can't use that regime: pilot 2's balanced accuracy was 0.33 / 0.33 / 0.00 at N = 5 / 10 / 20, against A-compact's 0.83 / 0.67 / 0.67.
- **Encoder:** RR-branch CNN-LSTM (`cnn_lstm_rr_seed0.pt`; user decision; single seed, reported as a limitation). Reference tiers are the rule applied to the RR encoder's own events, so pilot 2's windows and numbers are not reused.
- **Stage 0 baseline:** tier-stratified DS2 windows sampled with RR-encoder tiers (N in {1, 5, 10, 20}, 20 per (N, tier) cell, at most 3 per record per cell, seed 0), and A-compact run on them with pilots 1-2's scaffold and task.
- **Adapter:** N context vectors -> N x 4 virtual tokens (per-event Linear(32, 4x2560) + learned event-position embeddings for up to 20 slots + per-token L2 normalisation times one learnable scale, initialised to the median vocabulary embedding norm).
- **Task:** the most urgent tier among N consecutive beats. Deterministic window target in the ReasoningOutput schema, tier tokens weighted x4.
- **Training data:** DS1 windows, stratified by tier over N in {1, 5, 10, 20}, 100 per tier (300 total); validation = 60 windows from held-out DS1 records {109, 205, 223}.
- **Training:** AdamW lr 1e-3, gradient accumulation 8, clip 1.0. Every 20 updates, greedy validation generation; keep the checkpoint with the best validation balanced accuracy; patience 3; at most 3 epochs. Resumable.
- **Staging:** stage 0 baseline -> smoke test (peak memory < 11.5 GB at N = 20; token norms at scale) -> seed 101 -> **Gate A** (validation balanced accuracy >= 0.60 at N = 10 and parse rate >= 95%; otherwise stop and report) -> seeds 202, 303 -> DS2 evaluation.
- **Primary endpoint:** the mean over 3 seeds of MEA balanced accuracy on stage 0's DS2 windows is non-inferior to A-compact at each N in {5, 10, 20}, margin -0.05, with a 95% bootstrap CI over windows.
- **Secondary endpoints:** N = 1 cells; parse rate >= 95%; false-alarm rate on natural-prevalence windows <= A-compact's; per-slot label decoding (item 6 method); E3-style prefill and memory at N = 10 / 20.
- Every outcome is reported.
- **Implementation corrections (2026-09-24, during seed 101; the design above is unchanged).** (1) Memory: the first smoke test peaked at 15.3 GB at N = 20. Logits are now computed only for the target's last T+1 positions, and the frozen LM uses gradient checkpointing. The loss is identical (3.8367 both ways on the smoke window), and the peak is now 9.5 GB. (2) Validation generation length: the implementation used `max_new_tokens` = 64, but window targets are up to 66 tokens (priority / urgent 61-66, routine 44-45). Correct priority and urgent answers were truncated and scored as unparsed, which biased validation selection against those tiers. Seed 101 was stopped at a resume checkpoint and resumed with `max_new_tokens` = 96. Weights and optimizer state were kept. The three earlier validation scores (updates 20, 40 and 60; balanced accuracy 0.25, 0.37, 0.60) are reported but flagged as truncated, and best-checkpoint selection and patience were reset so that only evaluations at 96 tokens count. The DS2 evaluation uses the pilots' generation cap (256).
