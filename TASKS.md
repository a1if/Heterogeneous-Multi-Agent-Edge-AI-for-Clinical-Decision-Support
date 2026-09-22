# TASKS

Working checklist for Phase 1 (see `docs/analysis_plan.md`, `docs/publication_plan.md` §9).
Tick items when done; add new items as they are found.

## Run order: least GPU first (set 2026-09-23)
Step 1 was stopped during its CPU-only data replay (no GPU used yet); it resumes from its saved state later.
1. [x] Profile the replay. `predict()` is ~6 ms/beat; the ~70 min came from indexing an `NpzFile` inside the loop, which re-decompresses the whole array on every access. Fixed in `build_real_training_examples`: 62 s, identical 64 examples. The eval replays (`prepare_events`) were never affected
2. [x] RR encoder training: 10 epochs (early stop). DS2 acc 91.1% vs 85.4%; S Se 38.3% (+P 55.0%) vs 8.2% (+P 5.1%); V Se 96.9% vs 77.2%; F Se 0.0% vs 0.3% (`results/p1_rr_encoder_results.json`)
3. [x] Probe on the 32-d context vector, E80, dissertation protocol: reference 69.6% ± 9.9 (reproduces dissertation exactly), RR encoder 84.6% ± 5.5; paired Wilcoxon over 15 matched folds p = 0.0022 (folds overlap across repeats, so p is optimistic). Probe recall N 1.0 / S 0.7 / V 0.9 / F 0.8 vs 0.7 / 0.7 / 0.85 / 0.5 (`results/p1_rr_probe_results.json`). Decision gate for the RR adapters: passed
4. [x] Step 2: contested events (`results/p1_step2_contested_events.json`). The dissertation's hypothesis (errors on low-confidence / near-0.85 events) is not supported once rule path is controlled: within the "priority" path, confidence doesn't separate wrong from right, and near-threshold V/F events are not over-represented (Fisher p = 1.0 / 0.13 / 0.74). Instead, errors concentrate in the "priority" tier (predicted non-N, not urgent: 18 of 80 events), which Arm B mostly calls "routine": wrong on 3 / 12 / 17 of 18 for headline / seed 101 / seed 202, which accounts for most of the 95% vs 81% vs 79% seed spread. The adapter's 64 training examples are balanced on true class, but contain only 5 priority-tier examples (42 routine, 17 urgent), because the reference encoder predicts N for most S/F training beats
5. [ ] Step 9 prep: multi-flag decision target and unsupported-claim scorer, run on the existing Day 6 outputs (CPU)
6. [ ] Step 7: inverse decoder and confidence R² on adapter outputs (adapter forward pass only, seconds of GPU)
7. [ ] Step 8: pre-generation anomaly check on adapter outputs (CPU / seconds of GPU)
8. [ ] GPU-heavy, in rising cost: step 4 baseline family (~1.5 h), step 5 timing (~2 h), step 3 nondeterminism (~3 h), step 6 seeds × k (~4.5 h), step 1 seeds (~4 h, replay now ~1 min), RR-encoder adapters (~6 h), step 10 E400 (~8 h)

## Phase 1, step 1: seeded headline (stopped, resumable)
- [x] Write and commit the analysis plan (`c779b80`)
- [x] Step 1 harness `p1_step1_seeded_headline.py`
- [x] Drop the GPU E60 rerun; E60 is a subset of E80, so compute it offline (Deviation 1, `c6402c5`)
- [x] Cache the DS1 training-example replay (~70 min/seed) on disk (`fccb6ba`)
- [ ] Train seeds 101, 202, 303, 404, 505 (`reasoning/checkpoints/p1_k4_seed*.pt`)
- [ ] Determinism gate 6a (seeds 101/202 vs the E2 checkpoints)
- [ ] E80 evaluation for all seeds; token-count gate 6b
- [ ] E60/E80 offline invariance figures (gate 6c)
- [ ] Add step 1 results to `results_ledger.json`

## Phase 1, new step: RR-branch encoder (encoder sweep, plan §2.6.2)
- [x] Log Deviation 2 in the analysis plan (new, unplanned encoder arm)
- [x] `perception/rr_features.py`: pre-RR, post-RR, local average RR, ratios (per record)
- [x] Unit tests for the RR features (record boundaries, post-RR fallback, causality of the local average)
- [x] `perception/model_rr.py`: CNN-LSTM + RR branch fed into the context LSTM (context vector stays 32-d, same extraction point)
- [x] `train_perception_agent_rr.py`: same split, sampler and early stopping as the reference; seed recorded; writes `results/p1_rr_encoder_results.json`
- [x] CPU smoke test of training (a few batches)
- [x] Full training on GPU (seed 0, 10 epochs)
- [x] Compare DS2 accuracy and per-class Se/+P against the reference (same metric code): see run-order item 2
- [x] Integrate into `PerceptionAgent`: encoder chosen by checkpoint format; per-record RR history cleared by `reset_state()`; post-RR via `next_rr_interval_ms`, supplied by `replay_selected`; reference encoder bit-identical on 320 golden events; 4 parity tests
- [x] Probe recoverability at the encoder's context vector (see run-order item 3). The probe must replay with `replay_selected`: `day7_auditability_probe.py` and similar call `agent.predict(X[idx])` with no RR and out of order, which the RR agent now rejects on purpose
- [ ] Retrain k=4 adapters (5 seeds) on the RR encoder's context vectors, then run the E80 Arm B eval, so the encoder sweep reaches the adapter

## Found along the way
- [ ] New arm to test (log as a Deviation before running): adapter training examples stratified by reference tier (or predicted label) instead of true class. Directly targets the priority-tier failure behind the seed spread. GPU: training ~30 min/seed, plus E80 eval
- [ ] Dissertation §5.3 also says "three full-batch gradient steps" (same erratum as §3.5.1: 192 per-example updates)
- [x] Pre-check: RR features separate S from N in held-out data (DS2: 85% of S vs 4% of N have pre-RR ratio < 0.85)
- [ ] F-class RR distribution shifts between splits (DS1 post-ratio median 0.76, DS2 1.00); the RR encoder's F Se is 0.0%. Check whether this shift explains it
- [ ] RR encoder is a single seed (0); train more seeds before reporting it as a sweep point
- [ ] `tests/test_day2_baseline_arm.py::test_baseline_arm_produces_structured_output` crashes with a Windows access violation while loading Gemma (pre-existing, unrelated to Phase 1 changes)
- [x] DS1 `rr_interval_ms` has outliers up to 100,022 ms; RR features clip to [200, 3000] ms (`perception/rr_features.py`, tested)
- [ ] Services audit request: no `services/` directory and no issue link in this repo; waiting on the user
