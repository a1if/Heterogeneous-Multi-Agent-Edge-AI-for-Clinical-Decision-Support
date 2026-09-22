# TASKS

Working checklist for Phase 1 (see `docs/analysis_plan.md`, `docs/publication_plan.md` §9).
Tick items when done; add new items as they are found.

## Phase 1, step 1: seeded headline (running)
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
- [ ] Full training on GPU: queued, starts automatically when step 1's process exits (`logs/p1_rr_encoder.log`)
- [ ] Compare DS2 overall accuracy and per-class recall (S especially) against the reference CNN-LSTM (85.4%, S 8.2%)
- [x] Integrate into `PerceptionAgent`: encoder chosen by checkpoint format; per-record RR history cleared by `reset_state()`; post-RR via `next_rr_interval_ms`, supplied by `replay_selected`; reference encoder bit-identical on 320 golden events; 4 parity tests
- [ ] Probe recoverability at the encoder's context vector (attribution protocol) for the RR encoder. The probe must replay with `replay_selected`: `day7_auditability_probe.py` and similar call `agent.predict(X[idx])` with no RR and out of order, which the RR agent now rejects on purpose
- [ ] Retrain k=4 adapters (5 seeds) on the RR encoder's context vectors, then run the E80 Arm B eval, so the encoder sweep reaches the adapter

## Found along the way
- [x] Pre-check: RR features separate S from N in held-out data (DS2: 85% of S vs 4% of N have pre-RR ratio < 0.85)
- [ ] F-class RR distribution shifts between splits (DS1 post-ratio median 0.76, DS2 1.00); check F results for this
- [ ] `tests/test_day2_baseline_arm.py::test_baseline_arm_produces_structured_output` crashes with a Windows access violation while loading Gemma (pre-existing, unrelated to Phase 1 changes)
- [x] DS1 `rr_interval_ms` has outliers up to 100,022 ms; RR features clip to [200, 3000] ms (`perception/rr_features.py`, tested)
- [ ] Per-beat perception replay is slow (~143 ms/beat); profile `compute_sqi` / `estimate_qrs_duration_ms` before step 10 (E400)
- [ ] Services audit request: no `services/` directory and no issue link in this repo; waiting on the user
