# Confirmatory evaluation plan: `r4-confirmatory` on INCART

Written 2026-10-01, before any INCART data is downloaded, converted or inspected. Registered as Deviation 21 in `docs/analysis_plan.md`. Anything done differently afterwards is reported as a deviation from this page.

## 1. Locked system (release tag `r4-confirmatory`)
- **Perception:** RR-branch CNN-LSTM, seed 0 (`perception/checkpoints/cnn_lstm_rr_seed0.pt`).
- **Adapter:** recipe r4 checkpoints, seeds 101, 202 and 303. These are the existing outputs of the frozen DS1/r4 protocol (Deviation 18) and are **not retrained**.
  - Each event has a 35-d input: the 32-d vector plus heart rate, RR interval and run length.
  - Each event gets 4 virtual tokens, over at most 50 slots.
  - Training data, hard-negative selection, schedule, early stopping and checkpoint-selection rule are as registered.
- **Text baselines:**
  - **A-compact:** Deviation 12.
  - **A-filtered:** Deviation 20.
  - Both use the same scaffold, instructions and schema-constrained decoder (128 new tokens, field cap 40, greedy).
- **Calibration:** Deviation 17 primary rule. A routine-logit bias is chosen on the 147 DS1 validation windows; the bias values are frozen in the manifest and **not re-tuned on INCART**.
- **Freeze:** a manifest records the SHA-256 of every checkpoint, of the code files that build prompts, decode and analyse, and of the test-window builder, plus the package versions and the git commit. The tag is created only after the Deviation 20 analysis has fixed A-filtered's bias.

## 2. Data
- **Dataset:** St Petersburg INCART 12-lead Arrhythmia Database v1.0.0 (PhysioNet; 75 thirty-minute recordings from 32 patients). None of it has been used for any decision in this project. All 75 recordings form the test set; nothing on INCART is used for training, tuning or prompt selection.
- **Patients:** recordings are grouped by patient, using PhysioNet's record-to-patient information if it exists. If it doesn't, records sharing the same header metadata (age, sex, diagnoses) are grouped. If grouping by patient isn't possible, the bootstrap clusters by recording, and this is stated as a limitation.
- **Conversion** (fixed before inspection; mirrors `data_prep.py`):
  - Use lead II, resampled from 257 to 360 Hz with polyphase resampling (up 360, down 257).
  - Map annotation positions to the 360 Hz grid as round(sample × 360 / 257).
  - Take a 360-sample window centred on each annotated beat and z-score it.
  - Compute the RR interval from the original 257 Hz annotation positions. The first beat of a record gets the split's median RR. Gaps are capped at 11,999 ms.
- **Inclusion:** every beat whose symbol is in `data_prep.AAMI_MAP` (unchanged AAMI EC57 mapping: N, L, R, e, j → N; A, a, J, S → S; V, E → V; F → F; P, /, f, u → Q) and whose window lies inside the recording.
  - **Excluded:** other symbols, edge beats, and flat windows (SD < 1e-8), as in `data_prep.py`.
  - No recording is excluded.
- **Events and reference tier:** a chronological replay through the frozen perception agent produces the events. The reference tier is the existing rule applied to those events. True annotations are reported only for describing the data and for perception accuracy, which is reported but is not an endpoint.
- **Windows:** the `p1_item7_testset_v2.build` sampler, unchanged: seed 0; per tier, 60 windows at N = 1, 5, 10 and 20 and 40 at N = 50; class round-robin with at most 3 per record per cell; plus 100 natural 50-beat windows at N = 5, 10, 20 and 50. Cells that cannot be filled stay short (no top-up), and achieved counts are reported.

## 3. Arms and runs
- Arms: r4 seeds 101, 202 and 303; A-compact; A-filtered. Each is generated once through `p1_item7_eval.py`.
- Calibrated text answers come from tier logits with the frozen biases.
- **Deviation 25 (decided on DS2 on 2026-10-02, before the freeze):** stopping at the tier token reproduced every DS2 text-arm tier (2,390 of 2,390), so the INCART text arms are generated with `--stop-at-tier` and their field-cap rate is not reported. The adapter arms keep full generation.
- Timing (time to first token and time to the tier token) uses batch 1, on the first 7 stratified windows per tier per N (21 per N), with arm order rotated, on the same RTX 5070 used for the earlier timing runs.

## 4. Hypotheses (tested in this order; fixed sequence at α = 0.05, stopping at the first failure)
- **H1 (primary):** r4 (3-seed mean balanced accuracy) is **non-inferior** to calibrated A-compact at **N = 10, 20 and 50**. The margin is −0.05, and all three must pass.
- **H2:** r4 is **superior** to calibrated A-compact at N = 10 and 20 (replicating Deviation 18).
- **H3:** r4 is non-inferior to calibrated A-filtered at N = 10, 20 and 50.
- **H4:** r4's median time to first token is lower than A-compact's at N = 10, 20 and 50 (the 95% CI of the paired median relative difference excludes 0).
- **Reported without a hypothesis test:**
  - all arms at every N, default and calibrated;
  - r4 versus A-filtered on time to first token, time to decision and prompt tokens;
  - false-alarm rate on natural-prevalence routine windows, with patient-cluster CIs;
  - urgent recall by reason;
  - parse and field-cap rates;
  - perception accuracy.

## 5. Statistics and failures
- **Balanced accuracy:** mean recall over the tiers present, on stratified windows, paired across arms.
- **Confidence intervals:** patient-cluster bootstrap with 20,000 resamples (seed 0), percentile 95%.
  - Non-inferiority: the lower bound is above −0.05.
  - Superiority: the lower bound is above 0.
  - Timing uses a window bootstrap of the paired median relative difference.
- **Failed generation:** output from which no valid tier can be parsed under the schema.
  - A failure counts as **incorrect** for accuracy and as an **alarm** for false-alarm rates.
  - A runtime error (for example running out of memory) is retried once on the same window; if it fails again, it is a failed generation.
  - Parse rate is reported per arm.

## 6. Conduct
- Before the run, only these are inspected: conversion checks (beat counts per class and record, window counts) and the window list's hash. No model output is inspected.
- The locked system runs once. No changes after results, and no reruns except technical retries (Section 5).
- Every hypothesis outcome is reported, including failures. If H1 fails, the paper's accuracy claim is restricted to MIT-BIH and stated as not replicated externally.
