# Paper outline (draft 0, 2026-10-02)

**Working title:** *Latent communication from a non-transformer perception agent to a frozen LLM: efficiency, accuracy and information recoverability*

**Target:** TMLR (primary) or Engineering Applications of Artificial Intelligence. ECG triage is the case study, not the topic.

**Scope:** RQ1–RQ3, unchanged from the dissertation, extended from one event per call to many:
- **RQ1:** Does a learned adapter reduce the token count, latency and memory of the perception-to-LLM handoff compared with text interfaces?
- **RQ2:** Does it preserve downstream task accuracy?
- **RQ3:** How much information is recoverable from the virtual tokens, and how can that cost be measured rather than asserted?

## Abstract (skeleton; [..] = pending results)
- **Problem:** latent communication between agents has been restricted to transformer pairs.
- **Approach:** a learned adapter maps a non-transformer perception agent's per-event vectors (CNN-LSTM; replicated on a convolutional ResNet) into a frozen Gemma 4 E4B's embedding space at 4 tokens per event.
- **Evidence base:** pre-registered evaluation (24 deviations), 3 seeds, patient-level confidence intervals, an external confirmatory test on INCART.
- **RQ1:** prefill −19 to −78% and decision time −4 to −36% against full-record JSON at 10–50 events; [batching and prefix caching]; parity with sender-side filtered text.
- **RQ2:** better than full-record and calibrated JSON at N = 10–20; not preserved for a single event or against filtered text.
- **RQ3:** per-event label, tier, heart rate, RR interval and run length are recoverable from each slot; [compression curve].
- **Design rule:** filter when the receiver's question is known in advance; use a latent channel when it isn't, or when many events must be communicated at constant cost.

## Contributions
1. **A latent channel from a non-transformer sender into a frozen LLM,** scaled to 50 events per call and replicated on two sender architectures (RQ1–RQ3).
2. **A rigorous efficiency–accuracy evaluation:**
   - compact, full and **filtered** text baselines;
   - calibration;
   - serving with batching and prefix caching;
   - a pre-registered external confirmatory test.
3. **A measurement protocol for information recoverability** (per slot, per field), plus the rate–recoverability curve.
4. **An actionable protocol-selection rule,** and an honest negative result: filtered text beats the latent channel for a known single question.
5. **An open artifact:** adapter, training recipe, constrained decoder, measurement harness, and the analysis plan with its deviations.

## Sections
1. **Introduction:** the agent-communication cost; the non-transformer gap (*Beyond Tokens*); RQ1–RQ3; contributions.
2. **Related work** (see below).
3. **System:**
   - senders: CNN-LSTM-RR, and ResNet1D-RR as the replication;
   - multi-event adapter (4 tokens per event, position embeddings, L2 scale, side inputs);
   - schema-constrained decoding;
   - the task, framed explicitly as a communication-fidelity benchmark.
4. **Evaluation protocol:**
   - pre-registration and deviations;
   - data (MIT-BIH DS1 training / DS2 test; INCART confirmatory);
   - window sampling;
   - arms (A-compact, A-filtered, MEA);
   - statistics (patient bootstrap, non-inferiority margin, seeds).
5. **Results, RQ1 (efficiency):**
   - tokens, prefill, decision time, memory, energy against N (Table 3, Figure 3);
   - serving: batching and prefix caching [Deviation 23];
   - filtered-text cost growth with abnormal burden.
6. **Results, RQ2 (accuracy):**
   - against default, calibrated and filtered text (Tables 4–5, Figures 4–5);
   - false alarms (Figure 8);
   - mechanisms: lost in the middle, the numeric threshold;
   - the r3 → r4 iteration;
   - the second sender [Deviation 22];
   - INCART confirmatory [Deviation 21].
7. **Results, RQ3 (recoverability):**
   - per-slot decoding (Figures 9–10);
   - side inputs;
   - run ≥ 3;
   - encoder ceiling and 10-seed robustness;
   - the gap between decodable and used (0.92–0.95 against 0.83);
   - compression curve [Deviation 24].
8. **Discussion:**
   - the protocol-selection rule;
   - when latent beats text (unknown questions, many events);
   - limits of filtering;
   - engineering implications.
9. **Limitations:**
   - one receiver LLM (Qwen as future work);
   - one domain;
   - test-set reuse (answered by INCART);
   - the F class;
   - perception specificity;
   - a zero-shot text baseline under a frozen-model constraint.
10. **Conclusion.**

## Figures and tables → source files
| Item | Source |
|---|---|
| Pipeline | reports/figures/fig1_pipeline.png |
| Efficiency vs N | results/p1_item7_ttd.json, p1_item7_filtered_timing.json, E3/E3b |
| Serving (batch, cache) | results/p1_serving.json [Deviation 23] |
| Accuracy vs N (r3, r4, text, calibrated, filtered) | results/p1_item7_r4_analysis.json, p1_item7_filtered.json |
| Forest plots | the same files |
| False alarms | results/p1_item7_r4_analysis.json |
| Second sender | results/p1_second_sender_analysis.json [Deviation 22] |
| INCART confirmatory | results/p1_confirmatory_incart.json [Deviation 21] |
| Recoverability per slot | results/p1_item7_slot_decoder.json, p1_item7_r4_analysis.json, p1_item7_runlen.json |
| Compression curve | results/p1_sweep_analysis.json [Deviation 24] |
| RR encoder over 10 seeds | results/p1_rr_seeds.json |
| Deviations table | docs/analysis_plan.md |

## Related work (paragraph plan; keys from the dissertation's references.bib, new ones to verify)
1. **Latent communication between LLM agents:**
   - *Beyond Tokens* survey (liu2026beyondtokens);
   - agent primitives (jin2026agentprimitives);
   - embedding debate (pham2024ciphers);
   - thought communication (zheng2025thought);
   - latent collaboration (zou2025latent);
   - cache-to-cache (fu2025cachetocache);
   - latent cache flow (rossi2026latentcacheflow);
   - bicameral (flamant2026bicameral);
   - fully latent agents (du2026enabling).

   All of these require transformer senders. The heterogeneous exception is Vision Wormhole (liu2026visionwormhole).
2. **Sensor and signal encoders feeding LLMs:**
   - ECG-Chat (zhao2025ecgchat), ELF (han2026encoderfree), SensorLM (zhang2025sensorlm), NetLLM (wu2024netllm), fetal monitoring (wong2025llms);
   - to add: Time-LLM (ICLR 2024), LLaVA-style projectors (NeurIPS 2023).

   These mostly fine-tune the LLM or use one-shot embeddings, and don't measure communication cost or recoverability.
3. **Context and prompt compression,** to add and verify:
   - gist tokens (Mu et al., NeurIPS 2023);
   - in-context autoencoder (Ge et al., ICLR 2024);
   - xRAG (Cheng et al., NeurIPS 2024);
   - LLMLingua (Jiang et al., EMNLP 2023).

   These compress text into embeddings, whereas this work compresses a non-text agent's state.
4. **Long-context and calibration effects behind the text baseline's failures:**
   - lost in the middle (Liu et al., TACL 2024) and found in the middle (Hsieh et al., Findings of ACL 2024);
   - calibrate before use (Zhao et al., ICML 2021).
5. **Pre-registration and evaluation practice in ML:** a brief note justifying the deviation log.

## Writing schedule (docs/run_order.md)
| Dates | Sections |
|---|---|
| Fri 2 – Sat 3 | Outline (this file), related work, introduction |
| Sun 4 – Tue 6 | System and protocol |
| Wed 7 – Sat 10 | Results and discussion |
| Sun 11 | Full draft |
