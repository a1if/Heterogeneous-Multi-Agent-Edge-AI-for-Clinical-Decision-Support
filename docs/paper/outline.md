# Paper outline (draft 1, 2026-10-04)

**Working title:** *Latent communication from a non-transformer perception agent to a frozen LLM: efficiency, accuracy and information recoverability*

**Target:** *Information Fusion* (Elsevier; the LaTeX draft in this folder, `main.tex`, uses elsarticle with `\journal{Information Fusion}`). ECG triage is the case study, not the topic. The draft (abstract, introduction, related work, system, protocol, results, discussion; 49 references) already exists in `sections/`; this outline tracks what it must cover.

**Scope:** RQ1–RQ3, unchanged from the dissertation, extended from one event per call to many:
- **RQ1:** Does a learned adapter reduce the token count, latency and memory of the perception-to-LLM handoff compared with text interfaces?
- **RQ2:** Does it preserve downstream task accuracy?
- **RQ3:** How much information is recoverable from the virtual tokens, and how can that cost be measured rather than asserted?

## Abstract (skeleton; [..] = pending INCART)
- **Problem:** latent communication between agents has been restricted to transformer pairs; a non-transformer perception agent can only talk to an LLM through text.
- **Approach:** a learned adapter maps a non-transformer perception agent's per-event vectors (CNN-LSTM; replicated on a convolutional ResNet) into a frozen Gemma 4 E4B's embedding space, 4 tokens per event, up to 50 events per call.
- **Evidence base:** pre-registered evaluation (27 deviations, each logged before its data), 3 seeds throughout, patient-level confidence intervals, an external confirmatory test on INCART [result].
- **RQ1:** prefill −34 to −81% and energy per decision −25 to −62% against full-record JSON at 10–50 events; under batching the adapter is the only interface that fits every batch size on a 12 GB GPU, with 10–30% higher throughput than sender-side filtered text.
- **RQ2:** more accurate than full-record and calibrated JSON at N = 10–20 (replicated at N = 10 on the second sender); not better for a single event, and less accurate than filtered text when the receiver's question is known in advance.
- **RQ3:** per-event label, tier, heart rate, RR interval and run length are recoverable from every slot; facts are recoverable only when passed to the adapter, and the receiver uses them (they drive the accuracy gain and the false-alarm cut); compression to 1–2 tokens per event keeps the information recoverable but makes its use unreliable (decodable is not the same as used).
- **Design rule:** filter when the receiver's question is known in advance; use a latent channel when it isn't, when many events must be sent at constant and predictable cost, or when requests are batched.

## Contributions
1. **A latent channel from a non-transformer sender into a frozen LLM,** scaled to 50 events per call and replicated on two sender architectures (RQ1–RQ3).
2. **A rigorous efficiency–accuracy evaluation:**
   - compact, full and **filtered** text baselines, default and calibrated;
   - serving with batching, prefix caching and energy, under a GPU memory cap;
   - a pre-registered external confirmatory test.
3. **A measurement protocol for information recoverability** (per slot, per field) and two findings it enables: (a) the facts that are made recoverable are the facts the receiver uses (ablation); (b) recoverable is not the same as used (compression with three seeds).
4. **An actionable protocol-selection rule,** with honest negative results: filtered text beats the latent channel on accuracy for a known single question; fewer than 4 tokens per event is not reliable.
5. **An open artifact:** adapter, training recipe, constrained decoder, measurement harness, analysis plan with all deviations, results ledger.

## Sections
1. **Introduction:** the agent-communication cost; the non-transformer gap (*Beyond Tokens*); RQ1–RQ3; contributions.
2. **Related work** (see below).
3. **System:**
   - senders: CNN-LSTM-RR (main), ResNet1D-RR (replication);
   - multi-event adapter (k tokens per event, position embeddings, L2 scale, side inputs heart rate / RR / run length);
   - recipe r4 (hard negatives, side inputs, schedule, early stopping on validation balanced accuracy);
   - schema-constrained decoding;
   - the task, framed explicitly as a communication-fidelity benchmark (the receiver reproduces a fixed rule over perception outputs).
4. **Evaluation protocol:**
   - pre-registration and deviations (summary table; full log as supplementary material);
   - data (MIT-BIH DS1 training / DS2 test; INCART confirmatory);
   - window sampling (class-balanced stratified + natural prevalence);
   - arms (A-compact, A-filtered, calibrated variants, MEA);
   - statistics (patient-cluster bootstrap, non-inferiority margin −0.05, three seeds, fixed-sequence hypotheses on INCART).
5. **Results, RQ1 (efficiency):**
   - tokens, prefill, decision time, memory, energy against N (batch 1);
   - serving: batching, prefix caching, throughput, energy; what does not fit in memory;
   - filtered-text cost growth with abnormal burden;
   - second sender.
6. **Results, RQ2 (accuracy):**
   - against default, calibrated and filtered text; false alarms;
   - mechanisms of text failure: numeric threshold, lost in the middle, sparse evidence (second sender at N = 50);
   - the r3 → r4 iteration and the **ablation** (side inputs drive the gain; hard negatives alone change little);
   - the second sender;
   - INCART confirmatory [Deviation 21; H1–H4 as pre-registered].
7. **Results, RQ3 (recoverability):**
   - per-slot decoding of label, tier, heart rate, RR, run ≥ 3;
   - encoder ceiling and 10-seed robustness;
   - facts recoverable only when passed (ablation);
   - **compression:** 1 / 2 / 4 / 8 tokens per event, three seeds for 1, 2, 4: information recoverable at every k; use reliable only at k = 4 (one failed run at k = 1; false alarms ×2.4 at k = 2).
8. **Discussion:**
   - the protocol-selection rule;
   - when latent beats text (unknown questions, many events, batching and predictable memory);
   - decodable vs used: what recoverability metrics can and cannot certify;
   - limits of filtering (depends on the sender's evidence density);
   - engineering implications for agent pipelines.
9. **Limitations:**
   - one receiver LLM (second receiver as future work);
   - one domain; two databases;
   - test-set reuse on DS2 (answered by INCART);
   - the F class;
   - perception specificity;
   - zero-shot text baselines under a frozen-receiver constraint;
   - seed variability (validation 0.72–0.82 at k = 4; one failed run at k = 1);
   - k = 8 a single run; its heart-rate / RR decodability likely limited by the decoder (32 PCA components).
10. **Conclusion.**

## Key findings to state exactly (source: docs/analysis_plan.md, results_ledger.json)
| Finding | Deviation | Ledger |
|---|---|---|
| Adapter vs calibrated compact text +0.10 at N = 10 and 20 (superior) | 18 | p1.table.07 |
| Filtered text more accurate at every N (−0.11 to −0.17) | 20 | p1.table.09 |
| Second sender: superior to calibrated text at N = 10, non-inferior at 20 and 50 | 22 | p1.table.10 |
| Serving: adapter fits every batch; +10–30% throughput, −12–28% energy vs filtered text | 23 / 23a | p1.table.05 |
| Stop-at-tier reproduces every text tier (2,390 / 2,390) | 25 | (analysis_plan) |
| Ablation: side inputs +0.03 to +0.05 accuracy, false alarms 8.6% → 3.6%; hard negatives ≈ +0.01 | 26 | p1.table.11 |
| Compression: k = 1 and k = 2 not non-inferior to k = 4; information recoverable at every k | 24 / 27 | p1.table.12, p1.figure.12 |
| INCART H1–H4 | 21 | [pending] |

## Figures and tables → source files (all in results_ledger.json as p1.table.* / p1.figure.*)
Planned paper figures (5–7), redrawn from the report figures in the venue's style:
| Paper item | From report | Source |
|---|---|---|
| Fig. 1 Pipeline | Figure 1 | reports/figures/fig1_pipeline.png |
| Fig. 2 Efficiency vs N + serving | Figure 3, Table 5 | results/p1_item7_ttd.json, p1_e3*.json, p1_serving.json |
| Fig. 3 Accuracy vs N, both senders, all text arms | Figures 4, 11 | p1_item7_r4_analysis.json, p1_item7_filtered.json, p1_second_sender_analysis.json |
| Fig. 4 Why text fails | Figure 6 | results/p1_item7_eval_ds2v2_r4.json |
| Fig. 5 Recoverability per slot + side inputs | Figures 9, 10 | p1_item7_slot_decoder.json, p1_item7_runlen.json |
| Fig. 6 Compression: every run at each k | Figure 12 | p1_dev27_kseeds.json, p1_sweep_analysis.json |
| Table: ablation | Table 11 | results/p1_dev26_ablation.json |
| Table: INCART confirmatory | (new) | results/p1_confirmatory_incart.json |
| Table: deviations summary | Table 14 | docs/analysis_plan.md |
All numbers in the text come from ledger placeholders rendered by render_ledger.py (never retyped); add scalar keys for INCART after Monday.

## Related work (paragraph plan; keys from the dissertation's references.bib, new ones to verify before citing)
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

   These compress text into embeddings, whereas this work compresses a non-text agent's state; our compression study adds the reliability-of-use dimension they rarely report.
4. **Probing and the decodable-vs-used distinction:** linear/MLP probes and their limits (to add and verify: e.g. Hewitt & Liang, EMNLP 2019 control tasks; Belinkov, CL 2022 probing survey). Supports the RQ3 framing.
5. **Long-context and calibration effects behind the text baseline's failures:**
   - lost in the middle (Liu et al., TACL 2024) and found in the middle (Hsieh et al., Findings of ACL 2024);
   - calibrate before use (Zhao et al., ICML 2021).
6. **Pre-registration and evaluation practice in ML:** a brief note justifying the deviation log.

## Writing schedule (docs/run_order.md)
| Dates | Sections |
|---|---|
| Sun 4 – Mon 5 | Related work (verify references first) |
| Mon 5 | INCART freeze and run |
| Mon 5 – Tue 6 | Introduction |
| Tue 6 – Wed 7 | System and protocol |
| Thu 8 – Sat 10 | Results and discussion (INCART section after its result) |
| Sun 11 | Full draft, abstract |
| Mon 12 – Wed 14 | Supervisor review (with the updated technical report) |
| Thu 15 – Sat 17 | Revise; figures in venue style; code release |
| Sun 18 | Final checks (ledger render, references, anonymisation) |
| Mon 19 | Submission-ready |
