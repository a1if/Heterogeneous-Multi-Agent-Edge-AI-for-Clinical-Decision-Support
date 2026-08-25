<!--
Locked per 7day_dissertation_sprint_plan.md §3 (E0), Block 3.
This text is frozen (§2.3 chapter-freeze discipline) — any change goes through
revisions_queue.md, not an in-place edit. Numbers below are ledger placeholders
(double-curly-brace syntax), substituted by render_ledger.py before submission.
-->

Because Arm A attained {{armA.accuracy.n80}} accuracy on these 80 events, the discordant cell (Arm A incorrect / Arm B correct) is structurally empty. McNemar's two-sided exact test nonetheless allocates probability mass to that impossible direction; the one-sided exact test (p={{e0.mcnemar.p_onesided}}) is the appropriate directional instrument and does not reach α=0.05 at this sample size. We accordingly report both arms' accuracies with exact confidence intervals (Arm A: one-sided 95% lower bound {{e0.armA.accuracy.ci_exact95.lower}}; Arm B: Wilson 95% CI [{{e0.armB.accuracy.ci_wilson95.lower}}, {{e0.armB.accuracy.ci_wilson95.upper}}]) and treat the accuracy difference as unresolved rather than as evidence either of degradation or of preservation.
