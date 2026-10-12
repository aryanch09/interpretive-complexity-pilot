# Discarded: 14 firms scored on the DekaLLM backend (2026-10-05)

387 windows across 14 firms, served by DekaLLM at bf16. Discarded, not used.

**Why.** DekaLLM degraded sharply mid-run: a probe with realistic 1,200-word
payloads returned 3/5 successes at 22-27s latency (mean 24.1s), against 5.2s the
previous day, and 27 of 1,169 calls (2.3%) returned HTTP 200 with an empty
completion — which costs that window a persona. DeepInfra, which had been
rate-limited the day before, probed 5/5 clean at 5.2s.

Finishing the remaining ~840 calls on DekaLLM would have run past midnight and
kept feeding empty completions into the measure. Re-running all 30 firms on
DeepInfra took less time than finishing on DekaLLM.

**Why not just switch for the remainder.** The design claim is that the model is
held fixed and only the analytical lens varies. A dataset served partly by one
backend and partly by another falsifies that, and the served-backend audit raises
on it by design. One homogeneous dataset was worth more than 387 salvaged windows.

Kept for provenance only. Nothing here may be cited, and it must not be merged
with the DeepInfra run.
