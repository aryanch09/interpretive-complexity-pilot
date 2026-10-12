# Deviations from ANALYSIS_PLAN.md

Every departure from the pre-registered plan is recorded here with the date, what changed,
why, and **whether results had been seen at the time**. The paper reports this list. An
empty list is itself a claim and is published only if true.

| Date | Plan section | What changed | Why | Results seen first? |
|---|---|---|---|---|
| 2026-10-04 | §8 rule 4 | The pinned bf16 backend changed from DeepInfra to DekaLLM, and `MAX_RETRIES` 6 → 10. | DeepInfra began rate-limiting the model upstream hard enough to exhaust the retry ceiling (3/3 probe calls failed; DekaLLM 2/3 at 2.2 s). Of the 12 endpoints serving this model exactly two are bf16, so this is the only way to keep a single reference-precision backend. | **No.** Nothing was scored under v5 at the time; the 23 calls made before the switch were discarded and nothing was cached. |
| 2026-10-05 | §8 rule 1 | Windows where a persona's output could not be PARSED are now re-scored once and, if they still fail, dropped — previously they were kept, with `numeric_disagreement` computed from the two personas that did parse and, where REASONING could not be extracted, an imputed `text[:500]` string feeding the embedding measure. | The pre-registered rule drops any window where a persona failed and forbids imputation. The code keyed that rule on an API-level `failed` flag rather than on actually having three valid outputs, so a persona that returned unparseable text counted as a success. 25 of the first 331 scored windows were affected, skewed 14 large / 11 mid / 0 small — a tier-correlated bias in the comparison the paper is about. | **No.** Found by auditing the cache mid-run; no tier gradient, firm mean or test statistic had been computed. |
| 2026-10-05 | §8 rule 4 | Pinned backend returned to DeepInfra, and the 14 firms (387 windows) already scored on DekaLLM were DISCARDED rather than merged. Empty completions are now retried instead of parsed as empty text. | DekaLLM degraded mid-run: 3/5 success at 22-27s latency (mean 24.1s, against 5.2s the previous day) and 2.3% of calls returning HTTP 200 with no completion, which silently costs a window one persona. DeepInfra had recovered from the congestion that caused the original switch (5/5 at 5.2s). Re-running all 30 firms on DeepInfra was FASTER than finishing the remainder on DekaLLM, and keeps exactly one serving backend, which the design claim requires. | **No.** No gradient, firm mean or test statistic had been computed; the discarded data is archived unused in archive/dekallm_discarded/. |
| 2026-10-09 | — (interpretation, not a plan change) | The S1b result is NO LONGER reported as 'truncation changes the answer'. The decomposition shows the contrast is not attributable to coverage. | S1b's two arms differed in coverage, scoring unit and number of draws simultaneously. Isolating each: aggregation contributes +0.0002, and coverage with the scoring unit and backend held constant contributes **-0.0139** (paired -0.0252 [-0.0534, +0.0027], spans zero) — the opposite sign to the headline. The movement sits with the scoring unit. | **Yes**, the S1b number had been computed. This is a correction made because the contrast was audited after the fact, not a post-hoc exclusion; the original S1b figures are retained and reported alongside. |

The second entry is a genuine correction: the code was not implementing the rule the plan states.
Substituting `text[:500]` for reasoning a persona never produced is imputation, and a two-score
standard deviation is exactly the non-comparability the partial-failure rule exists to prevent.

Arguably this first entry is not a deviation at all: ANALYSIS_PLAN.md §8 rule 4 requires that
all calls come from "the single pinned bf16 backend", and they still do. Switching *which* bf16
backend is pinned is not the same as mixing quantizations, which is what the pin exists to
prevent. It is recorded here anyway, because a change to what served the model is exactly the
kind of thing a reader is entitled to know without having to ask.

## Not deviations (declared in the plan before scoring)

- The extraction fix and the per-window reasoning persistence, both landed before any
  filing was scored and both declared in `ANALYSIS_PLAN.md` §0.
- The missing-section rule (MD&A-only primary), declared in §2 before scoring, as the
  advisor required.
- Inclusion of HAYN with a pre-specified sensitivity analysis excluding it, declared in §1.
