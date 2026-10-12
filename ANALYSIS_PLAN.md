# Pre-registered analysis plan — Round 8 complete-section run

**Status: binding.** Committed to git before any filing is scored under cache version
`v5-c1200-penaigptoss20b`. Any departure from this document is recorded in `DEVIATIONS.md`
with its reason and reported in the paper. No rule in this file may be changed after
results are seen.

**Project:** interpretive complexity in SEC 10-K/10-Q narrative sections, measured as
disagreement between three analytical persona prompts under a single fixed model. One
model, three lenses, no interaction between calls. This is a prompt ensemble, not a
multi-agent system.

**Written:** 4 October 2026, before scoring.
**Commit hash of this plan:** recorded in the paper and in `DEVIATIONS.md` (the hash of
the commit that adds this file).

---

## 0. What changed before scoring, and why it is not a deviation

Two changes landed between the advisor's Round 8 reply and this plan. Both are declared
here, before any scoring, so neither is a post-hoc choice.

1. **Extraction fix.** `_html_to_text`'s sparse-table pruner deleted any table under 40
   words. Filings that render each Item heading as its own one-cell table therefore lost
   their section headers entirely. XOM's 10-K lost `ITEM 1A. RISK FACTORS` and returned 0
   Risk Factors words — the bug the advisor flagged. The same defect silently truncated
   three large-cap MD&As (PG 10-Q, AMZN 10-K, AMZN 10-Q) and one large-cap Risk Factors
   section (WMT 10-K). Fixed by unwrapping, rather than deleting, a sub-40-word table whose
   entire text is an Item heading, plus a one-character widening of the cross-reference
   rejector so a quoted `see "Item 1A. Risk Factors"` is not mistaken for a real header.
   The forbidden loosening of `_ITEM_PATTERN`'s line anchor was not used.
   **This matters for the paper's own claim:** before the fix, "complete sections" was
   false, and it was false disproportionately for large-caps — the same direction as the
   coverage confound the paper reports removing.

2. **Per-window reasoning text is now persisted** to `chunk_level_results.csv` and
   `chunk_cache.json`. Previously the three personas' reasoning was embedded and then
   discarded, so the embedding measure could not be recomputed with a different encoder,
   and no noise floor could be computed for it, without re-running every model call.

3. **Score parsing corrected.** This one does change the measure, so it is declared
   explicitly. The personas are asked for an integer on 0–100, which the parser divides by
   10. The retired parser divided only when the raw value exceeded 10, treating anything in
   [0, 10] as a legacy 0–10 answer and recording it **undivided**. The persona prompt's own
   calibration guide defines "0–14: overwhelmingly negative", so low raw scores are expected
   rather than anomalous — and the smoke test caught exactly that: a sentiment persona wrote
   "this consistent negative framing places the communication firmly in the overwhelmingly
   negative band" and scored 10, which the old parser would have recorded as **10.0, the
   maximum positive value**, silently and with no parse-error flag.
   Because numeric disagreement is the SD of the three persona scores, a single inverted
   score inflates that window's disagreement by roughly the full scale width. The error
   would also not have been randomly distributed: it fires on windows a persona reads as
   strongly negative, which means disproportionately on Risk Factors windows. The parser now
   divides unconditionally, as the 0–100 contract requires.
   This is a plausible partial explanation for the flat, non-monotone numeric measure
   reported in Round 6, and it is a further reason not to rest the paper on that measure.
   It is declared here, before scoring, and the full run is scored entirely by the corrected
   parser: the 69-call smoke test scored under the old parser was discarded rather than
   reused, so no row in the analysis dataset comes from a different code version.

Changes 1 and 2 touch neither the windowing algorithm, the persona prompts, the model, nor
any measure definition. Change 3 corrects a defect in how a persona's answer is read; it
does not alter what the personas are asked or how disagreement is computed from their
answers.

---

## 1. Sample and unit of analysis

- **Firms:** 30 (`FULL_PILOT_FIRMS`), 10 per size tier. Tier is the independent variable;
  point-in-time CRSP/Compustat values are not yet available and are not used.
- **Filings:** 59, frozen in `data/sample_manifest.csv` — the most recent 10-K and 10-Q
  per firm filed after 2024-07-01, pinned by (ticker, form, date) so the scored sample
  cannot drift if a firm files again mid-project. `data/edgar.py` reads the manifest in
  place of querying EDGAR for "most recent", so the whole run reproduces offline from
  cached raw HTML. HAYN contributes one filing —
  (10-Q only; delisted/acquired August 2024, CIK 858655, no 10-K will ever exist).
- **Scoring window:** 1,200 words (`CHUNK_WORDS`), non-overlapping, sections windowed
  independently. A remainder under 400 words (`MIN_TAIL_WORDS`) merges into the preceding
  window; a section under 400 words yields no windows.
- **Aggregation, fixed in advance:**
  window → filing (unweighted mean over that filing's windows **of the primary section**)
  → firm (unweighted mean over that firm's filings) → tier.
- **Unit of analysis for every confirmatory test: the firm (n = 30).** Filing-level and
  window-level analyses are secondary or diagnostic. This avoids double-counting firms that
  contribute two filings.

### HAYN rule (declared before scoring)

HAYN is **included**. Its single 10-Q yields a valid MD&A of 6 windows, so under the Round 8
definition of `extraction_short` it is not a failed extraction, and the Round 6 rule that
excluded it (single-filing detection under a fixed word cap) no longer applies. Because it
contributes one filing where every other firm contributes two, **the primary test is also
reported with HAYN excluded, as a pre-specified sensitivity.** If the two disagree, both are
reported and the discrepancy is discussed; neither is presented alone.

---

## 2. Missing-section rule (the advisor's Issue 2, decided before scoring)

**Primary analysis is MD&A only, all 59 filings.** This is the advisor's option (a).

- Every one of the 59 filings yields at least 2 MD&A windows (363 MD&A windows in total,
  tier medians 6.5 / 6.5 / 4.0 for large / mid / small), so the primary sample loses
  nothing and no filing is dropped by the `MIN_USABLE_CHUNKS = 2` gate on MD&A.
- **Risk Factors is a separate, secondary analysis** on the filings that have it, reported
  with its own n. It is **never pooled** with MD&A in the primary measure.
- After the §0 fix, **all ten** filings that yield no Risk windows are 10-Qs whose Item 1A is
  a short "no material change" cross-reference — the one genuine extraction failure (XOM's
  10-K, which the advisor flagged) is resolved, and XOM now contributes 5,098 Risk words.
  These ten are **legitimately absent, not extraction failures.** Each was verified individually by locating and reading the
  actual Item 1A text in the converted filing. The quoted sentence for each is recorded in
  the extraction report. These are 10-Qs incorporating the 10-K's risk factors by reference.
- Because MD&A and Risk Factors may carry different disagreement levels — which this design
  is the first to be able to measure — pooling them would make a filing's headline number
  depend on its own section composition. That is the inhomogeneity this plan exists to avoid.

All 638+ windows across both sections are still **scored**; the rule governs which windows
the confirmatory test **reads**.

---

## 3. Measures

| | Definition |
|---|---|
| **Embedding disagreement** (primary) | Mean pairwise cosine distance (1 − cos) between the three personas' reasoning embeddings. `all-MiniLM-L6-v2`, 384-d, mean-pooled, L2-normalised. Computed per window, then aggregated per §1. |
| **Numeric disagreement** (secondary) | Population SD (ddof=0) of the three personas' 0–100 scores, rescaled to 0–10. Computed per window, then aggregated per §1. |
| **Within-filing SD** (secondary) | SD (ddof=1) of window-level disagreement within a filing. |

**Reporting order is fixed: the numeric measure is reported first in every table and every
email, whatever it shows.** The numeric measure was flat and non-monotone at n=30 in Round 6
(Mid < Small < Large). That finding is not to be omitted or de-emphasised.

### Numeric retirement rule (the advisor's Issue 3)

If **Spearman(filing-level numeric disagreement, complete MD&A section word count) ≥ +0.30
with p < 0.05 at n = 59**, the numeric measure is **retired as a primary measure**, and the
paper states plainly that it is partly a length measure. Combined with its flat gradient,
that is sufficient grounds for retirement. The trigger threshold and the n are fixed here;
the Round 7 continuity figure (ρ = +0.524, p = 0.045, n = 15, all large-cap) is not evidence
and does not pre-commit the outcome either way.

---

## 4. Primary hypothesis and confirmatory test

**H1:** interpretive complexity, measured as inter-persona embedding disagreement, is higher
for smaller firms. Ordered alternative **Small > Mid > Large** on firm-level means.

**C1 — the single confirmatory test.** Jonckheere–Terpstra ordered-trend test on firm-level
primary-measure means across the three tiers, one-sided in the direction Small > Mid > Large,
α = 0.05, n = 30 firms (10 per tier). `scipy` has no Jonckheere–Terpstra implementation, so
it is implemented directly and unit-tested against a worked example with a known statistic
before use.

**C1-perm — robustness check on C1, not a second hypothesis.** The same ordered statistic
with its null built by permuting tier labels across firms, 10,000 draws, one-sided. Reported
alongside C1. Because it tests the identical hypothesis by another route, it carries no
multiple-testing penalty; if C1 and C1-perm disagree, both are reported and the permutation
p-value is preferred, since it assumes less.

The confirmatory family contains **one** test. No correction applies to a family of one.

---

## 5. The window-count battery (the advisor's Issue 1) — pass criteria fixed in advance

The advisor's concern: window count scales with section length, length scales with tier, and
an unweighted mean over 25 windows is mechanically less variable than one over 2, so the
predicted gradient could arrive for a purely statistical reason.

**Being precise about the mechanism, because it determines which tests are decisive.**
Averaging over more windows reduces the *variance* of a filing's mean but not its *expected
value*, so pure averaging adds noise to small-cap estimates rather than shifting tier means.
Two channels can actually bias the means, and both are tested:
- **(a) composition/position** — a 2-window MD&A is roughly half section-opening overview; a
  25-window MD&A is roughly 4%. If opening text reads differently, tier means shift.
- **(b) window size** — the tail merge produces windows of 403–1,598 words, and short
  sections are disproportionately small-cap, so window size itself correlates with tier.
  "Identically sized window" is therefore approximately, not exactly, true, and the deviation
  is correlated with the independent variable.

Every test below runs on cached window-level output with **zero additional model calls.**

| | Test | Pre-declared criterion |
|---|---|---|
| **W1** | Descriptives: windows per filing and per firm by tier; Spearman of window count with tier, with numeric, and with embedding disagreement. Same for mean window size. | Descriptive. No pass/fail. Reported whatever it shows. |
| **W2** | Matched comparison. Bin filings by MD&A window count (2–3, 4–5, 6–8, 9+); tier gradient within each bin; plus a coarsened exact matching estimate. | The primary sign must not be **contradicted** (reversed with a CI excluding 0) in any bin containing all three tiers. Overlap is reported honestly: at pre-registration, bins 4–5 (5 large / 8 mid / 6 small) and 6–8 (7 / 8 / 5) contain all three tiers; bin 2–3 (3 / 0 / 8) has no mid-caps and bin 9+ (5 / 4 / 0) has no small-caps, so those two bins are not identified for a three-tier comparison and are reported as such. |
| **W3** | **Equal-k subsampling — the decisive test.** For every filing draw k windows at random (k = 2, the sample minimum; and k = 4 among filings that have 4), recompute filing means, firm means and the tier gradient. 2,000 iterations. | **The primary sign must hold in ≥ 90% of k = 2 draws.** This equalises window count exactly, so a gradient that survives is not an averaging artifact. |
| **W4** | Position control. (i) first-k windows only; (ii) all windows excluding each section's first window; (iii) position-decile term in W5. | Directional agreement with the primary. Divergence is reported as a composition effect, which is itself a finding. |
| **W5** | Window-level mixed model: `disagreement_w ~ tier + log(window_words) + position_decile + form_type + (1 | firm/filing)`. Report tier coefficients with CIs and the ICCs at filing and firm level. | **The Small-vs-Large tier coefficient CI must exclude 0** in the primary direction. Honest limitation stated up front: `statsmodels` MixedLM supports nested variance components but is not a general crossed-effects engine, so the nesting specification used is stated explicitly and a firm-only random-intercept version is reported beside it. |
| **W6** | Null simulation. Hold each filing's real window count fixed; draw window disagreements from the pooled distribution with no tier effect; recompute the gradient 10,000 times. | Quantifies what averaging alone can produce. The observed gradient is reported against this null distribution. No pass/fail; this is the figure that makes the advisor's concern quantitative. |
| **W7** | Window-size check. Primary recomputed (i) on exact-1,200-word windows only, tails excluded, and (ii) on all windows with `window_words` as a covariate. | Directional agreement with the primary. |

**Overall gate.** The tier result is reported as **supported** only if it passes **W3 and W5**
and is not contradicted in an overlapping **W2** bin. If it fails any of these, the paper
reports it as **not robust to window count** — which is itself a result, and is reported with
the same prominence as a positive finding would get.

---

## 6. Secondary analyses (Holm-corrected family of three)

Two-sided, α = 0.05, Holm correction **within this family only**:

- **S1** Tier trend in the **numeric** measure (firm-level).
- **S2** Tier trend in **Risk Factors** embedding disagreement, on the filings that have Risk
  windows, with its own n stated.
- **S3** Tier trend in **within-filing SD** of embedding disagreement.

Every other analysis in the paper — truncation contrast, noise floor, persona-free baseline,
reliability, encoder robustness, placebo, discriminant validity, familiarity/anonymisation,
cross-model, at-scale extraction audit, case studies, any market link — is **exploratory or
descriptive and is labelled as such.** None of it can convert a failed confirmatory test into
a supported one.

---

## 7. What counts as a null (stated before seeing data)

The result is reported as a **null** if any of the following holds:

1. C1 is not significant at one-sided α = 0.05 (and C1-perm agrees); or
2. C1 is significant but the window-count gate in §5 fails; or
3. The firm-level tier means are not monotone in the predicted order.

**A null is reported as the headline finding, in the same place and with the same prominence
a positive result would get.** The paper's spine does not depend on H1: the coverage confound,
the truncation contrast, and the construct-validity battery stand whichever way the gradient
goes. The advisor's position is on the record — "a clean null on a well-identified measure is
a real contribution at a workshop."

A null is **not** grounds for switching the primary measure, the primary section, the
aggregation level, the tier definition, or the test.

---

## 8. Data-quality rules (pre-existing, restated as binding)

1. **Partial-failure rule.** Any window where any of the three personas failed is dropped —
   never imputed. A 2-persona distance is not comparable to a 3-persona mean pairwise
   distance, and failed calls store error strings that would corrupt the embedding.
2. A filing is dropped unless at least 2 windows survive **and** at least half its windows
   survive.
3. **No results are reported from a run with failed rows.** `results.csv` must show zero
   `any_failed = True` before anything goes to the advisor or into the paper.
4. **Served-backend audit must pass.** All calls must come from the single pinned bf16
   backend; the run fails otherwise. Unpinned routing across quantizations would make part of
   the measured disagreement a quantization artifact and the design claim false.
5. **413 means shrink the window, never retry.** The request size does not change on retry.
6. Every number in the paper and in any email is generated by a script reading `output/`.
   No hand-typed figures. Every figure in an email is verified against its attachment before
   it is sent.

---

## 9. Call budget and provenance of the counts

At pre-registration, re-derived by running the project's own `extract_sections_dict` and
`chunk_section` over the frozen sample **after** the §0 extraction fix: **667 windows →
2,001 persona calls** for the 59 filings. 363 of those windows (1,089 calls) are
MD&A — the primary section. Before the extraction fix the same computation gave 638
windows / 1,914 calls; the fix lengthens sections in 10 filings.

Generated by `analysis/extraction_report.py` into `output/extraction_report.{txt,csv}`,
which also classifies every filing yielding no Risk Factors windows and runs the anomaly
gate (no MD&A above 40,000 words, none below 1,000, no extraction failures).

Three other figures are in circulation and all are wrong: 1,869 (what the advisor was sent;
it comes from a `round(words / 1200)` estimator that implements neither the sub-400-word drop
nor the tail merge, so it is not the real chunker), 2,013, and 671 windows. The discrepancy
and its cause are disclosed in the Round 9 email.

**667 / 2,001 is the figure to use** until the run completes, at which point it is confirmed
against `len(chunk_level_results.csv)` and that single confirmed number is used everywhere.

---

## 10. Deviations

Any departure from this plan — including one that looks purely technical — is appended to
`DEVIATIONS.md` with the date, what changed, why, and whether results had been seen at the
time. The paper reports the deviation list. An empty list is itself a claim and is only
published if true.
