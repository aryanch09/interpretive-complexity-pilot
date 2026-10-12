# S1(a) — Truncation contrast

**Status: EXPLORATORY.** Under `ANALYSIS_PLAN.md` §6 the truncation contrast is explicitly
labelled exploratory/descriptive. It cannot convert a failed confirmatory test into a
supported one, and it is not part of the Holm-corrected secondary family.

**Cost: zero.** It reads `output/chunk_level_results.csv` and makes no model calls, no
network calls, and no writes outside its own output directory.

---

## The question

Standard practice in LLM studies of 10-K text is to truncate each filing to a fixed budget —
the first N words, or whatever fits a context window. Because larger firms file longer
narrative sections, a fixed budget reads a smaller **fraction** of a large firm's filing. How
much the model read therefore becomes a function of firm size: the very variable such studies
regress on.

This project has already measured that confound in its own sample. The retired 2,000-word cap
truncated **59 of 59** filings, and

> Spearman(tier rank [large = 3], fraction of the narrative sections read) = **−0.415,
> p = 0.0011**

(`output/coverage_diagnostic.txt`). That establishes *coverage was confounded with size*. It
does not yet establish that the confound mattered. This experiment asks the next question:
**does truncation change the answer?**

## The design

Using only the window-level output, recompute the tier gradient as it would have been measured
if the model had read only the **first k windows** of each filing's MD&A, for
**k = 1, 2, 4, 8, all**. Same model, same filings, same scored text, same aggregation.
Coverage is the only thing that varies.

Aggregation is the pre-registered chain (`ANALYSIS_PLAN.md` §1), unchanged:

```
window -> filing   unweighted mean over that filing's MD&A windows inside the budget
filing -> firm     unweighted mean over the firm's filings
firm   -> tier     unweighted mean over the tier's firms
```

* Unit of analysis: the **firm** (n = 30). Filing- and window-level numbers are diagnostic.
* **MD&A only.** Risk Factors is never pooled into the primary measure (§2).
* Failed windows are **dropped, never imputed** (§8.1); the count dropped is reported at every
  budget.
* The **numeric** measure is reported first in every table (§3), whatever it shows; the
  **embedding** measure is the primary one.
* Gradient = (small-cap tier mean) − (large-cap tier mean). Monotonicity
  (small > mid > large) is reported separately, since a gradient can be positive without being
  monotone.
* CIs are 95% percentile bootstrap, **resampling firms within tier** (the firm is the unit),
  10,000 draws by default, seed exposed as `--seed` and recorded in every output row, the
  report and the figure caption. Each (measure, variant, budget) cell draws from its own
  stream derived from that one seed, so results do not depend on evaluation order.
* The comparison against the complete-section estimate is a **paired** bootstrap: the same
  resampled firms are read both ways, because the question is about the same firms read twice,
  not two independent samples.

### Two budget rules — both reported, never averaged

| rule | what it does | why it is not enough on its own |
|---|---|---|
| **inclusive** | every filing contributes `min(k, its window count)` windows; a filing with fewer than k windows contributes all of them | this is what a fixed word budget actually does to a short filing — it reads all of it — but the *effective* k then varies by filing, and short filings are disproportionately small-cap |
| **strict** | only filings with at least k windows contribute at all | holds the per-filing window count at exactly k, at the cost of dropping filings; the drops are **not tier-neutral**, so at k = 8 the small-cap tier can fall to a handful of firms |

The two differ, and the difference is itself the finding: it is the same asymmetry — short
sections are small-cap sections — seen from the other side. Both are in the output CSV; the
figure plots both. Firm counts per tier are reported at every budget, and any cell where a
tier falls below five firms is flagged `THIN` in the report and logged as a warning.

### k = 1, explicitly

k = 1 cannot support a within-filing SD, and the pre-registered usability gate
(`MIN_USABLE_CHUNKS = 2`, §8.2) cannot be satisfied inside a 1-window budget. What this code
does, and why:

* The usability gate is evaluated **once, on each filing's complete MD&A**, not inside the
  budget. It is a data-quality statement about how the filing was scored, not about a
  simulated reading budget. A filing that fails it contributes to **no** budget; a filing that
  passes it contributes to every budget. So k = 1 is analysable, and the set of filings is the
  same at every k (under the inclusive rule).
* The number of filings that *would* fail a within-budget reading of the gate is reported at
  every k (`n_filings_failing_within_budget_gate`) so the stricter reading can be checked.
  At k = 1 that is every filing, by construction.
* **Within-filing SD is NaN at k = 1**, never 0: "no variation measured" is not "no
  variation". The `within_filing_sd_defined` column records this.

---

## Honesty constraint — read before quoting any number from here

**This is a SIMULATION of truncation over windows that were each scored in isolation. It is
NOT the same thing as a model reading a truncated document.** A persona scoring window 1 of 25
never saw windows 2–25: there is no long-context degradation, no carry-over from earlier text,
and no difference in the prompt. What varies here is only *which already-scored windows enter
the average*.

The exact replication without that caveat — re-scoring each filing's first 2,000 words of
MD&A as one document under the old cap, ~180 calls — is **S1(b)**, a separate experiment. It
is **not** run here and this code makes no model calls. The caveat appears in the figure
caption, the text report and the module docstring, not only in this README.

A second limitation, stated because the test suite makes it concrete: with any within-filing
measurement noise, a paired bootstrap over 30 firms is tight enough that a *substantively
negligible* change can produce a CI excluding zero. The output therefore reports the size of
the change beside its CI (`delta_vs_all_pct_of_complete`), and neither should be quoted
without the other.

---

## Running it

```bash
python3 experiments/truncation_contrast/run.py                    # reads output/chunk_level_results.csv
python3 experiments/truncation_contrast/run.py --input FIXTURE.csv --out-dir /tmp/tc \
    --n-boot 2000 --seed 20261004
```

Useful flags: `--budgets 1,2,4,8,all`, `--section mda|risk`, `--seed`, `--n-boot`,
`--no-figure`, `--quiet`. If the input file does not exist the script exits with an error
rather than producing anything.

### Outputs (written to `output/truncation_contrast/`)

| file | contents |
|---|---|
| `truncation_contrast_gradients.csv` | one row per (measure, budget rule, k): tier means, gradient, 95% CI, paired difference vs the complete section and its CI and relative size, monotonicity, firm counts per tier, filings with ≥ k windows, filings short of k, windows used and windows dropped as failed, mean fraction read by tier, Spearman(tier rank, fraction read), seed, draws |
| `truncation_contrast_tiers.csv` | one row per (measure, rule, k, tier): firms, filings, windows used, tier mean, tier SD, mean fraction of MD&A read |
| `truncation_contrast_firm_level.csv` | one row per (rule, k, firm): both measures and fraction read — the inputs to every tier number above |
| `truncation_contrast_diagnostics.csv` | per-cell drop accounting |
| `truncation_contrast.txt` | the same numbers as a readable report, with the caveat at the top |
| `truncation_contrast.pdf` | the figure (vector) |

### The figure

Three stacked panels sharing one x-axis (k = 1, 2, 4, 8, all): **(a)** numeric gradient
(secondary measure, reported first per §3), **(b)** embedding gradient (primary), **(c)** mean
fraction of the complete MD&A read, by tier. Gradient panels show both budget rules with 95%
bootstrap CIs, a zero line, and the complete-section estimate marked as a dashed reference
line with its CI as a band. Hue is colourblind-safe (validated: protan ΔE 25.5, tritan 36.9,
normal-vision ΔE 37.4) and every series is *also* distinguished by line style and marker, so
the figure survives greyscale printing. The caption carries the seed, the number of bootstrap
draws, the drop rule and the simulation caveat. The zero line stays in view on both gradient
panels deliberately: cropping to the data would make a small movement in the gradient look
like a large one.

---

## Result

**Not yet run on real data.** `output/chunk_level_results.csv` does not exist: the ~2,001-call
complete-section run has not happened. Every number below is a placeholder and is filled in by
running the command above — none is hand-typed (§8.6).

| | k = 1 | k = 2 | k = 4 | k = 8 | complete |
|---|---|---|---|---|---|
| numeric gradient (small − large) | — | — | — | — | — |
| embedding gradient (small − large) | — | — | — | — | — |
| 95% CI | — | — | — | — | — |
| difference vs complete (paired CI) | — | — | — | — | — |
| filings with ≥ k windows (of 59) | — | — | — | — | 59 |
| mean fraction of MD&A read: large / mid / small | — | — | — | — | 1.00 / 1.00 / 1.00 |

Verification against synthetic data with known ground truth is in
`tests/test_truncation_contrast.py` (fixtures: `tests/fixtures_truncation.py`): the analysis
recovers a planted tier gradient exactly at every budget, recovers a planted
rise-with-window-index as a material change in the gradient at every truncated budget, and
reports **no** material change on a flat fixture under the identical rule.

---

## What it means for the paper

If the gradient moves with k, the paper can say something stronger than "coverage was
confounded with firm size": it can say that **the standard fixed-budget protocol returns a
different answer from reading the complete section, on the same filings and the same model** —
turning a methodological caveat into a measured effect. If the gradient does not move, that is
equally reportable and considerably reassuring: it bounds how much the field's truncation
habit could have distorted results of this kind, and it strengthens the complete-section
estimate rather than weakening it. Either way the figure is the paper's clearest single
argument for reading complete sections in fixed windows, and panel (c) shows the mechanism
directly — at a fixed budget, how much of the filing the model reads is a function of firm
size.
