"""
Synthetic `output/chunk_level_results.csv` generator for the S1(a) truncation
contrast, with controllable ground truth.

(Named `fixtures_truncation` rather than `fixtures` because several analyses in
this project are being built in parallel and each ships its own generator; this
one is written for the truncation experiment's needs — a position effect along
`chunk_index` — and is imported only by `tests/test_truncation_contrast.py`.)

Why this exists
---------------
The complete-section run has not happened yet: `output/chunk_level_results.csv`
does not exist.  Analysis code written against a file that does not exist can only
be checked for "does it crash".  That is not a test.

So this module emits a window-level frame with the *exact* real schema and a
*known* data-generating process, and — independently of the analysis code — computes
what the correct answer is.  The tests then assert that the analysis recovers the
planted effect, and that it reports no effect when none was planted.

Data-generating process (per window)
------------------------------------
    disagreement = base[tier] + slope * chunk_index + Normal(0, noise_sd)

`slope` is applied to the *absolute* window index inside the section, not to a
relative position, because that is the structure the truncation experiment is
built to detect: if later windows disagree more, then reading only the first k
windows understates disagreement, and it understates it *unevenly*, because
filings differ in how many windows they have.  With `slope = 0` the process is
flat in window index and the truncation curve must come out flat.

The ground truth returned by `make_fixture` is computed by `_truth()` using plain
nested loops over the generated records — deliberately a second, independent
implementation of the pre-registered window -> filing -> firm -> tier aggregation,
so that agreement between it and `experiments/truncation_contrast/run.py` is
evidence rather than a tautology.

Everything is seeded; nothing here touches the network, an API, or a model.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# The exact schema of output/chunk_level_results.csv, in order, as written by
# main.py's chunk-level loop.  Column order is part of the contract: the fixture
# is useless if it is not shaped like the real file.
SCHEMA: tuple[str, ...] = (
    "ticker",
    "tier",
    "filing_type",
    "filing_date",
    "section",
    "chunk_index",
    "n_in_section",
    "word_count",
    "numeric_disagreement",
    "embedding_disagreement",
    "value_score",
    "sentiment_score",
    "fundamentals_score",
    "any_failed",
    "any_parse_error",
    "served_by",
    "value_reasoning",
    "sentiment_reasoning",
    "fundamentals_reasoning",
)

TIERS: tuple[str, ...] = ("large", "mid", "small")

# Real-sample shape (ANALYSIS_PLAN.md §1, §9): MD&A windows per filing, tier
# medians 6.5 / 6.5 / 4.0, min 2, max 25.
DEFAULT_WINDOW_MEDIANS: dict[str, float] = {"large": 6.5, "mid": 6.5, "small": 4.0}
MIN_WINDOWS = 2
MAX_WINDOWS = 25

# Window length range produced by the 1,200-word windowing with the MIN_TAIL_WORDS
# merge (ANALYSIS_PLAN.md §5(b)): 403-1,598 words.
FULL_WINDOW_WORDS = 1200
MIN_TAIL_WORDS = 400
MAX_MERGED_WORDS = 1598


def _draw_window_counts(rng: np.random.Generator, tier: str, median: float) -> int:
    """Lognormal window count, clipped to the real sample's observed [2, 25]."""
    n = int(round(float(rng.lognormal(mean=math.log(median), sigma=0.55))))
    return int(min(MAX_WINDOWS, max(MIN_WINDOWS, n)))


def _window_words(rng: np.random.Generator, index: int, n_in_section: int) -> int:
    """
    Word count for one window.

    Every window but the last is a full 1,200-word window.  The last absorbs the
    section's tail (the MIN_TAIL_WORDS merge), so it runs 1,200-1,598 words; a
    one-window section can be anywhere in 403-1,598.
    """
    if n_in_section == 1:
        return int(rng.integers(MIN_TAIL_WORDS + 3, MAX_MERGED_WORDS + 1))
    if index == n_in_section - 1:
        return int(rng.integers(FULL_WINDOW_WORDS, MAX_MERGED_WORDS + 1))
    return FULL_WINDOW_WORDS


def make_fixture(
    seed: int = 20261004,
    n_firms_per_tier: int = 10,
    # ── embedding measure (primary) ───────────────────────────────────────────
    embedding_base: dict[str, float] | None = None,
    embedding_slope: float = 0.0,
    embedding_noise_sd: float = 0.0,
    # ── numeric measure (secondary; flat in the real Round 6 data) ────────────
    numeric_base: dict[str, float] | None = None,
    numeric_slope: float = 0.0,
    numeric_noise_sd: float = 0.0,
    # ── sample shape ──────────────────────────────────────────────────────────
    window_medians: dict[str, float] | None = None,
    window_counts: dict[tuple[str, str], int] | None = None,
    include_risk: bool = True,
    risk_base: dict[str, float] | None = None,
    single_filing_firms: tuple[str, ...] = ("SML09",),
    # ── failure injection ─────────────────────────────────────────────────────
    n_failed_windows: int = 0,
    fail_section: str = "mda",
) -> tuple[pd.DataFrame, dict]:
    """
    Build a synthetic chunk-level frame plus its ground truth.

    Returns `(df, truth)`.

    `df` has exactly `SCHEMA` as its columns.

    `truth` contains, computed independently of the analysis code:
      window_counts     {(ticker, filing_type): n_mda_windows}
      mda_words         {(ticker, filing_type): total MD&A words}
      tier_means        {measure: {k: {tier: expected tier mean}}}
      gradients         {measure: {k: expected small - large}}
      frac_read         {k: {tier: expected mean fraction of MD&A read}}
      failed            set of (ticker, filing_type, section, chunk_index) dropped
      n_failed_mda      how many MD&A windows were failed
      params            the generating parameters

    `k = None` means "all windows" (the complete section).

    Parameters chosen so the planted effect is analytic:
      * `embedding_slope = 0` and `embedding_noise_sd = 0` -> every window of a
        tier carries exactly `embedding_base[tier]`, so every truncation level must
        return exactly the planted gradient.  Any drift is a bug in the analysis.
      * `embedding_slope > 0` -> later windows disagree more, so a budget of k
        windows understates disagreement, and the understatement differs by tier
        because window counts differ by tier.

    `risk_base` defaults to a *reversed* gradient so that a test can detect an
    analysis that leaks Risk Factors windows into the MD&A-only primary
    (ANALYSIS_PLAN.md §2: Risk Factors is never pooled with MD&A).
    """
    embedding_base = dict(embedding_base or {"large": 0.26, "mid": 0.30, "small": 0.33})
    numeric_base = dict(numeric_base or {"large": 0.60, "mid": 0.60, "small": 0.60})
    risk_base = dict(risk_base or {"large": 0.50, "mid": 0.40, "small": 0.20})
    window_medians = dict(window_medians or DEFAULT_WINDOW_MEDIANS)

    rng = np.random.default_rng(seed)

    firms: list[tuple[str, str]] = []
    for tier in TIERS:
        prefix = {"large": "LRG", "mid": "MID", "small": "SML"}[tier]
        for i in range(n_firms_per_tier):
            firms.append((f"{prefix}{i:02d}", tier))

    records: list[dict] = []
    counts: dict[tuple[str, str], int] = {}

    for ticker, tier in firms:
        forms = ("10-Q",) if ticker in single_filing_firms else ("10-K", "10-Q")
        for form in forms:
            key = (ticker, form)
            filing_date = "20250214" if form == "10-K" else "20250509"

            if window_counts is not None and key in window_counts:
                n_mda = int(window_counts[key])
            else:
                n_mda = _draw_window_counts(rng, tier, window_medians[tier])
            counts[key] = n_mda

            sections: list[tuple[str, int]] = [("mda", n_mda)]
            if include_risk:
                # Risk Factors windows exist for most filings; ten 10-Qs in the
                # real sample have none (ANALYSIS_PLAN.md §2).
                n_risk = 0 if (form == "10-Q" and rng.random() < 0.34) else max(
                    1, int(round(n_mda * 1.0))
                )
                if n_risk:
                    sections.append(("risk", n_risk))

            for section, n_in_section in sections:
                base = embedding_base if section == "mda" else risk_base
                slope = embedding_slope if section == "mda" else 0.0
                for idx in range(n_in_section):
                    emb = base[tier] + slope * idx
                    if embedding_noise_sd:
                        emb += float(rng.normal(0.0, embedding_noise_sd))
                    num = numeric_base[tier] + numeric_slope * idx
                    if numeric_noise_sd:
                        num += float(rng.normal(0.0, numeric_noise_sd))

                    # Persona scores are 0-10 after the parser's /10; their
                    # population SD is the numeric measure, so they are generated
                    # to be consistent with `num` rather than independently.
                    centre = 5.0 + 0.5 * (idx % 3)
                    offs = np.array([-math.sqrt(1.5), 0.0, math.sqrt(1.5)]) * num
                    scores = np.round(centre + offs, 2)

                    records.append({
                        "ticker": ticker,
                        "tier": tier,
                        "filing_type": form,
                        "filing_date": filing_date,
                        "section": section,
                        "chunk_index": idx,
                        "n_in_section": n_in_section,
                        "word_count": _window_words(rng, idx, n_in_section),
                        "numeric_disagreement": float(num),
                        "embedding_disagreement": float(emb),
                        "value_score": float(scores[0]),
                        "sentiment_score": float(scores[1]),
                        "fundamentals_score": float(scores[2]),
                        "any_failed": False,
                        "any_parse_error": False,
                        "served_by": "DeepInfra",
                        "value_reasoning": f"[synthetic value reasoning {ticker} {section} {idx}]",
                        "sentiment_reasoning": f"[synthetic sentiment reasoning {ticker} {section} {idx}]",
                        "fundamentals_reasoning": f"[synthetic fundamentals reasoning {ticker} {section} {idx}]",
                    })

    # ── Failure injection ─────────────────────────────────────────────────────
    # A failed window carries any_failed=True and NULL in both disagreement
    # columns.  ANALYSIS_PLAN.md §8.1: it is dropped, never imputed.
    failed: set[tuple[str, str, str, int]] = set()
    if n_failed_windows:
        candidates = [i for i, r in enumerate(records) if r["section"] == fail_section]
        if n_failed_windows > len(candidates):
            raise ValueError(
                f"asked to fail {n_failed_windows} windows but only "
                f"{len(candidates)} exist in section {fail_section!r}"
            )
        chosen = rng.choice(len(candidates), size=n_failed_windows, replace=False)
        for c in chosen:
            r = records[candidates[int(c)]]
            r["any_failed"] = True
            r["numeric_disagreement"] = None
            r["embedding_disagreement"] = None
            r["value_score"] = None
            r["sentiment_score"] = None
            r["fundamentals_score"] = None
            r["served_by"] = ""
            r["value_reasoning"] = "API call failed after all retries."
            r["sentiment_reasoning"] = "API call failed after all retries."
            r["fundamentals_reasoning"] = "API call failed after all retries."
            failed.add((r["ticker"], r["filing_type"], r["section"], r["chunk_index"]))

    df = pd.DataFrame.from_records(records, columns=list(SCHEMA))

    truth = _truth(
        records=records,
        counts=counts,
        firms=firms,
        params={
            "seed": seed,
            "n_firms_per_tier": n_firms_per_tier,
            "embedding_base": embedding_base,
            "embedding_slope": embedding_slope,
            "embedding_noise_sd": embedding_noise_sd,
            "numeric_base": numeric_base,
            "numeric_slope": numeric_slope,
            "numeric_noise_sd": numeric_noise_sd,
            "risk_base": risk_base,
            "include_risk": include_risk,
            "single_filing_firms": single_filing_firms,
            "n_failed_windows": n_failed_windows,
        },
    )
    truth["failed"] = failed
    truth["n_failed_mda"] = sum(1 for f in failed if f[2] == "mda")
    return df, truth


def _truth(records: list[dict], counts: dict, firms: list[tuple[str, str]], params: dict) -> dict:
    """
    Independent reference implementation of the pre-registered aggregation.

    Plain loops on purpose.  This is the oracle the analysis is checked against,
    so it shares no code with it.

      window -> filing : unweighted mean over the filing's MD&A windows inside the
                         budget that did not fail
      filing -> firm   : unweighted mean over the firm's filings
      firm -> tier     : unweighted mean over the tier's firms

    Budgets are applied on `chunk_index` (positional), which is what a fixed word
    budget would have cut; a filing with fewer than k windows contributes all of
    its windows.
    """
    budgets: list[int | None] = [1, 2, 4, 8, None]
    measures = ("numeric_disagreement", "embedding_disagreement")

    mda = [r for r in records if r["section"] == "mda"]
    mda_words: dict[tuple[str, str], int] = {}
    for r in mda:
        mda_words[(r["ticker"], r["filing_type"])] = (
            mda_words.get((r["ticker"], r["filing_type"]), 0) + int(r["word_count"])
        )

    tier_of = dict(firms)
    tier_means: dict[str, dict] = {m: {} for m in measures}
    gradients: dict[str, dict] = {m: {} for m in measures}
    frac_read: dict = {}

    for k in budgets:
        # filing level
        filing_vals: dict[str, dict[tuple[str, str], float]] = {m: {} for m in measures}
        filing_frac: dict[tuple[str, str], float] = {}
        for key in sorted(counts):
            rows = [
                r for r in mda
                if (r["ticker"], r["filing_type"]) == key
                and (k is None or r["chunk_index"] < k)
            ]
            in_budget_words = sum(int(r["word_count"]) for r in rows)
            filing_frac[key] = in_budget_words / mda_words[key]
            good = [r for r in rows if not r["any_failed"]]
            if not good:
                continue
            for m in measures:
                filing_vals[m][key] = sum(float(r[m]) for r in good) / len(good)

        # firm level
        firm_vals: dict[str, dict[str, float]] = {m: {} for m in measures}
        firm_frac: dict[str, float] = {}
        for ticker, _tier in firms:
            keys = [key for key in sorted(counts) if key[0] == ticker]
            fr = [filing_frac[key] for key in keys]
            firm_frac[ticker] = sum(fr) / len(fr)
            for m in measures:
                vals = [filing_vals[m][key] for key in keys if key in filing_vals[m]]
                if vals:
                    firm_vals[m][ticker] = sum(vals) / len(vals)

        # tier level
        for m in measures:
            tm: dict[str, float] = {}
            for tier in TIERS:
                vals = [v for t, v in firm_vals[m].items() if tier_of[t] == tier]
                tm[tier] = sum(vals) / len(vals) if vals else float("nan")
            tier_means[m][k] = tm
            gradients[m][k] = tm["small"] - tm["large"]

        ft: dict[str, float] = {}
        for tier in TIERS:
            vals = [v for t, v in firm_frac.items() if tier_of[t] == tier]
            ft[tier] = sum(vals) / len(vals)
        frac_read[k] = ft

    return {
        "window_counts": counts,
        "mda_words": mda_words,
        "tier_means": tier_means,
        "gradients": gradients,
        "frac_read": frac_read,
        "params": params,
        "budgets": budgets,
    }


def write_fixture(path: str | Path, **kwargs) -> tuple[Path, pd.DataFrame, dict]:
    """Write a fixture to `path` in the real file's CSV form; return it too."""
    df, truth = make_fixture(**kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path, df, truth


if __name__ == "__main__":  # pragma: no cover - manual inspection aid
    import argparse

    ap = argparse.ArgumentParser(description="Emit a synthetic chunk_level_results.csv")
    ap.add_argument("--out", default="/tmp/chunk_level_results_fixture.csv")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--embedding-slope", type=float, default=0.0)
    ap.add_argument("--noise-sd", type=float, default=0.0)
    ap.add_argument("--n-failed", type=int, default=0)
    a = ap.parse_args()

    p, df, truth = write_fixture(
        a.out,
        seed=a.seed,
        embedding_slope=a.embedding_slope,
        embedding_noise_sd=a.noise_sd,
        n_failed_windows=a.n_failed,
    )
    n_mda = int((df["section"] == "mda").sum())
    print(f"wrote {p}  rows={len(df)}  mda={n_mda}  risk={len(df) - n_mda}")
    print("planted embedding gradient (small - large):")
    for k in truth["budgets"]:
        print(f"  k={str(k):>4}  {truth['gradients']['embedding_disagreement'][k]:+.5f}")
