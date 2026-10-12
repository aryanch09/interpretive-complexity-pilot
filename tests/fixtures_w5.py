"""
Synthetic `chunk_level_results.csv` generator with controllable ground truth.

Why this exists
---------------
`output/chunk_level_results.csv` does not exist yet — it is produced by the
~2,001-call complete-section run.  The W5 window-level mixed model (ANALYSIS_PLAN
section 5) therefore has to be validated against data whose truth is known by
construction, so that when the real file lands we already know the estimator
recovers what it claims to recover and does not invent an effect that is not
there.

What is planted
---------------
The generator draws window-level disagreement from exactly the structure the
pre-registered W5 model assumes:

    disagreement_w = intercept
                   + tier_effect[tier]                       <- the quantity W5 tests
                   + beta_log_words * (log(word_count) - log(1200))
                   + beta_position_decile * position_decile
                   + beta_form_10q * 1{form == 10-Q}
                   + u_firm      ~ N(0, sd_firm^2)
                   + v_filing    ~ N(0, sd_filing^2)         (nested in firm)
                   + eps_window  ~ N(0, sd_resid^2)

The DGP is deliberately *linear* in `position_decile` and in `log(word_count)`,
i.e. correctly specified for the model under test.  That is what makes the
no-effect fixture a true null: if the generator introduced curvature the model
cannot represent, a rejection would be a misspecification result rather than a
false positive, and the test would no longer be testing what it claims to.
Misspecification is a separate study and is not simulated here.

Realism of the shape (matched to the pre-registration figures)
--------------------------------------------------------------
* 30 firms, 10 per tier; 59 filings (one firm contributes a single 10-Q, the
  HAYN case declared in ANALYSIS_PLAN section 1).
* MD&A windows per filing drawn from a discretised lognormal with tier medians
  6.5 / 6.5 / 4.0 (large / mid / small) and clipped to [2, 25].  Target totals
  are ~363 MD&A windows and ~667 windows overall; realised totals vary with the
  seed and are reported in the returned truth dict, never assumed.
* `word_count` follows the real chunker's arithmetic: every window but the last
  of a section is exactly `CHUNK_WORDS` = 1200 words, and the last is in
  [403, 1598] — below 1200 when the tail stood alone, above 1200 when a sub-400
  word tail was merged into it (analysis/chunking.chunk_section).  Window size is
  therefore correlated with within-section position, exactly as in the real data;
  that collinearity is a property of the design, not of this fixture.
* Ten 10-Qs yield no Risk Factors windows ("no material change" cross-references,
  ANALYSIS_PLAN section 2).

Internal consistency
--------------------
`numeric_disagreement` is recomputed from the three emitted persona scores as
their population SD (ddof=0), after those scores are rounded to the one decimal
place the real parser produces.  So the CSV is self-consistent: a referee (or a
test) can recompute the column from the score columns and get the same number.

The reasoning columns hold short synthetic strings.  They are NOT real text and
must never be fed to an encoder to "recompute" the embedding measure — they
carry no semantic content, and `embedding_disagreement` here is a planted draw,
not a cosine distance.

Nothing in this module makes a network call or a model call.

Run standalone:
    python3 -m tests.fixtures_w5 --out /tmp/chunk_level_results.csv --seed 20261004
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# The exact column order written by main.py's per-window record (one row per
# scored window).  Asserted by the test suite so this fixture cannot silently
# drift from the real schema.
CHUNK_SCHEMA: tuple[str, ...] = (
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

# Chunker constants (config.CHUNK_WORDS / config.MIN_TAIL_WORDS).  Mirrored, not
# imported, so the fixture runs without touching project config or its env/API
# validation.  The schema test cross-checks the implied word_count range.
CHUNK_WORDS = 1200
MIN_TAIL_WORDS = 400

# Observed bounds on windows per section in the pre-registration sample.
MIN_WINDOWS = 2
MAX_WINDOWS = 25

FAILED_SENTINEL = "API call failed after all retries."

# Three deviations with mean 0 and population SD exactly 1, so that
# m + s * OFFSETS has population SD exactly s before rounding.
_OFFSETS = np.array([-math.sqrt(1.5), 0.0, math.sqrt(1.5)])


def _draw_window_counts(
    rng: np.random.Generator,
    n: int,
    median: float,
    log_sd: float,
) -> tuple[np.ndarray, int]:
    """Discretised lognormal window counts, clipped to [MIN_WINDOWS, MAX_WINDOWS].

    Returns the counts and the number of draws the clip actually moved, so the
    bound is reported rather than applied silently.
    """
    raw = rng.lognormal(mean=math.log(median), sigma=log_sd, size=n)
    counts = np.rint(raw).astype(int)
    clipped = np.clip(counts, MIN_WINDOWS, MAX_WINDOWS)
    n_clipped = int((clipped != counts).sum())
    return clipped, n_clipped


def _word_counts_for_section(rng: np.random.Generator, n_windows: int) -> list[int]:
    """Replicate the real chunker's word-count pattern for one section.

    All windows but the last are exactly CHUNK_WORDS.  The last is drawn in
    [MIN_TAIL_WORDS + 3, CHUNK_WORDS + CHUNK_WORDS // 3 - 2] = [403, 1598], the
    range observed in the pre-registration sample: below 1200 when the remainder
    stood alone, above 1200 when a sub-400-word remainder was merged in.
    """
    last = int(rng.integers(MIN_TAIL_WORDS + 3, CHUNK_WORDS + CHUNK_WORDS // 3 - 1))
    if n_windows == 1:
        return [last]
    return [CHUNK_WORDS] * (n_windows - 1) + [last]


def position_decile(chunk_index: int, n_in_section: int) -> int:
    """Pre-registered position term: chunk_index / max(1, n_in_section - 1), binned.

    The fraction lies in [0, 1]; decile 9 absorbs the closed upper endpoint, so
    the 10 bins are [0,0.1), ..., [0.8,0.9), [0.9,1.0].
    """
    frac = chunk_index / max(1, n_in_section - 1)
    return int(min(9, math.floor(frac * 10)))


def generate_chunk_level_fixture(
    *,
    seed: int = 20261004,
    n_firms_per_tier: int = 10,
    # ---- embedding measure (primary) ------------------------------------
    intercept: float = 0.280,
    tier_effect: Mapping[str, float] | None = None,
    beta_log_words: float = 0.0,
    beta_position_decile: float = 0.0,
    beta_form_10q: float = 0.0,
    sd_firm: float = 0.030,
    sd_filing: float = 0.020,
    sd_resid: float = 0.040,
    # ---- numeric measure (secondary) ------------------------------------
    numeric_intercept: float = 0.600,
    numeric_tier_effect: Mapping[str, float] | None = None,
    numeric_sd_firm: float = 0.080,
    numeric_sd_filing: float = 0.060,
    numeric_sd_resid: float = 0.150,
    # ---- shape ----------------------------------------------------------
    mda_median_windows: Mapping[str, float] | None = None,
    risk_median_windows: Mapping[str, float] | None = None,
    window_count_log_sd: float = 0.55,
    n_single_filing_firms: int = 1,
    n_tenq_without_risk: int = 10,
    # ---- data-quality knobs ---------------------------------------------
    failure_rate: float = 0.0,
    parse_error_rate: float = 0.0,
    served_by: str = "DeepInfra",
    out_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Emit a synthetic chunk_level_results.csv with known ground truth.

    Every random draw comes from one explicitly seeded Generator; the same seed
    and arguments always produce byte-identical output.

    Returns `(df, truth)`.  `truth` records the seed, every planted parameter,
    the realised sample shape, and every bound that actually bit (window-count
    clipping, disagreement clipping, score clipping) — so no cap is silent.

    If `out_path` is given, writes the CSV there and the truth dict alongside it
    as `<out_path>.truth.json`.
    """
    rng = np.random.default_rng(seed)

    tier_effect = dict(tier_effect or {"large": 0.0, "mid": 0.0, "small": 0.0})
    numeric_tier_effect = dict(
        numeric_tier_effect or {"large": 0.0, "mid": 0.0, "small": 0.0}
    )
    mda_median_windows = dict(
        mda_median_windows or {"large": 6.5, "mid": 6.5, "small": 4.0}
    )
    risk_median_windows = dict(
        risk_median_windows or {"large": 6.0, "mid": 5.0, "small": 3.5}
    )
    for name, mapping in (("tier_effect", tier_effect),
                          ("numeric_tier_effect", numeric_tier_effect),
                          ("mda_median_windows", mda_median_windows),
                          ("risk_median_windows", risk_median_windows)):
        missing = [t for t in TIERS if t not in mapping]
        if missing:
            raise ValueError(f"{name} is missing tier(s): {missing}")

    # ---- firms and filings ------------------------------------------------
    firms: list[dict] = []
    for tier in TIERS:
        for i in range(n_firms_per_tier):
            firms.append({"ticker": f"{tier[:3].upper()}{i + 1:02d}", "tier": tier})

    # The HAYN analogue: the last small-cap firm(s) contribute a 10-Q only.
    single_filing = {f["ticker"] for f in firms[-n_single_filing_firms:]} if \
        n_single_filing_firms > 0 else set()

    filings: list[dict] = []
    for f in firms:
        forms = ["10-Q"] if f["ticker"] in single_filing else ["10-K", "10-Q"]
        for form in forms:
            # Deterministic, cutoff-respecting dates (all after 2024-07-01).
            if form == "10-K":
                day = int(rng.integers(1, 29))
                date = f"2026{int(rng.integers(1, 3)):02d}{day:02d}"
            else:
                day = int(rng.integers(1, 29))
                date = f"2026{int(rng.integers(3, 6)):02d}{day:02d}"
            filings.append({**f, "filing_type": form, "filing_date": date})

    # Which 10-Qs incorporate risk factors by reference and so yield no windows.
    tenq_idx = [i for i, fl in enumerate(filings) if fl["filing_type"] == "10-Q"]
    n_no_risk = min(n_tenq_without_risk, len(tenq_idx))
    no_risk = set(rng.choice(tenq_idx, size=n_no_risk, replace=False).tolist()) \
        if n_no_risk else set()

    # ---- window counts per (filing, section) ------------------------------
    n_clipped_total = 0
    for tier in TIERS:
        for section, medians in (("mda", mda_median_windows),
                                 ("risk", risk_median_windows)):
            idx = [i for i, fl in enumerate(filings) if fl["tier"] == tier]
            counts, n_clipped = _draw_window_counts(
                rng, len(idx), medians[tier], window_count_log_sd
            )
            n_clipped_total += n_clipped
            for j, i in enumerate(idx):
                filings[i][f"n_{section}"] = (
                    0 if (section == "risk" and i in no_risk) else int(counts[j])
                )

    # ---- random effects ---------------------------------------------------
    u_firm = {f["ticker"]: float(rng.normal(0.0, sd_firm)) for f in firms}
    u_firm_num = {f["ticker"]: float(rng.normal(0.0, numeric_sd_firm)) for f in firms}

    # ---- rows -------------------------------------------------------------
    rows: list[dict] = []
    n_emb_clipped = 0
    n_score_clipped = 0
    for fl in filings:
        v_filing = float(rng.normal(0.0, sd_filing))
        v_filing_num = float(rng.normal(0.0, numeric_sd_filing))
        is_10q = 1.0 if fl["filing_type"] == "10-Q" else 0.0

        for section in ("mda", "risk"):
            n_sec = int(fl.get(f"n_{section}", 0))
            if n_sec == 0:
                continue
            words = _word_counts_for_section(rng, n_sec)

            for k in range(n_sec):
                wc = words[k]
                pdec = position_decile(k, n_sec)

                mu = (
                    intercept
                    + tier_effect[fl["tier"]]
                    + beta_log_words * (math.log(wc) - math.log(CHUNK_WORDS))
                    + beta_position_decile * pdec
                    + beta_form_10q * is_10q
                    + u_firm[fl["ticker"]]
                    + v_filing
                )
                emb = mu + float(rng.normal(0.0, sd_resid))
                # Cosine distance is bounded in [0, 2]; a draw outside it would
                # be impossible in the real data.  Clipping is counted, not hidden.
                if emb < 0.0 or emb > 2.0:
                    n_emb_clipped += 1
                    emb = float(np.clip(emb, 0.0, 2.0))

                num_sd = (
                    numeric_intercept
                    + numeric_tier_effect[fl["tier"]]
                    + u_firm_num[fl["ticker"]]
                    + v_filing_num
                    + float(rng.normal(0.0, numeric_sd_resid))
                )
                num_sd = max(0.0, num_sd)

                # Three 0-10 persona scores whose population SD is num_sd, then
                # rounded to the one decimal the real parser emits (0-100 ints / 10).
                centre = float(rng.uniform(3.0, 7.0))
                raw_scores = centre + num_sd * _OFFSETS
                scores = np.round(raw_scores, 1)
                if scores.min() < 0.0 or scores.max() > 10.0:
                    n_score_clipped += 1
                    scores = np.clip(scores, 0.0, 10.0)
                scores = scores[rng.permutation(3)]
                # Recomputed from the emitted scores, so the CSV is self-consistent.
                numeric = float(np.std(scores, ddof=0))

                failed = bool(rng.random() < failure_rate)
                parse_err = bool(rng.random() < parse_error_rate)

                rows.append({
                    "ticker": fl["ticker"],
                    "tier": fl["tier"],
                    "filing_type": fl["filing_type"],
                    "filing_date": fl["filing_date"],
                    "section": section,
                    "chunk_index": k,
                    "n_in_section": n_sec,
                    "word_count": wc,
                    # The pre-registered partial-failure rule: a failed window
                    # carries NO disagreement value at all.  It is dropped
                    # downstream, never imputed.
                    "numeric_disagreement": None if failed else numeric,
                    "embedding_disagreement": None if failed else float(emb),
                    "value_score": None if failed else float(scores[0]),
                    "sentiment_score": None if failed else float(scores[1]),
                    "fundamentals_score": None if failed else float(scores[2]),
                    "any_failed": failed,
                    "any_parse_error": parse_err,
                    "served_by": served_by,
                    "value_reasoning": (
                        FAILED_SENTINEL if failed else
                        f"synthetic value reasoning {fl['ticker']} {section} {k}"
                    ),
                    "sentiment_reasoning": (
                        FAILED_SENTINEL if failed else
                        f"synthetic sentiment reasoning {fl['ticker']} {section} {k}"
                    ),
                    "fundamentals_reasoning": (
                        FAILED_SENTINEL if failed else
                        f"synthetic fundamentals reasoning {fl['ticker']} {section} {k}"
                    ),
                })

    df = pd.DataFrame(rows, columns=list(CHUNK_SCHEMA))

    mda = df[df.section == "mda"]
    truth = {
        "seed": seed,
        "generator": "tests.fixtures_w5.generate_chunk_level_fixture",
        "planted": {
            "intercept": intercept,
            "tier_effect": tier_effect,
            "tier_reference": "large",
            "small_vs_large": tier_effect["small"] - tier_effect["large"],
            "mid_vs_large": tier_effect["mid"] - tier_effect["large"],
            "beta_log_words": beta_log_words,
            "beta_position_decile": beta_position_decile,
            "beta_form_10q": beta_form_10q,
            "sd_firm": sd_firm,
            "sd_filing": sd_filing,
            "sd_resid": sd_resid,
            "var_firm": sd_firm ** 2,
            "var_filing": sd_filing ** 2,
            "var_resid": sd_resid ** 2,
            "icc_firm": sd_firm ** 2 / (sd_firm ** 2 + sd_filing ** 2 + sd_resid ** 2),
            "icc_filing": (sd_firm ** 2 + sd_filing ** 2)
            / (sd_firm ** 2 + sd_filing ** 2 + sd_resid ** 2),
            "numeric_intercept": numeric_intercept,
            "numeric_tier_effect": numeric_tier_effect,
            "failure_rate": failure_rate,
            "parse_error_rate": parse_error_rate,
        },
        "realised": {
            "n_firms": len(firms),
            "n_filings": len(filings),
            "n_windows": int(len(df)),
            "n_mda_windows": int(len(mda)),
            "n_risk_windows": int(len(df) - len(mda)),
            "n_failed_windows": int(df["any_failed"].sum()),
            "n_filings_without_risk": int(n_no_risk),
            "single_filing_firms": sorted(single_filing),
            "mda_windows_per_filing_median_by_tier": {
                t: float(
                    mda[mda.tier == t].groupby(["ticker", "filing_type", "filing_date"])
                    .size().median()
                ) if (mda.tier == t).any() else None
                for t in TIERS
            },
            "mda_windows_per_filing_min": int(
                mda.groupby(["ticker", "filing_type", "filing_date"]).size().min()
            ) if len(mda) else None,
            "mda_windows_per_filing_max": int(
                mda.groupby(["ticker", "filing_type", "filing_date"]).size().max()
            ) if len(mda) else None,
            "word_count_min": int(df["word_count"].min()) if len(df) else None,
            "word_count_max": int(df["word_count"].max()) if len(df) else None,
        },
        "bounds_applied": {
            "window_counts_clipped_to_2_25": n_clipped_total,
            "embedding_draws_clipped_to_0_2": n_emb_clipped,
            "score_triples_clipped_to_0_10": n_score_clipped,
        },
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        Path(str(out_path) + ".truth.json").write_text(
            json.dumps(truth, indent=2), encoding="utf-8"
        )

    return df, truth


def _cli(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--out", required=True, help="Where to write the synthetic CSV")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--small-effect", type=float, default=0.0,
                    help="Planted small-vs-large embedding tier effect")
    ap.add_argument("--mid-effect", type=float, default=0.0,
                    help="Planted mid-vs-large embedding tier effect")
    ap.add_argument("--failure-rate", type=float, default=0.0)
    args = ap.parse_args(argv)

    _, truth = generate_chunk_level_fixture(
        seed=args.seed,
        tier_effect={"large": 0.0, "mid": args.mid_effect, "small": args.small_effect},
        failure_rate=args.failure_rate,
        out_path=args.out,
    )
    print(json.dumps(truth, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
