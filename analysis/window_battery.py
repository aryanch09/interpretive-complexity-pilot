"""
The window-count battery (ANALYSIS_PLAN.md section 5, tests W1-W4 and W6-W7).

Why this exists
---------------
The advisor's Round 8 Issue 1: window count scales with section length, section
length scales with tier, and a filing-level mean is an unweighted mean over that
filing's windows.  A mean over 25 windows is mechanically less variable than a
mean over 2, so a tier gradient could in principle arrive for a purely
statistical reason rather than because small firms file more ambiguous prose.

ANALYSIS_PLAN.md section 5 is precise about the mechanism, and that precision
decides which tests are decisive:

  * Pure averaging reduces the VARIANCE of a filing mean, not its EXPECTED
    value.  It therefore adds noise to small-cap estimates rather than shifting
    tier means.  W6 quantifies exactly how much apparent gradient that noise can
    manufacture.
  * Two channels can actually bias the means:
      (a) composition / position - a 2-window MD&A is roughly half
          section-opening overview, a 25-window MD&A roughly 4%.  W4.
      (b) window size - the MIN_TAIL_WORDS merge produces windows of 403-1,598
          words, and short sections are disproportionately small-cap, so window
          size is itself correlated with the independent variable.  W7.

W5 (the window-level mixed model) is a separate module and is deliberately not
imported here.

Everything in this file runs on cached window-level output.  No EDGAR requests,
no model calls, no network access of any kind.

Pre-registered aggregation, used identically in every test below:

    window -> filing   unweighted mean over that filing's windows OF THE SECTION
                       BEING ANALYSED
    filing -> firm     unweighted mean over that firm's filings
    firm   -> tier     unweighted mean over the tier's firms

The confirmatory unit of analysis is the FIRM (n = 30).  Windows where any
persona failed (`any_failed = True`, both disagreement columns null) are DROPPED
and never imputed - ANALYSIS_PLAN.md section 8 rule 1.

Run:
    python -m analysis.window_battery \\
        --input output/chunk_level_results.csv \\
        --section mda --measure embedding_disagreement --seed 20261004
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

# Documented abseil mutex deadlock on this machine: TensorFlow 2.21 acquires a
# global mutex on import that never releases under Python 3.13 + PyTorch 2.12.
# Nothing here needs TensorFlow; block it before anything else can pull it in.
sys.modules["tensorflow"] = None  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed constants.  These encode pre-registered choices; none of them is tuned.
# ---------------------------------------------------------------------------

TIER_ORDER: tuple[str, ...] = ("large", "mid", "small")
#: Ascending in the hypothesised direction Small > Mid > Large.  Used by the
#: ordered-trend statistic: group order is large -> mid -> small.
TIER_ASCENDING: tuple[str, ...] = ("large", "mid", "small")
#: Tier rank with large = 3, matching analysis/coverage_diagnostic.py so the
#: sign of every Spearman in this file is comparable with the ones already sent
#: to the advisor.
TIER_RANK: dict[str, int] = {"large": 3, "mid": 2, "small": 1}

#: Pre-registered W2 bins on MD&A window count.
W2_BIN_EDGES: tuple[tuple[str, int, float], ...] = (
    ("2-3", 2, 3),
    ("4-5", 4, 5),
    ("6-8", 6, 8),
    ("9+", 9, float("inf")),
)

W3_ITERATIONS = 2_000          # ANALYSIS_PLAN.md section 5, W3
W3_K_VALUES: tuple[int, ...] = (2, 4)
W3_SIGN_SHARE_CRITERION = 0.90  # pre-declared: sign must hold in >= 90% of k=2
W6_ITERATIONS = 10_000         # ANALYSIS_PLAN.md section 5, W6
PERMUTATIONS = 10_000          # ANALYSIS_PLAN.md section 4, C1-perm
EXACT_WINDOW_WORDS = 1_200     # config.CHUNK_WORDS
MIN_USABLE_CHUNKS = 2          # analysis/disagreement.MIN_USABLE_CHUNKS
W4_FIRST_K = 2                 # "first-k windows only"; k = the sample minimum

#: Set False to force the label-permutation null down its generic, statistic-agnostic
#: loop.  The two paths are asserted equal in tests/test_window_battery.py.
_ALLOW_FAST_PERMUTATION = True

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "experiments" / "window_battery"

#: Exact schema of output/chunk_level_results.csv - one row per scored window.
CHUNK_SCHEMA: tuple[str, ...] = (
    "ticker", "tier", "filing_type", "filing_date", "section", "chunk_index",
    "n_in_section", "word_count", "numeric_disagreement", "embedding_disagreement",
    "value_score", "sentiment_score", "fundamentals_score", "any_failed",
    "any_parse_error", "served_by", "value_reasoning", "sentiment_reasoning",
    "fundamentals_reasoning",
)


# ===========================================================================
# Sibling-module resolution (analysis/trend_tests.py, written in parallel)
# ===========================================================================
#
# The real module is used when it is importable and its signature accepts this
# module's call; otherwise a thin internal fallback runs, so the battery and its
# tests work standalone.  Which implementation was used is recorded once in
# SIBLING_PROVENANCE and printed in the report, never silently swapped.
#
# One substantive note.  analysis.trend_tests.tier_trend_statistic returns the
# JONCKHEERE-TERPSTRA STATISTIC J - correctly, because C1 and C1-perm must
# permute the identical statistic.  J is non-negative and carries no sign, so it
# cannot be the quantity W2's confidence interval, W3's "share of draws in which
# the primary sign holds" and W6's one-sided percentile are defined on.  Those
# three tests need a SIGNED gradient, and the plan's own tables report the tier
# gradient as firm-level tier means and their range.  This module therefore:
#
#   * uses the sibling's tier_trend_statistic (J) where J is the right object -
#     reported next to the primary, and permuted by C1-perm; and
#   * defines the signed Small-minus-Large difference on firm-level means
#     locally, as `gradient_small_minus_large`, and uses that for W2/W3/W4/W6/W7.
#
# Both numbers appear in the report so neither is hidden behind the other.

try:
    from analysis import trend_tests as _TT
except ImportError:                                            # pragma: no cover
    _TT = None

#: Which implementation each entry point resolved to.  Filled in on first use.
SIBLING_PROVENANCE: dict[str, str] = {}


def _record(name: str, source: str, reason: str = "") -> None:
    """Record, once and loudly, which implementation of `name` is in use."""
    if name in SIBLING_PROVENANCE:
        return
    SIBLING_PROVENANCE[name] = source
    if source.startswith("window_battery.fallback"):
        logger.warning(
            "analysis.trend_tests.%s not usable from here%s - using this module's "
            "internal fallback", name, f" ({reason})" if reason else "")
    else:
        logger.info("using analysis.trend_tests.%s", name)


def _fellback(name: str) -> bool:
    return SIBLING_PROVENANCE.get(name, "").startswith("window_battery.fallback")


# ---------------------------------------------------------------------------
# Internal fallbacks
# ---------------------------------------------------------------------------

def _fallback_aggregate_to_firm(
    chunk_df: pd.DataFrame,
    section: str = "mda",
    measure: str = "embedding_disagreement",
) -> pd.DataFrame:
    """
    Pre-registered three-step aggregation.  One row per firm:
        ticker, tier, value, n_filings, n_windows
    """
    w = surviving_windows(chunk_df, section=section, measure=measure)
    if w.empty:
        return pd.DataFrame(columns=["ticker", "tier", "value", "n_filings", "n_windows"])
    filing = (
        w.groupby(["ticker", "tier", "filing_key"], as_index=False)
         .agg(filing_mean=(measure, "mean"), n_windows=(measure, "size"))
    )
    firm = (
        filing.groupby(["ticker", "tier"], as_index=False)
              .agg(value=("filing_mean", "mean"),
                   n_filings=("filing_mean", "size"),
                   n_windows=("n_windows", "sum"))
    )
    return firm.sort_values(["tier", "ticker"]).reset_index(drop=True)


def _fallback_jonckheere_terpstra(
    groups: Sequence[Sequence[float]],
    alternative: str = "increasing",
) -> dict[str, float]:
    """
    Jonckheere-Terpstra ordered-trend test.  `groups` must already be in the
    hypothesised ASCENDING order (large, mid, small).  scipy has no
    implementation, so this is written out directly and is unit-tested against a
    worked example with a known statistic.

    Ties contribute 0.5 to the count (the standard convention).  Normal
    approximation using the tie-free variance; the sibling module carries the
    tie-corrected variance, which is what the confirmatory C1 uses.
    """
    arrs = [np.asarray(g, dtype=float) for g in groups]
    arrs = [a[~np.isnan(a)] for a in arrs]
    ns = np.array([a.size for a in arrs], dtype=float)
    N = float(ns.sum())
    if len(arrs) < 2 or (ns == 0).any():
        return {"statistic": float("nan"), "z": float("nan"), "p_value": float("nan"),
                "n": N}

    jt = 0.0
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            xi = arrs[i][:, None]
            yj = arrs[j][None, :]
            jt += float((yj > xi).sum() + 0.5 * (yj == xi).sum())

    mean = (N ** 2 - (ns ** 2).sum()) / 4.0
    var = (N ** 2 * (2 * N + 3) - (ns ** 2 * (2 * ns + 3)).sum()) / 72.0
    z = (jt - mean) / np.sqrt(var) if var > 0 else float("nan")
    if alternative == "increasing":
        p = float(stats.norm.sf(z))
    elif alternative == "decreasing":
        p = float(stats.norm.cdf(z))
    else:
        p = float(2 * stats.norm.sf(abs(z)))
    return {"statistic": jt, "expected": float(mean), "variance": float(var),
            "z": float(z), "p_value": p, "n": N,
            "variance_note": "tie-free variance (internal fallback)"}


def _fallback_label_permutation_test(
    values: Sequence[float],
    labels: Sequence[str],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_draws: int = PERMUTATIONS,
    seed: int = 0,
    alternative: str = "greater",
) -> dict[str, Any]:
    """
    Null built by permuting tier labels across firms.  One-sided p-value uses the
    (1 + #{null at least as extreme}) / (1 + n_draws) convention, which is never
    0 and is valid for a finite number of draws.
    """
    v = np.asarray(list(values), dtype=float)
    lab = np.asarray(list(labels), dtype=object)
    rng = np.random.default_rng(seed)
    obs = float(statistic(v, lab))

    fast = _ALLOW_FAST_PERMUTATION and getattr(statistic, "__name__", "") == "_gradient_stat"
    if fast:
        # Permuting labels across firms is identical to permuting the firm values
        # and holding the labels fixed, so for the signed gradient the whole null
        # can be built in one vectorised pass.  Asserted equal to the generic
        # loop in tests/test_window_battery.py.
        small_pos = np.flatnonzero(lab == "small")
        large_pos = np.flatnonzero(lab == "large")
        perm = np.argsort(rng.random((n_draws, v.size)), axis=1)
        permuted = v[perm]
        null = permuted[:, small_pos].mean(axis=1) - permuted[:, large_pos].mean(axis=1)
    else:
        null = np.empty(n_draws, dtype=float)
        for i in range(n_draws):
            null[i] = statistic(v, rng.permutation(lab))

    if alternative == "greater":
        hits = int((null >= obs).sum())
    elif alternative == "less":
        hits = int((null <= obs).sum())
    else:
        hits = int((np.abs(null) >= abs(obs)).sum())
    return {
        "observed": obs,
        "p_value": (1 + hits) / (1 + n_draws),
        "n_draws_used": int(n_draws),
        "n_at_least_as_extreme": hits,
        "seed": int(seed),
        "alternative": alternative,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "null_percentiles": _percentiles(null),
    }


# ---------------------------------------------------------------------------
# Public wrappers: sibling first, fallback second, provenance always recorded
# ---------------------------------------------------------------------------

def _normalise_firm_frame(out: Any) -> pd.DataFrame:
    """Accept the sibling's firm frame and expose its measure column as `value`."""
    if not isinstance(out, pd.DataFrame):
        raise TypeError(f"aggregate_to_firm returned {type(out)!r}, not a DataFrame")
    df = out.copy()
    if "value" not in df.columns:
        for cand in ("firm_mean", "mean", "firm_value"):
            if cand in df.columns:
                df = df.rename(columns={cand: "value"})
                break
    for col in ("ticker", "tier", "value"):
        if col not in df.columns:
            raise KeyError(f"aggregate_to_firm frame has no {col!r} column "
                           f"(got {list(df.columns)})")
    for col in ("n_filings", "n_windows"):
        if col not in df.columns:
            df[col] = np.nan
    return df.reset_index(drop=True)


def aggregate_to_firm(
    chunk_df: pd.DataFrame,
    section: str = "mda",
    measure: str = "embedding_disagreement",
) -> pd.DataFrame:
    """
    window -> filing -> firm, per ANALYSIS_PLAN.md section 1.  Returns
    ticker, tier, value, n_filings, n_windows.
    """
    fn = getattr(_TT, "aggregate_to_firm", None) if _TT is not None else None
    if fn is not None and not _fellback("aggregate_to_firm"):
        try:
            out = _normalise_firm_frame(fn(chunk_df, section=section, measure=measure))
            _record("aggregate_to_firm", "analysis.trend_tests")
            return out
        except Exception as exc:
            _record("aggregate_to_firm", "window_battery.fallback", repr(exc))
    else:
        _record("aggregate_to_firm", "window_battery.fallback",
                "module or attribute absent" if fn is None else "")
    return _fallback_aggregate_to_firm(chunk_df, section=section, measure=measure)


def jonckheere_terpstra(
    groups: Sequence[Sequence[float]],
    alternative: str = "increasing",
) -> dict[str, Any]:
    """C1's ordered-trend test.  `groups` in ascending order (large, mid, small)."""
    fn = getattr(_TT, "jonckheere_terpstra", None) if _TT is not None else None
    if fn is not None and not _fellback("jonckheere_terpstra"):
        try:
            out = _dataclass_to_dict(fn(groups, alternative=alternative))
            if "statistic" not in out or "p_value" not in out:
                raise KeyError(f"unexpected JT result keys {sorted(out)}")
            out.setdefault("n", out.get("n_total", float("nan")))
            _record("jonckheere_terpstra", "analysis.trend_tests")
            return out
        except Exception as exc:
            _record("jonckheere_terpstra", "window_battery.fallback", repr(exc))
    else:
        _record("jonckheere_terpstra", "window_battery.fallback",
                "module or attribute absent" if fn is None else "")
    return _fallback_jonckheere_terpstra(groups, alternative=alternative)


def label_permutation_test(
    values: Sequence[float],
    labels: Sequence[str],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_draws: int = PERMUTATIONS,
    seed: int = 0,
    alternative: str = "greater",
) -> dict[str, Any]:
    """Null built by permuting tier labels across firms."""
    fn = getattr(_TT, "label_permutation_test", None) if _TT is not None else None
    if fn is not None and not _fellback("label_permutation_test"):
        try:
            res = fn(np.asarray(list(values), dtype=float),
                     np.asarray(list(labels), dtype=object),
                     statistic, n_draws=n_draws, seed=seed, alternative=alternative)
            out = _dataclass_to_dict(res, drop=("null_distribution",))
            null = getattr(res, "null_distribution", None)
            if null is not None:
                arr = np.asarray(null, dtype=float)
                out["null_mean"] = float(arr.mean())
                out["null_sd"] = float(arr.std(ddof=1))
                out["null_percentiles"] = _percentiles(arr)
            if "p_value" not in out:
                raise KeyError(f"unexpected permutation result keys {sorted(out)}")
            _record("label_permutation_test", "analysis.trend_tests")
            return out
        except Exception as exc:
            _record("label_permutation_test", "window_battery.fallback", repr(exc))
    else:
        _record("label_permutation_test", "window_battery.fallback",
                "module or attribute absent" if fn is None else "")
    return _fallback_label_permutation_test(
        values, labels, statistic, n_draws=n_draws, seed=seed, alternative=alternative)


def tier_trend_statistic(firm_df: pd.DataFrame, value_col: str = "value") -> float:
    """
    The sibling's ordered-trend scalar (the JT statistic J), which is what
    C1-perm permutes.  It is NOT signed, so it is reported rather than used as
    the battery's gradient - see the note at the top of this section.
    """
    fn = getattr(_TT, "tier_trend_statistic", None) if _TT is not None else None
    if fn is not None and not _fellback("tier_trend_statistic"):
        try:
            v = float(fn(firm_df, measure_col=value_col, tier_col="tier"))
            _record("tier_trend_statistic", "analysis.trend_tests")
            return v
        except Exception as exc:
            _record("tier_trend_statistic", "window_battery.fallback", repr(exc))
    else:
        _record("tier_trend_statistic", "window_battery.fallback",
                "module or attribute absent" if fn is None else "")
    return _jt_stat(firm_df[value_col].to_numpy(dtype=float), firm_df["tier"].to_numpy())


def gradient_small_minus_large(firm_df: pd.DataFrame, value_col: str = "value") -> float:
    """
    The battery's SIGNED tier gradient: small-cap firm mean minus large-cap firm
    mean.  Positive is the hypothesised direction (H1: Small > Mid > Large).
    Defined locally and never delegated, because W2's CI, W3's sign share and
    W6's one-sided percentile are all defined on a signed quantity.
    """
    means = {t: firm_df.loc[firm_df["tier"] == t, value_col].mean() for t in TIER_ORDER}
    lo, hi = means.get("large", np.nan), means.get("small", np.nan)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return float("nan")
    return float(hi - lo)


def _gradient_stat(values: np.ndarray, labels: np.ndarray) -> float:
    """Signed Small-minus-Large, as a (values, labels) callable for permutation."""
    v = np.asarray(values, dtype=float)
    lab = np.asarray(labels, dtype=object)
    return float(v[lab == "small"].mean() - v[lab == "large"].mean())


def _jt_stat(values: np.ndarray, labels: np.ndarray) -> float:
    """The JT statistic as a (values, labels) callable for permutation."""
    v = np.asarray(values, dtype=float)
    lab = np.asarray(labels, dtype=object)
    return float(jonckheere_terpstra([v[lab == t] for t in TIER_ASCENDING])["statistic"])


def _dataclass_to_dict(obj: Any, drop: Sequence[str] = ()) -> dict[str, Any]:
    """Flatten a sibling dataclass (or dict) result into a plain dict."""
    if isinstance(obj, dict):
        d = dict(obj)
    elif hasattr(obj, "__dataclass_fields__"):
        d = {f: getattr(obj, f) for f in obj.__dataclass_fields__}
        for extra in ("null_mean", "null_sd"):
            if isinstance(getattr(type(obj), extra, None), property):
                try:
                    d[extra] = getattr(obj, extra)
                except Exception:
                    pass
    else:
        d = {k: getattr(obj, k) for k in dir(obj)
             if not k.startswith("_") and not callable(getattr(obj, k))}
    return {k: v for k, v in d.items() if k not in drop}


# ===========================================================================
# Window-level preparation
# ===========================================================================

_PERCENTILE_GRID = (0.5, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5)


def _percentiles(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{q}": float("nan") for q in _PERCENTILE_GRID}
    return {f"p{q}": float(np.percentile(x, q)) for q in _PERCENTILE_GRID}


def _describe(x: Sequence[float]) -> dict[str, float]:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def filing_key(df: pd.DataFrame) -> pd.Series:
    """(ticker, filing_type, filing_date) as one string - the filing identity."""
    return (
        df["ticker"].astype(str) + "|" + df["filing_type"].astype(str)
        + "|" + df["filing_date"].astype(str)
    )


def check_schema(chunk_df: pd.DataFrame) -> list[str]:
    """Return the list of CHUNK_SCHEMA columns missing from `chunk_df`."""
    return [c for c in CHUNK_SCHEMA if c not in chunk_df.columns]


def surviving_windows(
    chunk_df: pd.DataFrame,
    section: str = "mda",
    measure: str = "embedding_disagreement",
) -> pd.DataFrame:
    """
    The analysable window set: one section, failed windows dropped, never imputed.

    A window is dropped if `any_failed` is true OR the measure is null.  Those
    are the same rows by construction (a failed window carries null in both
    disagreement columns), but both conditions are checked because the
    pre-registered rule is about the data, not about one column's bookkeeping.
    """
    missing = check_schema(chunk_df)
    if missing:
        raise ValueError(f"chunk frame is missing required columns: {missing}")
    if measure not in ("numeric_disagreement", "embedding_disagreement"):
        raise ValueError(f"measure must be a disagreement column, got {measure!r}")

    df = chunk_df.copy()
    df["any_failed"] = df["any_failed"].astype("object").map(_as_bool).fillna(False).astype(bool)
    df = df.loc[df["section"].astype(str) == section].copy()
    n_section = len(df)
    df[measure] = pd.to_numeric(df[measure], errors="coerce")

    failed = df["any_failed"].to_numpy()
    null = ~np.isfinite(df[measure].to_numpy(dtype=float))
    dropped = int((failed | null).sum())
    if dropped:
        logger.warning(
            "surviving_windows: dropped %d of %d '%s' windows "
            "(%d any_failed, %d null %s) - dropped, never imputed",
            dropped, n_section, section, int(failed.sum()), int(null.sum()), measure,
        )
    df = df.loc[~(failed | null)].copy()
    df["filing_key"] = filing_key(df)
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce")
    df["chunk_index"] = pd.to_numeric(df["chunk_index"], errors="coerce").astype("Int64")
    df["n_in_section"] = pd.to_numeric(df["n_in_section"], errors="coerce").astype("Int64")
    df["tier_rank"] = df["tier"].map(TIER_RANK)
    df.attrs["n_section_rows"] = n_section
    df.attrs["n_dropped_failed"] = dropped
    return df.reset_index(drop=True)


def _as_bool(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    if isinstance(v, (int, np.integer)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes"):
        return True
    if s in ("false", "f", "0", "no", "", "nan", "none"):
        return False
    return None


def position_decile_frame(
    chunk_df: pd.DataFrame,
    section: str | None = None,
    measure: str = "embedding_disagreement",
) -> pd.DataFrame:
    """
    W4 (iii).  Window-level frame carrying the position column the mixed model
    (W5) consumes:

        position_decile = chunk_index / max(1, n_in_section - 1)

    computed exactly as pre-registered, so it is 0.0 for a section's first
    window, 1.0 for its last, and 0.0 for a one-window section (the denominator
    is floored at 1 rather than dividing by zero).  `position_decile_bin` is the
    same quantity cut into ten equal-width bins 1..10, provided for convenience;
    the model term is the continuous column.

    Passing `section=None` returns every section, which is what W5 wants if it
    models sections jointly.  Failed windows are still dropped.
    """
    if section is None:
        parts = [
            surviving_windows(chunk_df, section=s, measure=measure)
            for s in sorted(chunk_df["section"].astype(str).unique())
        ]
        df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    else:
        df = surviving_windows(chunk_df, section=section, measure=measure)
    if df.empty:
        return df
    idx = df["chunk_index"].astype(float).to_numpy()
    n = df["n_in_section"].astype(float).to_numpy()
    denom = np.maximum(1.0, n - 1.0)
    df["position_decile"] = idx / denom
    df["position_decile_bin"] = np.clip(
        np.floor(df["position_decile"].to_numpy() * 10).astype(int) + 1, 1, 10)
    return df


# ===========================================================================
# Vectorised three-step aggregation (shared by W3 and W6)
# ===========================================================================

class _Structure:
    """
    Filing/firm/tier index structure for one window set, so the aggregation can
    be applied to thousands of simulated window-value matrices at once without
    ever changing the pre-registered formula.
    """

    def __init__(self, w: pd.DataFrame, measure: str):
        self.measure = measure
        order = w.sort_values(["ticker", "filing_key", "chunk_index"]).reset_index(drop=True)
        self.filing_keys: list[str] = list(dict.fromkeys(order["filing_key"]))
        fk_pos = {k: i for i, k in enumerate(self.filing_keys)}
        self.filing_values: list[np.ndarray] = [np.empty(0)] * len(self.filing_keys)
        self.filing_word_counts: list[np.ndarray] = [np.empty(0)] * len(self.filing_keys)
        self.filing_ticker: list[str] = [""] * len(self.filing_keys)
        for key, grp in order.groupby("filing_key", sort=False):
            i = fk_pos[key]
            self.filing_values[i] = grp[measure].to_numpy(dtype=float)
            self.filing_word_counts[i] = grp["word_count"].to_numpy(dtype=float)
            self.filing_ticker[i] = str(grp["ticker"].iloc[0])
        self.counts = np.array([v.size for v in self.filing_values], dtype=int)

        tiers = order.drop_duplicates("ticker").set_index("ticker")["tier"].astype(str)
        self.firm_tickers: list[str] = list(dict.fromkeys(order["ticker"]))
        self.firm_tiers: list[str] = [tiers[t] for t in self.firm_tickers]
        self.firm_filing_idx: list[np.ndarray] = [
            np.array([i for i, tk in enumerate(self.filing_ticker) if tk == firm], dtype=int)
            for firm in self.firm_tickers
        ]
        self.tier_firm_idx: dict[str, np.ndarray] = {
            t: np.array([i for i, ft in enumerate(self.firm_tiers) if ft == t], dtype=int)
            for t in TIER_ORDER
        }

    # -- observed ----------------------------------------------------------
    def observed_filing_means(self) -> np.ndarray:
        return np.array([v.mean() for v in self.filing_values], dtype=float)[None, :]

    # -- aggregation -------------------------------------------------------
    def firm_means(self, filing_means: np.ndarray) -> np.ndarray:
        """(iters, n_filings) -> (iters, n_firms), unweighted over the firm's filings."""
        return np.stack(
            [filing_means[:, idx].mean(axis=1) for idx in self.firm_filing_idx], axis=1)

    def tier_means(self, filing_means: np.ndarray) -> dict[str, np.ndarray]:
        fm = self.firm_means(filing_means)
        return {t: (fm[:, idx].mean(axis=1) if idx.size else np.full(fm.shape[0], np.nan))
                for t, idx in self.tier_firm_idx.items()}

    def gradient(self, filing_means: np.ndarray) -> np.ndarray:
        tm = self.tier_means(filing_means)
        return tm["small"] - tm["large"]

    def ordering_holds(self, filing_means: np.ndarray) -> np.ndarray:
        tm = self.tier_means(filing_means)
        return (tm["small"] > tm["mid"]) & (tm["mid"] > tm["large"])

    def firm_frame(self, filing_means: np.ndarray) -> pd.DataFrame:
        fm = self.firm_means(filing_means)[0]
        return pd.DataFrame({
            "ticker": self.firm_tickers,
            "tier": self.firm_tiers,
            "value": fm,
            "n_filings": [idx.size for idx in self.firm_filing_idx],
            "n_windows": [int(self.counts[idx].sum()) for idx in self.firm_filing_idx],
        })

    def tier_firm_counts(self) -> dict[str, int]:
        return {t: int(idx.size) for t, idx in self.tier_firm_idx.items()}


def _firm_frame_from_windows(w: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Observed firm-level frame straight from a window set (no simulation)."""
    if w.empty:
        return pd.DataFrame(columns=["ticker", "tier", "value", "n_filings", "n_windows"])
    st = _Structure(w, measure)
    return st.firm_frame(st.observed_filing_means())


def _tier_mean_dict(firm_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {t: _describe(firm_df.loc[firm_df["tier"] == t, "value"]) for t in TIER_ORDER}


# ===========================================================================
# Primary (the number every test is compared against)
# ===========================================================================

def primary_result(
    chunk_df: pd.DataFrame,
    section: str,
    measure: str,
    seed: int,
    n_permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    """
    The pre-registered primary: firm-level tier means on the analysed section,
    the signed Small-minus-Large gradient the battery's criteria are defined on,
    the Jonckheere-Terpstra ordered-trend test (C1), C1-perm (the label
    permutation of the SAME JT statistic), and - separately labelled - the
    label-permutation null of the signed gradient, which is the null W2 and W6
    are comparable with.
    """
    w = surviving_windows(chunk_df, section=section, measure=measure)
    firm_df = aggregate_to_firm(chunk_df, section=section, measure=measure)

    gradient = gradient_small_minus_large(firm_df)
    tier_means = _tier_mean_dict(firm_df)
    values = firm_df["value"].to_numpy(dtype=float)
    labels = firm_df["tier"].to_numpy()

    groups = [firm_df.loc[firm_df["tier"] == t, "value"].to_numpy(dtype=float)
              for t in TIER_ASCENDING]
    jt = jonckheere_terpstra(groups, alternative="increasing")
    jt_perm = label_permutation_test(values, labels, _jt_stat,
                                     n_draws=n_permutations, seed=seed,
                                     alternative="greater")
    grad_perm = label_permutation_test(values, labels, _gradient_stat,
                                       n_draws=n_permutations, seed=seed,
                                       alternative="greater")

    ordering = (
        tier_means["small"]["mean"] > tier_means["mid"]["mean"]
        > tier_means["large"]["mean"]
    )
    return {
        "section": section,
        "measure": measure,
        "n_windows_in_section": int(w.attrs.get("n_section_rows", len(w))),
        "n_windows_dropped_failed": int(w.attrs.get("n_dropped_failed", 0)),
        "n_windows_used": int(len(w)),
        "n_filings": int(w["filing_key"].nunique()),
        "n_firms": int(w["ticker"].nunique()),
        "n_firms_in_firm_frame": int(len(firm_df)),
        "tier_firm_counts": {t: int((firm_df["tier"] == t).sum()) for t in TIER_ORDER},
        "tier_means": tier_means,
        "gradient_small_minus_large": gradient,
        "gradient_sign": int(np.sign(gradient)) if np.isfinite(gradient) else 0,
        "gradient_definition": ("firm-level small-cap mean minus large-cap mean; this is "
                                "the signed quantity W2/W3/W4/W6/W7 are defined on"),
        "monotone_small_gt_mid_gt_large": bool(ordering),
        "tier_trend_statistic_J": float(tier_trend_statistic(firm_df)),
        "tier_trend_statistic_note": ("the ordered-trend (Jonckheere-Terpstra) statistic "
                                      "returned by analysis.trend_tests.tier_trend_statistic; "
                                      "non-negative, hence not usable as a signed gradient"),
        "jonckheere_terpstra": _to_jsonable(jt),
        "label_permutation": _to_jsonable(jt_perm),
        "label_permutation_statistic": "Jonckheere-Terpstra J (C1-perm as pre-registered)",
        "gradient_permutation": _to_jsonable(grad_perm),
        "gradient_permutation_statistic": "signed Small-minus-Large firm-mean difference",
        "firm_means": firm_df.to_dict(orient="records"),
    }


# ===========================================================================
# W1 - Descriptives
# ===========================================================================

def _spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, float]:
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    ok = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[ok], ya[ok]
    if xa.size < 3 or np.unique(xa).size < 2 or np.unique(ya).size < 2:
        return {"rho": float("nan"), "p_value": float("nan"), "n": int(xa.size)}
    res = stats.spearmanr(xa, ya)
    return {"rho": float(res.statistic), "p_value": float(res.pvalue), "n": int(xa.size)}


def w1_descriptives(
    chunk_df: pd.DataFrame,
    section: str,
    measure: str,
) -> dict[str, Any]:
    """
    W1.  Purely descriptive - no pass/fail, reported whatever it shows.

    Windows per filing and per firm by tier; mean window size by tier; and
    Spearman of window count and of mean window size against tier rank
    (large = 3), against numeric disagreement and against embedding
    disagreement, at both the filing and the firm level, each with its own n
    and p.
    """
    out: dict[str, Any] = {"section": section, "measure": measure, "criterion": None,
                           "note": "Descriptive. No pass/fail (ANALYSIS_PLAN.md s5, W1)."}

    wn = surviving_windows(chunk_df, section=section, measure="numeric_disagreement")
    we = surviving_windows(chunk_df, section=section, measure="embedding_disagreement")

    def _filing_table(w: pd.DataFrame, col: str) -> pd.DataFrame:
        return (w.groupby(["ticker", "tier", "filing_type", "filing_key"], as_index=False)
                 .agg(n_windows=(col, "size"),
                      mean_window_words=("word_count", "mean"),
                      filing_mean=(col, "mean")))

    f_num = _filing_table(wn, "numeric_disagreement").rename(
        columns={"filing_mean": "numeric_disagreement"})
    f_emb = _filing_table(we, "embedding_disagreement").rename(
        columns={"filing_mean": "embedding_disagreement"})
    filings = f_emb.merge(
        f_num[["filing_key", "numeric_disagreement"]], on="filing_key", how="outer")
    filings["tier_rank"] = filings["tier"].map(TIER_RANK)

    # -- windows per filing, per firm, and window size, by tier ------------
    out["windows_per_filing_by_tier"] = {
        t: _describe(filings.loc[filings["tier"] == t, "n_windows"]) for t in TIER_ORDER}
    out["windows_per_filing_overall"] = _describe(filings["n_windows"])

    firms = (filings.groupby(["ticker", "tier"], as_index=False)
                    .agg(windows_total=("n_windows", "sum"),
                         windows_per_filing=("n_windows", "mean"),
                         n_filings=("n_windows", "size"),
                         mean_window_words=("mean_window_words", "mean"),
                         embedding_disagreement=("embedding_disagreement", "mean"),
                         numeric_disagreement=("numeric_disagreement", "mean")))
    firms["tier_rank"] = firms["tier"].map(TIER_RANK)
    out["windows_per_firm_by_tier"] = {
        "total": {t: _describe(firms.loc[firms["tier"] == t, "windows_total"])
                  for t in TIER_ORDER},
        "mean_per_filing": {t: _describe(firms.loc[firms["tier"] == t, "windows_per_filing"])
                            for t in TIER_ORDER},
    }
    out["mean_window_words_by_tier"] = {
        "filing_level": {t: _describe(filings.loc[filings["tier"] == t, "mean_window_words"])
                         for t in TIER_ORDER},
        "firm_level": {t: _describe(firms.loc[firms["tier"] == t, "mean_window_words"])
                       for t in TIER_ORDER},
    }
    out["window_words_overall"] = _describe(we["word_count"])

    # -- Spearman block ----------------------------------------------------
    out["spearman"] = {
        "filing_level": {
            "window_count_vs_tier_rank": _spearman(filings["n_windows"], filings["tier_rank"]),
            "window_count_vs_numeric": _spearman(
                filings["n_windows"], filings["numeric_disagreement"]),
            "window_count_vs_embedding": _spearman(
                filings["n_windows"], filings["embedding_disagreement"]),
            "mean_window_words_vs_tier_rank": _spearman(
                filings["mean_window_words"], filings["tier_rank"]),
            "mean_window_words_vs_numeric": _spearman(
                filings["mean_window_words"], filings["numeric_disagreement"]),
            "mean_window_words_vs_embedding": _spearman(
                filings["mean_window_words"], filings["embedding_disagreement"]),
        },
        "firm_level": {
            "window_count_vs_tier_rank": _spearman(firms["windows_total"], firms["tier_rank"]),
            "window_count_vs_numeric": _spearman(
                firms["windows_total"], firms["numeric_disagreement"]),
            "window_count_vs_embedding": _spearman(
                firms["windows_total"], firms["embedding_disagreement"]),
            "mean_window_words_vs_tier_rank": _spearman(
                firms["mean_window_words"], firms["tier_rank"]),
            "mean_window_words_vs_numeric": _spearman(
                firms["mean_window_words"], firms["numeric_disagreement"]),
            "mean_window_words_vs_embedding": _spearman(
                firms["mean_window_words"], firms["embedding_disagreement"]),
        },
    }
    out["tier_rank_convention"] = "large=3, mid=2, small=1 (as in coverage_diagnostic.py)"
    out["filing_table"] = filings.to_dict(orient="records")
    return out


# ===========================================================================
# W2 - Matched comparison
# ===========================================================================

def _bin_label(n: int) -> str:
    for label, lo, hi in W2_BIN_EDGES:
        if lo <= n <= hi:
            return label
    return "below-gate"


def _welch_diff(a: Sequence[float], b: Sequence[float], conf: float = 0.95) -> dict[str, float]:
    """Mean(a) - mean(b) with a Welch (unequal-variance) confidence interval."""
    x = np.asarray(list(a), dtype=float); x = x[np.isfinite(x)]
    y = np.asarray(list(b), dtype=float); y = y[np.isfinite(y)]
    out = {"n_a": int(x.size), "n_b": int(y.size), "mean_a": float("nan"),
           "mean_b": float("nan"), "diff": float("nan"), "se": float("nan"),
           "ci_low": float("nan"), "ci_high": float("nan"), "df": float("nan"),
           "ci_excludes_zero": False}
    if x.size == 0 or y.size == 0:
        return out
    out["mean_a"], out["mean_b"] = float(x.mean()), float(y.mean())
    out["diff"] = out["mean_a"] - out["mean_b"]
    if x.size < 2 or y.size < 2:
        return out  # CI not identified from one observation on a side
    vx, vy = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    se = np.sqrt(vx + vy)
    if se == 0:
        return out
    df = (vx + vy) ** 2 / (vx ** 2 / (x.size - 1) + vy ** 2 / (y.size - 1))
    crit = float(stats.t.ppf(0.5 + conf / 2, df))
    out.update({"se": float(se), "df": float(df),
                "ci_low": out["diff"] - crit * se, "ci_high": out["diff"] + crit * se})
    out["ci_excludes_zero"] = bool(out["ci_low"] > 0 or out["ci_high"] < 0)
    return out


def w2_matched(
    chunk_df: pd.DataFrame,
    section: str,
    measure: str,
    primary_sign: int,
) -> dict[str, Any]:
    """
    W2.  Bin filings by window count (2-3, 4-5, 6-8, 9+), compute the tier
    gradient inside each bin, report every bin's tier composition, and state
    explicitly which bins contain all three tiers.

    A bin that does not contain all three tiers is LABELLED
    `identified_for_three_tier = False` and is excluded from the criterion -
    never quietly averaged into it.

    Criterion (pre-declared): the primary sign must not be CONTRADICTED -
    reversed with a CI excluding 0 - in any bin containing all three tiers.

    The within-bin unit is the FILING, not the firm: a firm's two filings can
    land in different bins, so a firm-level mean is not defined inside a bin.
    This is a deliberate, stated departure from the confirmatory unit, and it is
    why W2 is a diagnostic rather than a confirmatory test.
    """
    w = surviving_windows(chunk_df, section=section, measure=measure)
    filings = (w.groupby(["ticker", "tier", "filing_type", "filing_key"], as_index=False)
                .agg(n_windows=(measure, "size"), value=(measure, "mean")))
    filings["bin"] = filings["n_windows"].astype(int).map(_bin_label)

    below = filings.loc[filings["bin"] == "below-gate"]
    if len(below):
        logger.warning(
            "w2_matched: %d filing(s) have fewer than 2 surviving windows and fall "
            "outside the pre-registered bins: %s",
            len(below), sorted(below["filing_key"]))

    bins: dict[str, Any] = {}
    for label, _lo, _hi in W2_BIN_EDGES:
        sub = filings.loc[filings["bin"] == label]
        comp = {t: int((sub["tier"] == t).sum()) for t in TIER_ORDER}
        all_three = all(v > 0 for v in comp.values())
        tier_means = {t: _describe(sub.loc[sub["tier"] == t, "value"]) for t in TIER_ORDER}
        sl = _welch_diff(sub.loc[sub["tier"] == "small", "value"],
                         sub.loc[sub["tier"] == "large", "value"])
        grad = sl["diff"]
        contradicted = bool(
            all_three and np.isfinite(grad) and primary_sign != 0
            and np.sign(grad) == -primary_sign and sl["ci_excludes_zero"])
        bins[label] = {
            "n_filings": int(len(sub)),
            "tier_composition": comp,
            "identified_for_three_tier": all_three,
            "identification_note": (
                "contains all three tiers" if all_three else
                "NOT IDENTIFIED for a three-tier comparison - missing tier(s): "
                + ", ".join(t for t, v in comp.items() if v == 0)),
            "tier_means": tier_means,
            "gradient_small_minus_large": grad,
            "gradient_ci": {"low": sl["ci_low"], "high": sl["ci_high"],
                            "se": sl["se"], "df": sl["df"],
                            "excludes_zero": sl["ci_excludes_zero"],
                            "method": "Welch two-sample t, filing-level, 95%"},
            "monotone_small_gt_mid_gt_large": bool(
                np.isfinite(tier_means["small"]["mean"])
                and np.isfinite(tier_means["mid"]["mean"])
                and np.isfinite(tier_means["large"]["mean"])
                and tier_means["small"]["mean"] > tier_means["mid"]["mean"]
                > tier_means["large"]["mean"]),
            "contradicts_primary": contradicted,
        }

    identified = [b for b, d in bins.items() if d["identified_for_three_tier"]]
    contradicting = [b for b in identified if bins[b]["contradicts_primary"]]

    cem = _cem_small_vs_large(filings, primary_sign)

    return {
        "section": section,
        "measure": measure,
        "unit": "filing",
        "primary_sign": int(primary_sign),
        "bins": bins,
        "bins_identified_for_three_tier": identified,
        "bins_not_identified": [b for b, d in bins.items()
                                if not d["identified_for_three_tier"]],
        "filings_outside_bins": {
            "n": int(len(below)), "filing_keys": sorted(below["filing_key"])},
        "coarsened_exact_matching": cem,
        "criterion": ("the primary sign must not be CONTRADICTED (reversed with a CI "
                      "excluding 0) in any bin containing all three tiers"),
        "criterion_met": bool(not contradicting),
        "criterion_vacuous": bool(not identified),
        "criterion_note": (
            "NO bin contains all three tiers, so the criterion is satisfied VACUOUSLY: "
            "W2 provides no evidence either way and must not be read as support"
            if not identified else
            f"evaluated on {len(identified)} bin(s) containing all three tiers"),
        "contradicting_bins": contradicting,
        "filing_table": filings.to_dict(orient="records"),
    }


def _cem_small_vs_large(filings: pd.DataFrame, primary_sign: int) -> dict[str, Any]:
    """
    Coarsened exact matching on (window-count bin, filing_type).

    Strata containing both small- and large-cap filings are retained; strata
    lacking overlap are discarded and counted.  The reported estimate is the
    ATT-style weighted Small-minus-Large difference, weighting each retained
    stratum by its number of small-cap (treated) filings.  Mid-cap filings take
    no part in this two-tier contrast and are counted separately rather than
    being absorbed into either side.
    """
    sl = filings.loc[filings["tier"].isin(["small", "large"])].copy()
    strata: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    for (b, form), grp in sl.groupby(["bin", "filing_type"], sort=True):
        n_s = int((grp["tier"] == "small").sum())
        n_l = int((grp["tier"] == "large").sum())
        rec = {"bin": str(b), "filing_type": str(form), "n_small": n_s, "n_large": n_l}
        if n_s > 0 and n_l > 0:
            rec["mean_small"] = float(grp.loc[grp["tier"] == "small", "value"].mean())
            rec["mean_large"] = float(grp.loc[grp["tier"] == "large", "value"].mean())
            rec["diff"] = rec["mean_small"] - rec["mean_large"]
            strata.append(rec)
        else:
            rec["reason"] = "no overlap: " + ("no large-cap" if n_l == 0 else "no small-cap")
            discarded.append(rec)

    n_discarded = int(sum(d["n_small"] + d["n_large"] for d in discarded))
    n_mid = int((filings["tier"] == "mid").sum())
    if strata:
        wts = np.array([s["n_small"] for s in strata], dtype=float)
        diffs = np.array([s["diff"] for s in strata], dtype=float)
        att = float((wts * diffs).sum() / wts.sum())
        unweighted = float(diffs.mean())
    else:
        att = float("nan")
        unweighted = float("nan")
    logger.info(
        "CEM: %d stratum/strata retained, %d filing(s) discarded for lack of overlap, "
        "%d mid-cap filing(s) outside the Small-vs-Large contrast",
        len(strata), n_discarded, n_mid)
    return {
        "coarsening": "exact on (window-count bin, filing_type)",
        "estimand": "ATT-style weighted Small-minus-Large, weights = n_small per stratum",
        "strata_retained": strata,
        "strata_discarded": discarded,
        "n_strata_retained": len(strata),
        "n_filings_discarded_no_overlap": n_discarded,
        "n_mid_filings_outside_contrast": n_mid,
        "weighted_small_minus_large": att,
        "unweighted_stratum_mean_difference": unweighted,
        "sign_agrees_with_primary": bool(
            np.isfinite(att) and primary_sign != 0 and np.sign(att) == primary_sign),
    }


# ===========================================================================
# W3 - Equal-k subsampling (the decisive test)
# ===========================================================================

def w3_equal_k(
    chunk_df: pd.DataFrame,
    section: str,
    measure: str,
    seed: int,
    k_values: Sequence[int] = W3_K_VALUES,
    n_iterations: int = W3_ITERATIONS,
    primary_sign: int = 1,
) -> dict[str, Any]:
    """
    W3.  THE DECISIVE TEST.  For each filing draw k windows at random without
    replacement, recompute filing means -> firm means -> the tier gradient, and
    repeat `n_iterations` times from an explicit seed.

    k = 2 is the sample minimum and uses every filing.  k = 4 is restricted to
    filings with at least 4 surviving windows; filings - and therefore sometimes
    whole firms - drop out, and the drop-out is reported, not absorbed.

    PRE-DECLARED CRITERION: the primary sign must hold in >= 90% of k = 2 draws.
    `criterion_met` is reported for k = 2 only, as a plain boolean, next to the
    share it was computed from.
    """
    w = surviving_windows(chunk_df, section=section, measure=measure)
    out: dict[str, Any] = {
        "section": section, "measure": measure, "seed": int(seed),
        "n_iterations": int(n_iterations), "primary_sign": int(primary_sign),
        "criterion": (f"the primary sign must hold in >= {W3_SIGN_SHARE_CRITERION:.0%} "
                      "of k = 2 draws"),
        "by_k": {},
    }
    all_filings = set(w["filing_key"].unique())
    all_firms = set(w["ticker"].unique())

    for k in k_values:
        counts = w.groupby("filing_key")[measure].size()
        keep = set(counts.index[counts >= k])
        sub = w.loc[w["filing_key"].isin(keep)]
        dropped_filings = sorted(all_filings - keep)
        res: dict[str, Any] = {
            "k": int(k),
            "restriction": ("every filing (k equals the sample minimum)" if k == MIN_USABLE_CHUNKS
                            else f"filings with at least {k} surviving windows"),
            "n_filings_used": int(len(keep)),
            "n_filings_dropped": int(len(dropped_filings)),
            "filings_dropped": dropped_filings,
        }
        if sub.empty:
            res.update({"error": "no filing has k surviving windows", "criterion_met": False})
            out["by_k"][f"k={k}"] = res
            continue

        st = _Structure(sub, measure)
        res["n_firms_used"] = int(len(st.firm_tickers))
        res["firms_dropped"] = sorted(all_firms - set(st.firm_tickers))
        res["n_firms_dropped"] = int(len(res["firms_dropped"]))
        res["tier_firm_counts"] = st.tier_firm_counts()

        # Seed is derived from (seed, k) so k=2 and k=4 are independent draws and
        # both are reproducible from the single recorded seed.
        rng = np.random.default_rng([seed, k])
        filing_means = np.empty((n_iterations, len(st.filing_keys)), dtype=float)
        for j, vals in enumerate(st.filing_values):
            n = vals.size
            if n == k:
                filing_means[:, j] = vals.mean()           # the draw is the data
            else:
                pick = rng.random((n_iterations, n)).argsort(axis=1)[:, :k]
                filing_means[:, j] = vals[pick].mean(axis=1)

        grad = st.gradient(filing_means)
        order = st.ordering_holds(filing_means)
        tm = st.tier_means(filing_means)
        share_sign = float(np.mean(np.sign(grad) == primary_sign))
        res.update({
            "gradient_distribution": {
                **_describe(grad),
                "percentiles": _percentiles(grad),
                "share_gradient_positive": float(np.mean(grad > 0)),
                "share_gradient_zero": float(np.mean(grad == 0)),
            },
            "tier_mean_distributions": {
                t: {**_describe(v), "percentiles": _percentiles(v)} for t, v in tm.items()},
            "share_primary_sign_holds": share_sign,
            "share_ordering_small_gt_mid_gt_large": float(np.mean(order)),
            "n_filings_whose_draw_is_deterministic": int(
                sum(1 for v in st.filing_values if v.size == k)),
            "observed_gradient_on_this_subset": float(
                st.gradient(st.observed_filing_means())[0]),
        })
        if k == MIN_USABLE_CHUNKS:
            res["criterion_met"] = bool(share_sign >= W3_SIGN_SHARE_CRITERION)
            res["criterion_applies"] = True
        else:
            res["criterion_met"] = None
            res["criterion_applies"] = False
            res["criterion_note"] = (
                "the pre-declared >= 90% criterion is defined on k = 2 only; "
                "this row is reported for information")
        out["by_k"][f"k={k}"] = res

    k2 = out["by_k"].get(f"k={MIN_USABLE_CHUNKS}", {})
    out["criterion_met"] = bool(k2.get("criterion_met", False))
    out["share_primary_sign_holds_k2"] = k2.get("share_primary_sign_holds", float("nan"))
    return out


# ===========================================================================
# W4 - Position control
# ===========================================================================

def w4_position(
    chunk_df: pd.DataFrame,
    section: str,
    measure: str,
    primary_sign: int,
    first_k: int = W4_FIRST_K,
) -> dict[str, Any]:
    """
    W4.  Position control.

    (i)   first-k windows only, for every filing (k = 2, the sample minimum).
    (ii)  all windows EXCLUDING each section's first window.  A 2-window filing
          then contributes 1 window and a 1-window section contributes 0, so
          filings drop out; the count and the identities are reported.
    (iii) the `position_decile` column, computed as
          chunk_index / max(1, n_in_section - 1), exposed for the W5 mixed model
          via position_decile_frame().

    Criterion: directional agreement with the primary.  Divergence is a
    composition effect and is itself a finding, not a failure to hide.
    """
    w = surviving_windows(chunk_df, section=section, measure=measure)
    base_filings = set(w["filing_key"].unique())
    base_firms = set(w["ticker"].unique())

    variants: dict[str, Any] = {}

    # (i) first-k windows only
    first = w.loc[w["chunk_index"].astype(float) < first_k]
    variants[f"first_{first_k}_windows"] = _variant_summary(
        first, measure, primary_sign, base_filings, base_firms,
        note=f"windows with chunk_index < {first_k} in every filing")

    # (ii) drop each section's first window
    nofirst = w.loc[w["chunk_index"].astype(float) >= 1]
    summary = _variant_summary(
        nofirst, measure, primary_sign, base_filings, base_firms,
        note="all windows except each section's first (chunk_index >= 1)")
    two_window = (w.groupby("filing_key")[measure].size() == 2)
    summary["n_filings_reduced_to_one_window"] = int(two_window.sum())
    summary["n_one_window_sections_contributing_zero"] = int(
        (w.groupby("filing_key")[measure].size() == 1).sum())
    summary["note_on_dropout"] = (
        "a 2-window filing contributes 1 window here; a 1-window section "
        "contributes 0 and the filing drops out entirely")
    variants["excluding_first_window"] = summary

    # (iii) the position column for W5
    pos = position_decile_frame(chunk_df, section=section, measure=measure)
    pos_by_tier = {t: _describe(pos.loc[pos["tier"] == t, "position_decile"])
                   for t in TIER_ORDER}
    first_share = (pos.assign(is_first=pos["chunk_index"].astype(float) == 0)
                      .groupby("tier")["is_first"].mean().to_dict())

    return {
        "section": section, "measure": measure, "primary_sign": int(primary_sign),
        "first_k": int(first_k),
        "variants": variants,
        "position_decile": {
            "formula": "chunk_index / max(1, n_in_section - 1)",
            "exposed_by": "analysis.window_battery.position_decile_frame",
            "n_windows": int(len(pos)),
            "by_tier": pos_by_tier,
            "share_of_windows_that_are_section_opening_by_tier": {
                str(k): float(v) for k, v in first_share.items()},
            "spearman_position_vs_measure": _spearman(
                pos["position_decile"], pos[measure]),
        },
        "criterion": "directional agreement with the primary",
        "criterion_met": bool(all(v.get("sign_agrees_with_primary", False)
                                  for v in variants.values())),
    }


def _variant_summary(
    w: pd.DataFrame,
    measure: str,
    primary_sign: int,
    base_filings: set,
    base_firms: set,
    note: str,
) -> dict[str, Any]:
    """Firm-level gradient for a filtered window set, with its drop-out accounting."""
    if w.empty:
        return {"note": note, "error": "no windows survive this filter",
                "sign_agrees_with_primary": False}
    firm_df = _firm_frame_from_windows(w, measure)
    grad = gradient_small_minus_large(firm_df)
    tm = _tier_mean_dict(firm_df)
    kept_filings = set(w["filing_key"].unique())
    kept_firms = set(w["ticker"].unique())
    return {
        "note": note,
        "n_windows": int(len(w)),
        "n_filings": int(len(kept_filings)),
        "n_filings_dropped": int(len(base_filings - kept_filings)),
        "filings_dropped": sorted(base_filings - kept_filings),
        "n_firms": int(len(kept_firms)),
        "n_firms_dropped": int(len(base_firms - kept_firms)),
        "firms_dropped": sorted(base_firms - kept_firms),
        "tier_firm_counts": {t: int((firm_df["tier"] == t).sum()) for t in TIER_ORDER},
        "tier_means": tm,
        "gradient_small_minus_large": grad,
        "monotone_small_gt_mid_gt_large": bool(
            np.isfinite(tm["small"]["mean"]) and np.isfinite(tm["mid"]["mean"])
            and np.isfinite(tm["large"]["mean"])
            and tm["small"]["mean"] > tm["mid"]["mean"] > tm["large"]["mean"]),
        "sign_agrees_with_primary": bool(
            np.isfinite(grad) and primary_sign != 0 and np.sign(grad) == primary_sign),
    }


# ===========================================================================
# W6 - Null simulation
# ===========================================================================

def w6_null_simulation(
    chunk_df: pd.DataFrame,
    section: str,
    measure: str,
    seed: int,
    n_iterations: int = W6_ITERATIONS,
    observed_gradient: float | None = None,
) -> dict[str, Any]:
    """
    W6.  How much tier gradient can pure averaging over unequal window counts
    manufacture, when there is no tier effect at all?

    HELD FIXED
      * every filing's real surviving-window count,
      * the filing -> firm -> tier structure (which filings belong to which firm,
        which firm sits in which tier, and the tier sizes),
      * the pooled marginal distribution of window-level disagreement in this
        section (the simulated values are drawn from the observed values
        themselves, so the null has the real measure's scale, spread and shape),
      * the pre-registered aggregation.

    RANDOMISED
      * which value lands in which window.  Values are drawn i.i.d. with
        replacement from the pooled empirical distribution, so by construction
        the expected window-level disagreement is identical for every tier,
        every firm and every filing.  The tier signal is destroyed; only the
        unequal window counts remain.

    Reported: the null distribution of the gradient, the observed gradient's
    percentile within it, and a one-sided p-value in the pre-registered
    direction Small > Mid > Large.  No pass/fail - this is the figure that makes
    the advisor's concern quantitative.
    """
    w = surviving_windows(chunk_df, section=section, measure=measure)
    st = _Structure(w, measure)
    pooled = np.concatenate([v for v in st.filing_values if v.size]) if st.counts.sum() else np.empty(0)
    obs_here = float(st.gradient(st.observed_filing_means())[0]) if pooled.size else float("nan")
    obs = obs_here
    if observed_gradient is not None and np.isfinite(observed_gradient):
        # The primary's gradient comes from the usability-gated firm frame; this
        # module's window set is ungated.  Both are reported so any difference
        # between them is visible rather than absorbed.
        obs = float(observed_gradient)

    if pooled.size == 0:
        return {"section": section, "measure": measure, "error": "no surviving windows"}

    rng = np.random.default_rng([seed, 6])
    counts = st.counts
    total = int(counts.sum())
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(int)

    draws = rng.choice(pooled, size=(n_iterations, total), replace=True)
    sums = np.add.reduceat(draws, starts, axis=1)
    filing_means = sums / counts[None, :]

    grad = st.gradient(filing_means)
    order = st.ordering_holds(filing_means)
    tm = st.tier_means(filing_means)

    n_ge = int((grad >= obs).sum())
    p_one_sided = (1 + n_ge) / (1 + n_iterations)
    pct = float((grad < obs).mean() * 100.0)

    return {
        "section": section, "measure": measure, "seed": int(seed),
        "n_iterations": int(n_iterations),
        "held_fixed": [
            "each filing's real surviving-window count",
            "the filing -> firm -> tier structure and tier sizes",
            "the pooled empirical distribution of window-level disagreement",
            "the pre-registered window -> filing -> firm -> tier aggregation",
        ],
        "randomised": [
            "which pooled value lands in which window (i.i.d., with replacement)",
            "and therefore all tier, firm and filing signal in the measure",
        ],
        "pooled_distribution": {**_describe(pooled), "percentiles": _percentiles(pooled)},
        "window_counts_by_filing": {
            k: int(c) for k, c in zip(st.filing_keys, counts)},
        "window_count_summary": _describe(counts),
        "null_gradient": {**_describe(grad), "percentiles": _percentiles(grad)},
        "null_tier_means": {t: _describe(v) for t, v in tm.items()},
        "null_share_ordering_small_gt_mid_gt_large": float(np.mean(order)),
        "observed_gradient": obs,
        "observed_gradient_source": (
            "the primary (usability-gated firm frame)" if obs != obs_here
            else "this module's own window set"),
        "observed_gradient_on_w6_window_set": obs_here,
        "observed_percentile_in_null": pct,
        "p_value_one_sided": float(p_one_sided),
        "p_value_convention": "(1 + #{null >= observed}) / (1 + n_iterations), one-sided",
        "null_abs_gradient_p95": float(np.percentile(np.abs(grad), 95)),
        "interpretation": (
            "the 95th percentile of the null gradient is the largest Small-minus-Large "
            "difference that unequal window counts alone manufacture 5% of the time"),
        "criterion": None,
        "note": "No pass/fail (ANALYSIS_PLAN.md s5, W6).",
    }


# ===========================================================================
# W7 - Window-size check
# ===========================================================================

def _ols(
    y: np.ndarray,
    X: np.ndarray,
    names: Sequence[str],
    clusters: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Ordinary least squares by `numpy.linalg.lstsq`, with classical SEs and
    (optionally) one-way cluster-robust SEs.  Written out rather than taken from
    statsmodels so this module adds no dependency that is absent from
    requirements.txt.
    """
    n, kp = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - kp
    XtX_inv = np.linalg.pinv(X.T @ X)
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(XtX_inv * s2))

    se_cl = None
    n_clusters = None
    if clusters is not None:
        uniq = np.unique(clusters)
        n_clusters = int(uniq.size)
        meat = np.zeros((kp, kp))
        for g in uniq:
            m = clusters == g
            Xg, ug = X[m], resid[m]
            s = Xg.T @ ug
            meat += np.outer(s, s)
        adj = (n_clusters / max(n_clusters - 1, 1)) * ((n - 1) / max(dof, 1))
        V = XtX_inv @ meat @ XtX_inv * adj
        se_cl = np.sqrt(np.diag(V))

    crit = float(stats.t.ppf(0.975, dof))
    coefs = {}
    for i, nm in enumerate(names):
        rec = {
            "coef": float(beta[i]),
            "se": float(se[i]),
            "t": float(beta[i] / se[i]) if se[i] > 0 else float("nan"),
            "p_value": float(2 * stats.t.sf(abs(beta[i] / se[i]), dof)) if se[i] > 0 else float("nan"),
            "ci_low": float(beta[i] - crit * se[i]),
            "ci_high": float(beta[i] + crit * se[i]),
        }
        rec["ci_excludes_zero"] = bool(rec["ci_low"] > 0 or rec["ci_high"] < 0)
        if se_cl is not None:
            rec["se_cluster_firm"] = float(se_cl[i])
            rec["ci_low_cluster_firm"] = float(beta[i] - crit * se_cl[i])
            rec["ci_high_cluster_firm"] = float(beta[i] + crit * se_cl[i])
            rec["ci_excludes_zero_cluster_firm"] = bool(
                rec["ci_low_cluster_firm"] > 0 or rec["ci_high_cluster_firm"] < 0)
        coefs[nm] = rec
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n_observations": int(n),
        "n_parameters": int(kp),
        "df_residual": int(dof),
        "r_squared": float(1 - (resid @ resid) / ss_tot) if ss_tot > 0 else float("nan"),
        "n_clusters_firm": n_clusters,
        "coefficients": coefs,
        "se_note": ("classical SEs are the pre-registered 'simple OLS'; firm-clustered "
                    "SEs are reported beside them because windows are nested in filings "
                    "in firms and classical SEs understate uncertainty"),
    }


def w7_window_size(
    chunk_df: pd.DataFrame,
    section: str,
    measure: str,
    primary_sign: int,
    exact_words: int = EXACT_WINDOW_WORDS,
) -> dict[str, Any]:
    """
    W7.  Window-size check.

    (i)  the primary recomputed on exact-`exact_words`-word windows only, tails
         excluded.  Reports how many windows and filings that drops and whether
         any filing falls below the MIN_USABLE_CHUNKS = 2 gate.
    (ii) all windows with word_count as a covariate: a window-level OLS of
         disagreement on tier dummies (reference = large) plus log(word_count).
         The full mixed model is W5's job and is not duplicated here.

    Criterion: directional agreement with the primary.
    """
    w = surviving_windows(chunk_df, section=section, measure=measure)
    base_filings = set(w["filing_key"].unique())
    base_firms = set(w["ticker"].unique())

    # -- (i) exact-length windows only -------------------------------------
    exact = w.loc[w["word_count"].astype(float) == float(exact_words)]
    summary = _variant_summary(
        exact, measure, primary_sign, base_filings, base_firms,
        note=f"only windows of exactly {exact_words} words (tail-merged windows excluded)")
    summary["n_windows_dropped"] = int(len(w) - len(exact))
    summary["word_count_distribution_all_windows"] = _describe(w["word_count"])
    if not exact.empty:
        counts = exact.groupby("filing_key")[measure].size()
        below = sorted(counts.index[counts < MIN_USABLE_CHUNKS])
        summary["filings_below_min_usable_chunks_gate"] = below
        summary["n_filings_below_min_usable_chunks_gate"] = int(len(below))
        summary["any_filing_below_gate"] = bool(len(below) > 0)
        summary["gate_note"] = (
            f"MIN_USABLE_CHUNKS = {MIN_USABLE_CHUNKS}. These filings are RETAINED in the "
            "W7(i) recomputation and reported, so the sensitivity check is not quietly "
            "narrowed; the pre-registered gate governs the primary pipeline, and W7(i) "
            "is a sensitivity analysis of it.")
        gated = exact.loc[exact["filing_key"].isin(set(counts.index[counts >= MIN_USABLE_CHUNKS]))]
        if not gated.empty:
            gated_df = _firm_frame_from_windows(gated, measure)
            summary["gradient_if_gate_applied"] = gradient_small_minus_large(gated_df)
            summary["n_filings_if_gate_applied"] = int(gated["filing_key"].nunique())
            summary["n_firms_if_gate_applied"] = int(gated["ticker"].nunique())
            summary["sign_agrees_with_primary_if_gate_applied"] = bool(
                np.isfinite(summary["gradient_if_gate_applied"]) and primary_sign != 0
                and np.sign(summary["gradient_if_gate_applied"]) == primary_sign)
    else:
        summary["filings_below_min_usable_chunks_gate"] = sorted(base_filings)
        summary["n_filings_below_min_usable_chunks_gate"] = int(len(base_filings))
        summary["any_filing_below_gate"] = True

    # -- (ii) word_count as a covariate ------------------------------------
    y = w[measure].to_numpy(dtype=float)
    is_mid = (w["tier"].astype(str) == "mid").to_numpy(dtype=float)
    is_small = (w["tier"].astype(str) == "small").to_numpy(dtype=float)
    logwords = np.log(w["word_count"].to_numpy(dtype=float))
    X = np.column_stack([np.ones_like(y), is_mid, is_small, logwords])
    ols = _ols(y, X, ["intercept", "tier_mid", "tier_small", "log_word_count"],
               clusters=w["ticker"].to_numpy())
    ols["specification"] = (
        f"{measure}_window ~ 1 + tier_mid + tier_small + log(word_count); "
        "reference tier = large; unit = window")
    small_coef = ols["coefficients"]["tier_small"]["coef"]
    ols["sign_agrees_with_primary"] = bool(
        np.isfinite(small_coef) and primary_sign != 0 and np.sign(small_coef) == primary_sign)

    return {
        "section": section, "measure": measure, "primary_sign": int(primary_sign),
        "exact_window_words": int(exact_words),
        "exact_windows_only": summary,
        "word_count_covariate_ols": ols,
        "criterion": "directional agreement with the primary",
        "criterion_met": bool(summary.get("sign_agrees_with_primary", False)
                              and ols["sign_agrees_with_primary"]),
    }


# ===========================================================================
# Orchestration
# ===========================================================================

def run_battery(
    chunk_df: pd.DataFrame,
    section: str = "mda",
    measure: str = "embedding_disagreement",
    seed: int = 20261004,
    w3_iterations: int = W3_ITERATIONS,
    w6_iterations: int = W6_ITERATIONS,
    n_permutations: int = PERMUTATIONS,
    w3_k_values: Sequence[int] = W3_K_VALUES,
) -> dict[str, Any]:
    """
    Run W1-W4 and W6-W7 on a cached window-level frame.

    W5 (the window-level mixed model) belongs to another module and is neither
    run nor imported here; the battery's own gate therefore reports the W3 and
    W2 halves of the ANALYSIS_PLAN.md section 5 overall gate and says plainly
    that W5 is outstanding.

    Every number in the returned structure is computed from `chunk_df`.  The
    seed is recorded in the output and every random procedure derives its
    generator from it.
    """
    missing = check_schema(chunk_df)
    if missing:
        raise ValueError(f"chunk frame is missing required columns: {missing}")

    primary = primary_result(chunk_df, section, measure, seed, n_permutations)
    sign = primary["gradient_sign"]
    if sign == 0:
        logger.warning(
            "run_battery: the primary gradient is exactly 0 or undefined; sign-based "
            "criteria cannot be evaluated and are reported as not met")

    result: dict[str, Any] = {
        "meta": {
            "module": "analysis.window_battery",
            "tests": ["W1", "W2", "W3", "W4", "W6", "W7"],
            "w5_note": ("W5 (window-level mixed model) is implemented in a separate "
                        "module and is deliberately not imported here"),
            "section": section,
            "measure": measure,
            "seed": int(seed),
            "w3_iterations": int(w3_iterations),
            "w6_iterations": int(w6_iterations),
            "n_permutations": int(n_permutations),
            "aggregation": ("window -> filing (unweighted mean over that filing's windows "
                            "of the analysed section) -> firm (unweighted mean over the "
                            "firm's filings) -> tier"),
            "confirmatory_unit": "firm",
            "failed_window_rule": ("any_failed windows are DROPPED, never imputed "
                                   "(ANALYSIS_PLAN.md s8 rule 1)"),
            "usability_gate_note": (
                "the ANALYSIS_PLAN.md s8 rule 2 gate (>= 2 surviving windows AND >= half "
                "surviving) governs the PRIMARY pipeline and is applied there by "
                "aggregate_to_firm.  The battery's variants deliberately do not re-apply "
                "it, because W4(ii) is defined to leave a 2-window filing with 1 window "
                "and W7(i) is defined to exclude tails; re-gating would silently narrow "
                "the sensitivity checks.  Each variant reports its own filing and firm "
                "counts, and W7(i) also reports the gradient with the gate applied."),
            "tier_rank_convention": "large=3, mid=2, small=1",
            "sibling_provenance": SIBLING_PROVENANCE,
            "network_calls": 0,
            "model_calls": 0,
        },
        "primary": primary,
        "W1": w1_descriptives(chunk_df, section, measure),
        "W2": w2_matched(chunk_df, section, measure, sign),
        "W3": w3_equal_k(chunk_df, section, measure, seed,
                         k_values=w3_k_values, n_iterations=w3_iterations,
                         primary_sign=sign),
        "W4": w4_position(chunk_df, section, measure, sign),
        "W6": w6_null_simulation(chunk_df, section, measure, seed,
                                 n_iterations=w6_iterations,
                                 observed_gradient=primary["gradient_small_minus_large"]),
        "W7": w7_window_size(chunk_df, section, measure, sign),
    }
    result["gate"] = {
        "rule": ("ANALYSIS_PLAN.md s5: the tier result is reported as SUPPORTED only if it "
                 "passes W3 and W5 and is not contradicted in an overlapping W2 bin"),
        "W3_passed": bool(result["W3"]["criterion_met"]),
        "W2_not_contradicted": bool(result["W2"]["criterion_met"]),
        "W2_criterion_vacuous": bool(result["W2"].get("criterion_vacuous", False)),
        "W5_passed": None,
        "W5_status": "not evaluated here - separate module",
        "battery_half_of_gate_passed": bool(
            result["W3"]["criterion_met"] and result["W2"]["criterion_met"]),
        "W4_directional_agreement": bool(result["W4"]["criterion_met"]),
        "W7_directional_agreement": bool(result["W7"]["criterion_met"]),
        "verdict_pending_W5": (
            "W3 and W2 passed; final verdict requires W5"
            if result["W3"]["criterion_met"] and result["W2"]["criterion_met"]
            else "NOT ROBUST TO WINDOW COUNT on the battery's own tests - "
                 "ANALYSIS_PLAN.md s7 clause 2 applies"),
    }
    return result


# ===========================================================================
# Reporting
# ===========================================================================

def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return _to_jsonable(obj.to_dict(orient="records"))
    if obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    return str(obj)


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "n/a" if not np.isfinite(float(v)) else f"{float(v):.{nd}f}"
    return str(v)


def _tier_line(tm: dict[str, dict[str, float]]) -> str:
    return "  ".join(
        f"{t}: {_fmt(tm[t]['mean'])} (sd {_fmt(tm[t]['sd'])}, n {tm[t]['n']})"
        for t in TIER_ORDER)


def format_report(result: dict[str, Any]) -> str:
    """Render the battery as the human-readable report.txt body."""
    m, p = result["meta"], result["primary"]
    L: list[str] = []
    add = L.append

    add("=" * 78)
    add("WINDOW-COUNT BATTERY - W1-W4, W6-W7 (ANALYSIS_PLAN.md section 5)")
    add("=" * 78)
    add(f"section            : {m['section']}")
    add(f"measure            : {m['measure']}")
    add(f"seed               : {m['seed']}   (W3 {m['w3_iterations']} iters, "
        f"W6 {m['w6_iterations']} iters, perm {m['n_permutations']} draws)")
    add(f"aggregation        : {m['aggregation']}")
    add(f"confirmatory unit  : {m['confirmatory_unit']}")
    add(f"failed windows     : {m['failed_window_rule']}")
    add(f"usability gate     : {m['usability_gate_note']}")
    add(f"network/model calls: {m['network_calls']}/{m['model_calls']}")
    add("sibling provenance : " + "; ".join(f"{k}={v}" for k, v in m["sibling_provenance"].items()))
    add(f"W5                 : {m['w5_note']}")
    add("")

    add("-" * 78)
    add("PRIMARY (the number every test below is compared against)")
    add("-" * 78)
    add(f"windows in section {p['n_windows_in_section']}, dropped for failure "
        f"{p['n_windows_dropped_failed']}, used {p['n_windows_used']}")
    add(f"filings {p['n_filings']}, firms {p['n_firms']}  "
        f"(tier firm counts {p['tier_firm_counts']})")
    add("firm-level tier means: " + _tier_line(p["tier_means"]))
    add(f"gradient (Small - Large) = {_fmt(p['gradient_small_minus_large'])}   "
        f"sign {p['gradient_sign']:+d}   "
        f"monotone Small>Mid>Large: {_fmt(p['monotone_small_gt_mid_gt_large'])}")
    jt = p["jonckheere_terpstra"]
    add(f"C1  Jonckheere-Terpstra: JT = {_fmt(jt.get('statistic'), 1)}, "
        f"z = {_fmt(jt.get('z'), 3)}, one-sided p = {_fmt(jt.get('p_value'))}, "
        f"n = {_fmt(jt.get('n'), 0)}")
    pm = p["label_permutation"]
    add(f"C1-perm label permutation of {p['label_permutation_statistic']}:")
    add(f"      observed {_fmt(pm.get('observed'), 1)}, one-sided p = "
        f"{_fmt(pm.get('p_value'))} "
        f"({pm.get('n_draws_used')} draws, seed {pm.get('seed')})")
    gp = p["gradient_permutation"]
    add(f"label permutation of the {p['gradient_permutation_statistic']}:")
    add(f"      observed {_fmt(gp.get('observed'))}, one-sided p = "
        f"{_fmt(gp.get('p_value'))}, null mean {_fmt(gp.get('null_mean'))}, "
        f"null sd {_fmt(gp.get('null_sd'))}")
    add(f"tier_trend_statistic (JT J) = {_fmt(p['tier_trend_statistic_J'], 1)}  "
        f"[{p['tier_trend_statistic_note']}]")
    add(f"gradient definition: {p['gradient_definition']}")
    add("")

    # ---- W1 ----
    w1 = result["W1"]
    add("-" * 78)
    add("W1  DESCRIPTIVES  [descriptive - no pass/fail]")
    add("-" * 78)
    add("windows per filing, by tier:")
    for t in TIER_ORDER:
        d = w1["windows_per_filing_by_tier"][t]
        add(f"   {t:<6} n={d['n']:<3} mean {_fmt(d['mean'],2)}  median {_fmt(d['median'],2)}  "
            f"sd {_fmt(d['sd'],2)}  min {_fmt(d['min'],0)}  max {_fmt(d['max'],0)}")
    add("windows per firm (total across the firm's filings), by tier:")
    for t in TIER_ORDER:
        d = w1["windows_per_firm_by_tier"]["total"][t]
        add(f"   {t:<6} n={d['n']:<3} mean {_fmt(d['mean'],2)}  median {_fmt(d['median'],2)}  "
            f"sd {_fmt(d['sd'],2)}  min {_fmt(d['min'],0)}  max {_fmt(d['max'],0)}")
    add("mean window size (words), by tier, filing level:")
    for t in TIER_ORDER:
        d = w1["mean_window_words_by_tier"]["filing_level"][t]
        add(f"   {t:<6} n={d['n']:<3} mean {_fmt(d['mean'],1)}  median {_fmt(d['median'],1)}  "
            f"min {_fmt(d['min'],0)}  max {_fmt(d['max'],0)}")
    add(f"tier rank convention: {w1['tier_rank_convention']}")
    for level in ("filing_level", "firm_level"):
        add(f"Spearman ({level.replace('_',' ')}):")
        for name, s in w1["spearman"][level].items():
            add(f"   {name:<34} rho = {_fmt(s['rho'],3):>7}   "
                f"p = {_fmt(s['p_value'],4):>7}   n = {s['n']}")
    add("")

    # ---- W2 ----
    w2 = result["W2"]
    add("-" * 78)
    add("W2  MATCHED COMPARISON  [criterion: primary sign not CONTRADICTED in any "
        "all-three-tier bin]")
    add("-" * 78)
    add(f"unit of analysis inside a bin: {w2['unit']} "
        "(a firm's two filings can fall in different bins)")
    for label, d in w2["bins"].items():
        comp = d["tier_composition"]
        add(f" bin {label:<4} n={d['n_filings']:<3} composition "
            f"large/mid/small = {comp['large']}/{comp['mid']}/{comp['small']}")
        add(f"          {d['identification_note']}")
        add(f"          tier means: {_tier_line(d['tier_means'])}")
        ci = d["gradient_ci"]
        add(f"          Small-Large = {_fmt(d['gradient_small_minus_large'])}  "
            f"95% CI [{_fmt(ci['low'])}, {_fmt(ci['high'])}]  "
            f"excludes 0: {_fmt(ci['excludes_zero'])}  "
            f"contradicts primary: {_fmt(d['contradicts_primary'])}")
    add(f" bins identified for a three-tier comparison : {w2['bins_identified_for_three_tier']}")
    add(f" bins NOT identified (labelled, not averaged): {w2['bins_not_identified']}")
    if w2["filings_outside_bins"]["n"]:
        add(f" filings outside the pre-registered bins (<2 windows): "
            f"{w2['filings_outside_bins']['n']} {w2['filings_outside_bins']['filing_keys']}")
    cem = w2["coarsened_exact_matching"]
    add(" coarsened exact matching:")
    add(f"   coarsening  : {cem['coarsening']}")
    add(f"   estimand    : {cem['estimand']}")
    add(f"   strata kept : {cem['n_strata_retained']}")
    for s in cem["strata_retained"]:
        add(f"      ({s['bin']}, {s['filing_type']})  n_small={s['n_small']} "
            f"n_large={s['n_large']}  diff = {_fmt(s['diff'])}")
    add(f"   filings discarded for lack of overlap : "
        f"{cem['n_filings_discarded_no_overlap']}")
    for s in cem["strata_discarded"]:
        add(f"      ({s['bin']}, {s['filing_type']})  n_small={s['n_small']} "
            f"n_large={s['n_large']}  - {s['reason']}")
    add(f"   mid-cap filings outside the contrast  : "
        f"{cem['n_mid_filings_outside_contrast']}")
    add(f"   weighted Small-Large = {_fmt(cem['weighted_small_minus_large'])}  "
        f"(unweighted stratum mean {_fmt(cem['unweighted_stratum_mean_difference'])})  "
        f"sign agrees: {_fmt(cem['sign_agrees_with_primary'])}")
    add(f" W2 CRITERION MET: {_fmt(w2['criterion_met'])}"
        + ("  [VACUOUSLY]" if w2.get("criterion_vacuous") else "")
        + (f"   contradicting bins: {w2['contradicting_bins']}"
           if w2["contradicting_bins"] else ""))
    add(f"   {w2['criterion_note']}")
    add("")

    # ---- W3 ----
    w3 = result["W3"]
    add("-" * 78)
    add("W3  EQUAL-k SUBSAMPLING  [THE DECISIVE TEST]")
    add("-" * 78)
    add(f"criterion: {w3['criterion']}")
    for key, d in w3["by_k"].items():
        add(f" {key}  ({d['restriction']})")
        if "error" in d:
            add(f"      ERROR: {d['error']}")
            continue
        add(f"      filings used {d['n_filings_used']}, dropped {d['n_filings_dropped']}"
            + (f" {d['filings_dropped']}" if d["n_filings_dropped"] else ""))
        add(f"      firms used {d['n_firms_used']}, dropped {d['n_firms_dropped']}"
            + (f" {d['firms_dropped']}" if d["n_firms_dropped"] else "")
            + f", tier firm counts {d['tier_firm_counts']}")
        add(f"      filings whose k-draw is deterministic (n == k): "
            f"{d['n_filings_whose_draw_is_deterministic']}")
        g = d["gradient_distribution"]
        add(f"      gradient: mean {_fmt(g['mean'])}  sd {_fmt(g['sd'])}  "
            f"median {_fmt(g['median'])}  min {_fmt(g['min'])}  max {_fmt(g['max'])}")
        pc = g["percentiles"]
        add("      percentiles: " + "  ".join(
            f"{q}={_fmt(pc[f'p{q}'])}" for q in (1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99)))
        add(f"      observed gradient on this subset: "
            f"{_fmt(d['observed_gradient_on_this_subset'])}")
        add(f"      share primary sign holds           : "
            f"{d['share_primary_sign_holds']:.4f}")
        add(f"      share ordering Small>Mid>Large     : "
            f"{d['share_ordering_small_gt_mid_gt_large']:.4f}")
        if d["criterion_applies"]:
            add(f"      >>> PRE-DECLARED CRITERION MET: "
                f"{'YES' if d['criterion_met'] else 'NO'} "
                f"({d['share_primary_sign_holds']:.1%} vs required "
                f"{W3_SIGN_SHARE_CRITERION:.0%})")
        else:
            add(f"      criterion not applicable: {d['criterion_note']}")
    add(f" W3 CRITERION MET (k=2): {'YES' if w3['criterion_met'] else 'NO'}   "
        f"share = {_fmt(w3['share_primary_sign_holds_k2'])}")
    add("")

    # ---- W4 ----
    w4 = result["W4"]
    add("-" * 78)
    add("W4  POSITION CONTROL  [criterion: directional agreement with the primary]")
    add("-" * 78)
    for name, d in w4["variants"].items():
        add(f" {name}: {d['note']}")
        if "error" in d:
            add(f"      ERROR: {d['error']}")
            continue
        add(f"      windows {d['n_windows']}, filings {d['n_filings']} "
            f"(dropped {d['n_filings_dropped']}), firms {d['n_firms']} "
            f"(dropped {d['n_firms_dropped']})")
        if d["n_filings_dropped"]:
            add(f"      filings dropped: {d['filings_dropped']}")
        if "n_filings_reduced_to_one_window" in d:
            add(f"      filings reduced to a single window: "
                f"{d['n_filings_reduced_to_one_window']}; one-window sections "
                f"contributing zero: {d['n_one_window_sections_contributing_zero']}")
            add(f"      {d['note_on_dropout']}")
        add(f"      tier means: {_tier_line(d['tier_means'])}")
        add(f"      gradient {_fmt(d['gradient_small_minus_large'])}  "
            f"monotone {_fmt(d['monotone_small_gt_mid_gt_large'])}  "
            f"sign agrees with primary: {_fmt(d['sign_agrees_with_primary'])}")
    pd_ = w4["position_decile"]
    add(f" position_decile = {pd_['formula']}  (exposed by {pd_['exposed_by']})")
    for t in TIER_ORDER:
        d = pd_["by_tier"][t]
        add(f"      {t:<6} mean {_fmt(d['mean'],3)}  median {_fmt(d['median'],3)}  n {d['n']}")
    add("      share of windows that are section-opening, by tier: "
        + "  ".join(f"{k}={_fmt(v,3)}" for k, v in
                    pd_["share_of_windows_that_are_section_opening_by_tier"].items()))
    s = pd_["spearman_position_vs_measure"]
    add(f"      Spearman(position_decile, {w4['measure']}) rho = {_fmt(s['rho'],3)}, "
        f"p = {_fmt(s['p_value'])}, n = {s['n']}")
    add(f" W4 CRITERION MET: {_fmt(w4['criterion_met'])}")
    add("")

    # ---- W6 ----
    w6 = result["W6"]
    add("-" * 78)
    add("W6  NULL SIMULATION  [no pass/fail - quantifies the advisor's concern]")
    add("-" * 78)
    if "error" in w6:
        add(f" ERROR: {w6['error']}")
    else:
        add(" HELD FIXED:")
        for x in w6["held_fixed"]:
            add(f"   - {x}")
        add(" RANDOMISED:")
        for x in w6["randomised"]:
            add(f"   - {x}")
        pool = w6["pooled_distribution"]
        add(f" pooled window distribution: n {pool['n']}  mean {_fmt(pool['mean'])}  "
            f"sd {_fmt(pool['sd'])}  min {_fmt(pool['min'])}  max {_fmt(pool['max'])}")
        wc = w6["window_count_summary"]
        add(f" window counts held fixed: n_filings {wc['n']}  mean {_fmt(wc['mean'],2)}  "
            f"median {_fmt(wc['median'],1)}  min {_fmt(wc['min'],0)}  max {_fmt(wc['max'],0)}")
        g = w6["null_gradient"]
        add(f" null gradient: mean {_fmt(g['mean'])}  sd {_fmt(g['sd'])}  "
            f"min {_fmt(g['min'])}  max {_fmt(g['max'])}")
        pc = g["percentiles"]
        add("   percentiles: " + "  ".join(
            f"{q}={_fmt(pc[f'p{q}'])}" for q in (1, 2.5, 5, 25, 50, 75, 95, 97.5, 99)))
        add(f"   95th percentile of |null gradient| = {_fmt(w6['null_abs_gradient_p95'])}")
        add(f"   {w6['interpretation']}")
        add(f" null share ordering Small>Mid>Large = "
            f"{_fmt(w6['null_share_ordering_small_gt_mid_gt_large'])}")
        add(f" OBSERVED gradient {_fmt(w6['observed_gradient'])} "
            f"[{w6['observed_gradient_source']}] sits at the "
            f"{_fmt(w6['observed_percentile_in_null'],2)}th percentile of this null")
        if w6["observed_gradient"] != w6["observed_gradient_on_w6_window_set"]:
            add(f"   (gradient recomputed on W6's own ungated window set: "
                f"{_fmt(w6['observed_gradient_on_w6_window_set'])})")
        add(f" one-sided p = {_fmt(w6['p_value_one_sided'])}   "
            f"[{w6['p_value_convention']}]")
    add("")

    # ---- W7 ----
    w7 = result["W7"]
    add("-" * 78)
    add("W7  WINDOW-SIZE CHECK  [criterion: directional agreement with the primary]")
    add("-" * 78)
    e = w7["exact_windows_only"]
    add(f" (i) {e['note']}")
    if "error" in e:
        add(f"      ERROR: {e['error']}")
    else:
        add(f"      windows kept {e['n_windows']}, dropped {e['n_windows_dropped']}; "
            f"filings {e['n_filings']} (dropped {e['n_filings_dropped']}), "
            f"firms {e['n_firms']} (dropped {e['n_firms_dropped']})")
        if e["n_filings_dropped"]:
            add(f"      filings dropped: {e['filings_dropped']}")
        add(f"      filings now below the MIN_USABLE_CHUNKS={MIN_USABLE_CHUNKS} gate: "
            f"{e.get('n_filings_below_min_usable_chunks_gate')} "
            f"{e.get('filings_below_min_usable_chunks_gate')}")
        if "gate_note" in e:
            add(f"      {e['gate_note']}")
        add(f"      tier means: {_tier_line(e['tier_means'])}")
        add(f"      gradient {_fmt(e['gradient_small_minus_large'])}  "
            f"sign agrees: {_fmt(e['sign_agrees_with_primary'])}")
        if "gradient_if_gate_applied" in e:
            add(f"      gradient if the 2-window gate IS applied: "
                f"{_fmt(e['gradient_if_gate_applied'])} "
                f"({e['n_filings_if_gate_applied']} filings, "
                f"{e['n_firms_if_gate_applied']} firms), sign agrees: "
                f"{_fmt(e['sign_agrees_with_primary_if_gate_applied'])}")
    o = w7["word_count_covariate_ols"]
    add(f" (ii) {o['specification']}")
    add(f"      n = {o['n_observations']}, df_resid = {o['df_residual']}, "
        f"R2 = {_fmt(o['r_squared'])}, firm clusters = {o['n_clusters_firm']}")
    for nm, c in o["coefficients"].items():
        add(f"      {nm:<16} coef {_fmt(c['coef']):>9}  se {_fmt(c['se']):>8}  "
            f"p {_fmt(c['p_value']):>8}  95% CI [{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}]"
            f"  excl 0: {_fmt(c['ci_excludes_zero'])}")
        if "se_cluster_firm" in c:
            add(f"      {'':<16} firm-clustered se {_fmt(c['se_cluster_firm'])}  "
                f"95% CI [{_fmt(c['ci_low_cluster_firm'])}, "
                f"{_fmt(c['ci_high_cluster_firm'])}]  excl 0: "
                f"{_fmt(c['ci_excludes_zero_cluster_firm'])}")
    add(f"      {o['se_note']}")
    add(f"      tier_small coefficient sign agrees with primary: "
        f"{_fmt(o['sign_agrees_with_primary'])}")
    add(f" W7 CRITERION MET: {_fmt(w7['criterion_met'])}")
    add("")

    # ---- gate ----
    g = result["gate"]
    add("=" * 78)
    add("GATE (ANALYSIS_PLAN.md section 5)")
    add("=" * 78)
    add(f" {g['rule']}")
    add(f"   W3 passed                        : {_fmt(g['W3_passed'])}")
    add(f"   W2 not contradicted              : {_fmt(g['W2_not_contradicted'])}"
        + ("  [vacuous: no all-three-tier bin]" if g.get("W2_criterion_vacuous") else ""))
    add(f"   W5                               : {g['W5_status']}")
    add(f"   W4 directional agreement         : {_fmt(g['W4_directional_agreement'])}")
    add(f"   W7 directional agreement         : {_fmt(g['W7_directional_agreement'])}")
    add(f"   battery half of the gate passed  : "
        f"{_fmt(g['battery_half_of_gate_passed'])}")
    add(f"   {g['verdict_pending_W5']}")
    add("=" * 78)
    return "\n".join(L) + "\n"


def write_report(
    result: dict[str, Any],
    out_dir: Path | str | None = None,
) -> dict[str, Path]:
    """
    Write the battery to
        output/experiments/window_battery/report.txt   (human-readable)
        output/experiments/window_battery/report.json  (every number, machine-readable)

    numbers.tex is generated from report.json; nothing is transcribed by hand
    (ANALYSIS_PLAN.md section 8 rule 6).
    """
    d = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    d.mkdir(parents=True, exist_ok=True)
    txt, js = d / "report.txt", d / "report.json"
    txt.write_text(format_report(result), encoding="utf-8")
    js.write_text(json.dumps(_to_jsonable(result), indent=2, sort_keys=False, allow_nan=False)
                  + "\n", encoding="utf-8")
    logger.info("window battery written to %s and %s", txt, js)
    return {"report_txt": txt, "report_json": js}


# ===========================================================================
# CLI
# ===========================================================================

def load_chunk_results(path: Path | str) -> pd.DataFrame:
    """Read a chunk-level results CSV and validate its schema."""
    df = pd.read_csv(path, dtype={"filing_date": str})
    missing = check_schema(df)
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    return df


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="output/chunk_level_results.csv")
    ap.add_argument("--section", default="mda", choices=["mda", "risk"])
    ap.add_argument("--measure", default="embedding_disagreement",
                    choices=["embedding_disagreement", "numeric_disagreement"])
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--w3-iterations", type=int, default=W3_ITERATIONS)
    ap.add_argument("--w6-iterations", type=int, default=W6_ITERATIONS)
    ap.add_argument("--permutations", type=int, default=PERMUTATIONS)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    path = Path(args.input)
    if not path.exists():
        logger.error("%s does not exist. The scoring run has not produced it yet; "
                     "run the battery against tests/fixtures.py output to exercise it.", path)
        return 2
    df = load_chunk_results(path)
    res = run_battery(df, section=args.section, measure=args.measure, seed=args.seed,
                      w3_iterations=args.w3_iterations, w6_iterations=args.w6_iterations,
                      n_permutations=args.permutations)
    paths = write_report(res, out_dir=args.out_dir)
    print(format_report(res))
    print(f"written: {paths['report_txt']}\n         {paths['report_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
