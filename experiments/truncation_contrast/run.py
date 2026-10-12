"""
S1(a) — the truncation contrast.  EXPLORATORY (ANALYSIS_PLAN.md §6).

The question
------------
Standard practice in LLM studies of 10-K text is to truncate each filing to a
fixed budget — the first N words, or whatever fits a context window.  Because
larger firms file longer narrative sections, a fixed budget reads a smaller
*fraction* of a large firm's filing.  How much the model read therefore becomes a
function of firm size: the very variable such studies regress on.

In this project's own sample the retired 2,000-word cap truncated 59 of 59
filings, and Spearman(tier rank [large=3], fraction of the narrative sections
read) = -0.415, p = 0.0011.  That establishes that coverage was *confounded* with
size.  This experiment asks the next question: does it change the answer?

What this does
--------------
It re-measures the tier gradient as it would have been measured if the model had
read only the FIRST k windows of each filing's MD&A, for k = 1, 2, 4, 8 and all.
Same model, same filings, same scored text, same aggregation — coverage is the
only thing that varies.

Costs nothing: it reads `output/chunk_level_results.csv` and makes no model calls,
no network calls, and no writes outside its own output directory.

THE HONESTY CONSTRAINT — read this before quoting any number from here
----------------------------------------------------------------------
This is a SIMULATION of truncation over windows that were each scored in
isolation.  It is NOT the same thing as a model reading a truncated document.  A
persona scoring window 1 of 25 never saw windows 2-25, so there is no
long-context degradation, no carry-over of earlier content, and no difference in
the prompt.  What varies here is only *which already-scored windows enter the
average*.  The exact replication without that caveat — re-scoring each filing's
first 2,000 words of MD&A as one document under the old cap, ~180 calls — is a
separate experiment, S1(b), and is not run here.

Aggregation (pre-registered, ANALYSIS_PLAN.md §1)
-------------------------------------------------
    window -> filing   unweighted mean over that filing's MD&A windows inside the
                       budget that did not fail
    filing -> firm     unweighted mean over the firm's filings
    firm   -> tier     unweighted mean over the tier's firms
The unit of analysis is the FIRM (n = 30).  MD&A only; Risk Factors is never
pooled into the primary measure (§2).  Failed windows are dropped, never imputed
(§8.1).  The numeric measure is reported first in every table (§3), whatever it
shows; the embedding measure is the primary one.

Two budget rules, both reported
-------------------------------
  inclusive  every filing contributes min(k, its window count) windows.  A filing
             with fewer than k windows contributes all of them.  This is what a
             fixed word budget actually does to a short filing: it reads all of it.
  strict     only filings with at least k windows contribute at all.  This holds
             the per-filing window count at exactly k, at the cost of dropping
             filings — and the drops are not tier-neutral, because short filings
             are disproportionately small-cap.  The gap between the two variants
             is itself informative and is reported, never averaged away.

k = 1 is handled explicitly: see `--help` and the README.

Usage
-----
    python3 experiments/truncation_contrast/run.py
    python3 experiments/truncation_contrast/run.py --input /path/to/fixture.csv \
        --out-dir /tmp/tc --n-boot 2000 --seed 20261004
"""

from __future__ import annotations

import argparse
import logging
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# Documented abseil/TensorFlow mutex deadlock on this machine (CLAUDE.md): the
# guard must be in place before anything can reach `transformers`.  This module
# imports no transformers and no model code at all, but the project rule is
# unconditional and costs nothing here.
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None  # type: ignore[assignment]

logger = logging.getLogger("truncation_contrast")

# ── Pre-registered constants ──────────────────────────────────────────────────
TIER_ORDER: tuple[str, ...] = ("large", "mid", "small")
# Tier rank with large = 3, as used in output/coverage_diagnostic.txt, so the
# sign of Spearman(tier rank, fraction read) is comparable to the -0.415 already
# reported for the retired 2,000-word cap.
TIER_RANK: dict[str, int] = {"large": 3, "mid": 2, "small": 1}

PRIMARY_SECTION = "mda"
# Numeric first (ANALYSIS_PLAN.md §3), embedding second and primary.
MEASURES: tuple[str, ...] = ("numeric_disagreement", "embedding_disagreement")
PRIMARY_MEASURE = "embedding_disagreement"
DEFAULT_BUDGETS: tuple[int | None, ...] = (1, 2, 4, 8, None)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "ticker", "tier", "filing_type", "filing_date", "section", "chunk_index",
    "n_in_section", "word_count", "numeric_disagreement", "embedding_disagreement",
    "any_failed",
)

# ANALYSIS_PLAN.md §8.2.  Imported rather than restated so there is one definition
# of the usability gate in the project; the literal is only a fallback for the
# case where the analysis package cannot be imported (e.g. the file is copied out).
try:  # pragma: no cover - import shape, not logic
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from analysis.disagreement import MIN_USABLE_CHUNKS  # noqa: E402
except Exception:  # pragma: no cover
    MIN_USABLE_CHUNKS = 2

# Colourblind-safe, greyscale-legible pair (validated: protan dE 25.5, tritan
# 36.9, normal-vision dE 37.4, both >= 3:1 on a white surface).  Identity is
# carried by linestyle and marker as well as hue, so the figure survives
# greyscale printing and photocopying.
C_INCLUSIVE = "#184f95"
C_STRICT = "#eb6834"
C_TIER = {"large": "#184f95", "mid": "#eb6834", "small": "#1baf7a"}
C_REF = "#52514e"
C_GRID = "#d8d7d4"


def k_label(k: int | None) -> str:
    return "all" if k is None else str(k)


# ── Loading ───────────────────────────────────────────────────────────────────

def _to_bool(s: pd.Series) -> pd.Series:
    """
    Robust bool coercion for `any_failed`.

    pandas reads a clean column of True/False as bool, but one NaN turns it into
    object, and a round-trip through CSV can leave the strings "True"/"False".
    Anything unrecognised is treated as NOT failed; the null-disagreement check
    below is the independent second gate, so a mis-parsed flag cannot smuggle a
    failed window into an average.
    """
    if s.dtype == bool:
        return s
    return (
        s.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes"})
    )


def prepare(df: pd.DataFrame, source: str = "<frame>") -> pd.DataFrame:
    """
    Validate and normalise a window-level frame.

    Idempotent, and applied both to what is read from disk and to anything handed
    to `run_truncation_contrast` directly, so a caller that builds a frame in
    memory (a test, a notebook) goes through exactly the same checks as a caller
    that reads the real CSV.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{source} is missing required column(s): {missing}")

    df = df.copy()
    df["any_failed"] = _to_bool(df["any_failed"])
    for c in ("chunk_index", "n_in_section", "word_count"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for m in MEASURES:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    # A filing is identified by (ticker, form, date): a firm contributes a 10-K
    # and a 10-Q, and they must never be merged at the window level.
    df["filing_id"] = (
        df["ticker"].astype(str) + "|" + df["filing_type"].astype(str) + "|"
        + df["filing_date"].astype(str)
    )

    bad_tiers = sorted(set(df["tier"].dropna().unique()) - set(TIER_ORDER))
    if bad_tiers:
        raise SystemExit(f"unexpected tier value(s) in {source}: {bad_tiers}")
    return df


def load_window_level(path: Path) -> pd.DataFrame:
    """Read `chunk_level_results.csv` and normalise the columns this code uses."""
    if not path.exists():
        raise SystemExit(
            f"Window-level output not found: {path}\n"
            "This experiment reads cached output only and makes no model calls. "
            "Run the scoring pipeline first, or pass --input with a fixture "
            "(see tests/fixtures_truncation.py)."
        )
    return prepare(pd.read_csv(path, dtype={"filing_date": str}), source=str(path))


# ── Filing-level preparation ──────────────────────────────────────────────────

def section_frame(df: pd.DataFrame, section: str = PRIMARY_SECTION) -> pd.DataFrame:
    """Rows of one section only.  Risk Factors is never pooled in (§2)."""
    return df[df["section"] == section].copy()


def _is_usable_window(df: pd.DataFrame) -> pd.Series:
    """A window counts only if no persona failed AND both measures are present."""
    ok = ~df["any_failed"]
    for m in MEASURES:
        ok &= df[m].notna()
    return ok


def filing_inventory(sec: pd.DataFrame) -> pd.DataFrame:
    """
    One row per filing: window count, total section words, surviving windows, and
    whether it passes the pre-registered usability gate on its COMPLETE section.

    The gate (§8.2: >= MIN_USABLE_CHUNKS surviving windows and >= half surviving)
    is evaluated once, here, on the complete section — it is a data-quality
    statement about how the filing was scored, not about a simulated budget.  It
    is therefore NOT re-applied inside each truncation budget.  That is what makes
    k = 1 analysable at all: re-applying a ">= 2 windows" gate inside a 1-window
    budget would drop every filing by construction.  For transparency the count of
    filings that *would* fail a within-budget gate is reported at every k, so the
    stricter reading of the rule can be checked without changing the estimate.
    """
    usable = _is_usable_window(sec)
    g = sec.assign(_usable=usable).groupby("filing_id", sort=True)
    inv = g.agg(
        ticker=("ticker", "first"),
        tier=("tier", "first"),
        filing_type=("filing_type", "first"),
        n_windows=("chunk_index", "size"),
        section_words=("word_count", lambda s: int(pd.to_numeric(s).sum())),
        n_usable=("_usable", "sum"),
    ).reset_index()
    inv["n_usable"] = inv["n_usable"].astype(int)

    # Integrity check: the number of rows per filing must equal the `n_in_section`
    # the chunker recorded.  A mismatch means the window-level file is incomplete
    # (a run interrupted mid-filing writes the rows it has), and every mean built
    # on it would be computed over a truncated section without anyone noticing —
    # which is the exact failure mode this experiment exists to study.
    declared = sec.groupby("filing_id")["n_in_section"].max()
    mismatch = inv.assign(declared=inv["filing_id"].map(declared))
    mismatch = mismatch[mismatch["declared"].notna()
                        & (mismatch["declared"] != mismatch["n_windows"])]
    if not mismatch.empty:
        logger.warning(
            "window-count mismatch in %d filing(s): rows present != n_in_section "
            "(%s). The window-level file looks incomplete; results below are "
            "computed on the rows that are there.",
            len(mismatch),
            ", ".join(
                f"{r.filing_id}: {r.n_windows} rows vs n_in_section={int(r.declared)}"
                for r in mismatch.itertuples()
            ),
        )
    inv["n_windows_declared"] = inv["filing_id"].map(declared)

    inv["passes_gate"] = (
        (inv["n_usable"] >= MIN_USABLE_CHUNKS) & (inv["n_usable"] >= inv["n_windows"] / 2)
    )
    return inv


# ── One budget ────────────────────────────────────────────────────────────────

def filing_means_at_k(
    sec: pd.DataFrame, inv: pd.DataFrame, k: int | None, strict: bool
) -> tuple[pd.DataFrame, dict]:
    """
    Filing-level means under a budget of the first `k` windows.

    Returns `(filings, diag)` where `filings` has one row per contributing filing
    and `diag` records everything that was dropped and why — nothing is bounded
    silently.
    """
    eligible = inv[inv["passes_gate"]]
    scope_ids = set(eligible["filing_id"])
    diag: dict = {
        "k": k,
        "variant": "strict" if strict else "inclusive",
        "n_filings_gate_eligible": int(len(eligible)),
        "n_filings_gate_failed": int((~inv["passes_gate"]).sum()),
    }

    if k is None:
        kept_ids = scope_ids
        diag["n_filings_with_at_least_k"] = int(len(eligible))
        diag["n_filings_short_of_k"] = 0
        diag["n_filings_dropped_by_strict"] = 0
    else:
        has_k = set(eligible.loc[eligible["n_windows"] >= k, "filing_id"])
        diag["n_filings_with_at_least_k"] = int(len(has_k))
        diag["n_filings_short_of_k"] = int(len(scope_ids) - len(has_k))
        if strict:
            kept_ids = has_k
            diag["n_filings_dropped_by_strict"] = int(len(scope_ids) - len(has_k))
        else:
            kept_ids = scope_ids
            diag["n_filings_dropped_by_strict"] = 0

    rows = sec[sec["filing_id"].isin(kept_ids)]
    if k is not None:
        rows = rows[rows["chunk_index"] < k]

    diag["n_windows_in_budget"] = int(len(rows))
    usable_mask = _is_usable_window(rows)
    diag["n_windows_dropped_failed"] = int((~usable_mask).sum())
    good = rows[usable_mask]

    # Words the model would have read at this budget, over words in the complete
    # section.  Failed windows count as read: the failure is in the scoring, not
    # in the coverage, and coverage is what this ratio is about.
    words_in_budget = rows.groupby("filing_id")["word_count"].sum().astype(float)
    total_words = inv.set_index("filing_id")["section_words"].astype(float)

    agg = {m: (m, "mean") for m in MEASURES}
    filings = good.groupby("filing_id", sort=True).agg(
        ticker=("ticker", "first"),
        tier=("tier", "first"),
        filing_type=("filing_type", "first"),
        n_windows_used=("chunk_index", "size"),
        **agg,
    ).reset_index()

    # Within-filing SD (§3, secondary).  Undefined on one window — reported as
    # NaN rather than 0, because "no variation measured" is not "no variation".
    sd = good.groupby("filing_id")[PRIMARY_MEASURE].std(ddof=1)
    filings["within_filing_sd_" + PRIMARY_MEASURE.split("_")[0]] = (
        filings["filing_id"].map(sd)
    )
    filings["frac_section_read"] = filings["filing_id"].map(
        words_in_budget / total_words
    )
    filings["n_windows_in_budget"] = filings["filing_id"].map(
        rows.groupby("filing_id")["chunk_index"].size()
    )

    # How many contributing filings would fail the gate if it were re-applied
    # inside this budget.  Diagnostic only; it does not change the estimate.
    would_fail = int(
        (
            (filings["n_windows_used"] < MIN_USABLE_CHUNKS)
            | (filings["n_windows_used"] < filings["n_windows_in_budget"] / 2)
        ).sum()
    )
    diag["n_filings_failing_within_budget_gate"] = would_fail
    diag["n_filings_contributing"] = int(len(filings))
    diag["n_filings_no_surviving_window"] = int(len(kept_ids) - len(filings))
    diag["n_windows_used"] = int(len(good))
    # Coverage fraction is defined for every filing in scope, including any whose
    # windows all failed, so it is reported on the budget scope.
    diag["frac_read_scope_ids"] = sorted(kept_ids)
    return filings, diag


def firm_means(filings: pd.DataFrame) -> pd.DataFrame:
    """filing -> firm: unweighted mean over the firm's filings (§1)."""
    cols = list(MEASURES) + ["frac_section_read"]
    out = filings.groupby(["ticker", "tier"], sort=True).agg(
        n_filings=("filing_id", "size"), **{c: (c, "mean") for c in cols}
    ).reset_index()
    return out


def tier_stats(firms: pd.DataFrame, measure: str) -> pd.DataFrame:
    """firm -> tier: unweighted mean over the tier's firms (§1)."""
    rows = []
    for tier in TIER_ORDER:
        v = firms.loc[firms["tier"] == tier, measure].dropna().to_numpy(float)
        rows.append({
            "tier": tier,
            "n_firms": int(v.size),
            "mean": float(v.mean()) if v.size else np.nan,
            "sd": float(v.std(ddof=1)) if v.size > 1 else np.nan,
        })
    return pd.DataFrame(rows)


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _rng_for(seed: int, label: str) -> np.random.Generator:
    """
    Deterministic, label-dependent generator.

    Each (measure, variant, budget) cell gets its own stream derived from the one
    user-supplied seed, so results are reproducible and independent of the order
    in which the cells are computed.  CRC32 of the label is a stable spawn key
    across processes and Python versions (unlike `hash`, which is salted).
    """
    return np.random.default_rng(
        np.random.SeedSequence(entropy=int(seed), spawn_key=(zlib.crc32(label.encode()),))
    )


def _draw_indices(
    tier_sizes: dict[str, int], n_boot: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Stratified cluster bootstrap: resample FIRMS with replacement within tier."""
    return {
        t: rng.integers(0, n, size=(n_boot, n)) if n > 0 else np.zeros((n_boot, 0), int)
        for t, n in tier_sizes.items()
    }


def bootstrap_gradient(
    firm_vectors: dict[str, np.ndarray],
    ref_vectors: dict[str, np.ndarray] | None,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """
    Percentile bootstrap CI for the tier gradient (small - large), and for the
    PAIRED difference against a reference budget when one is given.

    The firm is the unit of analysis (§1), so the firm is the resampling unit.
    `ref_vectors` must be aligned firm-for-firm with `firm_vectors`; the same
    resampled firms are then used for both, which is what makes the difference a
    paired comparison — the question "does truncation change the answer" is about
    the same firms read two ways, not about two independent samples.
    """
    sizes = {t: firm_vectors[t].size for t in TIER_ORDER}
    idx = _draw_indices(sizes, n_boot, rng)

    def grad(vectors: dict[str, np.ndarray]) -> np.ndarray:
        means = {}
        for t in TIER_ORDER:
            v = vectors[t]
            means[t] = (
                v[idx[t]].mean(axis=1) if v.size else np.full(n_boot, np.nan)
            )
        return means["small"] - means["large"]

    def pct(a: np.ndarray, q: float) -> float:
        """Percentile that returns NaN for an all-NaN draw instead of warning.

        An all-NaN draw means a tier contributed no firms at this budget — which
        happens under the strict rule at large k — and NaN is the honest answer.
        """
        return float(np.nanpercentile(a, q)) if np.isfinite(a).any() else float("nan")

    def point(v: dict[str, np.ndarray]) -> float:
        if v["small"].size == 0 or v["large"].size == 0:
            return float("nan")
        return float(v["small"].mean() - v["large"].mean())

    boot = grad(firm_vectors)
    out = {
        "ci_lo": pct(boot, 2.5),
        "ci_hi": pct(boot, 97.5),
        "boot_sd": float(np.nanstd(boot, ddof=1)) if np.isfinite(boot).sum() > 1 else float("nan"),
        "n_boot": int(n_boot),
    }
    if ref_vectors is not None:
        delta = boot - grad(ref_vectors)
        out.update({
            "delta_vs_all": point(firm_vectors) - point(ref_vectors),
            "delta_ci_lo": pct(delta, 2.5),
            "delta_ci_hi": pct(delta, 97.5),
        })
    return out


def _aligned_vectors(
    firms: pd.DataFrame, ref_firms: pd.DataFrame | None, measure: str
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, list[str]]:
    """
    Per-tier firm-mean vectors, in a fixed ticker order.

    When a reference frame is supplied the two are intersected on ticker, so the
    paired bootstrap compares the same firms under both budgets.  Firms lost to
    the intersection are returned so the caller can report them.
    """
    vectors: dict[str, np.ndarray] = {}
    ref_vectors: dict[str, np.ndarray] | None = {} if ref_firms is not None else None
    dropped: list[str] = []
    for tier in TIER_ORDER:
        a = firms[(firms["tier"] == tier) & firms[measure].notna()]
        if ref_firms is None:
            vectors[tier] = a.sort_values("ticker")[measure].to_numpy(float)
            continue
        b = ref_firms[(ref_firms["tier"] == tier) & ref_firms[measure].notna()]
        common = sorted(set(a["ticker"]) & set(b["ticker"]))
        dropped += sorted((set(a["ticker"]) | set(b["ticker"])) - set(common))
        vectors[tier] = (
            a.set_index("ticker").loc[common, measure].to_numpy(float)
            if common else np.zeros(0)
        )
        ref_vectors[tier] = (  # type: ignore[index]
            b.set_index("ticker").loc[common, measure].to_numpy(float)
            if common else np.zeros(0)
        )
    return vectors, ref_vectors, dropped


# ── The experiment ────────────────────────────────────────────────────────────

def run_truncation_contrast(
    df: pd.DataFrame,
    section: str = PRIMARY_SECTION,
    budgets: tuple[int | None, ...] = DEFAULT_BUDGETS,
    seed: int = 20261004,
    n_boot: int = 10000,
    variants: tuple[str, ...] = ("inclusive", "strict"),
) -> dict:
    """
    Recompute the tier gradient at every truncation budget.

    Returns a dict of three tidy frames — `tiers`, `gradients`, `firm_level` — plus
    `diagnostics` (one row per cell) and `meta` (seed, n_boot, counts).  Every
    number in them is computed here; none is carried in from elsewhere.
    """
    sec = section_frame(prepare(df), section)
    if sec.empty:
        raise SystemExit(f"no windows with section == {section!r}")
    inv = filing_inventory(sec)

    if "all" not in [k_label(k) for k in budgets]:
        raise SystemExit("budgets must include the complete-section case (None)")

    tiers_rows: list[dict] = []
    grad_rows: list[dict] = []
    firm_rows: list[dict] = []
    diag_rows: list[dict] = []

    for variant in variants:
        strict = variant == "strict"
        per_k: dict[int | None, tuple[pd.DataFrame, pd.DataFrame, dict]] = {}
        for k in budgets:
            filings, diag = filing_means_at_k(sec, inv, k, strict=strict)
            firms = firm_means(filings)
            per_k[k] = (filings, firms, diag)
            diag_rows.append({"variant": variant, **{
                a: b for a, b in diag.items() if a != "frac_read_scope_ids"
            }})

            for _, r in firms.iterrows():
                firm_rows.append({
                    "variant": variant, "k": k_label(k), "ticker": r["ticker"],
                    "tier": r["tier"], "n_filings": int(r["n_filings"]),
                    "numeric_disagreement": r["numeric_disagreement"],
                    "embedding_disagreement": r["embedding_disagreement"],
                    "frac_section_read": r["frac_section_read"],
                })

        ref_firms = per_k[None][1]
        ref_gradients = {
            m: _tier_gap(tier_stats(ref_firms, m)) for m in MEASURES
        }
        for k in budgets:
            filings, firms, diag = per_k[k]
            # Coverage: fraction of the complete MD&A read at this budget,
            # aggregated firm-first like every other quantity here.
            cov = tier_stats(firms, "frac_section_read")

            for measure in MEASURES:
                ref_gradient = ref_gradients[measure]
                ts = tier_stats(firms, measure)
                for _, row in ts.iterrows():
                    t = row["tier"]
                    sub = filings[filings["tier"] == t]
                    tiers_rows.append({
                        "measure": measure, "variant": variant, "k": k_label(k),
                        "tier": t, "n_firms": int(row["n_firms"]),
                        "n_filings": int(len(sub)),
                        "n_windows_used": int(sub["n_windows_used"].sum()),
                        "tier_mean": row["mean"], "tier_sd": row["sd"],
                        "mean_frac_section_read": float(
                            cov.loc[cov["tier"] == t, "mean"].iloc[0]
                        ),
                    })

                means = {r["tier"]: r["mean"] for _, r in ts.iterrows()}
                vecs, ref_vecs, dropped = _aligned_vectors(
                    firms, None if k is None else ref_firms, measure
                )
                label = f"{measure}|{variant}|{k_label(k)}"
                boot = bootstrap_gradient(
                    vecs, ref_vecs, n_boot, _rng_for(seed, label)
                )
                rho, p, n_rho = _spearman_tier_frac(firms)
                grad_rows.append({
                    "measure": measure, "variant": variant, "k": k_label(k),
                    "k_numeric": (np.nan if k is None else k),
                    "mean_large": means["large"], "mean_mid": means["mid"],
                    "mean_small": means["small"],
                    "gradient_small_minus_large": means["small"] - means["large"],
                    "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
                    "boot_sd": boot["boot_sd"],
                    "gradient_mid_minus_large": means["mid"] - means["large"],
                    "gradient_small_minus_mid": means["small"] - means["mid"],
                    "monotone_small_gt_mid_gt_large": bool(
                        means["small"] > means["mid"] > means["large"]
                    ),
                    "delta_vs_all": boot.get("delta_vs_all", 0.0 if k is None else np.nan),
                    "delta_ci_lo": boot.get("delta_ci_lo", np.nan),
                    "delta_ci_hi": boot.get("delta_ci_hi", np.nan),
                    # Size of the truncation effect relative to the complete-section
                    # gradient.  A CI on the paired delta can exclude 0 for a change
                    # of no substantive size once within-filing noise is averaged
                    # over 30 firms, so the magnitude is reported beside the CI and
                    # neither is quoted without the other.
                    "delta_vs_all_pct_of_complete": _pct_of(
                        boot.get("delta_vs_all", 0.0 if k is None else np.nan),
                        ref_gradient,
                    ),
                    "n_firms_large": int(ts.loc[ts["tier"] == "large", "n_firms"].iloc[0]),
                    "n_firms_mid": int(ts.loc[ts["tier"] == "mid", "n_firms"].iloc[0]),
                    "n_firms_small": int(ts.loc[ts["tier"] == "small", "n_firms"].iloc[0]),
                    "min_tier_n_firms": int(ts["n_firms"].min()),
                    "n_firms_paired": int(sum(v.size for v in vecs.values())),
                    "n_firms_dropped_from_pairing": len(dropped),
                    "firms_dropped_from_pairing": ",".join(dropped),
                    "n_filings_contributing": diag["n_filings_contributing"],
                    "n_filings_no_surviving_window": diag["n_filings_no_surviving_window"],
                    "n_filings_with_at_least_k": diag["n_filings_with_at_least_k"],
                    "n_filings_short_of_k": diag["n_filings_short_of_k"],
                    "n_filings_dropped_by_strict": diag["n_filings_dropped_by_strict"],
                    "n_filings_failing_within_budget_gate":
                        diag["n_filings_failing_within_budget_gate"],
                    "n_windows_in_budget": diag["n_windows_in_budget"],
                    "n_windows_used": diag["n_windows_used"],
                    "n_windows_dropped_failed": diag["n_windows_dropped_failed"],
                    "mean_frac_read_large": float(
                        cov.loc[cov["tier"] == "large", "mean"].iloc[0]),
                    "mean_frac_read_mid": float(
                        cov.loc[cov["tier"] == "mid", "mean"].iloc[0]),
                    "mean_frac_read_small": float(
                        cov.loc[cov["tier"] == "small", "mean"].iloc[0]),
                    "spearman_tierrank_fracread_rho": rho,
                    "spearman_tierrank_fracread_p": p,
                    "spearman_n_firms": n_rho,
                    "within_filing_sd_defined": bool(k != 1),
                    "seed": int(seed), "n_boot": int(n_boot),
                    "bootstrap_stream": label,
                })

    grad_df = pd.DataFrame(grad_rows)
    thin = grad_df[grad_df["min_tier_n_firms"] < 5]
    for _, r in thin.iterrows():
        logger.warning(
            "firm attrition: %s / %s / k=%s rests on %d/%d/%d firms "
            "(large/mid/small). A tier with fewer than 5 firms cannot support a "
            "bootstrap CI that means anything; reported, not hidden.",
            r["measure"], r["variant"], r["k"],
            r["n_firms_large"], r["n_firms_mid"], r["n_firms_small"],
        )

    meta = {
        "seed": int(seed),
        "n_boot": int(n_boot),
        "section": section,
        "budgets": [k_label(k) for k in budgets],
        "n_windows_total_in_section": int(len(sec)),
        "n_filings_in_section": int(len(inv)),
        "n_filings_gate_failed": int((~inv["passes_gate"]).sum()),
        "filings_gate_failed": sorted(inv.loc[~inv["passes_gate"], "filing_id"]),
        "n_windows_failed_in_section": int((~_is_usable_window(sec)).sum()),
        "min_usable_chunks": int(MIN_USABLE_CHUNKS),
        "measures_reported_in_order": list(MEASURES),
        "primary_measure": PRIMARY_MEASURE,
    }
    return {
        "tiers": pd.DataFrame(tiers_rows),
        "gradients": grad_df,
        "firm_level": pd.DataFrame(firm_rows),
        "diagnostics": pd.DataFrame(diag_rows),
        "inventory": inv,
        "meta": meta,
    }


def _tier_gap(ts: pd.DataFrame) -> float:
    """small - large from a tier_stats frame."""
    small = ts.loc[ts["tier"] == "small", "mean"].iloc[0]
    large = ts.loc[ts["tier"] == "large", "mean"].iloc[0]
    return float(small - large)


def _pct_of(delta: float, reference: float) -> float:
    """`delta` as a percentage of the complete-section gradient, or NaN."""
    if reference is None or not np.isfinite(reference) or reference == 0:
        return float("nan")
    if not np.isfinite(delta):
        return float("nan")
    return float(100.0 * delta / abs(reference))


def _spearman_tier_frac(firms: pd.DataFrame) -> tuple[float, float, int]:
    """
    Spearman(tier rank [large=3], firm-level fraction of MD&A read).

    This is the coverage-confound statistic in the form already reported for the
    retired 2,000-word cap (rho = -0.415, p = 0.0011).  At k = all every firm
    reads 100%, so the statistic is undefined (no variation) and returns NaN
    rather than a spurious zero.
    """
    from scipy.stats import spearmanr

    sub = firms.dropna(subset=["frac_section_read"])
    if len(sub) < 3 or sub["frac_section_read"].nunique() < 2:
        return (np.nan, np.nan, int(len(sub)))
    rho, p = spearmanr(
        sub["tier"].map(TIER_RANK).to_numpy(float),
        sub["frac_section_read"].to_numpy(float),
    )
    return (float(rho), float(p), int(len(sub)))


# ── Reporting ─────────────────────────────────────────────────────────────────

CAVEAT = (
    "SIMULATED truncation: each window was scored in isolation, so a persona "
    "scoring window 1 of 25 never saw the rest of the filing. This is not the same "
    "as a model reading a truncated document. The exact replication is S1(b), "
    "which is not run here."
)


def format_report(res: dict) -> str:
    meta = res["meta"]
    g = res["gradients"]
    L: list[str] = []
    L.append("S1(a) Truncation contrast — EXPLORATORY (ANALYSIS_PLAN.md §6)")
    L.append("=" * 78)
    L.append("")
    L.append(f"section={meta['section']}  windows={meta['n_windows_total_in_section']}  "
             f"filings={meta['n_filings_in_section']}  "
             f"failed windows dropped={meta['n_windows_failed_in_section']}")
    L.append(f"filings failing the complete-section usability gate "
             f"(>= {meta['min_usable_chunks']} usable and >= half): "
             f"{meta['n_filings_gate_failed']}"
             + (f"  {meta['filings_gate_failed']}" if meta["filings_gate_failed"] else ""))
    L.append(f"seed={meta['seed']}  bootstrap draws={meta['n_boot']}  "
             f"unit of analysis = firm")
    L.append("")
    L.append("CAVEAT: " + CAVEAT)
    L.append("")

    for measure in meta["measures_reported_in_order"]:
        tag = "PRIMARY" if measure == meta["primary_measure"] else "secondary"
        L.append(f"{measure}  [{tag}]")
        L.append("-" * 78)
        for variant in sorted(g["variant"].unique(), reverse=True):
            sub = g[(g["measure"] == measure) & (g["variant"] == variant)]
            L.append(f"  variant = {variant}")
            L.append("    k     large     mid     small    gradient (small-large)"
                     "        95% CI            d vs all        mono  firms filings")
            for _, r in sub.iterrows():
                dv = "" if r["k"] == "all" else f"{r['delta_vs_all']:+.4f}"
                dci = ("" if r["k"] == "all"
                       else f"[{r['delta_ci_lo']:+.4f},{r['delta_ci_hi']:+.4f}]")
                L.append(
                    f"  {r['k']:>4}  {r['mean_large']:7.4f} {r['mean_mid']:7.4f} "
                    f"{r['mean_small']:7.4f}   {r['gradient_small_minus_large']:+.4f}   "
                    f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  {dv:>8} {dci:>18}  "
                    f"{'Y' if r['monotone_small_gt_mid_gt_large'] else 'n':>4}  "
                    f"{r['n_firms_paired']:>5} {r['n_filings_contributing']:>7}"
                )
            L.append("")

    L.append("Coverage at each budget (mean fraction of the complete MD&A read, "
             "firm-level)")
    L.append("-" * 78)
    cov = g[(g["measure"] == PRIMARY_MEASURE)]
    for variant in sorted(cov["variant"].unique(), reverse=True):
        sub = cov[cov["variant"] == variant]
        L.append(f"  variant = {variant}")
        L.append("    k     large     mid    small   Spearman(tier rank, frac read)"
                 "        filings >= k  short of k  dropped(strict)")
        for _, r in sub.iterrows():
            rho = ("     n/a (all firms read 100%)" if np.isnan(r["spearman_tierrank_fracread_rho"])
                   else f"rho={r['spearman_tierrank_fracread_rho']:+.3f} "
                        f"p={r['spearman_tierrank_fracread_p']:.4f} "
                        f"n={int(r['spearman_n_firms'])}")
            L.append(
                f"  {r['k']:>4}  {r['mean_frac_read_large']:7.3f} "
                f"{r['mean_frac_read_mid']:6.3f} {r['mean_frac_read_small']:6.3f}   "
                f"{rho:<34} {int(r['n_filings_with_at_least_k']):>6} "
                f"{int(r['n_filings_short_of_k']):>10} "
                f"{int(r['n_filings_dropped_by_strict']):>14}"
            )
        L.append("")

    L.append("Firm attrition by budget (firms per tier entering the tier mean)")
    L.append("-" * 78)
    att = g[g["measure"] == PRIMARY_MEASURE]
    for variant in sorted(att["variant"].unique(), reverse=True):
        sub = att[att["variant"] == variant]
        L.append(f"  variant = {variant}")
        L.append("    k    large  mid  small   note")
        for _, r in sub.iterrows():
            note = ("" if r["min_tier_n_firms"] >= 5
                    else "THIN — a tier below 5 firms cannot support a meaningful CI")
            L.append(
                f"  {r['k']:>4}  {int(r['n_firms_large']):>5} {int(r['n_firms_mid']):>4} "
                f"{int(r['n_firms_small']):>6}   {note}"
            )
        L.append("")

    L.append("Budget rules")
    L.append("-" * 78)
    L.append("  inclusive: every filing contributes min(k, its window count) windows;")
    L.append("             a filing with fewer than k windows contributes all of them.")
    L.append("  strict   : only filings with >= k windows contribute. Drops are not")
    L.append("             tier-neutral — short filings are disproportionately small-cap.")
    L.append("  k = 1    : the pre-registered usability gate is evaluated once on each")
    L.append("             filing's COMPLETE MD&A, not inside the budget, so k = 1 is")
    L.append("             analysable. Within-filing SD is undefined at k = 1 and is")
    L.append("             reported as NaN, never 0. The count of filings that would")
    L.append("             fail a within-budget gate is in the diagnostics frame.")
    return "\n".join(L)


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(res: dict, out_path: Path) -> Path:
    """
    Gradient vs truncation budget, with bootstrap CIs and the complete-section
    estimate marked.  Vector PDF; colourblind-safe hues with linestyle and marker
    carrying the same distinction so it survives greyscale.
    """
    import matplotlib
    matplotlib.use("Agg")  # no display, no network, deterministic output
    import matplotlib.pyplot as plt

    g = res["gradients"]
    meta = res["meta"]
    order = [k for k in meta["budgets"]]
    xs = {k: i for i, k in enumerate(order)}

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.8), sharex=True)
    fig.subplots_adjust(left=0.135, right=0.975, top=0.905, bottom=0.155, hspace=0.30)

    panels = [
        (axes[0], "numeric_disagreement",
         "(a) Numeric disagreement — secondary measure"),
        (axes[1], "embedding_disagreement",
         "(b) Embedding disagreement — PRIMARY measure"),
    ]

    for ax, measure, title in panels:
        for variant, colour, ls, marker, mfc, off in (
            ("inclusive", C_INCLUSIVE, "-", "o", C_INCLUSIVE, -0.055),
            ("strict", C_STRICT, "--", "s", "white", +0.055),
        ):
            sub = g[(g["measure"] == measure) & (g["variant"] == variant)]
            if sub.empty:
                continue
            x = np.array([xs[k] + off for k in sub["k"]])
            y = sub["gradient_small_minus_large"].to_numpy(float)
            lo = y - sub["ci_lo"].to_numpy(float)
            hi = sub["ci_hi"].to_numpy(float) - y
            ax.errorbar(
                x, y, yerr=[lo, hi], color=colour, ls=ls, lw=2.0, marker=marker,
                ms=8, mfc=mfc, mec=colour, mew=1.8, capsize=3.5, elinewidth=1.4,
                label=f"{variant} budget", zorder=3,
            )
            if variant == "inclusive":
                row = sub[sub["k"] == "all"]
                if not row.empty:
                    ref = float(row["gradient_small_minus_large"].iloc[0])
                    ax.axhspan(
                        float(row["ci_lo"].iloc[0]), float(row["ci_hi"].iloc[0]),
                        color=C_REF, alpha=0.10, zorder=0,
                    )
                    ax.axhline(ref, color=C_REF, lw=1.3, ls=(0, (6, 3)), zorder=1)
                    ax.annotate(
                        "complete section", xy=(0.32, ref),
                        xycoords=("axes fraction", "data"),
                        xytext=(0, -5), textcoords="offset points",
                        fontsize=8.5, color=C_REF, va="top",
                    )
        ax.axhline(0, color="#9b9a96", lw=0.9, zorder=1)
        ax.set_title(title, fontsize=10.5, loc="left", pad=11)
        ax.set_ylabel("gradient  (small − large)", fontsize=9.5)
        ax.margins(y=0.16)
        ax.grid(axis="y", color=C_GRID, lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.legend(frameon=False, fontsize=9, loc="best")

    ax = axes[2]
    cov = g[(g["measure"] == PRIMARY_MEASURE) & (g["variant"] == "inclusive")]
    for tier, ls, marker in (("large", "-", "o"), ("mid", "--", "s"), ("small", ":", "^")):
        y = cov[f"mean_frac_read_{tier}"].to_numpy(float)
        x = np.array([xs[k] for k in cov["k"]], float)
        ax.plot(x, y, color=C_TIER[tier], ls=ls, lw=2.0, marker=marker, ms=7,
                mfc="white" if tier != "large" else C_TIER[tier],
                mec=C_TIER[tier], mew=1.8, label=f"{tier}-cap", zorder=3)
    ax.set_title("(c) Mean fraction of the complete MD&A read, by tier "
                 "(inclusive budget)", fontsize=10.5, loc="left", pad=11)
    ax.set_ylabel("fraction of MD&A read", fontsize=9.5)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color=C_GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="best")

    axes[-1].set_xticks(range(len(order)))
    axes[-1].set_xticklabels(
        [("all\n(complete)" if k == "all" else k) for k in order], fontsize=9
    )
    axes[-1].set_xlabel("budget: first k windows of the filing's MD&A", fontsize=9.5)
    axes[-1].set_xlim(-0.35, len(order) - 0.65)

    fig.suptitle(
        "Truncation changes the measured tier gradient",
        fontsize=12.5, x=0.135, ha="left", y=0.985,
    )
    fig.text(
        0.135, 0.012,
        "Firm-level means; window → filing → firm → tier, all unweighted. Bars are "
        "95% percentile bootstrap CIs\n"
        f"over {meta['n_boot']:,} draws resampling firms within tier, seed "
        f"{meta['seed']}. Failed windows are dropped, never imputed.\n"
        "SIMULATED truncation: every window was scored in isolation, so a persona "
        "scoring window 1 of 25 never\nsaw the rest of the filing. This is not a model "
        "reading a truncated document; that is S1(b), not run here.",
        fontsize=7.9, color="#3a3a38", va="bottom", ha="left", linespacing=1.55,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")  # vector
    plt.close(fig)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(
        description="S1(a) truncation contrast — cached output only, no model calls."
    )
    ap.add_argument("--input", type=Path,
                    default=repo / "output" / "chunk_level_results.csv",
                    help="window-level CSV (default: output/chunk_level_results.csv)")
    ap.add_argument("--out-dir", type=Path,
                    default=repo / "output" / "truncation_contrast",
                    help="where the CSVs, the report and the PDF are written")
    ap.add_argument("--section", default=PRIMARY_SECTION, choices=["mda", "risk"],
                    help="MD&A is the pre-registered primary section (§2)")
    ap.add_argument("--budgets", default="1,2,4,8,all",
                    help="comma-separated window budgets; 'all' = complete section")
    ap.add_argument("--seed", type=int, default=20261004,
                    help="bootstrap seed; recorded in every output")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--no-figure", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    budgets = tuple(
        None if b.strip().lower() == "all" else int(b)
        for b in args.budgets.split(",") if b.strip()
    )

    df = load_window_level(args.input)
    res = run_truncation_contrast(
        df, section=args.section, budgets=budgets, seed=args.seed, n_boot=args.n_boot
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    res["tiers"].to_csv(out / "truncation_contrast_tiers.csv", index=False)
    res["gradients"].to_csv(out / "truncation_contrast_gradients.csv", index=False)
    res["firm_level"].to_csv(out / "truncation_contrast_firm_level.csv", index=False)
    res["diagnostics"].to_csv(out / "truncation_contrast_diagnostics.csv", index=False)
    report = format_report(res)
    (out / "truncation_contrast.txt").write_text(report + "\n")

    logger.info("dropped %d failed window(s) from the %s section (never imputed)",
                res["meta"]["n_windows_failed_in_section"], res["meta"]["section"])
    logger.info("%d filing(s) failed the complete-section usability gate",
                res["meta"]["n_filings_gate_failed"])

    if not args.no_figure:
        fig_path = make_figure(res, out / "truncation_contrast.pdf")
        logger.info("figure written: %s", fig_path)

    if not args.quiet:
        print(report)
    logger.info("outputs written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
