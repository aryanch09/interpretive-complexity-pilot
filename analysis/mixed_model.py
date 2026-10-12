"""
W5 — the window-level mixed model (ANALYSIS_PLAN.md section 5).

Pre-registered specification
----------------------------
    disagreement_w ~ tier + log(window_words) + position_decile + form_type
                     + (1 | firm/filing)

with `position_decile` derived from `chunk_index / max(1, n_in_section - 1)`.

Pre-declared criterion (fixed before any filing was scored): **the Small-vs-Large
tier coefficient CI must exclude 0 in the primary direction** (positive — H1 is
that smaller firms produce *more* inter-persona disagreement).  Tier coefficients
are reported with CIs, together with the ICCs at the filing and firm levels.

What W5 is for
--------------
The advisor's Issue 1: window count scales with section length, length scales
with tier, and a filing-level mean over 25 windows is mechanically less variable
than one over 2.  W5 sidesteps filing-level averaging entirely by modelling the
window and letting the nesting absorb the firm and filing levels, while holding
the two channels that can actually bias a tier mean -- window size and
within-section position -- as explicit covariates.

The honest problem with (1 | firm/filing)
-----------------------------------------
`statsmodels` MixedLM is not a general crossed/nested random-effects engine.  It
fits ONE grouping factor and expresses further structure through variance
components (`vc_formula`).  Rather than present one specification as if the
software supported the formula literally, this module fits three and reports
them side by side:

  (a) `nested`     -- firm as the grouping factor with a filing-level variance
                      component nested inside it.  This is the closest thing to
                      (1 | firm/filing) that MixedLM actually supports, and it
                      is the only specification that yields BOTH ICCs.
  (b) `firm_only`  -- a firm random intercept and nothing else.  Fewer
                      assumptions, and the fallback if (a) will not converge.
  (c) `ols_cluster`-- OLS with standard errors clustered by firm.  A
                      specification check that does not rely on any
                      random-effects assumption; point estimates are unweighted
                      so they differ from (a)/(b) by construction, which is the
                      point of including it.

Which one the paper's W5 claim rests on is decided by a rule stated before the
fit, printed in the report, and NOT chosen after seeing which one passes:
(a) if it converges, else (b) with (a)'s convergence diagnostics reported in
full.  (c) is never the primary -- it ignores the nesting the design has -- but a
disagreement between (c) and the primary is reported as a limitation, not hidden.

Pre-registered data rules enforced here
---------------------------------------
* A window with `any_failed = True` (both disagreement columns null) is DROPPED,
  never imputed (section 8.1).  Dropped counts are reported.
* A filing is used only if at least 2 of its windows survive AND at least half
  survive (section 8.2), evaluated within the section being analysed -- section 2
  states that gate as applying to MD&A, the primary section.
* Primary section is MD&A only; Risk Factors is a separate secondary run with its
  own n (section 2).  Sections are never pooled.

No network calls, no model calls: this reads `output/chunk_level_results.csv` and
nothing else.  Every number printed is computed from that file.

Run:
    python3 -m analysis.mixed_model                        # MD&A, embedding measure
    python3 -m analysis.mixed_model --section risk
    python3 -m analysis.mixed_model --measure numeric_disagreement
    python3 -m analysis.mixed_model --chunks /path/to/chunk_level_results.csv
"""

from __future__ import annotations

import sys

# Documented abseil mutex deadlock on this machine: TensorFlow 2.21 + Python 3.13
# + PyTorch 2.12 hangs the process on import.  Nothing here imports transformers,
# but the guard is set before any project import in case a transitive import
# reaches analysis.embeddings.  Do not remove (CLAUDE.md, Critical Technical Notes).
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = None  # type: ignore[assignment]

import argparse
import json
import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

logger = logging.getLogger("mixed_model")

TIERS: tuple[str, ...] = ("large", "mid", "small")

# Reference levels. Made explicit because the SIGN of the Small-vs-Large
# coefficient depends on them, and the pre-declared criterion is directional.
TIER_REFERENCE = "large"
FORM_REFERENCE = "10-K"

# ANALYSIS_PLAN section 8.2 / analysis.disagreement.MIN_USABLE_CHUNKS.
MIN_USABLE_WINDOWS = 2

# The pre-registered scoring window (config.CHUNK_WORDS). Used only to centre
# log(window_words) for reporting; centring a covariate cannot change any tier
# coefficient in a model without interactions.
CHUNK_WORDS = 1200

PRIMARY_MEASURE = "embedding_disagreement"
PRIMARY_SECTION = "mda"

# H1: smaller firms -> MORE disagreement, so the pre-declared direction for the
# Small-vs-Large coefficient is positive.
PRIMARY_DIRECTION = "positive"

COEF_LABELS: dict[str, str] = {
    "Intercept": "intercept",
    "tier_mid": f"tier[mid vs {TIER_REFERENCE}]",
    "tier_small": f"tier[small vs {TIER_REFERENCE}]",
    "log_window_words": "log(window_words)",
    "position_decile": "position_decile",
    "form_10Q": f"form_type[10-Q vs {FORM_REFERENCE}]",
}

SMALL_VS_LARGE = "tier_small"
MID_VS_LARGE = "tier_mid"


# ───────────────────────────── data structures ──────────────────────────────


@dataclass
class QualityReport:
    """Exactly what was read, dropped, and kept. Nothing is dropped silently."""

    csv_path: str
    section: str
    measure: str
    n_rows_file: int = 0
    n_rows_section: int = 0
    n_failed_dropped: int = 0
    n_null_measure_dropped: int = 0
    n_filings_dropped_gate: int = 0
    filings_dropped_gate: list[str] = field(default_factory=list)
    n_windows_dropped_gate: int = 0
    n_windows_used: int = 0
    n_filings_used: int = 0
    n_firms_used: int = 0
    windows_by_tier: dict[str, int] = field(default_factory=dict)
    firms_by_tier: dict[str, int] = field(default_factory=dict)
    filings_by_tier: dict[str, int] = field(default_factory=dict)
    single_filing_firms: list[str] = field(default_factory=list)
    n_in_section_mismatches: list[str] = field(default_factory=list)
    served_by_values: list[str] = field(default_factory=list)
    n_parse_errors_kept: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SpecResult:
    """One of the three specifications."""

    key: str
    label: str
    engine: str
    inference: str
    converged: bool
    message: str = ""
    coefficients: dict[str, dict[str, float]] = field(default_factory=dict)
    variances: dict[str, Optional[float]] = field(default_factory=dict)
    iccs: dict[str, Optional[float]] = field(default_factory=dict)
    n_obs: int = 0
    n_groups: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def criterion(self, coef: str = SMALL_VS_LARGE,
                  direction: str = PRIMARY_DIRECTION) -> Optional[bool]:
        """Does this spec meet the pre-declared W5 criterion?

        None when the coefficient is unavailable (e.g. the fit failed), so a
        missing fit can never be read as a pass.
        """
        c = self.coefficients.get(coef)
        if c is None or not self.converged:
            return None
        if direction == "positive":
            return bool(c["ci_low"] > 0.0)
        if direction == "negative":
            return bool(c["ci_high"] < 0.0)
        raise ValueError(f"direction must be 'positive' or 'negative', got {direction!r}")

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["criterion_small_vs_large"] = self.criterion()
        return d


@dataclass
class W5Result:
    seed: int
    section: str
    measure: str
    direction: str
    tier_reference: str
    form_reference: str
    position_term: str
    quality: QualityReport
    specs: dict[str, SpecResult]
    primary_spec_key: str
    primary_spec_reason: str
    variance_decomposition: dict[str, Any]
    collinearity: dict[str, Any]
    bootstrap: dict[str, Any]
    design_summary: dict[str, Any]

    @property
    def primary(self) -> SpecResult:
        return self.specs[self.primary_spec_key]

    def criterion_met(self) -> Optional[bool]:
        return self.primary.criterion(SMALL_VS_LARGE, self.direction)

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "section": self.section,
            "measure": self.measure,
            "direction": self.direction,
            "tier_reference": self.tier_reference,
            "form_reference": self.form_reference,
            "position_term": self.position_term,
            "quality": self.quality.as_dict(),
            "specs": {k: v.as_dict() for k, v in self.specs.items()},
            "primary_spec": self.primary_spec_key,
            "primary_spec_reason": self.primary_spec_reason,
            "criterion_met": self.criterion_met(),
            "variance_decomposition": self.variance_decomposition,
            "collinearity": self.collinearity,
            "bootstrap": self.bootstrap,
            "design_summary": self.design_summary,
        }


# ──────────────────────────────── loading ────────────────────────────────────


def filing_id(row: pd.Series) -> str:
    return f"{row['ticker']}|{row['filing_type']}|{row['filing_date']}"


def position_decile_from(chunk_index: int, n_in_section: int) -> int:
    """Pre-registered position term, binned into 10 equal-width bins of the
    fraction chunk_index / max(1, n_in_section - 1).

    The fraction is closed at 1.0, so decile 9 covers [0.9, 1.0]; a single-window
    section has fraction 0 and so decile 0.
    """
    frac = chunk_index / max(1, n_in_section - 1)
    return int(min(9, math.floor(frac * 10)))


def load_window_frame(
    csv_path: str | Path,
    section: str = PRIMARY_SECTION,
    measure: str = PRIMARY_MEASURE,
) -> tuple[pd.DataFrame, QualityReport]:
    """Read chunk_level_results.csv and apply the pre-registered data rules.

    Returns the analysable window frame plus a QualityReport accounting for every
    row that was read and every row that was dropped.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} does not exist. W5 runs on cached window-level output only; "
            "it makes no model calls. Run the scoring pipeline first, or point "
            "--chunks at a synthetic fixture (tests/fixtures_w5.py)."
        )

    raw = pd.read_csv(csv_path)
    q = QualityReport(csv_path=str(csv_path), section=section, measure=measure,
                      n_rows_file=len(raw))

    required = {"ticker", "tier", "filing_type", "filing_date", "section",
                "chunk_index", "n_in_section", "word_count", "any_failed", measure}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required column(s): {missing}")

    df = raw[raw["section"].astype(str) == section].copy()
    q.n_rows_section = len(df)
    if df.empty:
        raise ValueError(f"No rows with section == {section!r} in {csv_path}")

    df["filing"] = df.apply(filing_id, axis=1).astype(str)
    df["firm"] = df["ticker"].astype(str)

    # Integrity check: the stored n_in_section must equal the number of scored
    # windows present for that (filing, section). A mismatch means rows are
    # missing from the file, which would silently distort position_decile.
    counts = df.groupby("filing")["chunk_index"].size()
    for fid, n_rows in counts.items():
        stored = df.loc[df["filing"] == fid, "n_in_section"].unique()
        if len(stored) != 1 or int(stored[0]) != int(n_rows):
            q.n_in_section_mismatches.append(
                f"{fid}: n_in_section={list(stored)} but {n_rows} row(s) present"
            )
    if q.n_in_section_mismatches:
        logger.warning(
            "n_in_section disagrees with the row count for %d filing(s); "
            "position_decile uses the STORED n_in_section, which is the chunker's "
            "own count. Listed in the report.",
            len(q.n_in_section_mismatches),
        )

    if "served_by" in df.columns:
        q.served_by_values = sorted(
            {b for v in df["served_by"].astype(str) for b in v.split("|") if b}
        )

    # ---- pre-registered rule 8.1: drop failed windows, never impute ---------
    failed_mask = df["any_failed"].fillna(False).astype(bool)
    q.n_failed_dropped = int(failed_mask.sum())
    kept = df[~failed_mask].copy()

    null_mask = kept[measure].isna()
    q.n_null_measure_dropped = int(null_mask.sum())
    if q.n_null_measure_dropped:
        logger.warning(
            "%d window(s) have any_failed=False but a null %s; dropped, not imputed.",
            q.n_null_measure_dropped, measure,
        )
    kept = kept[~null_mask].copy()

    # ---- pre-registered rule 8.2: filing-level usability gate --------------
    scored_per_filing = df.groupby("filing").size()
    used_per_filing = kept.groupby("filing").size()
    drop_filings: list[str] = []
    for fid, n_total in scored_per_filing.items():
        n_used = int(used_per_filing.get(fid, 0))
        if n_used < MIN_USABLE_WINDOWS or n_used < n_total / 2:
            drop_filings.append(f"{fid}: {n_used} of {n_total} window(s) usable")
    if drop_filings:
        dropped_ids = [s.split(":")[0] for s in drop_filings]
        q.n_windows_dropped_gate = int(kept["filing"].isin(dropped_ids).sum())
        kept = kept[~kept["filing"].isin(dropped_ids)].copy()
    q.filings_dropped_gate = drop_filings
    q.n_filings_dropped_gate = len(drop_filings)

    if kept.empty:
        raise ValueError(
            f"No usable windows remain for section={section!r} after the "
            "pre-registered failure and filing-gate rules."
        )

    if "any_parse_error" in kept.columns:
        q.n_parse_errors_kept = int(kept["any_parse_error"].fillna(False).astype(bool).sum())

    q.n_windows_used = len(kept)
    q.n_filings_used = kept["filing"].nunique()
    q.n_firms_used = kept["firm"].nunique()
    for t in TIERS:
        sub = kept[kept["tier"] == t]
        q.windows_by_tier[t] = int(len(sub))
        q.firms_by_tier[t] = int(sub["firm"].nunique())
        q.filings_by_tier[t] = int(sub["filing"].nunique())
    per_firm = kept.groupby("firm")["filing"].nunique()
    q.single_filing_firms = sorted(per_firm[per_firm == 1].index.tolist())

    return kept.reset_index(drop=True), q


def prepare_design(
    df: pd.DataFrame,
    measure: str = PRIMARY_MEASURE,
    position_as_factor: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the model matrix columns and return (frame, regressor names).

    Tier is a 3-level factor with `large` as the reference, coded as explicit
    0/1 dummies rather than through patsy's `C()` so that the reference level is
    unambiguous in both the formula and the printed coefficient names.
    """
    d = df.copy()
    d["y"] = pd.to_numeric(d[measure], errors="raise").astype(float)
    d["log_window_words"] = np.log(pd.to_numeric(d["word_count"]).astype(float))
    d["window_words_centred_log"] = d["log_window_words"] - math.log(CHUNK_WORDS)
    d["position_frac"] = [
        int(ci) / max(1, int(n) - 1)
        for ci, n in zip(d["chunk_index"], d["n_in_section"])
    ]
    d["position_decile"] = [
        position_decile_from(int(ci), int(n))
        for ci, n in zip(d["chunk_index"], d["n_in_section"])
    ]
    d["form_10Q"] = (d["filing_type"].astype(str) == "10-Q").astype(float)
    for t in TIERS:
        if t == TIER_REFERENCE:
            continue
        d[f"tier_{t}"] = (d["tier"].astype(str) == t).astype(float)
    d["firm"] = d["firm"].astype(str)
    d["filing"] = d["filing"].astype(str)

    regressors: list[str] = []
    for t in TIERS:
        if t == TIER_REFERENCE:
            continue
        col = f"tier_{t}"
        if d[col].sum() == 0:
            logger.warning(
                "Tier %r has no windows in this sample; its dummy is dropped. "
                "The tier factor is then 2-level, not 3-level.", t,
            )
            continue
        regressors.append(col)
    regressors.append("log_window_words")

    if position_as_factor:
        present = sorted(d["position_decile"].unique())
        base = present[0]
        for v in present[1:]:
            col = f"position_decile_{v}"
            d[col] = (d["position_decile"] == v).astype(float)
            regressors.append(col)
        logger.info(
            "position_decile entered as a %d-level factor (reference decile %d).",
            len(present), base,
        )
    else:
        regressors.append("position_decile")

    if 0 < d["form_10Q"].sum() < len(d):
        regressors.append("form_10Q")
    else:
        logger.warning(
            "form_type has no variation in this sample (all %s); the form term "
            "is dropped.", d["filing_type"].astype(str).unique().tolist(),
        )

    return d, regressors


# ──────────────────────────────── fitting ────────────────────────────────────


def _coef_table(params, ses, ci, pvals, names) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for n in names:
        if n not in params.index:
            continue
        out[n] = {
            "estimate": float(params[n]),
            "se": float(ses[n]),
            "ci_low": float(ci.loc[n, 0]),
            "ci_high": float(ci.loc[n, 1]),
            "p": float(pvals[n]),
        }
    return out


def _formula(regressors: list[str]) -> str:
    return "y ~ " + " + ".join(regressors)


def fit_nested(
    df: pd.DataFrame,
    regressors: list[str],
    reml: bool = True,
) -> SpecResult:
    """(a) firm grouping factor + filing variance component nested within firm.

    `vc_formula={'filing': '0 + C(filing)'}` is evaluated inside each firm group,
    so the filing levels are nested, not crossed, and ONE variance is estimated
    for the whole filing component.
    """
    spec = SpecResult(
        key="nested",
        label="(a) firm grouping + filing variance component nested in firm",
        engine="statsmodels MixedLM (REML)" if reml else "statsmodels MixedLM (ML)",
        inference="Wald z, normal-approximation CIs (MixedLM does not provide "
                  "Satterthwaite/Kenward-Roger df)",
        converged=False,
        n_obs=len(df),
        n_groups=int(df["firm"].nunique()),
    )
    names = ["Intercept"] + regressors
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            md = smf.mixedlm(
                _formula(regressors),
                data=df,
                groups=df["firm"],
                re_formula="1",
                vc_formula={"filing": "0 + C(filing)"},
            )
            res = md.fit(reml=reml)
            spec.warnings = sorted({str(w.message) for w in caught})
    except Exception as exc:  # pragma: no cover - reported, not swallowed
        spec.message = f"{type(exc).__name__}: {exc}"
        logger.error("Nested specification (a) failed to fit: %s", spec.message)
        return spec

    spec.converged = bool(res.converged)
    spec.message = "converged" if res.converged else (
        "DID NOT CONVERGE - coefficients and variance components from this "
        "specification are not trustworthy and W5 falls back to (b)."
    )
    spec.coefficients = _coef_table(
        res.fe_params, res.bse_fe, res.conf_int().loc[names], res.pvalues, names
    )

    # In statsmodels, res.cov_re and res.vcomp are already on the variance scale
    # (res.cov_re_unscaled is the one divided by scale); res.scale is the
    # residual variance. Verified numerically against a known-truth simulation.
    var_firm = float(np.asarray(res.cov_re)[0, 0])
    var_filing = float(np.asarray(res.vcomp)[0]) if len(np.asarray(res.vcomp)) else float("nan")
    var_resid = float(res.scale)
    total = var_firm + var_filing + var_resid
    spec.variances = {
        "firm": var_firm,
        "filing_within_firm": var_filing,
        "residual_within_filing": var_resid,
        "total": total,
    }
    spec.iccs = {
        # Correlation between two windows of the same firm but DIFFERENT filings.
        "icc_firm": var_firm / total if total > 0 else None,
        # Correlation between two windows of the SAME filing.
        "icc_filing": (var_firm + var_filing) / total if total > 0 else None,
    }
    spec.diagnostics = {
        "loglike": float(res.llf),
        "method": "REML" if reml else "ML",
        "converged_flag": bool(res.converged),
        "n_firms": int(df["firm"].nunique()),
        "n_filings": int(df["filing"].nunique()),
        "min_group_size": int(df.groupby("firm").size().min()),
        "max_group_size": int(df.groupby("firm").size().max()),
        "vc_names": list(getattr(res, "vcomp_names", []) or ["filing"]),
    }
    return spec


def fit_firm_only(
    df: pd.DataFrame,
    regressors: list[str],
    reml: bool = True,
) -> SpecResult:
    """(b) firm random intercept only - the fallback, with fewer assumptions."""
    spec = SpecResult(
        key="firm_only",
        label="(b) firm random intercept only",
        engine="statsmodels MixedLM (REML)" if reml else "statsmodels MixedLM (ML)",
        inference="Wald z, normal-approximation CIs",
        converged=False,
        n_obs=len(df),
        n_groups=int(df["firm"].nunique()),
    )
    names = ["Intercept"] + regressors
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            md = smf.mixedlm(_formula(regressors), data=df, groups=df["firm"],
                             re_formula="1")
            res = md.fit(reml=reml)
            spec.warnings = sorted({str(w.message) for w in caught})
    except Exception as exc:  # pragma: no cover
        spec.message = f"{type(exc).__name__}: {exc}"
        logger.error("Firm-only specification (b) failed to fit: %s", spec.message)
        return spec

    spec.converged = bool(res.converged)
    spec.message = "converged" if res.converged else "DID NOT CONVERGE"
    spec.coefficients = _coef_table(
        res.fe_params, res.bse_fe, res.conf_int().loc[names], res.pvalues, names
    )
    var_firm = float(np.asarray(res.cov_re)[0, 0])
    var_resid = float(res.scale)
    total = var_firm + var_resid
    spec.variances = {
        "firm": var_firm,
        "filing_within_firm": None,  # not estimable in this specification
        "residual_within_filing": var_resid,
        "total": total,
    }
    spec.iccs = {
        "icc_firm": var_firm / total if total > 0 else None,
        # Not identified here: with no filing component, two windows of the same
        # filing and two windows of the same firm have the same modelled
        # correlation. Reporting icc_firm as if it were icc_filing would be wrong.
        "icc_filing": None,
    }
    spec.diagnostics = {
        "loglike": float(res.llf),
        "method": "REML" if reml else "ML",
        "converged_flag": bool(res.converged),
        "n_firms": int(df["firm"].nunique()),
    }
    return spec


def fit_ols_clustered(df: pd.DataFrame, regressors: list[str]) -> SpecResult:
    """(c) OLS with firm-clustered SEs - no random-effects assumptions at all."""
    n_clusters = int(df["firm"].nunique())
    spec = SpecResult(
        key="ols_cluster",
        label="(c) OLS, SEs clustered by firm",
        engine="statsmodels OLS",
        inference="cluster-robust by firm",
        converged=True,
        n_obs=len(df),
        n_groups=n_clusters,
    )
    names = ["Intercept"] + regressors
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # use_t=True is not the statsmodels default but is the right
            # small-sample choice: with cov_type="cluster" statsmodels sets
            # df_resid_inference = n_clusters - 1, so the CIs become t(G-1)
            # instead of normal.  With 30 firms that difference is not cosmetic.
            res = smf.ols(_formula(regressors), data=df).fit(
                cov_type="cluster", cov_kwds={"groups": df["firm"]}, use_t=True
            )
            spec.warnings = sorted({str(w.message) for w in caught})
    except Exception as exc:  # pragma: no cover
        spec.converged = False
        spec.message = f"{type(exc).__name__}: {exc}"
        logger.error("Clustered OLS specification (c) failed: %s", spec.message)
        return spec

    df_inf = getattr(res, "df_resid_inference", None)
    spec.inference = (
        "cluster-robust (CR1) by firm, "
        + (f"t with {df_inf:.0f} df (= n_firms - 1)" if res.use_t and df_inf is not None
           else "normal-approximation CIs")
    )
    spec.message = "fitted (closed form; no iterative convergence to check)"
    spec.coefficients = _coef_table(
        res.params, res.bse, res.conf_int(), res.pvalues, names
    )
    spec.variances = {
        "firm": None, "filing_within_firm": None,
        "residual_within_filing": float(res.mse_resid), "total": None,
    }
    spec.iccs = {"icc_firm": None, "icc_filing": None}
    spec.diagnostics = {
        "r_squared": float(res.rsquared),
        "n_clusters": n_clusters,
        "use_t": bool(res.use_t),
        "df_resid_ols": float(res.df_resid),
        "df_resid_inference": None if df_inf is None else float(df_inf),
        "note": (
            f"{n_clusters} clusters is few for cluster-robust inference; CR1 SEs "
            "can be anti-conservative below ~40 clusters."
        ),
    }
    return spec


# ──────────────────── model-free variance decomposition ─────────────────────


def variance_decomposition(df: pd.DataFrame, value_col: str = "y") -> dict[str, Any]:
    """Descriptive nested decomposition of total window-level variance.

    SS_total = SS_between_firms + SS_between_filings_within_firm + SS_within_filings
    (exactly additive, using observation-weighted group means).

    Two caveats, both reported rather than buried:

    1. These SHARES are descriptive, not variance-component estimates.  Under
       unbalanced group sizes each group mean carries residual noise, so the
       higher-level shares are biased UPWARD relative to the true variance
       components.  The REML components from specification (a) are the estimates;
       the shares' value is that they need no convergence and no distributional
       assumption, so they survive a failure of (a).
    2. Run on the raw measure, the between-firm share also contains whatever the
       fixed effects explain -- tier is a between-firm variable, so a real tier
       effect inflates it.  `run_w5` therefore reports this decomposition twice:
       on the raw measure, and on the residuals from the fixed-effect fit
       (`value_col="y_resid"`), which is the one comparable to the REML
       components.
    """
    y = df[value_col].to_numpy(dtype=float)
    grand = float(y.mean())
    ss_total = float(((y - grand) ** 2).sum())

    firm_mean = df.groupby("firm")[value_col].transform("mean").to_numpy(dtype=float)
    filing_mean = df.groupby("filing")[value_col].transform("mean").to_numpy(dtype=float)

    ss_firm = float(((firm_mean - grand) ** 2).sum())
    ss_filing = float(((filing_mean - firm_mean) ** 2).sum())
    ss_within = float(((y - filing_mean) ** 2).sum())

    n_firms = int(df["firm"].nunique())
    n_filings = int(df["filing"].nunique())
    n_obs = int(len(df))
    df_firm = max(n_firms - 1, 0)
    df_filing = max(n_filings - n_firms, 0)
    df_within = max(n_obs - n_filings, 0)

    def share(x: float) -> Optional[float]:
        return x / ss_total if ss_total > 0 else None

    return {
        "value_col": value_col,
        "n_obs": n_obs,
        "n_firms": n_firms,
        "n_filings": n_filings,
        "grand_mean": grand,
        "sd_total": float(y.std(ddof=1)) if n_obs > 1 else None,
        "ss_total": ss_total,
        "ss_between_firms": ss_firm,
        "ss_between_filings_within_firm": ss_filing,
        "ss_within_filings": ss_within,
        "share_between_firms": share(ss_firm),
        "share_between_filings_within_firm": share(ss_filing),
        "share_within_filings": share(ss_within),
        "df_between_firms": df_firm,
        "df_between_filings_within_firm": df_filing,
        "df_within_filings": df_within,
        "ms_between_firms": ss_firm / df_firm if df_firm else None,
        "ms_between_filings_within_firm": ss_filing / df_filing if df_filing else None,
        "ms_within_filings": ss_within / df_within if df_within else None,
        "additivity_check_abs_error": abs(
            ss_total - (ss_firm + ss_filing + ss_within)
        ),
    }


def collinearity_report(df: pd.DataFrame, regressors: list[str]) -> dict[str, Any]:
    """Pairwise correlations and VIFs for the fixed-effect design.

    Reported because the design makes window size and within-section position
    mechanically related: every window but a section's last is exactly 1,200
    words, so `log_window_words` varies almost only at the final (highest
    position) window.  A referee will ask whether the two covariates can be
    separated; this is the answer, computed rather than asserted.
    """
    cols = [c for c in regressors if pd.api.types.is_numeric_dtype(df[c])]
    out: dict[str, Any] = {"regressors": cols}
    if len(cols) >= 2:
        out["correlations"] = (
            df[cols].corr().round(4).to_dict()
        )
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X = sm.add_constant(df[cols].astype(float).to_numpy())
        out["vif"] = {
            c: float(variance_inflation_factor(X, i + 1)) for i, c in enumerate(cols)
        }
    except Exception as exc:  # pragma: no cover
        out["vif_error"] = f"{type(exc).__name__}: {exc}"

    last = df["chunk_index"] == (df["n_in_section"] - 1)
    out["n_windows_exactly_chunk_words"] = int((df["word_count"] == CHUNK_WORDS).sum())
    out["n_windows_final_in_section"] = int(last.sum())
    out["mean_log_words_final_vs_other"] = {
        "final": float(df.loc[last, "log_window_words"].mean()) if last.any() else None,
        "other": float(df.loc[~last, "log_window_words"].mean()) if (~last).any() else None,
    }
    return out


# ─────────────────────────── cluster bootstrap ───────────────────────────────


def firm_cluster_bootstrap(
    df: pd.DataFrame,
    regressors: list[str],
    coef: str = SMALL_VS_LARGE,
    n_boot: int = 0,
    seed: int = 20261004,
) -> dict[str, Any]:
    """Firm-level cluster bootstrap of an OLS coefficient (percentile CI).

    Off by default (`n_boot=0`).  It exists because 30 clusters is few for CR1
    SEs, so a resampling CI that makes no normality assumption is a useful
    companion to specification (c).  It is a robustness companion, NOT the
    pre-declared criterion, which is evaluated on the primary specification.

    Draws that cannot be fitted (e.g. a resample in which one tier is absent, so
    its dummy is collinear) are COUNTED and reported, never silently skipped.
    """
    out: dict[str, Any] = {
        "n_requested": int(n_boot), "seed": int(seed), "coef": coef,
        "n_successful": 0, "n_failed": 0, "failure_reasons": {},
    }
    if n_boot <= 0:
        out["status"] = "not run (n_boot=0)"
        return out

    rng = np.random.default_rng(seed)
    firms = df["firm"].unique()
    frames = {f: df[df["firm"] == f] for f in firms}
    ests: list[float] = []
    for _ in range(int(n_boot)):
        picks = rng.choice(firms, size=len(firms), replace=True)
        parts = []
        for j, f in enumerate(picks):
            g = frames[f].copy()
            # Re-label so a firm drawn twice counts as two independent clusters.
            g["firm"] = f"{f}__b{j}"
            g["filing"] = g["filing"] + f"__b{j}"
            parts.append(g)
        boot = pd.concat(parts, ignore_index=True)
        try:
            res = smf.ols(_formula(regressors), data=boot).fit()
            if coef not in res.params.index or not np.isfinite(res.params[coef]):
                raise ValueError(f"{coef} not estimable in this resample")
            ests.append(float(res.params[coef]))
        except Exception as exc:
            out["n_failed"] += 1
            key = type(exc).__name__
            out["failure_reasons"][key] = out["failure_reasons"].get(key, 0) + 1

    out["n_successful"] = len(ests)
    if ests:
        a = np.asarray(ests, dtype=float)
        out["mean"] = float(a.mean())
        out["sd"] = float(a.std(ddof=1)) if len(a) > 1 else None
        out["ci_low"] = float(np.percentile(a, 2.5))
        out["ci_high"] = float(np.percentile(a, 97.5))
        out["share_positive"] = float((a > 0).mean())
    out["status"] = "complete"
    return out


# ───────────────────────────────── driver ────────────────────────────────────


def run_w5(
    csv_path: str | Path,
    section: str = PRIMARY_SECTION,
    measure: str = PRIMARY_MEASURE,
    seed: int = 20261004,
    direction: str = PRIMARY_DIRECTION,
    position_as_factor: bool = False,
    n_boot: int = 0,
    reml: bool = True,
) -> W5Result:
    """Fit all three W5 specifications on cached window-level output."""
    df0, quality = load_window_frame(csv_path, section=section, measure=measure)
    df, regressors = prepare_design(df0, measure=measure,
                                    position_as_factor=position_as_factor)

    specs = {
        "nested": fit_nested(df, regressors, reml=reml),
        "firm_only": fit_firm_only(df, regressors, reml=reml),
        "ols_cluster": fit_ols_clustered(df, regressors),
    }

    # Residuals from the fixed-effect part, so the nested SS decomposition can
    # also be reported CONDITIONAL on the covariates (tier is between-firm, so
    # the unconditional between-firm share necessarily contains the tier effect).
    df = df.copy()
    try:
        df["y_resid"] = smf.ols(_formula(regressors), data=df).fit().resid
        decomp_resid = variance_decomposition(df, value_col="y_resid")
    except Exception as exc:  # pragma: no cover
        logger.error("Conditional variance decomposition unavailable: %s", exc)
        decomp_resid = {"error": f"{type(exc).__name__}: {exc}"}

    # Pre-stated rule, applied without reference to which spec passes.
    if specs["nested"].converged and SMALL_VS_LARGE in specs["nested"].coefficients:
        primary_key = "nested"
        reason = (
            "Specification (a) converged, so the W5 claim rests on it: it is the "
            "closest representation of the pre-registered (1 | firm/filing) that "
            "statsmodels MixedLM supports, and the only one of the three that "
            "yields both the firm-level and the filing-level ICC."
        )
    else:
        primary_key = "firm_only"
        reason = (
            "Specification (a) did not converge (or did not return the tier "
            "coefficient), so by the rule stated before the fit the W5 claim "
            "falls back to (b), the firm-only random intercept. (a)'s "
            "convergence diagnostics are reported in full and the filing-level "
            "ICC is NOT available from (b)."
        )

    design_summary = {
        "formula_preregistered": (
            "disagreement_w ~ tier + log(window_words) + position_decile "
            "+ form_type + (1 | firm/filing)"
        ),
        "fixed_effects_fitted": _formula(regressors),
        "tier_levels_present": sorted(df["tier"].astype(str).unique().tolist()),
        "tier_is_three_level": sorted(df["tier"].astype(str).unique().tolist()) ==
                               sorted(TIERS),
        "position_decile_counts": {
            int(k): int(v) for k, v in
            df["position_decile"].value_counts().sort_index().items()
        },
        "position_frac_range": [float(df["position_frac"].min()),
                                float(df["position_frac"].max())],
        "word_count_range": [int(df["word_count"].min()), int(df["word_count"].max())],
        "measure_mean": float(df["y"].mean()),
        "measure_sd": float(df["y"].std(ddof=1)) if len(df) > 1 else None,
        "measure_mean_by_tier": {
            t: float(df.loc[df["tier"] == t, "y"].mean())
            for t in TIERS if (df["tier"] == t).any()
        },
        "windows_per_filing": {
            "min": int(df.groupby("filing").size().min()),
            "median": float(df.groupby("filing").size().median()),
            "max": int(df.groupby("filing").size().max()),
        },
    }

    return W5Result(
        seed=seed,
        section=section,
        measure=measure,
        direction=direction,
        tier_reference=TIER_REFERENCE,
        form_reference=FORM_REFERENCE,
        position_term="10-level factor" if position_as_factor else "linear in decile 0-9",
        quality=quality,
        specs=specs,
        primary_spec_key=primary_key,
        primary_spec_reason=reason,
        variance_decomposition={
            "raw": variance_decomposition(df, value_col="y"),
            "conditional_on_fixed_effects": decomp_resid,
        },
        collinearity=collinearity_report(df, regressors),
        bootstrap=firm_cluster_bootstrap(df, regressors, SMALL_VS_LARGE, n_boot, seed),
        design_summary=design_summary,
    )


# ──────────────────────────────── reporting ──────────────────────────────────


def _fmt(x: Optional[float], nd: int = 4, width: int = 0) -> str:
    s = "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) \
        else f"{x:.{nd}f}"
    return s.rjust(width) if width else s


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100.0 * x:5.1f}%"


def format_report(r: W5Result) -> str:
    L: list[str] = []
    A = L.append

    A("W5 - window-level mixed model (ANALYSIS_PLAN.md section 5)")
    A("=" * 78)
    A("")
    A("Pre-registered specification:")
    A(f"  {r.design_summary['formula_preregistered']}")
    A("Pre-declared criterion:")
    A(f"  the Small-vs-Large tier coefficient CI must exclude 0 in the "
      f"{r.direction} direction")
    A(f"  (H1: smaller firms -> MORE inter-persona disagreement).")
    A("")
    A(f"Input file      : {r.quality.csv_path}")
    A(f"Section         : {r.section}"
      + ("  (the pre-registered primary section)" if r.section == PRIMARY_SECTION
         else "  (SECONDARY - reported with its own n, never pooled with MD&A)"))
    A(f"Measure         : {r.measure}"
      + ("  (primary)" if r.measure == PRIMARY_MEASURE else "  (secondary)"))
    A(f"Tier reference  : {r.tier_reference}  <- the sign of the Small-vs-Large "
      f"coefficient is relative to this")
    A(f"Form reference  : {r.form_reference}")
    A(f"Position term   : position_decile = floor(10 * chunk_index / "
      f"max(1, n_in_section - 1)), capped at 9; entered {r.position_term}")
    A(f"Seed            : {r.seed}  (the three fits are deterministic given the "
      f"input; the seed governs the optional cluster bootstrap)")
    A("")

    q = r.quality
    A("Sample actually used")
    A("-" * 78)
    A(f"  rows in file                     {q.n_rows_file}")
    A(f"  rows in section {r.section!r:<16} {q.n_rows_section}")
    A(f"  dropped, any_failed=True         {q.n_failed_dropped}"
      f"   (pre-registered rule 8.1: dropped, NEVER imputed)")
    A(f"  dropped, null {r.measure:<18} {q.n_null_measure_dropped}")
    A(f"  filings dropped by the 8.2 gate  {q.n_filings_dropped_gate}"
      f"   ({q.n_windows_dropped_gate} window(s) with them)")
    for s in q.filings_dropped_gate:
        A(f"      - {s}")
    A("")
    A(f"  WINDOWS used                     {q.n_windows_used}")
    A(f"  FILINGS used                     {q.n_filings_used}")
    A(f"  FIRMS used                       {q.n_firms_used}")
    A(f"  {'tier':<8}{'windows':>10}{'filings':>10}{'firms':>8}")
    for t in TIERS:
        A(f"  {t:<8}{q.windows_by_tier.get(t, 0):>10}"
          f"{q.filings_by_tier.get(t, 0):>10}{q.firms_by_tier.get(t, 0):>8}")
    A("")
    A(f"  tier is a {'3-level' if r.design_summary['tier_is_three_level'] else 'REDUCED'}"
      f" factor, levels present: {r.design_summary['tier_levels_present']}")
    A(f"  firms contributing a single filing: "
      f"{q.single_filing_firms if q.single_filing_firms else 'none'}")
    A(f"  windows per filing: min {r.design_summary['windows_per_filing']['min']}, "
      f"median {r.design_summary['windows_per_filing']['median']}, "
      f"max {r.design_summary['windows_per_filing']['max']}")
    A(f"  window word_count range: {r.design_summary['word_count_range']}")
    if q.served_by_values:
        A(f"  served_by backends present: {q.served_by_values}"
          + ("" if len(q.served_by_values) == 1
             else "   <- MORE THAN ONE BACKEND: the served-backend audit "
                  "(plan 8.4) fails; do not report these results"))
    if q.n_parse_errors_kept:
        A(f"  windows kept with any_parse_error=True: {q.n_parse_errors_kept} "
          f"(a parse error is not a failure; the plan drops only any_failed)")
    for m in q.n_in_section_mismatches:
        A(f"  INTEGRITY: {m}")
    A("")
    A(f"  raw window-level mean of {r.measure} by tier (no covariates, no nesting):")
    for t, v in r.design_summary["measure_mean_by_tier"].items():
        A(f"      {t:<8}{_fmt(v)}")
    A(f"      overall  {_fmt(r.design_summary['measure_mean'])}  "
      f"(sd {_fmt(r.design_summary['measure_sd'])})")
    A("")

    A("Three specifications, side by side")
    A("-" * 78)
    A("statsmodels MixedLM fits ONE grouping factor; (1 | firm/filing) is not")
    A("directly supported, so the nesting is expressed as a variance component")
    A("and all three specifications are reported rather than one presented as")
    A("if the software had fitted the formula literally.")
    A("")
    for key in ("nested", "firm_only", "ols_cluster"):
        s = r.specs[key]
        A(f"{s.label}")
        A(f"  engine    : {s.engine}")
        A(f"  inference : {s.inference}")
        A(f"  status    : {s.message}")
        A(f"  n_obs     : {s.n_obs}   groups (firms): {s.n_groups}")
        if s.coefficients:
            A(f"  {'term':<30}{'est':>10}{'se':>9}{'ci_low':>10}"
              f"{'ci_high':>10}{'p':>9}")
            for name, c in s.coefficients.items():
                A(f"  {COEF_LABELS.get(name, name):<30}"
                  f"{_fmt(c['estimate'], 4, 10)}{_fmt(c['se'], 4, 9)}"
                  f"{_fmt(c['ci_low'], 4, 10)}{_fmt(c['ci_high'], 4, 10)}"
                  f"{_fmt(c['p'], 4, 9)}")
        else:
            A("  no coefficients available (fit failed)")
        if any(v is not None for v in s.variances.values()):
            A("  variance components:")
            for k in ("firm", "filing_within_firm", "residual_within_filing"):
                v = s.variances.get(k)
                A(f"      {k:<26}{_fmt(v, 6, 12)}"
                  + ("" if v is None else f"   (sd {_fmt(math.sqrt(v), 4)})"))
        if any(v is not None for v in s.iccs.values()):
            A("  ICCs:")
            if s.iccs.get("icc_filing") is not None:
                A(f"      icc_filing  {_fmt(s.iccs['icc_filing'])}   "
                  f"= corr(two windows of the SAME filing)")
            if s.iccs.get("icc_firm") is not None:
                A(f"      icc_firm    {_fmt(s.iccs['icc_firm'])}   "
                  f"= corr(two windows of the same firm, DIFFERENT filings)")
            if s.iccs.get("icc_filing") is None and key == "firm_only":
                A("      icc_filing  n/a - not identified without a filing "
                  "variance component")
        crit = s.criterion(SMALL_VS_LARGE, r.direction)
        if crit is None:
            A("  criterion: NOT EVALUABLE (no usable fit)")
        else:
            A(f"  criterion (Small-vs-Large CI excludes 0, {r.direction}): "
              f"{'PASS' if crit else 'FAIL'}")
        if s.diagnostics:
            A(f"  diagnostics: {json.dumps(s.diagnostics, default=str)}")
        for w in s.warnings:
            A(f"  warning   : {w}")
        A("")

    A("Which specification the paper's W5 claim rests on")
    A("-" * 78)
    p = r.primary
    A(f"  PRIMARY: {p.label}")
    A(f"  {r.primary_spec_reason}")
    crit = r.criterion_met()
    A("")
    c = p.coefficients.get(SMALL_VS_LARGE)
    if c is not None:
        A(f"  Small-vs-{r.tier_reference}: {c['estimate']:+.4f} "
          f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  p={c['p']:.4f}")
    cm = p.coefficients.get(MID_VS_LARGE)
    if cm is not None:
        A(f"  Mid-vs-{r.tier_reference}:   {cm['estimate']:+.4f} "
          f"[{cm['ci_low']:+.4f}, {cm['ci_high']:+.4f}]  p={cm['p']:.4f}")
    A("")
    if crit is None:
        A("  W5 CRITERION: NOT EVALUABLE - no specification produced a usable "
          "tier coefficient.")
    elif crit:
        A("  W5 CRITERION: MET on the primary specification.")
    else:
        A("  W5 CRITERION: NOT MET on the primary specification. Under the "
          "section 5 overall gate")
        A("  the tier result is then reported as NOT ROBUST TO WINDOW COUNT, "
          "with the same")
        A("  prominence a positive finding would get (section 7).")
    agree = {k: s.criterion(SMALL_VS_LARGE, r.direction) for k, s in r.specs.items()}
    A(f"  criterion across all three specifications: {agree}")
    if len({v for v in agree.values() if v is not None}) > 1:
        A("  THE THREE SPECIFICATIONS DISAGREE. The primary is as stated above; "
          "the disagreement")
        A("  is a limitation to report, not a menu to choose from.")
    A("")

    A("Variance decomposition of window-level disagreement")
    A("-" * 78)
    A("  How much of the variance in window-level disagreement sits BETWEEN firms,")
    A("  BETWEEN filings within a firm, and WITHIN filings.  Reported three ways,")
    A("  because no single one of them is both assumption-free and unbiased:")
    A("")
    A("   (i) descriptive nested sums of squares on the raw measure - needs no")
    A("       model and no convergence, but the between-firm share also contains")
    A("       whatever the fixed effects explain (tier is a between-firm variable),")
    A("       and under unbalanced group sizes the higher-level shares are biased")
    A("       UPWARD because each group mean carries residual noise;")
    A("  (ii) the same decomposition on the residuals from the fixed-effect fit,")
    A("       which removes (i)'s first problem but not its second;")
    A(" (iii) the REML variance components from specification (a) - these are the")
    A("       actual estimates, and the ones to quote, when (a) converges.")
    A("")
    for label, key in (("(i) raw measure", "raw"),
                       ("(ii) residuals, conditional on the fixed effects",
                        "conditional_on_fixed_effects")):
        vd = r.variance_decomposition.get(key, {})
        A(f"  {label}")
        if "error" in vd:
            A(f"      unavailable: {vd['error']}")
            A("")
            continue
        A(f"      {'level':<34}{'SS':>12}{'share':>9}{'df':>8}{'MS':>12}")
        for lab, ss, sh, dfk, ms in (
            ("between firms", "ss_between_firms", "share_between_firms",
             "df_between_firms", "ms_between_firms"),
            ("between filings within a firm", "ss_between_filings_within_firm",
             "share_between_filings_within_firm", "df_between_filings_within_firm",
             "ms_between_filings_within_firm"),
            ("within filings (window to window)", "ss_within_filings",
             "share_within_filings", "df_within_filings", "ms_within_filings"),
        ):
            A(f"      {lab:<34}{_fmt(vd[ss], 5, 12)}{_pct(vd[sh]):>9}"
              f"{vd[dfk]:>8}{_fmt(vd[ms], 6, 12)}")
        A(f"      {'total':<34}{_fmt(vd['ss_total'], 5, 12)}{'100.0%':>9}")
        A(f"      additivity check |SS_total - sum| = "
          f"{vd['additivity_check_abs_error']:.3e}")
        A("")

    nested = r.specs["nested"]
    headline_share: Optional[float] = None
    headline_basis = ""
    if nested.converged:
        tot = nested.variances.get("total")
        A("  (iii) REML variance components from specification (a):")
        A(f"      {'level':<34}{'variance':>12}{'share':>9}{'sd':>10}")
        for k, lab in (("firm", "between firms"),
                       ("filing_within_firm", "between filings within a firm"),
                       ("residual_within_filing", "within filings")):
            v = nested.variances.get(k)
            A(f"      {lab:<34}{_fmt(v, 6, 12)}"
              f"{(_pct(v / tot) if (v is not None and tot) else 'n/a'):>9}"
              f"{(_fmt(math.sqrt(v), 4) if v is not None and v >= 0 else 'n/a'):>10}")
        if tot:
            headline_share = nested.variances["residual_within_filing"] / tot
            headline_basis = "REML components from specification (a)"
        A("")
        # How reliable is a filing-level mean, given these components?  A filing's
        # true level varies with (var_firm + var_filing); the mean of m windows
        # carries var_resid / m of noise on top of it.  This is the number that
        # decides whether filing-level means are worth reporting at all.
        vb = (nested.variances.get("firm") or 0.0) + \
             (nested.variances.get("filing_within_firm") or 0.0)
        ve = nested.variances.get("residual_within_filing") or 0.0
        if vb > 0 and ve > 0:
            med = r.design_summary["windows_per_filing"]["median"]
            A("  Reliability of a filing-level mean implied by those components")
            A("  (= between-filing variance / (between-filing + within/m)):")
            for m in sorted({2.0, float(med),
                             float(r.design_summary['windows_per_filing']['max'])}):
                rel = vb / (vb + ve / m)
                tag = ""
                if abs(m - 2.0) < 1e-9:
                    tag = "  <- the sample minimum"
                elif abs(m - float(med)) < 1e-9:
                    tag = "  <- the sample median"
                A(f"      m = {m:>5.1f} windows:  reliability {rel:.3f}{tag}")
            A("")
    else:
        A("  (iii) REML components unavailable: specification (a) did not converge,")
        A("        so the decomposition below rests on the descriptive shares.")
        A("")
        raw = r.variance_decomposition.get("raw", {})
        headline_share = raw.get("share_within_filings")
        headline_basis = "descriptive SS shares on the raw measure"

    if headline_share is not None:
        w = headline_share
        A(f"  SUBSTANTIVE READ (basis: {headline_basis})")
        if w >= 0.70:
            A(f"      {_pct(w)} of the variance in window-level disagreement sits")
            A("      WITHIN filings.  Filing-level means are therefore mostly")
            A("      averaging noise, and the paper should say so explicitly: a")
            A("      filing's headline number is a weak summary of a highly")
            A("      variable within-document quantity, and the firm-level means")
            A("      the confirmatory test uses inherit that noise.")
        elif w >= 0.50:
            A(f"      {_pct(w)} of the variance is within filings - the single")
            A("      largest component.  Filing-level means carry real but limited")
            A("      information and their uncertainty should be reported with them.")
        else:
            A(f"      only {_pct(w)} of the variance is within filings, so most of")
            A("      the variation is between documents and between firms;")
            A("      filing-level means are a meaningful summary of a filing.")
    A("")

    co = r.collinearity
    A("Design diagnostics a referee will ask for")
    A("-" * 78)
    A(f"  windows of exactly {CHUNK_WORDS} words: "
      f"{co['n_windows_exactly_chunk_words']} of {q.n_windows_used}; "
      f"final-in-section windows: {co['n_windows_final_in_section']}")
    A("  (every window but a section's last is exactly one CHUNK_WORDS window, so")
    A("   log(window_words) varies almost only at the last, highest-position")
    A("   window - the two covariates are related by construction)")
    if "correlations" in co:
        names = co["regressors"]
        A(f"  {'':<20}" + "".join(f"{n[:16]:>18}" for n in names))
        for a in names:
            A(f"  {a[:20]:<20}"
              + "".join(f"{co['correlations'][a][b]:>18.3f}" for b in names))
    if "vif" in co:
        A("  VIF: " + ", ".join(f"{k}={v:.2f}" for k, v in co["vif"].items()))
    elif "vif_error" in co:
        A(f"  VIF unavailable: {co['vif_error']}")
    A(f"  position_decile counts: {r.design_summary['position_decile_counts']}")
    A("")

    b = r.bootstrap
    if b.get("status") != "not run (n_boot=0)":
        A("Firm-level cluster bootstrap of the Small-vs-Large OLS coefficient")
        A("-" * 78)
        A(f"  seed {b['seed']}, {b['n_requested']} requested, "
          f"{b['n_successful']} fitted, {b['n_failed']} failed "
          f"{b['failure_reasons'] or ''}")
        if b.get("ci_low") is not None:
            A(f"  mean {_fmt(b['mean'])}  sd {_fmt(b['sd'])}  "
              f"percentile CI [{_fmt(b['ci_low'])}, {_fmt(b['ci_high'])}]  "
              f"share > 0: {_pct(b['share_positive'])}")
        A("  Robustness companion only - the pre-declared criterion is evaluated")
        A("  on the primary specification above, not here.")
        A("")

    A("Limitations of W5 as specified")
    A("-" * 78)
    A("  * (1 | firm/filing) is approximated, not fitted literally: MixedLM has")
    A("    one grouping factor and expresses the filing level as a variance")
    A("    component inside the firm group.")
    A("  * MixedLM CIs are Wald/normal; no Satterthwaite or Kenward-Roger")
    A("    correction is available, so with ~30 firms they are optimistic.")
    A(f"  * {q.n_firms_used} clusters is few for cluster-robust inference in (c).")
    A("  * W5 is one row of the section 5 battery. The section 5 overall gate")
    A("    requires W3 and W5 to pass and W2 not to be contradicted; W5 alone")
    A("    neither establishes nor refutes H1.")
    A("  * The confirmatory unit of analysis remains the FIRM (n=30, test C1).")
    A("    W5 is a window-level diagnostic, not the confirmatory test.")
    return "\n".join(L)


def coefficient_frame(r: W5Result) -> pd.DataFrame:
    """Long-format coefficient table across the three specifications."""
    rows = []
    for key, s in r.specs.items():
        for name, c in s.coefficients.items():
            rows.append({
                "section": r.section,
                "measure": r.measure,
                "spec": key,
                "spec_label": s.label,
                "is_primary_spec": key == r.primary_spec_key,
                "converged": s.converged,
                "term": name,
                "term_label": COEF_LABELS.get(name, name),
                "tier_reference": r.tier_reference,
                "estimate": c["estimate"],
                "se": c["se"],
                "ci_low": c["ci_low"],
                "ci_high": c["ci_high"],
                "p": c["p"],
                "n_obs": s.n_obs,
                "n_firms": r.quality.n_firms_used,
                "n_filings": r.quality.n_filings_used,
                "icc_firm": s.iccs.get("icc_firm"),
                "icc_filing": s.iccs.get("icc_filing"),
                "var_firm": s.variances.get("firm"),
                "var_filing_within_firm": s.variances.get("filing_within_firm"),
                "var_residual": s.variances.get("residual_within_filing"),
                "seed": r.seed,
            })
    return pd.DataFrame(rows)


# ────────────────────────────────── CLI ──────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="W5 window-level mixed model (pre-registered; cached output only)."
    )
    ap.add_argument("--chunks", default=None,
                    help="Path to chunk_level_results.csv "
                         "(default: <project>/output/chunk_level_results.csv)")
    ap.add_argument("--section", default=PRIMARY_SECTION, choices=["mda", "risk"],
                    help="mda is the pre-registered primary section")
    ap.add_argument("--measure", default=PRIMARY_MEASURE,
                    choices=["embedding_disagreement", "numeric_disagreement"])
    ap.add_argument("--seed", type=int, default=20261004,
                    help="Seed for the optional cluster bootstrap; recorded in output")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="Firm-cluster bootstrap replications (0 = off)")
    ap.add_argument("--position-as-factor", action="store_true",
                    help="Enter position_decile as a 10-level factor instead of linear")
    ap.add_argument("--ml", action="store_true",
                    help="Fit by ML instead of the default REML")
    ap.add_argument("--out-prefix", default=None,
                    help="Output path prefix (default: <project>/output/w5_mixed_model)")
    ap.add_argument("--no-write", action="store_true",
                    help="Print the report without writing any file")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    )

    project_root = Path(__file__).resolve().parents[1]
    chunks = Path(args.chunks) if args.chunks else \
        project_root / "output" / "chunk_level_results.csv"
    prefix = Path(args.out_prefix) if args.out_prefix else \
        project_root / "output" / "w5_mixed_model"

    try:
        result = run_w5(
            chunks,
            section=args.section,
            measure=args.measure,
            seed=args.seed,
            position_as_factor=args.position_as_factor,
            n_boot=args.bootstrap,
            reml=not args.ml,
        )
    except (FileNotFoundError, ValueError) as exc:
        # A missing or unusable input is a message, not a traceback. W5 never
        # fabricates data to fit.
        raise SystemExit(str(exc)) from None
    report = format_report(result)
    print("\n" + report)

    if not args.no_write:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        suffix = "" if (args.section == PRIMARY_SECTION
                        and args.measure == PRIMARY_MEASURE) \
            else f"_{args.section}_{args.measure.split('_')[0]}"
        txt = prefix.with_name(prefix.name + suffix + ".txt")
        csv = prefix.with_name(prefix.name + suffix + ".csv")
        js = prefix.with_name(prefix.name + suffix + ".json")
        txt.write_text(report + "\n", encoding="utf-8")
        coefficient_frame(result).to_csv(csv, index=False)
        js.write_text(json.dumps(result.as_dict(), indent=2, default=str),
                      encoding="utf-8")
        logger.info("wrote %s, %s, %s", txt, csv, js)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
