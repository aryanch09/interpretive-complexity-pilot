"""
The pre-registered secondary family (ANALYSIS_PLAN.md section 6).

Three tests, TWO-SIDED, alpha = 0.05, Holm-corrected WITHIN this family only:

  S1  tier trend in the NUMERIC measure, firm level
  S2  tier trend in RISK FACTORS embedding disagreement, on the filings that
      have Risk windows, with its own n stated
  S3  tier trend in the WITHIN-FILING SD of embedding disagreement

This was specified before scoring and must be reported whatever it shows. It is
run here as a single unit so the correction is applied to the family the plan
defined, rather than to whichever tests happened to get run.

Note on sidedness: the confirmatory test C1 is one-sided in the direction
Small > Mid > Large, because the hypothesis is directional. The secondary family
is explicitly TWO-SIDED in the plan, so a trend in either direction counts. That
matters here, because the numeric measure runs the wrong way and a one-sided
test would hide it behind a p near 1.

Zero model calls.

Run:  python3 -m analysis.secondary_family
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("tensorflow", None)

from analysis.trend_tests import (
    DEFAULT_SEED,
    MIN_USABLE_CHUNKS,
    TIER_ORDER,
    aggregate_to_firm,
    holm,
    jonckheere_terpstra_labeled,
)
from config import Config

logger = logging.getLogger(__name__)


def _two_sided_jt(firm: pd.DataFrame, col: str = "firm_mean") -> dict:
    """JT in both directions; the two-sided p is 2*min(one-sided), capped at 1."""
    up = jonckheere_terpstra_labeled(firm[col].to_numpy(), firm.tier.to_numpy(),
                                     group_order=TIER_ORDER, alternative="increasing")
    dn = jonckheere_terpstra_labeled(firm[col].to_numpy(), firm.tier.to_numpy(),
                                     group_order=TIER_ORDER, alternative="decreasing")
    p_two = min(1.0, 2.0 * min(float(up.p_value), float(dn.p_value)))
    means = {t: float(firm[firm.tier == t][col].mean()) for t in TIER_ORDER}
    return {
        "tier_means": means,
        "gradient_small_minus_large": means["small"] - means["large"],
        "monotone_increasing": bool(means["large"] < means["mid"] < means["small"]),
        "monotone_decreasing": bool(means["large"] > means["mid"] > means["small"]),
        "jt_statistic": float(up.statistic), "jt_z": float(up.z),
        "p_one_sided_increasing": float(up.p_value),
        "p_one_sided_decreasing": float(dn.p_value),
        "p_two_sided": p_two,
        "n_firms": int(len(firm)),
    }


def _within_filing_sd(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Firm-level mean of each filing's within-filing SD of window disagreement."""
    d = chunk_df[(chunk_df.section == "mda")
                 & (~chunk_df.any_failed.astype(str).str.lower().isin(["true", "1"]))
                 & chunk_df.embedding_disagreement.notna()]
    rows = []
    for (tic, form, date), g in d.groupby(["ticker", "filing_type", "filing_date"]):
        if len(g) < 2:          # SD undefined on one window; dropped, never imputed
            continue
        rows.append({"ticker": tic, "tier": g.tier.iloc[0],
                     "sd": float(g.embedding_disagreement.std(ddof=1))})
    f = pd.DataFrame(rows)
    return (f.groupby(["ticker", "tier"], as_index=False).sd.mean()
             .rename(columns={"sd": "firm_mean"}))


def run(config: Config, seed: int = DEFAULT_SEED) -> dict:
    ch = pd.read_csv(config.OUTPUT_DIR / "chunk_level_results.csv")

    # S1 — numeric measure, MD&A (the primary section), firm level.
    s1_firm = aggregate_to_firm(ch, section="mda", measure="numeric_disagreement")
    s1 = _two_sided_jt(s1_firm)
    s1["description"] = "tier trend in numeric disagreement (MD&A, firm level)"

    # S2 — Risk Factors embedding disagreement, only filings that have Risk windows.
    s2_firm = aggregate_to_firm(ch, section="risk",
                                measure="embedding_disagreement")
    s2 = _two_sided_jt(s2_firm)
    risk = ch[(ch.section == "risk")
              & (~ch.any_failed.astype(str).str.lower().isin(["true", "1"]))]
    s2["n_filings_with_risk"] = int(
        risk.groupby(["ticker", "filing_type", "filing_date"]).ngroups)
    s2["description"] = ("tier trend in Risk Factors embedding disagreement, "
                         "filings with Risk windows only")

    # S3 — within-filing SD of embedding disagreement.
    s3_firm = _within_filing_sd(ch)
    s3 = _two_sided_jt(s3_firm)
    s3["description"] = "tier trend in within-filing SD of embedding disagreement"

    tests = {"S1_numeric": s1, "S2_risk_factors": s2, "S3_within_filing_sd": s3}
    names = list(tests)
    pvals = [tests[n]["p_two_sided"] for n in names]
    h = holm(pvals, alpha=0.05)

    for i, n in enumerate(names):
        tests[n]["p_holm_adjusted"] = float(h.adjusted[i])
        tests[n]["reject_at_05"] = bool(h.reject[i])

    res = {
        "seed": seed,
        "family": "ANALYSIS_PLAN.md section 6 — two-sided, Holm within family",
        "alpha": 0.05,
        "n_tests": len(names),
        "tests": tests,
        "any_rejected": bool(any(h.reject)),
    }
    out = config.OUTPUT_DIR / "experiments" / "secondary_family"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str),
                                      encoding="utf-8")
    txt = report(res)
    (out / "report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return res


def report(r: dict) -> str:
    L = ["Pre-registered secondary family (ANALYSIS_PLAN.md section 6)", "=" * 78, ""]
    L.append("Two-sided, alpha = 0.05, Holm-corrected within this family of three.")
    L.append("Specified before scoring; reported whatever it shows.")
    L.append("")
    for name, t in r["tests"].items():
        L.append(f"{name}  —  {t['description']}")
        L.append("-" * 78)
        mu = t["tier_means"]
        L.append(f"  large {mu['large']:.4f}   mid {mu['mid']:.4f}   "
                 f"small {mu['small']:.4f}   n = {t['n_firms']} firms")
        if "n_filings_with_risk" in t:
            L.append(f"  filings contributing: {t['n_filings_with_risk']}")
        direction = ("increasing (small highest)" if t["monotone_increasing"]
                     else "decreasing (small lowest)" if t["monotone_decreasing"]
                     else "non-monotone")
        L.append(f"  gradient (small - large) = {t['gradient_small_minus_large']:+.4f}   "
                 f"{direction}")
        L.append(f"  two-sided p = {t['p_two_sided']:.4f}   "
                 f"Holm-adjusted p = {t['p_holm_adjusted']:.4f}   "
                 f"reject at 0.05: {t['reject_at_05']}")
        L.append("")
    L.append(f"Any test rejected after correction: {r['any_rejected']}")
    L.append("")
    L.append("  The family is two-sided by design. The numeric measure runs AGAINST the")
    L.append("  hypothesis, and a one-sided test in the hypothesised direction would")
    L.append("  bury that behind a p near 1 rather than report it.")
    return "\n".join(L)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(0 if run(Config()) else 1)
