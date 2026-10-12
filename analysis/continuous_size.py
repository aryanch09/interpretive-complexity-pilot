"""
Continuous size test — does the null survive a better size variable?

The pre-registered confirmatory test uses a three-level size TIER, because
point-in-time CRSP/Compustat values were not available. Three ordered bins throw
away most of the variation in firm size, and the obvious objection to a null
result is that the size variable was too coarse to see anything.

This answers that objection with the filer's OWN reported public float
(dei:EntityPublicFloat from the 10-K cover page, via the SEC XBRL API) -- an
SEC-filed value, not a reconstruction from market data, which is what the
project's rules require. Across the 30 firms it spans $27.6M to $3.6T.

STATUS: EXPLORATORY. The plan fixed tier as the independent variable and this is
not a substitute for the confirmatory test; it is a power check on the null. A
null that also holds against a continuous variable with five orders of magnitude
of variation is a considerably stronger null than one that holds across three
bins.

Zero model calls.

Run:  python3 -m analysis.continuous_size
"""
from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.modules.setdefault("tensorflow", None)

from analysis.trend_tests import DEFAULT_SEED, TIER_ORDER, aggregate_to_firm, bootstrap_ci
from config import Config

logger = logging.getLogger(__name__)


def run(config: Config, seed: int = DEFAULT_SEED) -> dict:
    flo = pd.read_csv(config.OUTPUT_DIR / "firm_public_float.csv").dropna(
        subset=["float_usd"])
    ch = pd.read_csv(config.OUTPUT_DIR / "chunk_level_results.csv")

    res: dict = {"seed": seed, "status": "EXPLORATORY — the pre-registered IV is tier",
                 "n_firms_with_float": int(len(flo)),
                 "float_range_usd": [float(flo.float_usd.min()),
                                     float(flo.float_usd.max())],
                 "measures": {}}

    for label, section, measure in (
        ("mda_embedding", "mda", "embedding_disagreement"),
        ("mda_numeric", "mda", "numeric_disagreement"),
        ("risk_embedding", "risk", "embedding_disagreement"),
    ):
        firm = aggregate_to_firm(ch, section=section, measure=measure)
        m = firm.merge(flo[["ticker", "float_usd"]], on="ticker", how="inner")
        if len(m) < 5:
            continue
        logf = np.log10(m.float_usd.to_numpy())
        y = m.firm_mean.to_numpy()
        rs, ps = spearmanr(m.float_usd, y)
        rp, pp = pearsonr(logf, y)
        slope = float(np.polyfit(logf, y, 1)[0])

        # bootstrap_ci resamples a 1-D array; resample firm INDICES so the
        # (float, measure) pairing is preserved, then re-correlate.
        rng = np.random.default_rng(seed)
        n = len(m)
        boots = []
        for _ in range(10000):
            idx = rng.integers(0, n, n)
            if len(np.unique(_y_idx := y[idx])) < 2:
                continue
            boots.append(float(spearmanr(logf[idx], _y_idx).statistic))
        boots = np.array([b for b in boots if np.isfinite(b)])
        boot_lo = float(np.percentile(boots, 2.5)) if len(boots) else float("nan")
        boot_hi = float(np.percentile(boots, 97.5)) if len(boots) else float("nan")
        res["measures"][label] = {
            "n_firms": int(len(m)),
            "spearman_rho": float(rs), "spearman_p": float(ps),
            "spearman_ci": [boot_lo, boot_hi],
            "pearson_log10float_r": float(rp), "pearson_p": float(pp),
            "slope_per_decade_of_float": slope,
            "interpretation": (
                f"a 10x increase in public float moves firm-level "
                f"{measure} by {slope:+.4f}"),
        }

    # Sanity: does tier actually order float the way the design assumes?
    firm = aggregate_to_firm(ch, section="mda", measure="embedding_disagreement")
    m = firm.merge(flo[["ticker", "float_usd"]], on="ticker", how="inner")
    res["tier_vs_float"] = {
        t: {"n": int((m.tier == t).sum()),
            "median_float_usd": float(m[m.tier == t].float_usd.median())}
        for t in TIER_ORDER}
    rank = {"small": 1, "mid": 2, "large": 3}
    rr, rp_ = spearmanr(m.tier.map(rank), m.float_usd)
    res["tier_float_consistency"] = {"spearman_rho": float(rr), "p": float(rp_)}

    out = config.OUTPUT_DIR / "experiments" / "continuous_size"
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "firm_float_and_measure.csv", index=False)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str),
                                      encoding="utf-8")
    txt = report(res)
    (out / "report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return res


def report(r: dict) -> str:
    lo, hi = r["float_range_usd"]
    L = ["Continuous size test — public float instead of three tiers", "=" * 78, ""]
    L.append(f"{r['n_firms_with_float']} firms, public float ${lo:,.0f} to ${hi:,.0f} "
             f"({np.log10(hi/lo):.1f} orders of magnitude).")
    L.append(f"{r['status']}")
    L.append("")
    L.append("Sanity: does tier order float as the design assumes?")
    L.append("-" * 78)
    for t, v in r["tier_vs_float"].items():
        L.append(f"  {t:6s} n={v['n']:2d}  median float ${v['median_float_usd']:,.0f}")
    c = r["tier_float_consistency"]
    L.append(f"  Spearman(tier rank, float) = {c['spearman_rho']:+.3f}, p = {c['p']:.3g}")
    L.append("")
    L.append("Disagreement against public float")
    L.append("-" * 78)
    for k, v in r["measures"].items():
        L.append(f"  {k}  (n = {v['n_firms']} firms)")
        L.append(f"    Spearman rho = {v['spearman_rho']:+.3f}  p = {v['spearman_p']:.4f}  "
                 f"CI [{v['spearman_ci'][0]:+.3f}, {v['spearman_ci'][1]:+.3f}]")
        L.append(f"    per decade of float: {v['slope_per_decade_of_float']:+.4f}")
        L.append("")
    L.append("  The hypothesis predicts a NEGATIVE correlation: smaller firms, more")
    L.append("  interpretive complexity. A null here cannot be blamed on the size")
    L.append("  variable being three coarse bins, which is the first objection to the")
    L.append("  pre-registered result.")
    return "\n".join(L)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(Config())
