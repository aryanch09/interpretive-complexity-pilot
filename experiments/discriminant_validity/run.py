"""
D1 — Discriminant validity.

The obvious objection to any new text measure: it is just length, readability or
tone wearing a new name. This tests that directly, by correlating filing-level
persona disagreement with each of those, and then asking whether the size-tier
relationship survives controlling for all of them at once.

Uses the REAL Loughran-McDonald Master Dictionary (see analysis/lm_dictionary.py
for why that matters — the list previously vendored under the name `lm_neg_pct`
covers 12.4% of L-M's Negative category and includes 32 words L-M assigns
elsewhere or not at all).

Labelled DESCRIPTIVE in ANALYSIS_PLAN.md section 6. It cannot convert a failed
confirmatory test into a supported one. A tier effect that vanishes under these
controls is a finding to report, not a result to suppress.

No model or API calls: text comes from the cached filings, disagreement from the
scored window-level output.

Run:  python3 -m experiments.discriminant_validity.run
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.modules.setdefault("tensorflow", None)

from analysis.lm_dictionary import (
    DictionaryUnavailable,
    category_shares,
    compare_with_legacy_list,
)
from analysis.trend_tests import MIN_USABLE_CHUNKS, TIER_ORDER
from config import Config

logger = logging.getLogger(__name__)

CONTROLS = ["log_mda_words", "fog_index", "lm_negative", "lm_uncertainty",
            "lm_litigious", "type_token_ratio"]


def _filing_measures(chunk_df: pd.DataFrame, section: str, measure: str) -> pd.DataFrame:
    """Filing-level disagreement under the pre-registered rules."""
    d = chunk_df[chunk_df["section"] == section].copy()
    failed = d["any_failed"].astype(str).str.lower().isin({"true", "1"})
    d = d[~failed]
    d = d[d[measure].notna()]
    rows = []
    for (tic, form, date), g in d.groupby(["ticker", "filing_type", "filing_date"]):
        total = int(g["n_in_section"].iloc[0]) if "n_in_section" in g else len(g)
        if len(g) < MIN_USABLE_CHUNKS or len(g) < total / 2:
            continue
        rows.append({"ticker": tic, "filing_type": form, "filing_date": str(date),
                     "tier": g["tier"].iloc[0], measure: float(g[measure].mean()),
                     "n_windows": int(len(g))})
    return pd.DataFrame(rows)


def _text_features(config: Config, keys: pd.DataFrame) -> pd.DataFrame:
    """Re-extract each filing's MD&A offline and compute the control variables."""
    from analysis.textual_baselines import compute_textual_features
    from data.edgar import _html_to_text, extract_sections_dict

    rows = []
    for _, r in keys.iterrows():
        date = str(r["filing_date"]).replace("-", "")
        path = config.FILINGS_DIR / f"{r['ticker']}_{r['filing_type']}_{date}.txt"
        if not path.exists():
            logger.warning("missing cached filing %s", path.name)
            continue
        sections, _ = extract_sections_dict(
            _html_to_text(path.read_text(encoding="utf-8", errors="ignore")),
            r["filing_type"],
        )
        mda = sections.get("mda") or ""
        if not mda:
            continue
        feats = compute_textual_features(mda)
        lm = category_shares(mda, config.BASE_DIR)
        n_words = len(mda.split())
        rows.append({
            "ticker": r["ticker"], "filing_type": r["filing_type"],
            "filing_date": str(r["filing_date"]),
            "mda_words": n_words,
            "log_mda_words": float(np.log(max(n_words, 1))),
            "fog_index": feats.get("fog_index"),
            "type_token_ratio": feats.get("type_token_ratio"),
            "lm_negative": lm.get("lm_negative"),
            "lm_uncertainty": lm.get("lm_uncertainty"),
            "lm_litigious": lm.get("lm_litigious"),
            "lm_positive": lm.get("lm_positive"),
            # The legacy column, kept so the two can be compared explicitly.
            "legacy_lm_neg_pct": feats.get("lm_neg_pct"),
        })
    return pd.DataFrame(rows)


def run(config: Config, chunk_path: Path | None = None, section: str = "mda",
        measure: str = "embedding_disagreement") -> dict:
    chunk_path = chunk_path or (config.OUTPUT_DIR / "chunk_level_results.csv")
    if not chunk_path.exists():
        raise FileNotFoundError(f"{chunk_path} not found — run the scoring pipeline first.")
    chunk_df = pd.read_csv(chunk_path)

    fil = _filing_measures(chunk_df, section, measure)
    num = _filing_measures(chunk_df, section, "numeric_disagreement")
    feats = _text_features(config, fil[["ticker", "filing_type", "filing_date"]])

    df = fil.merge(feats, on=["ticker", "filing_type", "filing_date"], how="inner")
    df = df.merge(num[["ticker", "filing_type", "filing_date", "numeric_disagreement"]],
                  on=["ticker", "filing_type", "filing_date"], how="left")

    out_dir = config.OUTPUT_DIR / "experiments" / "discriminant_validity"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "filing_features.csv", index=False)

    results: dict = {
        "section": section, "measure": measure, "n_filings": int(len(df)),
        "dictionary": compare_with_legacy_list(config.BASE_DIR),
        "correlations": {},
    }

    # Simple correlations against each candidate confound.
    for target in (measure, "numeric_disagreement"):
        if target not in df.columns:
            continue
        block = {}
        for c in CONTROLS + ["legacy_lm_neg_pct"]:
            if c not in df.columns:
                continue
            ok = df[c].notna() & df[target].notna()
            if ok.sum() < 3:
                continue
            rho, p = spearmanr(df.loc[ok, c], df.loc[ok, target])
            block[c] = {"spearman_rho": float(rho), "p": float(p), "n": int(ok.sum())}
        results["correlations"][target] = block

    # Does the tier relationship survive the controls?
    try:
        import statsmodels.formula.api as smf

        d = df.dropna(subset=[measure] + CONTROLS).copy()
        d["tier"] = pd.Categorical(d["tier"], categories=list(TIER_ORDER), ordered=True)
        formula = f"{measure} ~ C(tier) + " + " + ".join(CONTROLS)
        model = smf.ols(formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["ticker"]}
        )
        results["ols"] = {
            "formula": formula,
            "n_obs": int(model.nobs),
            "n_clusters": int(d["ticker"].nunique()),
            "se": "clustered by firm",
            "reference_tier": TIER_ORDER[0],
            "r_squared": float(model.rsquared),
            "coefficients": {
                k: {"estimate": float(model.params[k]),
                    "se": float(model.bse[k]),
                    "p": float(model.pvalues[k]),
                    "ci": [float(model.conf_int().loc[k, 0]),
                           float(model.conf_int().loc[k, 1])]}
                for k in model.params.index
            },
        }
        tier_terms = [k for k in model.params.index if k.startswith("C(tier)")]
        results["ols"]["tier_survives_controls"] = bool(
            any(model.pvalues[k] < 0.05 for k in tier_terms)
        )
    except Exception as exc:  # noqa: BLE001
        results["ols"] = {"error": str(exc)}
        logger.exception("OLS failed")

    (out_dir / "results.json").write_text(
        json.dumps(results, indent=1, default=str), encoding="utf-8")
    return results


def write_report(config: Config, **kw) -> str:
    r = run(config, **kw)
    L: list[str] = []
    L.append("D1 — Discriminant validity")
    L.append("=" * 78)
    L.append("")
    L.append(f"Filings: {r['n_filings']}   Section: {r['section']}   "
             f"Measure: {r['measure']}")
    L.append("")
    L.append("Dictionary note")
    L.append("-" * 78)
    dic = r.get("dictionary", {})
    if "coverage_of_lm_negative" in dic:
        L.append(f"  These use the real Loughran-McDonald Master Dictionary "
                 f"({dic['lm_negative_size']:,} negative words).")
        L.append(f"  The list previously reported as 'lm_neg_pct' covers "
                 f"{dic['coverage_of_lm_negative']*100:.1f}% of it and contains "
                 f"{dic['legacy_words_not_in_lm_negative']} words L-M does not "
                 f"classify as negative.")
        L.append(f"  Both are reported below so the difference is visible.")
    L.append("")
    L.append("Correlations with candidate confounds (Spearman)")
    L.append("-" * 78)
    for target, block in r.get("correlations", {}).items():
        L.append(f"  {target}")
        for c, v in block.items():
            flag = " *" if v["p"] < 0.05 else ""
            L.append(f"    {c:22s} rho = {v['spearman_rho']:+.3f}  "
                     f"p = {v['p']:.4f}  n = {v['n']}{flag}")
        L.append("")
    ols = r.get("ols", {})
    L.append("Does the tier relationship survive the controls?")
    L.append("-" * 78)
    if "error" in ols:
        L.append(f"  OLS failed: {ols['error']}")
    else:
        L.append(f"  {ols['formula']}")
        L.append(f"  n = {ols['n_obs']} filings, {ols['n_clusters']} firm clusters, "
                 f"SEs {ols['se']}, R^2 = {ols['r_squared']:.3f}")
        L.append(f"  Reference tier: {ols['reference_tier']}")
        L.append("")
        for k, v in ols["coefficients"].items():
            if k == "Intercept":
                continue
            L.append(f"    {k:28s} {v['estimate']:+.5f} "
                     f"[{v['ci'][0]:+.5f}, {v['ci'][1]:+.5f}]  p = {v['p']:.4f}")
        L.append("")
        L.append(f"  Tier term significant at 0.05 after controls: "
                 f"{ols.get('tier_survives_controls')}")
    L.append("")
    L.append("  Descriptive, not confirmatory. If the tier term does not survive these")
    L.append("  controls, that is reported as what it is: the measure covaries with")
    L.append("  length, readability or tone strongly enough that size adds little")
    L.append("  beyond them in this sample.")
    L.append("")
    text = "\n".join(L)
    (config.OUTPUT_DIR / "experiments" / "discriminant_validity" / "report.txt").write_text(
        text, encoding="utf-8")
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        print(write_report(Config()))
    except DictionaryUnavailable as exc:
        print(exc)
        raise SystemExit(1)
