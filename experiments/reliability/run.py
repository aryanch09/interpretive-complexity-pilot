"""
R1 — Reliability of the disagreement measure.

Two questions a referee will ask about any new measure, and that LLM-measure
papers rarely answer:

  (i)  SPLIT-HALF: within a single filing, do odd-numbered and even-numbered
       scoring windows give the same answer? If they do not, a filing-level mean
       is not measuring a property of the filing.
  (ii) TEST-RETEST ACROSS DOCUMENTS: is a firm's disagreement stable between its
       10-K and its 10-Q? If interpretive complexity is a firm property rather
       than a document accident, these should be clearly positively correlated.

Both run on cached window-level output. No model calls, no network.

A caveat stated up front, because it cuts the other way too: a LOW test-retest
correlation is not automatically a failure of the measure. A 10-K and a 10-Q are
different documents with different content and very different lengths, so some
of the gap is real cross-document variation rather than measurement noise. What
split-half isolates is closer to pure measurement reliability; test-retest
bounds how much of the measure is a stable firm characteristic.

Run:  python3 -m experiments.reliability.run
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.modules.setdefault("tensorflow", None)

from analysis.trend_tests import (
    DEFAULT_SEED,
    MIN_USABLE_CHUNKS,
    bootstrap_ci,
    spearman_brown,
)
from config import Config

logger = logging.getLogger(__name__)

OUT_DIR_NAME = "reliability"


def _surviving(chunk_df: pd.DataFrame, section: str, measure: str) -> pd.DataFrame:
    """Windows that survive the pre-registered partial-failure rule."""
    d = chunk_df[chunk_df["section"] == section].copy()
    failed = d["any_failed"].astype(str).str.lower().isin({"true", "1"})
    d = d[~failed]
    return d[d[measure].notna()]


def split_half(chunk_df: pd.DataFrame, section: str, measure: str,
               seed: int = DEFAULT_SEED) -> dict:
    """
    Odd vs even windows within each filing, correlated across filings, then
    Spearman-Brown corrected for half-length.

    Only filings with at least 2*MIN_USABLE_CHUNKS surviving windows are used, so
    each half independently clears the usability gate the plan sets for a filing.
    """
    d = _surviving(chunk_df, section, measure)
    key = ["ticker", "filing_type", "filing_date"]

    rows = []
    dropped = []
    for k, g in d.groupby(key, sort=True):
        g = g.sort_values("chunk_index")
        if len(g) < 2 * MIN_USABLE_CHUNKS:
            dropped.append({"filing": "_".join(map(str, k)), "n_windows": int(len(g)),
                            "reason": f"fewer than {2 * MIN_USABLE_CHUNKS} surviving windows"})
            continue
        odd = g[g["chunk_index"] % 2 == 1][measure]
        even = g[g["chunk_index"] % 2 == 0][measure]
        if len(odd) < MIN_USABLE_CHUNKS or len(even) < MIN_USABLE_CHUNKS:
            dropped.append({"filing": "_".join(map(str, k)), "n_windows": int(len(g)),
                            "reason": "a half fell below the usability gate"})
            continue
        rows.append({
            "ticker": k[0], "filing_type": k[1], "filing_date": k[2],
            "tier": g["tier"].iloc[0],
            "n_windows": int(len(g)),
            "half_even": float(even.mean()),
            "half_odd": float(odd.mean()),
        })

    hdf = pd.DataFrame(rows)
    out: dict = {
        "section": section, "measure": measure, "seed": seed,
        "n_filings_used": int(len(hdf)),
        "n_filings_dropped": len(dropped),
        "filings_dropped": dropped,
        "gate": f"a filing needs >= {2 * MIN_USABLE_CHUNKS} surviving windows, "
                f"and each half >= {MIN_USABLE_CHUNKS}",
    }
    if len(hdf) < 3:
        out["note"] = "too few filings for a correlation"
        return out

    r_p, p_p = pearsonr(hdf["half_even"], hdf["half_odd"])
    r_s, p_s = spearmanr(hdf["half_even"], hdf["half_odd"])
    out.update({
        "pearson_half_half": float(r_p), "pearson_p": float(p_p),
        "spearman_half_half": float(r_s), "spearman_p": float(p_s),
        "spearman_brown_from_pearson": float(spearman_brown(r_p)) if r_p > -1 else None,
        "spearman_brown_from_spearman": float(spearman_brown(r_s)) if r_s > -1 else None,
        "interpretation": (
            "Spearman-Brown estimates the reliability of the FULL-length filing "
            "mean from the correlation between two half-length means. It assumes "
            "the halves are equivalent and equally long; window counts here are "
            "odd/even splits of an often-odd total, so the halves can differ by "
            "one window."
        ),
    })

    try:
        paired = hdf[["half_even", "half_odd"]].to_numpy()
        ci = bootstrap_ci(
            paired[:, 0] - paired[:, 1],
            statistic=np.mean, n_boot=10000, seed=seed, method="percentile",
        )
        out["mean_half_difference"] = float(ci.observed)
        out["mean_half_difference_ci"] = [float(ci.low), float(ci.high)]
        out["halves_unbiased"] = bool(ci.low <= 0 <= ci.high)
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        out["mean_half_difference_error"] = str(exc)

    out["_per_filing"] = hdf
    return out


def test_retest(chunk_df: pd.DataFrame, section: str, measure: str,
                seed: int = DEFAULT_SEED) -> dict:
    """
    A firm's 10-K disagreement against its 10-Q disagreement, across firms.

    Filing-level means are computed with the pre-registered unweighted mean over
    that filing's surviving windows, and the usability gate applied.
    """
    d = _surviving(chunk_df, section, measure)
    key = ["ticker", "filing_type", "filing_date"]

    fil = []
    for k, g in d.groupby(key, sort=True):
        total = int(g["n_in_section"].iloc[0]) if "n_in_section" in g else len(g)
        if len(g) < MIN_USABLE_CHUNKS or len(g) < total / 2:
            continue
        fil.append({"ticker": k[0], "filing_type": k[1], "tier": g["tier"].iloc[0],
                    "value": float(g[measure].mean()), "n_windows": int(len(g))})
    fdf = pd.DataFrame(fil)

    wide = fdf.pivot_table(index=["ticker", "tier"], columns="filing_type",
                           values="value", aggfunc="mean").reset_index()
    out: dict = {"section": section, "measure": measure, "seed": seed}
    if not {"10-K", "10-Q"}.issubset(wide.columns):
        out["note"] = "need both 10-K and 10-Q filings to correlate"
        return out

    both = wide.dropna(subset=["10-K", "10-Q"])
    single = wide[wide[["10-K", "10-Q"]].isna().any(axis=1)]
    out.update({
        "n_firms_with_both": int(len(both)),
        "n_firms_with_one": int(len(single)),
        "firms_with_one": list(single["ticker"]),
    })
    if len(both) < 3:
        out["note"] = "too few firms with both forms"
        return out

    r_p, p_p = pearsonr(both["10-K"], both["10-Q"])
    r_s, p_s = spearmanr(both["10-K"], both["10-Q"])
    out.update({
        "pearson_10K_10Q": float(r_p), "pearson_p": float(p_p),
        "spearman_10K_10Q": float(r_s), "spearman_p": float(p_s),
        "mean_10K": float(both["10-K"].mean()),
        "mean_10Q": float(both["10-Q"].mean()),
        "caveat": (
            "A 10-K and a 10-Q are different documents of very different length, "
            "so this is an attenuated lower bound on firm-level stability, not a "
            "pure reliability coefficient. Split-half is the cleaner measurement-"
            "reliability estimate."
        ),
    })
    out["_per_firm"] = both
    return out


def write_report(config: Config, chunk_path: Path | None = None,
                 section: str = "mda", seed: int = DEFAULT_SEED) -> str:
    chunk_path = chunk_path or (config.OUTPUT_DIR / "chunk_level_results.csv")
    if not chunk_path.exists():
        raise FileNotFoundError(
            f"{chunk_path} not found — the scoring run has not produced "
            f"window-level output yet."
        )
    chunk_df = pd.read_csv(chunk_path)

    out_dir = config.OUTPUT_DIR / "experiments" / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {"seed": seed, "section": section,
                     "n_windows_in_file": int(len(chunk_df))}
    L: list[str] = []
    L.append("R1 — Reliability of the disagreement measure")
    L.append("=" * 78)
    L.append("")
    L.append(f"Window-level input: {chunk_path.name} ({len(chunk_df):,} rows)")
    L.append(f"Section analysed  : {section}   (the pre-registered primary section)")
    L.append(f"Seed              : {seed}")
    L.append("")

    for measure, label in (("embedding_disagreement", "EMBEDDING (primary)"),
                           ("numeric_disagreement", "NUMERIC (secondary)")):
        if measure not in chunk_df.columns:
            continue
        sh = split_half(chunk_df, section, measure, seed=seed)
        tr = test_retest(chunk_df, section, measure, seed=seed)

        per_filing = sh.pop("_per_filing", None)
        per_firm = tr.pop("_per_firm", None)
        if per_filing is not None:
            per_filing.to_csv(out_dir / f"split_half_{measure}.csv", index=False)
        if per_firm is not None:
            per_firm.to_csv(out_dir / f"test_retest_{measure}.csv", index=False)
        results[measure] = {"split_half": sh, "test_retest": tr}

        L.append(f"{label} — {measure}")
        L.append("-" * 78)
        if "spearman_half_half" in sh:
            L.append(f"  Split-half (odd vs even windows within a filing), "
                     f"n = {sh['n_filings_used']} filings")
            L.append(f"    Pearson  r = {sh['pearson_half_half']:+.3f} "
                     f"(p = {sh['pearson_p']:.4f})")
            L.append(f"    Spearman r = {sh['spearman_half_half']:+.3f} "
                     f"(p = {sh['spearman_p']:.4f})")
            sb = sh.get("spearman_brown_from_pearson")
            if sb is not None:
                L.append(f"    Spearman-Brown corrected reliability of the full "
                         f"filing mean = {sb:+.3f}")
            if "mean_half_difference_ci" in sh:
                lo, hi = sh["mean_half_difference_ci"]
                L.append(f"    Mean odd-even difference = "
                         f"{sh['mean_half_difference']:+.4f} "
                         f"[{lo:+.4f}, {hi:+.4f}] — "
                         f"{'halves unbiased' if sh.get('halves_unbiased') else 'HALVES DIFFER'}")
            if sh["n_filings_dropped"]:
                L.append(f"    {sh['n_filings_dropped']} filing(s) dropped "
                         f"(too few windows to split)")
        else:
            L.append(f"  Split-half: {sh.get('note', 'unavailable')}")
        L.append("")
        if "spearman_10K_10Q" in tr:
            L.append(f"  Test-retest (10-K vs 10-Q, same firm), "
                     f"n = {tr['n_firms_with_both']} firms")
            L.append(f"    Pearson  r = {tr['pearson_10K_10Q']:+.3f} "
                     f"(p = {tr['pearson_p']:.4f})")
            L.append(f"    Spearman r = {tr['spearman_10K_10Q']:+.3f} "
                     f"(p = {tr['spearman_p']:.4f})")
            L.append(f"    Mean 10-K = {tr['mean_10K']:.4f}, "
                     f"mean 10-Q = {tr['mean_10Q']:.4f}")
            if tr.get("firms_with_one"):
                L.append(f"    Firms contributing one form only: "
                         f"{', '.join(tr['firms_with_one'])}")
        else:
            L.append(f"  Test-retest: {tr.get('note', 'unavailable')}")
        L.append("")

    L.append("How to read these numbers")
    L.append("-" * 78)
    L.append("  Split-half is the measurement-reliability estimate: it asks whether two")
    L.append("  disjoint halves of the SAME filing agree. Test-retest asks a different")
    L.append("  and harder question — whether disagreement is a stable property of the")
    L.append("  FIRM across two different documents — so it is attenuated by genuine")
    L.append("  cross-document variation and is a lower bound, not a reliability")
    L.append("  coefficient. A high split-half with a low test-retest would mean the")
    L.append("  measure is reliable but document-specific, which is itself a finding and")
    L.append("  would need saying plainly rather than hiding.")
    L.append("")
    L.append("  Labelled EXPLORATORY in ANALYSIS_PLAN.md section 6: not part of the")
    L.append("  confirmatory family, and cannot convert a failed confirmatory test into a")
    L.append("  supported one.")
    L.append("")

    text = "\n".join(L)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=1, default=str), encoding="utf-8"
    )
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print(write_report(Config()))
