"""
How many windows per filing would this measure need to be usable?

"The measure is unreliable" is a dead end for a reader. "The measure needs N
windows per filing" is actionable. Spearman-Brown inverts to give N directly.

Free: reads the scored window-level output, makes no model calls.

Run:  python3 -m experiments.reliability.extrapolate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("tensorflow", None)
from config import Config


def main() -> int:
    cfg = Config()
    out_dir = cfg.OUTPUT_DIR / "experiments" / "reliability"
    rel = json.loads((out_dir / "results.json").read_text())
    emb = rel["embedding_disagreement"]["split_half"]

    # Split-half r is the reliability of a HALF-LENGTH filing mean. Spearman-Brown
    # steps it to the full mean (x2) and to any other multiple of the half.
    r_half = float(emb["pearson_half_half"])
    sb = lambda r, k: (k * r) / (1 + (k - 1) * r)  # noqa: E731

    ch = pd.read_csv(cfg.OUTPUT_DIR / "chunk_level_results.csv")
    mda = ch[(ch.section == "mda")
             & (~ch.any_failed.astype(str).str.lower().isin(["true", "1"]))]
    per_filing = mda.groupby(["ticker", "filing_type", "filing_date"]).size()
    median_windows = float(per_filing.median())
    half_windows = median_windows / 2.0

    rows = []
    for target in (0.60, 0.70, 0.80, 0.90):
        # Solve k in Spearman-Brown for the target, where k multiplies the HALF.
        k = (target * (1 - r_half)) / (r_half * (1 - target))
        rows.append({
            "target_reliability": target,
            "k_multiples_of_half": k,
            "windows_per_filing_needed": k * half_windows,
            "factor_vs_current": (k * half_windows) / median_windows,
        })
    df = pd.DataFrame(rows)

    L = ["Reliability extrapolation — what would this measure need?", "=" * 74, ""]
    L.append(f"observed split-half r (half-length filing mean) : {r_half:+.3f}")
    L.append(f"Spearman-Brown reliability of the FULL mean     : "
             f"{sb(r_half, 2):+.3f}")
    L.append(f"median MD&A windows per filing                 : {median_windows:.0f}")
    L.append("")
    L.append("Windows per filing required to reach a target reliability:")
    L.append(f"  {'target':>8s} {'windows needed':>16s} {'vs current':>12s}")
    for _, r in df.iterrows():
        L.append(f"  {r.target_reliability:8.2f} {r.windows_per_filing_needed:16.0f} "
                 f"{r.factor_vs_current:11.1f}x")
    L.append("")
    words = median_windows * cfg.CHUNK_WORDS
    need70 = float(df[df.target_reliability == 0.70].windows_per_filing_needed.iloc[0])
    L.append(f"At {cfg.CHUNK_WORDS} words per window, the median filing currently supplies")
    L.append(f"~{words:,.0f} words of MD&A. Reaching the conventional 0.70 threshold would")
    L.append(f"need ~{need70:.0f} windows, i.e. ~{need70*cfg.CHUNK_WORDS:,.0f} words — far more MD&A than")
    L.append("a typical filing contains.")
    L.append("")
    L.append("READING: the shortfall is not a sampling accident that a bigger firm")
    L.append("sample would fix. Filing-level aggregation of this measure is limited by")
    L.append("how much narrative a filing HAS. Raising reliability requires a different")
    L.append("estimator — pooling across filings or years, modelling at the window level")
    L.append("rather than averaging to the filing, or a lens set that disagrees more")
    L.append("systematically — not more firms.")
    text = "\n".join(L)
    (out_dir / "extrapolation.txt").write_text(text, encoding="utf-8")
    df.to_csv(out_dir / "extrapolation.csv", index=False)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
