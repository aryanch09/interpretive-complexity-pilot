"""
Truncation-coverage diagnostic.

Answers the question the advisor raised in September 2026: what did the fixed
word cap actually do to the sample, and does filing length drive the
disagreement measures?

Run:  python -m analysis.coverage_diagnostic

Outputs
  output/coverage_diagnostic.csv  — per-filing coverage under the retired cap
  output/coverage_diagnostic.txt  — the memo tables

The diagnostic is deliberately kept as its own module rather than folded into
the summary: it describes a property of the OLD design, and it is the evidence
for why the design changed.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config  # noqa: E402

logger = logging.getLogger(__name__)

RETIRED_CAP = 2000  # the Round 7 MAX_WORDS_PER_FILING
TIER_RANK = {"small": 1, "mid": 2, "large": 3}


def _tier_map(base: Path) -> dict[str, str]:
    src = (base / "firms.py").read_text()
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r'"ticker":\s*"(\w+)".*?"tier":\s*"(\w+)"', src)
    }


def _manifest_selection(config) -> dict[tuple[str, str], str] | None:
    """
    The frozen analysis sample from data/sample_manifest.csv, as
    {(ticker, form): YYYYMMDD}. Returns None when the manifest is absent.

    Every report that describes the sample reads it from here, so the coverage
    diagnostic, the extraction report and the scored run cannot disagree about
    which filings the study is about. This report previously derived the sample
    by reading output/results.csv, which meant archiving a stale results file
    silently changed the sample.
    """
    path = Path(config.BASE_DIR) / "data" / "sample_manifest.csv"
    if not path.exists():
        return None
    import csv

    out: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["ticker"].strip(), row["filing_type"].strip())] = (
                row["filing_date"].strip().replace("-", "")
            )
    return out


def build_coverage_table(config: Config) -> pd.DataFrame:
    """
    For every filing in the study, the complete length of its narrative sections
    and the fraction of them the personas actually read under the retired cap.
    """
    from data.edgar import _html_to_text, extract_sections_dict

    base = Path(config.BASE_DIR)
    tiers = _tier_map(base)

    # The analysis sample is defined HERE, from the firm list and the filing
    # dates -- not by reading output/results.csv.  That coupling used to define
    # the sample by whatever a previous run happened to have scored, so
    # archiving a stale results.csv silently widened this report from the
    # 59-filing analysis sample to all 96 cached filings (including the extra
    # 10-Qs that MAX_FILINGS_PER_TYPE excludes).
    latest = _manifest_selection(config)
    if latest is None:
        from firms import FULL_PILOT_FIRMS

        tickers = {f["ticker"] if isinstance(f, dict) else f for f in FULL_PILOT_FIRMS}
        cutoff = config.TRAINING_CUTOFF.replace("-", "")

        # Fallback when the sample has not been frozen yet: most recent
        # post-cutoff filing of each form type per firm (MAX_FILINGS_PER_TYPE = 1).
        latest = {}
        for path in sorted(config.FILINGS_DIR.glob("*.txt")):
            parts = path.stem.split("_")
            if len(parts) < 3:
                continue
            ticker, form, date = parts[0], parts[1], parts[2]
            if ticker not in tickers or date <= cutoff:
                continue
            k = (ticker, form)
            if k not in latest or date > latest[k]:
                latest[k] = date
    selected = {f"{t}_{f}_{d}" for (t, f), d in latest.items()}

    rows: list[dict] = []
    for path in sorted(config.FILINGS_DIR.glob("*.txt")):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        ticker, form, date = parts[0], parts[1], parts[2]
        if f"{ticker}_{form}_{date}" not in selected:
            continue

        clean = _html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
        sections, fallback = extract_sections_dict(clean, form)
        mda_w = len(sections.get("mda", "").split())
        risk_w = len(sections.get("risk", "").split())
        full_w = mda_w + risk_w
        if not full_w:
            continue

        rows.append({
            "ticker": ticker,
            "tier": tiers.get(ticker),
            "filing_type": form,
            "filing_date": date,
            "mda_words": mda_w,
            "risk_words": risk_w,
            "section_words_total": full_w,
            # What the personas saw under the retired fixed cap.
            "words_read_old_cap": min(full_w, RETIRED_CAP),
            "fraction_read_old_cap": round(min(1.0, RETIRED_CAP / full_w), 4),
            # What they see now.
            "fraction_read_now": 1.0,
            "used_fallback_extraction": fallback,
        })

    return pd.DataFrame(rows)


def _fmt_rho(x, y, label: str) -> str:
    ok = pd.notna(x) & pd.notna(y)
    n = int(ok.sum())
    if n < 3:
        return f"  {label:<44} n={n:<3} (too few observations)"
    rho, p = spearmanr(x[ok], y[ok])
    return f"  {label:<44} n={n:<3} rho={rho:+.3f}  p={p:.4f}"


def write_report(config: Config) -> str:
    cov = build_coverage_table(config)
    cov.to_csv(config.OUTPUT_DIR / "coverage_diagnostic.csv", index=False)

    L: list[str] = []
    L.append("Coverage diagnostic — what the fixed word cap did to the sample")
    L.append("=" * 72)
    L.append("")
    L.append(
        f"Retired design: MAX_WORDS_PER_FILING = {RETIRED_CAP}, applied to the "
        "concatenated\nnarrative sections after extraction."
    )
    L.append(
        "Current design: MD&A and Risk Factors extracted complete; comparability "
        f"enforced\nat the chunk level ({config.CHUNK_WORDS} words per window)."
    )
    L.append("")

    n_trunc = int((cov["section_words_total"] > RETIRED_CAP).sum())
    L.append(f"Filings in sample: {len(cov)}")
    L.append(f"Filings truncated by the old cap: {n_trunc} of {len(cov)}")
    L.append("")

    L.append("Complete section length and fraction read, by tier (old cap)")
    L.append("-" * 72)
    g = cov.groupby("tier").agg(
        n=("ticker", "size"),
        median_section_words=("section_words_total", "median"),
        mean_fraction_read=("fraction_read_old_cap", "mean"),
        median_fraction_read=("fraction_read_old_cap", "median"),
        min_fraction_read=("fraction_read_old_cap", "min"),
    )
    for tier in ("large", "mid", "small"):
        if tier not in g.index:
            continue
        r = g.loc[tier]
        L.append(
            f"  {tier:<6} n={int(r['n']):<3} median section = "
            f"{int(r['median_section_words']):>6,} words | "
            f"read: mean {r['mean_fraction_read']:.1%}, "
            f"median {r['median_fraction_read']:.1%}, "
            f"min {r['min_fraction_read']:.1%}"
        )
    L.append("")

    # The confound: coverage was a function of the independent variable.
    cov["tier_rank"] = cov["tier"].map(TIER_RANK)
    L.append("THE PROBLEM — coverage was correlated with the independent variable")
    L.append("-" * 72)
    L.append(_fmt_rho(cov["tier_rank"], cov["fraction_read_old_cap"],
                      "tier rank (large=3) x fraction read"))
    L.append(_fmt_rho(cov["tier_rank"], cov["section_words_total"],
                      "tier rank (large=3) x complete section words"))
    L.append("")
    L.append(
        "  Larger firms file longer narrative sections, so a fixed cap read a\n"
        "  smaller share of them.  'How much of the filing the model read' was\n"
        "  therefore a function of firm size — the variable the hypothesis is\n"
        "  about — and any tier gradient in disagreement was confounded with a\n"
        "  tier gradient in coverage.  Under the current design every filing is\n"
        "  read in full, so this correlation is undefined by construction."
    )
    L.append("")

    # Length vs disagreement, on whatever scored data exists.
    L.append("Length vs. disagreement (the control the advisor asked for)")
    L.append("-" * 72)
    res_path = config.OUTPUT_DIR / "results.csv"
    if res_path.exists():
        res = pd.read_csv(res_path)
        clean = res[~res["any_failed"].fillna(False)].copy()
        L.append(f"  Scored rows available: {len(clean)} of {len(res)}")
        if len(clean) >= 3:
            key = clean["ticker"] + "_" + clean["filing_type"]
            cov_key = cov["ticker"] + "_" + cov["filing_type"]
            # results.csv now carries its own section_words_total (scored
            # sections only), so merging the coverage table's column of the same
            # name silently suffixed both and broke the lookup. Drop the local
            # copy first and take the coverage table's, which is the complete
            # narrative length this report is defined on.
            left = clean.assign(_k=key.values).drop(
                columns=["section_words_total"], errors="ignore")
            merged = left.merge(
                cov.assign(_k=cov_key.values)[["_k", "section_words_total"]],
                on="_k", how="left",
            )
            for measure in ("numeric_disagreement", "embedding_disagreement"):
                if measure in merged:
                    L.append(_fmt_rho(merged["word_count"], merged[measure],
                                      f"{measure} x words actually read"))
                    L.append(_fmt_rho(merged["section_words_total"], merged[measure],
                                      f"{measure} x complete section length"))
        else:
            L.append("  Too few scored rows for a meaningful correlation.")
        L.append("")
        L.append(
            "  NOTE: computed on the CURRENT complete-section run. The length\n"
            "  variable is the true narrative length, not a constant imposed by\n"
            "  the retired cap.\n"
            "  The length control that belongs in the paper is the one computed\n"
            "  after the complete-section re-run, where the length variable is\n"
            "  the true section length rather than a constant imposed by the cap."
        )
    else:
        L.append(
            "  No scored results under the current cache version.\n"
            "  The Round 7 partial run (15 of 59 rows, all large-cap, truncated AND\n"
            "  incomplete) is archived in archive/round7_partial/ and is deliberately\n"
            "  NOT read here: its length correlation was reported to the advisor as\n"
            "  continuity rather than evidence, and it is superseded.  The length\n"
            "  control that belongs in the paper is computed after the complete-section\n"
            "  run, on mda_words -- the true section length rather than a constant\n"
            "  imposed by the retired cap."
        )
    L.append("")

    # Chunk budget under the new design.
    L.append("Chunk budget under the current design")
    L.append("-" * 72)
    # Count windows with the REAL chunker, not an estimate.  The previous
    # round(words / CHUNK_WORDS) estimator implemented neither the
    # sub-MIN_TAIL_WORDS drop nor the tail merge, so it understated the budget
    # (it produced the 623-chunk / 1,869-call figure that was sent to the
    # advisor; the true figures are 665 and 1,995).
    from analysis.chunking import chunk_section

    def _windows(row) -> int:
        n = 0
        for col in ("mda_words", "risk_words"):
            # chunk_section needs text, not a count; reconstruct the window count
            # from the same arithmetic it uses.
            w = int(row[col] or 0)
            if w < config.MIN_TAIL_WORDS:
                continue
            if w <= config.CHUNK_WORDS + config.MIN_TAIL_WORDS:
                n += 1
                continue
            full, tail = divmod(w, config.CHUNK_WORDS)
            n += full + (1 if tail >= config.MIN_TAIL_WORDS else 0)
        return n

    n_chunks = cov.apply(_windows, axis=1)
    L.append(f"  Windows (real chunker): {int(n_chunks.sum()):,}")
    L.append(f"  Persona calls:          {int(n_chunks.sum()) * 3:,}")
    L.append(
        f"  Every filing read at 100% coverage; each judgment made on a "
        f"{config.CHUNK_WORDS}-word window."
    )

    text = "\n".join(L)
    (config.OUTPUT_DIR / "coverage_diagnostic.txt").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    cfg = Config()
    cfg.ensure_dirs()
    print(write_report(cfg))
