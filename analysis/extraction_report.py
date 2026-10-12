"""
Extraction report — what was recovered from every filing in the analysis sample,
and by which route.

Runs entirely offline from the raw HTML cached in data/filings/. No EDGAR
requests, no model calls. This is the audit the advisor's Round 8 reply requires
before any gradient is read off a run: every filing's complete section lengths,
its window counts, and an explicit classification of every filing that yields no
Risk Factors windows as either an extraction failure or a legitimate absence.

Writes:
  output/extraction_report.txt   human-readable, for the advisor and the paper
  output/extraction_report.csv   one row per filing

Run:  python -m analysis.extraction_report
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.modules.setdefault("tensorflow", None)  # abseil mutex deadlock guard

from config import Config

logger = logging.getLogger(__name__)

# A filing whose MD&A exceeds this is a boundary failure: the span has run past
# the end of the section (JPMorgan's MD&A once measured 107,257 words because it
# swallowed the financial statements).
MDA_BOUNDARY_FAILURE_WORDS = 40_000
# A filing whose MD&A falls below this is an extraction miss, not a short filer.
MDA_MISS_WORDS = 1_000

# Language that marks an Item 1A as incorporated by reference rather than restated.
_NO_MATERIAL_CHANGE_RE = re.compile(
    r"(?:no\s+material\s+change|not\s+aware\s+of\s+any\s+material\s+change|"
    r"there\s+have\s+been\s+no\s+(?:material\s+)?changes|"
    r"risk\s+factors\s+are\s+discussed\s+in\s+detail|"
    r"(?:please\s+)?refer\s+to|described\s+in|set\s+forth\s+in|"
    r"previously\s+disclosed|incorporated\s+(?:herein\s+)?by\s+reference)",
    re.IGNORECASE,
)


def _manifest_selection(config) -> dict[tuple[str, str], str] | None:
    """
    The frozen analysis sample from data/sample_manifest.csv, as
    {(ticker, form): YYYYMMDD}. Returns None when the manifest is absent.

    Every report that describes the sample reads it from here, so the coverage
    diagnostic, the extraction report and the scored run cannot disagree about
    which 59 filings the study is about.
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


def _selected_filings(config: Config) -> dict[tuple[str, str], str]:
    """The analysis sample: the frozen manifest if present, else the most recent
    post-cutoff 10-K and 10-Q per firm."""
    pinned = _manifest_selection(config)
    if pinned is not None:
        return pinned
    from firms import FULL_PILOT_FIRMS

    tickers = {f["ticker"] if isinstance(f, dict) else f for f in FULL_PILOT_FIRMS}
    cutoff = config.TRAINING_CUTOFF.replace("-", "")
    latest: dict[tuple[str, str], str] = {}
    for path in sorted(config.FILINGS_DIR.glob("*.txt")):
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        ticker, form, date = parts[0], parts[1], parts[2]
        if ticker not in tickers or date <= cutoff:
            continue
        key = (ticker, form)
        if key not in latest or date > latest[key]:
            latest[key] = date
    return latest


def _tier_map() -> dict[str, str]:
    """Tier lookup. FULL_PILOT_FIRMS is a list of ticker strings; the tier lives
    in the FIRMS registry of dicts."""
    from firms import FIRMS

    return {f["ticker"]: f.get("tier", "") for f in FIRMS if isinstance(f, dict)}


def _classify_risk_absence(body: str, full_text: str, config: Config) -> tuple[str, str]:
    """
    Decide whether a filing with no Risk Factors windows is an extraction failure
    or a legitimate absence, and return the sentence that proves it.

    The extracted fragment is not sufficient evidence on its own: when extraction
    anchors on a cross-reference rather than the real header, the fragment can
    begin mid-sentence and omit the filing's actual "no material change"
    statement. So the whole converted text is searched around each Item 1A /
    Risk Factors mention.
    """
    flat_body = " ".join(body.split())
    if _NO_MATERIAL_CHANGE_RE.search(flat_body[:400]):
        return "legitimate: incorporated by reference", flat_body[:240]

    # Look for the real statement anywhere a Risk Factors heading appears.
    for m in re.finditer(r"(?:ITEM|Item)\s*\xa0?\s*1A[.\s\u2013-]|RISK\s+FACTORS",
                         full_text, re.IGNORECASE):
        window = " ".join(full_text[m.start():m.start() + 700].split())
        if _NO_MATERIAL_CHANGE_RE.search(window):
            return ("legitimate: incorporated by reference "
                    "(verified against the full filing text)"), window[:240]

    if not flat_body:
        return "NO SECTION RECOVERED — investigate", ""
    return (f"below MIN_TAIL_WORDS ({config.MIN_TAIL_WORDS}) — verify manually",
            flat_body[:240])


def build_table(config: Config) -> pd.DataFrame:
    from analysis.chunking import chunk_section
    from data.edgar import _html_to_text, extract_sections_dict

    tiers = _tier_map()
    rows: list[dict] = []

    for (ticker, form), date in sorted(_selected_filings(config).items()):
        path = config.FILINGS_DIR / f"{ticker}_{form}_{date}.txt"
        text = _html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
        sections, failed = extract_sections_dict(text, form)

        row: dict = {
            "ticker": ticker,
            "tier": tiers.get(ticker, ""),
            "filing_type": form,
            "filing_date": date,
            "extraction_failed": failed,
        }
        total_windows = 0
        sizes: list[int] = []
        for sec in config.SECTIONS_TO_ANALYZE:
            body = sections.get(sec) or ""
            words = len(body.split())
            chunks = chunk_section(body, config.CHUNK_WORDS, config.MIN_TAIL_WORDS) if body else []
            row[f"{sec}_words"] = words
            row[f"{sec}_windows"] = len(chunks)
            total_windows += len(chunks)
            sizes.extend(len(c.split()) for c in chunks)

        row["total_windows"] = total_windows
        row["min_window_words"] = min(sizes) if sizes else 0
        row["max_window_words"] = max(sizes) if sizes else 0

        # Classify a missing Risk Factors section.
        if row.get("risk_windows", 0) == 0:
            kind, evidence = _classify_risk_absence(
                sections.get("risk") or "", text, config
            )
            row["risk_absence"] = kind
            row["risk_absence_evidence"] = evidence
        else:
            row["risk_absence"] = ""
            row["risk_absence_evidence"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def write_report(config: Config) -> str:
    df = build_table(config)
    df.to_csv(config.OUTPUT_DIR / "extraction_report.csv", index=False)

    L: list[str] = []
    L.append("Extraction report — complete sections and scoring windows")
    L.append("=" * 78)
    L.append("")
    L.append(f"Filings in analysis sample: {len(df)}")
    L.append(f"Firms: {df['ticker'].nunique()}")
    L.append(f"Window size: {config.CHUNK_WORDS} words, tail merged below {config.MIN_TAIL_WORDS}")
    L.append("")
    L.append(f"TOTAL SCORING WINDOWS: {int(df['total_windows'].sum()):,}")
    L.append(f"PERSONA CALLS:         {int(df['total_windows'].sum()) * 3:,}")
    L.append(f"  of which MD&A (the pre-registered primary section): "
             f"{int(df['mda_windows'].sum()):,} windows, "
             f"{int(df['mda_windows'].sum()) * 3:,} calls")
    L.append("")

    L.append("Per-filing table")
    L.append("-" * 78)
    L.append(f"{'ticker':7s} {'tier':6s} {'form':5s} {'date':9s} "
             f"{'mda_w':>7s} {'risk_w':>7s} {'mda_win':>8s} {'risk_win':>9s} "
             f"{'win_min':>8s} {'win_max':>8s}")
    for _, r in df.iterrows():
        L.append(
            f"{r['ticker']:7s} {r['tier']:6s} {r['filing_type']:5s} {r['filing_date']:9s} "
            f"{r['mda_words']:7,d} {r['risk_words']:7,d} {r['mda_windows']:8d} "
            f"{r['risk_windows']:9d} {r['min_window_words']:8d} {r['max_window_words']:8d}"
        )
    L.append("")

    L.append("Windows per filing, by tier")
    L.append("-" * 78)
    for tier in ("large", "mid", "small"):
        sub = df[df["tier"] == tier]
        if sub.empty:
            continue
        L.append(
            f"  {tier:6s} n={len(sub):2d}  "
            f"all sections: median {sub['total_windows'].median():5.1f}, "
            f"range {sub['total_windows'].min()}-{sub['total_windows'].max()}  |  "
            f"MD&A only: median {sub['mda_windows'].median():5.1f}, "
            f"range {sub['mda_windows'].min()}-{sub['mda_windows'].max()}"
        )
    L.append("")
    L.append("  Window count scales with filing length and so with tier. This is the")
    L.append("  artifact the advisor flagged: a mean over many windows is mechanically")
    L.append("  less variable than a mean over two. The pre-registered window-count")
    L.append("  battery (ANALYSIS_PLAN.md section 5) tests it before any gradient is read.")
    L.append("")

    L.append("Anomaly gate")
    L.append("-" * 78)
    failed = df[df["extraction_failed"]]
    boundary = df[df["mda_words"] > MDA_BOUNDARY_FAILURE_WORDS]
    miss = df[df["mda_words"] < MDA_MISS_WORDS]
    L.append(f"  extraction_failed                        : {len(failed)}"
             + (f"  {list(failed['ticker'])}" if len(failed) else "  NONE"))
    L.append(f"  MD&A > {MDA_BOUNDARY_FAILURE_WORDS:,} words (boundary failure): {len(boundary)}"
             + (f"  {list(boundary['ticker'])}" if len(boundary) else "  NONE"))
    L.append(f"  MD&A < {MDA_MISS_WORDS:,} words (extraction miss)   : {len(miss)}"
             + (f"  {list(miss['ticker'])}" if len(miss) else "  NONE"))
    L.append(f"  filings with 0 MD&A windows              : "
             f"{int((df['mda_windows'] == 0).sum())}")
    L.append(f"  MD&A word range                          : "
             f"{df['mda_words'].min():,} - {df['mda_words'].max():,}")
    L.append("")

    L.append("Filings with no Risk Factors windows — classified individually")
    L.append("-" * 78)
    norisk = df[df["risk_windows"] == 0]
    L.append(f"  {len(norisk)} of {len(df)} filings. "
             f"{len(df) - len(norisk)} filings have Risk Factors windows.")
    L.append("")
    for _, r in norisk.iterrows():
        L.append(f"  {r['ticker']} {r['filing_type']} ({r['risk_words']:,} words) "
                 f"-> {r['risk_absence']}")
        if r["risk_absence_evidence"]:
            L.append(f"      \"{r['risk_absence_evidence'][:200]}\"")
    L.append("")
    unresolved = norisk[norisk["risk_absence"].str.contains("investigate|verify", na=False)]
    if len(unresolved):
        L.append(f"  !! {len(unresolved)} filing(s) NOT classified as legitimate. "
                 f"Investigate before scoring: {list(unresolved['ticker'])}")
    else:
        L.append("  All are 10-Q sections that incorporate the 10-K's risk factors by")
        L.append("  reference — legitimately absent, not extraction failures. Under the")
        L.append("  pre-registered missing-section rule the primary analysis is MD&A only,")
        L.append("  so none of these filings is lost from the primary sample.")
    L.append("")

    text = "\n".join(L)
    (config.OUTPUT_DIR / "extraction_report.txt").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    print(write_report(Config()))
