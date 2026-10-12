"""
F1 (case studies) and F2 (where disagreement lives).

After E4 showed that content-free boilerplate produces MORE persona disagreement
than real MD&A text, the obvious question is what the measure is actually
responding to. These two analyses answer it from data already scored, with no
model calls.

F1 — CASE STUDIES. The highest- and lowest-disagreement windows, with each
persona's own reasoning, so a reader can see what a high score looks like. If the
high-disagreement windows are thin, generic or boilerplate-like, that corroborates
the placebo result directly rather than asserting it.

F2 — WHERE DISAGREEMENT LIVES. Disagreement by position within a section, by
section, and by simple topic tags. Also the relationship with window
"informativeness" proxies: how many numbers the window contains, its type-token
ratio, and its length. The placebo result predicts disagreement should FALL as a
window carries more concrete content.

Run:  python3 -m experiments.qualitative.run
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.modules.setdefault("tensorflow", None)

from config import Config

logger = logging.getLogger(__name__)

TOPICS = {
    "liquidity": r"\b(liquidity|cash flow|working capital|credit facilit|borrow)",
    "guidance": r"\b(guidance|outlook|expect to|anticipate|forecast)",
    "segments": r"\b(segment|division|business unit|geograph)",
    "litigation": r"\b(litigation|lawsuit|legal proceeding|settlement|court)",
    "macro": r"\b(inflation|interest rate|recession|macroeconomic|tariff|supply chain)",
    "accounting": r"\b(impairment|goodwill|amortizat|depreciat|accrual|estimate)",
}
_NUM = re.compile(r"\b\d[\d,.]*\b")


def _window_text(config: Config, row) -> str | None:
    from analysis.chunking import chunk_section
    from data.edgar import _html_to_text, extract_sections_dict

    safe = str(row["filing_date"]).replace("-", "")
    path = config.FILINGS_DIR / f"{row['ticker']}_{row['filing_type']}_{safe}.txt"
    if not path.exists():
        return None
    sections, _ = extract_sections_dict(
        _html_to_text(path.read_text(encoding="utf-8", errors="ignore")),
        row["filing_type"])
    chunks = chunk_section(sections.get(row["section"]) or "",
                           config.CHUNK_WORDS, config.MIN_TAIL_WORDS)
    i = int(row["chunk_index"])
    return chunks[i] if i < len(chunks) else None


def enrich(config: Config, df: pd.DataFrame) -> pd.DataFrame:
    """Add position, informativeness proxies and topic tags to each window."""
    out = []
    for _, r in df.iterrows():
        text = _window_text(config, r)
        if text is None:
            continue
        words = text.split()
        n = max(1, len(words))
        nums = _NUM.findall(text)
        rec = dict(r)
        rec["position"] = (int(r.chunk_index) / max(1, int(r.n_in_section) - 1)
                           if int(r.n_in_section) > 1 else 0.0)
        rec["numeric_density"] = len(nums) / n
        rec["type_token_ratio"] = len(set(w.lower() for w in words)) / n
        rec["mean_word_len"] = float(np.mean([len(w) for w in words])) if words else 0.0
        low = text.lower()
        for t, pat in TOPICS.items():
            rec[f"topic_{t}"] = bool(re.search(pat, low))
        rec["_text"] = text
        out.append(rec)
    return pd.DataFrame(out)


def f1_cases(df: pd.DataFrame, chunk_df: pd.DataFrame, k: int = 3) -> list[dict]:
    """Top-k and bottom-k windows by disagreement, with persona reasoning."""
    d = df.sort_values("embedding_disagreement")
    picks = pd.concat([d.head(k).assign(_band="lowest"),
                       d.tail(k).assign(_band="highest")])
    cases = []
    for _, r in picks.iterrows():
        src = chunk_df[(chunk_df.ticker == r.ticker)
                       & (chunk_df.filing_type == r.filing_type)
                       & (chunk_df.section == r.section)
                       & (chunk_df.chunk_index == r.chunk_index)]
        reasoning = {}
        if len(src):
            s = src.iloc[0]
            for role in ("value", "sentiment", "fundamentals"):
                col = f"{role}_reasoning"
                if col in s and isinstance(s[col], str):
                    reasoning[role] = " ".join(s[col].split())[:260]
        cases.append({
            "band": r._band, "ticker": r.ticker, "tier": r.tier,
            "filing_type": r.filing_type, "section": r.section,
            "chunk_index": int(r.chunk_index),
            "disagreement": float(r.embedding_disagreement),
            "numeric_density": float(r.numeric_density),
            "excerpt": " ".join(r._text.split())[:380],
            "reasoning": reasoning,
        })
    return cases


def f2_location(df: pd.DataFrame) -> dict:
    res: dict = {"n_windows": int(len(df))}

    # Position within section.
    bins = pd.cut(df.position, [-0.01, 0.25, 0.5, 0.75, 1.01],
                  labels=["first quarter", "second", "third", "last quarter"])
    res["by_position"] = {
        str(k): {"n": int(v["embedding_disagreement"].size),
                 "mean": float(v["embedding_disagreement"].mean())}
        for k, v in df.groupby(bins, observed=True)
    }
    rho, p = spearmanr(df.position, df.embedding_disagreement)
    res["position_vs_disagreement"] = {"rho": float(rho), "p": float(p)}

    res["by_section"] = {
        str(k): {"n": int(len(v)), "mean": float(v.embedding_disagreement.mean())}
        for k, v in df.groupby("section")
    }

    # Informativeness proxies. The placebo result predicts NEGATIVE correlations:
    # the more concrete content a window carries, the less the lenses diverge.
    res["informativeness"] = {}
    for col in ("numeric_density", "type_token_ratio", "word_count", "mean_word_len"):
        if col in df:
            ok = df[col].notna() & df.embedding_disagreement.notna()
            r_, p_ = spearmanr(df.loc[ok, col], df.loc[ok, "embedding_disagreement"])
            res["informativeness"][col] = {"rho": float(r_), "p": float(p_),
                                           "n": int(ok.sum())}

    res["by_topic"] = {}
    for t in TOPICS:
        col = f"topic_{t}"
        if col not in df:
            continue
        a = df.loc[df[col], "embedding_disagreement"]
        b = df.loc[~df[col], "embedding_disagreement"]
        if len(a) >= 5 and len(b) >= 5:
            res["by_topic"][t] = {
                "n_with": int(len(a)), "mean_with": float(a.mean()),
                "n_without": int(len(b)), "mean_without": float(b.mean()),
                "difference": float(a.mean() - b.mean()),
            }
    return res


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    cfg = Config()
    out_dir = cfg.OUTPUT_DIR / "experiments" / "qualitative"
    out_dir.mkdir(parents=True, exist_ok=True)

    ch = pd.read_csv(cfg.OUTPUT_DIR / "chunk_level_results.csv")
    clean = ch[(~ch.any_failed.astype(str).str.lower().isin(["true", "1"]))
               & ch.embedding_disagreement.notna()]
    print(f"enriching {len(clean)} windows (re-extracting text offline)...")
    df = enrich(cfg, clean)
    df.drop(columns=["_text"]).to_csv(out_dir / "window_features.csv", index=False)

    cases = f1_cases(df, ch)
    loc = f2_location(df)
    (out_dir / "results.json").write_text(
        json.dumps({"F1_cases": cases, "F2_location": loc}, indent=1, default=str),
        encoding="utf-8")

    L = ["F1/F2 — What is the measure actually responding to?", "=" * 78, ""]
    L.append(f"windows: {loc['n_windows']}   (no model calls)")
    L.append("")
    L.append("F2  Informativeness proxies vs disagreement")
    L.append("-" * 78)
    L.append("  The placebo result (boilerplate scores HIGHER than real text) predicts")
    L.append("  that disagreement FALLS as a window carries more concrete content.")
    L.append("")
    for k, v in loc["informativeness"].items():
        flag = " *" if v["p"] < 0.05 else ""
        L.append(f"    {k:20s} rho = {v['rho']:+.3f}  p = {v['p']:.4f}  "
                 f"n = {v['n']}{flag}")
    L.append("")
    L.append("F2  Position within section")
    L.append("-" * 78)
    for k, v in loc["by_position"].items():
        L.append(f"    {k:16s} n = {v['n']:3d}   mean = {v['mean']:.4f}")
    pv = loc["position_vs_disagreement"]
    L.append(f"    Spearman(position, disagreement) rho = {pv['rho']:+.3f}, "
             f"p = {pv['p']:.4f}")
    L.append("")
    L.append("F2  By section")
    L.append("-" * 78)
    for k, v in loc["by_section"].items():
        L.append(f"    {k:8s} n = {v['n']:3d}   mean = {v['mean']:.4f}")
    L.append("")
    if loc["by_topic"]:
        L.append("F2  By topic (window mentions the topic vs not)")
        L.append("-" * 78)
        for t, v in sorted(loc["by_topic"].items(),
                           key=lambda x: -abs(x[1]["difference"])):
            L.append(f"    {t:12s} with {v['mean_with']:.4f} (n={v['n_with']:3d})  "
                     f"without {v['mean_without']:.4f} (n={v['n_without']:3d})  "
                     f"diff {v['difference']:+.4f}")
        L.append("")
    L.append("F1  Case studies")
    L.append("=" * 78)
    for c in cases:
        L.append("")
        L.append(f"[{c['band'].upper()}] {c['ticker']} ({c['tier']}) "
                 f"{c['filing_type']} {c['section']}#{c['chunk_index']}  "
                 f"disagreement = {c['disagreement']:.4f}  "
                 f"numeric density = {c['numeric_density']:.3f}")
        L.append(f"  excerpt: \"{c['excerpt'][:300]}...\"")
        for role, txt in c["reasoning"].items():
            L.append(f"    {role:13s} {txt[:200]}")
    text = "\n".join(L)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    print(text[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
