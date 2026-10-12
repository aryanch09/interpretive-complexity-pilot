"""
H1 step 1 — build a blinded rating sheet.

Pairs a HIGH-disagreement window with a LOW-disagreement one, matched on tier and
on length, so neither firm size nor window length can drive the rater's choice.
Presentation order within each pair is randomised and the key is written to a
separate file the rater must not open.

ON EXCERPTS. The measure is computed on full 1,200-word windows, but 30 such
pairs is ~72,000 words, roughly five hours of careful reading -- the
"one hour" in the original plan was wrong. The default here shows the first 400
words of each window, which keeps the task near an hour. That is a real
limitation and the scorer prints it: the rater judges an excerpt while the
measure saw the whole window. Use --excerpt-words 0 for full windows if you have
the time; the comparison is cleaner that way.

Run:  python3 -m experiments.human_check.make_pairs --pairs 20 --excerpt-words 400
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("tensorflow", None)
from config import Config


def _window_text(config: Config, row) -> str | None:
    from analysis.chunking import chunk_section
    from data.edgar import _html_to_text, extract_sections_dict

    safe = str(row["filing_date"]).replace("-", "")
    p = config.FILINGS_DIR / f"{row['ticker']}_{row['filing_type']}_{safe}.txt"
    if not p.exists():
        return None
    sections, _ = extract_sections_dict(
        _html_to_text(p.read_text(encoding="utf-8", errors="ignore")),
        row["filing_type"])
    ch = chunk_section(sections.get(row["section"]) or "",
                       config.CHUNK_WORDS, config.MIN_TAIL_WORDS)
    i = int(row["chunk_index"])
    return ch[i] if i < len(ch) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--excerpt-words", type=int, default=400,
                    help="0 = full window")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--max-length-diff", type=int, default=120,
                    help="max word-count difference within a pair")
    a = ap.parse_args(argv)

    cfg = Config()
    out = cfg.OUTPUT_DIR / "experiments" / "human_check"
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(cfg.OUTPUT_DIR / "chunk_level_results.csv")
    d = d[(d.section == "mda")
          & (~d.any_failed.astype(str).str.lower().isin(["true", "1"]))
          & d.embedding_disagreement.notna()]
    lo_cut, hi_cut = d.embedding_disagreement.quantile([0.25, 0.75])
    lo = d[d.embedding_disagreement <= lo_cut]
    hi = d[d.embedding_disagreement >= hi_cut]

    rng = np.random.default_rng(a.seed)
    pairs, used_lo = [], set()
    for _, h in hi.sample(frac=1.0, random_state=a.seed).iterrows():
        if len(pairs) >= a.pairs:
            break
        cand = lo[(lo.tier == h.tier)
                  & (~lo.index.isin(used_lo))
                  & ((lo.word_count - h.word_count).abs() <= a.max_length_diff)]
        if cand.empty:
            continue
        l = cand.iloc[int(rng.integers(len(cand)))]
        used_lo.add(l.name)
        pairs.append((h, l))

    if len(pairs) < a.pairs:
        print(f"note: only {len(pairs)} pairs met the matching constraints "
              f"(wanted {a.pairs})")

    sheet, key = [], []
    sheet.append("# Which excerpt is HARDER TO INTERPRET?\n")
    sheet.append(
        "For each pair, read both and decide which is harder to form a clear view "
        "from — harder to tell what it implies about the company's prospects. "
        "This is NOT about vocabulary difficulty or sentence length. Write one "
        "letter (A or B) per line in `h1_answers.txt`, in order.\n")
    sheet.append("Do not open `h1_key.csv` until you are finished.\n")
    sheet.append("---\n")

    for i, (h, l) in enumerate(pairs, 1):
        th, tl = _window_text(cfg, h), _window_text(cfg, l)
        if th is None or tl is None:
            continue
        if a.excerpt_words:
            th = " ".join(th.split()[:a.excerpt_words])
            tl = " ".join(tl.split()[:a.excerpt_words])
        # randomise which side the high-disagreement window appears on
        high_is_a = bool(rng.integers(2))
        A, B = (th, tl) if high_is_a else (tl, th)
        sheet.append(f"\n## Pair {i}\n")
        sheet.append(f"**A.** {A}\n")
        sheet.append(f"**B.** {B}\n")
        key.append({
            "pair": i,
            "high_disagreement_side": "A" if high_is_a else "B",
            "high_ticker": h.ticker, "low_ticker": l.ticker, "tier": h.tier,
            "high_score": float(h.embedding_disagreement),
            "low_score": float(l.embedding_disagreement),
            "high_words": int(h.word_count), "low_words": int(l.word_count),
        })

    (out / "h1_rating_sheet.md").write_text("\n".join(sheet), encoding="utf-8")
    pd.DataFrame(key).to_csv(out / "h1_key.csv", index=False)
    if not (out / "h1_answers.txt").exists():
        (out / "h1_answers.txt").write_text(
            "# One letter per line, A or B, one per pair, in order.\n"
            "# Lines starting with # are ignored.\n", encoding="utf-8")

    words = sum(len(str(s).split()) for s in sheet)
    print(f"wrote {len(key)} pairs to {out/'h1_rating_sheet.md'}")
    print(f"  tiers: {pd.DataFrame(key).tier.value_counts().to_dict()}")
    print(f"  mean separation: high {pd.DataFrame(key).high_score.mean():.4f} "
          f"vs low {pd.DataFrame(key).low_score.mean():.4f}")
    print(f"  ~{words:,} words to read, roughly "
          f"{words/220/60:.1f} hours at careful pace")
    print(f"\nNext: fill in {out/'h1_answers.txt'}, then run")
    print(f"  python3 -m experiments.human_check.score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
