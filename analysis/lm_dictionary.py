"""
Loughran-McDonald Master Dictionary — the real one.

WHY THIS MODULE EXISTS. `analysis/textual_baselines.py` carries a hand-curated
290-word list named `LM_NEGATIVE_WORDS`, and the pipeline reports the resulting
share as `lm_neg_pct`. Checked against the actual Master Dictionary, that list is
neither complete nor correct:

  * it covers 290 of the 2,345 words in L-M's Negative category (12.4%), and
  * 32 of its 290 words are NOT in L-M Negative at all — 23 appear in the
    dictionary with no sentiment category, 5 are Uncertainty, 3 Constraining and
    1 Litigious. Examples: ABSENT, ARBITRARY, CLAIM, CONSTRAINT, ELIMINATE.

So `lm_neg_pct` is not the Loughran-McDonald negative measure, despite the name.
Anything reported under that label would be wrong, and a referee who knows the
dictionary would catch it immediately. The paper uses this module instead, and
the discrepancy is disclosed.

THE DICTIONARY IS NOT VENDORED. It is free for academic research but
redistribution is not explicitly granted, so it is gitignored and fetched by
`scripts/fetch_lm_dictionary.sh`. Cite Loughran and McDonald (2011) when using it.

Run:  python3 -m analysis.lm_dictionary   (prints a summary of what loaded)
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DICT_FILENAME = "Loughran-McDonald_MasterDictionary_1993-2025.csv"
CATEGORIES = (
    "Negative", "Positive", "Uncertainty", "Litigious",
    "Strong_Modal", "Weak_Modal", "Constraining",
)
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


class DictionaryUnavailable(RuntimeError):
    """Raised when the Master Dictionary is not on disk."""


def dictionary_path(base_dir: str | Path) -> Path:
    return Path(base_dir) / "data" / "dictionaries" / DICT_FILENAME


@lru_cache(maxsize=4)
def load_categories(base_dir: str) -> dict[str, frozenset[str]]:
    """{category: frozenset of uppercase words}. Cached across calls."""
    path = dictionary_path(base_dir)
    if not path.exists():
        raise DictionaryUnavailable(
            f"{path} not found. Fetch it with scripts/fetch_lm_dictionary.sh "
            f"(free for academic research; see sraf.nd.edu). It is not vendored "
            f"because redistribution is not explicitly granted."
        )
    df = pd.read_csv(path)
    words = df["Word"].astype(str).str.upper()
    out: dict[str, frozenset[str]] = {}
    for cat in CATEGORIES:
        if cat in df.columns:
            out[cat] = frozenset(words[df[cat] > 0])
    out["_vocabulary"] = frozenset(words)
    return out


def tokenize(text: str) -> list[str]:
    return [t.upper() for t in _TOKEN_RE.findall(text or "")]


def category_shares(text: str, base_dir: str | Path) -> dict[str, float]:
    """
    Share of alphabetic tokens falling in each L-M category.

    Denominator is ALL alphabetic tokens, which is the convention in Loughran and
    McDonald (2011) — not the in-dictionary tokens only. Reported as proportions,
    not percentages; multiply at the presentation layer.
    """
    cats = load_categories(str(base_dir))
    toks = tokenize(text)
    n = len(toks)
    if n == 0:
        return {f"lm_{c.lower()}": float("nan") for c in CATEGORIES} | {"n_tokens": 0}

    out: dict[str, float] = {"n_tokens": n}
    for cat in CATEGORIES:
        if cat in cats:
            out[f"lm_{cat.lower()}"] = sum(1 for t in toks if t in cats[cat]) / n
    out["lm_in_vocabulary"] = sum(1 for t in toks if t in cats["_vocabulary"]) / n
    return out


def compare_with_legacy_list(base_dir: str | Path) -> dict:
    """
    Quantify how far the vendored `LM_NEGATIVE_WORDS` list departs from the real
    Negative category. Used to disclose the discrepancy rather than bury it.
    """
    cats = load_categories(str(base_dir))
    try:
        from analysis.textual_baselines import LM_NEGATIVE_WORDS as legacy
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    legacy_set = {str(w).upper() for w in legacy}
    negative = cats["Negative"]
    not_negative = sorted(legacy_set - negative)

    placement: dict[str, list[str]] = {}
    for w in not_negative:
        hits = [c for c in CATEGORIES if c in cats and w in cats[c]]
        key = "+".join(hits) if hits else (
            "in dictionary, no sentiment category" if w in cats["_vocabulary"]
            else "not in dictionary at all"
        )
        placement.setdefault(key, []).append(w)

    return {
        "legacy_list_size": len(legacy_set),
        "lm_negative_size": len(negative),
        "coverage_of_lm_negative": len(legacy_set & negative) / len(negative),
        "legacy_words_not_in_lm_negative": len(not_negative),
        "misplacement_breakdown": {k: len(v) for k, v in placement.items()},
        "examples": not_negative[:15],
        "verdict": (
            "The legacy list is not the Loughran-McDonald Negative category. It "
            "covers a small fraction of it and includes words L-M assigns to other "
            "categories or to none. Reporting it as 'L-M negative' would be wrong."
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import json
    import sys

    sys.modules.setdefault("tensorflow", None)
    from config import Config

    cfg = Config()
    try:
        cats = load_categories(str(cfg.BASE_DIR))
    except DictionaryUnavailable as exc:
        print(exc)
        raise SystemExit(1)

    print(f"Loaded {DICT_FILENAME}")
    print(f"  vocabulary: {len(cats['_vocabulary']):,} words")
    for c in CATEGORIES:
        if c in cats:
            print(f"  {c:14s}: {len(cats[c]):>6,}")
    print()
    print("Comparison with the vendored LM_NEGATIVE_WORDS list:")
    print(json.dumps(compare_with_legacy_list(cfg.BASE_DIR), indent=1))
