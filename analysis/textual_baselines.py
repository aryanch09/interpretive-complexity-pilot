"""
Textual baseline controls for the interpretive complexity measure.

Motivation (Bushee, Gow & Taylor, 2018; Loughran & McDonald, 2014):
  Linguistic complexity in firm disclosures has two components:
    (a) information component — genuine business complexity
    (b) obfuscation component — deliberately vague or evasive language

  Our LLM disagreement measure could be picking up either component, or
  simple filing length. To test this, we compute textual features of the
  exact text the agents saw, then partial-out their effect from disagreement.
  The residual ("adjusted disagreement") maps to genuine interpretive
  complexity rather than surface-level textual noise.

Features computed here:
  word_count       — total words in the agent-facing text (post-truncation)
  lm_neg_pct       — Loughran-McDonald negative word % (tone proxy)
  avg_sent_len     — average sentence length in words (complexity proxy)
  type_token_ratio — unique words / total words (lexical diversity)
  fog_index        — Gunning Fog Index: 0.4*(words/sents + 100*complex/words)
                     where "complex" = words with ≥3 syllables (readability proxy)

LM word list: Loughran & McDonald (2011 JF), "When is a Liability not a
  Liability?" Full master dictionary: https://sraf.nd.edu/loughranmcdonald/resources/
  The list below is a curated 280-word subset of their negative word category.
  Limitation: this subset covers the highest-frequency terms but omits ~2,075
  words from the full L-M dictionary. Results using lm_neg_pct should be
  interpreted as a lower bound on negative tone, and ideally replicated with
  the full L-M master dictionary (available at the URL above).

Fog Index: Gunning (1952). Words with ≥3 syllables are counted as "complex."
  Syllables estimated by counting vowel groups (aeiou) after stripping common
  suffixes (es, ed, ing). This is an approximation — the CMU Pronouncing
  Dictionary provides exact syllable counts for English words but requires an
  additional dependency (nltk or pronouncing).
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Loughran-McDonald negative word list (curated subset) ─────────────────────
# Source: Loughran & McDonald (2011 JF). This is a 280-word subset of their
# ~2,355-word negative category. See module docstring for limitation note.

LM_NEGATIVE_WORDS: frozenset = frozenset({
    # A
    "abandon", "abandons", "abandoned", "abandonment",
    "abrupt", "abruptly",
    "absence", "absent",
    "abuse", "abuses", "abused", "abusive",
    "adverse", "adversely", "adversity", "adversities",
    "allegation", "allegations", "alleged", "allegedly", "allege", "alleges",
    "anomaly", "anomalies", "anomalous",
    "arbitrary", "artificially",
    "attrition",
    # B
    "bankrupt", "bankruptcy", "bankruptcies",
    "breach", "breaches", "breached",
    "bribery",
    "burden", "burdens", "burdensome",
    # C
    "cancel", "cancels", "cancelled", "cancellation", "cancellations",
    "challenge", "challenges", "challenging",
    "circumvent", "circumvented", "circumventing",
    "claim", "claims",
    "closure", "closures",
    "collapse", "collapses", "collapsed",
    "complaint", "complaints",
    "concern", "concerns",
    "conflict", "conflicts",
    "constraint", "constraints",
    "contested",
    "corruption", "corrupt",
    "counterclaim", "counterclaims",
    "criminal", "criminally",
    # D
    "damage", "damages", "damaged",
    "decline", "declines", "declined", "declining",
    "default", "defaults", "defaulted",
    "defective",
    "deficiency", "deficiencies", "deficient",
    "deficit", "deficits",
    "delay", "delays", "delayed",
    "delisted", "delisting",
    "deny", "denial", "denials",
    "departure", "departures",
    "destruction", "destroy",
    "deteriorate", "deteriorating", "deterioration",
    "difficult", "difficulty", "difficulties",
    "diminish", "diminished", "diminishing",
    "discontinued", "discontinue",
    "dismiss", "dismissed",
    "dispute", "disputes", "disputed",
    "disrupt", "disrupted", "disruption", "disruptions",
    "downgrade", "downgrades", "downgraded",
    "downturn", "downturns",
    "doubt", "doubts",
    # E
    "eliminate", "eliminated", "elimination",
    "enforcement",
    "error", "errors",
    "excessive",
    "exhausted", "exhaustion",
    "expiration",
    "exploitation", "exploit",
    # F
    "fail", "fails", "failed", "failing",
    "failure", "failures",
    "falling",
    "fault", "faults",
    "fine", "fines",
    "forfeit", "forfeiture",
    "fraud", "fraudulent", "fraudulently",
    "frivolous",
    # G
    "guilty",
    # H
    "harm", "harms", "harmful", "harmed",
    "harsh", "harshly",
    "hazard", "hazardous", "hazards",
    "heavy",
    # I
    "illegal", "illegally",
    "impair", "impaired", "impairment", "impairments",
    "impediment", "impediments",
    "impede", "impedes",
    "inadequate", "inadequacy",
    "inability",
    "inaccuracy", "inaccuracies", "inaccurate",
    "inconsistency", "inconsistencies",
    "insufficient",
    "investigate", "investigated", "investigation", "investigations",
    "involuntary",
    # J
    "jeopardize", "jeopardizes",
    "judgment", "judgments",
    # L
    "lacking", "lack",
    "late",
    "layoff", "layoffs",
    "liability", "liabilities",
    "limitation", "limitations",
    "liquidate", "liquidation",
    "litigation",
    "loss", "losses", "lost",
    "lowered",
    # M
    "misconduct",
    "mismanagement",
    "misrepresentation", "misrepresentations",
    "missing",
    "misstated", "misstate",
    # N
    "negligence", "negligent",
    "noncompliance",
    # O
    "obsolete", "obsolescence",
    "omission", "omissions",
    "overstate", "overstated",
    # P
    "penalty", "penalties",
    "plaintiff", "plaintiffs",
    "poor",
    "problem", "problems",
    "punitive",
    # R
    "recall", "recalls",
    "reduced", "reduction",
    "rejected", "rejection",
    "restate", "restatement",
    "restrictive",
    "risk", "risks", "risky",
    # S
    "severe", "severely",
    "shortage", "shortages",
    "slow", "slowed", "slowdown",
    "substandard",
    "suffer", "suffered", "suffering",
    "suspension", "suspensions",
    # T
    "terminated", "termination", "terminations",
    "threat", "threatened", "threatening", "threats",
    # U
    "uncertainty", "uncertainties",
    "unfavorable",
    "unresolved",
    "unsuccessful",
    # V
    "violate", "violates", "violated", "violation", "violations",
    "volatile", "volatility",
    # W
    "warning", "warnings",
    "weakness", "weaknesses",
    "wrong", "wrongful", "wrongfully",
    "worsen", "worsened", "worsening",
    "writedown", "writeoff",
})

_SENT_SPLITTER = re.compile(r"(?<=[.!?])\s+")
_WORD_TOKENIZER = re.compile(r"\b[a-zA-Z]+\b")
_VOWEL_GROUPS = re.compile(r"[aeiou]+")
_SUFFIX_STRIP = re.compile(r"(ing|ed|es|ly|tion|ness|ment|ful|ous|ive|er|est)$")


def _count_syllables(word: str) -> int:
    """Estimate syllable count by counting vowel groups after stripping suffixes."""
    w = word.lower()
    w = _SUFFIX_STRIP.sub("", w)
    count = len(_VOWEL_GROUPS.findall(w))
    return max(1, count)


def _count_complex_words(words: list[str]) -> int:
    """Count words with ≥3 estimated syllables (Gunning Fog 'hard words')."""
    return sum(1 for w in words if _count_syllables(w) >= 3)


# ── Feature computation ───────────────────────────────────────────────────────

def compute_textual_features(text: str) -> dict:
    """
    Compute Loughran-McDonald, readability, and Fog Index features for one text.

    All features are computed on the exact text the agents saw (post-truncation),
    so they measure the information environment of the agent call, not the full
    filing. This matters for the Bushee et al. decomposition: we are partialing
    out the textual characteristics of the INPUT, not the full 10-K.

    Returns:
      word_count       — int
      lm_neg_pct       — float in [0, 1]  (LM subset; see module docstring)
      avg_sent_len     — float (words per sentence)
      type_token_ratio — float in [0, 1]  (lexical diversity)
      fog_index        — float (Gunning Fog Index; higher = harder to read)
    """
    if not text or not text.strip():
        return {
            "word_count": 0,
            "lm_neg_pct": 0.0,
            "avg_sent_len": 0.0,
            "type_token_ratio": 0.0,
            "fog_index": 0.0,
        }

    words = _WORD_TOKENIZER.findall(text)
    word_count = len(words)

    if word_count == 0:
        return {
            "word_count": 0,
            "lm_neg_pct": 0.0,
            "avg_sent_len": 0.0,
            "type_token_ratio": 0.0,
            "fog_index": 0.0,
        }

    words_lower = [w.lower() for w in words]

    # L-M negative word percentage (subset — see module docstring for limitation)
    n_neg = sum(1 for w in words_lower if w in LM_NEGATIVE_WORDS)
    lm_neg_pct = n_neg / word_count

    # Sentence-level features
    sentences = [s.strip() for s in _SENT_SPLITTER.split(text) if s.strip()]
    n_sents = len(sentences) if sentences else 1
    if sentences:
        sent_lengths = [len(_WORD_TOKENIZER.findall(s)) for s in sentences]
        avg_sent_len = float(np.mean(sent_lengths))
    else:
        avg_sent_len = float(word_count)

    # Type-token ratio (lexical diversity)
    type_token_ratio = len(set(words_lower)) / word_count

    # Gunning Fog Index: 0.4 * (words/sentences + 100 * complex_words/words)
    n_complex = _count_complex_words(words)
    fog_index = round(0.4 * (word_count / n_sents + 100 * n_complex / word_count), 2)

    return {
        "word_count": word_count,
        "lm_neg_pct": round(lm_neg_pct, 5),
        "avg_sent_len": round(avg_sent_len, 2),
        "type_token_ratio": round(type_token_ratio, 4),
        "fog_index": fog_index,
    }


def enrich_results_with_text_features(
    results_df: pd.DataFrame,
    processed_dir: Path,
) -> pd.DataFrame:
    """
    Add textual feature columns to results_df by loading the processed filing
    text files that the agents actually saw.

    File naming: {ticker}_{form_type}_{date_safe}_sections.json  (Round 8+)
    falling back to  {ticker}_{form_type}_{date_safe}_extracted.txt  (Round 7 and
    earlier, where the text was a single truncated blob).
    date_safe = filing_date.replace("-", "")

    Features are computed on the concatenation of the sections actually sent to
    the personas, so word_count here is the complete-section length — the length
    control reported to the advisor.

    Works for both old cached runs (which lack these columns) and new runs
    where the features were already computed inline.
    """
    feature_cols = ["word_count", "lm_neg_pct", "avg_sent_len", "type_token_ratio", "fog_index"]
    if all(c in results_df.columns for c in feature_cols):
        logger.info("Textual features already present; skipping enrichment")
        return results_df

    df = results_df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    found, missing = 0, 0
    for idx, row in df.iterrows():
        ticker = row["ticker"]
        form_type = row.get("filing_type", "")
        filing_date = str(row.get("filing_date", ""))
        date_safe = filing_date.replace("-", "")

        sections_path = processed_dir / f"{ticker}_{form_type}_{date_safe}_sections.json"
        legacy_path = processed_dir / f"{ticker}_{form_type}_{date_safe}_extracted.txt"

        if sections_path.exists():
            sections = json.loads(sections_path.read_text(encoding="utf-8"))
            # Match the scored corpus: MD&A + Risk Factors only.
            text = "\n\n".join(
                sections[k] for k in ("mda", "risk") if k in sections
            )
        elif legacy_path.exists():
            text = legacy_path.read_text(encoding="utf-8")
        else:
            logger.debug("Processed file not found: %s", sections_path)
            missing += 1
            continue
        features = compute_textual_features(text)
        for col, val in features.items():
            df.at[idx, col] = val
        found += 1

    logger.info(
        "Textual features enriched: %d rows loaded from disk, %d not found",
        found, missing,
    )
    return df
