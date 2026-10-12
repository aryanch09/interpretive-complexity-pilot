"""
Disagreement measures.

Two complementary metrics:
  numeric_disagreement   — std of three persona SCORE values (what they conclude)
  embedding_disagreement — mean pairwise cosine distance of REASONING embeddings
                           (why they diverge), computed via neural sentence embeddings

Both are computed over the outputs of a prompt ensemble: one model called three
times with different analytical persona prompts. Disagreement reflects how far
a filing pulls different analytical lenses apart, not disagreement between
independent reasoners.

See analysis/embeddings.py for the embedding implementation.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# A filing-level mean resting on too few surviving chunks is not comparable to
# one resting on all of them; such filings are dropped rather than reported.
MIN_USABLE_CHUNKS = 2


# ── Numeric disagreement ──────────────────────────────────────────────────────

def numeric_disagreement(scores: list) -> float:
    """Population std of three persona score values (ddof=0)."""
    valid = [s for s in scores if s is not None and not np.isnan(float(s))]
    if len(valid) < 2:
        return 0.0
    return float(np.std(valid, ddof=0))


# ── Embedding disagreement ─────────────────────────────────────────────────────

def embedding_disagreement(
    reasoning_texts: list[str],
    embedding_model_name: str,
    cache_path: Optional[Path] = None,
) -> float:
    """
    Mean pairwise cosine distance between neural embeddings of agent reasoning texts.

    Uses sentence-transformers/all-MiniLM-L6-v2 loaded via the `transformers`
    library directly (sentence-transformers wrapper is bypassed due to a
    TensorFlow/abseil mutex deadlock on this machine — see analysis/embeddings.py).

    Returns 0.0 if fewer than 2 non-empty texts are provided or if the
    embedding computation fails.
    """
    from analysis.embeddings import compute_embedding_disagreement
    return compute_embedding_disagreement(reasoning_texts, cache_path=cache_path)


# ── Convenience wrapper ───────────────────────────────────────────────────────

def compute_disagreement(
    agent_outputs: dict[str, dict],
    config,
    cache_path: Optional[Path] = None,
) -> dict[str, Optional[float]]:
    """
    Given the dict returned by runner.run_all_agents, compute both disagreement
    measures and return them as a dict.

    Returns None for both measures if any agent failed. A 2-agent distance is not
    comparable to the 3-agent mean pairwise distance, and failed agents store error
    strings in the reasoning field that would corrupt the embedding computation.
    Callers should treat None as a signal to drop the row from analysis.
    """
    roles = ["value", "sentiment", "fundamentals"]

    n_failed = sum(1 for r in roles if agent_outputs[r].get("failed", False))
    if n_failed > 0:
        logger.warning(
            "compute_disagreement: %d of 3 agent(s) failed — returning None for both "
            "measures. Row must be excluded from analysis.",
            n_failed,
        )
        return {"numeric_disagreement": None, "embedding_disagreement": None}

    scores = [agent_outputs[r].get("score") for r in roles]
    reasonings = [agent_outputs[r].get("reasoning") or "" for r in roles]

    emb_cache = cache_path if cache_path is not None else None

    return {
        "numeric_disagreement": numeric_disagreement(scores),
        "embedding_disagreement": embedding_disagreement(
            reasoning_texts=reasonings,
            embedding_model_name=config.EMBEDDING_MODEL,
            cache_path=emb_cache,
        ),
    }


# ── Chunk-level aggregation (Round 8) ─────────────────────────────────────────

def aggregate_chunk_disagreement(chunk_rows: list[dict]) -> dict:
    """
    Aggregate per-chunk disagreement up to the filing level.

    `chunk_rows` is a list of dicts, one per scored chunk, each carrying at least:
        section, chunk_index, word_count,
        numeric_disagreement, embedding_disagreement, any_failed

    Aggregation rules
    -----------------
    * A chunk with any failed persona call contributes nothing (the Round 3 rule:
      a 2-agent distance is not the same statistic as a 3-agent one).
    * The filing-level measure is the UNWEIGHTED mean over surviving chunks.
      Unweighted, not word-weighted: chunks are already equal-sized by
      construction, and weighting by words would quietly restore length as a
      driver of the filing-level number.
    * A filing is usable only if at least `MIN_USABLE_CHUNKS` chunks survived AND
      at least half its chunks survived.  A filing whose mean rests on 2 of 19
      chunks is not measuring the same thing as one resting on 19 of 19.

    Also returns, as controls and as new dependent variables in their own right:
        n_chunks_total / n_chunks_used  — length, now observable
        within_filing_sd_embedding      — how much interpretive disagreement
                                          varies ACROSS parts of one filing
        mda_* / risk_*                  — section-level means, so the hypothesis
                                          can be tested separately on the two
                                          sections the advisor named
    """
    total = len(chunk_rows)
    good = [
        c for c in chunk_rows
        if not c.get("any_failed", False)
        and c.get("embedding_disagreement") is not None
        and c.get("numeric_disagreement") is not None
    ]

    out: dict = {
        "n_chunks_total": total,
        "n_chunks_used": len(good),
        "chunk_coverage": round(len(good) / total, 4) if total else None,
        "numeric_disagreement": None,
        "embedding_disagreement": None,
        "within_filing_sd_embedding": None,
        "within_filing_sd_numeric": None,
        "any_failed": True,
    }
    for key in ("mda", "risk"):
        out[f"{key}_embedding_disagreement"] = None
        out[f"{key}_numeric_disagreement"] = None
        out[f"{key}_n_chunks"] = 0

    if total == 0:
        return out

    if len(good) < MIN_USABLE_CHUNKS or len(good) < total / 2:
        logger.warning(
            "aggregate_chunk_disagreement: only %d of %d chunks usable "
            "(need >=%d and >=half) — filing excluded from analysis.",
            len(good), total, MIN_USABLE_CHUNKS,
        )
        return out

    emb = [float(c["embedding_disagreement"]) for c in good]
    num = [float(c["numeric_disagreement"]) for c in good]

    out["numeric_disagreement"] = float(np.mean(num))
    out["embedding_disagreement"] = float(np.mean(emb))
    # ddof=1: these are estimates of variation across the filing's parts.
    out["within_filing_sd_embedding"] = float(np.std(emb, ddof=1)) if len(emb) > 1 else 0.0
    out["within_filing_sd_numeric"] = float(np.std(num, ddof=1)) if len(num) > 1 else 0.0
    out["any_failed"] = False

    for key in ("mda", "risk"):
        sub = [c for c in good if c.get("section") == key]
        out[f"{key}_n_chunks"] = len(sub)
        if sub:
            out[f"{key}_embedding_disagreement"] = float(
                np.mean([float(c["embedding_disagreement"]) for c in sub])
            )
            out[f"{key}_numeric_disagreement"] = float(
                np.mean([float(c["numeric_disagreement"]) for c in sub])
            )

    return out
