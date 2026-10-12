"""
Fixed-window chunking of complete filing sections.

Why this exists
---------------
Through Round 7 the pipeline enforced uniformity across firms with a global word
cap (MAX_WORDS_PER_FILING).  That was the wrong instrument.  Measured on the
30-firm sample, the cap truncated 59 of 59 filings, and the *fraction* of each
narrative section that survived truncation was itself correlated with firm size:

    Spearman(tier rank [large=3], fraction of section read) = -0.301, p = 0.0205

Median fraction read was 15.7% for large-caps, 17.0% for mid-caps and 27.5% for
small-caps (median complete section length 12,733 / 11,750 / 7,285 words).  So
"how much of the filing the model actually read" varied systematically with the
independent variable, and any tier gradient in disagreement was confounded with
a tier gradient in coverage.

The fix is to make the *unit of analysis* uniform instead of the *document*:

  1. Extract MD&A and Risk Factors complete (data/edgar.extract_sections_dict).
  2. Split each section into fixed-size windows of config.CHUNK_WORDS words.
  3. Every persona scores every chunk.
  4. Disagreement is computed per chunk, then aggregated to the filing.

Every firm is now read in full, and every persona judgment is made on an
identically sized window.  Filing length enters the design as a chunk COUNT —
an observable control variable (n_chunks) — rather than as an unobserved
difference in how much of the document each firm's score was based on.

A second benefit: holding the window size fixed also holds the model's effective
reading condition fixed.  A 20B model given 100k tokens degrades on
long-context retrieval, and that degradation would be worst for the longest
filings, i.e. the largest firms — reintroducing the same confound in a subtler
form.  Fixed windows remove it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Human-readable labels shown to the personas so a chunk is not read out of context.
SECTION_LABELS: dict[str, str] = {
    "mda": "Management's Discussion and Analysis",
    "risk": "Risk Factors",
    "market_risk": "Quantitative and Qualitative Disclosures About Market Risk",
}


def chunk_section(
    text: str,
    chunk_words: int,
    min_tail_words: int,
) -> list[str]:
    """
    Split one section into consecutive windows of `chunk_words` words.

    A trailing remainder shorter than `min_tail_words` is merged into the previous
    chunk rather than emitted on its own.  Emitting a 40-word final chunk would
    create an observation whose disagreement is dominated by having almost nothing
    to disagree about, and short sections would produce proportionally more of
    them — a length artifact of exactly the kind this module exists to remove.
    """
    words = text.split()
    # A section too short to fill one scoreable window is not scored at all.
    # Emitting it anyway would put a 224-word window and a 1,200-word window into
    # the same unweighted mean, which is the length artifact this module exists
    # to remove, reintroduced through the back door.
    if len(words) < min_tail_words:
        return []
    if len(words) <= chunk_words + min_tail_words:
        return [" ".join(words)]

    chunks: list[str] = []
    for start in range(0, len(words), chunk_words):
        chunks.append(" ".join(words[start:start + chunk_words]))

    # Merge an undersized tail into its predecessor.
    if len(chunks) > 1 and len(chunks[-1].split()) < min_tail_words:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()

    return chunks


def build_filing_chunks(
    sections: dict[str, str],
    form_type: str,
    filing_date: str,
    chunk_words: int,
    min_tail_words: int,
    sections_to_analyze: tuple[str, ...] = ("mda", "risk"),
) -> list[dict]:
    """
    Turn a filing's complete sections into the list of chunk records the personas
    will score.

    Returns a list of dicts:
      section       — 'mda' | 'risk'
      chunk_index   — 0-based position within that section
      n_in_section   — total chunks in that section
      word_count    — words in this chunk
      text          — the prompt-ready text, with a short context header

    Only MD&A and Risk Factors are analyzed by default: those are the two sections
    the hypothesis is about, and the two the advisor named.  Item 7A / Item 3
    (Market Risk) is still extracted and cached, but is not scored, so that the
    scored corpus is identical in composition across all firms.

    The context header tells the persona which section and which part of it it is
    reading.  It is deliberately identical in structure for every chunk of every
    firm, and carries no firm identity, so it adds no cross-firm variation.
    """
    records: list[dict] = []

    for key in sections_to_analyze:
        body = sections.get(key)
        if not body:
            continue

        pieces = chunk_section(body, chunk_words, min_tail_words)
        label = SECTION_LABELS.get(key, key)

        for i, piece in enumerate(pieces):
            header = (
                f"[Section: {label} | part {i + 1} of {len(pieces)} "
                f"| form: {form_type} | filed: {filing_date}]"
            )
            records.append({
                "section": key,
                "chunk_index": i,
                "n_in_section": len(pieces),
                "word_count": len(piece.split()),
                "text": f"{header}\n\n{piece}",
            })

    if not records:
        logger.warning(
            "build_filing_chunks: no analyzable sections among %s for %s %s",
            sections_to_analyze, form_type, filing_date,
        )

    return records
