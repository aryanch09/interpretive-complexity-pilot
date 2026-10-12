"""
SEC EDGAR data acquisition and filing text extraction.

Flow per firm:
  1. Fetch submissions JSON  →  find post-cutoff 10-K / 10-Q accession numbers
  2. For each accession, resolve the human-readable primary document URL
  3. Download and clean the HTML/text
  4. Extract only narrative sections (MD&A, Risk Factors, Market Risk)
  5. Verify extraction quality: if fewer than MIN_EXTRACTION_WORDS words, fall back
     to the first MAX_WORDS words of the full cleaned text rather than silently
     passing a one-paragraph stub to the agents.
  6. Truncate to config.MAX_WORDS_PER_FILING and save to disk

Section extraction strategy:
  - Regex finds all "ITEM X." markers and their text spans
  - We keep the longest occurrence of each target item number to skip the TOC
  - Sections shorter than MIN_SECTION_CHARS are treated as TOC entries and dropped
  - If total extracted length is below MIN_EXTRACTION_WORDS, we fall back to the
    full document rather than passing an uninformative stub to the agents.
    A flag 'extraction_short' is set in the returned filing dict when this happens.

Quality note: verified on pilot data — JPMorgan 10-Qs yielded <210 words after
section extraction (the filing's structure uses XBRL tables rather than narrative
Item headers). The fallback catches these cases and logs a warning.
"""

import json
import logging
import re
import time
import warnings
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"

MIN_SECTION_CHARS = 800    # shorter → assume TOC entry, skip
MIN_EXTRACTION_WORDS = 300  # fewer after section extraction → fall back to full text


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url: str, user_agent: str, sleep: float = 0.5, timeout: int = 60) -> requests.Response:
    headers = {"User-Agent": user_agent}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    time.sleep(sleep)
    return resp


# ── Filing discovery ──────────────────────────────────────────────────────────

def get_recent_filings(
    cik: int,
    form_type: str,
    user_agent: str,
    after_date: str = "2023-11-01",
    max_results: int = 2,
) -> list[dict]:
    """Return up to max_results post-cutoff filings of the given form type."""
    cik_padded = str(cik).zfill(10)
    url = f"{EDGAR_BASE}/submissions/CIK{cik_padded}.json"
    resp = _get(url, user_agent)
    data = resp.json()

    filings = data["filings"]["recent"]
    results: list[dict] = []

    for i, form in enumerate(filings["form"]):
        if form == form_type and filings["filingDate"][i] >= after_date:
            results.append({
                "accessionNumber": filings["accessionNumber"][i],
                "filingDate": filings["filingDate"][i],
                "primaryDocument": filings["primaryDocument"][i],
                "form": form_type,
            })
            if len(results) >= max_results:
                break

    # Check extra pagination pages (rarely needed for recent filings)
    if len(results) < max_results and "files" in data["filings"]:
        for page_meta in data["filings"]["files"][:2]:
            page_url = f"{EDGAR_BASE}/submissions/{page_meta['name']}"
            try:
                page_resp = _get(page_url, user_agent)
                page_data = page_resp.json()
                for i, form in enumerate(page_data["form"]):
                    if form == form_type and page_data["filingDate"][i] >= after_date:
                        results.append({
                            "accessionNumber": page_data["accessionNumber"][i],
                            "filingDate": page_data["filingDate"][i],
                            "primaryDocument": page_data["primaryDocument"][i],
                            "form": form_type,
                        })
                        if len(results) >= max_results:
                            break
            except Exception as exc:
                logger.debug("Could not fetch extra submissions page: %s", exc)
            if len(results) >= max_results:
                break

    return results


# ── Document URL resolution ───────────────────────────────────────────────────

def _resolve_document_url(
    cik: int,
    accession_number: str,
    primary_document: str,
    user_agent: str,
    form_type: str = "",
) -> str:
    """
    Build the direct URL to the filing's human-readable document.

    If primaryDocument is an XML/XBRL file, fetch the filing index and pick
    the best .htm file — preferring files whose name contains the form type
    keyword (e.g. "10-k" or "10-q") over files that look like exhibits.
    """
    acc_clean = accession_number.replace("-", "")
    base = f"{EDGAR_ARCHIVES}/{cik}/{acc_clean}"
    doc_url = f"{base}/{primary_document}"

    if primary_document.lower().endswith((".xml", ".xsd", "_htm.xml")):
        doc_url = _find_htm_document(base, user_agent, form_type) or doc_url

    return doc_url


def _find_htm_document(filing_base_url: str, user_agent: str, form_type: str = "") -> Optional[str]:
    """
    Fetch the filing index and return the URL of the most likely primary document.

    Selection priority:
      1. .htm files whose filename contains the form type keyword (10-k, 10-q)
      2. The largest .htm file by byte size (approximated by link text length)
    Exhibits (filenames starting with 'ex' or containing 'exhibit') are deprioritized.
    """
    try:
        resp = _get(filing_base_url + "/", user_agent)
        soup = BeautifulSoup(resp.text, "lxml")
        candidates: list[tuple[int, str]] = []  # (priority_score, url)

        form_keyword = form_type.lower().replace("-", "")  # "10k" or "10q"

        for a in soup.find_all("a", href=True):
            href: str = a["href"]
            if not href.lower().endswith((".htm", ".html")):
                continue
            fname = href.split("/")[-1].lower()
            link = filing_base_url + "/" + fname

            # Deprioritize known exhibit patterns
            is_exhibit = fname.startswith("ex") or "exhibit" in fname or fname.startswith("r")
            if is_exhibit:
                continue

            # Score: higher = better candidate
            score = len(a.get_text())
            if form_keyword and form_keyword in fname:
                score += 100_000  # strong preference for form-type-named file

            candidates.append((score, link))

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
    except Exception as exc:
        logger.debug("Could not fetch filing index: %s", exc)
    return None


# ── Text extraction and cleaning ──────────────────────────────────────────────

_HIDDEN_STYLE_RE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*hidden", re.IGNORECASE
)
# A sparse <table> whose entire text is nothing but an Item heading.  Some filers
# (XOM, AMZN) render every Item heading as its own one-cell table, which the
# sparse-table pruner below would otherwise delete along with the real tables,
# leaving the document with no Item headers at all.  Anchored at both ends so a
# table containing a heading plus any other text does not match.  The character
# class allows curly apostrophes and en/em dashes ("MANAGEMENT’S"), and \s
# covers the non-breaking spaces these filers use ("Item\xa01A").
_ITEM_HEADER_TABLE_RE = re.compile(
    r"^(?:ITEM|Item)\s+\d+[A-Za-z]?\s*[.\-:)\u2013\u2014]?\s*"
    r"[A-Za-z][A-Za-z0-9\s,&'\u2018\u2019\u201c\u201d.:;/()\[\]\-\u2013\u2014]{3,90}$"
)
# Matches any namespace-prefixed token: us-gaap:LongToken, pg:segment, xbrli:shares,
# iso4217:USD, pg:A0.110NotesDue2026Member (dots allowed in right-hand side).
_XBRL_TOKEN_RE = re.compile(r"\b[a-z][\w-]*:[A-Za-z][\w.]{2,}\b")
# EDGAR CIK padding / bare 10-digit identifiers that slip through as "words"
_CIK_RE = re.compile(r"\b0{4,}\d{5,}\b")


def _is_prose_quality(text: str, garbage_threshold: float = 0.40) -> bool:
    """
    Return True if the text looks like real English prose.

    Checks the fraction of whitespace-split tokens that look like XBRL metadata
    (namespace:token, bare long integers, or ISO date strings).  If that fraction
    exceeds garbage_threshold the text is considered extraction garbage.
    """
    words = text.split()
    if not words:
        return False
    garbage = sum(
        1 for w in words
        if ":" in w
        or re.match(r"^\d{8,}$", w)
        or re.match(r"^\d{4}-\d{2}-\d{2}$", w)
    )
    return (garbage / len(words)) < garbage_threshold


def _html_to_text(html: str) -> str:
    """Strip HTML tags and return clean prose text."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup.find_all(["script", "style", "img", "head", "meta", "link"]):
        tag.decompose()

    # Strip inline iXBRL context blocks — these sit inside <div style="display:none">
    # in SEC inline XBRL filings (PG, AMZN, GOOGL, etc.) and contain only metadata,
    # not readable prose.  Must run BEFORE get_text() or their text content leaks through.
    for tag in soup.find_all(style=_HIDDEN_STYLE_RE):
        tag.decompose()

    # Replace sparse tables with their text; discard truly empty ones
    for table in soup.find_all("table"):
        text_in_table = table.get_text(" ", strip=True)
        if len(text_in_table.split()) < 40:
            if (
                len(text_in_table.split()) <= 16
                and _ITEM_HEADER_TABLE_RE.match(text_in_table)
            ):
                # Keep the heading as text.  The leading newline is required:
                # _ITEM_PATTERN anchors on a line start, so without it the header
                # would be glued to the preceding line and never seen.
                table.replace_with(soup.new_string("\n" + text_in_table + "\n"))
            else:
                table.decompose()
        else:
            table.replace_with(soup.new_string(text_in_table + "\n"))

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Strip any remaining XBRL namespace tokens and taxonomy URIs.
    text = _XBRL_TOKEN_RE.sub(" ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = _CIK_RE.sub(" ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


# Item heading regex: "ITEM 1A." "Item 7." "ITEM 7A -" at a line start.
#
# The line anchor is load-bearing and was kept after testing the alternative.
# Dropping it to catch inline headers in iXBRL-derived text (where "Item 7."
# can appear mid-line) tripled the marker count but admitted far more
# cross-references than real headers — "...as described in Item 1A. Risk
# Factors of this Annual Report..." names the section, so even requiring the
# title in the header does not screen it out.  Median 10-K MD&A length fell
# from ~7,600 to ~3,000 words.  Filings whose body headers are genuinely
# inline are handled by _title_based_sections instead.
_ITEM_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:ITEM|Item)\s+(\d+[A-Za-z]?)\s*[.\-:)]\s*([^\n]{0,120})",
    re.MULTILINE,
)

# Part II sections that share item numbers with Part I narrative sections.
# For a 10-Q, Item 2 in Part I is MD&A; Item 2 in Part II is "Unregistered Sales of
# Equity Securities and Use of Proceeds" — a boilerplate table, not narrative text.
# Reject any Item 2 candidate whose opening 300 chars match these patterns.
_PART_II_ITEM2_RE = re.compile(
    r"Unregistered\s+Sales|Use\s+of\s+Proceeds|Mine\s+Safety|Other\s+Information",
    re.IGNORECASE,
)

# Title-based section boundaries (used when ITEM markers are absent or only in TOC).
# Each tuple: (regex_pattern, section_key).  The LAST match that is NOT a TOC entry
# (detected by absence of a trailing page-number like " 14" or " 30") is used.
_TITLE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Management.s Discussion and Analysis", re.IGNORECASE), "mda"),
    (re.compile(r"Quantitative and Qualitative Disclosures About Market Risk", re.IGNORECASE), "market_risk"),
    (re.compile(r"(?<!\w)Risk Factors(?!\w)", re.IGNORECASE), "risk"),
]
# TOC entries are followed by a space + page number (1-3 digits) near end of line
_TOC_ENTRY_RE = re.compile(r"\s+\d{1,3}\s*(?:\n|$)")


# Titles that terminate a narrative section when ITEM markers are unusable.
_BOUNDARY_TITLES: list[re.Pattern] = [
    re.compile(r"Management.s Discussion and Analysis", re.IGNORECASE),
    re.compile(r"Quantitative and Qualitative Disclosures About Market Risk", re.IGNORECASE),
    re.compile(r"(?<!\w)Risk Factors(?!\w)", re.IGNORECASE),
    re.compile(r"Financial Statements and Supplementary Data", re.IGNORECASE),
    re.compile(r"Report of Independent Registered Public Accounting Firm", re.IGNORECASE),
    re.compile(r"Controls and Procedures", re.IGNORECASE),
    re.compile(r"(?<!\w)Legal Proceedings(?!\w)", re.IGNORECASE),
    re.compile(r"Unresolved Staff Comments", re.IGNORECASE),
    re.compile(r"Directors, Executive Officers", re.IGNORECASE),
    re.compile(r"Exhibits(?:,| and) Financial Statement Schedules", re.IGNORECASE),
    re.compile(r"Notes to (?:the )?Consolidated Financial Statements", re.IGNORECASE),
    # Part II headings of a 10-Q.  Without these, a Part II "Risk Factors" span
    # ran to the next boundary of any kind — often the end of the document —
    # producing an 18,993-word "Risk Factors" section for a JPMorgan 10-Q whose
    # actual Item 1A states there have been no material changes.
    re.compile(r"Unregistered Sales of Equity Securities", re.IGNORECASE),
    re.compile(r"Defaults Upon Senior Securities", re.IGNORECASE),
    re.compile(r"Mine Safety Disclosures", re.IGNORECASE),
    re.compile(r"(?<!\w)Other Information(?!\w)", re.IGNORECASE),
    re.compile(r"(?<!\w)Exhibits(?!\w)", re.IGNORECASE),
    re.compile(r"Critical Accounting (?:Estimates|Policies)", re.IGNORECASE),
]
# Indices 0/1/2 above are MD&A / Market Risk / Risk Factors and are referenced by
# position in _title_based_sections; append new boundaries, never insert.


def _title_based_sections(text: str) -> dict[str, str]:
    """
    Recover MD&A and Risk Factors by section TITLE when ITEM markers are unusable.

    Needed for iXBRL filers (AMZN, GOOGL, WMT) whose body text carries section
    titles but no "Item 7." markers, and for filers like JPMorgan whose Item 7
    span is a one-line incorporation by reference.

    Two rules make this reliable:

    * Every section is BOUNDED — it runs from its title to the next boundary
      title of a different kind.  Without an end boundary this returned
      everything from the anchor to the end of the document, which is how a
      "MD&A" of 107,257 words containing the complete financial statements was
      reaching the personas.

    * The end boundary must lie at least MIN_SECTION_CHARS past the start, and
      among all candidate anchors the LONGEST valid span wins.  Narrative
      sections open by cross-referencing their neighbours ("...see Risk Factors
      below..."), and taking the first boundary hit unconditionally truncated
      MD&A to a single paragraph — 149 words for Amazon's 10-K.  Requiring
      distance skips those in-paragraph mentions; taking the longest span skips
      running-header repetitions of the title.

    The table of contents is ignored (first 15% of the document); a TOC lists
    every one of these titles in order and would otherwise supply all anchors.
    """
    text_len = len(text)
    toc_cutoff = int(text_len * 0.15)

    hits: list[tuple[int, int]] = []  # (position, boundary_index)
    for b_idx, pattern in enumerate(_BOUNDARY_TITLES):
        for m in pattern.finditer(text):
            if m.start() > toc_cutoff:
                hits.append((m.start(), b_idx))
    hits.sort()
    if not hits:
        return {}

    wanted = {0: "mda", 1: "market_risk", 2: "risk"}
    sections: dict[str, str] = {}

    for b_idx, key in wanted.items():
        best = ""
        for pos, hit_b in hits:
            if hit_b != b_idx:
                continue
            # End at the next boundary of a different kind that is far enough
            # away to be a real section break rather than a cross-reference.
            end = text_len
            for next_pos, next_b in hits:
                if (
                    next_b != b_idx
                    and next_pos >= pos + MIN_SECTION_CHARS
                ):
                    end = next_pos
                    break
            content = text[pos:end].strip()
            if (
                len(content) >= MIN_SECTION_CHARS
                and len(content) > len(best)
                and _is_prose_quality(content)
            ):
                best = content
        if best:
            sections[key] = best

    return sections


# Subsection headings that Regulation S-K Item 303 effectively requires inside
# MD&A.  "Liquidity and Capital Resources" in particular appears in virtually
# every MD&A and virtually nowhere else in a filing, which makes it a reliable
# structural marker when no section header survives HTML-to-text conversion.
_LIQUIDITY_RE = re.compile(r"Liquidity\s+and\s+Capital\s+Resources", re.IGNORECASE)
_MDA_OPENING_RE = [
    re.compile(r"(?<!\w)Overview(?!\w)", re.IGNORECASE),
    re.compile(r"Results\s+of\s+Operations", re.IGNORECASE),
]


def _structural_mda(text: str) -> str:
    """
    Last-resort MD&A recovery for filings with no usable MD&A header at all.

    Walmart's and Amazon's 10-Ks reach the text layer with their MD&A heading
    present only in the table of contents — the body section is unlabelled, so
    both the ITEM path and the title path return nothing, and two large-cap
    observations would be lost.  Rather than hand-tune anchors per filer (which
    would mean tuning extraction against the independent variable), this locates
    MD&A by its mandated internal structure: the section is bracketed by its
    conventional opening heading and by "Liquidity and Capital Resources".

    Returns "" if the structure is not found, in which case the filing is
    correctly treated as unusable rather than guessed at.
    """
    text_len = len(text)
    toc_cutoff = int(text_len * 0.15)

    liq = next(
        (m.start() for m in _LIQUIDITY_RE.finditer(text) if m.start() > toc_cutoff),
        None,
    )
    if liq is None:
        return ""

    # Opening heading: the last one before Liquidity, preferring "Overview".
    start = None
    for pattern in _MDA_OPENING_RE:
        hits = [m.start() for m in pattern.finditer(text) if toc_cutoff < m.start() < liq]
        if hits:
            start = hits[-1]
            break
    if start is None:
        return ""

    # End at the next major boundary title after the Liquidity heading.
    end = text_len
    for pattern in _BOUNDARY_TITLES:
        for m in pattern.finditer(text):
            if m.start() > liq and m.start() < end:
                end = m.start()
                break

    content = text[start:end].strip()
    if len(content) < MIN_SECTION_CHARS or not _is_prose_quality(content):
        return ""
    return content


def _title_based_extraction(text: str) -> str:
    """Backward-compatible wrapper returning the concatenated title-based sections."""
    sections = _title_based_sections(text)
    return "\n\n".join(
        sections[k] for k in ("mda", "risk", "market_risk") if k in sections
    )


# ── Section boundary tables ───────────────────────────────────────────────────
# A section runs from its own ITEM header to the header of the next section in
# the FORM's canonical item order — not to "the next ITEM marker of any kind".
# That distinction matters: filings routinely omit items, print them out of
# order, or mention them only in cross-references, and a span that runs to the
# next marker found anywhere in the document then swallows the rest of the
# filing.  Before this table was introduced, JPMorgan's 10-K "MD&A" came back as
# 107,257 words — the entire remainder of the document, financial statements
# included — while its real Item 7 span was 395 characters.
_SUCCESSORS: dict[str, dict[str, tuple[str, ...]]] = {
    "10-K": {
        "7":  ("7a", "8"),
        "7a": ("8",),
        "1a": ("1b", "1c", "2"),
    },
    "10-Q": {
        "2":  ("3", "4"),
        "3":  ("4",),
        "1a": ("2", "3", "4", "5", "6"),
    },
}

# Expected title text on a genuine section header, used to score candidates.
# Whitespace-tolerant: HTML-to-text conversion routinely puts the apostrophe of
# "MANAGEMENT'S DISCUSSION" on its own line ("MANAGEMENT\n’\nS DISCUSSION"), and a
# pattern that allowed only a single character there silently rejected the real
# MD&A header in Clearfield's 10-K and Cathay's and Insteel's 10-Qs.
_EXPECTED_TITLE: dict[str, re.Pattern] = {
    "mda": re.compile(r"Management\s*[’'`\u00b4]?\s*s\s+Discussion", re.IGNORECASE),
    "risk": re.compile(r"Risk\s+Factors", re.IGNORECASE),
    "market_risk": re.compile(r"Quantitative\s+and\s+Qualitative", re.IGNORECASE),
}

# Language that marks an ITEM mention as a cross-reference inside a sentence
# ("...as described in Item 1A. Risk Factors of this Annual Report...") rather
# than an actual section header.  Checked on the text just before the marker and
# on the header's own trailing text.
_XREF_BEFORE_RE = re.compile(
    r"(?:see|refer(?:red|ence)?\s+to|described|discussed|included|contained|set\s+forth|"
    # "incorporated" is only a cross-reference cue in the phrase "incorporated
    # (herein) by reference"; as a bare trailing word it is almost always the tail
    # of a company name.  AMZN's 10-K lists "Chairman, CEO, and President, Corning
    # Incorporated" in a directors table immediately before its real
    # "Item 1A. Risk Factors" header, and the bare form rejected that real header
    # as a cross-reference, costing the whole section.
    r"incorporated\s+(?:herein\s+)?by\s+reference(?:\s+to)?|"
    r"listed|identified|disclosed|under|in|of|and|,)\s*[\"'\u2018\u201c]?\s*$",
    re.IGNORECASE,
)
_XREF_AFTER_RE = re.compile(
    r"^\s*[^\n]{0,120}?\b(?:of\s+this|in\s+this|to\s+this|of\s+our|of\s+Part|in\s+Part|"
    r"of\s+the\s+(?:Annual|Quarterly)|beginning\s+on\s+page|herein|above|below)\b",
    re.IGNORECASE,
)


def _is_cross_reference(text: str, match: re.Match) -> bool:
    """True when an ITEM marker is a mid-sentence cross-reference, not a header."""
    before = text[max(0, match.start() - 60):match.start()]
    # A real header starts a line; a cross-reference sits inside running prose.
    tail = before.rstrip()
    if tail and not tail.endswith((".", ":", ";", ")", "\n")):
        if _XREF_BEFORE_RE.search(tail):
            return True
    header_tail = match.group(2) or ""
    return bool(_XREF_AFTER_RE.match(header_tail))


def _section_span(
    text: str,
    matches: list[re.Match],
    idx: int,
    successors: tuple[str, ...],
) -> str:
    """
    Text from marker `idx` to the next marker whose item number is one of
    `successors`.  Falls back to the next marker of any kind, then to end of
    document, so a filing that omits the successor item still yields a section.
    """
    start = matches[idx].start()
    for j in range(idx + 1, len(matches)):
        if matches[j].group(1).lower() in successors:
            return text[start:matches[j].start()].strip()
    if idx + 1 < len(matches):
        return text[start:matches[idx + 1].start()].strip()
    return text[start:].strip()


def extract_sections_dict(text: str, form_type: str) -> tuple[dict[str, str], bool]:
    """
    Pull out the narrative sections as SEPARATE, COMPLETE sections.

    10-K : Item 7  (MD&A), Item 1A (Risk Factors), Item 7A (Market Risk)
    10-Q : Item 2  (MD&A), Item 1A (Risk Factors), Item 3  (Market Risk)

    Returns ({section_key: full_section_text}, extraction_failed).

    Round 8 change (advisor directive, September 2026)
    --------------------------------------------------
    Sections are returned COMPLETE — there is no word cap applied here and none
    applied downstream.  A fixed global word cap (previously 2000) truncated
    every filing at a different point in its MD&A: 58 of 59 filings in the
    30-firm sample were cut mid-section, and the surviving fraction of the
    section was itself correlated with firm size (Spearman rho = -0.344,
    p = 0.008 against tier rank).  That made "how much of the filing the model
    read" a function of the independent variable.  Uniform treatment is now
    enforced at the CHUNK level instead: complete sections are split into
    fixed-size windows (config.CHUNK_WORDS) so every firm is read in full and
    every persona judgment is made on an identically sized window.
    """
    if form_type == "10-K":
        targets = {"7": "mda", "1a": "risk", "7a": "market_risk"}
    else:
        targets = {"2": "mda", "1a": "risk", "3": "market_risk"}

    successors = _SUCCESSORS.get(form_type, {})
    sections: dict[str, str] = {}
    matches = list(_ITEM_PATTERN.finditer(text))

    for item_num, key in targets.items():
        candidates: list[tuple[int, int, str]] = []  # (title_match, length, content)
        for idx, m in enumerate(matches):
            if m.group(1).lower() != item_num:
                continue
            if _is_cross_reference(text, m):
                logger.debug("Item %s at %d rejected as cross-reference", item_num, m.start())
                continue
            content = _section_span(text, matches, idx, successors.get(item_num, ()))
            if len(content) < MIN_SECTION_CHARS:
                continue
            if form_type == "10-Q" and item_num == "2":
                # Part II Item 2 is "Unregistered Sales of Equity Securities",
                # not MD&A.  It is often the longer of the two after cleaning.
                if _PART_II_ITEM2_RE.search(content[:300]):
                    continue
            if not _is_prose_quality(content):
                continue
            # The header must name the section.  With the relaxed marker pattern
            # most surviving "Item 7"/"Item 1A" hits in iXBRL filings are
            # cross-references whose following words are ordinary prose; requiring
            # the section title in the header text is what separates a real
            # section header from a sentence that happens to cite one.
            title_re = _EXPECTED_TITLE.get(key)
            if title_re and not title_re.search(content[:200]):
                continue
            candidates.append((1, len(content), content))

        if candidates:
            # Prefer a candidate whose header names the section; among those,
            # take the longest.
            candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
            sections[key] = candidates[0][2]

    # ── Title-based recovery for whatever the ITEM path could not deliver ─────
    # Applied per-section, not as an all-or-nothing replacement: a filing may have
    # a perfectly good Item 7 span but no usable Item 1A, and there is no reason
    # to discard the good one.
    missing = [k for k in ("mda", "risk") if k not in sections]
    used_fallback = False
    if missing:
        recovered = _title_based_sections(text)
        for key in missing:
            if key in recovered:
                sections[key] = recovered[key]
                used_fallback = True
                logger.info("Recovered '%s' via title-based extraction (%d words)",
                            key, len(recovered[key].split()))
    # NOTE: used_fallback is informational only.  Before Round 8 the title-based
    # path returned everything from its anchor to the end of the document, so
    # using it did mean the personas saw the wrong text and the filing was flagged
    # unusable.  It is now bounded per section and produces sections of the same
    # kind as the ITEM path, so taking that route is not itself a defect.

    # Also applied when a span WAS found but is implausibly short: Amazon's 10-K
    # yields a 425-word "MD&A" from a cross-reference paragraph, which would be
    # scored as if it were the section itself.  Take whichever recovery is longer.
    if len(sections.get("mda", "").split()) < MIN_EXTRACTION_WORDS * 2:
        structural = _structural_mda(text)
        if structural and len(structural.split()) > len(sections.get("mda", "").split()):
            sections["mda"] = structural
            used_fallback = True
            logger.info(
                "Recovered MD&A structurally via Liquidity anchor (%d words)",
                len(structural.split()),
            )

    if "mda" not in sections:
        # MD&A is the section the hypothesis is about; without it the filing
        # cannot contribute a comparable observation.
        logger.warning("No usable MD&A recovered for this %s — filing unusable", form_type)
        return sections, True

    if "risk" not in sections:
        # Legitimate for many 10-Qs, which state "no material changes to the risk
        # factors disclosed in our Annual Report" instead of restating them.
        logger.info("No Risk Factors section in this %s (common for 10-Q)", form_type)

    if used_fallback:
        logger.info("Filing used title-based or structural recovery for one or more sections")

    # Not a failure: MD&A was recovered, by whichever route.
    return sections, False


def _load_sample_manifest(filings_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """
    Load data/sample_manifest.csv, which freezes the analysis sample.

    Columns: ticker, filing_type, filing_date (YYYY-MM-DD). When the file is
    present it REPLACES the "most recent post-cutoff filing" query, so the
    scored sample is the pre-registered one and does not change if a firm files
    again mid-project. Returns {} when absent, restoring the live-query
    behaviour.

    A pinned filing must already be cached under filings_dir; a pinned entry
    with no cached raw HTML is an error rather than a silent EDGAR fetch,
    because the point of the manifest is reproducibility without network access.
    """
    path = Path(filings_dir).parent / "sample_manifest.csv"
    if not path.exists():
        return {}

    import csv

    out: dict[tuple[str, str], list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ticker = row["ticker"].strip()
            form = row["filing_type"].strip()
            date = row["filing_date"].strip()
            safe = date.replace("-", "")
            raw = Path(filings_dir) / f"{ticker}_{form}_{safe}.txt"
            if not raw.exists():
                raise FileNotFoundError(
                    f"sample_manifest.csv pins {ticker} {form} {date} but "
                    f"{raw} is not cached. Fetch it before running, or remove "
                    f"the row; the manifest must not trigger a silent download."
                )
            out.setdefault((ticker, form), []).append({
                "filingDate": date,
                "accessionNumber": row.get("accession_number", ""),
                "primaryDocument": row.get("primary_document", ""),
            })
    return out


def _extract_sections(text: str, form_type: str) -> tuple[str, bool]:
    """
    Backward-compatible wrapper: returns the concatenated narrative sections.
    MD&A first, then Risk Factors, then Market Risk.  Used by callers that want
    a single text blob (calibration, anonymization diagnostics).
    """
    sections, failed = extract_sections_dict(text, form_type)
    ordered = [sections[k] for k in ("mda", "risk", "market_risk") if k in sections]
    return "\n\n".join(ordered), failed


def _truncate(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# ── Public interface ──────────────────────────────────────────────────────────

def fetch_firm_filings(
    cik: int,
    ticker: str,
    filings_dir: Path,
    processed_dir: Path,
    user_agent: str,
    filing_cutoff: str,
    max_words: int = 0,
    max_quarterly: int = 2,
) -> list[dict]:
    """
    Fetch, extract, and cache filing text for a single firm.

    Returns a list of dicts with keys:
      form, filing_date, sections, text, text_path, extraction_short, word_count_raw

    `sections` is {section_key: complete_section_text} with keys drawn from
    'mda', 'risk', 'market_risk'.  Sections are COMPLETE — no word cap is applied
    (Round 8, advisor directive).  `text` is the concatenation, retained for
    textual-baseline features and the anonymization diagnostic.

    `max_words` is accepted for backward compatibility and IGNORED.  Uniformity
    across firms is enforced at the chunk level downstream, not by truncating
    each filing at an arbitrary and firm-size-correlated cut point.

    extraction_short=True flags filings where section extraction failed or where
    the title-based fallback had to be used (MD&A and Risk Factors could not be
    separated).  These observations are excluded from section-level results.
    """
    if max_words:
        logger.debug(
            "fetch_firm_filings: max_words=%d ignored — sections are returned complete",
            max_words,
        )

    filings_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    manifest = _load_sample_manifest(filings_dir)

    for form_type, max_n in [("10-Q", max_quarterly), ("10-K", 1)]:
        pinned = manifest.get((ticker, form_type)) if manifest else None
        if pinned:
            # The analysis sample is FROZEN by data/sample_manifest.csv, so the
            # scored sample cannot drift when a firm files again mid-project.
            # Selection then needs no EDGAR request at all: every pinned filing's
            # raw HTML is already cached, which is what lets the whole pipeline
            # re-run offline.
            filing_list = pinned
            logger.info("  %s: %d pinned %s filing(s) from sample manifest",
                        ticker, len(filing_list), form_type)
        else:
            filing_list = get_recent_filings(cik, form_type, user_agent, filing_cutoff, max_n)
            logger.info("  %s: found %d post-cutoff %s filing(s)",
                        ticker, len(filing_list), form_type)

        for filing_meta in filing_list:
            date_str = filing_meta["filingDate"]
            safe_date = date_str.replace("-", "")
            raw_path = filings_dir / f"{ticker}_{form_type}_{safe_date}.txt"
            stem = f"{ticker}_{form_type}_{safe_date}"
            sections_path = processed_dir / f"{stem}_sections.json"
            quality_path = processed_dir / f"{stem}_quality.txt"

            # ── Cached processed version ──────────────────────────────────────
            if sections_path.exists():
                sections = json.loads(sections_path.read_text(encoding="utf-8"))
                extraction_short = (
                    quality_path.exists() and quality_path.read_text().strip() == "short"
                )
                text = "\n\n".join(
                    sections[k] for k in ("mda", "risk", "market_risk") if k in sections
                )
                logger.info(
                    "    %s %s: loaded from cache (%s; %d words total%s)",
                    form_type, date_str,
                    ", ".join(f"{k}={len(v.split())}w" for k, v in sections.items()),
                    len(text.split()),
                    ", extraction_short" if extraction_short else "",
                )
                results.append({
                    "form": form_type,
                    "filing_date": date_str,
                    "sections": sections,
                    "text": text,
                    "text_path": str(sections_path),
                    "extraction_short": extraction_short,
                    "word_count_raw": len(text.split()),
                    # Per-section counts.  word_count_raw includes market_risk,
                    # which is extracted and cached but NOT scored, so it is the
                    # wrong length control.  These are the scored sections.
                    "mda_words": len((sections.get("mda") or "").split()),
                    "risk_words": len((sections.get("risk") or "").split()),
                })
                continue

            # ── Fetch / re-extract ────────────────────────────────────────────
            try:
                if raw_path.exists():
                    # Raw HTML already downloaded — re-extract offline, no EDGAR hit.
                    raw_html = raw_path.read_text(encoding="utf-8", errors="ignore")
                    logger.info("    Re-extracting %s %s from cached HTML", form_type, date_str)
                else:
                    doc_url = _resolve_document_url(
                        cik,
                        filing_meta["accessionNumber"],
                        filing_meta["primaryDocument"],
                        user_agent,
                        form_type=form_type,
                    )
                    logger.info("    Fetching %s ...", doc_url)
                    raw_html = _get(doc_url, user_agent, sleep=1.0).text
                    raw_path.write_text(raw_html, encoding="utf-8")

                clean_text = _html_to_text(raw_html)
                sections, extraction_short = extract_sections_dict(clean_text, form_type)

                if not sections:
                    logger.error(
                        "    %s %s %s: no narrative sections recovered — skipping filing",
                        ticker, form_type, date_str,
                    )
                    continue

                sections_path.write_text(
                    json.dumps(sections, indent=1), encoding="utf-8"
                )
                quality_path.write_text(
                    "short" if extraction_short else "ok", encoding="utf-8"
                )

                text = "\n\n".join(
                    sections[k] for k in ("mda", "risk", "market_risk") if k in sections
                )
                logger.info(
                    "    Extracted %s from %s %s (%d words total)%s",
                    ", ".join(f"{k}={len(v.split())}w" for k, v in sections.items()),
                    form_type, date_str, len(text.split()),
                    " [EXTRACTION_SHORT]" if extraction_short else "",
                )

                results.append({
                    "form": form_type,
                    "filing_date": date_str,
                    "sections": sections,
                    "text": text,
                    "text_path": str(sections_path),
                    "extraction_short": extraction_short,
                    "word_count_raw": len(text.split()),
                    # Per-section counts.  word_count_raw includes market_risk,
                    # which is extracted and cached but NOT scored, so it is the
                    # wrong length control.  These are the scored sections.
                    "mda_words": len((sections.get("mda") or "").split()),
                    "risk_words": len((sections.get("risk") or "").split()),
                })

            except Exception as exc:
                logger.error(
                    "    Failed to fetch %s %s for %s: %s",
                    form_type, date_str, ticker, exc,
                )

    return results
