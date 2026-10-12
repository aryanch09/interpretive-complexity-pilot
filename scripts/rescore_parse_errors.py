"""
Re-score windows whose persona output could not be parsed.

WHY. The pre-registered rule (ANALYSIS_PLAN.md §8.1) drops any window where a
persona failed, and forbids imputation. The pipeline keyed that rule on an
API-level `failed` flag rather than on actually having three valid outputs, so a
persona that returned unparseable text counted as a success. The consequences in
the affected windows:

  * `numeric_disagreement` computed from the two personas that did parse, and
  * where REASONING could not be extracted, an imputed `text[:500]` string
    feeding the primary embedding measure.

In the first 331 scored windows, 25 were affected and skewed 14 large / 11 mid /
0 small — a tier-correlated bias in the exact comparison the paper is about.

WHAT THIS DOES. Temperature is 0.3 and these were transient formatting slips, so
each affected window is re-scored once. Windows that parse cleanly the second
time are repaired in place. Windows that still fail are marked
`any_failed = True`, which every downstream path already drops — so the fix needs
no change to any aggregation logic.

The window text is not stored, but it is exactly reproducible: extraction and
chunking are deterministic, so the filing is re-extracted from cached HTML and
the window recovered by (section, chunk_index). The script asserts the
reconstructed window's word count matches the recorded one before re-scoring,
and refuses to touch the window if it does not.

Dry run first:  python3 -m scripts.rescore_parse_errors
Apply:          python3 -m scripts.rescore_parse_errors --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.modules.setdefault("tensorflow", None)

from config import Config

logger = logging.getLogger(__name__)
ROLES = ("value", "sentiment", "fundamentals")


def _is_true(v) -> bool:
    return str(v).strip().lower() in {"true", "1"}


def _window_text(config: Config, ticker: str, form: str, date: str,
                 section: str, chunk_index: int) -> str | None:
    """Reproduce a window's exact text from the cached raw filing."""
    from analysis.chunking import chunk_section
    from data.edgar import _html_to_text, extract_sections_dict

    safe = str(date).replace("-", "")
    path = config.FILINGS_DIR / f"{ticker}_{form}_{safe}.txt"
    if not path.exists():
        logger.error("cached filing missing: %s", path.name)
        return None
    sections, _ = extract_sections_dict(
        _html_to_text(path.read_text(encoding="utf-8", errors="ignore")), form
    )
    body = sections.get(section) or ""
    chunks = chunk_section(body, config.CHUNK_WORDS, config.MIN_TAIL_WORDS)
    if chunk_index >= len(chunks):
        logger.error("%s %s %s/%s: chunk_index out of range (%d of %d)",
                     ticker, form, section, chunk_index, chunk_index, len(chunks))
        return None
    return chunks[chunk_index]


def _collect(cache: dict) -> list[tuple[str, int, dict]]:
    """(ticker, position, row) for every window flagged with a parse error."""
    out = []
    for ticker, rows in cache.items():
        if ticker.startswith("__") or not isinstance(rows, list):
            continue
        for i, r in enumerate(rows):
            if _is_true(r.get("any_parse_error")) and not _is_true(r.get("any_failed")):
                out.append((ticker, i, r))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually re-score and rewrite; default is a dry run")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--agent-sleep", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N windows (for testing)")
    args = ap.parse_args(argv)

    config = Config()
    config.API_PROVIDER = args.provider
    config.apply_provider()
    config.AGENT_SLEEP_SECONDS = args.agent_sleep

    cache_path = config.OUTPUT_DIR / "chunk_cache.json"
    csv_path = config.OUTPUT_DIR / "chunk_level_results.csv"
    if not cache_path.exists():
        print(f"{cache_path} not found — nothing to do.")
        return 1

    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    targets = _collect(cache)
    if args.limit:
        targets = targets[:args.limit]

    total = sum(len(v) for k, v in cache.items()
                if not k.startswith("__") and isinstance(v, list))
    print(f"chunk cache: {total:,} windows")
    print(f"flagged with a parse error, not already failed: {len(targets)}")
    if not targets:
        return 0

    import collections
    print("  by tier   :", dict(collections.Counter(r.get("tier") for _, _, r in targets)))
    print("  by section:", dict(collections.Counter(r.get("section") for _, _, r in targets)))
    print(f"  calls required: {len(targets) * 3}")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to re-score.")
        return 0

    backup = cache_path.with_suffix(
        f".backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    shutil.copy2(cache_path, backup)
    print(f"\nbacked up cache to {backup.name}")

    from agents.runner import run_all_agents
    from analysis.disagreement import compute_disagreement

    repaired = failed = skipped = 0
    for n, (ticker, pos, row) in enumerate(targets, 1):
        text = _window_text(config, ticker, row.get("filing_type"),
                            row.get("filing_date"), row.get("section"),
                            int(row.get("chunk_index", 0)))
        if text is None:
            skipped += 1
            continue
        recorded = int(row.get("word_count") or 0)
        actual = len(text.split())
        if recorded and abs(actual - recorded) > 0:
            logger.error("%s %s %s#%s: reconstructed window is %d words but %d was "
                         "recorded — refusing to re-score",
                         ticker, row.get("filing_type"), row.get("section"),
                         row.get("chunk_index"), actual, recorded)
            skipped += 1
            continue

        print(f"[{n}/{len(targets)}] {ticker} {row.get('filing_type')} "
              f"{row.get('section')}#{row.get('chunk_index')} ...", end=" ", flush=True)
        outputs = run_all_agents(text, config=config)
        clean = all(
            outputs.get(r, {}).get("score") is not None
            and not outputs.get(r, {}).get("parse_error")
            and not outputs.get(r, {}).get("failed")
            for r in ROLES
        )
        if clean:
            dis = compute_disagreement(
                outputs, config,
                cache_path=config.OUTPUT_DIR / "embedding_cache.json",
            )
            row.update({
                "numeric_disagreement": dis.get("numeric_disagreement"),
                "embedding_disagreement": dis.get("embedding_disagreement"),
                "any_parse_error": False,
                "any_failed": False,
            })
            for r in ROLES:
                row[f"{r}_score"] = outputs[r].get("score")
                row[f"{r}_reasoning"] = outputs[r].get("reasoning")
            served = {str(outputs[r].get("served_by")) for r in ROLES
                      if outputs[r].get("served_by")}
            if served:
                row["served_by"] = "|".join(sorted(served))
            repaired += 1
            print("repaired")
        else:
            # Still unparseable: a persona did not produce a usable judgment, so
            # the window is a failure under the pre-registered rule. Marking
            # any_failed is sufficient — every downstream path already drops it.
            row["any_failed"] = True
            row["numeric_disagreement"] = None
            row["embedding_disagreement"] = None
            failed += 1
            print("still unparseable -> marked failed")
        cache[ticker][pos] = row
        cache_path.write_text(json.dumps(cache, indent=1, default=str),
                              encoding="utf-8")

    rows = [r for k, v in cache.items()
            if not k.startswith("__") and isinstance(v, list) for r in v]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nrepaired: {repaired}   marked failed: {failed}   skipped: {skipped}")
    print(f"rewrote {csv_path.name} ({len(rows):,} windows)")
    print("\nNOTE: output/results.csv and summary.txt are filing-level aggregates and")
    print("are now stale. The paper's analyses read chunk_level_results.csv directly")
    print("and are unaffected; regenerate the filing-level artifacts before sending")
    print("anything to the advisor.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
