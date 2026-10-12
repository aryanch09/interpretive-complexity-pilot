"""
Interpretive Complexity Pilot Pipeline
=======================================
Entry point. Run:

    python main.py                            # full 30-firm sample
    python main.py --firm-set pilot           # 11-firm initial pilot
    python main.py --firms AAPL,MSFT          # ad-hoc subset for testing
    python main.py --anonymize                # run anonymization test alongside main pipeline
    python main.py --robustness               # run V2 + V3 prompt variants after V1
    python main.py --skip-agents              # reuse cached results, re-run validation only
    python main.py --calibrate --firms AAPL,NGVC  # temperature noise calibration test

CACHE VERSION:
  The pipeline stores intermediate agent results in output/intermediate_results.json.
  Each result is tagged with the CACHE_VERSION from config.py. If you change the
  scoring scale, model, or prompts, increment CACHE_VERSION in config.py and delete
  the intermediate file (or let the pipeline do it automatically with --clear-cache).
  Mixing cache generations produces invalid disagreement comparisons.

CURRENT CONFIG:
  Model:   openai/gpt-oss-20b (single model via config.PROVIDERS, persona-only disagreement)
  Scale:   0-100 integer (parsed to 0.0-10.0 by dividing by 10)
  Cutoff:  2024-07-01 (filings after this date only; see config.TRAINING_CUTOFF)
  Words:   no cap — MD&A and Risk Factors complete, scored in 1,200-word windows
"""

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path

import pandas as pd

from config import Config
from firms import FIRMS, PILOT_FIRMS, FULL_PILOT_FIRMS, resolve_ciks  # noqa: F401
from data.edgar import fetch_firm_filings
from agents.runner import run_all_agents
from agents.prompts import get_agents_for_variant, PROMPT_VARIANTS
from analysis.disagreement import (
    MIN_USABLE_CHUNKS,
    aggregate_chunk_disagreement,
    compute_disagreement,
)
from analysis.chunking import build_filing_chunks
from analysis.anonymize import anonymize_text
from analysis.textual_baselines import enrich_results_with_text_features, compute_textual_features
from analysis.validation import (
    get_complexity_proxies,
    run_validation,
    create_plots,
    build_analyst_coverage_table,
    compute_adjusted_disagreement,
    validate_trading_volume,
)


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(log_file: Path) -> logging.Logger:
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)
    return logging.getLogger("main")


# ── Intermediate result persistence with cache versioning ─────────────────────

def load_intermediate(path: Path, expected_version: str) -> dict:
    """
    Load cached intermediate results. Returns {} if the file does not exist
    or if its cache_version tag does not match expected_version (stale cache).
    """
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    stored_version = data.get("__cache_version__")
    if stored_version != expected_version:
        logging.getLogger("main").warning(
            "Cache version mismatch: stored=%s expected=%s — ignoring stale cache. "
            "Delete %s to suppress this warning, or use --clear-cache.",
            stored_version, expected_version, path,
        )
        return {}
    return {k: v for k, v in data.items() if not k.startswith("__")}


def save_intermediate(results: dict, path: Path, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"__cache_version__": version}
    payload.update(results)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def load_chunk_cache(path: Path, expected_version: str) -> dict:
    """
    Chunk-level rows keyed by ticker, version-checked like the filing-level cache.

    Kept in its own file rather than inside intermediate_results.json so that the
    filing-level cache format stays unchanged.  This matters because the run is
    executed in nightly batches against a daily token cap: a firm's chunk rows
    must survive across runs, or chunk_level_results.csv would only ever contain
    the firms scored on the final night.
    """
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    if data.get("__cache_version__") != expected_version:
        return {}
    return {k: v for k, v in data.items() if not k.startswith("__")}


def save_chunk_cache(cache: dict, path: Path, version: str) -> None:
    payload = {"__cache_version__": version}
    payload.update(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=str)


# ── Single-firm processing ────────────────────────────────────────────────────

def process_firm(
    firm: dict,
    config: Config,
    logger: logging.Logger,
    anonymize: bool = False,
    agents_override: dict | None = None,
) -> list[dict]:
    """
    Fetch filings, run agents, compute disagreement for one firm.
    Returns a list of row dicts (one per filing processed).
    """
    ticker = firm["ticker"]
    cik = firm["cik"]

    if not cik:
        raise ValueError(f"No CIK resolved for {ticker}")

    from datetime import date
    cutoff = date.fromisoformat(config.TRAINING_CUTOFF)

    filings = fetch_firm_filings(
        cik=cik,
        ticker=ticker,
        filings_dir=config.FILINGS_DIR,
        processed_dir=config.PROCESSED_DIR,
        user_agent=config.EDGAR_USER_AGENT,
        filing_cutoff=config.FILING_CUTOFF,
        max_words=config.MAX_WORDS_PER_FILING,
        max_quarterly=config.MAX_FILINGS_PER_TYPE,
    )

    if not filings:
        raise ValueError(f"No post-cutoff filings found for {ticker}")

    rows: list[dict] = []
    all_chunk_rows: list[dict] = []

    for filing in filings:
        filing_text = filing["text"]
        form_type = filing["form"]
        filing_date = filing["filing_date"]
        extraction_short = filing.get("extraction_short", False)

        # Hard check: skip any filing dated on or before training cutoff
        try:
            fd = date.fromisoformat(str(filing_date)[:10])
            if fd <= cutoff:
                logger.warning(
                    "Skipping %s %s %s — on or before training cutoff %s",
                    ticker, form_type, filing_date, config.TRAINING_CUTOFF,
                )
                continue
        except (ValueError, TypeError):
            logger.warning("Could not parse filing date '%s' for %s; skipping", filing_date, ticker)
            continue

        if extraction_short:
            logger.warning(
                "  %s %s %s: extraction flagged — MD&A could not be recovered; "
                "this observation is excluded from tier means.",
                ticker, form_type, filing_date,
            )

        logger.info("  Processing %s %s %s", ticker, form_type, filing_date)

        # Optional anonymization (familiarity alternative explanation test)
        anon_text: str | None = None
        anon_replacements: list[str] = []
        if anonymize:
            anon_text, anon_replacements = anonymize_text(filing_text, ticker)
            logger.info(
                "  Anonymized %s: %d replacement(s) made",
                ticker, len(anon_replacements),
            )

        # ── Chunk-level scoring (Round 8) ─────────────────────────────────────
        # Every persona reads every chunk of the complete MD&A and Risk Factors.
        # Disagreement is computed per chunk and averaged; see analysis/chunking.py
        # for why uniformity is enforced here rather than by truncating the filing.
        emb_cache = config.OUTPUT_DIR / "embedding_cache.json"

        chunks = build_filing_chunks(
            sections=filing.get("sections", {}),
            form_type=form_type,
            filing_date=filing_date,
            chunk_words=config.CHUNK_WORDS,
            min_tail_words=config.MIN_TAIL_WORDS,
            sections_to_analyze=tuple(config.SECTIONS_TO_ANALYZE),
        )
        if not chunks:
            logger.error(
                "  %s %s %s: no analyzable chunks — skipping filing",
                ticker, form_type, filing_date,
            )
            continue

        logger.info(
            "  %s %s %s: %d chunk(s) × 3 personas = %d call(s)",
            ticker, form_type, filing_date, len(chunks), len(chunks) * 3,
        )

        chunk_rows: list[dict] = []
        last_outputs: dict | None = None
        for ch in chunks:
            ch_outputs = run_all_agents(
                ch["text"], config, agents_override=agents_override
            )
            ch_dis = compute_disagreement(ch_outputs, config, cache_path=emb_cache)
            roles = ["value", "sentiment", "fundamentals"]
            chunk_rows.append({
                "ticker": ticker,
                "tier": firm["tier"],
                "filing_type": form_type,
                "filing_date": filing_date,
                "section": ch["section"],
                "chunk_index": ch["chunk_index"],
                "n_in_section": ch["n_in_section"],
                "word_count": ch["word_count"],
                "numeric_disagreement": ch_dis["numeric_disagreement"],
                "embedding_disagreement": ch_dis["embedding_disagreement"],
                "value_score": ch_outputs["value"]["score"],
                "sentiment_score": ch_outputs["sentiment"]["score"],
                "fundamentals_score": ch_outputs["fundamentals"]["score"],
                "any_failed": any(ch_outputs[r].get("failed", False) for r in roles),
                "any_parse_error": any(
                    ch_outputs[r].get("parse_error", False) for r in roles
                ),
                # Which inference backend served this chunk's three calls.  With
                # OpenRouter pinned to one backend this should be constant across
                # the whole run; it is recorded so that can be checked rather than
                # assumed (see the served-by audit after the firm loop).
                "served_by": "|".join(sorted({
                    str(ch_outputs[r].get("served_by")) for r in roles
                })),
                # Persona reasoning text, persisted per window.  Otherwise it
                # exists only inside the embedding call: the embedding cache is
                # hash->vector and stores no text, so the only reasoning that
                # survives is the last non-failed window's "representative"
                # columns in results.csv.  Keeping it here lets the embedding
                # measure be recomputed with a different encoder, and a noise
                # floor computed for it, with no new API calls.
                "value_reasoning": ch_outputs["value"]["reasoning"],
                "sentiment_reasoning": ch_outputs["sentiment"]["reasoning"],
                "fundamentals_reasoning": ch_outputs["fundamentals"]["reasoning"],
            })
            if not chunk_rows[-1]["any_failed"]:
                last_outputs = ch_outputs

        agg = aggregate_chunk_disagreement(chunk_rows)
        all_chunk_rows.extend(chunk_rows)

        # Representative persona outputs, kept so results.csv still carries readable
        # reasoning text.  These are one chunk's outputs, not the filing-level
        # measure — the measure is the chunk mean in `agg`.
        agent_outputs = last_outputs or {
            r: {"score": None, "confidence": None, "reasoning": "",
                "key_factors": "", "parse_error": True, "failed": True}
            for r in ["value", "sentiment", "fundamentals"]
        }
        disagree = {
            "numeric_disagreement": agg["numeric_disagreement"],
            "embedding_disagreement": agg["embedding_disagreement"],
        }

        # Textual features on agent-facing text (inline, for new runs)
        text_features = compute_textual_features(filing_text)

        row: dict = {
            "ticker": ticker,
            "company_name": firm["name"],
            "tier": firm["tier"],
            "filing_type": form_type,
            "filing_date": filing_date,
            "value_score": agent_outputs["value"]["score"],
            "sentiment_score": agent_outputs["sentiment"]["score"],
            "fundamentals_score": agent_outputs["fundamentals"]["score"],
            "value_confidence": agent_outputs["value"].get("confidence"),
            "sentiment_confidence": agent_outputs["sentiment"].get("confidence"),
            "fundamentals_confidence": agent_outputs["fundamentals"].get("confidence"),
            "numeric_disagreement": disagree["numeric_disagreement"],
            "embedding_disagreement": disagree["embedding_disagreement"],
            "value_reasoning": agent_outputs["value"].get("reasoning", ""),
            "sentiment_reasoning": agent_outputs["sentiment"].get("reasoning", ""),
            "fundamentals_reasoning": agent_outputs["fundamentals"].get("reasoning", ""),
            "value_key_factors": agent_outputs["value"].get("key_factors", ""),
            "sentiment_key_factors": agent_outputs["sentiment"].get("key_factors", ""),
            "fundamentals_key_factors": agent_outputs["fundamentals"].get("key_factors", ""),
            "any_parse_error": any(
                agent_outputs[r].get("parse_error", False)
                for r in ["value", "sentiment", "fundamentals"]
            ),
            # Filing is unusable if too few of its chunks produced 3 valid personas.
            "any_failed": agg["any_failed"],
            "partial_failure": (
                agg["n_chunks_used"] < agg["n_chunks_total"]
                and agg["n_chunks_used"] > 0
            ),
            "extraction_short": extraction_short,
            # ── Chunk-level design columns (Round 8) ───────────────────────────
            "n_chunks_total": agg["n_chunks_total"],
            "n_chunks_used": agg["n_chunks_used"],
            "chunk_coverage": agg["chunk_coverage"],
            "within_filing_sd_embedding": agg["within_filing_sd_embedding"],
            "within_filing_sd_numeric": agg["within_filing_sd_numeric"],
            "mda_embedding_disagreement": agg["mda_embedding_disagreement"],
            "mda_numeric_disagreement": agg["mda_numeric_disagreement"],
            "mda_n_chunks": agg["mda_n_chunks"],
            "risk_embedding_disagreement": agg["risk_embedding_disagreement"],
            "risk_numeric_disagreement": agg["risk_numeric_disagreement"],
            "risk_n_chunks": agg["risk_n_chunks"],
            # Complete-section word count: the true length of what was read, and
            # the length control the advisor asked to have reported.
            # section_words_total counts ONLY the scored sections (MD&A + Risk).
            # filing["word_count_raw"] also includes market_risk, which is
            # extracted and cached but never scored, so it would be the wrong
            # length control.  mda_words is the variable the pre-registered
            # numeric-retirement rule is defined on.
            "section_words_total": (filing.get("mda_words") or 0) + (filing.get("risk_words") or 0),
            "mda_words": filing.get("mda_words"),
            "risk_words": filing.get("risk_words"),
            # Inline textual features
            "word_count": text_features["word_count"],
            "lm_neg_pct": text_features["lm_neg_pct"],
            "avg_sent_len": text_features["avg_sent_len"],
            "type_token_ratio": text_features["type_token_ratio"],
            "fog_index": text_features["fog_index"],
        }

        # If anonymize: run agents on scrubbed text and record delta
        if anonymize and anon_text:
            anon_outputs = run_all_agents(anon_text, config, agents_override=agents_override)
            anon_disagree = compute_disagreement(anon_outputs, config, cache_path=emb_cache)
            row["anon_numeric_disagreement"] = anon_disagree["numeric_disagreement"]
            row["anon_embedding_disagreement"] = anon_disagree["embedding_disagreement"]
            row["anon_replacements_count"] = len(anon_replacements)
            row["familiarity_flag"] = (
                "high_familiarity" if firm["tier"] == "large" else
                "mid_familiarity" if firm["tier"] == "mid" else
                "low_familiarity"
            )
            # Delta: positive means agreement INCREASED after anonymization
            # (i.e. the name was contributing to disagreement, not familiarity)
            # Negative means anonymization REDUCED agreement → familiarity was suppressing disagreement
            raw_num = disagree["numeric_disagreement"]
            anon_num = anon_disagree["numeric_disagreement"]
            row["anon_delta_numeric"] = (
                round(anon_num - raw_num, 4) if raw_num is not None and anon_num is not None else None
            )
            raw_emb = disagree["embedding_disagreement"]
            anon_emb = anon_disagree["embedding_disagreement"]
            row["anon_delta_embedding"] = (
                round(anon_emb - raw_emb, 4) if raw_emb is not None and anon_emb is not None else None
            )

        rows.append(row)

    return rows, all_chunk_rows


# ── Output writers ────────────────────────────────────────────────────────────

def write_results_csv(all_rows: list[dict], path: Path) -> pd.DataFrame:
    df = pd.DataFrame(all_rows)
    df.to_csv(path, index=False)
    return df


def write_validation_csv(validation: dict, path: Path) -> None:
    validation["merged"].to_csv(path, index=False)


def write_summary(
    results_df: pd.DataFrame,
    validation: dict,
    failed_firms: list[str],
    path: Path,
    prompt_variant: str = "V1",
    vol_df=None,
    n_excluded_rows: int = 0,
    n_total_rows: int = 0,
    n_short_excluded: int = 0,
    config: Config | None = None,
) -> None:
    from config import Config as _Config
    # Read the LIVE config instance where the value can differ from the class
    # attribute: apply_provider() mutates MODEL_NAME on the instance, so under
    # --provider cerebras the class attribute names the wrong model.
    cfg = config if config is not None else _Config()
    lines: list[str] = []
    lines.append("Interpretive Complexity Pilot — Results Memo")
    lines.append("=" * 60)
    lines.append("")

    n_firms = results_df["ticker"].nunique()
    n_filings = len(results_df)
    n_parse_errors = int(results_df["any_parse_error"].sum()) if "any_parse_error" in results_df.columns else 0
    n_short = int(results_df["extraction_short"].sum()) if "extraction_short" in results_df.columns else 0
    total_rows = n_total_rows if n_total_rows else n_filings

    lines.append(
        f"Sample: {n_firms} firms, {n_filings} filing-observations in analysis sample "
        f"({total_rows} total rows collected; see results.csv)."
    )
    lines.append(
        f"Design: prompt ensemble — {cfg.MODEL_NAME}, three analytical persona prompts "
        f"(value / sentiment / fundamentals), no inter-persona interaction. "
        f"Disagreement reflects analytical perspective, not model architecture."
    )
    lines.append(
        f"Scoring: 0-100 integer → 0.0–10.0 after /10. "
        f"Filings after {_Config.TRAINING_CUTOFF} only."
    )
    if prompt_variant != "V1":
        lines.append(f"Prompt variant: {prompt_variant}.")
    if failed_firms:
        lines.append(f"Firms excluded due to complete API failure: {', '.join(failed_firms)}.")
    if n_excluded_rows:
        lines.append(
            f"Rows excluded from analysis: {n_excluded_rows} (any_failed=True — partial or complete "
            "persona-call failure). A 2-agent distance is not comparable to the 3-agent mean pairwise "
            "distance; failed agents also store error strings in the reasoning field that corrupt "
            "embedding computation. These rows are retained in results.csv for documentation but "
            "contribute no data to tier statistics, Spearman correlations, or adjusted disagreement."
        )
    if n_parse_errors:
        lines.append(f"Rows with ≥1 malformed persona response: {n_parse_errors} (any_parse_error=True).")
    if n_short:
        lines.append(
            f"Rows with extraction_short=True: {n_short} (MD&A could not be recovered, or the "
            f"filing yields fewer than {MIN_USABLE_CHUNKS} scoring windows). "
            "These rows are excluded from tier means but included in correlation tests. "
            "Excluded rows are retained in results_clean.csv for documentation."
        )
    if n_short_excluded:
        lines.append(
            f"  Rows excluded from tier means due to extraction_short: {n_short_excluded}. "
            "Tier means below reflect only rows with extraction_short=False."
        )
    lines.append("")

    # ── Table 1: Descriptive statistics ───────────────────────────────────────
    if "descriptive" in validation:
        desc_df = validation["descriptive"]
        lines.append("Table 1: Descriptive Statistics (firm-level means)")
        lines.append("")
        lines.append(f"{'Variable':<32} {'n':>4} {'mean':>8} {'std':>8} {'min':>8} {'median':>8} {'max':>8}")
        lines.append("-" * 80)
        for _, r in desc_df.iterrows():
            lines.append(
                f"  {r['variable']:<30} {int(r['n']):>4} {r['mean']:>8.4f} {r['std']:>8.4f} "
                f"{r['min']:>8.4f} {r['median']:>8.4f} {r['max']:>8.4f}"
            )
        lines.append("")

    # ── Disagreement by tier ───────────────────────────────────────────────────
    tier_df = validation["tier_stats"]
    lines.append("Disagreement by tier")
    lines.append("")

    # Check if numeric_disagreement is low-resolution in this dataset
    merged = validation["merged"]
    num_n_distinct = merged["numeric_disagreement"].nunique() if "numeric_disagreement" in merged.columns else 0
    if num_n_distinct <= 5:
        lines.append(
            f"  NOTE: numeric_disagreement takes only {num_n_distinct} distinct value(s) at the "
            "firm level. Spearman p-values and within-tier std are unreliable when most ranks "
            "are tied. This is an artifact of the old integer scoring scale; re-run agents "
            "with the 0-100 scale to obtain continuous disagreement values."
        )
        lines.append("")

    header = f"{'':22s}  {'Numeric disagreement':26s}  {'Embedding disagreement':26s}"
    lines.append(header)
    sub_hdr = f"{'':22s}  {'mean':>8s}  {'std':>8s}  {'distinct':>8s}      {'mean':>8s}  {'std':>8s}"
    lines.append(sub_hdr)
    for _, row in tier_df.iterrows():
        label = f"{row['tier'].capitalize()}-cap (n={int(row['n_firms'])})"
        nd = int(row.get("numeric_disagreement_n_distinct", 0))
        lines.append(
            f"{label:<22}  {row['numeric_disagreement_mean']:>8.4f}  "
            f"{row['numeric_disagreement_std']:>8.4f}  {nd:>8d}      "
            f"{row['embedding_disagreement_mean']:>8.4f}  "
            f"{row['embedding_disagreement_std']:>8.4f}"
        )
    lines.append("")

    # ── Flat numeric finding (report first, not buried) ────────────────────
    num_vals = [
        (r["tier"], r["numeric_disagreement_mean"])
        for _, r in tier_df.iterrows()
        if not pd.isna(r.get("numeric_disagreement_mean"))
    ]
    if len(num_vals) == 3:
        tiers_ordered = {t: v for t, v in num_vals}
        is_monotone = (
            tiers_ordered.get("large", float("inf")) <
            tiers_ordered.get("mid", float("inf")) <
            tiers_ordered.get("small", float("-inf"))
        )
        if not is_monotone:
            lines.append(
                "FINDING — NUMERIC MEASURE: numeric disagreement is flat and non-monotone across tiers "
                f"(Large={tiers_ordered.get('large', float('nan')):.3f}, "
                f"Mid={tiers_ordered.get('mid', float('nan')):.3f}, "
                f"Small={tiers_ordered.get('small', float('nan')):.3f}). "
                "The numeric measure does not separate tiers. This is a finding, not a failure: "
                f"at n={n_firms} firms, the hypothesis rests entirely on the embedding measure. "
                "Report this result before the embedding gradient in any advisor communication."
            )
            lines.append("")

    # ── Spearman correlations ──────────────────────────────────────────────────
    corr_df = validation["correlations"]
    bonf_alpha = validation.get("bonferroni_alpha", 0.05)
    n_tests = validation.get("n_total_tests", 1)
    # Largest n actually used in the Spearman tests (the proxy-data subset).
    n_corr = int(corr_df["n"].max()) if "n" in corr_df.columns and not corr_df.empty else n_firms

    lines.append("Spearman correlations with complexity proxies")
    lines.append("")
    lines.append(
        f"Bonferroni correction: α = 0.05 / {n_tests} tests = {bonf_alpha:.5f}. "
        "CI = bootstrap 95% confidence interval (1,000 resamples)."
    )
    lines.append(
        "Significance stars (*) are reported only for embedding_disagreement. "
        "Significance levels are not reported for numeric_disagreement because "
        f"its distributional properties at n={n_corr} do not support standard significance "
        "thresholds — the measure may carry heavy ties even after continuous rescoring. "
        "The embedding measure is the primary test of the hypothesis."
    )
    lines.append("")

    hdr = f"{'Disagreement':<32}  {'Proxy':<28}  {'n':>4}  {'rho':>6}  {'[95% CI]':>18}  {'p':>8}  {'sig':>5}  {'note'}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    raw_rows = corr_df[~corr_df["adjusted"]]
    for _, row in raw_rows.iterrows():
        low_res_flag = "[LOW_RES]" if row.get("low_resolution") else ""
        is_numeric = "numeric_disagreement" in str(row.get("disagreement", ""))
        # Never star numeric_disagreement regardless of p-value
        sig = "*" if (row.get("bonferroni_sig") and not row.get("low_resolution") and not is_numeric) else ""
        ci_lo = row.get("ci_lower_95", float("nan"))
        ci_hi = row.get("ci_upper_95", float("nan"))
        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if not (pd.isna(ci_lo) or pd.isna(ci_hi)) else "[n/a, n/a]"
        rho = row.get("spearman_rho", float("nan"))
        pval = row.get("p_value", float("nan"))
        rho_str = f"{rho:>6.3f}" if not pd.isna(rho) else "   nan"
        pval_str = f"{pval:>8.4f}" if not pd.isna(pval) else "     nan"
        lines.append(
            f"  {str(row['disagreement']):<30}  {str(row['proxy']):<28}  {int(row['n']):>4}  "
            f"{rho_str}  {ci_str:>18}  {pval_str}  "
            f"{sig:>5}  {low_res_flag}"
        )
    lines.append("")
    # Analyst count note: pull actual computed ρ to avoid hardcoded stale values
    _ac_row = raw_rows[
        (raw_rows["disagreement"] == "embedding_disagreement") &
        (raw_rows["proxy"] == "analyst_count")
    ]
    if not _ac_row.empty:
        _ac_rho = _ac_row.iloc[0].get("spearman_rho", float("nan"))
        _ac_p = _ac_row.iloc[0].get("p_value", float("nan"))
        _rho_str = f"{_ac_rho:.3f}" if not pd.isna(_ac_rho) else "n/a"
        _p_str = f"{_ac_p:.2f}" if not pd.isna(_ac_p) else "n/a"
        lines.append(
            f"Note on analyst_count: the embedding_disagreement × analyst_count correlation "
            f"(ρ = {_rho_str}, p = {_p_str}) is consistent with zero — not directional support for "
            "the hypothesis. Three of four proxies (market_cap, idiosyncratic_volatility, "
            "firm_age_years) are directionally consistent for embedding_disagreement; "
            "analyst_count is flat at this sample size."
        )
        lines.append("")

    # ── Textual baseline controls ──────────────────────────────────────────────
    text_cols = ["word_count", "lm_neg_pct"]
    has_text = all(c in results_df.columns for c in text_cols)

    if has_text:
        lines.append("Textual baseline controls (Loughran & McDonald 2014; Bushee et al. 2018)")
        lines.append("")
        mean_wc = results_df["word_count"].mean()
        median_wc = results_df["word_count"].median()
        mean_neg = results_df["lm_neg_pct"].mean() * 100
        mean_fog = results_df["fog_index"].mean() if "fog_index" in results_df.columns else None
        lines.append(
            f"Mean word count: {mean_wc:.0f} (median: {median_wc:.0f}).  "
            f"extraction_short rule: MD&A not recovered, or fewer than {MIN_USABLE_CHUNKS} "
            "scoring windows — no word-count threshold is applied.  "
            f"Mean L-M neg %: {mean_neg:.2f}% (curated 280-word subset of ~2,355-word L-M list)."
        )
        if mean_fog:
            lines.append(f"Mean Gunning Fog Index: {mean_fog:.1f}.")

        # Word count vs. disagreement correlations (K Round 6: "word count is a control from here on")
        if "correlations" in validation:
            corr_df_all = validation["correlations"]
            wc_rows = corr_df_all[
                (corr_df_all["proxy"] == "word_count") &
                (~corr_df_all["adjusted"])
            ]
            if not wc_rows.empty:
                lines.append("")
                lines.append("Word count vs. disagreement (length-artifact control):")
                for _, wr in wc_rows.iterrows():
                    rho = wr.get("spearman_rho", float("nan"))
                    pval = wr.get("p_value", float("nan"))
                    n_wc = int(wr.get("n", 0))
                    lines.append(
                        f"  {str(wr['disagreement']):<32} word_count ρ={rho:+.3f}, p={pval:.3f}, n={n_wc}"
                    )
        lines.append(
            "NOTE: lm_neg_pct uses a curated 280-word subset of the full L-M negative dictionary. "
            "Results should be replicated with the complete L-M master dictionary "
            "(sraf.nd.edu/loughranmcdonald/resources/) before publication."
        )

        if "adjusted_numeric_disagreement" in results_df.columns:
            lines.append("")
            lines.append("Raw vs. adjusted Spearman correlations (text controls partialed out):")
            lines.append("")
            adj_rows = corr_df[corr_df["adjusted"]]
            lines.append(
                f"{'Measure':<38}  {'Proxy':<28}  {'raw ρ':>7}  {'adj ρ':>7}  {'CI (adj)':>18}"
            )
            lines.append("-" * 100)
            for px in ["market_cap", "analyst_count", "idiosyncratic_volatility"]:
                for dm in ["numeric_disagreement", "embedding_disagreement"]:
                    raw_r = raw_rows[(raw_rows["disagreement"] == dm) & (raw_rows["proxy"] == px)]
                    adj_r = adj_rows[(adj_rows["disagreement"] == f"adjusted_{dm}") & (adj_rows["proxy"] == px)]
                    if raw_r.empty:
                        continue
                    raw_rho = f"{raw_r.iloc[0]['spearman_rho']:.3f}"
                    adj_rho = f"{adj_r.iloc[0]['spearman_rho']:.3f}" if not adj_r.empty else "n/a"
                    ci_str = (
                        f"[{adj_r.iloc[0]['ci_lower_95']:.3f}, {adj_r.iloc[0]['ci_upper_95']:.3f}]"
                        if not adj_r.empty else ""
                    )
                    lines.append(
                        f"  {dm:<36}  {px:<28}  {raw_rho:>7}  {adj_rho:>7}  {ci_str:>18}"
                    )
        lines.append("")

    # ── Trading volume validation ──────────────────────────────────────────────
    if vol_df is not None and len(vol_df) > 0:
        lines.append("Post-filing trading volume validation (Bhagwat et al. 2025)")
        lines.append("")
        mean_abvol = vol_df["abnormal_volume"].mean()
        med_abvol = vol_df["abnormal_volume"].median()
        lines.append(
            f"Observations with volume data: {len(vol_df)}. "
            f"Mean abnormal volume: {mean_abvol:+.3f}. Median: {med_abvol:+.3f}."
        )
        lines.append(
            "Caution: this association is consistent with the familiarity-confound story "
            "as well as the interpretive-complexity story. A significant β here does NOT "
            "establish that disagreement measures filing ambiguity rather than filing prominence."
        )
        lines.append("")

    # ── Design note ────────────────────────────────────────────────────────────
    lines.append("Design")
    lines.append("-" * 40)
    # Read the LIVE config: apply_provider() mutates the instance, not the class,
    # so a class-level read would name the wrong model/provider in a file that
    # goes to the advisor.
    lines.append(cfg.DESIGN_DESCRIPTION)
    lines.append("")

    # ── Limitations ────────────────────────────────────────────────────────────
    lines.append("Limitations and open questions")
    lines.append("-" * 40)
    lines.append(
        "1. PROMPT-ENSEMBLE DESIGN: all three analytical personas use the same model "
        f"({cfg.MODEL_NAME}). Cross-persona disagreement reflects analytical framing "
        "differences, not model-architecture differences. This is intentional — per advisor "
        "guidance (June 2026), it isolates the effect of analytical perspective."
    )
    lines.append(
        "2. FAMILIARITY CONFOUND: the model may have stronger priors for large-cap firms "
        "(AAPL, MSFT, JPM) from pre-training, which could suppress persona disagreement "
        "regardless of filing content. The anonymization test (output/anonymization_test.csv) "
        f"provides anonymization_delta as a firm-level control variable at n={n_firms} firms, "
        "not a pass/fail test."
    )
    lines.append(
        f"3. SAMPLE SIZE: n={n_firms} firms, {n_filings} filing-observations in this run. "
        "Proxy correlations are computed on the subset with available CRSP/Compustat data. "
        "All tier gradient results should be treated as measure-validation evidence, not the "
        "definitive return-test sample (which requires many more firms and time-series depth)."
    )
    lines.append(
        "4. EMBEDDING MEASURE: neural embeddings (all-MiniLM-L6-v2) are new to this run. "
        "Interpret with caution until validated on a larger sample. If embedding_disagreement "
        f"shows no variation at n={n_firms} firms, investigate the embedding pipeline."
    )
    lines.append(
        "5. PROXY ASYNCHRONY: market_cap, analyst_count, and idiosyncratic_volatility are "
        "measured as of today, not at the filing date. For volatile small-caps this can differ "
        "substantially from the filing-date value."
    )
    lines.append(
        "6. MULTIPLE TESTING: Bonferroni correction applied with α=0.05/n_tests across all "
        "Spearman tests. No correction applied to the trading volume regression (separate "
        "pre-specified test)."
    )
    lines.append(
        "7. CAUSAL IDENTIFICATION: cross-sectional Spearman ρ does not establish direction of "
        "causality. Higher LLM disagreement → lower market cap is equally consistent with "
        "disagreement being a cause (complexity discount) or an effect (small firms are less "
        "covered and more ambiguous)."
    )
    lines.append("")
    lines.append("Raw data: output/results.csv, output/validation.csv")
    lines.append("Plots: output/plots/")

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    print(text)


def write_anonymization_csv(all_rows: list[dict], path: Path) -> None:
    anon_cols = [
        "ticker", "tier", "filing_type", "filing_date",
        "numeric_disagreement", "anon_numeric_disagreement", "anon_delta_numeric",
        "embedding_disagreement", "anon_embedding_disagreement", "anon_delta_embedding",
        "anon_replacements_count", "familiarity_flag",
    ]
    available = [c for c in anon_cols if c in (all_rows[0] if all_rows else {})]
    if not available:
        return
    df = pd.DataFrame(all_rows)[available]
    if not df.empty:
        df.to_csv(path, index=False)


def write_robustness_csv(robustness_rows: list[dict], path: Path) -> None:
    if not robustness_rows:
        return
    pd.DataFrame(robustness_rows).to_csv(path, index=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpretive Complexity Pilot Pipeline")
    parser.add_argument("--firms", type=str, default=None,
                        help="Comma-separated tickers (e.g. AAPL,MSFT). Overrides --firm-set.")
    parser.add_argument("--firm-set", choices=["pilot", "full"], default="full",
                        help=f"'pilot' = {len(PILOT_FIRMS)}-firm sample; "
                             f"'full' = all {len(FULL_PILOT_FIRMS)} firms (default).")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["groq", "openrouter", "cerebras", "ollama"],
                        help="Inference provider (default: config.API_PROVIDER / $API_PROVIDER). "
                             "Changing provider changes the served model — bump CACHE_VERSION.")
    parser.add_argument("--chunk-words", type=int, default=None,
                        help="Scoring window size in words (default: config.CHUNK_WORDS). "
                             "Results are cached separately per window size.")
    parser.add_argument("--agent-sleep", type=float, default=None,
                        help="Seconds between persona calls (default: config.AGENT_SLEEP_SECONDS). "
                             "Use ~0.5 on the Groq Developer tier; 30 on the free tier.")
    parser.add_argument("--skip-agents", action="store_true",
                        help="Skip agent API calls; reuse existing intermediate results.")
    parser.add_argument("--skip-proxies", action="store_true",
                        help="Skip yfinance proxy collection; reuse proxy_cache.csv.")
    parser.add_argument("--anonymize", action="store_true",
                        help=(
                            "Run familiarity test: scrub company identifiers, re-run agents, "
                            "compare disagreement. Outputs anonymization_test.csv."
                        ))
    parser.add_argument("--robustness", action="store_true",
                        help="After V1 run, re-run with V2 and V3 prompt variants.")
    parser.add_argument("--returns", action="store_true",
                        help="(Stub) Future return-prediction pipeline. Not yet implemented.")
    parser.add_argument("--trading-volume", action="store_true",
                        help="Compute post-filing abnormal volume. Outputs trading_volume_validation.csv.")
    parser.add_argument("--calibrate", action="store_true",
                        help=(
                            "Run temperature noise calibration on selected firms: repeat same "
                            "agent prompt 5x, measure intra-agent variance vs. inter-agent variance."
                        ))
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete intermediate_results.json before starting (forces fresh agent runs).")
    parser.add_argument("--dry-run", action="store_true",
                        help=(
                            "Print the list of firms that would be processed and exit. "
                            "No EDGAR fetches, no API calls, no files written. "
                            "Use to verify --firm-set full before committing to the run."
                        ))
    return parser.parse_args()


def _build_firm_list(args: argparse.Namespace, resolved_all: list[dict]) -> list[dict]:
    if args.firms:
        requested = {t.strip().upper() for t in args.firms.split(",")}
        subset = [f for f in resolved_all if f["ticker"] in requested]
        if not subset:
            print(f"ERROR: None of the requested tickers matched the firm list: {args.firms}")
            sys.exit(1)
        return subset
    if args.firm_set == "pilot":
        return [f for f in resolved_all if f["ticker"] in PILOT_FIRMS]
    if args.firm_set == "full":
        return [f for f in resolved_all if f["ticker"] in FULL_PILOT_FIRMS]
    return resolved_all


def main() -> None:
    args = parse_args()
    config = Config()
    if args.provider:
        config.API_PROVIDER = args.provider
    config.apply_provider()
    if args.chunk_words:
        config.CHUNK_WORDS = args.chunk_words
        config.MIN_TAIL_WORDS = max(400, args.chunk_words // 3)
    if args.agent_sleep is not None:
        config.AGENT_SLEEP_SECONDS = args.agent_sleep
    config.ensure_dirs()

    logger = setup_logging(config.LOG_FILE)
    logger.info(
        "Pipeline starting — model=%s cutoff=%s scale=0-%d cache_version=%s",
        config.MODEL_NAME, config.TRAINING_CUTOFF,
        config.SCORING_SCALE, config.effective_cache_version,
    )
    logger.info("Design: %s", config.DESIGN_DESCRIPTION)

    if not args.skip_agents:
        config.validate()

    if args.returns:
        logger.warning("--returns: return prediction pipeline not yet implemented (see analysis/returns.py).")

    # ── 1. Clear cache if requested ───────────────────────────────────────────
    if args.clear_cache and config.INTERMEDIATE_FILE.exists():
        config.INTERMEDIATE_FILE.unlink()
        logger.info("Cleared intermediate cache: %s", config.INTERMEDIATE_FILE)

    # ── 2. Resolve firm list ──────────────────────────────────────────────────
    resolved_all = resolve_ciks(FIRMS, cache_file=config.CIK_CACHE_FILE)
    resolved_firms = _build_firm_list(args, resolved_all)
    logger.info("Firm set: %s (%d firms)", args.firm_set if not args.firms else "custom", len(resolved_firms))

    # ── 2a. Dry-run: print firm list and exit ─────────────────────────────────
    if args.dry_run:
        print(f"\nDRY RUN — {len(resolved_firms)} firms would be processed:")
        for firm in resolved_firms:
            cik = firm.get("cik") or "CIK-MISSING"
            print(f"  {firm['ticker']:<8} {firm['tier']:<8} CIK={cik}  {firm['name']}")
        print("\nNo API calls made. Remove --dry-run to execute.")
        return

    # ── 3. Load intermediate results (with cache version check) ───────────────
    intermediate = load_intermediate(config.INTERMEDIATE_FILE, config.effective_cache_version)
    chunk_cache_path = config.OUTPUT_DIR / "chunk_cache.json"
    chunk_cache = load_chunk_cache(chunk_cache_path, config.effective_cache_version)
    all_rows: list[dict] = []
    all_chunk_rows: list[dict] = []
    failed_firms: list[str] = []

    for ticker, rows in intermediate.items():
        if isinstance(rows, list) and rows:
            all_rows.extend(rows)
    for ticker, crows in chunk_cache.items():
        if isinstance(crows, list) and crows:
            all_chunk_rows.extend(crows)

    # ── 4. Calibration mode ───────────────────────────────────────────────────
    if args.calibrate:
        from analysis.calibration import run_full_calibration, write_calibration_report
        logger.info("Running temperature noise calibration ...")
        # Use the first firm that has a cached processed file
        calibration_done = False
        for firm in resolved_firms:
            ticker = firm["ticker"]
            # Find a cached text file
            import glob
            pattern = str(config.PROCESSED_DIR / f"{ticker}_*_sections.json")
            matches = glob.glob(pattern)
            if not matches:
                continue
            # NOTE: this still calibrates on the whole concatenated sections, not on a
            # single CHUNK_WORDS scoring window as the pipeline now scores.
            _cal_sections = json.loads(Path(matches[0]).read_text(encoding="utf-8"))
            cal_text = "\n\n".join(
                _cal_sections[k] for k in config.SECTIONS_TO_ANALYZE if k in _cal_sections
            )
            logger.info("Calibrating on %s (%s)", ticker, matches[0])
            cal_results = run_full_calibration(cal_text, config, n_runs=5)
            # Compare to inter-agent std from results if available
            inter_std = None
            if ticker in intermediate:
                for row in intermediate[ticker]:
                    v = row.get("numeric_disagreement")
                    if v is not None:
                        inter_std = float(v)
                        break
            write_calibration_report(
                cal_results, inter_std, ticker,
                config.OUTPUT_DIR / "calibration_report.txt",
            )
            calibration_done = True
            break
        if not calibration_done:
            logger.warning("Calibration: no cached processed files found. Run pipeline first.")
        if args.skip_agents:
            return

    # ── 5. Process each firm ──────────────────────────────────────────────────
    for firm in resolved_firms:
        ticker = firm["ticker"]

        if ticker in intermediate:
            logger.info("Skipping %s (already in cache)", ticker)
            continue

        if args.skip_agents:
            logger.info("Skipping %s (--skip-agents, no cache)", ticker)
            continue

        logger.info("Processing %s (%s tier) ...", ticker, firm["tier"])
        try:
            rows, chunk_rows = process_firm(firm, config, logger, anonymize=args.anonymize)
            intermediate[ticker] = rows
            all_rows.extend(rows)
            all_chunk_rows.extend(chunk_rows)
            if chunk_rows:
                # Persist incrementally: a long chunk-level run spread over several
                # nightly batches should never lose completed work to a later failure.
                chunk_cache[ticker] = chunk_rows
                save_chunk_cache(chunk_cache, chunk_cache_path, config.effective_cache_version)
                pd.DataFrame(all_chunk_rows).to_csv(
                    config.OUTPUT_DIR / "chunk_level_results.csv", index=False
                )
            save_intermediate(intermediate, config.INTERMEDIATE_FILE, config.effective_cache_version)
            logger.info("  Done: %d filing(s) for %s", len(rows), ticker)
        except Exception as exc:
            logger.error("FAILED %s: %s", ticker, exc)
            logger.debug(traceback.format_exc())
            failed_firms.append(ticker)

    if not all_rows:
        logger.error("No results collected. Check errors above.")
        sys.exit(1)

    # ── Served-backend audit ──────────────────────────────────────────────────
    # The design claim is that the model is held fixed across all persona calls.
    # If more than one backend served the run, that claim needs verifying before
    # the results are used: different backends may serve different quantizations
    # of the same weights, and that difference would show up as disagreement.
    if all_chunk_rows and any("served_by" in r for r in all_chunk_rows):
        # Per-window served_by is a "|"-joined set of the window's three calls, so
        # it must be split before counting: a window genuinely served by two
        # backends would otherwise collapse to one string ("DeepInfra|Together")
        # and, if every window shared it, pass as a single backend.  The failed-call
        # sentinel carries no served_by key, which str() turns into "None" inside
        # the join; that is missing data, not a backend, so it is excluded.
        backends = sorted({
            b
            for r in all_chunk_rows
            for b in str(r.get("served_by") or "").split("|")
            if b and b != "None"
        })
        if len(backends) > 1:
            logger.error(
                "SERVED-BACKEND MISMATCH: %d distinct backends served this run (%s). "
                "The 'model held fixed' claim does not hold as stated. Investigate "
                "before reporting any results.",
                len(backends), ", ".join(backends),
            )
            raise RuntimeError(
                f"Served-backend audit FAILED: {len(backends)} distinct backends "
                f"served this run ({', '.join(backends)}). The design claim is that"
                " the model is held fixed and only the analytical lens varies; a run"
                " served partly by a different quantization makes that claim false."
                " Stopping before summary.txt and per_filing_table.csv are written."
            )
        elif backends:
            logger.info("Served-backend check PASSED — all calls served by %s", backends[0])
        else:
            logger.warning(
                "Served-backend check INCONCLUSIVE — no backend recorded on any window."
            )

    if all_chunk_rows:
        pd.DataFrame(all_chunk_rows).to_csv(
            config.OUTPUT_DIR / "chunk_level_results.csv", index=False
        )
        logger.info("chunk_level_results.csv written (%d chunk rows)", len(all_chunk_rows))

    # ── 6. Write results.csv ──────────────────────────────────────────────────
    results_df = write_results_csv(all_rows, config.OUTPUT_DIR / "results.csv")
    logger.info("results.csv written (%d rows)", len(results_df))

    if args.anonymize and "anon_numeric_disagreement" in results_df.columns:
        write_anonymization_csv(all_rows, config.OUTPUT_DIR / "anonymization_test.csv")
        logger.info("anonymization_test.csv written")

    # ── 6b. Validate embedding variation (catch silent TF-IDF fallback) ──────
    if "embedding_disagreement" in results_df.columns:
        from analysis.embeddings import validate_embedding_variation
        emb_scores = results_df["embedding_disagreement"].dropna().tolist()
        try:
            validate_embedding_variation(emb_scores, threshold=config.EMBEDDING_FLAT_THRESHOLD)
            logger.info("Embedding variation check PASSED (spread=%.4f)", max(emb_scores) - min(emb_scores))
        except RuntimeError as e:
            logger.error("EMBEDDING VARIATION CHECK FAILED: %s", e)
            raise

    # ── 7. Textual baseline controls ──────────────────────────────────────────
    logger.info("Computing textual baseline controls ...")
    results_df = enrich_results_with_text_features(results_df, config.PROCESSED_DIR)

    # ── 7b. Flag extraction_short rows ────────────────────────────────────────
    # The Round 6 rule flagged anything below half the sample median word count.
    # That rule is retired here, because it no longer means what it meant then.
    #
    # Under the old fixed word cap, every filing was truncated to the same length,
    # so a below-median word count could only mean extraction had failed.  Now
    # that MD&A and Risk Factors are extracted complete, word count is genuinely
    # cross-sectional — small firms really do file shorter MD&As, and that is
    # variation the hypothesis is about, not a defect.  Applying a half-median cut
    # to complete sections would systematically discard small-cap filings, i.e.
    # exactly the tier the hypothesis predicts is most interesting.
    #
    # extraction_short now means what its name says: extraction did not deliver a
    # usable section. Two ways that happens:
    #   (a) the parser fell back to title-based extraction and could not separate
    #       MD&A from Risk Factors (flag set in data/edgar.extract_sections_dict);
    #   (b) the filing yielded too few chunks for a stable filing-level mean
    #       (n_chunks_total < MIN_USABLE_CHUNKS) — this catches genuinely broken
    #       extractions such as a mega-cap 10-Q that yields 500 words.
    if "n_chunks_total" in results_df.columns:
        too_few = results_df["n_chunks_total"].fillna(0) < MIN_USABLE_CHUNKS
        results_df["extraction_short"] = (
            too_few | results_df.get("extraction_short", False).fillna(False)
        )
        n_short_total = int(results_df["extraction_short"].sum())
        logger.info(
            "extraction_short: %d row(s) flagged "
            "(extraction fallback used, or fewer than %d chunks).",
            n_short_total, MIN_USABLE_CHUNKS,
        )
        for _, sr in results_df[results_df["extraction_short"]].iterrows():
            logger.warning(
                "  extraction_short: %s %s — %s chunk(s), %s section words",
                sr["ticker"], sr["filing_type"],
                sr.get("n_chunks_total"), sr.get("section_words_total"),
            )

    # ── 7d. Drop failed rows from analysis ───────────────────────────────────
    # Any row where any agent failed produces invalid disagreement measures:
    # a 2-agent distance is not comparable to the 3-agent mean pairwise distance,
    # and failed agents store error strings in reasoning that corrupt embeddings.
    clean_results_df = results_df[~results_df["any_failed"].fillna(False)].copy()
    n_excluded = len(results_df) - len(clean_results_df)
    if n_excluded:
        excluded = (
            results_df[results_df["any_failed"].fillna(False)][["ticker", "filing_type"]]
            .to_dict("records")
        )
        logger.warning(
            "Excluding %d row(s) with any_failed=True from analysis "
            "(kept in results.csv for documentation): %s",
            n_excluded, excluded,
        )

    # ── 8. Adjusted disagreement (on clean rows only) ─────────────────────────
    logger.info("Computing text-adjusted disagreement (Bushee et al. decomposition) ...")
    clean_results_df = compute_adjusted_disagreement(clean_results_df)
    # Write the clean analysis frame back so results.csv reflects adjusted cols for clean rows
    # Full frame (incl. failed rows) is preserved via all_rows; only analysis uses clean frame.
    clean_results_df.to_csv(config.OUTPUT_DIR / "results_clean.csv", index=False)

    # Flag single-filing firms (only 1 form type, no 10-K/10-Q pair).
    # K Round 6: "HAYN is still a single 10-Q with no 10-K ... flag it and hold it out of
    # the tier means." HAYN was acquired/delisted after August 2024; no post-cutoff 10-K exists.
    filing_type_counts = clean_results_df.groupby("ticker")["filing_type"].nunique()
    single_filing_tickers = set(filing_type_counts[filing_type_counts == 1].index)
    if single_filing_tickers:
        logger.warning(
            "Single-filing firms (only 10-K or only 10-Q, no matched pair): %s. "
            "Setting extraction_short=True to exclude from tier means.",
            sorted(single_filing_tickers),
        )
        clean_results_df.loc[
            clean_results_df["ticker"].isin(single_filing_tickers), "extraction_short"
        ] = True
        clean_results_df.to_csv(config.OUTPUT_DIR / "results_clean.csv", index=False)

    # Tier means exclude extraction_short rows (K Round 6: "exclude flagged rows from tier means
    # until the extraction is uniform"). Correlations and descriptive stats use all clean rows.
    tier_clean_df = clean_results_df[~clean_results_df["extraction_short"].fillna(False)].copy()
    n_short_excluded = len(clean_results_df) - len(tier_clean_df)
    if n_short_excluded:
        logger.info(
            "%d extraction_short row(s) excluded from tier means (kept in correlations).",
            n_short_excluded,
        )

    # ── 9. Collect complexity proxies ─────────────────────────────────────────
    proxy_cache_path = config.OUTPUT_DIR / "proxy_cache.csv"

    if args.skip_proxies and proxy_cache_path.exists():
        logger.info("Loading proxy data from cache")
        proxy_df = pd.read_csv(proxy_cache_path)
    else:
        tickers_needed = results_df["ticker"].unique().tolist()
        cik_map = {f["ticker"]: f.get("cik") for f in resolved_all}
        logger.info("Collecting complexity proxies for %d tickers ...", len(tickers_needed))
        proxy_rows: list[dict] = []
        for t in tickers_needed:
            logger.info("  Proxies for %s ...", t)
            proxy_rows.append(get_complexity_proxies(t, cik=cik_map.get(t)))
        proxy_df = pd.DataFrame(proxy_rows)
        proxy_df.to_csv(proxy_cache_path, index=False)

    # ── 10. Validation ────────────────────────────────────────────────────────
    logger.info("Running validation checks ...")
    validation = run_validation(clean_results_df, proxy_df, tier_results_df=tier_clean_df)

    write_validation_csv(validation, config.OUTPUT_DIR / "validation.csv")
    logger.info("validation.csv written")

    # Write descriptive stats
    if "descriptive" in validation:
        validation["descriptive"].to_csv(config.OUTPUT_DIR / "descriptive_stats.csv", index=False)
        logger.info("descriptive_stats.csv written")

    # Write correlation table with CIs and Bonferroni flags
    validation["correlations"].to_csv(config.OUTPUT_DIR / "correlations.csv", index=False)
    logger.info("correlations.csv written")

    # ── 11. Analyst coverage table ────────────────────────────────────────────
    cov_table = build_analyst_coverage_table(proxy_df, validation["merged"])
    cov_table.to_csv(config.OUTPUT_DIR / "analyst_coverage_table.csv", index=False)
    logger.info("analyst_coverage_table.csv written")

    # ── 12. Trading volume validation ─────────────────────────────────────────
    vol_df = None
    if args.trading_volume:
        logger.info("Running post-filing trading volume validation ...")
        vol_df = validate_trading_volume(results_df)
        if vol_df is not None:
            vol_df.to_csv(config.OUTPUT_DIR / "trading_volume_validation.csv", index=False)
            logger.info("trading_volume_validation.csv written (%d observations)", len(vol_df))

    # ── 13. Robustness variants ────────────────────────────────────────────────
    if args.robustness and not args.skip_agents:
        logger.info("Running prompt robustness variants (V2, V3) ...")
        from scipy.stats import spearmanr as _spearmanr
        robustness_rows: list[dict] = []

        v1_firm = (
            results_df.groupby("ticker")["numeric_disagreement"].mean().reset_index()
            .rename(columns={"numeric_disagreement": "v1_num"})
        )

        for variant in ["V2", "V3"]:
            logger.info("  Variant %s ...", variant)
            agents_v = get_agents_for_variant(variant)
            variant_rows: list[dict] = []
            for firm in resolved_firms:
                ticker = firm["ticker"]
                try:
                    rows_v, _ = process_firm(firm, config, logger, anonymize=False, agents_override=agents_v)
                    for r in rows_v:
                        r["prompt_variant"] = variant
                    variant_rows.extend(rows_v)
                except Exception as exc:
                    logger.error("Robustness %s FAILED %s: %s", variant, ticker, exc)

            if variant_rows:
                vdf = pd.DataFrame(variant_rows)
                vdf_firm = (
                    vdf.groupby("ticker")["numeric_disagreement"].mean().reset_index()
                    .rename(columns={"numeric_disagreement": f"{variant.lower()}_num"})
                )
                merged_v = v1_firm.merge(vdf_firm, on="ticker", how="inner")
                if len(merged_v) >= 4:
                    rho, pval = _spearmanr(merged_v["v1_num"], merged_v[f"{variant.lower()}_num"])
                    robustness_rows.append({
                        "variant_pair": f"V1_vs_{variant}",
                        "n_firms": len(merged_v),
                        "spearman_rho": round(float(rho), 4),
                        "p_value": round(float(pval), 4),
                    })
                    logger.info(
                        "  V1 vs %s ranking ρ=%.3f, p=%.4f (n=%d)",
                        variant, rho, pval, len(merged_v),
                    )
                vdf.to_csv(config.OUTPUT_DIR / f"results_{variant.lower()}.csv", index=False)

        write_robustness_csv(robustness_rows, config.OUTPUT_DIR / "robustness_rankings.csv")
        logger.info("robustness_rankings.csv written")

    # ── 14. Plots ─────────────────────────────────────────────────────────────
    logger.info("Generating plots ...")
    create_plots(validation["merged"], config.PLOTS_DIR)

    # ── 15. Summary ───────────────────────────────────────────────────────────
    write_summary(
        clean_results_df, validation, failed_firms,
        config.OUTPUT_DIR / "summary.txt",
        prompt_variant="V1",
        vol_df=vol_df,
        n_excluded_rows=n_excluded,
        n_total_rows=len(results_df),
        n_short_excluded=n_short_excluded,
        config=config,
    )
    # ── 16. Per-filing table for K ────────────────────────────────────────────
    pft_cols = [
        "ticker", "tier", "filing_type", "filing_date", "word_count",
        "numeric_disagreement", "embedding_disagreement", "extraction_short",
    ]
    pft_available = [c for c in pft_cols if c in clean_results_df.columns]
    pft = clean_results_df[pft_available].copy()
    pft = pft.sort_values(["tier", "ticker", "filing_type"])
    tier_order_map = {"large": 0, "mid": 1, "small": 2}
    if "tier" in pft.columns:
        pft["_tier_rank"] = pft["tier"].map(tier_order_map).fillna(9)
        pft = pft.sort_values(["_tier_rank", "ticker", "filing_type"]).drop(columns=["_tier_rank"])
    pft.to_csv(config.OUTPUT_DIR / "per_filing_table.csv", index=False)
    logger.info("per_filing_table.csv written (%d rows)", len(pft))

    logger.info("Pipeline complete. See output/ for all results.")


if __name__ == "__main__":
    main()
