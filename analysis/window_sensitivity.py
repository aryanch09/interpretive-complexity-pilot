"""
Scoring-window sensitivity test.

The chunk-level design fixes the confound the fixed word cap created, but it
introduces a free parameter: the window size.  If filing-level disagreement
depends materially on how the complete sections are cut into windows, then the
window is doing part of the measuring and the choice has to be justified rather
than assumed.

This runs a firm subset at two window sizes and reports:
  * rank correlation of filing-level disagreement between the two sizes
  * the tier gradient computed under each
  * whether either size changes the sign or monotonicity of the gradient

A high rank correlation and a stable gradient is the result we want: it says the
measure is a property of the filing, not of the cutting.  A low one is also
worth knowing, and belongs in the paper's robustness section either way.

Run:
  python -m analysis.window_sensitivity --firms AAPL,MSFT,HD,DRI,WYNN,TOL,NGVC,OFLX,UTMD \\
      --sizes 1200,2000 --agent-sleep 0.5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config  # noqa: E402

logger = logging.getLogger("window_sensitivity")

TIER_ORDER = ["large", "mid", "small"]


def run_size(firms: list[dict], config: Config, chunk_words: int) -> pd.DataFrame:
    """Score every firm at one window size, reusing that size's own cache."""
    from main import process_firm, load_intermediate, save_intermediate

    config.CHUNK_WORDS = chunk_words
    config.MIN_TAIL_WORDS = max(400, chunk_words // 3)

    cache_path = config.OUTPUT_DIR / f"window_cache_c{chunk_words}.json"
    cache = load_intermediate(cache_path, config.effective_cache_version)

    rows: list[dict] = []
    for firm in firms:
        ticker = firm["ticker"]
        if ticker in cache:
            logger.info("[c%d] %s: cached", chunk_words, ticker)
            rows.extend(cache[ticker])
            continue
        logger.info("[c%d] %s: scoring ...", chunk_words, ticker)
        try:
            firm_rows, _ = process_firm(firm, config, logger)
        except Exception as exc:
            logger.error("[c%d] %s FAILED: %s", chunk_words, ticker, exc)
            continue
        cache[ticker] = firm_rows
        save_intermediate(cache, cache_path, config.effective_cache_version)
        rows.extend(firm_rows)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["chunk_words"] = chunk_words
    return df


def compare(frames: dict[int, pd.DataFrame], config: Config) -> str:
    L: list[str] = []
    L.append("Scoring-window sensitivity")
    L.append("=" * 68)
    L.append("")

    empty = [cw for cw, df in frames.items() if df.empty or "any_failed" not in df]
    if empty:
        raise SystemExit(
            "No scored rows for window size(s) "
            + ", ".join(str(c) for c in empty)
            + ". Check the log above for per-firm errors."
        )

    sizes = sorted(frames)
    for cw in sizes:
        df = frames[cw]
        clean = df[~df["any_failed"].fillna(False)]
        L.append(
            f"window {cw:>5}w: {len(clean)} usable of {len(df)} filings, "
            f"{int(df['n_chunks_total'].sum())} windows scored"
        )
    L.append("")

    # ── Do the two window sizes rank filings the same way? ────────────────────
    L.append("Agreement between window sizes (filing level)")
    L.append("-" * 68)
    a, b = sizes[0], sizes[-1]
    ka = frames[a].assign(_k=frames[a].ticker + "_" + frames[a].filing_type)
    kb = frames[b].assign(_k=frames[b].ticker + "_" + frames[b].filing_type)
    m = ka.merge(kb, on="_k", suffixes=(f"_{a}", f"_{b}"))
    m = m[~m[f"any_failed_{a}"].fillna(False) & ~m[f"any_failed_{b}"].fillna(False)]

    for measure in ("embedding_disagreement", "numeric_disagreement"):
        x, y = m[f"{measure}_{a}"], m[f"{measure}_{b}"]
        ok = x.notna() & y.notna()
        if ok.sum() >= 3:
            rho, p = spearmanr(x[ok], y[ok])
            L.append(f"  {measure:<26} rho={rho:+.3f}  p={p:.4f}  n={int(ok.sum())}")
            L.append(
                f"  {'':<26} mean {a}w={x[ok].mean():.4f}   "
                f"mean {b}w={y[ok].mean():.4f}"
            )
        else:
            L.append(f"  {measure:<26} too few paired observations")
    L.append("")
    L.append(
        "  A high rank correlation means the two window sizes order filings the\n"
        "  same way — disagreement is a property of the filing, not of the cut.\n"
        "  A low one means the window is doing part of the measuring."
    )
    L.append("")

    # ── Does the tier gradient survive under both? ────────────────────────────
    L.append("Tier gradient under each window size (firm-level means)")
    L.append("-" * 68)
    L.append(f"  {'tier':<8}" + "".join(f"{('c' + str(c)):>14}" for c in sizes))
    for tier in TIER_ORDER:
        cells = []
        for cw in sizes:
            df = frames[cw]
            clean = df[
                ~df["any_failed"].fillna(False)
                & ~df["extraction_short"].fillna(False)
                & (df["tier"] == tier)
            ]
            firm_means = clean.groupby("ticker")["embedding_disagreement"].mean()
            cells.append(f"{firm_means.mean():.4f} (n={len(firm_means)})"
                         if len(firm_means) else "     —      ")
        L.append(f"  {tier:<8}" + "".join(f"{c:>14}" for c in cells))
    L.append("")
    L.append("  (embedding_disagreement — the primary measure)")
    L.append("")

    for cw in sizes:
        df = frames[cw]
        clean = df[~df["any_failed"].fillna(False) & ~df["extraction_short"].fillna(False)]
        means = [
            clean[clean.tier == t].groupby("ticker")["embedding_disagreement"].mean().mean()
            for t in TIER_ORDER
        ]
        if all(pd.notna(means)):
            mono = means[0] < means[1] < means[2]
            L.append(
                f"  c{cw}: {'monotone Large<Mid<Small' if mono else 'NOT monotone'}"
                f"  (spread {max(means) - min(means):+.4f})"
            )
    L.append("")

    # ── Does window size correlate with the measure? ──────────────────────────
    L.append("Length control under each window size")
    L.append("-" * 68)
    for cw in sizes:
        df = frames[cw]
        clean = df[~df["any_failed"].fillna(False)]
        for col, label in (("n_chunks_total", "n windows"),
                           ("section_words_total", "section words")):
            if col in clean and clean[col].notna().sum() >= 3:
                rho, p = spearmanr(clean[col], clean["embedding_disagreement"])
                L.append(f"  c{cw}  embedding x {label:<15} rho={rho:+.3f}  p={p:.4f}")
    L.append("")

    out = "\n".join(L)
    (config.OUTPUT_DIR / "window_sensitivity.txt").write_text(out, encoding="utf-8")
    combined = pd.concat(frames.values(), ignore_index=True)
    combined.to_csv(config.OUTPUT_DIR / "window_sensitivity.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--firms", required=True, help="Comma-separated tickers")
    ap.add_argument("--sizes", default="1200,2000", help="Comma-separated window sizes")
    ap.add_argument("--agent-sleep", type=float, default=None)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    )

    config = Config()
    config.validate()
    config.ensure_dirs()
    if args.agent_sleep is not None:
        config.AGENT_SLEEP_SECONDS = args.agent_sleep

    # FIRMS holds the dicts; FULL_PILOT_FIRMS/PILOT_FIRMS are ticker lists.
    from firms import FIRMS, resolve_ciks
    wanted = [t.strip().upper() for t in args.firms.split(",")]
    selected = [f for f in FIRMS if f["ticker"] in wanted]
    missing = sorted(set(wanted) - {f["ticker"] for f in selected})
    if missing:
        raise SystemExit(f"Unknown tickers (not in firms.FIRMS): {', '.join(missing)}")
    resolved = resolve_ciks(selected, cache_file=config.CIK_CACHE_FILE)

    frames: dict[int, pd.DataFrame] = {}
    for cw in [int(x) for x in args.sizes.split(",")]:
        frames[cw] = run_size(resolved, config, cw)

    print("\n" + compare(frames, config))


if __name__ == "__main__":
    main()
