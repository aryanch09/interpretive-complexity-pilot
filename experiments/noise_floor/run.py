"""
E1 (noise floor) and E2 (persona-free baseline) — the two experiments that decide
whether "interpretive complexity" is a construct or an artifact.

They answer the first two questions on any referee's list:

  E1. IS IT JUST TEMPERATURE NOISE? Run the SAME persona on the SAME window three
      times at the production temperature. The spread between those three repeats
      is the noise floor. If inter-persona disagreement is not clearly larger than
      intra-persona disagreement, the measure is sampling noise with a label.

  E2. WOULD ANY THREE SAMPLES DISAGREE, PERSONA OR NOT? Run three copies of a
      single NEUTRAL analyst prompt on the same window. Any disagreement there is
      what you get from sampling a model three times without varying the lens.
      Real-persona disagreement MINUS this baseline is the part attributable to
      the analytical lens — which is exactly what the paper claims to measure.

E2 is the sharper test. E1 can be passed by a measure that merely responds to
prompt wording; E2 cannot.

Both make model calls, so they are deliberately kept out of `make all`.

Design notes that matter for interpretation:
  * The subset is STRATIFIED by tier and drawn with a fixed seed, so the
    comparison is not dominated by one tier's windows.
  * Repeats use the production temperature (0.3). Running at temperature 0 would
    understate the floor and flatter the measure.
  * Embedding distances are computed with the same encoder and the same
    1-minus-cosine, mean-over-pairs recipe as the primary measure, so the numbers
    are directly comparable rather than merely similar.

Run:  python3 -m experiments.noise_floor.run --n-per-tier 20          (dry run)
      python3 -m experiments.noise_floor.run --n-per-tier 20 --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("tensorflow", None)

from analysis.trend_tests import DEFAULT_SEED, bootstrap_ci
from config import Config

logger = logging.getLogger(__name__)

ROLES = ("value", "sentiment", "fundamentals")
N_REPEATS = 3

# E2's neutral lens: deliberately generic, with no analytical stance. Three
# identical copies of this isolate sampling variation from lens variation.
NEUTRAL_PROMPT = """You are a financial analyst. Read the following excerpt from \
an SEC filing and assess what it implies about the company's outlook.

SCORE: [An integer from 0 to 100, where 50 = neutral (no directional view).
  This is divided by 10 during analysis to give a 0.0-10.0 scale.
  Calibration guide - use the full range, do not cluster near 50:
    85-100 : overwhelmingly positive / exceptional outlook
    65-84  : clearly positive, investable, meaningful upside
    55-64  : mildly positive, slight lean bullish
    45-54  : neutral, roughly balanced signals
    35-44  : mildly negative, some concern
    15-34  : clearly negative, significant concern or deterioration
    0-14   : overwhelmingly negative / severe expected underperformance
  Write a single integer with no decimal point.]
CONFIDENCE: [A decimal from 0.0 to 1.0.]
REASONING: [2-4 sentences explaining your assessment.]
KEY_FACTORS: [Up to four short phrases, comma separated.]"""


def select_windows(chunk_df: pd.DataFrame, n_per_tier: int, section: str,
                   seed: int) -> pd.DataFrame:
    """Stratified sample of clean windows, balanced across tiers."""
    d = chunk_df[(chunk_df["section"] == section)].copy()
    d = d[~d["any_failed"].astype(str).str.lower().isin({"true", "1"})]
    d = d[~d["any_parse_error"].astype(str).str.lower().isin({"true", "1"})]
    rng = np.random.default_rng(seed)
    picks = []
    for tier in ("large", "mid", "small"):
        sub = d[d["tier"] == tier]
        if sub.empty:
            logger.warning("no clean windows for tier %s", tier)
            continue
        k = min(n_per_tier, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=k, replace=False)
        picks.append(sub.loc[idx])
        if k < n_per_tier:
            logger.warning("tier %s: only %d windows available, wanted %d",
                           tier, k, n_per_tier)
    return pd.concat(picks).reset_index(drop=True)


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


def _mean_pairwise(texts: list[str], config: Config) -> float:
    """Same recipe as the primary measure: 1 - cosine, averaged over pairs."""
    from analysis.embeddings import encode_with_cache

    usable = [t for t in texts if t and t.strip()]
    if len(usable) < 2:
        return float("nan")
    vecs = encode_with_cache(usable, cache_path=config.OUTPUT_DIR / "embedding_cache.json")
    ds = []
    for a, b in combinations(range(len(vecs)), 2):
        va, vb = vecs[a], vecs[b]
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom > 0:
            ds.append(float(np.clip(1.0 - float(np.dot(va, vb)) / denom, 0.0, 2.0)))
    return float(np.mean(ds)) if ds else float("nan")


def run(config: Config, n_per_tier: int, section: str, seed: int,
        apply: bool) -> dict:
    chunk_path = config.OUTPUT_DIR / "chunk_level_results.csv"
    if not chunk_path.exists():
        raise FileNotFoundError(f"{chunk_path} not found — run the scoring pipeline first.")
    chunk_df = pd.read_csv(chunk_path)
    sel = select_windows(chunk_df, n_per_tier, section, seed)

    n_calls = len(sel) * (len(ROLES) * N_REPEATS + N_REPEATS)
    print(f"windows selected : {len(sel)}  "
          f"({dict(sel['tier'].value_counts())})")
    print(f"E1 calls         : {len(sel) * len(ROLES) * N_REPEATS}  "
          f"(3 personas x {N_REPEATS} repeats)")
    print(f"E2 calls         : {len(sel) * N_REPEATS}  "
          f"({N_REPEATS} copies of one neutral prompt)")
    print(f"TOTAL            : {n_calls} calls")
    print(f"estimated cost   : ${n_calls * 2700 / 1e6 * 0.029 + n_calls * 450 / 1e6 * 0.14:.3f}")
    if not apply:
        print("\nDRY RUN — nothing called. Re-run with --apply.")
        return {"dry_run": True, "n_windows": len(sel), "n_calls": n_calls}

    from agents.prompts import AGENTS
    from agents.runner import run_agent

    out_dir = config.OUTPUT_DIR / "experiments" / "noise_floor"
    out_dir.mkdir(parents=True, exist_ok=True)

    # RESUME: skip windows already present in noise_floor.csv. This experiment is
    # 12 calls per window and has been interrupted by rate limiting twice.
    done_path = out_dir / "noise_floor.csv"
    records, done = [], set()
    if done_path.exists():
        prev = pd.read_csv(done_path)
        records = prev.to_dict("records")
        done = {(x["ticker"], x["filing_type"], x["section"], int(x["chunk_index"]))
                for x in records}
        print(f"resuming: {len(done)} window(s) already done, skipping them")

    for n, (_, row) in enumerate(sel.iterrows(), 1):
        if (row.ticker, row.filing_type, row.section, int(row.chunk_index)) in done:
            continue
        text = _window_text(config, row)
        if text is None:
            logger.warning("could not rebuild window for %s", row["ticker"])
            continue
        print(f"[{n}/{len(sel)}] {row['ticker']} {row['filing_type']} "
              f"{row['section']}#{row['chunk_index']}", flush=True)

        rec = {k: row[k] for k in ("ticker", "tier", "filing_type", "filing_date",
                                   "section", "chunk_index", "word_count")}
        rec["inter_persona_observed"] = float(row["embedding_disagreement"])

        # E1: each persona repeated N times.
        intra = {}
        first_pass = {}
        for role in ROLES:
            reasonings, scores = [], []
            for rep in range(N_REPEATS):
                out = run_agent(AGENTS[role]["prompt"], text,
                                AGENTS[role]["name"], config)
                reasonings.append(out.get("reasoning") or "")
                scores.append(out.get("score"))
                if rep == 0:
                    first_pass[role] = out.get("reasoning") or ""
            intra[role] = _mean_pairwise(reasonings, config)
            rec[f"intra_{role}_embedding"] = intra[role]
            valid = [s for s in scores if s is not None]
            rec[f"intra_{role}_score_sd"] = (
                float(np.std(valid, ddof=0)) if len(valid) >= 2 else float("nan"))
        rec["intra_persona_mean"] = float(np.nanmean(list(intra.values())))
        # Inter-persona distance recomputed from THIS run's first repeat, so the
        # comparison is within one sampling draw rather than against the main run.
        rec["inter_persona_rerun"] = _mean_pairwise(
            [first_pass[r] for r in ROLES], config)

        # E2: three copies of one neutral prompt.
        neutral_reasonings = []
        for _rep in range(N_REPEATS):
            out = run_agent(NEUTRAL_PROMPT, text, "Neutral Analyst", config)
            neutral_reasonings.append(out.get("reasoning") or "")
        rec["persona_free_baseline"] = _mean_pairwise(neutral_reasonings, config)

        records.append(rec)
        pd.DataFrame(records).to_csv(out_dir / "noise_floor.csv", index=False)

    df = pd.DataFrame(records)
    results = _summarise(df, seed)
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=1, default=str), encoding="utf-8")
    (out_dir / "report.txt").write_text(_report(results, df), encoding="utf-8")
    return results


def _summarise(df: pd.DataFrame, seed: int) -> dict:
    out: dict = {"seed": seed, "n_windows": int(len(df)), "by_tier": {}}
    if df.empty:
        return out

    def ci(series):
        s = series.dropna().to_numpy()
        if len(s) < 3:
            return None
        b = bootstrap_ci(s, statistic=np.mean, n_boot=10000, seed=seed,
                         method="percentile")
        return {"mean": float(b.observed), "ci": [float(b.low), float(b.high)]}

    out["inter_persona"] = ci(df["inter_persona_rerun"])
    out["intra_persona"] = ci(df["intra_persona_mean"])
    out["persona_free_baseline"] = ci(df["persona_free_baseline"])

    df = df.copy()
    df["ratio"] = df["inter_persona_rerun"] / df["intra_persona_mean"]
    df["lens_effect"] = df["inter_persona_rerun"] - df["persona_free_baseline"]
    out["ratio_inter_over_intra"] = ci(df["ratio"])
    out["lens_effect_over_baseline"] = ci(df["lens_effect"])

    for tier in ("large", "mid", "small"):
        s = df[df["tier"] == tier]
        if s.empty:
            continue
        out["by_tier"][tier] = {
            "n": int(len(s)),
            "inter_persona": ci(s["inter_persona_rerun"]),
            "intra_persona": ci(s["intra_persona_mean"]),
            "persona_free_baseline": ci(s["persona_free_baseline"]),
            "ratio": ci(s["ratio"]),
            "lens_effect": ci(s["lens_effect"]),
        }

    le = out.get("lens_effect_over_baseline")
    out["verdict"] = {
        "inter_exceeds_intra": bool(
            out["ratio_inter_over_intra"]
            and out["ratio_inter_over_intra"]["ci"][0] > 1.0),
        "lens_effect_positive": bool(le and le["ci"][0] > 0),
        "reading": (
            "The measure survives both checks only if inter-persona disagreement "
            "exceeds the intra-persona noise floor AND exceeds the persona-free "
            "baseline. Failing the second means three samples of one neutral "
            "prompt disagree about as much as three different lenses do, in which "
            "case the measure is sampling variation rather than analytical "
            "perspective, and the paper must say so."
        ),
    }
    return out


def _report(r: dict, df: pd.DataFrame) -> str:
    L = ["E1/E2 — Noise floor and persona-free baseline", "=" * 78, ""]
    L.append(f"Windows: {r['n_windows']}   seed: {r['seed']}")
    L.append("")

    def line(label, v):
        if not v:
            return f"  {label:34s} (insufficient data)"
        return (f"  {label:34s} {v['mean']:.4f}  "
                f"[{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]")

    L.append("Embedding distance, mean with 95% bootstrap CI")
    L.append("-" * 78)
    L.append(line("inter-persona (3 lenses)", r.get("inter_persona")))
    L.append(line("intra-persona (same lens x3)", r.get("intra_persona")))
    L.append(line("persona-free (neutral x3)", r.get("persona_free_baseline")))
    L.append("")
    L.append(line("ratio inter / intra", r.get("ratio_inter_over_intra")))
    L.append(line("lens effect (inter - neutral)", r.get("lens_effect_over_baseline")))
    L.append("")
    v = r.get("verdict", {})
    L.append(f"  inter-persona exceeds the noise floor : {v.get('inter_exceeds_intra')}")
    L.append(f"  lens effect above the baseline       : {v.get('lens_effect_positive')}")
    L.append("")
    if r.get("by_tier"):
        L.append("By tier")
        L.append("-" * 78)
        for tier, b in r["by_tier"].items():
            rr = b.get("ratio")
            le = b.get("lens_effect")
            L.append(f"  {tier:6s} n={b['n']:3d}  "
                     f"ratio {rr['mean']:.2f} [{rr['ci'][0]:.2f}, {rr['ci'][1]:.2f}]  "
                     if rr else f"  {tier:6s} n={b['n']:3d}  ")
            if le:
                L.append(f"          lens effect {le['mean']:+.4f} "
                         f"[{le['ci'][0]:+.4f}, {le['ci'][1]:+.4f}]")
    L.append("")
    L.append(v.get("reading", ""))
    L.append("")
    L.append("Labelled EXPLORATORY in ANALYSIS_PLAN.md section 6.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-per-tier", type=int, default=20)
    ap.add_argument("--section", default="mda")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--agent-sleep", type=float, default=0.5)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)

    config = Config()
    config.API_PROVIDER = a.provider
    config.apply_provider()
    config.AGENT_SLEEP_SECONDS = a.agent_sleep

    r = run(config, a.n_per_tier, a.section, a.seed, a.apply)
    if a.apply:
        print()
        print(_report(r, pd.DataFrame()))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
