"""
Synthetic `chunk_level_results.csv` generator with controllable ground truth.

Why this exists
---------------
Nothing is scored yet: the ~2,001-call Round 8 run has not produced
`output/chunk_level_results.csv`.  The analysis code still has to be proved
correct before it ever sees real numbers, so this module emits a file with the
EXACT real schema and a KNOWN data-generating process, and the tests assert that
the analysis recovers a planted tier effect and does not invent one when none is
planted.

This file writes SYNTHETIC data only.  It never writes into `output/`, and every
file it produces is named with a `SYNTHETIC_` prefix plus a sidecar
`.params.json` recording the seed and every generating parameter, so a synthetic
CSV can never be mistaken for a scored run.  Nothing here makes a network call
or a model call.

Sample shape (matched to the real frozen sample, ANALYSIS_PLAN.md §1, §9)
-------------------------------------------------------------------------
    30 firms, 10 per tier
    59 filings  (29 firms x {10-K, 10-Q}; HAYN contributes only a 10-Q)
    667 windows total, of which 363 are MD&A
    MD&A windows per filing: median 6.5 large / 6.5 mid / 4.0 small, min 2, max 25
    10 filings yield no Risk Factors windows, all of them 10-Qs (plan §2)
    word_count: 1,200 for every window except the last of each section, which
    carries the MIN_TAIL_WORDS merge and lands in [403, 1598]

The per-tier MD&A window-count lists below are fixed, hand-built integer lists,
not random draws, precisely so those five published sample statistics (59 / 667 /
363 / the three medians / min / max) hold EXACTLY and are asserted on every
generation.  Anyone editing them will trip `_check_shape`.

Data-generating process
-----------------------
Per window, for the embedding measure:

    emb = base_embedding
        + tier_effect * tier_rank          (tier_rank: large 0, mid 1, small 2)
        + firm_effect[ticker]              ~ N(0, firm_sd)
        + filing_effect[filing]            ~ N(0, filing_sd)
        + window_noise                     ~ N(0, window_sd)

`tier_effect = 0.0` plants NO tier effect; a positive value plants a monotone
Small > Mid > Large gradient of exactly that size per tier step, on the
firm-level means that `aggregate_to_firm` produces.  Default magnitudes are
taken from the superseded Round 6 scale (firm means ~0.27-0.33, within-tier SD
~0.05-0.08) so the fixture is not easier to analyse than the real thing.

The numeric measure is built the other way round, from the scores, so the file
is internally consistent with the plan's definition: three integer persona
scores on 0-100 are emitted, and `numeric_disagreement` is recomputed as their
population SD (ddof=0) after dividing by 10.  A test checks that identity holds
in the written file.

Run it
------
    python3 -m tests.fixture_chunks            # writes tests/fixtures/
    python3 -m tests.fixture_chunks --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

sys.modules.setdefault("tensorflow", None)  # see tests/__init__.py

logger = logging.getLogger(__name__)

# ── The exact schema of output/chunk_level_results.csv, in order ──────────────
# Mirrors the dict literal built in main.py (the chunk_rows.append(...) block).
CHUNK_SCHEMA: tuple[str, ...] = (
    "ticker", "tier", "filing_type", "filing_date", "section",
    "chunk_index", "n_in_section", "word_count",
    "numeric_disagreement", "embedding_disagreement",
    "value_score", "sentiment_score", "fundamentals_score",
    "any_failed", "any_parse_error", "served_by",
    "value_reasoning", "sentiment_reasoning", "fundamentals_reasoning",
)

PERSONAS: tuple[str, ...] = ("value", "sentiment", "fundamentals")
TIER_RANK: Mapping[str, int] = {"large": 0, "mid": 1, "small": 2}

CHUNK_WORDS = 1200          # config.CHUNK_WORDS
TAIL_MIN, TAIL_MAX = 403, 1598   # observed merged-tail range in the real sample

# 10 firms per tier, from firms.FULL_PILOT_FIRMS.
TICKERS: Mapping[str, tuple[str, ...]] = {
    "large": ("AAPL", "AMZN", "BAC", "GOOGL", "HD", "JPM", "MSFT", "PG", "WMT", "XOM"),
    "mid":   ("CATY", "CLFD", "DRI", "HAS", "IBP", "KFRC", "PLAB", "TOL", "WTS", "WYNN"),
    "small": ("CLIR", "HAYN", "IIIN", "LWAY", "MLAB", "NGVC", "OFLX", "RELL", "UFPT", "UTMD"),
}
#: HAYN was delisted/acquired in August 2024 and will never file a 10-K
#: (CLAUDE.md); it contributes one filing where every other firm contributes two.
SINGLE_FILING_TICKERS: Mapping[str, str] = {"HAYN": "10-Q"}

# MD&A windows per filing, per tier.  Sorted ascending; see _check_shape for the
# five invariants these satisfy.  Sums: 163 + 123 + 77 = 363.
MDA_COUNTS: Mapping[str, tuple[int, ...]] = {
    "large": (2, 3, 4, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 11, 13, 19, 25),
    "mid":   (2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 9, 9, 10),
    "small": (2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8),
}

TARGET_FILINGS = 59
TARGET_WINDOWS = 667
TARGET_MDA_WINDOWS = 363
TARGET_RISK_WINDOWS = TARGET_WINDOWS - TARGET_MDA_WINDOWS   # 304
#: ANALYSIS_PLAN.md §2: ten filings yield no Risk windows, all 10-Qs.
#: (CLAUDE.md's older run note says 53/59 have Risk windows, i.e. six missing.
#: The plan is the binding document and is the later of the two, so ten is used
#: here and the discrepancy is reported.)
N_FILINGS_WITHOUT_RISK = 10

RISK_FROM_MDA_RATIO = 0.9   # Risk sections run a little shorter than MD&A
FAILED_SENTINEL = "API call failed after all retries."
BACKEND = "DeepInfra"       # the pinned bf16 backend; constant across a clean run


@dataclass
class FixtureParams:
    """Every knob, recorded verbatim into the sidecar JSON beside the CSV."""

    seed: int = 20261004
    label: str = "effect"
    # ── ground truth ──
    tier_effect: float = 0.04          # embedding disagreement per tier step
    numeric_tier_effect: float = 0.0   # 0.0 mirrors the real flat numeric finding
    base_embedding: float = 0.28
    base_numeric: float = 0.60
    firm_sd: float = 0.03
    filing_sd: float = 0.02
    window_sd: float = 0.06
    numeric_window_sd: float = 0.15
    # ── failures ──
    failure_rate: float = 0.0
    gate_breaking_filings: tuple[tuple[str, str], ...] = ()
    parse_error_rate: float = 0.0
    # ── shape ──
    n_filings_without_risk: int = N_FILINGS_WITHOUT_RISK
    notes: str = ""
    realised: dict = field(default_factory=dict)


# ── shape construction ────────────────────────────────────────────────────────

def _filing_plan() -> list[dict]:
    """Deterministic (ticker, tier, filing_type, filing_date, mda_k) list.

    Window counts are assigned by splitting each tier's sorted count list: the
    larger half to 10-Ks, the smaller half to 10-Qs (annual MD&As are longer),
    then by alphabetical ticker within each form.  Fully deterministic, no RNG.
    """
    plan: list[dict] = []
    for tier, tickers in TICKERS.items():
        counts = sorted(MDA_COUNTS[tier])
        tenk_tickers = [t for t in tickers if SINGLE_FILING_TICKERS.get(t, "10-K") == "10-K"]
        tenq_tickers = [t for t in tickers if SINGLE_FILING_TICKERS.get(t, "10-Q") == "10-Q"]
        n_k = len(tenk_tickers)
        if n_k + len(tenq_tickers) != len(counts):
            raise AssertionError(
                f"{tier}: {n_k} 10-K + {len(tenq_tickers)} 10-Q != "
                f"{len(counts)} window counts"
            )
        q_counts, k_counts = counts[: len(tenq_tickers)], counts[len(tenq_tickers):]
        for tkr, k in zip(tenk_tickers, k_counts):
            plan.append({"ticker": tkr, "tier": tier, "filing_type": "10-K",
                         "filing_date": _filing_date(tkr, "10-K"), "mda_k": int(k)})
        for tkr, k in zip(tenq_tickers, q_counts):
            plan.append({"ticker": tkr, "tier": tier, "filing_type": "10-Q",
                         "filing_date": _filing_date(tkr, "10-Q"), "mda_k": int(k)})
    plan.sort(key=lambda r: (r["ticker"], r["filing_type"]))
    return plan


def _filing_date(ticker: str, form: str) -> str:
    """Deterministic post-2024-07-01 filing date, YYYYMMDD, from the ticker."""
    if ticker == "HAYN":
        return "20240801"   # the real last filing before delisting
    h = sum(ord(c) * (i + 1) for i, c in enumerate(ticker))
    if form == "10-K":
        month = 1 + h % 12
        day = 1 + h % 27
        return f"2025{month:02d}{day:02d}"
    month = 1 + (h + 5) % 9          # keep the 10-Q inside 2026 and after the 10-K
    day = 1 + (h + 11) % 27
    return f"2026{month:02d}{day:02d}"


def _assign_risk_counts(plan: list[dict], n_without_risk: int) -> None:
    """Give each filing a `risk_k`, 0 for the chosen 10-Qs, summing to 304.

    The filings with no Risk windows are 10-Qs incorporating the 10-K's risk
    factors by reference (plan §2).  They are picked by stepping through the
    10-Qs in ticker order at a fixed stride so the missingness is spread across
    tiers rather than concentrated in one -- a deliberately conservative choice,
    since concentrating it in small-caps would make the secondary Risk analysis
    look better identified than it is.

    The residual needed to hit the documented 304-window total is distributed
    +/-1 at a time, in a deterministic order, never below 2 windows, and the
    total adjustment is logged.
    """
    tenq = [r for r in plan if r["filing_type"] == "10-Q"]
    if n_without_risk > len(tenq):
        raise ValueError(
            f"n_filings_without_risk={n_without_risk} exceeds the {len(tenq)} 10-Qs"
        )
    stride = max(1, len(tenq) // n_without_risk)
    chosen = [tenq[(i * stride) % len(tenq)] for i in range(n_without_risk)]
    seen: set[int] = set()
    unique_chosen = []
    for r in chosen:
        if id(r) not in seen:
            seen.add(id(r))
            unique_chosen.append(r)
    i = 0
    while len(unique_chosen) < n_without_risk:
        cand = tenq[i % len(tenq)]
        if id(cand) not in seen:
            seen.add(id(cand))
            unique_chosen.append(cand)
        i += 1
    for r in plan:
        r["risk_k"] = 0 if id(r) in seen else max(
            2, int(round(RISK_FROM_MDA_RATIO * r["mda_k"]))
        )

    with_risk = [r for r in plan if r["risk_k"] > 0]
    delta = TARGET_RISK_WINDOWS - sum(r["risk_k"] for r in with_risk)
    step = 1 if delta > 0 else -1
    moved = 0
    order = sorted(with_risk, key=lambda r: (-r["mda_k"], r["ticker"], r["filing_type"]))
    guard = 0
    while delta != 0:
        progressed = False
        for r in order:
            if delta == 0:
                break
            if step < 0 and r["risk_k"] <= 2:
                continue
            r["risk_k"] += step
            delta -= step
            moved += 1
            progressed = True
        guard += 1
        if not progressed or guard > 10_000:
            raise AssertionError(
                f"could not reach {TARGET_RISK_WINDOWS} Risk windows; {delta} left over"
            )
    logger.info(
        "fixture: Risk window counts adjusted by %d single-window steps to hit the "
        "documented total of %d across %d filings (%d filings have none)",
        moved, TARGET_RISK_WINDOWS, len(with_risk), len(plan) - len(with_risk),
    )


def _check_shape(plan: list[dict]) -> dict:
    """Assert the five published sample statistics hold exactly."""
    n_filings = len(plan)
    n_mda = sum(r["mda_k"] for r in plan)
    n_risk = sum(r["risk_k"] for r in plan)
    medians = {
        tier: float(np.median([r["mda_k"] for r in plan if r["tier"] == tier]))
        for tier in TIER_RANK
    }
    allk = [r["mda_k"] for r in plan]
    realised = {
        "n_firms": len({r["ticker"] for r in plan}),
        "n_filings": n_filings,
        "n_windows": n_mda + n_risk,
        "n_mda_windows": n_mda,
        "n_risk_windows": n_risk,
        "mda_median_by_tier": medians,
        "mda_min": min(allk),
        "mda_max": max(allk),
        "n_filings_without_risk": sum(1 for r in plan if r["risk_k"] == 0),
    }
    expected = {
        "n_firms": 30, "n_filings": TARGET_FILINGS, "n_windows": TARGET_WINDOWS,
        "n_mda_windows": TARGET_MDA_WINDOWS, "n_risk_windows": TARGET_RISK_WINDOWS,
        "mda_median_by_tier": {"large": 6.5, "mid": 6.5, "small": 4.0},
        "mda_min": 2, "mda_max": 25,
    }
    for key, want in expected.items():
        if realised[key] != want:
            raise AssertionError(
                f"fixture shape drifted: {key} = {realised[key]!r}, expected {want!r}"
            )
    if any(r["filing_type"] != "10-Q" for r in plan if r["risk_k"] == 0):
        raise AssertionError("a non-10-Q was given zero Risk windows (plan §2)")
    return realised


# ── generation ────────────────────────────────────────────────────────────────

def _scores_for(target_numeric: float, rng: np.random.Generator) -> tuple[list[int], float]:
    """Three integer 0-100 persona scores whose /10 population SD is ~target.

    For scores (m-a, m, m+a) the population SD is a*sqrt(2/3), so a target SD of
    t on the 0-10 scale needs a = 10*t/sqrt(2/3) on the 0-100 scale.  The scores
    are then rounded to integers (as the real parser produces) and the measure is
    RECOMPUTED from the rounded values, so the written file satisfies the plan's
    definition exactly rather than approximately.
    """
    t = max(0.0, float(target_numeric))
    a = 10.0 * t / np.sqrt(2.0 / 3.0)
    a = float(np.clip(a, 0.0, 25.0))
    m = float(rng.uniform(35.0, 65.0))
    raw = [m - a, m, m + a]
    scores = [int(np.clip(round(v), 0, 100)) for v in raw]
    rng.shuffle(scores)   # persona order carries no information
    measure = float(np.std(np.asarray(scores, dtype=float) / 10.0, ddof=0))
    return scores, measure


def generate_chunk_level_results(params: Optional[FixtureParams] = None) -> pd.DataFrame:
    """Build a synthetic chunk-level frame with the exact real schema.

    Deterministic in `params.seed`: the same params always give byte-identical
    output.  Returns the frame; `params.realised` is filled in with the shape
    statistics that were actually produced.
    """
    p = params or FixtureParams()
    if p.failure_rate < 0.0 or p.failure_rate > 1.0:
        raise ValueError(f"failure_rate must lie in [0,1], got {p.failure_rate}")
    rng = np.random.default_rng(p.seed)

    plan = _filing_plan()
    _assign_risk_counts(plan, p.n_filings_without_risk)
    p.realised = _check_shape(plan)
    p.realised["seed"] = int(p.seed)

    tickers = sorted({r["ticker"] for r in plan})
    firm_effect = {t: float(rng.normal(0.0, p.firm_sd)) for t in tickers}
    filing_effect = {
        (r["ticker"], r["filing_type"]): float(rng.normal(0.0, p.filing_sd))
        for r in plan
    }

    gate_break = {(str(a), str(b)) for a, b in p.gate_breaking_filings}
    unknown_break = gate_break - {(r["ticker"], r["filing_type"]) for r in plan}
    if unknown_break:
        raise ValueError(f"gate_breaking_filings names unknown filings: {sorted(unknown_break)}")

    rows: list[dict] = []
    n_forced_failed = 0
    for rec in plan:
        tier = rec["tier"]
        rank = TIER_RANK[tier]
        for section in ("mda", "risk"):
            k = rec["mda_k"] if section == "mda" else rec["risk_k"]
            if k == 0:
                continue
            breaking = (section == "mda" and (rec["ticker"], rec["filing_type"]) in gate_break)
            # Leave exactly one window alive: fails both ">= 2 surviving" and
            # ">= half surviving" limbs of the rule-8.2 gate for any k >= 2.
            survivor = int(rng.integers(0, k)) if breaking else -1
            for idx in range(k):
                words = CHUNK_WORDS if idx < k - 1 else int(
                    rng.integers(TAIL_MIN, TAIL_MAX + 1)
                )
                emb = (
                    p.base_embedding
                    + p.tier_effect * rank
                    + firm_effect[rec["ticker"]]
                    + filing_effect[(rec["ticker"], rec["filing_type"])]
                    + float(rng.normal(0.0, p.window_sd))
                )
                emb = float(np.clip(emb, 0.0, 1.0))
                num_target = (
                    p.base_numeric
                    + p.numeric_tier_effect * rank
                    + float(rng.normal(0.0, p.numeric_window_sd))
                )
                scores, numeric = _scores_for(num_target, rng)

                failed = breaking and idx != survivor
                if not failed and p.failure_rate > 0.0:
                    failed = bool(rng.random() < p.failure_rate)
                parse_err = (not failed) and p.parse_error_rate > 0.0 and bool(
                    rng.random() < p.parse_error_rate
                )
                if failed:
                    n_forced_failed += 1

                reasoning = {
                    r: f"{r} lens on {rec['ticker']} {rec['filing_type']} {section} "
                       f"window {idx}: margin, liquidity and guidance language."
                    for r in PERSONAS
                }
                row = {
                    "ticker": rec["ticker"],
                    "tier": tier,
                    "filing_type": rec["filing_type"],
                    "filing_date": rec["filing_date"],
                    "section": section,
                    "chunk_index": idx,
                    "n_in_section": k,
                    "word_count": words,
                    # Rule 8.1: a failed window stores NO disagreement at all.
                    "numeric_disagreement": None if failed else numeric,
                    "embedding_disagreement": None if failed else emb,
                    "value_score": scores[0],
                    "sentiment_score": scores[1],
                    "fundamentals_score": scores[2],
                    "any_failed": bool(failed),
                    "any_parse_error": bool(parse_err),
                    "served_by": f"{BACKEND}|None" if failed else BACKEND,
                    "value_reasoning": reasoning["value"],
                    "sentiment_reasoning": reasoning["sentiment"],
                    "fundamentals_reasoning": reasoning["fundamentals"],
                }
                if failed:
                    # One persona's call is what failed: its score is absent and
                    # its reasoning holds the real sentinel string, exactly as
                    # agents/runner.py writes it.
                    dead = PERSONAS[int(rng.integers(0, len(PERSONAS)))]
                    row[f"{dead}_score"] = None
                    row[f"{dead}_reasoning"] = FAILED_SENTINEL
                rows.append(row)

    df = pd.DataFrame(rows, columns=list(CHUNK_SCHEMA))
    if list(df.columns) != list(CHUNK_SCHEMA):
        raise AssertionError("column order drifted from CHUNK_SCHEMA")
    if len(df) != TARGET_WINDOWS:
        raise AssertionError(f"wrote {len(df)} windows, expected {TARGET_WINDOWS}")

    p.realised["n_failed_windows"] = int(df["any_failed"].sum())
    p.realised["word_count_min"] = int(df["word_count"].min())
    p.realised["word_count_max"] = int(df["word_count"].max())
    p.realised["word_count_median"] = float(df["word_count"].median())
    logger.info(
        "fixture: %d windows (%d MD&A) across %d filings / %d firms, seed=%d, "
        "tier_effect=%.4f, %d failed window(s)",
        len(df), int((df["section"] == "mda").sum()), p.realised["n_filings"],
        p.realised["n_firms"], p.seed, p.tier_effect, p.realised["n_failed_windows"],
    )
    return df


def write_fixture(
    params: Optional[FixtureParams] = None,
    out_dir: Path | str = Path(__file__).parent / "fixtures",
) -> tuple[Path, pd.DataFrame]:
    """Write `SYNTHETIC_chunk_level_results_<label>.csv` plus its params sidecar.

    The seed and every generating parameter land in the sidecar JSON, because the
    CSV schema is fixed by the real pipeline and must not grow a column.
    """
    p = params or FixtureParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate_chunk_level_results(p)
    csv_path = out_dir / f"SYNTHETIC_chunk_level_results_{p.label}.csv"
    if "output" in csv_path.resolve().parts:
        raise RuntimeError(
            f"refusing to write synthetic data under output/: {csv_path}"
        )
    df.to_csv(csv_path, index=False)
    sidecar = csv_path.with_suffix(".params.json")
    sidecar.write_text(json.dumps(asdict(p), indent=2, default=str))
    logger.info("fixture: wrote %s and %s", csv_path, sidecar)
    return csv_path, df


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--seed", type=int, default=FixtureParams.seed)
    ap.add_argument("--tier-effect", type=float, default=None,
                    help="embedding disagreement planted per tier step "
                         "(0.0 = no effect). Default: write both 'effect' and 'null'.")
    ap.add_argument("--failure-rate", type=float, default=0.0)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "fixtures")
    args = ap.parse_args(argv)

    specs = (
        [FixtureParams(seed=args.seed, label="effect", tier_effect=args.tier_effect,
                       failure_rate=args.failure_rate)]
        if args.tier_effect is not None else
        [FixtureParams(seed=args.seed, label="effect", tier_effect=0.04,
                       failure_rate=args.failure_rate),
         FixtureParams(seed=args.seed, label="null", tier_effect=0.0,
                       failure_rate=args.failure_rate)]
    )
    for spec in specs:
        path, df = write_fixture(spec, out_dir=args.out_dir)
        print(f"{path}  ({len(df)} windows, seed={spec.seed}, "
              f"tier_effect={spec.tier_effect})")
        print(json.dumps(spec.realised, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
