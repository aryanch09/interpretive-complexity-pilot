"""
Synthetic chunk_level_results.csv generator with controllable ground truth.

Why this exists
---------------
output/chunk_level_results.csv does not exist yet: the ~2,001-call scoring run
has not been made.  The window-count battery therefore has to be testable
against data whose truth is known by construction, so that "the analysis found a
tier effect" can be distinguished from "the analysis would find a tier effect in
anything".

Two fixtures matter, and they are the pair the battery has to be able to tell
apart:

  planted_effect_fixture()
      A real tier effect at the WINDOW level: every window of a small-cap
      filing has a higher expected disagreement than every window of a large-cap
      filing.  Window counts still differ by tier, exactly as in the real
      sample, so the effect is not *made* by the counts.  W3 (equal-k
      subsampling) must RECOVER this.

  count_artifact_fixture()
      No tier effect whatsoever: the expected window-level disagreement is
      IDENTICAL for every tier.  What differs is the window count - large-caps
      get ~12 windows per filing, small-caps exactly 2 - so the variance of a
      filing mean differs by tier and a gradient can appear by chance.  The seed
      is searched (deterministically, and the offset is recorded) until the
      realised sample does show an apparent Small > Mid > Large gradient, so
      there is something for the battery to be fooled by.  W3 must FAIL here.

Everything is seeded and every random procedure takes its seed as a parameter.
No network calls, no model calls.

Schema emitted is exactly analysis.window_battery.CHUNK_SCHEMA, in that order.

Run:
    python3 -m tests.fixtures_window_battery --out /tmp/chunk_level_results.csv --kind planted
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.window_battery import CHUNK_SCHEMA  # noqa: E402

TIERS = ("large", "mid", "small")
#: small is the hypothesised high end: step 0 = large, 1 = mid, 2 = small.
TIER_STEP = {"large": 0, "mid": 1, "small": 2}

EXACT_WORDS = 1200
MIN_TAIL_WORDS = 400
MAX_TAIL_WORDS = 1598      # schema's documented upper bound
MIN_WORDS = 403            # schema's documented lower bound

FAILED_SENTINEL = "API call failed after all retries."
SERVED_BY = "DeepInfra"    # the single pinned bf16 backend


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FixtureSpec:
    """
    Ground truth for one synthetic sample.  Every field is a knob a test can
    turn; nothing is read from the environment.
    """
    seed: int = 1
    n_firms_per_tier: int = 10
    #: Window-level mean shift PER TIER STEP in the embedding measure.
    #: 0.0 means no tier effect exists at all.
    tier_effect: float = 0.030
    base_embedding: float = 0.280
    firm_sd: float = 0.010          # firm random effect
    filing_sd: float = 0.006        # filing random effect
    window_sd: float = 0.030        # within-filing window noise
    #: Window-count regime. "realistic" reproduces the real sample's medians
    #: (6.5 / 6.5 / 4.0 MD&A windows per filing, min 2, max 25).  "fixed" uses
    #: `fixed_counts`, which is how the count-artifact fixture is built.
    count_regime: str = "realistic"
    mda_median_counts: dict[str, float] = field(
        default_factory=lambda: {"large": 6.5, "mid": 6.5, "small": 4.0})
    count_log_sd: dict[str, float] = field(
        default_factory=lambda: {"large": 0.60, "mid": 0.55, "small": 0.40})
    fixed_counts: dict[str, Sequence[int]] = field(
        default_factory=lambda: {"large": (12,), "mid": (7,), "small": (2,)})
    count_min: int = 2
    count_max: int = 25
    #: Numeric (secondary) measure: 0.0 reproduces the real sample's flat,
    #: non-monotone numeric gradient.
    numeric_tier_effect: float = 0.0
    numeric_base_spread: float = 6.0     # SD of persona scores on the 0-100 scale
    numeric_spread_sd: float = 2.0
    include_risk: bool = True
    risk_median_counts: dict[str, float] = field(
        default_factory=lambda: {"large": 7.0, "mid": 6.0, "small": 4.0})
    #: Share of filings with no Risk Factors windows (10-Qs incorporating the
    #: 10-K's risk factors by reference) - 10 of 59 in the real sample.
    n_filings_without_risk: int = 10
    #: Failed windows: any_failed=True and BOTH disagreement columns null.
    n_failed_windows: int = 0
    #: Rows with a parse error but no failure - must be KEPT.
    n_parse_error_windows: int = 0
    #: One firm contributes a single 10-Q (the HAYN case), giving 59 filings.
    single_filing_firm: bool = True
    #: Probability that a section's length is an exact multiple of the window
    #: size, so all of its windows are exactly 1,200 words.
    p_exact_multiple: float = 0.10


# ---------------------------------------------------------------------------
# Independent aggregation, used ONLY by the seed search
# ---------------------------------------------------------------------------

def apparent_firm_gradient(
    df: pd.DataFrame,
    section: str = "mda",
    measure: str = "embedding_disagreement",
) -> tuple[float, bool, dict[str, float]]:
    """
    window -> filing -> firm -> tier, written out independently of
    analysis.window_battery so the fixtures do not depend on the module under
    test for their own ground truth.  Returns (Small - Large, monotone, means).
    """
    w = df[(df["section"] == section) & (~df["any_failed"].astype(bool))].copy()
    w = w[pd.to_numeric(w[measure], errors="coerce").notna()]
    w["fk"] = w["ticker"] + "|" + w["filing_type"] + "|" + w["filing_date"].astype(str)
    filing = w.groupby(["ticker", "tier", "fk"])[measure].mean().reset_index()
    firm = filing.groupby(["ticker", "tier"])[measure].mean().reset_index()
    means = {t: float(firm.loc[firm["tier"] == t, measure].mean()) for t in TIERS}
    monotone = means["small"] > means["mid"] > means["large"]
    return means["small"] - means["large"], bool(monotone), means


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _firm_table(spec: FixtureSpec) -> list[dict[str, Any]]:
    firms: list[dict[str, Any]] = []
    for tier in TIERS:
        prefix = {"large": "LRG", "mid": "MID", "small": "SML"}[tier]
        for i in range(1, spec.n_firms_per_tier + 1):
            firms.append({"ticker": f"{prefix}{i:02d}", "tier": tier, "idx": i})
    return firms


def _filings_for(firm: dict[str, Any], spec: FixtureSpec) -> list[tuple[str, str]]:
    """(filing_type, filing_date) pairs.  Dates are deterministic, never Date.now()."""
    i = firm["idx"]
    tenk = f"2025{((i - 1) % 12) + 1:02d}{((i * 3) % 27) + 1:02d}"
    tenq = f"2026{((i + 3) % 12) + 1:02d}{((i * 5) % 27) + 1:02d}"
    if spec.single_filing_firm and firm["tier"] == "small" and i == spec.n_firms_per_tier:
        return [("10-Q", tenq)]          # the HAYN case: one filing only
    return [("10-K", tenk), ("10-Q", tenq)]


def _draw_count(rng: np.random.Generator, tier: str, spec: FixtureSpec,
                medians: dict[str, float]) -> int:
    if spec.count_regime == "fixed":
        choices = spec.fixed_counts[tier]
        return int(choices[rng.integers(0, len(choices))])
    n = int(np.round(np.exp(rng.normal(np.log(medians[tier]), spec.count_log_sd[tier]))))
    return int(np.clip(n, spec.count_min, spec.count_max))


def _word_counts(rng: np.random.Generator, n: int, spec: FixtureSpec) -> list[int]:
    """
    Reproduce analysis/chunking.chunk_section's output lengths: all windows are
    exactly `EXACT_WORDS` except the last, which carries the tail (and the tail
    merge, hence the 403-1,598 range).  Occasionally the section length is an
    exact multiple and every window is exactly 1,200.
    """
    if rng.random() < spec.p_exact_multiple:
        return [EXACT_WORDS] * n
    tail = int(rng.integers(MIN_WORDS, MAX_TAIL_WORDS + 1))
    if tail == EXACT_WORDS:
        tail += 1
    if n == 1:
        return [tail]
    return [EXACT_WORDS] * (n - 1) + [tail]


def _reasoning(persona: str, ticker: str, section: str, idx: int) -> str:
    return (f"{persona} lens on {ticker} {section} window {idx}: the disclosure is read "
            f"as {'ambiguous' if idx % 2 else 'directionally clear'} on margin trajectory "
            f"and liquidity, with the stated drivers only partly quantified.")


def generate_chunk_level_results(spec: FixtureSpec) -> pd.DataFrame:
    """
    Emit a synthetic chunk_level_results.csv frame with the exact real schema.

    Ground truth is `spec`: in particular `spec.tier_effect` is the true
    window-level mean difference per tier step in `embedding_disagreement`, so
    `tier_effect == 0` means NO tier effect exists and any gradient in the
    realised sample is sampling noise over unequal window counts.
    """
    rng = np.random.default_rng(spec.seed)
    firms = _firm_table(spec)

    # Which filings carry no Risk Factors windows (deterministic given the seed).
    all_filings: list[tuple[dict, str, str]] = []
    for firm in firms:
        for form, date in _filings_for(firm, spec):
            all_filings.append((firm, form, date))
    no_risk: set[int] = set()
    if spec.include_risk and spec.n_filings_without_risk > 0:
        q_idx = [i for i, (_f, form, _d) in enumerate(all_filings) if form == "10-Q"]
        n_pick = min(spec.n_filings_without_risk, len(q_idx))
        no_risk = set(rng.choice(q_idx, size=n_pick, replace=False).tolist())

    firm_re = {f["ticker"]: rng.normal(0.0, spec.firm_sd) for f in firms}
    firm_spread_re = {f["ticker"]: rng.normal(0.0, spec.numeric_spread_sd) for f in firms}

    rows: list[dict[str, Any]] = []
    for fi, (firm, form, date) in enumerate(all_filings):
        tier = firm["tier"]
        step = TIER_STEP[tier]
        filing_re = rng.normal(0.0, spec.filing_sd)
        filing_spread_re = rng.normal(0.0, spec.numeric_spread_sd)

        sections: list[tuple[str, dict[str, float]]] = [("mda", spec.mda_median_counts)]
        if spec.include_risk and fi not in no_risk:
            sections.append(("risk", spec.risk_median_counts))

        for section, medians in sections:
            n = _draw_count(rng, tier, spec, medians)
            words = _word_counts(rng, n, spec)
            # Risk Factors sits at a slightly different level from MD&A - the two
            # sections are never pooled (ANALYSIS_PLAN.md s2), so this only has to
            # be non-identical, not calibrated.
            section_shift = 0.0 if section == "mda" else 0.015

            for ci in range(n):
                mu = (spec.base_embedding + spec.tier_effect * step
                      + section_shift + firm_re[firm["ticker"]] + filing_re)
                emb = float(np.clip(mu + rng.normal(0.0, spec.window_sd), 0.001, 1.999))

                spread = max(
                    0.5,
                    spec.numeric_base_spread + spec.numeric_tier_effect * step
                    + firm_spread_re[firm["ticker"]] + filing_spread_re)
                centre = float(rng.uniform(25.0, 75.0))
                scores = [int(np.clip(round(centre + rng.normal(0.0, spread)), 0, 100))
                          for _ in range(3)]
                numeric = float(np.std(np.asarray(scores, dtype=float) / 10.0, ddof=0))

                rows.append({
                    "ticker": firm["ticker"],
                    "tier": tier,
                    "filing_type": form,
                    "filing_date": date,
                    "section": section,
                    "chunk_index": ci,
                    "n_in_section": n,
                    "word_count": int(words[ci]),
                    "numeric_disagreement": numeric,
                    "embedding_disagreement": emb,
                    "value_score": scores[0],
                    "sentiment_score": scores[1],
                    "fundamentals_score": scores[2],
                    "any_failed": False,
                    "any_parse_error": False,
                    "served_by": SERVED_BY,
                    "value_reasoning": _reasoning("value", firm["ticker"], section, ci),
                    "sentiment_reasoning": _reasoning("sentiment", firm["ticker"], section, ci),
                    "fundamentals_reasoning": _reasoning(
                        "fundamentals", firm["ticker"], section, ci),
                })

    df = pd.DataFrame(rows, columns=list(CHUNK_SCHEMA))

    # -- inject failed windows: any_failed=True, BOTH measures null ---------
    if spec.n_failed_windows > 0:
        # Only from filings that can spare a window and still clear the
        # MIN_USABLE_CHUNKS=2 gate, so the fixture stays analysable.
        mda = df.index[df["section"] == "mda"]
        sizes = df.loc[mda].groupby(
            ["ticker", "filing_type", "filing_date"])["chunk_index"].transform("size")
        eligible = np.asarray(mda)[(sizes.to_numpy() >= 4)]
        n_pick = min(spec.n_failed_windows, eligible.size)
        if n_pick < spec.n_failed_windows:
            print(f"[fixtures] WARNING: asked for {spec.n_failed_windows} failed windows, "
                  f"only {n_pick} eligible (filings must keep >= 2 usable windows)")
        pick = rng.choice(eligible, size=n_pick, replace=False)
        df.loc[pick, "any_failed"] = True
        df.loc[pick, ["numeric_disagreement", "embedding_disagreement"]] = np.nan
        df.loc[pick, ["value_score", "sentiment_score", "fundamentals_score"]] = np.nan
        for col in ("value_reasoning", "sentiment_reasoning", "fundamentals_reasoning"):
            df.loc[pick, col] = FAILED_SENTINEL

    # -- inject parse-error-but-not-failed rows: these must be KEPT ---------
    if spec.n_parse_error_windows > 0:
        ok = df.index[~df["any_failed"].astype(bool)].to_numpy()
        n_pick = min(spec.n_parse_error_windows, ok.size)
        df.loc[rng.choice(ok, size=n_pick, replace=False), "any_parse_error"] = True

    df.attrs["spec"] = spec
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# The two fixtures the battery has to tell apart
# ---------------------------------------------------------------------------

def planted_effect_fixture(seed: int = 11, tier_effect: float = 0.030, **over: Any
                           ) -> tuple[pd.DataFrame, FixtureSpec]:
    """
    A genuine window-level tier effect of `tier_effect` per step, on top of
    realistic tier-varying window counts.  The effect is present at equal k by
    construction, because it lives in every window's expected value, so W3 must
    recover it.
    """
    spec = FixtureSpec(seed=seed, tier_effect=tier_effect, count_regime="realistic", **over)
    return generate_chunk_level_results(spec), spec


def count_artifact_fixture(
    seed: int = 101,
    target_gradient: tuple[float, float] = (0.004, 0.014),
    require_monotone: bool = True,
    max_seed_offsets: int = 4_000,
    **over: Any,
) -> tuple[pd.DataFrame, FixtureSpec, dict[str, Any]]:
    """
    NO tier effect: `tier_effect = 0`, so every tier has the same expected
    window-level disagreement.  Only the window count differs - large-caps ~12
    windows per filing, mid ~7, small exactly 2 - so the variance of a filing
    mean differs by tier.

    The realised sample's apparent gradient is then pure noise, which is
    sometimes positive and sometimes negative.  A fixture that happened to come
    out negative would be no test at all (the battery would reject it for the
    wrong reason), so the seed is advanced deterministically from `seed` until
    the realised sample shows an apparent gradient inside `target_gradient` and,
    if `require_monotone`, apparent monotonicity Small > Mid > Large.  The
    offset that was used is returned, so the search is fully reproducible and
    nothing about it is hidden.

    Small-caps are given exactly 2 windows on purpose: it is the hardest case
    for W3, because a k = 2 draw from a 2-window filing IS the filing and adds
    no resampling noise on the small-cap side.
    """
    base = dict(
        tier_effect=0.0,
        count_regime="fixed",
        fixed_counts={"large": (11, 12, 13), "mid": (6, 7, 8), "small": (2,)},
        window_sd=0.090,       # wide enough that noise can manufacture a gradient
        firm_sd=0.0,           # no firm-level signal either
        filing_sd=0.0,
        risk_median_counts={"large": 11.0, "mid": 7.0, "small": 2.0},
    )
    base.update(over)

    lo, hi = target_gradient
    tried: list[dict[str, Any]] = []
    for off in range(max_seed_offsets):
        spec = FixtureSpec(seed=seed + off, **base)
        df = generate_chunk_level_results(spec)
        grad, monotone, means = apparent_firm_gradient(df)
        if lo <= grad <= hi and (monotone or not require_monotone):
            search = {
                "base_seed": seed, "seed_offset": off, "seed_used": spec.seed,
                "n_seeds_tried": off + 1, "apparent_gradient": grad,
                "apparent_monotone": monotone, "apparent_tier_means": means,
                "target_gradient": [lo, hi], "require_monotone": require_monotone,
                "true_tier_effect": 0.0,
                "note": ("seed advanced deterministically until the realised sample showed "
                         "an apparent gradient in range; the TRUE tier effect is exactly 0"),
            }
            return df, spec, search
        tried.append({"seed": spec.seed, "gradient": grad, "monotone": monotone})
    raise RuntimeError(
        f"count_artifact_fixture: no seed in [{seed}, {seed + max_seed_offsets}) produced an "
        f"apparent gradient in {target_gradient} (monotone={require_monotone}). "
        f"Last 5 attempts: {tried[-5:]}")


def write_fixture(df: pd.DataFrame, path: Path | str) -> Path:
    """Write the frame as a real chunk_level_results.csv (nulls as empty cells)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, columns=list(CHUNK_SCHEMA))
    return p


def describe(df: pd.DataFrame) -> str:
    mda = df[df["section"] == "mda"]
    per_filing = mda.groupby(
        ["ticker", "tier", "filing_type", "filing_date"])["chunk_index"].size().reset_index(
        name="n")
    lines = [
        f"rows (windows)        : {len(df)}",
        f"  mda                 : {len(mda)}",
        f"  risk                : {int((df['section'] == 'risk').sum())}",
        f"firms                 : {df['ticker'].nunique()}",
        f"filings               : {df.groupby(['ticker','filing_type','filing_date']).ngroups}",
        f"any_failed rows       : {int(df['any_failed'].astype(bool).sum())}",
        f"word_count range      : {int(df['word_count'].min())}-{int(df['word_count'].max())}",
        f"exact-{EXACT_WORDS}-word windows : "
        f"{int((df['word_count'] == EXACT_WORDS).sum())}",
        "mda windows per filing, median by tier:",
    ]
    for t in TIERS:
        sub = per_filing[per_filing["tier"] == t]["n"]
        lines.append(f"  {t:<6} n_filings={len(sub):<3} median={sub.median():<5} "
                     f"min={sub.min()} max={sub.max()}")
    grad, mono, means = apparent_firm_gradient(df)
    lines.append(f"apparent firm-level tier means (embedding): "
                 + "  ".join(f"{t}={means[t]:.4f}" for t in TIERS))
    lines.append(f"apparent gradient Small-Large = {grad:+.4f}   monotone = {mono}")
    gnum, mnum, mnums = apparent_firm_gradient(df, measure="numeric_disagreement")
    lines.append(f"numeric (secondary) tier means             : "
                 + "  ".join(f"{t}={mnums[t]:.4f}" for t in TIERS))
    lines.append(f"numeric gradient Small-Large  = {gnum:+.4f}   monotone = {mnum}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["planted", "artifact", "null"], default="planted")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="write the CSV here (optional)")
    args = ap.parse_args(argv)

    if args.kind == "planted":
        df, spec = planted_effect_fixture(seed=args.seed if args.seed is not None else 11)
        search = None
    elif args.kind == "artifact":
        df, spec, search = count_artifact_fixture(
            seed=args.seed if args.seed is not None else 101)
    else:
        spec = FixtureSpec(seed=args.seed if args.seed is not None else 7, tier_effect=0.0)
        df = generate_chunk_level_results(spec)
        search = None

    print(f"=== fixture: {args.kind} ===")
    print(f"seed = {spec.seed}   true tier_effect = {spec.tier_effect}")
    if search:
        print(f"seed search: offset {search['seed_offset']} of "
              f"{search['n_seeds_tried']} tried; target {search['target_gradient']}")
    print(describe(df))
    if args.out:
        print(f"written: {write_fixture(df, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
