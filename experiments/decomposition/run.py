"""
S1c — Decomposing the truncation contrast.

WHY THIS EXISTS. S1b compared "first 2,000 words scored as one unit" against
"complete sections scored in windows and averaged" and found a significant size
gradient in the first and none in the second. Read carelessly that says
truncation changes the answer. It does not, because the two arms differ in THREE
ways at once: how much text was read, how large the scoring unit was, and
whether the result was a single draw or a mean over several.

The existing data already shows this matters. S1a reads about 2,400 words as two
windows and averages them, giving a gradient of +0.0204. S1b reads 2,000 words as
one block, giving +0.0571. Almost the same coverage, nearly triple the gradient.
So most of the S1b effect is NOT coverage.

THE DESIGN. A 2x2 over {truncated, complete} x {single unit, windowed-and-averaged}:

                     single unit            windowed + averaged
  truncated (2kw)    S1b          (have)    S1a k=2          (have)
  complete           S1c          (THIS)    main run         (have)

With the fourth cell:
  S1b vs S1c  = pure COVERAGE effect, scoring unit held constant
  S1c vs main = pure AGGREGATION effect, coverage held constant

A HONEST CAVEAT, stated before the result. S1c is the design this project
deliberately rejected: feeding a whole section in one call. A 20B-class model
degrades on long-context retrieval, and that degradation is worst for the longest
sections, which belong to the largest firms. So S1c carries its own size-
correlated bias in the opposite direction. It is run to MEASURE the aggregation
effect, not because it is a design anyone should use. The decomposition is
therefore a bound, not a clean causal split, and must be reported that way.

Cost: 3 calls per filing (59 filings = 177 calls), with large inputs.

Run:  python3 -m experiments.decomposition.run            (dry run)
      python3 -m experiments.decomposition.run --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("tensorflow", None)

from analysis.trend_tests import (
    DEFAULT_SEED,
    TIER_ORDER,
    bootstrap_ci,
    jonckheere_terpstra_labeled,
)
from config import Config

logger = logging.getLogger(__name__)
ROLES = ("value", "sentiment", "fundamentals")


RETIRED_CAP = 2000


def _mda(config: Config, ticker: str, form: str, date: str,
         cap: int | None = None) -> str | None:
    t = _complete_mda(config, ticker, form, date)
    if t is None:
        return None
    return " ".join(t.split()[:cap]) if cap else t


def _complete_mda(config: Config, ticker: str, form: str, date: str) -> str | None:
    from data.edgar import _html_to_text, extract_sections_dict

    path = config.FILINGS_DIR / f"{ticker}_{form}_{str(date).replace('-','')}.txt"
    if not path.exists():
        return None
    sections, _ = extract_sections_dict(
        _html_to_text(path.read_text(encoding="utf-8", errors="ignore")), form)
    return sections.get("mda") or None


def run(config: Config, apply: bool, seed: int) -> dict:
    man = pd.read_csv(Path(config.BASE_DIR) / "data" / "sample_manifest.csv")
    res = pd.read_csv(config.OUTPUT_DIR / "results.csv")[
        ["ticker", "filing_type", "filing_date", "tier"]].copy()
    for f in (man, res):
        f["filing_date"] = f.filing_date.astype(str).str.replace("-", "", regex=False)
    fil = man.merge(res, on=["ticker", "filing_type", "filing_date"], how="left")

    out_dir = config.OUTPUT_DIR / "experiments" / "decomposition"
    out_dir.mkdir(parents=True, exist_ok=True)
    arm = getattr(config, "_ARM", "complete")
    cap = RETIRED_CAP if arm == "truncated" else None
    done_path = out_dir / f"{arm}_single_unit.csv"

    print(f"filings: {len(fil)}   calls: {len(fil)*3}")
    if not apply:
        # Show the input sizes so the long-context caveat is quantified up front.
        sizes = []
        for r in fil.itertuples():
            t = _complete_mda(config, r.ticker, r.filing_type, r.filing_date)
            if t:
                sizes.append(len(t.split()))
        if sizes:
            print(f"MD&A words per call: median {int(np.median(sizes)):,}  "
                  f"max {max(sizes):,}  (~{max(sizes)*1.33/1000:.0f}k tokens worst case)")
        print("\nDRY RUN — nothing called. Re-run with --apply.")
        return {"dry_run": True}

    from agents.prompts import AGENTS
    from agents.runner import run_agent
    from analysis.disagreement import compute_disagreement

    rows, done = [], set()
    if done_path.exists():
        prev = pd.read_csv(done_path)
        prev["filing_date"] = prev.filing_date.astype(str).str.replace("-", "", regex=False)
        rows = prev.to_dict("records")
        done = {(x["ticker"], x["filing_type"], str(x["filing_date"])) for x in rows}
        print(f"resuming: {len(done)} filing(s) already scored")

    for n, r in enumerate(fil.itertuples(), 1):
        if (r.ticker, r.filing_type, str(r.filing_date)) in done:
            continue
        text = _mda(config, r.ticker, r.filing_type, r.filing_date, cap)
        if not text:
            continue
        print(f"[{n}/{len(fil)}] {arm} {r.ticker} {r.filing_type} "
              f"({len(text.split()):,}w)", flush=True)
        outs = {role: run_agent(AGENTS[role]["prompt"], text,
                                AGENTS[role]["name"], config) for role in ROLES}
        failed = any(outs[x].get("failed") for x in ROLES)
        dis = compute_disagreement(
            outs, config, cache_path=config.OUTPUT_DIR / "embedding_cache.json")
        rows.append({
            "ticker": r.ticker, "tier": r.tier, "filing_type": r.filing_type,
            "filing_date": r.filing_date, "words_read": len(text.split()),
            "any_failed": failed,
            "embedding_disagreement": dis.get("embedding_disagreement"),
            "numeric_disagreement": dis.get("numeric_disagreement"),
        })
        pd.DataFrame(rows).to_csv(done_path, index=False)

    return summarise(pd.DataFrame(rows), config, out_dir, seed)


def _firm_gradient(df: pd.DataFrame, col: str = "embedding_disagreement") -> dict:
    clean = df[(~df.any_failed.astype(str).str.lower().isin(["true", "1"]))
               & df[col].notna()]
    firm = (clean.groupby(["ticker", "tier"], as_index=False)[col].mean()
                 .rename(columns={col: "firm_mean"}))
    means = {t: float(firm[firm.tier == t].firm_mean.mean()) for t in TIER_ORDER}
    jt = jonckheere_terpstra_labeled(firm.firm_mean.to_numpy(), firm.tier.to_numpy(),
                                     group_order=TIER_ORDER, alternative="increasing")
    return {"tier_means": means,
            "gradient": means["small"] - means["large"],
            "monotone": bool(means["large"] < means["mid"] < means["small"]),
            "jt_p": float(jt.p_value), "n_firms": int(len(firm)),
            "_firm": firm}


def summarise(df: pd.DataFrame, config: Config, out_dir: Path, seed: int) -> dict:
    from analysis.trend_tests import aggregate_to_firm

    s1c = _firm_gradient(df)

    # The other three cells.
    s1b_raw = pd.read_csv(
        config.OUTPUT_DIR / "experiments" / "truncation_replication" /
        "truncated_scoring.csv")
    s1b = _firm_gradient(s1b_raw)

    ch = pd.read_csv(config.OUTPUT_DIR / "chunk_level_results.csv")
    main_firm = aggregate_to_firm(ch, section="mda",
                                  measure="embedding_disagreement")
    mmeans = {t: float(main_firm[main_firm.tier == t].firm_mean.mean())
              for t in TIER_ORDER}
    main = {"tier_means": mmeans, "gradient": mmeans["small"] - mmeans["large"],
            "_firm": main_firm.rename(columns={"firm_mean": "firm_mean"})}

    g = pd.read_csv(config.OUTPUT_DIR / "experiments" / "truncation_contrast" /
                    "truncation_contrast_gradients.csv")
    g = g[(g.measure == "embedding_disagreement") & (g.variant == "inclusive")
          & (g.k.astype(str) == "2")]
    s1a_k2 = float(g.gradient_small_minus_large.iloc[0]) if len(g) else None

    def paired(a: pd.DataFrame, b: pd.DataFrame) -> dict:
        m = a[["ticker", "firm_mean"]].merge(b[["ticker", "firm_mean"]],
                                             on="ticker", suffixes=("_a", "_b"))
        d = (m.firm_mean_a - m.firm_mean_b).to_numpy()
        ci = bootstrap_ci(d, statistic=np.mean, n_boot=10000, seed=seed,
                          method="percentile")
        return {"mean": float(ci.observed), "ci": [float(ci.low), float(ci.high)],
                "n_firms": int(len(m)),
                "differs": bool(not (ci.low <= 0 <= ci.high))}

    res = {
        "seed": seed,
        "cells": {
            "truncated_single_unit_S1b": {k: v for k, v in s1b.items() if k != "_firm"},
            "complete_single_unit_S1c": {k: v for k, v in s1c.items() if k != "_firm"},
            "complete_windowed_main": {"tier_means": mmeans, "gradient": main["gradient"]},
            "truncated_windowed_S1a_k2_gradient": s1a_k2,
        },
        "coverage_effect_S1b_minus_S1c": {
            "gradient_shift": s1b["gradient"] - s1c["gradient"],
            "paired": paired(s1b["_firm"], s1c["_firm"]),
            "note": "scoring unit held constant (both single-unit); only coverage differs",
        },
        "aggregation_effect_S1c_minus_main": {
            "gradient_shift": s1c["gradient"] - main["gradient"],
            "paired": paired(s1c["_firm"], main["_firm"]),
            "note": "coverage held constant (both complete); only the scoring unit "
                    "and the averaging differ",
        },
        "total_S1b_minus_main": {
            "gradient_shift": s1b["gradient"] - main["gradient"],
            "paired": paired(s1b["_firm"], main["_firm"]),
        },
        "caveat": (
            "S1c scores a complete MD&A in one call, which is the design this "
            "project rejected because long-context degradation is worst for the "
            "longest sections and therefore for the largest firms. S1c carries a "
            "size-correlated bias of its own, so this decomposition bounds the two "
            "effects rather than cleanly identifying them."
        ),
    }
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str),
                                          encoding="utf-8")
    txt = report(res)
    (out_dir / "report.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    return res


def report(r: dict) -> str:
    c = r["cells"]
    L = ["S1c — Decomposing the truncation contrast", "=" * 74, ""]
    L.append("2x2 over coverage x scoring unit; firm-level gradient (small - large)")
    L.append("-" * 74)
    L.append(f"  {'':22s} {'single unit':>14s} {'windowed+avg':>14s}")
    L.append(f"  {'truncated (2,000w)':22s} "
             f"{c['truncated_single_unit_S1b']['gradient']:+14.4f} "
             f"{(c['truncated_windowed_S1a_k2_gradient'] or float('nan')):+14.4f}")
    L.append(f"  {'complete':22s} "
             f"{c['complete_single_unit_S1c']['gradient']:+14.4f} "
             f"{c['complete_windowed_main']['gradient']:+14.4f}")
    L.append("")
    for key, label in (("coverage_effect_S1b_minus_S1c", "COVERAGE effect"),
                       ("aggregation_effect_S1c_minus_main", "AGGREGATION effect"),
                       ("total_S1b_minus_main", "TOTAL (S1b vs main)")):
        b = r[key]
        p = b["paired"]
        L.append(f"  {label:20s} gradient shift {b['gradient_shift']:+.4f}   "
                 f"paired {p['mean']:+.4f} [{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]  "
                 f"{'differs' if p['differs'] else 'spans zero'}")
    L.append("")
    L.append(f"  S1c alone: monotone {c['complete_single_unit_S1c']['monotone']}, "
             f"JT p = {c['complete_single_unit_S1c']['jt_p']:.4f}")
    L.append("")
    L.append("  " + r["caveat"])
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--agent-sleep", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--arm", choices=["complete", "truncated"], default="complete",
                    help="complete MD&A, or the retired 2,000-word cap. Both are "
                         "scored as a SINGLE unit; running both on one backend is "
                         "what makes the coverage contrast internally valid.")
    ap.add_argument("--backend", default=None,
                    help="pin a specific OpenRouter backend for BOTH arms")
    a = ap.parse_args(argv)
    config = Config()
    config.API_PROVIDER = a.provider
    config.apply_provider()
    config.AGENT_SLEEP_SECONDS = a.agent_sleep
    config._ARM = a.arm
    if a.backend:
        config.PROVIDERS[config.API_PROVIDER]["extra_body"] = {
            "provider": {"order": [a.backend], "allow_fallbacks": False,
                         "quantizations": ["bf16"]}}
        print(f"backend pinned to {a.backend} (bf16) for this arm")
    run(config, a.apply, a.seed)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
