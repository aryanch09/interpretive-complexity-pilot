"""
S1b — Exact replication of the retired practice.

S1a simulates truncation by averaging only the first k windows of a filing that
was scored window-by-window. That is informative but not the same thing: a
persona scoring window 1 of 25 never sees the rest, whereas a persona given a
truncated DOCUMENT reads 2,000 words as a whole.

This experiment removes that caveat. Each filing's first 2,000 words of MD&A --
exactly what the retired cap produced -- is scored as a SINGLE unit by the three
personas, and the tier gradient from that is compared with the complete-section
gradient from the main run. Same model, same filings, same personas; the only
thing that differs is how much of the document the model read, and whether it
read it whole.

If the two gradients differ, truncation changes the answer. That is the paper's
second contribution, and this is the version of it with no simulation caveat.

Cost: 3 calls per filing (59 filings = 177 calls).

Run:  python3 -m experiments.truncation_replication.run            (dry run)
      python3 -m experiments.truncation_replication.run --apply
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

RETIRED_CAP = 2000
ROLES = ("value", "sentiment", "fundamentals")


def _filings(config: Config) -> pd.DataFrame:
    man = pd.read_csv(Path(config.BASE_DIR) / "data" / "sample_manifest.csv")
    res = pd.read_csv(config.OUTPUT_DIR / "results.csv")[
        ["ticker", "filing_type", "filing_date", "tier"]].copy()
    res["filing_date"] = res.filing_date.astype(str).str.replace("-", "", regex=False)
    man["filing_date"] = man.filing_date.astype(str).str.replace("-", "", regex=False)
    return man.merge(res, on=["ticker", "filing_type", "filing_date"], how="left")


def _truncated_mda(config: Config, ticker: str, form: str, date: str) -> str | None:
    """The first 2,000 words of MD&A — exactly what the retired cap produced."""
    from data.edgar import _html_to_text, extract_sections_dict

    path = config.FILINGS_DIR / f"{ticker}_{form}_{date}.txt"
    if not path.exists():
        return None
    sections, _ = extract_sections_dict(
        _html_to_text(path.read_text(encoding="utf-8", errors="ignore")), form)
    mda = (sections.get("mda") or "").split()
    return " ".join(mda[:RETIRED_CAP]) if mda else None


def run(config: Config, apply: bool, seed: int) -> dict:
    fil = _filings(config)
    print(f"filings: {len(fil)}   calls required: {len(fil) * 3}")
    est = len(fil) * 3
    print(f"estimated cost: ${est*2700/1e6*0.03 + est*450/1e6*0.14:.3f}")
    if not apply:
        print("\nDRY RUN — nothing called. Re-run with --apply.")
        return {"dry_run": True}

    from agents.prompts import AGENTS
    from agents.runner import run_agent
    from analysis.disagreement import compute_disagreement

    out_dir = config.OUTPUT_DIR / "experiments" / "truncation_replication"
    out_dir.mkdir(parents=True, exist_ok=True)

    # RESUME: anything already in truncated_scoring.csv is not re-scored. Calls
    # are slow and rate-limited, so an interrupted run must cost only the filing
    # it was mid-way through, never the whole set.
    done_path = out_dir / "truncated_scoring.csv"
    rows = []
    done = set()
    if done_path.exists():
        prev = pd.read_csv(done_path)
        prev["filing_date"] = prev.filing_date.astype(str).str.replace("-", "", regex=False)
        rows = prev.to_dict("records")
        done = {(x["ticker"], x["filing_type"], str(x["filing_date"])) for x in rows}
        print(f"resuming: {len(done)} filing(s) already scored, skipping them")

    for n, r in enumerate(fil.itertuples(), 1):
        if (r.ticker, r.filing_type, str(r.filing_date)) in done:
            continue
        text = _truncated_mda(config, r.ticker, r.filing_type, r.filing_date)
        if not text:
            logger.warning("no MD&A for %s %s", r.ticker, r.filing_type)
            continue
        print(f"[{n}/{len(fil)}] {r.ticker} {r.filing_type} "
              f"({len(text.split())}w)", flush=True)
        outs = {}
        for role in ROLES:
            outs[role] = run_agent(AGENTS[role]["prompt"], text,
                                   AGENTS[role]["name"], config)
        failed = any(outs[x].get("failed") for x in ROLES)
        dis = compute_disagreement(
            outs, config, cache_path=config.OUTPUT_DIR / "embedding_cache.json")
        rows.append({
            "ticker": r.ticker, "tier": r.tier, "filing_type": r.filing_type,
            "filing_date": r.filing_date, "words_read": len(text.split()),
            "any_failed": failed,
            "numeric_disagreement": dis.get("numeric_disagreement"),
            "embedding_disagreement": dis.get("embedding_disagreement"),
            **{f"{x}_reasoning": outs[x].get("reasoning") for x in ROLES},
        })
        pd.DataFrame(rows).to_csv(out_dir / "truncated_scoring.csv", index=False)

    df = pd.DataFrame(rows)
    return summarise(df, config, out_dir, seed)


def summarise(df: pd.DataFrame, config: Config, out_dir: Path, seed: int) -> dict:
    from analysis.trend_tests import aggregate_to_firm

    clean = df[(~df.any_failed) & df.embedding_disagreement.notna()]
    res: dict = {"seed": seed, "n_filings_scored": int(len(df)),
                 "n_filings_clean": int(len(clean)),
                 "cap_words": RETIRED_CAP}

    # Truncated arm: filing -> firm, same chain as the main analysis.
    firm = (clean.groupby(["ticker", "tier"], as_index=False)
                 .embedding_disagreement.mean()
                 .rename(columns={"embedding_disagreement": "firm_mean"}))
    means = {t: float(firm[firm.tier == t].firm_mean.mean()) for t in TIER_ORDER}
    jt = jonckheere_terpstra_labeled(firm.firm_mean.to_numpy(), firm.tier.to_numpy(),
                                     group_order=TIER_ORDER, alternative="increasing")
    grad = means["small"] - means["large"]
    res["truncated"] = {
        "tier_means": means, "gradient_small_minus_large": grad,
        "monotone": bool(means["large"] < means["mid"] < means["small"]),
        "jt_p_one_sided": float(jt.p_value), "jt_z": float(jt.z),
        "n_firms": int(len(firm)),
    }

    # Complete-section arm, from the main run.
    ch = pd.read_csv(config.OUTPUT_DIR / "chunk_level_results.csv")
    cf = aggregate_to_firm(ch, section="mda", measure="embedding_disagreement")
    cmeans = {t: float(cf[cf.tier == t].firm_mean.mean()) for t in TIER_ORDER}
    cjt = jonckheere_terpstra_labeled(cf.firm_mean.to_numpy(), cf.tier.to_numpy(),
                                      group_order=TIER_ORDER, alternative="increasing")
    cgrad = cmeans["small"] - cmeans["large"]
    res["complete"] = {
        "tier_means": cmeans, "gradient_small_minus_large": cgrad,
        "monotone": bool(cmeans["large"] < cmeans["mid"] < cmeans["small"]),
        "jt_p_one_sided": float(cjt.p_value), "n_firms": int(len(cf)),
    }

    # Paired difference, firm by firm — the quantity the claim is about.
    paired = firm.merge(cf[["ticker", "firm_mean"]], on="ticker",
                        suffixes=("_trunc", "_complete"))
    paired["delta"] = paired.firm_mean_trunc - paired.firm_mean_complete
    ci = bootstrap_ci(paired.delta.to_numpy(), statistic=np.mean,
                      n_boot=10000, seed=seed, method="percentile")
    res["paired_delta"] = {
        "n_firms": int(len(paired)),
        "mean_truncated_minus_complete": float(ci.observed),
        "ci": [float(ci.low), float(ci.high)],
        "differs_from_zero": bool(not (ci.low <= 0 <= ci.high)),
    }
    res["gradient_shift"] = float(grad - cgrad)
    paired.to_csv(out_dir / "paired_firm_means.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str),
                                          encoding="utf-8")
    print("\n" + report(res))
    (out_dir / "report.txt").write_text(report(res), encoding="utf-8")
    return res


def report(r: dict) -> str:
    L = ["S1b — Exact replication of the retired 2,000-word cap", "=" * 74, ""]
    if r.get("dry_run"):
        return "\n".join(L + ["dry run"])
    t, c = r["truncated"], r["complete"]
    L.append(f"Filings scored under the cap: {r['n_filings_clean']}/{r['n_filings_scored']}")
    L.append("")
    L.append(f"{'arm':12s} {'large':>8s} {'mid':>8s} {'small':>8s} "
             f"{'gradient':>10s} {'monotone':>9s} {'JT p':>7s}")
    for name, a in (("truncated", t), ("complete", c)):
        m = a["tier_means"]
        L.append(f"{name:12s} {m['large']:8.4f} {m['mid']:8.4f} {m['small']:8.4f} "
                 f"{a['gradient_small_minus_large']:+10.4f} "
                 f"{str(a['monotone']):>9s} {a['jt_p_one_sided']:7.4f}")
    L.append("")
    L.append(f"Gradient shift (truncated - complete): {r['gradient_shift']:+.4f}")
    pd_ = r["paired_delta"]
    L.append(f"Paired firm-level difference: {pd_['mean_truncated_minus_complete']:+.4f} "
             f"[{pd_['ci'][0]:+.4f}, {pd_['ci'][1]:+.4f}]  "
             f"n = {pd_['n_firms']} firms")
    L.append(f"  differs from zero: {pd_['differs_from_zero']}")
    L.append("")
    L.append("  This is the same model, the same filings and the same personas. The")
    L.append("  only difference is how much of the document was read, and whether it")
    L.append("  was read whole. Unlike S1a this involves no simulation: the truncated")
    L.append("  arm really was scored on 2,000 words as a single unit.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--agent-sleep", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    a = ap.parse_args(argv)
    config = Config()
    config.API_PROVIDER = a.provider
    config.apply_provider()
    config.AGENT_SLEEP_SECONDS = a.agent_sleep
    run(config, a.apply, a.seed)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
