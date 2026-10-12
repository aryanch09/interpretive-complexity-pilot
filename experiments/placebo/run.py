"""
E4 (content placebo) and E6 (familiarity / anonymisation) on one subset.

Both ask whether the measure is responding to what we claim it responds to, and
both are run on the same stratified window subset so they are directly comparable.

E4 — CONTENT PLACEBO. The objection: persona disagreement might track vocabulary
rather than argument. Three conditions on the same windows:
   real        the window as scored in the main run
   shuffled    the same sentences in random order — argument destroyed,
               vocabulary and length preserved exactly
   boilerplate ~1,200 words of forward-looking-statement disclaimer, which is
               real filing language carrying no firm-specific content
Expected if the measure reads content: real > shuffled > boilerplate. If
shuffled matches real, the measure is a bag of words and the construct claim
fails.

E6 — FAMILIARITY. The objection that cuts against this project's own result: all
three personas share one model's priors, and the model knows AAPL far better than
it knows a micro-cap. Strong priors could suppress disagreement for large firms —
and large-cap is exactly where the main run measures the LOWEST disagreement.
Scoring anonymised windows (firm names, executives and products stripped) and
comparing the delta BY TIER tests whether the tier pattern is a familiarity
artifact.

Cost: 3 personas x (shuffled + boilerplate + anonymised) per window.

Run:  python3 -m experiments.placebo.run --n-per-tier 12            (dry run)
      python3 -m experiments.placebo.run --n-per-tier 12 --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("tensorflow", None)

from analysis.trend_tests import DEFAULT_SEED, TIER_ORDER, bootstrap_ci
from config import Config

logger = logging.getLogger(__name__)
ROLES = ("value", "sentiment", "fundamentals")

# Real 10-K safe-harbour language: filing prose with no firm-specific content.
BOILERPLATE = """
This report contains forward-looking statements within the meaning of the Private
Securities Litigation Reform Act of 1995. Forward-looking statements may be
identified by words such as anticipates, believes, estimates, expects, intends,
plans, projects, targets, and similar expressions. These statements are based on
current expectations and assumptions that are subject to risks and uncertainties
that could cause actual results to differ materially from those expressed or
implied. Factors that could cause actual results to differ include, but are not
limited to, general economic and business conditions, changes in demand, the
competitive environment, the availability and cost of capital, changes in laws
and regulations, litigation and regulatory proceedings, disruptions to
information technology systems, the ability to attract and retain qualified
personnel, and other factors described from time to time in our filings with the
Securities and Exchange Commission. Readers are cautioned not to place undue
reliance on forward-looking statements, which speak only as of the date on which
they are made. We undertake no obligation to publicly update or revise any
forward-looking statement, whether as a result of new information, future events
or otherwise, except as required by law. The risks described are not the only
risks facing us. Additional risks and uncertainties not currently known to us or
that we currently deem immaterial may also materially adversely affect our
business, financial condition, results of operations, cash flows, or prospects.
"""


def _shuffle_sentences(text: str, rng: random.Random) -> str:
    """Destroy argument order, preserve vocabulary and length exactly."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p for p in parts if p.strip()]
    rng.shuffle(parts)
    return " ".join(parts)


def _boilerplate(target_words: int) -> str:
    base = " ".join(BOILERPLATE.split())
    out = []
    while len(" ".join(out).split()) < target_words:
        out.append(base)
    return " ".join(" ".join(out).split()[:target_words])


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


def select(chunk_df: pd.DataFrame, n_per_tier: int, seed: int) -> pd.DataFrame:
    d = chunk_df[(chunk_df.section == "mda")
                 & (~chunk_df.any_failed.astype(str).str.lower().isin(["true", "1"]))
                 & chunk_df.embedding_disagreement.notna()]
    rng = np.random.default_rng(seed)
    picks = []
    for tier in TIER_ORDER:
        sub = d[d.tier == tier]
        k = min(n_per_tier, len(sub))
        picks.append(sub.loc[rng.choice(sub.index.to_numpy(), size=k, replace=False)])
    return pd.concat(picks).reset_index(drop=True)


def run(config: Config, n_per_tier: int, seed: int, apply: bool) -> dict:
    ch = pd.read_csv(config.OUTPUT_DIR / "chunk_level_results.csv")
    sel = select(ch, n_per_tier, seed)
    n_calls = len(sel) * 3 * 3
    print(f"windows: {len(sel)}  ({dict(sel.tier.value_counts())})")
    print(f"conditions: shuffled, boilerplate, anonymised  ->  {n_calls} calls")
    print(f"estimated cost: ${n_calls*2700/1e6*0.03 + n_calls*450/1e6*0.14:.3f}")
    if not apply:
        print("\nDRY RUN — nothing called. Re-run with --apply.")
        return {"dry_run": True}

    from agents.prompts import AGENTS
    from agents.runner import run_agent
    from analysis.anonymize import anonymize_text
    from analysis.disagreement import compute_disagreement

    out_dir = config.OUTPUT_DIR / "experiments" / "placebo"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    cache = config.OUTPUT_DIR / "embedding_cache.json"

    def score(text: str) -> dict:
        outs = {r: run_agent(AGENTS[r]["prompt"], text, AGENTS[r]["name"], config)
                for r in ROLES}
        if any(outs[r].get("failed") for r in ROLES):
            return {"embedding_disagreement": None, "numeric_disagreement": None}
        return compute_disagreement(outs, config, cache_path=cache)

    # RESUME: skip windows already present in placebo.csv.
    done_path = out_dir / "placebo.csv"
    rows, done = [], set()
    if done_path.exists():
        prev = pd.read_csv(done_path)
        rows = prev.to_dict("records")
        done = {(x["ticker"], x["filing_type"], x["section"], int(x["chunk_index"]))
                for x in rows}
        print(f"resuming: {len(done)} window(s) already done, skipping them")

    for n, (_, row) in enumerate(sel.iterrows(), 1):
        if (row.ticker, row.filing_type, row.section, int(row.chunk_index)) in done:
            continue
        text = _window_text(config, row)
        if not text:
            continue
        print(f"[{n}/{len(sel)}] {row.ticker} {row.filing_type} "
              f"{row.section}#{row.chunk_index} ({row.tier})", flush=True)
        rec = {k: row[k] for k in ("ticker", "tier", "filing_type", "filing_date",
                                   "section", "chunk_index", "word_count")}
        rec["real"] = float(row.embedding_disagreement)

        sh = score(_shuffle_sentences(text, rng))
        rec["shuffled"] = sh.get("embedding_disagreement")

        bp = score(_boilerplate(len(text.split())))
        rec["boilerplate"] = bp.get("embedding_disagreement")

        try:
            anon, _removed = anonymize_text(text, row.ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("anonymise failed: %s", exc)
            anon = None
        rec["anonymised"] = score(anon).get("embedding_disagreement") if anon else None

        rows.append(rec)
        pd.DataFrame(rows).to_csv(out_dir / "placebo.csv", index=False)

    return summarise(pd.DataFrame(rows), out_dir, seed)


def summarise(df: pd.DataFrame, out_dir: Path, seed: int) -> dict:
    def ci(s):
        s = pd.Series(s).dropna().to_numpy()
        if len(s) < 3:
            return None
        b = bootstrap_ci(s, statistic=np.mean, n_boot=10000, seed=seed,
                         method="percentile")
        return {"mean": float(b.observed), "ci": [float(b.low), float(b.high)]}

    res: dict = {"seed": seed, "n_windows": int(len(df)), "conditions": {}}
    for c in ("real", "shuffled", "boilerplate", "anonymised"):
        if c in df:
            res["conditions"][c] = ci(df[c])
    for c in ("shuffled", "boilerplate", "anonymised"):
        if c in df:
            res[f"real_minus_{c}"] = ci(df["real"] - df[c])

    # E6 is a TIER question: does the anonymisation delta differ by tier?
    res["anonymisation_delta_by_tier"] = {}
    if "anonymised" in df:
        for t in TIER_ORDER:
            s = df[df.tier == t]
            if len(s) >= 3:
                res["anonymisation_delta_by_tier"][t] = ci(s["real"] - s["anonymised"])

    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str),
                                          encoding="utf-8")
    print("\n" + report(res))
    (out_dir / "report.txt").write_text(report(res), encoding="utf-8")
    return res


def report(r: dict) -> str:
    L = ["E4/E6 — Content placebo and familiarity control", "=" * 74, ""]
    if r.get("dry_run"):
        return "\n".join(L + ["dry run"])
    L.append(f"windows: {r['n_windows']}")
    L.append("")
    L.append("Mean embedding disagreement by condition (95% bootstrap CI)")
    L.append("-" * 74)
    for c, v in r["conditions"].items():
        if v:
            L.append(f"  {c:14s} {v['mean']:.4f}  [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]")
    L.append("")
    L.append("Differences from the real text")
    L.append("-" * 74)
    for c in ("shuffled", "boilerplate", "anonymised"):
        v = r.get(f"real_minus_{c}")
        if v:
            sig = "differs" if not (v["ci"][0] <= 0 <= v["ci"][1]) else "NOT distinguishable"
            L.append(f"  real - {c:12s} {v['mean']:+.4f} "
                     f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]  {sig}")
    L.append("")
    L.append("E4 reading: if 'real - shuffled' is not distinguishable from zero, the")
    L.append("  measure responds to vocabulary rather than argument, and the construct")
    L.append("  claim fails. Boilerplate is the floor: real filing language with no")
    L.append("  firm-specific content.")
    L.append("")
    L.append("E6 — anonymisation delta by tier")
    L.append("-" * 74)
    for t, v in (r.get("anonymisation_delta_by_tier") or {}).items():
        if v:
            L.append(f"  {t:6s} {v['mean']:+.4f}  [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
    L.append("")
    L.append("E6 reading: all three personas share one model's priors, and the model")
    L.append("  knows large-caps far better than micro-caps. If anonymisation RAISES")
    L.append("  disagreement more for large-caps, familiarity was suppressing it there —")
    L.append("  which is the direction that would matter, because large-cap is where the")
    L.append("  main run measures the lowest disagreement.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-tier", type=int, default=12)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--agent-sleep", type=float, default=0.5)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    config = Config()
    config.API_PROVIDER = a.provider
    config.apply_provider()
    config.AGENT_SLEEP_SECONDS = a.agent_sleep
    run(config, a.n_per_tier, a.seed, a.apply)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
