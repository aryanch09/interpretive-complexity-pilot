"""
E7 — Cross-model replication of the placebo result.

E4 found the result that decides what this measure is: on gpt-oss-20b, persona
disagreement on content-free BOILERPLATE is HIGHER than on real MD&A text, and
sentence-shuffled text is indistinguishable from real. That says the measure
responds to informational emptiness rather than interpretive difficulty.

The obvious objection is that this is a quirk of one 20B model. This experiment
tests it on a different model family at the same precision. If the inversion
replicates, it is a property of the MEASURE -- three prompted lenses reading the
same text -- and not of the model, which makes it a finding about the method
rather than about gpt-oss-20b.

Design discipline, unchanged from the main run: one model, one backend, one
precision, fallbacks off, three personas, identical prompts, identical windows,
identical embedding recipe. The ONLY thing that differs from E4 is the scoring
model.

This is a labelled robustness check. It is never merged into the main results,
and it gets its own output directory.

Run:  python3 -m experiments.cross_model.run --n-per-tier 6          (dry run)
      python3 -m experiments.cross_model.run --n-per-tier 6 --apply
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("tensorflow", None)

from analysis.trend_tests import DEFAULT_SEED, TIER_ORDER, bootstrap_ci
from config import Config
from experiments.placebo.run import (
    _boilerplate,
    _shuffle_sentences,
    _window_text,
    select,
)

logger = logging.getLogger(__name__)
ROLES = ("value", "sentiment", "fundamentals")

CROSS_MODEL = "meta-llama/llama-3.3-70b-instruct"
CROSS_BACKEND = "Novita"          # the one endpoint serving this model at bf16


def run(config: Config, n_per_tier: int, seed: int, apply: bool) -> dict:
    ch = pd.read_csv(config.OUTPUT_DIR / "chunk_level_results.csv")
    sel = select(ch, n_per_tier, seed)
    n_calls = len(sel) * 3 * 3   # real, shuffled, boilerplate x 3 personas
    print(f"cross model  : {CROSS_MODEL}  (backend {CROSS_BACKEND}, bf16)")
    print(f"windows      : {len(sel)}  ({dict(sel.tier.value_counts())})")
    print(f"conditions   : real, shuffled, boilerplate")
    print(f"calls        : {n_calls}")
    print(f"est. cost    : ${n_calls*2700/1e6*0.10 + n_calls*450/1e6*0.30:.3f}")
    if not apply:
        print("\nDRY RUN — nothing called. Re-run with --apply.")
        return {"dry_run": True}

    # Point the whole client at the cross-check model, pinned exactly as the
    # main run pins its own: one backend, bf16, no fallbacks.
    config.MODEL_NAME = CROSS_MODEL
    config.PROVIDERS[config.API_PROVIDER]["model"] = CROSS_MODEL
    config.PROVIDERS[config.API_PROVIDER]["extra_body"] = {
        "provider": {"order": [CROSS_BACKEND], "allow_fallbacks": False,
                     "quantizations": ["bf16"]}
    }

    from agents.prompts import AGENTS
    from agents.runner import run_agent
    from analysis.disagreement import compute_disagreement

    out_dir = config.OUTPUT_DIR / "experiments" / "cross_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "cross_model.csv"
    rng = random.Random(seed)
    # A separate embedding cache is unnecessary -- the encoder is unchanged and
    # keyed by text hash -- but the REASONING text differs, so entries are new.
    cache = config.OUTPUT_DIR / "embedding_cache.json"

    rows, done = [], set()
    if done_path.exists():
        prev = pd.read_csv(done_path)
        rows = prev.to_dict("records")
        done = {(x["ticker"], x["filing_type"], x["section"], int(x["chunk_index"]))
                for x in rows}
        print(f"resuming: {len(done)} window(s) already done")

    def score(text: str) -> tuple[float | None, str]:
        outs = {r: run_agent(AGENTS[r]["prompt"], text, AGENTS[r]["name"], config)
                for r in ROLES}
        served = sorted({str(outs[r].get("served_by")) for r in ROLES
                         if outs[r].get("served_by")})
        if any(outs[r].get("failed") for r in ROLES):
            return None, "|".join(served)
        d = compute_disagreement(outs, config, cache_path=cache)
        return d.get("embedding_disagreement"), "|".join(served)

    for n, (_, row) in enumerate(sel.iterrows(), 1):
        key = (row.ticker, row.filing_type, row.section, int(row.chunk_index))
        if key in done:
            continue
        text = _window_text(config, row)
        if not text:
            continue
        print(f"[{n}/{len(sel)}] {row.ticker} {row.filing_type} "
              f"{row.section}#{row.chunk_index} ({row.tier})", flush=True)

        real, served = score(text)
        shuf, _ = score(_shuffle_sentences(text, rng))
        boil, _ = score(_boilerplate(len(text.split())))
        rows.append({
            "ticker": row.ticker, "tier": row.tier, "filing_type": row.filing_type,
            "filing_date": row.filing_date, "section": row.section,
            "chunk_index": int(row.chunk_index), "word_count": row.word_count,
            "real": real, "shuffled": shuf, "boilerplate": boil,
            "served_by": served,
            # The same window's value under the MAIN model, for a paired view.
            "real_main_model": float(row.embedding_disagreement),
        })
        pd.DataFrame(rows).to_csv(done_path, index=False)

    return summarise(pd.DataFrame(rows), out_dir, seed)


def summarise(df: pd.DataFrame, out_dir: Path, seed: int) -> dict:
    def ci(s):
        s = pd.Series(s).dropna().to_numpy()
        if len(s) < 3:
            return None
        b = bootstrap_ci(s, statistic=np.mean, n_boot=10000, seed=seed,
                         method="percentile")
        return {"mean": float(b.observed), "ci": [float(b.low), float(b.high)]}

    res: dict = {
        "seed": seed, "cross_model": CROSS_MODEL, "backend": CROSS_BACKEND,
        "n_windows": int(len(df)),
        "served_by": sorted(set(df.served_by.dropna())) if "served_by" in df else [],
        "conditions": {c: ci(df[c]) for c in ("real", "shuffled", "boilerplate")
                       if c in df},
        "main_model_real": ci(df["real_main_model"]) if "real_main_model" in df else None,
    }
    for c in ("shuffled", "boilerplate"):
        if c in df:
            res[f"real_minus_{c}"] = ci(df["real"] - df[c])
    # A single boolean is too crude to report. Replication has two parts, and
    # they can disagree: the ORDERING can hold while the contrast is merely
    # underpowered. Both are recorded so the paper can state which is which.
    conds = res.get("conditions") or {}
    means = {k: (v or {}).get("mean") for k, v in conds.items()}
    res["ordering_matches"] = bool(
        all(means.get(k) is not None for k in ("real", "shuffled", "boilerplate"))
        and means["real"] < means["shuffled"] < means["boilerplate"]
    )
    rmb = res.get("real_minus_boilerplate") or {}
    res["boilerplate_contrast_significant"] = bool(
        rmb.get("ci") and rmb["ci"][1] < 0)
    res["inversion_replicates"] = res["ordering_matches"]
    res["replication_note"] = (
        "ordering_matches is the directional claim; "
        "boilerplate_contrast_significant is whether this arm alone clears zero. "
        "At a smaller n the second can be False while the first is True, which is "
        "an underpowered contrast rather than a failed replication."
    )
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str),
                                          encoding="utf-8")
    txt = report(res)
    (out_dir / "report.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)
    return res


def report(r: dict) -> str:
    L = ["E7 — Cross-model replication of the placebo result", "=" * 74, ""]
    if r.get("dry_run"):
        return "\n".join(L + ["dry run"])
    L.append(f"cross model : {r['cross_model']}")
    L.append(f"served by   : {r['served_by']}")
    L.append(f"windows     : {r['n_windows']}")
    L.append("")
    L.append("Mean embedding disagreement (95% bootstrap CI)")
    L.append("-" * 74)
    for c, v in r["conditions"].items():
        if v:
            L.append(f"  {c:14s} {v['mean']:.4f}  [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]")
    mm = r.get("main_model_real")
    if mm:
        L.append(f"  {'(main model,':14s} real text: {mm['mean']:.4f} "
                 f"[{mm['ci'][0]:.4f}, {mm['ci'][1]:.4f}])")
    L.append("")
    for c in ("shuffled", "boilerplate"):
        v = r.get(f"real_minus_{c}")
        if v:
            verdict = ("differs" if not (v["ci"][0] <= 0 <= v["ci"][1])
                       else "NOT distinguishable")
            L.append(f"  real - {c:12s} {v['mean']:+.4f} "
                     f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]  {verdict}")
    L.append("")
    L.append(f"  ordering real < shuffled < boilerplate matches: "
             f"{r.get('ordering_matches')}")
    L.append(f"  boilerplate contrast clears zero on this arm : "
             f"{r.get('boilerplate_contrast_significant')}")
    L.append("")
    L.append("  If it replicates on a second model family at the same precision, the")
    L.append("  inversion is a property of the measure -- three prompted lenses reading")
    L.append("  one text -- rather than a quirk of the model used for the main run.")
    L.append("  Labelled robustness check; never merged into the main results.")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-tier", type=int, default=6)
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
