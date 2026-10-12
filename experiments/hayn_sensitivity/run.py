"""Pre-registered HAYN-excluded sensitivity (ANALYSIS_PLAN.md section 1).

HAYN contributes one filing where every other firm contributes two, so the plan
requires the primary test (C1 + C1-perm) to be re-run with HAYN excluded.
Zero model calls; reads output/chunk_level_results.csv only.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from analysis.trend_tests import (aggregate_to_firm, jonckheere_terpstra_labeled,
                                  permutation_test_tier_trend, TIER_ORDER)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "experiments" / "hayn_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)
chunks = pd.read_csv(ROOT / "output" / "chunk_level_results.csv")
res = {}
for label, excl in [("all_firms", ()), ("hayn_excluded", ("HAYN",))]:
    firm = aggregate_to_firm(chunks, section="mda", measure="embedding_disagreement",
                             exclude_tickers=excl)
    means = firm.groupby("tier")["firm_mean"].mean().to_dict()
    jt = jonckheere_terpstra_labeled(firm["firm_mean"].to_numpy(float),
                                     firm["tier"].to_numpy(), TIER_ORDER,
                                     alternative="increasing")
    perm = permutation_test_tier_trend(firm)
    res[label] = {"n_firms": int(len(firm)),
                  "tier_means": {k: float(v) for k, v in means.items()},
                  "gradient": float(means["small"] - means["large"]),
                  "jt_stat": float(jt.statistic), "jt_z": float(jt.z),
                  "jt_p": float(jt.p_value), "perm_p": float(perm.p_value)}
(OUT / "results.json").write_text(json.dumps(res, indent=1))
lines = ["HAYN-excluded sensitivity (pre-registered, ANALYSIS_PLAN.md s1)", "=" * 66]
for k, v in res.items():
    m = v["tier_means"]
    lines.append(f"{k:14s} n={v['n_firms']}  large {m['large']:.4f} mid {m['mid']:.4f} "
                 f"small {m['small']:.4f}  grad {v['gradient']:+.4f}  JT p {v['jt_p']:.4f}  perm p {v['perm_p']:.4f}")
(OUT / "report.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
