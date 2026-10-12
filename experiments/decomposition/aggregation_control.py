"""
Isolating the AGGREGATION effect, free, from existing window-level output.

S1b compares "first 2,000 words as one unit" with "complete sections windowed
and averaged" and the two arms differ in coverage, scoring unit AND number of
draws. The number of draws is the one component that can be isolated with no new
model calls, because the main run already scored every window independently.

Taking ONE window per filing at random gives complete-coverage sampling with
single-draw aggregation. Comparing that with the averaged main run holds coverage
and unit size fixed and varies only the number of draws.

Theory says averaging reduces the VARIANCE of a filing mean without moving its
expectation, so this difference should be ~0. Confirming it empirically matters,
because it rules out the explanation a referee reaches for first.

Run:  python3 -m experiments.decomposition.aggregation_control
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.modules.setdefault("tensorflow", None)
from analysis.trend_tests import TIER_ORDER
from config import Config

N_DRAWS = 2000
SEED = 20261004


def _gradient(pairs) -> float:
    f = pd.DataFrame(pairs, columns=["ticker", "tier", "v"])
    f["v"] = f.v.astype(float)
    firm = f.groupby(["ticker", "tier"], as_index=False).v.mean()
    mu = {t: firm[firm.tier == t].v.mean() for t in TIER_ORDER}
    return float(mu["small"] - mu["large"])


def main() -> int:
    cfg = Config()
    out_dir = cfg.OUTPUT_DIR / "experiments" / "decomposition"
    out_dir.mkdir(parents=True, exist_ok=True)

    ch = pd.read_csv(cfg.OUTPUT_DIR / "chunk_level_results.csv")
    d = ch[(ch.section == "mda")
           & (~ch.any_failed.astype(str).str.lower().isin(["true", "1"]))
           & ch.embedding_disagreement.notna()].copy()
    d["fid"] = d.ticker + "|" + d.filing_type + "|" + d.filing_date.astype(str)
    groups = {fid: g[["ticker", "tier", "embedding_disagreement"]].to_numpy(object)
              for fid, g in d.groupby("fid")}

    rng = np.random.default_rng(SEED)
    gs = np.array([_gradient([g[rng.integers(len(g))] for g in groups.values()])
                   for _ in range(N_DRAWS)])

    fm = d.groupby(["fid", "ticker", "tier"], as_index=False).embedding_disagreement.mean()
    g_main = _gradient(fm[["ticker", "tier", "embedding_disagreement"]].to_numpy(object))

    res = {
        "seed": SEED, "n_draws": N_DRAWS,
        "complete_all_windows_averaged": g_main,
        "complete_one_random_window": {
            "mean": float(gs.mean()),
            "ci": [float(np.percentile(gs, 2.5)), float(np.percentile(gs, 97.5))],
        },
        "aggregation_effect": float(gs.mean() - g_main),
        "interpretation": (
            "Coverage and scoring unit are identical across these two arms; only "
            "the number of draws differs. An effect near zero confirms that "
            "averaging changes the variance of a filing mean rather than its "
            "expectation, so the S1b contrast cannot be explained by aggregation."
        ),
    }
    (out_dir / "aggregation_control.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8")
    print(f"complete, all windows averaged : {g_main:+.4f}")
    print(f"complete, one random window    : {gs.mean():+.4f} "
          f"[{res['complete_one_random_window']['ci'][0]:+.4f}, "
          f"{res['complete_one_random_window']['ci'][1]:+.4f}]")
    print(f"aggregation effect             : {res['aggregation_effect']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
