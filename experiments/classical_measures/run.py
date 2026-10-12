"""
S3 — Does truncation bias ESTABLISHED textual measures, not just ours?

The paper's first contribution is that truncating filings to a fixed budget makes
coverage a function of firm size. Demonstrated only on our own persona-
disagreement measure, that is a weak claim, because our measure turned out not to
measure what it was built to measure. If the same preprocessing step distorts
Fog, Loughran-McDonald category shares and type-token ratio -- measures this
literature has used for fifteen years -- then the finding is about the field's
preprocessing, not about one experimental instrument.

DESIGN. Each filing's MD&A is measured twice with identical code: once complete,
once truncated to the retired 2,000-word cap. Nothing about the measures changes.
Firm-level tier means and the ordered-trend test are computed for each arm, and
the paired per-firm difference is bootstrapped.

READ THE LENGTH-SENSITIVITY COLUMN BEFORE THE RESULT. These measures are not
equally fair tests:

  lm_negative, lm_uncertainty   length-NORMALISED (share of tokens). A shift
                                under truncation is a real compositional finding.
  fog_index                     mildly length-sensitive; it averages sentence
                                length and complex-word share, both of which can
                                drift across a document.
  type_token_ratio              MECHANICALLY length-dependent. TTR always falls
                                as text lengthens, because the vocabulary
                                saturates. Any TTR difference between a 2,000-word
                                and a 12,000-word arm is partly arithmetic, and it
                                is reported here only so that the artifact is
                                visible rather than mistaken for a finding.

Zero model calls.

Run:  python3 -m experiments.classical_measures.run
"""
from __future__ import annotations

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
MEASURES = {
    "lm_negative": "length-normalised",
    "lm_uncertainty": "length-normalised",
    "lm_litigious": "length-normalised",
    "fog_index": "mildly length-sensitive",
    "type_token_ratio": "MECHANICALLY length-dependent - artifact, not finding",
}


def build(config: Config) -> pd.DataFrame:
    from analysis.lm_dictionary import category_shares
    from analysis.textual_baselines import compute_textual_features
    from data.edgar import _html_to_text, extract_sections_dict

    man = pd.read_csv(Path(config.BASE_DIR) / "data" / "sample_manifest.csv")
    res = pd.read_csv(config.OUTPUT_DIR / "results.csv")[
        ["ticker", "filing_type", "filing_date", "tier"]]
    for f in (man, res):
        f["filing_date"] = f.filing_date.astype(str).str.replace("-", "", regex=False)
    fil = man.merge(res, on=["ticker", "filing_type", "filing_date"], how="left")

    rows = []
    for r in fil.itertuples():
        p = config.FILINGS_DIR / f"{r.ticker}_{r.filing_type}_{r.filing_date}.txt"
        if not p.exists():
            continue
        sections, _ = extract_sections_dict(
            _html_to_text(p.read_text(encoding="utf-8", errors="ignore")),
            r.filing_type)
        mda = sections.get("mda") or ""
        if not mda:
            continue
        words = mda.split()
        for arm, txt in (("complete", mda),
                         ("truncated", " ".join(words[:RETIRED_CAP]))):
            feats = compute_textual_features(txt)
            lm = category_shares(txt, config.BASE_DIR)
            rows.append({
                "ticker": r.ticker, "tier": r.tier, "filing_type": r.filing_type,
                "arm": arm, "words": len(txt.split()),
                "complete_words": len(words),
                "fog_index": feats.get("fog_index"),
                "type_token_ratio": feats.get("type_token_ratio"),
                "lm_negative": lm.get("lm_negative"),
                "lm_uncertainty": lm.get("lm_uncertainty"),
                "lm_litigious": lm.get("lm_litigious"),
            })
    return pd.DataFrame(rows)


def analyse(df: pd.DataFrame, seed: int) -> dict:
    res: dict = {"seed": seed, "cap_words": RETIRED_CAP,
                 "n_filings": int(df.arm.eq("complete").sum()),
                 "n_firms": int(df.ticker.nunique()), "measures": {}}

    for m, sensitivity in MEASURES.items():
        block: dict = {"length_sensitivity": sensitivity, "arms": {}}
        firm_tables = {}
        for arm in ("truncated", "complete"):
            sub = df[(df.arm == arm) & df[m].notna()]
            firm = sub.groupby(["ticker", "tier"], as_index=False)[m].mean()
            firm_tables[arm] = firm
            means = {t: float(firm[firm.tier == t][m].mean()) for t in TIER_ORDER}
            jt = jonckheere_terpstra_labeled(
                firm[m].to_numpy(), firm.tier.to_numpy(),
                group_order=TIER_ORDER, alternative="increasing")
            block["arms"][arm] = {
                "tier_means": means,
                "gradient_small_minus_large": means["small"] - means["large"],
                "monotone": bool(means["large"] < means["mid"] < means["small"]),
                "jt_p_one_sided": float(jt.p_value),
                "n_firms": int(len(firm)),
            }

        t, c = firm_tables["truncated"], firm_tables["complete"]
        merged = t.merge(c, on=["ticker", "tier"], suffixes=("_t", "_c"))
        delta = (merged[f"{m}_t"] - merged[f"{m}_c"]).to_numpy()
        ci = bootstrap_ci(delta, statistic=np.mean, n_boot=10000, seed=seed,
                          method="percentile")
        block["paired_truncated_minus_complete"] = {
            "mean": float(ci.observed), "ci": [float(ci.low), float(ci.high)],
            "n_firms": int(len(merged)),
            "differs": bool(not (ci.low <= 0 <= ci.high)),
        }
        # Is the distortion itself tier-dependent? That is what turns a level
        # shift into a confound for a size regression.
        block["distortion_by_tier"] = {}
        for tier in TIER_ORDER:
            s = merged[merged.tier == tier]
            if len(s) >= 3:
                d = (s[f"{m}_t"] - s[f"{m}_c"]).to_numpy()
                b = bootstrap_ci(d, statistic=np.mean, n_boot=10000, seed=seed,
                                 method="percentile")
                block["distortion_by_tier"][tier] = {
                    "mean": float(b.observed), "ci": [float(b.low), float(b.high)]}
        block["gradient_shift"] = (block["arms"]["truncated"]["gradient_small_minus_large"]
                                   - block["arms"]["complete"]["gradient_small_minus_large"])
        res["measures"][m] = block
    return res


def report(r: dict) -> str:
    L = ["S3 — Does truncation bias established textual measures?", "=" * 78, ""]
    L.append(f"{r['n_filings']} filings, {r['n_firms']} firms. Each MD&A measured twice "
             f"with identical code:")
    L.append(f"complete, and truncated to the retired {r['cap_words']:,}-word cap. "
             f"Zero model calls.")
    L.append("")
    for m, b in r["measures"].items():
        t, c = b["arms"]["truncated"], b["arms"]["complete"]
        L.append(f"{m}   [{b['length_sensitivity']}]")
        L.append("-" * 78)
        L.append(f"  {'arm':11s} {'large':>10s} {'mid':>10s} {'small':>10s} "
                 f"{'gradient':>11s} {'JT p':>8s}")
        for name, a in (("truncated", t), ("complete", c)):
            mu = a["tier_means"]
            L.append(f"  {name:11s} {mu['large']:10.4f} {mu['mid']:10.4f} "
                     f"{mu['small']:10.4f} {a['gradient_small_minus_large']:+11.4f} "
                     f"{a['jt_p_one_sided']:8.4f}")
        p = b["paired_truncated_minus_complete"]
        L.append(f"  gradient shift from truncation: {b['gradient_shift']:+.4f}")
        L.append(f"  paired per-firm distortion    : {p['mean']:+.4f} "
                 f"[{p['ci'][0]:+.4f}, {p['ci'][1]:+.4f}]  "
                 f"{'differs' if p['differs'] else 'spans zero'}")
        tiers = b.get("distortion_by_tier") or {}
        if tiers:
            bits = "  ".join(f"{k} {v['mean']:+.4f}" for k, v in tiers.items())
            L.append(f"  distortion by tier            : {bits}")
        L.append("")
    L.append("How to read this")
    L.append("-" * 78)
    L.append("  A level shift under truncation is unsurprising on its own. What matters")
    L.append("  for a size regression is whether the distortion is TIER-DEPENDENT, and")
    L.append("  whether it moves the measured gradient. Where it does, truncation is not")
    L.append("  merely adding noise; it is changing the cross-sectional answer, for")
    L.append("  measures this literature has used for years and that have nothing to do")
    L.append("  with language models.")
    L.append("")
    L.append("  type_token_ratio is reported for completeness and should NOT be read as")
    L.append("  a finding: TTR falls mechanically as text lengthens, so any difference")
    L.append("  between a 2,000-word and a complete arm is partly arithmetic.")
    return "\n".join(L)


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    cfg = Config()
    out_dir = cfg.OUTPUT_DIR / "experiments" / "classical_measures"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build(cfg)
    df.to_csv(out_dir / "classical_measures.csv", index=False)
    res = analyse(df, DEFAULT_SEED)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str),
                                          encoding="utf-8")
    txt = report(res)
    (out_dir / "report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
