"""
H1 step 2 — score the rater's answers.

Reads h1_answers.txt against the hidden key and reports agreement with the
measure, an exact binomial test, and a confidence interval.

Interpretation is NOT symmetric, and the report says so. The project's other
evidence says the measure responds to vagueness rather than to interpretive
difficulty. So:

  agreement well above 50%  the measure tracks something a human also calls
                            "hard to interpret". The negative framing elsewhere
                            in the paper is then too strong and must be softened:
                            vague text may genuinely BE harder to interpret, and
                            the construct definition, not the measure, was wrong.
  agreement near 50%        the measure does not track human-judged difficulty,
                            which corroborates the placebo and informativeness
                            results with an external criterion.
  agreement well below 50%  the measure runs OPPOSITE to human judgment, the
                            strongest version of the negative result.

One self-rating by the paper's own author is weak evidence either way and the
report states that limit rather than leaving it implied.

Run:  python3 -m experiments.human_check.score
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

sys.modules.setdefault("tensorflow", None)
from config import Config


def main() -> int:
    cfg = Config()
    out = cfg.OUTPUT_DIR / "experiments" / "human_check"
    key = pd.read_csv(out / "h1_key.csv")

    raw = (out / "h1_answers.txt").read_text(encoding="utf-8").splitlines()
    ans = [l.strip().upper() for l in raw
           if l.strip() and not l.strip().startswith("#")]
    ans = [x for x in ans if x in ("A", "B")]

    if len(ans) != len(key):
        print(f"ERROR: {len(ans)} answers for {len(key)} pairs. "
              f"One letter (A or B) per line, in order.")
        return 1

    key = key.copy()
    key["answer"] = ans
    key["agrees"] = key.answer == key.high_disagreement_side
    k, n = int(key.agrees.sum()), len(key)
    bt = binomtest(k, n, 0.5, alternative="two-sided")
    ci = bt.proportion_ci(confidence_level=0.95)

    res = {
        "n_pairs": n, "n_agree": k, "agreement": k / n,
        "binomial_p_two_sided": float(bt.pvalue),
        "ci95": [float(ci.low), float(ci.high)],
        "by_tier": {t: {"n": int(len(g)), "agreement": float(g.agrees.mean())}
                    for t, g in key.groupby("tier")},
        "rater_note": ("single rater, who is an author of the paper; weak "
                       "evidence in either direction and labelled as such"),
    }
    (out / "h1_results.json").write_text(json.dumps(res, indent=1),
                                         encoding="utf-8")
    key.to_csv(out / "h1_scored.csv", index=False)

    L = ["H1 — human rating against the measure", "=" * 70, ""]
    L.append(f"  pairs        : {n}")
    L.append(f"  agreement    : {k}/{n} = {k/n:.1%}  "
             f"95% CI [{ci.low:.1%}, {ci.high:.1%}]")
    L.append(f"  binomial p   : {bt.pvalue:.4f} (two-sided, against 50%)")
    L.append("")
    for t, b in res["by_tier"].items():
        L.append(f"    {t:6s} n={b['n']:2d}  agreement {b['agreement']:.1%}")
    L.append("")
    if ci.low > 0.5:
        L.append("  READING: agreement exceeds chance. The measure tracks something a")
        L.append("  human also calls hard to interpret, so the paper's negative framing")
        L.append("  is too strong and should be softened: vague text may genuinely be")
        L.append("  harder to interpret, and the construct DEFINITION was the problem.")
    elif ci.high < 0.5:
        L.append("  READING: agreement is below chance. The measure runs opposite to")
        L.append("  human judgment -- the strongest form of the negative result.")
    else:
        L.append("  READING: agreement is indistinguishable from chance. The measure")
        L.append("  does not track human-judged interpretive difficulty, corroborating")
        L.append("  the placebo and informativeness results with an external criterion.")
    L.append("")
    L.append("  LIMIT: one rater, who is an author. Weak evidence either way. A second")
    L.append("  independent rater would change that; a self-rating of one's own measure")
    L.append("  is the kind of evidence a referee discounts, and should be.")
    txt = "\n".join(L)
    (out / "h1_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
