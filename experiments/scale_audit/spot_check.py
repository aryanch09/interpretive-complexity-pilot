"""
S2 spot-check — manual validation of at-scale extraction.

The at-scale coverage claim rests on extraction succeeding on 785 filings from
filers this project has never looked at. A 98.1% success RATE is not evidence
that the successes are correct, so a random sample is checked against two
criteria a human would apply:

  starts_with_header  the extracted MD&A actually begins at an MD&A heading,
                      rather than at a cross-reference or mid-document
  plausible           1,000 to 40,000 words, and not obviously the financial
                      statements (the classic boundary-overrun failure)

Run:  python3 -m experiments.scale_audit.spot_check --n 20
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.modules.setdefault("tensorflow", None)
from config import Config

MDA_RE = re.compile(r"management.s\s+discussion", re.I)
BALANCE_SHEET = "Total current liabilities"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261004)
    a = ap.parse_args(argv)

    cfg = Config()
    out_dir = cfg.OUTPUT_DIR / "experiments" / "scale_audit"
    d = pd.read_csv(out_dir / "scale_audit_usable.csv")
    samp = d.sample(min(a.n, len(d)), random_state=a.seed)
    cache = Path(cfg.BASE_DIR) / "data" / "scale_audit_filings"

    from data.edgar import _html_to_text, extract_sections_dict

    rows = []
    for r in samp.itertuples():
        f = list(cache.glob(f"{r.cik}_*.txt"))
        if not f:
            continue
        sections, _ = extract_sections_dict(
            _html_to_text(f[0].read_text(encoding="utf-8", errors="ignore")), "10-K")
        mda = sections.get("mda") or ""
        n = len(mda.split())
        rows.append({
            "cik": int(r.cik),
            "mda_words": n,
            "risk_words": len((sections.get("risk") or "").split()),
            "starts_with_header": bool(MDA_RE.search(mda[:300])),
            "plausible": bool(1000 <= n <= 40000 and BALANCE_SHEET not in mda[:2000]),
            "first_words": " ".join(mda.split()[:12]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "spot_check.csv", index=False)

    res = {
        "n_checked": int(len(df)),
        "seed": a.seed,
        "starts_with_header": int(df.starts_with_header.sum()),
        "plausible": int(df.plausible.sum()),
        "pass_rate": float(df.plausible.mean()) if len(df) else None,
    }
    (out_dir / "spot_check.json").write_text(json.dumps(res, indent=1),
                                             encoding="utf-8")
    print(f"spot-check: {res['plausible']}/{res['n_checked']} plausible, "
          f"{res['starts_with_header']}/{res['n_checked']} start at an MD&A header")
    if res["plausible"] < res["n_checked"]:
        print(df[~df.plausible].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
