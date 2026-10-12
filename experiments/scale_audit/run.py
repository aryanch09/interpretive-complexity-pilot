"""
S2 — The coverage confound at scale, with zero model calls.

The 59-filing pilot shows that a fixed reading budget reads proportionally less
of a large firm's filing, so coverage becomes a function of firm size. That is
the paper's central methodological claim, and on 59 filings it is a claim about
our sample. This experiment tests it on a large cross-section of filers so it
becomes a claim about the field.

Design:
  * Firm size is PUBLIC FLOAT as the filer itself reports it on the 10-K cover
    page (dei:EntityPublicFloat, via the SEC XBRL frames API). This is an
    SEC-filed value, not a reconstruction from market data, which matters because
    the project forbids rebuilding point-in-time size from free market sources.
  * Narrative length is MD&A + Risk Factors measured with the project's own
    extractor, so the at-scale numbers are produced by exactly the same code as
    the pilot's.
  * Reported: the elasticity of narrative length with respect to public float;
    the fraction of the narrative read under budgets common in the literature,
    by float decile; and extraction success rates, so the reader can see how much
    of the sample the measurement actually covers.

NO model or API calls. EDGAR only, at the documented rate limit, with the
project User-Agent. Downloads are cached so a re-run is offline.

Run:  python3 -m experiments.scale_audit.run --target 600 --time-box-min 300
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.modules.setdefault("tensorflow", None)

from config import Config

logger = logging.getLogger(__name__)

FRAMES = "https://data.sec.gov/api/xbrl/frames/dei/EntityPublicFloat/USD/{period}.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

# Budgets common in the LLM-on-filings literature, in WORDS.
# Token budgets converted at 0.75 words/token.
BUDGETS = {
    "512 tokens": int(512 * 0.75),
    "2,000 words": 2000,
    "4k tokens": int(4096 * 0.75),
    "8k tokens": int(8192 * 0.75),
    "32k tokens": int(32768 * 0.75),
    "128k tokens": int(131072 * 0.75),
}


class Fetcher:
    """EDGAR client honouring the 10 req/s limit with the project User-Agent."""

    def __init__(self, user_agent: str, rate: float = 8.0):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent,
                               "Accept-Encoding": "gzip, deflate"})
        self._min_interval = 1.0 / rate
        self._last = 0.0
        import threading
        self._lock = threading.Lock()

    def get(self, url: str, timeout: float = 30.0):
        with self._lock:
            wait = self._min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
        return self.s.get(url, timeout=timeout)


def gather_float(fetcher: Fetcher, periods: list[str]) -> pd.DataFrame:
    """Public float per CIK, taking each filer's most recent reported value."""
    rows = []
    for p in periods:
        try:
            r = fetcher.get(FRAMES.format(period=p))
            if r.status_code != 200:
                logger.warning("frames %s -> HTTP %s", p, r.status_code)
                continue
            data = r.json().get("data", [])
            for d in data:
                if d.get("val"):
                    rows.append({"cik": int(d["cik"]), "entity": d.get("entityName"),
                                 "float_usd": float(d["val"]), "period": p,
                                 "end": d.get("end")})
            logger.info("frames %s: %d filers", p, len(data))
        except Exception as exc:  # noqa: BLE001
            logger.warning("frames %s failed: %s", p, exc)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("end").groupby("cik", as_index=False).last()
    return df[df.float_usd > 0]


def stratified_sample(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Sample evenly across float deciles so the size range is covered."""
    df = df.copy()
    df["decile"] = pd.qcut(df.float_usd, 10, labels=False, duplicates="drop")
    per = max(1, target // df.decile.nunique())
    rng = np.random.default_rng(seed)
    picks = []
    for _, g in df.groupby("decile"):
        k = min(per, len(g))
        picks.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    return pd.concat(picks).reset_index(drop=True)


def latest_10k(fetcher: Fetcher, cik: int, after: str) -> dict | None:
    r = fetcher.get(SUBMISSIONS.format(cik=cik))
    if r.status_code != 200:
        return None
    recent = (r.json().get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    for i, f in enumerate(forms):
        if f != "10-K":
            continue
        date = recent["filingDate"][i]
        if date <= after:
            continue
        return {"accession": recent["accessionNumber"][i],
                "doc": recent["primaryDocument"][i], "date": date}
    return None


def measure(fetcher: Fetcher, cik: int, meta: dict, cache_dir: Path) -> dict | None:
    from data.edgar import _html_to_text, extract_sections_dict

    acc = meta["accession"].replace("-", "")
    cache = cache_dir / f"{cik}_{meta['date'].replace('-','')}.txt"
    if cache.exists():
        html = cache.read_text(encoding="utf-8", errors="ignore")
    else:
        url = ARCHIVE.format(cik=cik, acc_nodash=acc, doc=meta["doc"])
        r = fetcher.get(url, timeout=60.0)
        if r.status_code != 200 or not r.text:
            return None
        html = r.text
        cache.write_text(html, encoding="utf-8")

    text = _html_to_text(html)
    sections, failed = extract_sections_dict(text, "10-K")
    mda = len((sections.get("mda") or "").split())
    risk = len((sections.get("risk") or "").split())
    return {"cik": cik, "filing_date": meta["date"], "extraction_failed": bool(failed),
            "mda_words": mda, "risk_words": risk, "narrative_words": mda + risk,
            "raw_words": len(text.split())}


def run(config: Config, target: int, seed: int, time_box_s: float,
        workers: int) -> dict:
    out_dir = config.OUTPUT_DIR / "experiments" / "scale_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config.BASE_DIR) / "data" / "scale_audit_filings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    fetcher = Fetcher(config.EDGAR_USER_AGENT)
    periods = [f"CY{y}Q{q}I" for y in (2025, 2026) for q in (1, 2, 3, 4)]
    floats = gather_float(fetcher, periods)
    print(f"filers with public float: {len(floats):,}")
    if floats.empty:
        return {"error": "no float data"}
    floats.to_csv(out_dir / "public_float.csv", index=False)

    sample = stratified_sample(floats, target, seed)
    print(f"stratified sample: {len(sample)} filers across "
          f"{sample.decile.nunique()} float deciles")

    start = time.time()
    results, misses = [], 0

    def work(row):
        meta = latest_10k(fetcher, int(row.cik), config.TRAINING_CUTOFF)
        if not meta:
            return None
        m = measure(fetcher, int(row.cik), meta, cache_dir)
        if m:
            m.update({"entity": row.entity, "float_usd": row.float_usd,
                      "decile": int(row.decile)})
        return m

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, r): r for r in sample.itertuples()}
        for n, fut in enumerate(as_completed(futures), 1):
            if time.time() - start > time_box_s:
                print(f"\nTIME BOX reached after {n} filings — stopping cleanly.")
                for f in futures:
                    f.cancel()
                break
            try:
                m = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.debug("worker failed: %s", exc)
                m = None
            if m:
                results.append(m)
            else:
                misses += 1
            if n % 25 == 0:
                print(f"  {n}/{len(sample)} attempted, {len(results)} measured, "
                      f"{misses} unavailable, {time.time()-start:.0f}s", flush=True)

    df = pd.DataFrame(results)
    if df.empty:
        return {"error": "no filings measured"}
    df.to_csv(out_dir / "scale_audit.csv", index=False)
    return summarise(df, misses, out_dir, seed)


def summarise(df: pd.DataFrame, misses: int, out_dir: Path, seed: int) -> dict:
    from scipy.stats import spearmanr

    ok = df[(~df.extraction_failed) & (df.mda_words >= 1000)].copy()
    res: dict = {
        "seed": seed,
        "n_attempted_measured": int(len(df)),
        "n_unavailable": int(misses),
        "n_usable": int(len(ok)),
        "extraction_success_rate": float(len(ok) / max(1, len(df))),
        "float_range_usd": [float(df.float_usd.min()), float(df.float_usd.max())],
    }

    # Elasticity of narrative length w.r.t. public float.
    ok = ok[ok.narrative_words > 0]
    x = np.log(ok.float_usd.to_numpy())
    y = np.log(ok.narrative_words.to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    rho, p = spearmanr(ok.float_usd, ok.narrative_words)
    res["elasticity_log_narrative_on_log_float"] = float(slope)
    res["spearman_float_vs_narrative"] = {"rho": float(rho), "p": float(p),
                                          "n": int(len(ok))}

    # Fraction read under each budget, by float decile.
    frac = {}
    for name, budget in BUDGETS.items():
        ok[f"frac_{name}"] = np.minimum(1.0, budget / ok.narrative_words)
        by_dec = ok.groupby("decile")[f"frac_{name}"].median()
        r2, p2 = spearmanr(ok.decile, ok[f"frac_{name}"])
        frac[name] = {
            "budget_words": budget,
            "median_overall": float(ok[f"frac_{name}"].median()),
            "median_bottom_decile": float(by_dec.iloc[0]) if len(by_dec) else None,
            "median_top_decile": float(by_dec.iloc[-1]) if len(by_dec) else None,
            "spearman_decile_vs_fraction_read": {"rho": float(r2), "p": float(p2)},
            "share_truncated": float((ok.narrative_words > budget).mean()),
        }
    res["budgets"] = frac
    ok.to_csv(out_dir / "scale_audit_usable.csv", index=False)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str),
                                          encoding="utf-8")
    return res


def report(r: dict) -> str:
    L = ["S2 — The coverage confound at scale (zero model calls)", "=" * 78, ""]
    if "error" in r:
        return "\n".join(L + [f"FAILED: {r['error']}"])
    L.append(f"Filings measured : {r['n_attempted_measured']:,}   "
             f"usable {r['n_usable']:,}   unavailable {r['n_unavailable']:,}")
    L.append(f"Extraction success: {r['extraction_success_rate']*100:.1f}%")
    lo, hi = r["float_range_usd"]
    L.append(f"Public float range: ${lo:,.0f} .. ${hi:,.0f}")
    L.append("")
    s = r["spearman_float_vs_narrative"]
    L.append("Do larger firms file longer narratives?")
    L.append("-" * 78)
    L.append(f"  elasticity of log(MD&A + Risk words) on log(public float) = "
             f"{r['elasticity_log_narrative_on_log_float']:+.4f}")
    L.append(f"  Spearman(public float, narrative words) rho = {s['rho']:+.3f}  "
             f"p = {s['p']:.3g}  n = {s['n']:,}")
    L.append("")
    L.append("Fraction of the narrative read under budgets common in the literature")
    L.append("-" * 78)
    L.append(f"  {'budget':14s} {'words':>7s} {'median read':>12s} "
             f"{'bottom decile':>14s} {'top decile':>11s} {'truncated':>10s}")
    for name, b in r["budgets"].items():
        L.append(f"  {name:14s} {b['budget_words']:7,d} "
                 f"{b['median_overall']*100:11.1f}% "
                 f"{(b['median_bottom_decile'] or 0)*100:13.1f}% "
                 f"{(b['median_top_decile'] or 0)*100:10.1f}% "
                 f"{b['share_truncated']*100:9.1f}%")
    L.append("")
    L.append("  'bottom decile' is the smallest firms by public float, 'top decile' the")
    L.append("  largest. A budget reads a SMALLER share of a larger firm's filing, so")
    L.append("  under any fixed budget coverage is a function of firm size — which is")
    L.append("  the variable size-effect studies regress on.")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--time-box-min", type=float, default=300)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)

    config = Config()
    r = run(config, a.target, a.seed, a.time_box_min * 60, a.workers)
    text = report(r)
    print(text)
    (config.OUTPUT_DIR / "experiments" / "scale_audit" / "report.txt").write_text(
        text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
