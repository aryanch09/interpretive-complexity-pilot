"""
Firm list with metadata and CIK resolution.

CIKs for large-cap firms are hardcoded (high confidence).
CIKs for mid/small-cap firms are resolved automatically from EDGAR's
company_tickers.json and cached in .cik_cache.json.

Run `python firms.py` to print the resolved firm list and verify CIKs.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

EDGAR_USER_AGENT = "Research Project aryanchhabra6996@gmail.com"

# ── Firm subsets ─────────────────────────────────────────────────────────────

# Firms actually run in the initial 11-firm pilot (5 large / 3 mid / 3 small)
PILOT_FIRMS: list[str] = [
    "AAPL", "MSFT", "JPM", "PG", "HD",       # large-cap
    "DRI", "TOL", "WYNN",                      # mid-cap
    "NGVC", "UTMD", "OFLX",                    # small-cap
]

# 30-firm expansion set — ready to run, awaiting advisor sign-off.
# Target: ~10 per tier, sector-diverse, all with post-cutoff EDGAR filings.
# Do NOT run this set until the advisor has seen the 11-firm pilot results.
# Launch with: python main.py --firm-set full
FULL_PILOT_FIRMS: list[str] = [
    # Large-cap (10): 5 pilot + 5 new
    "AAPL", "MSFT", "JPM", "PG", "HD",
    "AMZN", "GOOGL", "XOM", "WMT", "BAC",
    # Mid-cap (10): 3 pilot + 7 new
    "DRI", "WYNN", "TOL",
    "HAS", "WTS", "IBP", "CATY", "KFRC", "PLAB", "CLFD",
    # Small-cap (10): 3 pilot + 7 new
    "NGVC", "OFLX", "UTMD",
    "HAYN", "MLAB", "UFPT", "CLIR", "IIIN", "RELL", "LWAY",
]

# fmt: off
FIRMS: list[dict] = [
    # ── Large-cap tier ── expected LOW disagreement ──────────────────────────
    {"ticker": "AAPL",  "name": "Apple Inc.",                          "tier": "large", "cik": 320193},
    {"ticker": "MSFT",  "name": "Microsoft Corp.",                     "tier": "large", "cik": 789019},
    {"ticker": "JPM",   "name": "JPMorgan Chase",                      "tier": "large", "cik": 19617},
    {"ticker": "JNJ",   "name": "Johnson & Johnson",                   "tier": "large", "cik": 200406},
    {"ticker": "WMT",   "name": "Walmart Inc.",                        "tier": "large", "cik": 104169},
    {"ticker": "XOM",   "name": "ExxonMobil",                          "tier": "large", "cik": 34088},
    {"ticker": "PG",    "name": "Procter & Gamble",                    "tier": "large", "cik": 80424},
    {"ticker": "HD",    "name": "Home Depot",                          "tier": "large", "cik": 354950},
    {"ticker": "BAC",   "name": "Bank of America",                     "tier": "large", "cik": 70858},

    # ── Mid-cap tier ── expected MODERATE disagreement ────────────────────────
    {"ticker": "HAS",   "name": "Hasbro",                              "tier": "mid",   "cik": None},
    {"ticker": "DRI",   "name": "Darden Restaurants",                  "tier": "mid",   "cik": None},
    {"ticker": "WYNN",  "name": "Wynn Resorts",                        "tier": "mid",   "cik": None},
    {"ticker": "THS",   "name": "Treehouse Foods",                     "tier": "mid",   "cik": 1320695},
    {"ticker": "IBP",   "name": "Installed Building Products",         "tier": "mid",   "cik": None},
    {"ticker": "WTS",   "name": "Watts Water Technologies",            "tier": "mid",   "cik": None},
    {"ticker": "CATY",  "name": "Cathay General Bancorp",              "tier": "mid",   "cik": None},
    {"ticker": "TOL",   "name": "Toll Brothers",                       "tier": "mid",   "cik": None},
    {"ticker": "FNB",   "name": "F.N.B. Corporation",                  "tier": "mid",   "cik": None},

    # ── Small-cap tier ── expected HIGH disagreement ──────────────────────────
    {"ticker": "NGVC",  "name": "Natural Grocers by Vitamin Cottage",  "tier": "small", "cik": None},
    {"ticker": "OFLX",  "name": "Omega Flex",                          "tier": "small", "cik": None},
    {"ticker": "UTMD",  "name": "Utah Medical Products",               "tier": "small", "cik": None},
    {"ticker": "LWAY",  "name": "Lifeway Foods",                       "tier": "small", "cik": None},
    # PCTI (PCTEL) deregistered Dec 2023 — substituted with SPOK Holdings:
    # niche healthcare-paging / software platform, minimal analyst coverage
    {"ticker": "SPOK",  "name": "Spok Holdings",                       "tier": "small", "cik": 1289945},
    {"ticker": "IIIN",  "name": "Insteel Industries",                  "tier": "small", "cik": None},
    {"ticker": "RELL",  "name": "Richardson Electronics",              "tier": "small", "cik": None},
    {"ticker": "HAFC",  "name": "Hanmi Financial",                     "tier": "small", "cik": None},
    {"ticker": "CLMT",  "name": "Calumet Specialty Products",          "tier": "small", "cik": None},

    # ── Full-pilot expansion firms (not in initial 11-firm pilot) ────────────
    # Large-cap additions
    {"ticker": "AMZN",  "name": "Amazon.com Inc.",                     "tier": "large", "cik": 1018724},
    {"ticker": "GOOGL", "name": "Alphabet Inc.",                       "tier": "large", "cik": 1652044},

    # Mid-cap additions
    {"ticker": "KFRC",  "name": "Kforce Inc.",                         "tier": "mid",   "cik": None},
    {"ticker": "PLAB",  "name": "Photronics Inc.",                     "tier": "mid",   "cik": None},
    {"ticker": "CLFD",  "name": "Clearfield Inc.",                     "tier": "mid",   "cik": None},

    # Small-cap additions
    {"ticker": "HAYN",  "name": "Haynes International Inc.",           "tier": "small", "cik": 858655},
    {"ticker": "MLAB",  "name": "Mesa Laboratories Inc.",              "tier": "small", "cik": None},
    {"ticker": "UFPT",  "name": "UFP Technologies Inc.",               "tier": "small", "cik": None},
    {"ticker": "CLIR",  "name": "ClearSign Technologies Corp.",        "tier": "small", "cik": None},
]
# fmt: on


def resolve_ciks(
    firms: list[dict],
    cache_file: Path = Path(".cik_cache.json"),
) -> list[dict]:
    """
    Fill in any missing CIK values by querying EDGAR's company_tickers.json.
    Results are cached locally so subsequent runs don't hit the network.
    """
    cache: dict[str, int] = {}
    if cache_file.exists():
        with open(cache_file) as f:
            cache = json.load(f)

    needs_lookup = [
        f for f in firms
        if f["cik"] is None and f["ticker"] not in cache
    ]

    if needs_lookup:
        logger.info("Fetching EDGAR company ticker → CIK mapping (%d firms need lookup)", len(needs_lookup))
        headers = {"User-Agent": EDGAR_USER_AGENT}
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        time.sleep(0.5)

        # EDGAR returns {"0": {"cik_str": "320193", "ticker": "AAPL", ...}, ...}
        ticker_to_cik: dict[str, int] = {}
        for entry in resp.json().values():
            ticker_to_cik[entry["ticker"].upper()] = int(entry["cik_str"])

        for firm in needs_lookup:
            t = firm["ticker"]
            if t in ticker_to_cik:
                cache[t] = ticker_to_cik[t]
                logger.info("  Resolved %s → CIK %d", t, cache[t])
            else:
                logger.warning("  Could not resolve CIK for %s", t)

        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=2)

    resolved = []
    for firm in firms:
        f = firm.copy()
        if f["cik"] is None:
            f["cik"] = cache.get(f["ticker"])
        resolved.append(f)

    return resolved


def get_firm(ticker: str, firms: Optional[list[dict]] = None) -> Optional[dict]:
    source = firms if firms is not None else FIRMS
    for f in source:
        if f["ticker"] == ticker:
            return f
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    resolved = resolve_ciks(FIRMS)
    print(f"\n{'Ticker':<8} {'Tier':<8} {'CIK':>10}  Name")
    print("-" * 60)
    for f in resolved:
        cik_str = str(f["cik"]) if f["cik"] else "MISSING"
        print(f"{f['ticker']:<8} {f['tier']:<8} {cik_str:>10}  {f['name']}")
    missing = [f["ticker"] for f in resolved if not f["cik"]]
    if missing:
        print(f"\nWARNING: no CIK found for: {', '.join(missing)}")
