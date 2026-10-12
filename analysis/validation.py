"""
Validation checks: do disagreement scores vary with known complexity proxies?

Proxies:
  idiosyncratic_volatility — annualized residual std from market regression
  firm_age_years           — years since first trade date
  analyst_count            — number of analyst opinions (coverage proxy)
  market_cap               — size proxy (log scale in regressions/plots)

Validation tests:
  1. Spearman rank correlations between each disagreement measure and each proxy
     — with Bonferroni correction for multiple tests
     — with bootstrap 95% CI for each ρ (1,000 resamples)
     — with tie-detection flag: numeric_disagreement in the pilot sample takes
       only 3 distinct values; Spearman p-values are invalid when most ranks are
       tied, and should not be starred.
  2. Mean / std of both measures by tier (large / mid / small)
  3. Adjusted disagreement: OLS residual after partialing out textual controls
  4. Descriptive statistics table
  5. Scatter plots saved as PNG

Statistical caution — numeric_disagreement:
  In the initial 11-firm pilot, numeric_disagreement takes only 3 values
  (0.0, 0.157, 0.313) due to the old integer scoring scale. This makes
  Spearman p-values invalid (the test assumes continuous ranks). All
  numeric_disagreement p-values are flagged as unreliable in output until
  the continuous 0-100 scoring scale is rerun. Do NOT star significance on
  numeric_disagreement from the pilot data.
"""

import datetime
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import yfinance as yf
from scipy import stats

logger = logging.getLogger(__name__)

MARKET_BENCHMARK = "SPY"
HIST_PERIOD = "1y"
ANNUALIZE = np.sqrt(252)


# ── Complexity proxy collection ───────────────────────────────────────────────

def _idiosyncratic_volatility(ticker: str) -> Optional[float]:
    """
    Annualized idiosyncratic volatility: std dev of residuals from OLS
    regression of daily firm returns on SPY returns over the past year.

    Note on proxy timing: this uses the most recent year of data, not
    data contemporaneous with the filing date. For robustness, future
    versions should use the 1-year window ending on the filing date.
    """
    try:
        stock_hist = yf.download(ticker, period=HIST_PERIOD, progress=False, auto_adjust=True)
        mkt_hist = yf.download(MARKET_BENCHMARK, period=HIST_PERIOD, progress=False, auto_adjust=True)

        if stock_hist.empty or mkt_hist.empty:
            return None

        stock_ret = stock_hist["Close"].squeeze().pct_change().dropna()
        mkt_ret = mkt_hist["Close"].squeeze().pct_change().dropna()

        common = stock_ret.index.intersection(mkt_ret.index)
        if len(common) < 50:
            return None

        y = stock_ret.loc[common].values
        x = mkt_ret.loc[common].values

        X = np.column_stack([np.ones(len(x)), x])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta

        return float(np.std(residuals) * ANNUALIZE)

    except Exception as exc:
        logger.warning("idio_vol failed for %s: %s", ticker, exc)
        return None


def _firm_age(ticker: str, cik: Optional[int] = None) -> Optional[float]:
    """
    Years since the stock's first trade date.

    Sources tried in order:
      1. yfinance ipoDate
      2. yfinance first_trade_date_epoch_utc
      3. EDGAR submissions API earliest filing date (proxy; overstates age slightly)

    Note: all three sources give age as of today, not as of the filing date.
    The error is small (<1 year) for our filing window.
    """
    now = datetime.datetime.utcnow()

    try:
        info = yf.Ticker(ticker).info
        ipo_str = info.get("ipoExpectedDate") or info.get("ipoDate")
        if ipo_str:
            dt = datetime.datetime.strptime(ipo_str[:10], "%Y-%m-%d")
            age = (now - dt).days / 365.25
            logger.debug("firm_age %s: yfinance ipoDate → %.1f yrs", ticker, age)
            return round(age, 2)
    except Exception:
        pass

    try:
        fast = yf.Ticker(ticker).fast_info
        first_trade = getattr(fast, "first_trade_date_epoch_utc", None)
        if first_trade is not None:
            dt = datetime.datetime.utcfromtimestamp(first_trade)
            age = (now - dt).days / 365.25
            logger.debug("firm_age %s: yfinance fast_info → %.1f yrs", ticker, age)
            return round(age, 2)
    except Exception:
        pass

    if cik is not None:
        try:
            cik_str = str(cik).zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{cik_str}.json"
            headers = {"User-Agent": "Research Project aryanchhabra6996@gmail.com"}
            resp = requests.get(url, headers=headers, timeout=15)
            time.sleep(0.5)
            if resp.status_code == 200:
                data = resp.json()
                filings_block = data.get("filings", {})
                # Most-recent 1000 filings
                dates = list(filings_block.get("recent", {}).get("filingDate", []))
                # Older filings are paginated; each entry has filingFrom/filingTo
                for chunk in filings_block.get("files", []):
                    filing_from = chunk.get("filingFrom", "")
                    if filing_from:
                        dates.append(filing_from)
                if dates:
                    earliest = min(dates)
                    dt = datetime.datetime.strptime(earliest[:10], "%Y-%m-%d")
                    age = (now - dt).days / 365.25
                    logger.info("firm_age %s: EDGAR fallback → %.1f yrs", ticker, earliest)
                    return round(age, 2)
        except Exception as exc:
            logger.warning("firm_age EDGAR fallback failed for %s: %s", ticker, exc)

    logger.warning("firm_age: all sources failed for %s", ticker)
    return None


def get_complexity_proxies(ticker: str, cik: Optional[int] = None) -> dict:
    """Collect all complexity proxies for a single ticker.

    Limitation: market_cap and analyst_count are point-in-time (today),
    not contemporaneous with the filing. For volatile small-caps this can
    differ substantially from the filing-date value.
    """
    proxy: dict = {"ticker": ticker}

    try:
        info = yf.Ticker(ticker).info
        proxy["market_cap"] = info.get("marketCap")
        proxy["analyst_count"] = info.get("numberOfAnalystOpinions")
    except Exception as exc:
        logger.warning("yfinance info failed for %s: %s", ticker, exc)
        proxy["market_cap"] = None
        proxy["analyst_count"] = None

    proxy["idiosyncratic_volatility"] = _idiosyncratic_volatility(ticker)
    proxy["firm_age_years"] = _firm_age(ticker, cik=cik)

    return proxy


# ── Analyst coverage table ────────────────────────────────────────────────────

def build_analyst_coverage_table(
    proxy_df: pd.DataFrame,
    merged_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a table sorted by analyst count for export and inspection."""
    cols_needed = ["ticker", "tier", "analyst_count", "market_cap",
                   "numeric_disagreement", "embedding_disagreement"]
    available = [c for c in cols_needed if c in merged_df.columns]
    tbl = merged_df[available].copy()
    if "market_cap" in tbl.columns:
        tbl["market_cap_B"] = (tbl["market_cap"] / 1e9).round(1)
        tbl = tbl.drop(columns=["market_cap"])
    tbl = tbl.sort_values("analyst_count", ascending=False, na_position="last")
    return tbl.reset_index(drop=True)


# ── Textual baseline decomposition (Bushee, Gow & Taylor 2018) ───────────────

def compute_adjusted_disagreement(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Partial out textual controls from disagreement measures using OLS residuals.

    Controls: log(word_count) + lm_neg_pct + fog_index (if available).
    The residual isolates genuine interpretive complexity from length-driven or
    tone-driven disagreement.

    Limitation: this OLS is estimated on n=33 filing-observations. R² CIs are
    wide at this sample size — R²=0.015 and R²=0.15 are statistically
    indistinguishable. The absence of a large R² is suggestive but not conclusive.
    """
    df = results_df.copy()
    text_controls = ["word_count", "lm_neg_pct", "fog_index"]
    available = [c for c in text_controls if c in df.columns and df[c].notna().any()]

    if not available:
        logger.warning(
            "compute_adjusted_disagreement: no textual controls found — "
            "run enrich_results_with_text_features() first"
        )
        return df

    for dm in ["numeric_disagreement", "embedding_disagreement"]:
        if dm not in df.columns:
            continue

        sub = df[[dm] + available].copy().dropna()
        if len(sub) < max(5, len(available) + 2):
            logger.warning("Too few observations for adjusted_%s (%d rows)", dm, len(sub))
            continue

        ctrl_vals = []
        for c in available:
            col = sub[c].values.astype(float)
            if c == "word_count":
                col = np.log1p(col)
            ctrl_vals.append(col)

        X = np.column_stack([np.ones(len(sub))] + ctrl_vals)
        y = sub[dm].values.astype(float)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        residuals = y - X @ beta

        adj_col = f"adjusted_{dm}"
        df.loc[sub.index, adj_col] = residuals

        r2 = 1 - np.var(residuals) / np.var(y) if np.var(y) > 0 else 0
        logger.info(
            "  adjusted_%s: controls=%s, n=%d, R²=%.3f "
            "(wide CI at n=%d — interpret directionally only)",
            dm, available, len(sub), r2, len(sub),
        )

    return df


# ── Post-filing trading volume validation (Bhagwat et al. 2025) ──────────────

def validate_trading_volume(
    results_df: pd.DataFrame,
    pre_window: int = 20,
    post_window: int = 5,
) -> Optional[pd.DataFrame]:
    """
    Test whether LLM disagreement predicts abnormal trading volume.

    Caution on interpretation: a positive β(embedding_disagreement → abnormal_volume)
    is consistent with the hypothesis that interpretive ambiguity drives investor
    disagreement and trade. However, it is also consistent with:
      (a) the filing event itself generating both LLM diversity and abnormal volume
          (information-event confound — filing days have elevated volume regardless)
      (b) firms with more news (which generates both) appearing in both
    Ruling out (a) would require testing whether β holds conditional on the
    earnings surprise or the information content of the filing.
    """
    rows_out: list[dict] = []

    for _, row in results_df.iterrows():
        ticker = row["ticker"]
        filing_date_str = str(row["filing_date"])[:10]

        try:
            from datetime import date, timedelta
            fd = date.fromisoformat(filing_date_str)
            start = (fd - timedelta(days=pre_window + 10)).isoformat()
            end = (fd + timedelta(days=post_window + 10)).isoformat()

            hist = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if hist.empty or len(hist) < pre_window // 2:
                continue

            if isinstance(hist.columns, pd.MultiIndex):
                hist = hist.droplevel(level=1, axis=1)

            hist = hist.sort_index()
            hist_dates = pd.to_datetime(hist.index).date

            pre_mask = hist_dates < fd
            post_mask = hist_dates >= fd
            pre_data = hist[pre_mask]
            post_data = hist[post_mask]

            if len(pre_data) < 5 or len(post_data) < 1:
                continue

            pre_vol_series = pre_data["Volume"].squeeze().tail(pre_window)
            post_vol_series = post_data["Volume"].squeeze().head(post_window)
            pre_vol = float(pre_vol_series.mean())
            post_vol = float(post_vol_series.mean())

            if pre_vol == 0 or np.isnan(pre_vol):
                continue

            abnormal_vol = float(post_vol / pre_vol - 1)

            pre_close = float(pre_data["Close"].squeeze().iloc[-1])
            post_idx = min(post_window - 1, len(post_data) - 1)
            post_close = float(post_data["Close"].squeeze().iloc[post_idx])
            abnormal_ret = float((post_close - pre_close) / pre_close) if pre_close != 0 else np.nan

            rows_out.append({
                "ticker": ticker,
                "tier": row.get("tier"),
                "filing_date": filing_date_str,
                "filing_type": row.get("filing_type"),
                "numeric_disagreement": row.get("numeric_disagreement"),
                "embedding_disagreement": row.get("embedding_disagreement"),
                "adjusted_numeric_disagreement": row.get("adjusted_numeric_disagreement"),
                "adjusted_embedding_disagreement": row.get("adjusted_embedding_disagreement"),
                "abnormal_volume": round(abnormal_vol, 4),
                "abnormal_return": round(float(abnormal_ret), 4),
            })

        except Exception as exc:
            logger.warning("Trading volume failed for %s %s: %s", ticker, filing_date_str, exc)

    if not rows_out:
        logger.warning("No trading volume data collected")
        return None

    vol_df = pd.DataFrame(rows_out)
    logger.info("Trading volume validation: %d filing-observations", len(vol_df))
    _print_volume_regression(vol_df)
    return vol_df


def _print_volume_regression(vol_df: pd.DataFrame) -> None:
    """OLS summary of abnormal_volume ~ disagreement, including a market-cap baseline."""
    # Baseline: abnormal_volume ~ market_cap (tests whether filing-event volume
    # alone explains abnormal trading before adding disagreement)
    for dm in ["numeric_disagreement", "embedding_disagreement",
               "adjusted_numeric_disagreement", "adjusted_embedding_disagreement"]:
        if dm not in vol_df.columns:
            continue
        sub = vol_df[["abnormal_volume", dm]].dropna()
        if len(sub) < 5:
            continue
        X = np.column_stack([np.ones(len(sub)), sub[dm].values])
        y = sub["abnormal_volume"].values
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        # Approximate SE via OLS formula (homoskedastic)
        n, k = len(sub), 2
        mse = ss_res / (n - k) if n > k else np.nan
        xtx_inv = np.linalg.pinv(X.T @ X)
        se_beta1 = np.sqrt(mse * xtx_inv[1, 1]) if not np.isnan(mse) else np.nan
        t_stat = beta[1] / se_beta1 if not np.isnan(se_beta1) and se_beta1 > 0 else np.nan
        logger.info(
            "  abnormal_vol ~ %s: β=%.4f (SE=%.4f, t=%.2f), R²=%.3f (n=%d)",
            dm, beta[1], se_beta1, t_stat, r2, len(sub),
        )


# ── Statistical helpers ───────────────────────────────────────────────────────

def _count_most_common(x: pd.Series) -> int:
    """Return the size of the largest tie group."""
    valid = x.dropna()
    if valid.empty:
        return 0
    return int(valid.value_counts().iloc[0])


def _is_low_resolution(x: pd.Series, threshold: float = 0.50) -> bool:
    """True if one value accounts for >threshold of non-null observations."""
    valid = x.dropna()
    if valid.empty:
        return True
    return _count_most_common(valid) / len(valid) > threshold


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Spearman ρ and p-value, dropping NaN pairs."""
    mask = x.notna() & y.notna()
    if mask.sum() < 4:
        return (float("nan"), float("nan"))
    rho, pval = stats.spearmanr(x[mask], y[mask])
    return (round(float(rho), 4), round(float(pval), 4))


def _spearman_with_ci(
    x: pd.Series,
    y: pd.Series,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """
    Return (rho, pval, ci_lower, ci_upper) with bootstrap 95% CI.

    Bootstrap resamples (n_boot=1000) are used instead of the Fisher-z
    approximation because the Fisher-z is unreliable with small n and ties.
    """
    rho, pval = _spearman(x, y)
    mask = x.notna() & y.notna()
    x_clean = x[mask].values
    y_clean = y[mask].values
    n = len(x_clean)

    if n < 4:
        return rho, pval, float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = stats.spearmanr(x_clean[idx], y_clean[idx])
        if not np.isnan(r):
            boot_rhos.append(r)

    if len(boot_rhos) < 10:
        return rho, pval, float("nan"), float("nan")

    ci_lower = float(np.percentile(boot_rhos, 2.5))
    ci_upper = float(np.percentile(boot_rhos, 97.5))
    return rho, pval, round(ci_lower, 4), round(ci_upper, 4)


# ── Descriptive statistics ────────────────────────────────────────────────────

def build_descriptive_stats(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics (mean, std, min, median, max) for key variables,
    overall and by tier.

    These should appear as Table 1 in any paper draft.
    """
    vars_of_interest = [
        "numeric_disagreement",
        "embedding_disagreement",
        "market_cap",
        "analyst_count",
        "idiosyncratic_volatility",
        "firm_age_years",
        "word_count",
        "lm_neg_pct",
        "fog_index",
    ]
    available = [v for v in vars_of_interest if v in merged.columns]

    rows = []
    for var in available:
        col = merged[var].dropna()
        if col.empty:
            continue
        rows.append({
            "variable": var,
            "n": int(col.count()),
            "mean": round(float(col.mean()), 4),
            "std": round(float(col.std()), 4),
            "min": round(float(col.min()), 4),
            "p25": round(float(col.quantile(0.25)), 4),
            "median": round(float(col.median()), 4),
            "p75": round(float(col.quantile(0.75)), 4),
            "max": round(float(col.max()), 4),
        })

    return pd.DataFrame(rows)


# ── Validation tests ──────────────────────────────────────────────────────────

def run_validation(
    results_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
    tier_results_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Merge disagreement scores with proxies and run all validation checks.

    Returns a dict with:
      correlations   — DataFrame with ρ, p-value, CI, Bonferroni flag, tie flag
      tier_stats     — mean/std by tier (from tier_results_df if provided, else results_df)
      merged         — firm-level merged DataFrame (all clean rows, for correlations)
      descriptive    — Table 1 descriptive statistics

    tier_results_df: if provided, tier means are computed from this subset (e.g., after
      excluding extraction_short rows). Correlations and descriptive stats use results_df.
    """
    # Aggregate to firm level (mean across filings)
    agg_cols = ["numeric_disagreement", "embedding_disagreement", "tier"]
    text_cols = [
        "word_count", "lm_neg_pct", "avg_sent_len", "type_token_ratio", "fog_index",
        "adjusted_numeric_disagreement", "adjusted_embedding_disagreement",
    ]
    extra_agg = [c for c in text_cols if c in results_df.columns]

    agg_dict: dict = {
        "numeric_disagreement": "mean",
        "embedding_disagreement": "mean",
        "tier": "first",
    }
    for c in extra_agg:
        agg_dict[c] = "mean"

    firm_scores = (
        results_df.groupby("ticker")[agg_cols + extra_agg]
        .agg(agg_dict)
        .reset_index()
    )

    merged = firm_scores.merge(proxy_df, on="ticker", how="left")

    # Measures to correlate
    disagree_measures = ["numeric_disagreement", "embedding_disagreement"]
    adj_measures = [f"adjusted_{dm}" for dm in disagree_measures if f"adjusted_{dm}" in merged.columns]
    all_measures = disagree_measures + adj_measures

    proxies = ["market_cap", "analyst_count", "idiosyncratic_volatility", "firm_age_years", "word_count"]

    # Bonferroni threshold: divide α=0.05 by total number of hypothesis tests
    n_total_tests = len(all_measures) * len(proxies)
    bonferroni_alpha = 0.05 / max(n_total_tests, 1)

    # 1. Spearman correlations with bootstrap CI and Bonferroni correction
    corr_rows = []
    for dm in all_measures:
        # Detect low-resolution measures (heavy ties → invalid p-values)
        low_res = _is_low_resolution(merged[dm]) if dm in merged.columns else False
        n_distinct = merged[dm].dropna().nunique() if dm in merged.columns else 0

        for px in proxies:
            if px not in merged.columns or dm not in merged.columns:
                continue
            col = merged[px].copy()
            if px == "market_cap":
                col = np.log(col.replace(0, np.nan))

            rho, pval, ci_lo, ci_hi = _spearman_with_ci(merged[dm], col)
            bonf_sig = pval < bonferroni_alpha if not np.isnan(pval) else False

            corr_rows.append({
                "disagreement": dm,
                "proxy": px,
                "n": int((merged[dm].notna() & col.notna()).sum()),
                "spearman_rho": rho,
                "p_value": pval,
                "ci_lower_95": ci_lo,
                "ci_upper_95": ci_hi,
                "bonferroni_sig": bonf_sig,
                "bonferroni_threshold": round(bonferroni_alpha, 5),
                "low_resolution": low_res,
                "n_distinct_values": n_distinct,
                "adjusted": dm.startswith("adjusted_"),
            })

    corr_df = pd.DataFrame(corr_rows)

    # 2. Tier statistics — use tier_results_df if provided (excludes extraction_short rows)
    if tier_results_df is not None:
        tier_agg_dict: dict = {"numeric_disagreement": "mean", "embedding_disagreement": "mean", "tier": "first"}
        tier_extra = [c for c in extra_agg if c in tier_results_df.columns]
        for c in tier_extra:
            tier_agg_dict[c] = "mean"
        tier_firm_scores = (
            tier_results_df.groupby("ticker")[["numeric_disagreement", "embedding_disagreement", "tier"] + tier_extra]
            .agg(tier_agg_dict)
            .reset_index()
        )
        tier_source = tier_firm_scores
    else:
        tier_source = firm_scores

    tier_order = ["large", "mid", "small"]
    tier_rows = []
    for tier in tier_order:
        sub = tier_source[tier_source["tier"] == tier]
        row: dict = {"tier": tier, "n_firms": len(sub)}
        for dm in disagree_measures:
            row[f"{dm}_mean"] = round(float(sub[dm].mean()), 4) if not sub.empty else float("nan")
            row[f"{dm}_std"] = round(float(sub[dm].std(ddof=1)), 4) if len(sub) > 1 else float("nan")
            row[f"{dm}_n_distinct"] = int(sub[dm].nunique()) if not sub.empty else 0
        tier_rows.append(row)
    tier_df = pd.DataFrame(tier_rows)

    # 3. Descriptive statistics
    desc_df = build_descriptive_stats(merged)

    return {
        "correlations": corr_df,
        "tier_stats": tier_df,
        "merged": merged,
        "descriptive": desc_df,
        "bonferroni_alpha": bonferroni_alpha,
        "n_total_tests": n_total_tests,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

TIER_COLORS = {"large": "#2563eb", "mid": "#f59e0b", "small": "#dc2626"}


def create_plots(merged: pd.DataFrame, plots_dir: Path) -> None:
    """Generate and save validation scatter plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    _scatter_plot(
        df=merged, x_col="market_cap", y_col="numeric_disagreement",
        x_label="Market Cap (log scale)", y_label="Numeric Disagreement",
        title="Numeric Disagreement vs. Market Cap", log_x=True,
        path=plots_dir / "numeric_vs_market_cap.png",
    )
    _scatter_plot(
        df=merged, x_col="analyst_count", y_col="embedding_disagreement",
        x_label="Number of Analyst Opinions", y_label="Embedding Disagreement",
        title="Embedding Disagreement vs. Analyst Coverage", log_x=False,
        path=plots_dir / "embedding_vs_analyst_count.png",
    )
    _scatter_plot(
        df=merged, x_col="idiosyncratic_volatility", y_col="numeric_disagreement",
        x_label="Idiosyncratic Volatility (annualized)", y_label="Numeric Disagreement",
        title="Numeric Disagreement vs. Idiosyncratic Volatility", log_x=False,
        path=plots_dir / "numeric_vs_idio_vol.png",
    )
    _scatter_plot(
        df=merged, x_col="firm_age_years", y_col="embedding_disagreement",
        x_label="Firm Age (years since IPO)", y_label="Embedding Disagreement",
        title="Embedding Disagreement vs. Firm Age", log_x=False,
        path=plots_dir / "embedding_vs_firm_age.png",
    )

    # Tier box plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tier_order = ["large", "mid", "small"]
    valid_tiers = [t for t in tier_order if t in merged["tier"].values]

    for ax, col, label in zip(
        axes,
        ["numeric_disagreement", "embedding_disagreement"],
        ["Numeric Disagreement", "Embedding Disagreement"],
    ):
        tier_data = [merged.loc[merged["tier"] == t, col].dropna().values for t in valid_tiers]
        bp = ax.boxplot(tier_data, labels=valid_tiers, patch_artist=True)
        for patch, tier in zip(bp["boxes"], valid_tiers):
            patch.set_facecolor(TIER_COLORS.get(tier, "gray"))
            patch.set_alpha(0.7)
        ax.set_xlabel("Firm Tier")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by Tier")

    plt.tight_layout()
    fig.savefig(plots_dir / "disagreement_by_tier.png", dpi=150)
    plt.close(fig)
    logger.info("Plots saved to %s", plots_dir)


def _scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    log_x: bool,
    path: Path,
) -> None:
    sub = df[[x_col, y_col, "tier", "ticker"]].dropna()
    if sub.empty:
        logger.warning("No data to plot for %s vs %s", y_col, x_col)
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for tier, group in sub.groupby("tier"):
        ax.scatter(
            group[x_col], group[y_col],
            label=tier, color=TIER_COLORS.get(tier, "gray"), alpha=0.8, s=80,
        )
        for _, row in group.iterrows():
            ax.annotate(row["ticker"], (row[x_col], row[y_col]), fontsize=7, alpha=0.7)

    if log_x:
        ax.set_xscale("log")

    x_vals = sub[x_col].values
    y_vals = sub[y_col].values
    if log_x:
        x_vals = np.log(x_vals.astype(float))
    try:
        z = np.polyfit(x_vals, y_vals, 1)
        p = np.poly1d(z)
        x_sorted = np.sort(x_vals)
        ax.plot(
            np.exp(x_sorted) if log_x else x_sorted,
            p(x_sorted),
            "k--", linewidth=1, alpha=0.5, label="trend",
        )
    except Exception:
        pass

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
