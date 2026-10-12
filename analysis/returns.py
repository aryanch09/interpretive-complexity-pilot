"""
Return prediction infrastructure — stubs for future implementation.

This module provides the scaffolding for testing whether interpretive complexity
predicts future returns, as proposed in the full paper.

Two approaches are outlined:
  1. Quintile sort: sort firms by complexity measure each period, long top
     quintile, short bottom quintile. Report mean return of each quintile.
  2. Fama-MacBeth regression: each period regress returns on lagged complexity
     and control variables; report time-series average of period coefficients.

Data sources:
  - Monthly stock returns: yfinance (free)
  - Fama-French factors: Ken French data library (CSV download)

Status: STUB. These functions are not called by the main pipeline. Wire up with
  python main.py --returns
once the complexity dataset is large enough (≥27 firms, ≥6 months of returns).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Data acquisition ──────────────────────────────────────────────────────────

def get_monthly_returns(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Download monthly total returns for a list of tickers.

    Returns a DataFrame with columns = tickers, index = month-end dates (UTC).
    Prices are adjusted for splits and dividends (auto_adjust=True).
    """
    raise NotImplementedError(
        "get_monthly_returns is not yet implemented. "
        "Plan: use yf.download(tickers, start=start, end=end, interval='1mo', auto_adjust=True), "
        "extract 'Close', compute pct_change(), resample to month-end."
    )


def get_factor_returns(start: str, end: str) -> pd.DataFrame:
    """
    Download Fama-French 3-factor monthly returns from Ken French's data library.

    Returns a DataFrame with columns [Mkt-RF, SMB, HML, RF], index = month-end dates.
    Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
    (F-F_Research_Data_Factors CSV, monthly, 1926–present, free download)
    """
    raise NotImplementedError(
        "get_factor_returns is not yet implemented. "
        "Plan: download F-F_Research_Data_Factors.zip, parse the CSV, "
        "convert YYYYMM index to month-end dates, divide values by 100."
    )


# ── Idiosyncratic return ──────────────────────────────────────────────────────

def compute_idiosyncratic_return(
    monthly_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    ticker: str,
) -> pd.Series:
    """
    Compute monthly idiosyncratic return as the residual from a CAPM regression.

    For each rolling 12-month window, regress stock returns on market excess return.
    The residual series is the idiosyncratic component used as the dependent variable
    in return prediction tests.
    """
    raise NotImplementedError(
        "compute_idiosyncratic_return is not yet implemented. "
        "Plan: align monthly_returns[ticker] with factor_returns['Mkt-RF'] + factor_returns['RF'], "
        "run rolling OLS (window=12), extract residuals."
    )


# ── Cross-sectional tests ─────────────────────────────────────────────────────

def quintile_sort(
    firm_data: pd.DataFrame,
    complexity_col: str,
    forward_return_col: str,
    n_quintiles: int = 5,
) -> pd.DataFrame:
    """
    Sort firms into quintiles by complexity measure each period; compute mean
    forward return per quintile.

    Args:
        firm_data: DataFrame with columns [date, ticker, complexity_col, forward_return_col]
        complexity_col: 'numeric_disagreement' or 'embedding_disagreement'
        forward_return_col: name of the column containing the next-period return

    Returns:
        DataFrame with columns [quintile, mean_return, n_firms, t_stat] and a
        long_short row for quintile 5 − quintile 1.
    """
    raise NotImplementedError(
        "quintile_sort is not yet implemented. "
        "Plan: pd.qcut each period's complexity into n_quintiles bins, "
        "compute mean forward_return per bin, scipy.stats.ttest_1samp for t-stats, "
        "append long_short row as Q5 - Q1."
    )


def fama_macbeth(
    firm_data: pd.DataFrame,
    complexity_col: str,
    controls: Optional[list[str]] = None,
    forward_return_col: str = "fwd_return_1m",
) -> pd.DataFrame:
    """
    Fama-MacBeth (1973) regression of forward returns on lagged complexity.

    Each period: regress cross-section of returns on [complexity, controls].
    Final output: time-series mean and Newey-West std error of period coefficients.

    Args:
        firm_data: panel with [date, ticker, complexity_col, *controls, forward_return_col]
        complexity_col: main independent variable
        controls: optional list of additional regressors (e.g., ['log_market_cap', 'momentum'])
        forward_return_col: dependent variable column name

    Returns:
        DataFrame with columns [variable, coefficient, std_error, t_stat, p_value]
        Time-series average of cross-sectional OLS coefficients.
        Std errors are Newey-West adjusted for serial correlation (lag=3).
    """
    raise NotImplementedError(
        "fama_macbeth is not yet implemented. "
        "Plan: group by date, run OLS each period (numpy.linalg.lstsq or statsmodels), "
        "stack period coefficients into a time series, compute mean and Newey-West SE "
        "using statsmodels.stats.sandwich_covariance or manual Bartlett kernel."
    )


# ── Summary table ─────────────────────────────────────────────────────────────

def summarize_return_tests(
    quintile_results: pd.DataFrame,
    fmb_results: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Write a summary table combining quintile sort and Fama-MacBeth results
    to a text file for inclusion in the paper.
    """
    raise NotImplementedError(
        "summarize_return_tests is not yet implemented. "
        "Plan: format quintile_results and fmb_results as aligned text tables, "
        "write to output_path."
    )
