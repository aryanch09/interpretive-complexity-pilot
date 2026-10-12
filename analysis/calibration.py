"""
Temperature noise calibration: separate persona signal from sampling randomness.

Problem (identified in pilot review):
  With temperature > 0, re-running the same agent on the same text produces
  different scores. The numeric_disagreement measure is std(value_score,
  sentiment_score, fundamentals_score). If intra-agent variance (same prompt,
  different runs) is comparable to inter-agent variance (different prompts, same
  run), then the disagreement measure is dominated by temperature noise rather
  than persona signal, and the measure has no construct validity.

This module provides the calibration test to quantify the signal-to-noise ratio.

Test design:
  1. Take a single filing text
  2. Run the SAME agent prompt N times (default 5)
  3. Compute intra-agent score std (temperature noise baseline)
  4. Compare to inter-agent score std from the main pipeline run
  5. Signal-to-noise ratio = inter_agent_std / intra_agent_std
     SNR > 2.0: persona is driving more than half the variance → signal is real
     SNR < 1.5: measure is mostly noise → report with caution

Usage:
  python main.py --calibrate --firms AAPL,NGVC
  (runs calibration on one large-cap and one small-cap filing)

Output: output/calibration_report.txt
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

logger = logging.getLogger(__name__)


def run_intra_agent_calibration(
    filing_text: str,
    config: Config,
    agent_role: str = "value",
    n_runs: int = 5,
) -> dict:
    """
    Run the SAME agent prompt n_runs times on the same filing text.

    Returns:
      scores        — list of n_runs scores
      score_mean    — mean score across runs
      score_std     — std of scores (temperature noise baseline)
      parse_errors  — number of runs with parse errors
      raw_outputs   — list of raw agent responses (for debugging)
    """
    from agents.prompts import AGENTS
    from agents.runner import run_agent

    if agent_role not in AGENTS:
        raise ValueError(f"Unknown agent role '{agent_role}'. Choose from: {list(AGENTS)}")

    meta = AGENTS[agent_role]
    scores = []
    raw_outputs = []
    parse_errors = 0

    for i in range(n_runs):
        logger.info("  Calibration run %d/%d (%s) ...", i + 1, n_runs, meta["name"])
        result = run_agent(
            system_prompt=meta["prompt"],
            filing_text=filing_text,
            agent_name=f"{meta['name']}_calibration_run{i+1}",
            config=config,
        )
        raw_outputs.append(result.get("raw_output", ""))
        if result.get("score") is not None and not result.get("failed"):
            scores.append(result["score"])
        else:
            parse_errors += 1

    if not scores:
        logger.warning("Calibration: no valid scores in %d runs for %s", n_runs, agent_role)
        return {
            "agent_role": agent_role,
            "n_runs": n_runs,
            "scores": [],
            "score_mean": None,
            "score_std": None,
            "parse_errors": parse_errors,
            "raw_outputs": raw_outputs,
        }

    return {
        "agent_role": agent_role,
        "n_runs": n_runs,
        "scores": scores,
        "score_mean": round(float(np.mean(scores)), 3),
        "score_std": round(float(np.std(scores, ddof=0)), 4),
        "parse_errors": parse_errors,
        "raw_outputs": raw_outputs,
    }


def run_full_calibration(
    filing_text: str,
    config: Config,
    n_runs: int = 5,
) -> dict:
    """
    Run calibration for all three agent roles on a single filing.

    Returns a dict with per-role results and an overall SNR estimate.
    SNR is computed after the main pipeline has also been run on this
    filing (pass inter_agent_std to compute_snr).
    """
    results = {}
    for role in ["value", "sentiment", "fundamentals"]:
        results[role] = run_intra_agent_calibration(filing_text, config, agent_role=role, n_runs=n_runs)

    # Pool intra-agent stds across all three roles
    stds = [r["score_std"] for r in results.values() if r["score_std"] is not None]
    pooled_noise_std = float(np.mean(stds)) if stds else None

    return {
        "per_role": results,
        "pooled_intra_agent_std": pooled_noise_std,
        "n_runs_per_role": n_runs,
        "interpretation": _interpret_noise(pooled_noise_std),
    }


def compute_snr(inter_agent_std: float, intra_agent_std: Optional[float]) -> Optional[float]:
    """
    Signal-to-noise ratio: inter-agent std / intra-agent std.
    SNR > 2.0 → persona drives more variance than temperature noise.
    SNR < 1.5 → measure is dominated by temperature randomness.
    """
    if intra_agent_std is None or intra_agent_std == 0:
        return None
    return round(inter_agent_std / intra_agent_std, 3)


def _interpret_noise(intra_std: Optional[float]) -> str:
    if intra_std is None:
        return "could not estimate intra-agent noise (parse errors)"
    if intra_std < 0.3:
        return "LOW noise (std < 0.3): temperature barely affects scores — persona signal should dominate"
    if intra_std < 0.7:
        return "MODERATE noise (std 0.3–0.7): persona signal likely dominates but check SNR per filing"
    return "HIGH noise (std > 0.7): temperature variance is large — SNR check required before interpreting disagreement"


def write_calibration_report(
    calibration_results: dict,
    inter_agent_std: Optional[float],
    ticker: str,
    output_path,
) -> None:
    """Write a plain-text calibration report to output_path."""
    lines = [
        "Calibration Report — Temperature Noise Baseline",
        f"Ticker: {ticker}",
        f"Runs per role: {calibration_results['n_runs_per_role']}",
        "",
        "Intra-agent score std (temperature noise) by role:",
    ]

    for role, res in calibration_results["per_role"].items():
        if res["score_std"] is not None:
            lines.append(
                f"  {role:<14}: std = {res['score_std']:.4f}  "
                f"(scores: {[round(s, 2) for s in res['scores']]})"
            )
        else:
            lines.append(f"  {role:<14}: could not compute (parse errors: {res['parse_errors']})")

    pooled = calibration_results["pooled_intra_agent_std"]
    lines.append("")
    lines.append(f"Pooled intra-agent std (temperature noise): {pooled:.4f}" if pooled else "Pooled: N/A")

    if inter_agent_std is not None and pooled is not None:
        snr = compute_snr(inter_agent_std, pooled)
        lines.append(f"Inter-agent std (persona signal, this filing): {inter_agent_std:.4f}")
        lines.append(f"Signal-to-noise ratio (SNR): {snr}")
        if snr is not None:
            if snr >= 2.0:
                lines.append("  → SNR ≥ 2.0: persona signal dominates temperature noise (GOOD)")
            elif snr >= 1.5:
                lines.append("  → SNR 1.5–2.0: marginal signal, interpret with caution")
            else:
                lines.append("  → SNR < 1.5: temperature noise dominates — disagreement measure is unreliable")

    lines.append("")
    lines.append(calibration_results["interpretation"])

    from pathlib import Path
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    for line in lines:
        logger.info("CALIBRATION: %s", line)
