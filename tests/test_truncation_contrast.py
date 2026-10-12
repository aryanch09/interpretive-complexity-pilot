"""
Tests for experiments/truncation_contrast/run.py (S1(a)).

These are recovery tests, not smoke tests.  Each one plants a known structure in
a synthetic window-level frame (tests/fixtures_truncation.py) and asserts that the analysis
returns the planted answer — and, just as important, that it returns *no* effect
when none was planted.

Run:
    python3 -m pytest tests/ -q
    python3 -m tests.test_truncation_contrast

No network, no API, no model.  Everything is seeded.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.fixtures_truncation import SCHEMA, make_fixture  # noqa: E402


def _load_run_module():
    """Load the experiment by path; it lives outside any package."""
    path = REPO / "experiments" / "truncation_contrast" / "run.py"
    spec = importlib.util.spec_from_file_location("truncation_contrast_run", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tc = _load_run_module()

SEED = 20261004
NB = 1000  # bootstrap draws in tests; the shipped default is 10,000


def grad_table(res, measure="embedding_disagreement", variant="inclusive"):
    g = res["gradients"]
    return g[(g["measure"] == measure) & (g["variant"] == variant)].set_index("k")


def tier_table(res, measure="embedding_disagreement", variant="inclusive"):
    t = res["tiers"]
    return t[(t["measure"] == measure) & (t["variant"] == variant)].set_index(["k", "tier"])


# Criterion used below to decide whether a truncation budget MATERIALLY changed
# the answer.  Both halves are required, and the same rule is applied to the
# fixture with a planted position effect and to the flat one — a rule that only
# fires on the fixture it was written for proves nothing.
#   (i)  the paired bootstrap CI for (gradient at k - gradient at all) excludes 0
#   (ii) the change is at least 5% of the complete-section gradient
# (ii) is needed because (i) alone will fire on a change of no substantive size:
# averaging within-filing measurement noise over 30 firms makes the paired CI very
# tight, so a 2% shift can clear it.
MATERIAL_PCT = 5.0


def is_material(row) -> bool:
    excludes_zero = not (row["delta_ci_lo"] <= 0 <= row["delta_ci_hi"])
    return bool(excludes_zero and abs(row["delta_vs_all_pct_of_complete"]) >= MATERIAL_PCT)


def run(df, **kw):
    kw.setdefault("seed", SEED)
    kw.setdefault("n_boot", NB)
    return tc.run_truncation_contrast(df, **kw)


# ── 1. The fixture is shaped like the real file ───────────────────────────────

def test_fixture_has_the_exact_real_schema():
    df, _ = make_fixture()
    assert tuple(df.columns) == SCHEMA
    assert set(df["tier"]) <= {"large", "mid", "small"}
    assert set(df["filing_type"]) <= {"10-K", "10-Q"}
    assert set(df["section"]) <= {"mda", "risk"}
    assert df["word_count"].between(403, 1598).all()
    assert df["filing_date"].astype(str).str.fullmatch(r"\d{8}").all()
    # chunk_index is 0-based and contiguous inside each (filing, section)
    for _, sub in df.groupby(["ticker", "filing_type", "section"]):
        idx = sorted(sub["chunk_index"])
        assert idx == list(range(len(idx)))
        assert set(sub["n_in_section"]) == {len(idx)}


def test_fixture_resembles_the_real_sample_shape():
    df, _ = make_fixture()
    mda = df[df["section"] == "mda"]
    per_filing = mda.groupby(["ticker", "filing_type"]).size()
    assert df.groupby(["ticker", "filing_type"]).ngroups == 59  # 30 firms, HAYN-like
    assert per_filing.min() >= 2
    assert per_filing.max() <= 25


# ── 2. The aggregation is the pre-registered one, on numbers done by hand ─────

def _hand_frame() -> pd.DataFrame:
    """
    Six filings with window values chosen so that the pre-registered
    window -> filing -> firm -> tier chain gives a DIFFERENT answer from a pooled
    mean over windows.  The expected numbers in the test below are arithmetic done
    by hand, not produced by this code.

    Risk windows with absurd values are included to prove they never enter.
    """
    spec = [
        ("LRG01", "large", "10-K", [0.1, 0.1, 0.1, 0.7]),
        ("LRG01", "large", "10-Q", [0.5, 0.5]),
        ("LRG02", "large", "10-K", [0.2, 0.4]),
        ("MID01", "mid", "10-K", [0.4, 0.4]),
        ("MID02", "mid", "10-K", [0.5, 0.5]),
        ("SML01", "small", "10-K", [0.6, 0.8]),
        ("SML02", "small", "10-Q", [0.5, 0.5, 0.9]),
    ]
    rows = []
    for ticker, tier, form, vals in spec:
        for section, series in (("mda", vals), ("risk", [9.0] * len(vals))):
            for i, v in enumerate(series):
                rows.append({
                    "ticker": ticker, "tier": tier, "filing_type": form,
                    "filing_date": "20250101", "section": section, "chunk_index": i,
                    "n_in_section": len(series), "word_count": 1200,
                    "numeric_disagreement": 1.0 - v, "embedding_disagreement": v,
                    "value_score": 5.0, "sentiment_score": 5.0,
                    "fundamentals_score": 5.0, "any_failed": False,
                    "any_parse_error": False, "served_by": "DeepInfra",
                    "value_reasoning": "x", "sentiment_reasoning": "y",
                    "fundamentals_reasoning": "z",
                })
    return pd.DataFrame.from_records(rows, columns=list(SCHEMA))


def test_aggregation_matches_hand_computation():
    res = run(_hand_frame())
    t = tier_table(res)

    # complete section:
    #   LRG01 = mean(mean(.1,.1,.1,.7)=.25, mean(.5,.5)=.5) = .375 ; LRG02 = .3
    #   large = mean(.375, .3) = .3375      (pooled window mean would be .325)
    #   mid   = mean(.4, .5)   = .45
    #   SML01 = .7 ; SML02 = mean(.5,.5,.9) = .6333... ; small = .6666...
    assert t.loc[("all", "large"), "tier_mean"] == pytest.approx(0.3375)
    assert t.loc[("all", "mid"), "tier_mean"] == pytest.approx(0.45)
    assert t.loc[("all", "small"), "tier_mean"] == pytest.approx(2 / 3)

    g = grad_table(res)
    assert g.loc["all", "gradient_small_minus_large"] == pytest.approx(2 / 3 - 0.3375)

    # k = 2: large = mean(mean(.1,.5), .3) = .3 ; small = mean(.7, .5) = .6
    assert t.loc[("2", "large"), "tier_mean"] == pytest.approx(0.3)
    assert t.loc[("2", "small"), "tier_mean"] == pytest.approx(0.6)
    assert g.loc["2", "gradient_small_minus_large"] == pytest.approx(0.3)

    # k = 1: large = mean(mean(.1,.5), .2) = .25 ; mid = .45 ; small = mean(.6,.5) = .55
    assert t.loc[("1", "large"), "tier_mean"] == pytest.approx(0.25)
    assert t.loc[("1", "mid"), "tier_mean"] == pytest.approx(0.45)
    assert t.loc[("1", "small"), "tier_mean"] == pytest.approx(0.55)

    # The numeric column is 1 - embedding here, so a swapped column would show up
    # as the wrong sign.
    gn = grad_table(res, measure="numeric_disagreement")
    assert gn.loc["all", "gradient_small_minus_large"] == pytest.approx(0.3375 - 2 / 3)


def test_risk_windows_are_never_pooled_into_the_primary():
    # The hand frame's risk windows are all 9.0; if any leaked in, the MD&A means
    # above would be far larger.  Re-check directly against the no-risk frame.
    full = _hand_frame()
    mda_only = full[full["section"] == "mda"].copy()
    a = grad_table(run(full))["gradient_small_minus_large"]
    b = grad_table(run(mda_only))["gradient_small_minus_large"]
    pd.testing.assert_series_equal(a, b)

    df, truth = make_fixture(include_risk=True, risk_base={"large": 0.9, "mid": 0.5, "small": 0.1})
    got = grad_table(run(df)).loc["all", "gradient_small_minus_large"]
    assert got == pytest.approx(truth["gradients"]["embedding_disagreement"][None])


# ── 3. Recovery of a planted tier effect ──────────────────────────────────────

def test_recovers_planted_tier_effect_at_every_budget():
    """Flat in window index, +0.07 small-vs-large planted: every k must return it."""
    df, truth = make_fixture(
        embedding_base={"large": 0.26, "mid": 0.30, "small": 0.33},
        embedding_slope=0.0, embedding_noise_sd=0.0,
    )
    res = run(df)
    g = grad_table(res)
    for k in ("1", "2", "4", "8", "all"):
        key = None if k == "all" else int(k)
        assert g.loc[k, "gradient_small_minus_large"] == pytest.approx(
            truth["gradients"]["embedding_disagreement"][key], abs=1e-12
        ) == pytest.approx(0.07, abs=1e-12)
        assert g.loc[k, "ci_lo"] > 0, f"CI must exclude 0 at k={k}"
        assert bool(g.loc[k, "monotone_small_gt_mid_gt_large"])


def test_recovers_planted_effect_with_noise():
    df, truth = make_fixture(
        embedding_base={"large": 0.26, "mid": 0.30, "small": 0.33},
        embedding_slope=0.0, embedding_noise_sd=0.03, seed=7,
    )
    res = run(df)
    g = grad_table(res)
    for k in ("1", "2", "4", "8", "all"):
        key = None if k == "all" else int(k)
        got = g.loc[k, "gradient_small_minus_large"]
        # matches the independently computed oracle exactly ...
        assert got == pytest.approx(truth["gradients"]["embedding_disagreement"][key], abs=1e-12)
        # ... and is close to, and significantly above, the planted 0.07
        assert abs(got - 0.07) < 0.02
        assert g.loc[k, "ci_lo"] > 0


# ── 4. No effect when none is planted ─────────────────────────────────────────

def test_finds_no_effect_when_none_is_planted():
    df, _ = make_fixture(
        embedding_base={"large": 0.30, "mid": 0.30, "small": 0.30},
        embedding_slope=0.0, embedding_noise_sd=0.03, seed=11,
    )
    g = grad_table(run(df))
    for k in ("1", "2", "4", "8", "all"):
        assert abs(g.loc[k, "gradient_small_minus_large"]) < 0.02
        assert g.loc[k, "ci_lo"] <= 0 <= g.loc[k, "ci_hi"], f"false positive at k={k}"
        assert not bool(g.loc[k, "monotone_small_gt_mid_gt_large"])


def test_null_rejection_rate_is_near_nominal():
    """
    Calibration, not a single draw: over 12 independent null fixtures the 95%
    bootstrap CI for the complete-section gradient should exclude 0 only rarely.
    A procedure that fires on most null fixtures would pass the single-seed test
    above by luck; this catches it.
    """
    excl = 0
    for s in range(12):
        df, _ = make_fixture(
            embedding_base={"large": 0.30, "mid": 0.30, "small": 0.30},
            embedding_slope=0.0, embedding_noise_sd=0.03, seed=100 + s,
        )
        g = grad_table(run(df, n_boot=600))
        lo, hi = g.loc["all", "ci_lo"], g.loc["all", "ci_hi"]
        excl += int(not (lo <= 0 <= hi))
    assert excl <= 3, f"{excl}/12 null fixtures produced a CI excluding 0"


# ── 5. The experiment's own claim: does truncation change the answer? ─────────

def test_detects_disagreement_rising_with_window_index():
    """
    Plant: disagreement rises with window index.  Truncating to the first k
    windows must therefore UNDERSTATE disagreement, and must understate it
    unevenly across tiers (because window counts differ by tier), so the measured
    gradient at small k must differ from the complete-section gradient.
    """
    df, truth = make_fixture(
        embedding_base={"large": 0.26, "mid": 0.30, "small": 0.33},
        embedding_slope=0.01, embedding_noise_sd=0.0,
        n_firms_per_tier=20, seed=3,
    )
    res = run(df)
    t = tier_table(res)
    g = grad_table(res)

    # (i) every tier's mean rises monotonically with the budget: truncation to
    #     early windows understates disagreement.
    for tier in ("large", "mid", "small"):
        means = [t.loc[(k, tier), "tier_mean"] for k in ("1", "2", "4", "8", "all")]
        assert all(b > a for a, b in zip(means, means[1:])), (tier, means)
        assert means[0] < means[-1] - 0.01

    # (ii) the oracle agrees, number for number
    for k in ("1", "2", "4", "8", "all"):
        key = None if k == "all" else int(k)
        assert g.loc[k, "gradient_small_minus_large"] == pytest.approx(
            truth["gradients"]["embedding_disagreement"][key], abs=1e-12)

    # (iii) the measured gradient differs materially from the complete-section
    #       gradient at every truncated budget: truncation changes the answer.
    for k in ("1", "2", "4", "8"):
        assert is_material(g.loc[k]), (
            k, g.loc[k, ["delta_vs_all", "delta_vs_all_pct_of_complete",
                         "delta_ci_lo", "delta_ci_hi"]].to_dict()
        )
    assert abs(g.loc["1", "delta_vs_all_pct_of_complete"]) > 15.0
    assert g.loc["all", "delta_vs_all"] == 0.0


def test_flat_in_window_index_gives_a_flat_truncation_curve():
    """The control: no position structure, so the curve must not move."""
    df, _ = make_fixture(
        embedding_base={"large": 0.26, "mid": 0.30, "small": 0.33},
        embedding_slope=0.0, embedding_noise_sd=0.02,
        n_firms_per_tier=20, seed=3,
    )
    res = run(df)
    t = tier_table(res)
    g = grad_table(res)

    for tier in ("large", "mid", "small"):
        means = np.array([t.loc[(k, tier), "tier_mean"] for k in ("1", "2", "4", "8", "all")])
        assert means.max() - means.min() < 0.012, (tier, means)

    # No budget registers as a material change under the same rule that fires at
    # every budget on the position-effect fixture.
    for k in ("1", "2", "4", "8"):
        assert not is_material(g.loc[k]), (
            k, g.loc[k, ["delta_vs_all", "delta_vs_all_pct_of_complete",
                         "delta_ci_lo", "delta_ci_hi"]].to_dict()
        )
    spread = (g["gradient_small_minus_large"].max()
              - g["gradient_small_minus_large"].min())
    assert spread < 0.012, spread


def test_flat_fixture_with_no_noise_is_exactly_flat():
    df, _ = make_fixture(embedding_slope=0.0, embedding_noise_sd=0.0)
    g = grad_table(run(df))["gradient_small_minus_large"]
    assert g.max() - g.min() == pytest.approx(0.0, abs=1e-12)


# ── 6. Pre-registered data rules ──────────────────────────────────────────────

def test_failed_windows_are_dropped_never_imputed():
    df, truth = make_fixture(
        embedding_slope=0.01, embedding_noise_sd=0.0, n_failed_windows=25, seed=5,
    )
    res = run(df)
    meta = res["meta"]
    assert meta["n_windows_failed_in_section"] == truth["n_failed_mda"] == 25

    g = grad_table(res)
    for k in ("1", "2", "4", "8", "all"):
        key = None if k == "all" else int(k)
        # the oracle recomputes the means with those windows removed; imputing
        # (e.g. filling 0 or the filing mean) would not match
        assert g.loc[k, "gradient_small_minus_large"] == pytest.approx(
            truth["gradients"]["embedding_disagreement"][key], abs=1e-12)
        assert np.isfinite(g.loc[k, "mean_large"])
        assert g.loc[k, "n_windows_used"] + g.loc[k, "n_windows_dropped_failed"] == \
            g.loc[k, "n_windows_in_budget"]
    assert g.loc["all", "n_windows_dropped_failed"] > 0


def test_filing_failing_the_usability_gate_is_excluded_once_for_all_budgets():
    """
    The §8.2 gate (>= 2 usable windows and >= half usable) is evaluated on the
    COMPLETE section, so a filing that fails it contributes to no budget at all.
    """
    df, _ = make_fixture(embedding_noise_sd=0.0)
    # kill one window of a 2-window filing: 1 of 2 survives -> fails the gate
    mda = df[df["section"] == "mda"]
    counts = mda.groupby(["ticker", "filing_type"]).size()
    victim = counts[counts == 2].index[0]
    mask = (
        (df["ticker"] == victim[0]) & (df["filing_type"] == victim[1])
        & (df["section"] == "mda") & (df["chunk_index"] == 0)
    )
    df.loc[mask, "any_failed"] = True
    df.loc[mask, ["numeric_disagreement", "embedding_disagreement"]] = np.nan

    res = run(df)
    assert res["meta"]["n_filings_gate_failed"] == 1
    assert any(str(victim[0]) in f for f in res["meta"]["filings_gate_failed"])
    fl = res["firm_level"]
    for k in ("1", "2", "4", "8", "all"):
        sub = fl[(fl["variant"] == "inclusive") & (fl["k"] == k)]
        n_filings = sub["n_filings"].sum()
        assert n_filings == 58, (k, n_filings)


def test_null_disagreement_without_the_flag_is_still_dropped():
    """Second gate: a null measure is excluded even if any_failed says False."""
    df, _ = make_fixture(embedding_slope=0.01, embedding_noise_sd=0.0)
    flagged = df.copy()
    target = (flagged["section"] == "mda") & (flagged["chunk_index"] == 1)
    flagged.loc[target, ["numeric_disagreement", "embedding_disagreement"]] = np.nan
    flagged.loc[target, "any_failed"] = True
    silent = flagged.copy()
    silent.loc[target, "any_failed"] = False  # flag lost, values still null
    a = grad_table(run(flagged))["gradient_small_minus_large"]
    b = grad_table(run(silent))["gradient_small_minus_large"]
    pd.testing.assert_series_equal(a, b)


# ── 7. Budget rules, coverage and k = 1 ───────────────────────────────────────

def test_inclusive_and_strict_budget_rules():
    df, truth = make_fixture(embedding_slope=0.01, embedding_noise_sd=0.0, seed=2)
    res = run(df)
    counts = pd.Series(truth["window_counts"])

    for k in (1, 2, 4, 8):
        expected_has_k = int((counts >= k).sum())
        gi = grad_table(res).loc[str(k)]
        gs = grad_table(res, variant="strict").loc[str(k)]
        assert gi["n_filings_with_at_least_k"] == expected_has_k
        # inclusive keeps every filing; strict keeps only those with >= k windows
        assert gi["n_filings_contributing"] == len(counts)
        assert gs["n_filings_contributing"] == expected_has_k
        assert gs["n_filings_dropped_by_strict"] == len(counts) - expected_has_k
        if expected_has_k < len(counts):
            # the two variants genuinely differ — that difference is the finding
            assert gi["gradient_small_minus_large"] != gs["gradient_small_minus_large"]

    # and the strict drops are not tier-neutral: short filings are small-cap
    t = tier_table(res, variant="strict")
    assert t.loc[("8", "small"), "n_firms"] <= t.loc[("8", "large"), "n_firms"]


def test_coverage_reproduces_the_size_coverage_confound():
    df, truth = make_fixture(embedding_noise_sd=0.0)
    res = run(df)
    g = grad_table(res)
    for k in ("1", "2", "4", "8"):
        for tier in ("large", "mid", "small"):
            f = g.loc[k, f"mean_frac_read_{tier}"]
            assert 0 < f <= 1
            key = None if k == "all" else int(k)
            assert f == pytest.approx(truth["frac_read"][key][tier], abs=1e-12)
        # a fixed budget reads a smaller fraction of the (longer) large-cap MD&A
        assert g.loc[k, "mean_frac_read_small"] > g.loc[k, "mean_frac_read_large"]
        assert g.loc[k, "spearman_tierrank_fracread_rho"] < 0
    for tier in ("large", "mid", "small"):
        assert g.loc["all", f"mean_frac_read_{tier}"] == pytest.approx(1.0)
    # at k = all every firm reads 100%: no variation, so no correlation is defined
    assert np.isnan(g.loc["all", "spearman_tierrank_fracread_rho"])


def test_k1_is_handled_explicitly():
    df, _ = make_fixture(embedding_noise_sd=0.0)
    sec = tc.section_frame(tc.prepare(df))
    inv = tc.filing_inventory(sec)
    filings, diag = tc.filing_means_at_k(sec, inv, k=1, strict=False)

    # every filing contributes exactly one window; none is dropped by a gate
    assert (filings["n_windows_used"] == 1).all()
    assert diag["n_filings_contributing"] == len(inv)
    # the within-filing SD is undefined on one window: NaN, never 0
    assert filings["within_filing_sd_embedding"].isna().all()
    # ... and the within-budget reading of the gate is reported, not silently used
    assert diag["n_filings_failing_within_budget_gate"] == len(inv)

    g = grad_table(run(df))
    assert not bool(g.loc["1", "within_filing_sd_defined"])
    assert bool(g.loc["all", "within_filing_sd_defined"])


def test_incomplete_window_file_is_flagged(caplog):
    """
    A run interrupted mid-filing writes only the windows it finished.  Silently
    averaging over those would be truncation by accident — the exact failure this
    experiment studies — so it must be flagged.
    """
    df, _ = make_fixture(embedding_noise_sd=0.0)
    mda = df[df["section"] == "mda"]
    counts = mda.groupby(["ticker", "filing_type"]).size()
    victim = counts[counts >= 4].index[0]
    drop = (
        (df["ticker"] == victim[0]) & (df["filing_type"] == victim[1])
        & (df["section"] == "mda")
        & (df["chunk_index"] == int(counts[victim]) - 1)
    )
    truncated = df[~drop]

    with caplog.at_level("WARNING", logger="truncation_contrast"):
        res = run(truncated)
    assert any("window-count mismatch" in r.message for r in caplog.records)
    inv = res["inventory"]
    row = inv[inv["ticker"] == victim[0]].iloc[0]
    assert row["n_windows"] == row["n_windows_declared"] - 1


# ── 8. Determinism, provenance, and no side channels ──────────────────────────

def test_seed_is_exposed_recorded_and_respected():
    df, _ = make_fixture(embedding_noise_sd=0.03, seed=4)
    a = run(df, seed=123)
    b = run(df, seed=123)
    c = run(df, seed=999)

    pd.testing.assert_frame_equal(a["gradients"], b["gradients"])
    assert (a["gradients"]["seed"] == 123).all()
    assert (c["gradients"]["seed"] == 999).all()
    # the point estimate does not depend on the seed; the CI does
    ga, gc = grad_table(a), grad_table(c)
    assert ga["gradient_small_minus_large"].equals(gc["gradient_small_minus_large"])
    assert ga.loc["all", "ci_lo"] != gc.loc["all", "ci_lo"]
    # ... and the two seeds' CIs agree to within Monte-Carlo error
    assert abs(ga.loc["all", "ci_lo"] - gc.loc["all", "ci_lo"]) < 0.01


def test_no_network_or_model_modules_are_imported():
    for mod in ("requests", "openai", "groq", "transformers", "torch", "yfinance"):
        assert mod not in sys.modules, f"{mod} was imported by the experiment"
    assert sys.modules.get("tensorflow") is None  # the abseil deadlock guard


def test_writes_a_vector_pdf_and_tidy_csvs(tmp_path):
    df, _ = make_fixture(embedding_slope=0.005, embedding_noise_sd=0.01, seed=6)
    fixture_csv = tmp_path / "chunk_level_results.csv"
    df.to_csv(fixture_csv, index=False)
    out = tmp_path / "out"
    rc = tc.main([
        "--input", str(fixture_csv), "--out-dir", str(out),
        "--n-boot", "300", "--seed", "20261004", "--quiet",
    ])
    assert rc == 0
    pdf = out / "truncation_contrast.pdf"
    assert pdf.exists() and pdf.read_bytes()[:5] == b"%PDF-"

    grads = pd.read_csv(out / "truncation_contrast_gradients.csv")
    tiers = pd.read_csv(out / "truncation_contrast_tiers.csv")
    assert len(grads) == 2 * 2 * 5          # measures x variants x budgets
    assert len(tiers) == 2 * 2 * 5 * 3      # ... x tiers
    assert set(grads["k"]) == {"1", "2", "4", "8", "all"}
    assert (grads["seed"] == 20261004).all() and (grads["n_boot"] == 300).all()
    # numeric is reported before embedding (ANALYSIS_PLAN.md §3)
    assert list(grads["measure"].unique())[0] == "numeric_disagreement"
    for f in ("truncation_contrast_firm_level.csv",
              "truncation_contrast_diagnostics.csv", "truncation_contrast.txt"):
        assert (out / f).exists()
    assert "SIMULATED truncation" in (out / "truncation_contrast.txt").read_text()


def test_missing_input_fails_loudly_rather_than_inventing_numbers(tmp_path):
    with pytest.raises(SystemExit) as e:
        tc.load_window_level(tmp_path / "nope.csv")
    assert "no model calls" in str(e.value)


if __name__ == "__main__":  # python3 -m tests.test_truncation_contrast
    raise SystemExit(pytest.main([str(Path(__file__)), "-q"]))
