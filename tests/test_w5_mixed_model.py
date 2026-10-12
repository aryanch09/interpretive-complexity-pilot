"""
Tests for analysis/mixed_model.py (W5) against synthetic data with known truth.

These are recovery tests, not smoke tests.  Each one plants a specific truth in
`tests/fixtures_w5.py` and asserts that W5 recovers it, or -- just as important --
that W5 does NOT find an effect that was never planted.  A test that only checks
"runs without error" would not catch the failure mode that matters here: an
estimator that manufactures a tier gradient out of the window-count imbalance.

All seeds are fixed, so every assertion is deterministic: there is no flaky
tolerance in this file.  Tolerances are stated absolutely AND paired with a
CI-coverage assertion, which is the statistically meaningful form of "recovered".

No network calls, no model calls.

Run:
    python3 -m pytest tests/ -q
    python3 -m tests.test_w5_mixed_model        # same assertions, prints estimates
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import mixed_model as mm  # noqa: E402
from tests.fixtures_w5 import (  # noqa: E402
    CHUNK_SCHEMA,
    CHUNK_WORDS,
    generate_chunk_level_fixture,
    position_decile,
)

SPEC_KEYS = ("nested", "firm_only", "ols_cluster")

# Collected by the recovery tests and printed by __main__ so the numbers in the
# write-up are the ones the tests actually produced.
RECOVERED: list[dict] = []


# ────────────────────────────── helpers ──────────────────────────────────────


def _write_fixture(tmpdir: str, **kwargs) -> tuple[Path, dict]:
    path = Path(tmpdir) / "chunk_level_results.csv"
    _, truth = generate_chunk_level_fixture(out_path=path, **kwargs)
    return path, truth


def _record(test: str, spec: str, term: str, planted: float, c: dict) -> None:
    RECOVERED.append({
        "test": test, "spec": spec, "term": term, "planted": planted,
        "estimate": c["estimate"], "se": c["se"],
        "ci_low": c["ci_low"], "ci_high": c["ci_high"], "p": c["p"],
    })


def _assert_recovered(
    test: str, spec: str, term: str, planted: float, c: dict, tol: float
) -> None:
    """Recovery = the 95% CI covers the planted value AND the point estimate is
    close to it.

    CI coverage is the statistically meaningful half: an unbiased estimator with
    honest standard errors covers the truth, and a single point estimate is
    never expected to equal it.  The point check is a guard against a grossly
    wrong estimate, so its bound is `max(tol, 3 * se)`: a stated absolute
    tolerance, widened when the design itself cannot pin the coefficient down
    (one draw of a coefficient with a large SE being 2 SE from the truth is
    ordinary sampling variation, not an estimator defect).
    """
    _record(test, spec, term, planted, c)
    bound = max(tol, 3.0 * c["se"])
    assert abs(c["estimate"] - planted) <= bound, (
        f"{test}/{spec}: {term} estimate {c['estimate']:+.4f} is further than "
        f"{bound:.4f} (= max({tol}, 3*se={3 * c['se']:.4f})) from the planted "
        f"{planted:+.4f}"
    )
    assert c["ci_low"] <= planted <= c["ci_high"], (
        f"{test}/{spec}: {term} 95% CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] "
        f"does not cover the planted {planted:+.4f}"
    )


# ───────────────────────── fixture integrity tests ───────────────────────────


def test_fixture_has_the_exact_real_schema() -> None:
    """The synthetic file must be column-for-column the real file, in order."""
    with tempfile.TemporaryDirectory() as td:
        path, truth = _write_fixture(td, seed=11)
        df = pd.read_csv(path)

    assert tuple(df.columns) == CHUNK_SCHEMA, (
        f"column mismatch:\n got      {tuple(df.columns)}\n expected {CHUNK_SCHEMA}"
    )
    # Shape matched to the pre-registration sample.
    assert truth["realised"]["n_firms"] == 30
    assert truth["realised"]["n_filings"] == 59
    assert set(df["tier"]) == {"large", "mid", "small"}
    assert set(df["filing_type"]) <= {"10-K", "10-Q"}
    assert set(df["section"]) <= {"mda", "risk"}
    assert df["filing_date"].astype(str).str.fullmatch(r"\d{8}").all()
    assert (df["filing_date"].astype(int) > 20240701).all(), \
        "fixture must respect config.TRAINING_CUTOFF"

    # Window word counts follow the real chunker: every window but a section's
    # last is exactly CHUNK_WORDS; the last is in the observed 403-1598 band.
    assert df["word_count"].between(403, 1598).all()
    last = df["chunk_index"] == df["n_in_section"] - 1
    assert (df.loc[~last, "word_count"] == CHUNK_WORDS).all()

    # chunk_index is 0-based, contiguous, and agrees with n_in_section.
    g = df.groupby(["ticker", "filing_type", "filing_date", "section"])
    for key, sub in g:
        idx = sorted(sub["chunk_index"].tolist())
        assert idx == list(range(len(sub))), f"{key}: non-contiguous chunk_index {idx}"
        assert sub["n_in_section"].nunique() == 1
        assert int(sub["n_in_section"].iloc[0]) == len(sub), f"{key}: n_in_section wrong"

    # Pre-registration MD&A shape: min 2, max 25 windows per filing.
    mda = df[df.section == "mda"].groupby(
        ["ticker", "filing_type", "filing_date"]).size()
    assert mda.min() >= 2 and mda.max() <= 25
    assert set(df["served_by"]) == {"DeepInfra"}, \
        "one pinned backend, so the served-backend audit (plan 8.4) can pass"


def test_fixture_numeric_disagreement_is_the_sd_of_its_own_scores() -> None:
    """The CSV must be internally consistent: a referee can recompute the column."""
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_fixture(td, seed=12)
        df = pd.read_csv(path)
    ok = ~df["any_failed"].astype(bool)
    scores = df.loc[ok, ["value_score", "sentiment_score", "fundamentals_score"]]
    recomputed = scores.to_numpy(dtype=float).std(axis=1, ddof=0)
    assert np.allclose(recomputed, df.loc[ok, "numeric_disagreement"].to_numpy(),
                       atol=1e-12)
    assert scores.to_numpy().min() >= 0.0 and scores.to_numpy().max() <= 10.0


def test_fixture_is_deterministic_in_its_seed() -> None:
    with tempfile.TemporaryDirectory() as td:
        a, _ = _write_fixture(td, seed=99)
        first = a.read_bytes()
        b, _ = _write_fixture(td, seed=99)
        assert b.read_bytes() == first, "same seed must give byte-identical output"
        c, _ = _write_fixture(td, seed=100)
        assert c.read_bytes() != first, "a different seed must give different data"


def test_fixture_reports_every_bound_it_applied() -> None:
    """No silent caps: the truth dict accounts for each bound that actually bit."""
    with tempfile.TemporaryDirectory() as td:
        _, truth = _write_fixture(td, seed=13)
    b = truth["bounds_applied"]
    assert set(b) == {
        "window_counts_clipped_to_2_25",
        "embedding_draws_clipped_to_0_2",
        "score_triples_clipped_to_0_10",
    }
    assert all(isinstance(v, int) and v >= 0 for v in b.values())


def test_position_decile_matches_the_preregistered_definition() -> None:
    # chunk_index / max(1, n_in_section - 1), binned into 10 bins, 9 closed.
    assert position_decile(0, 1) == 0          # single-window section
    assert position_decile(0, 2) == 0
    assert position_decile(1, 2) == 9          # fraction 1.0 -> top bin
    assert position_decile(12, 25) == 5        # 12/24 = 0.5
    assert position_decile(24, 25) == 9
    assert mm.position_decile_from(1, 2) == position_decile(1, 2)
    for n in (1, 2, 3, 7, 25):
        for i in range(n):
            assert 0 <= mm.position_decile_from(i, n) <= 9


# ───────────────────────── recovery: planted effects ─────────────────────────

PLANT_SMALL = 0.060
PLANT_MID = 0.030
PLANT_SEED = 20261004
TIER_TOL = 0.025   # ~1.8 SE at this design; seeds are fixed, so not flaky


def test_recovers_planted_tier_effect_in_all_three_specifications() -> None:
    """A planted Small-vs-Large effect of +0.060 must come back, in all three specs."""
    with tempfile.TemporaryDirectory() as td:
        path, truth = _write_fixture(
            td, seed=PLANT_SEED,
            tier_effect={"large": 0.0, "mid": PLANT_MID, "small": PLANT_SMALL},
            sd_firm=0.030, sd_filing=0.020, sd_resid=0.040,
        )
        r = mm.run_w5(path, seed=PLANT_SEED)

    assert truth["planted"]["small_vs_large"] == PLANT_SMALL
    assert r.quality.n_firms_used == 30
    assert r.quality.n_filings_used == 59
    assert r.quality.n_windows_used == truth["realised"]["n_mda_windows"]

    for key in SPEC_KEYS:
        s = r.specs[key]
        assert s.converged, f"{key} did not converge: {s.message}"
        _assert_recovered("planted_tier_effect", key, mm.SMALL_VS_LARGE,
                          PLANT_SMALL, s.coefficients[mm.SMALL_VS_LARGE], TIER_TOL)
        _assert_recovered("planted_tier_effect", key, mm.MID_VS_LARGE,
                          PLANT_MID, s.coefficients[mm.MID_VS_LARGE], TIER_TOL)
        assert s.criterion() is True, f"{key} failed the pre-declared criterion"

    assert r.criterion_met() is True
    assert r.primary_spec_key == "nested"


def test_recovers_variance_components_and_iccs() -> None:
    """The nested spec must recover the planted firm/filing/residual variances."""
    sd_firm, sd_filing, sd_resid = 0.060, 0.040, 0.050
    with tempfile.TemporaryDirectory() as td:
        path, truth = _write_fixture(
            td, seed=4242,
            tier_effect={"large": 0.0, "mid": PLANT_MID, "small": PLANT_SMALL},
            sd_firm=sd_firm, sd_filing=sd_filing, sd_resid=sd_resid,
        )
        r = mm.run_w5(path, seed=4242)

    s = r.specs["nested"]
    assert s.converged, s.message
    v = s.variances
    planted = truth["planted"]

    # Variance components are estimated from 30 firms / 59 filings, so their own
    # sampling error is large; tolerances are on the SD scale, where they are
    # interpretable, and are generous for the two higher levels by necessity.
    for key, truth_sd, tol in (
        ("firm", sd_firm, 0.025),
        ("filing_within_firm", sd_filing, 0.020),
        ("residual_within_filing", sd_resid, 0.008),
    ):
        est_sd = math.sqrt(v[key])
        RECOVERED.append({
            "test": "variance_components", "spec": "nested", "term": f"sd_{key}",
            "planted": truth_sd, "estimate": est_sd, "se": None,
            "ci_low": None, "ci_high": None, "p": None,
        })
        assert abs(est_sd - truth_sd) <= tol, (
            f"sd_{key}: recovered {est_sd:.4f}, planted {truth_sd:.4f}"
        )

    for key, tol in (("icc_firm", 0.12), ("icc_filing", 0.12)):
        est, tru = s.iccs[key], planted[key]
        RECOVERED.append({
            "test": "variance_components", "spec": "nested", "term": key,
            "planted": tru, "estimate": est, "se": None,
            "ci_low": None, "ci_high": None, "p": None,
        })
        assert abs(est - tru) <= tol, f"{key}: recovered {est:.4f}, planted {tru:.4f}"

    # icc_filing must exceed icc_firm by construction (it adds a component).
    assert s.iccs["icc_filing"] > s.iccs["icc_firm"]
    # The firm-only spec cannot identify a filing-level ICC and must say so
    # rather than pass off icc_firm as icc_filing.
    assert r.specs["firm_only"].iccs["icc_filing"] is None
    assert r.specs["ols_cluster"].iccs["icc_firm"] is None


def test_recovers_planted_covariate_coefficients() -> None:
    """log(window_words), position_decile and form_type must also come back."""
    b_log, b_pos, b_form = 0.080, 0.0040, 0.0150
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_fixture(
            td, seed=777,
            tier_effect={"large": 0.0, "mid": 0.0, "small": 0.0},
            beta_log_words=b_log, beta_position_decile=b_pos, beta_form_10q=b_form,
            sd_firm=0.030, sd_filing=0.020, sd_resid=0.040,
        )
        r = mm.run_w5(path, seed=777)

    s = r.primary
    assert s.converged, s.message
    # log(window_words) is weakly identified by construction -- only a section's
    # final window departs from exactly 1,200 words -- so coverage of the CI is
    # the meaningful assertion and the point tolerance is correspondingly wide.
    _assert_recovered("covariates", s.key, "log_window_words", b_log,
                      s.coefficients["log_window_words"], 0.060)
    _assert_recovered("covariates", s.key, "position_decile", b_pos,
                      s.coefficients["position_decile"], 0.0025)
    _assert_recovered("covariates", s.key, "form_10Q", b_form,
                      s.coefficients["form_10Q"], 0.015)


def test_power_across_seeds_with_a_planted_effect() -> None:
    """With +0.060 planted, the criterion should be met on most seeds."""
    seeds = list(range(5001, 5041))
    hits, ests = 0, []
    for sd in seeds:
        with tempfile.TemporaryDirectory() as td:
            path, _ = _write_fixture(
                td, seed=sd,
                tier_effect={"large": 0.0, "mid": PLANT_MID, "small": PLANT_SMALL},
                sd_firm=0.030, sd_filing=0.020, sd_resid=0.040,
            )
            r = mm.run_w5(path, seed=sd)
        ests.append(r.primary.coefficients[mm.SMALL_VS_LARGE]["estimate"])
        hits += bool(r.criterion_met())
    RECOVERED.append({
        "test": "power_40_seeds", "spec": "primary", "term": mm.SMALL_VS_LARGE,
        "planted": PLANT_SMALL, "estimate": float(np.mean(ests)), "se": None,
        "ci_low": None, "ci_high": None, "p": f"{hits}/{len(seeds)} criterion met",
    })
    assert hits >= 36, f"only {hits}/{len(seeds)} seeds met the criterion"
    assert abs(float(np.mean(ests)) - PLANT_SMALL) <= 0.010, (
        f"mean estimate over {len(seeds)} seeds is {np.mean(ests):+.4f}, "
        f"planted {PLANT_SMALL:+.4f} -- the estimator looks biased"
    )


# ────────────────── the null: no effect planted, none found ──────────────────


def test_no_tier_effect_ci_covers_zero() -> None:
    """With nothing planted, every specification's tier CI must cover 0.

    This is the test that matters most: the window-count imbalance the advisor
    flagged (large-caps have many windows, small-caps few) is present in the
    fixture by construction, so an estimator that converts that imbalance into a
    tier gradient fails here.
    """
    with tempfile.TemporaryDirectory() as td:
        path, truth = _write_fixture(
            td, seed=20261005,
            tier_effect={"large": 0.0, "mid": 0.0, "small": 0.0},
            sd_firm=0.030, sd_filing=0.020, sd_resid=0.040,
        )
        r = mm.run_w5(path, seed=20261005)

    assert truth["planted"]["small_vs_large"] == 0.0
    # The imbalance is really there, or the test would be vacuous.
    per_tier = {t: r.quality.windows_by_tier[t] for t in ("large", "mid", "small")}
    assert per_tier["large"] > per_tier["small"], per_tier

    for key in SPEC_KEYS:
        s = r.specs[key]
        for term in (mm.SMALL_VS_LARGE, mm.MID_VS_LARGE):
            c = s.coefficients[term]
            _record("no_tier_effect", key, term, 0.0, c)
            assert c["ci_low"] <= 0.0 <= c["ci_high"], (
                f"{key}: {term} CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] "
                "excludes 0 although no tier effect was planted"
            )
        assert s.criterion() is False, f"{key} reported a pass on a true null"

    assert r.criterion_met() is False


def test_false_positive_rate_under_the_null_across_seeds() -> None:
    """The criterion should fire on few seeds when nothing is planted."""
    seeds = list(range(6001, 6041))
    hits, ests = 0, []
    for sd in seeds:
        with tempfile.TemporaryDirectory() as td:
            path, _ = _write_fixture(
                td, seed=sd, tier_effect={"large": 0.0, "mid": 0.0, "small": 0.0},
                sd_firm=0.030, sd_filing=0.020, sd_resid=0.040,
            )
            r = mm.run_w5(path, seed=sd)
        ests.append(r.primary.coefficients[mm.SMALL_VS_LARGE]["estimate"])
        hits += bool(r.criterion_met())
    RECOVERED.append({
        "test": "null_40_seeds", "spec": "primary", "term": mm.SMALL_VS_LARGE,
        "planted": 0.0, "estimate": float(np.mean(ests)), "se": None,
        "ci_low": None, "ci_high": None,
        "p": f"{hits}/{len(seeds)} criterion met (one-sided, so <=~1 expected)",
    })
    # Excluding 0 upward with a 95% two-sided CI is a ~2.5% event per seed, so
    # 0-3 of 40 is the expected range.  Measured over 100 independent seeds the
    # primary specification's rate is 2.0% (and CI coverage of the true 0 is
    # 97%), i.e. nominal; 5+ of 40 here would indicate a broken estimator.
    assert hits <= 4, f"{hits}/{len(seeds)} false positives under the null"
    assert abs(float(np.mean(ests))) <= 0.010, (
        f"mean estimate under the null is {np.mean(ests):+.4f}, not ~0"
    )


# ─────────────────── pre-registered data rules are enforced ──────────────────


def test_failed_windows_are_dropped_never_imputed() -> None:
    with tempfile.TemporaryDirectory() as td:
        path, truth = _write_fixture(
            td, seed=31337, failure_rate=0.12,
            tier_effect={"large": 0.0, "mid": PLANT_MID, "small": PLANT_SMALL},
            sd_firm=0.030, sd_filing=0.020, sd_resid=0.040,
        )
        raw = pd.read_csv(path)
        df, q = mm.load_window_frame(path, section="mda")
        r = mm.run_w5(path, seed=31337)

    mda = raw[raw.section == "mda"]
    n_failed = int(mda["any_failed"].astype(bool).sum())
    assert n_failed > 0, "fixture planted no failures; the test would be vacuous"
    assert q.n_failed_dropped == n_failed
    assert q.n_rows_section == len(mda)
    assert q.n_windows_used == len(mda) - n_failed - q.n_windows_dropped_gate

    # Nothing imputed: no failed row survives, and no null reaches the model.
    assert not df["any_failed"].astype(bool).any()
    assert df["embedding_disagreement"].notna().all()
    design, _ = mm.prepare_design(df)
    assert np.isfinite(design["y"].to_numpy()).all()

    # Dropping is MCAR here, so the planted effect must still be recovered.
    assert r.primary.converged
    _assert_recovered("with_failures", r.primary_spec_key, mm.SMALL_VS_LARGE,
                      PLANT_SMALL, r.primary.coefficients[mm.SMALL_VS_LARGE],
                      TIER_TOL)


def test_filing_gate_drops_a_filing_with_too_few_surviving_windows() -> None:
    """Plan 8.2: <2 surviving windows, or <half surviving, and the filing goes."""
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_fixture(td, seed=55, failure_rate=0.0)
        raw = pd.read_csv(path)

        mda = raw[raw.section == "mda"]
        key = ["ticker", "filing_type", "filing_date"]
        sizes = mda.groupby(key).size()
        victim = sizes[sizes >= 4].index[0]          # fail all but one window
        survivor_ok = sizes[sizes >= 4].index[1]     # fail just one window

        def rows_of(k, df=raw):
            m = pd.Series(True, index=df.index)
            for col, val in zip(key, k):
                m &= df[col] == val
            return m & (df["section"] == "mda")

        v = raw.index[rows_of(victim)].tolist()
        for i in v[1:]:
            raw.loc[i, "any_failed"] = True
            raw.loc[i, ["numeric_disagreement", "embedding_disagreement"]] = np.nan
        s = raw.index[rows_of(survivor_ok)].tolist()
        raw.loc[s[0], "any_failed"] = True
        raw.loc[s[0], ["numeric_disagreement", "embedding_disagreement"]] = np.nan

        path2 = Path(td) / "gated.csv"
        raw.to_csv(path2, index=False)
        _, q = mm.load_window_frame(path2, section="mda")

    vid = "|".join(str(x) for x in victim)
    sid = "|".join(str(x) for x in survivor_ok)
    assert q.n_filings_dropped_gate == 1, q.filings_dropped_gate
    assert any(s.startswith(vid) for s in q.filings_dropped_gate), q.filings_dropped_gate
    assert not any(s.startswith(sid) for s in q.filings_dropped_gate)
    assert q.n_windows_dropped_gate == 1   # the victim's one surviving window


def test_sections_are_never_pooled() -> None:
    with tempfile.TemporaryDirectory() as td:
        path, truth = _write_fixture(td, seed=66)
        mda, qm = mm.load_window_frame(path, section="mda")
        risk, qr = mm.load_window_frame(path, section="risk")
    assert set(mda["section"]) == {"mda"}
    assert set(risk["section"]) == {"risk"}
    assert qm.n_windows_used == truth["realised"]["n_mda_windows"]
    assert qm.n_windows_used + qr.n_windows_used == truth["realised"]["n_windows"]
    # Risk Factors is absent from some 10-Qs and so has its own, smaller n.
    assert qr.n_filings_used < qm.n_filings_used


def test_missing_input_file_raises_rather_than_inventing_data() -> None:
    try:
        mm.load_window_frame("/nonexistent/chunk_level_results.csv")
    except FileNotFoundError as exc:
        assert "makes no model calls" in str(exc)
    else:
        raise AssertionError("a missing input file must raise")


# ───────────────── reporting contract a referee will check ───────────────────


def test_reference_levels_are_explicit_in_the_output() -> None:
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_fixture(
            td, seed=808,
            tier_effect={"large": 0.0, "mid": PLANT_MID, "small": PLANT_SMALL},
        )
        r = mm.run_w5(path, seed=808)
        report = mm.format_report(r)

    assert r.tier_reference == "large" and r.form_reference == "10-K"
    assert r.design_summary["tier_is_three_level"] is True
    assert "Tier reference  : large" in report
    assert "tier[small vs large]" in report
    assert "Seed            : 808" in report
    assert str(r.quality.n_windows_used) in report
    assert "WINDOWS used" in report and "FILINGS used" in report and "FIRMS used" in report
    assert "icc_filing" in report and "icc_firm" in report
    assert "W5 CRITERION" in report
    # Which specification the claim rests on, and why, must be in the text.
    assert "PRIMARY:" in report and r.primary_spec_reason[:40] in report
    frame = mm.coefficient_frame(r)
    assert set(frame["spec"]) == set(SPEC_KEYS)
    assert frame.loc[frame["is_primary_spec"], "spec"].unique().tolist() == ["nested"]
    assert (frame["tier_reference"] == "large").all()


def test_nonconvergence_of_spec_a_falls_back_to_spec_b_and_says_so() -> None:
    """If (a) will not converge, W5 must report the failure and fall back to (b),
    not present (a)'s numbers anyway and not silently drop the fallback note."""
    real_fit_nested = mm.fit_nested

    def broken_fit_nested(df, regressors, reml=True):
        spec = real_fit_nested(df, regressors, reml=reml)
        spec.converged = False
        spec.message = ("DID NOT CONVERGE - simulated failure; coefficients and "
                        "variance components from this specification are not "
                        "trustworthy and W5 falls back to (b).")
        spec.diagnostics["converged_flag"] = False
        return spec

    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_fixture(
            td, seed=2024,
            tier_effect={"large": 0.0, "mid": PLANT_MID, "small": PLANT_SMALL})
        mm.fit_nested = broken_fit_nested
        try:
            r = mm.run_w5(path, seed=2024)
        finally:
            mm.fit_nested = real_fit_nested
        report = mm.format_report(r)

    assert r.primary_spec_key == "firm_only"
    assert "did not converge" in r.primary_spec_reason
    assert r.specs["nested"].criterion() is None, \
        "a non-converged fit must never be read as a pass or a fail"
    # The criterion is then evaluated on (b), which still recovers the effect.
    _assert_recovered("fallback_to_b", "firm_only", mm.SMALL_VS_LARGE, PLANT_SMALL,
                      r.primary.coefficients[mm.SMALL_VS_LARGE], TIER_TOL)
    assert r.criterion_met() is True
    assert "DID NOT CONVERGE" in report
    assert "REML components unavailable" in report
    assert "NOT EVALUABLE" in report      # (a)'s own criterion line
    # The filing-level ICC is genuinely unavailable in the fallback.
    assert r.primary.iccs["icc_filing"] is None
    assert "not identified without a filing variance component" in report


def test_position_decile_as_a_factor_also_runs() -> None:
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_fixture(
            td, seed=1357,
            tier_effect={"large": 0.0, "mid": PLANT_MID, "small": PLANT_SMALL},
            beta_position_decile=0.003)
        r = mm.run_w5(path, seed=1357, position_as_factor=True)
    assert r.position_term == "10-level factor"
    terms = list(r.primary.coefficients)
    assert sum(t.startswith("position_decile_") for t in terms) >= 8
    assert "position_decile" not in terms   # the linear term is replaced
    _assert_recovered("position_as_factor", r.primary_spec_key, mm.SMALL_VS_LARGE,
                      PLANT_SMALL, r.primary.coefficients[mm.SMALL_VS_LARGE],
                      TIER_TOL)


def test_variance_decomposition_is_additive_and_sums_to_one() -> None:
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_fixture(
            td, seed=909, sd_firm=0.030, sd_filing=0.020, sd_resid=0.040)
        r = mm.run_w5(path, seed=909)

    for key in ("raw", "conditional_on_fixed_effects"):
        vd = r.variance_decomposition[key]
        shares = (vd["share_between_firms"],
                  vd["share_between_filings_within_firm"],
                  vd["share_within_filings"])
        assert abs(sum(shares) - 1.0) < 1e-9, (key, shares)
        assert all(0.0 <= x <= 1.0 for x in shares), (key, shares)
        assert vd["additivity_check_abs_error"] < 1e-9
        assert (vd["df_between_firms"] + vd["df_between_filings_within_firm"]
                + vd["df_within_filings"]) == vd["n_obs"] - 1

    # With sd_resid (0.040) the largest planted component, the within-filing
    # share should dominate in the REML decomposition.
    s = r.specs["nested"]
    assert s.converged
    assert s.variances["residual_within_filing"] > s.variances["firm"]


def test_within_filing_share_tracks_the_planted_residual_variance() -> None:
    """The substantive claim -- 'most variance is within filings' -- must be
    driven by the data, not by the report text."""
    common = dict(tier_effect={"large": 0.0, "mid": 0.0, "small": 0.0}, seed=1234)
    with tempfile.TemporaryDirectory() as td:
        noisy, _ = _write_fixture(
            td, sd_firm=0.010, sd_filing=0.010, sd_resid=0.080, **common)
        r_noisy = mm.run_w5(noisy, seed=1234)
        path2 = Path(td) / "structured.csv"
        generate_chunk_level_fixture(
            out_path=path2, sd_firm=0.080, sd_filing=0.040, sd_resid=0.015, **common)
        r_struct = mm.run_w5(path2, seed=1234)

    w_noisy = r_noisy.specs["nested"].variances["residual_within_filing"] / \
        r_noisy.specs["nested"].variances["total"]
    w_struct = r_struct.specs["nested"].variances["residual_within_filing"] / \
        r_struct.specs["nested"].variances["total"]
    RECOVERED.append({
        "test": "within_filing_share", "spec": "nested",
        "term": "within-filing variance share",
        "planted": 0.0064 / (0.0001 + 0.0001 + 0.0064),
        "estimate": w_noisy, "se": None, "ci_low": None, "ci_high": None,
        "p": f"structured fixture: {w_struct:.3f}",
    })
    assert w_noisy > 0.85, w_noisy
    assert w_struct < 0.15, w_struct
    assert "within filings" in mm.format_report(r_noisy)


def test_cluster_bootstrap_is_deterministic_in_its_seed() -> None:
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_fixture(
            td, seed=4321,
            tier_effect={"large": 0.0, "mid": PLANT_MID, "small": PLANT_SMALL})
        a = mm.run_w5(path, seed=7, n_boot=40).bootstrap
        b = mm.run_w5(path, seed=7, n_boot=40).bootstrap
        c = mm.run_w5(path, seed=8, n_boot=40).bootstrap

    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]
    assert (a["ci_low"], a["ci_high"]) != (c["ci_low"], c["ci_high"])
    assert a["n_successful"] + a["n_failed"] == 40
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    # The planted effect should sit inside the bootstrap interval.
    assert a["ci_low"] <= PLANT_SMALL <= a["ci_high"]


def test_numeric_measure_runs_as_the_secondary_measure() -> None:
    with tempfile.TemporaryDirectory() as td:
        path, _ = _write_fixture(
            td, seed=1515,
            numeric_tier_effect={"large": 0.0, "mid": 0.10, "small": 0.20})
        r = mm.run_w5(path, seed=1515, measure="numeric_disagreement")
    s = r.primary
    assert s.converged
    c = s.coefficients[mm.SMALL_VS_LARGE]
    RECOVERED.append({
        "test": "numeric_measure", "spec": s.key, "term": mm.SMALL_VS_LARGE,
        "planted": 0.20, "estimate": c["estimate"], "se": c["se"],
        "ci_low": c["ci_low"], "ci_high": c["ci_high"], "p": c["p"],
    })
    # Rounding the three persona scores to one decimal perturbs the measure, so
    # only CI coverage is asserted here, not a tight point tolerance.
    assert c["ci_low"] <= 0.20 <= c["ci_high"], c
    assert "numeric_disagreement  (secondary)" in mm.format_report(r)


# ──────────────────────────── plain-python runner ────────────────────────────


def _all_tests() -> list:
    g = globals()
    return [g[n] for n in sorted(g) if n.startswith("test_") and callable(g[n])]


def main() -> int:
    failures = []
    for fn in _all_tests():
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:
            failures.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}: {type(exc).__name__}: {exc}")

    if RECOVERED:
        print("\nRecovered values (planted -> estimated)")
        print("-" * 104)
        print(f"{'test':<26}{'spec':<12}{'term':<28}{'planted':>9}"
              f"{'est':>10}{'ci_low':>10}{'ci_high':>10}")
        for row in RECOVERED:
            def f(x, nd=4):
                return "       n/a" if x is None else f"{x:>10.{nd}f}"
            print(f"{row['test']:<26}{row['spec']:<12}{row['term'][:27]:<28}"
                  f"{row['planted']:>9.4f}{f(row['estimate'])}"
                  f"{f(row['ci_low'])}{f(row['ci_high'])}"
                  + (f"   {row['p']}" if isinstance(row["p"], str) else ""))

    print(f"\n{len(_all_tests()) - len(failures)} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
