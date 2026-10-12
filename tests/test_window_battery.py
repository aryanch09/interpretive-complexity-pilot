"""
Tests for analysis/window_battery.py (W1-W4, W6-W7).

The point of this file is NOT to check that the battery runs.  It is to check
that the battery's DECISIVE test can tell two situations apart:

  * a real window-level tier effect, which survives equalising the window count
    (W3 must PASS), and
  * an apparent gradient produced only by unequal window counts, with the true
    tier effect exactly 0 (W3 must FAIL).

Both fixtures are built by tests/fixtures.py from the same generator, differing
only in `tier_effect` (0.030 vs 0.0) and the window-count regime, so nothing but
the ground truth changes between them.

Run:
    python3 -m pytest tests/ -q
    python3 -m tests.test_window_battery
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import window_battery as wb                            # noqa: E402
from tests import fixtures_window_battery as fx                        # noqa: E402

SEED = 20261004
# Smaller iteration counts for the test suite; the pre-registered 2,000 / 10,000
# are the module defaults and are exercised in the two decisive tests below.
FAST_W3 = 2_000
FAST_W6 = 2_000
FAST_PERM = 1_000


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def planted():
    df, spec = fx.planted_effect_fixture(seed=11)
    return df, spec


@pytest.fixture(scope="module")
def artifact():
    df, spec, search = fx.count_artifact_fixture(seed=101)
    return df, spec, search


@pytest.fixture(scope="module")
def planted_battery(planted):
    df, _ = planted
    return wb.run_battery(df, section="mda", measure="embedding_disagreement",
                          seed=SEED, n_permutations=FAST_PERM)


@pytest.fixture(scope="module")
def artifact_battery(artifact):
    df, _, _ = artifact
    return wb.run_battery(df, section="mda", measure="embedding_disagreement",
                          seed=SEED, n_permutations=FAST_PERM)


# ===========================================================================
# 1. The fixture itself
# ===========================================================================

def test_fixture_has_the_exact_real_schema(planted):
    df, _ = planted
    assert list(df.columns) == list(wb.CHUNK_SCHEMA)
    assert set(df["tier"]) <= {"large", "mid", "small"}
    assert set(df["filing_type"]) <= {"10-K", "10-Q"}
    assert set(df["section"]) <= {"mda", "risk"}
    assert df["filing_date"].astype(str).str.fullmatch(r"\d{8}").all()
    assert df["word_count"].between(403, 1598).all()
    assert df["chunk_index"].min() == 0
    # chunk_index is 0-based within section and n_in_section is its own group size
    g = df.groupby(["ticker", "filing_type", "filing_date", "section"])
    assert (g["chunk_index"].max() + 1 == g["chunk_index"].size()).all()
    assert (g["n_in_section"].nunique() == 1).all()
    assert (g["n_in_section"].first() == g["chunk_index"].size()).all()
    assert df["served_by"].nunique() == 1          # the single pinned backend


def test_fixture_matches_the_real_sample_shape(planted):
    df, _ = planted
    assert df["ticker"].nunique() == 30
    assert df.groupby(["ticker", "filing_type", "filing_date"]).ngroups == 59
    per_tier = df.drop_duplicates("ticker")["tier"].value_counts().to_dict()
    assert per_tier == {"large": 10, "mid": 10, "small": 10}
    mda = df[df["section"] == "mda"]
    counts = mda.groupby(["ticker", "filing_type", "filing_date"])["chunk_index"].size()
    assert counts.min() >= 2
    assert counts.max() <= 25
    # the real sample has 667 windows of which 363 are MD&A; the fixture only has
    # to be in that neighbourhood, not identical
    assert 550 <= len(df) <= 800
    assert 300 <= len(mda) <= 450


def test_fixture_numeric_measure_is_consistent_with_the_three_scores(planted):
    df, _ = planted
    ok = df[~df["any_failed"].astype(bool)]
    scores = ok[["value_score", "sentiment_score", "fundamentals_score"]].to_numpy(float)
    recomputed = np.std(scores / 10.0, axis=1, ddof=0)
    assert np.allclose(recomputed, ok["numeric_disagreement"].to_numpy(float))


def test_window_counts_differ_by_tier_in_the_planted_fixture(planted):
    """If the counts did not differ, W3 would be a trivial test."""
    df, _ = planted
    mda = df[df["section"] == "mda"]
    med = (mda.groupby(["ticker", "tier", "filing_type", "filing_date"])["chunk_index"]
              .size().reset_index(name="n").groupby("tier")["n"].median())
    assert med["small"] < med["large"]
    assert med["small"] < med["mid"]


# ===========================================================================
# 2. Failed windows are dropped, never imputed
# ===========================================================================

def test_failed_windows_are_dropped_never_imputed():
    spec = fx.FixtureSpec(seed=5, n_failed_windows=12, n_parse_error_windows=7)
    df = fx.generate_chunk_level_results(spec)
    n_failed = int(df["any_failed"].astype(bool).sum())
    assert n_failed == 12
    failed = df[df["any_failed"].astype(bool)]
    assert failed["numeric_disagreement"].isna().all()
    assert failed["embedding_disagreement"].isna().all()

    w = wb.surviving_windows(df, section="mda", measure="embedding_disagreement")
    n_mda_failed = int(df[(df["section"] == "mda")]["any_failed"].astype(bool).sum())
    assert n_mda_failed == 12                       # the generator only fails MD&A rows
    assert len(w) == int((df["section"] == "mda").sum()) - 12
    assert w.attrs["n_dropped_failed"] == 12
    assert not w["any_failed"].any()

    # a parse error alone is not a failure: those rows stay
    assert int(df["any_parse_error"].astype(bool).sum()) == 7
    mda_pe = df[(df["section"] == "mda") & df["any_parse_error"].astype(bool)
                & ~df["any_failed"].astype(bool)]
    surviving_keys = set(zip(w["ticker"], w["filing_type"], w["filing_date"],
                             w["chunk_index"].astype(int)))
    for _, r in mda_pe.iterrows():
        assert (r["ticker"], r["filing_type"], r["filing_date"],
                int(r["chunk_index"])) in surviving_keys

    # and nothing is imputed: the firm means equal those from the clean subset
    firm = wb._fallback_aggregate_to_firm(df, section="mda",
                                          measure="embedding_disagreement")
    manual = (df[(df["section"] == "mda") & ~df["any_failed"].astype(bool)]
              .assign(fk=lambda d: d["ticker"] + d["filing_type"] + d["filing_date"])
              .groupby(["ticker", "tier", "fk"])["embedding_disagreement"].mean()
              .reset_index().groupby(["ticker", "tier"])["embedding_disagreement"].mean())
    merged = firm.set_index(["ticker", "tier"])["value"]
    assert np.allclose(merged.sort_index().to_numpy(),
                       manual.sort_index().to_numpy())


def test_csv_round_trip_preserves_the_drop_rule(planted):
    """A failed window read back from disk (empty cells, 'True' strings) still drops."""
    spec = fx.FixtureSpec(seed=9, n_failed_windows=6)
    df = fx.generate_chunk_level_results(spec)
    with tempfile.TemporaryDirectory() as d:
        path = fx.write_fixture(df, Path(d) / "chunk_level_results.csv")
        back = wb.load_chunk_results(path)
    assert list(back.columns) == list(wb.CHUNK_SCHEMA)
    w_disk = wb.surviving_windows(back, section="mda")
    w_mem = wb.surviving_windows(df, section="mda")
    assert len(w_disk) == len(w_mem)
    assert math.isclose(w_disk["embedding_disagreement"].mean(),
                        w_mem["embedding_disagreement"].mean(), rel_tol=1e-12)


# ===========================================================================
# 3. The aggregation chain, by hand
# ===========================================================================

def _tiny_frame(rows: list[dict]) -> pd.DataFrame:
    base = {c: None for c in wb.CHUNK_SCHEMA}
    out = []
    for r in rows:
        d = dict(base)
        d.update({"any_failed": False, "any_parse_error": False, "served_by": "DeepInfra",
                  "word_count": 1200, "numeric_disagreement": 0.0,
                  "value_score": 50, "sentiment_score": 50, "fundamentals_score": 50,
                  "value_reasoning": "a", "sentiment_reasoning": "b",
                  "fundamentals_reasoning": "c"})
        d.update(r)
        out.append(d)
    return pd.DataFrame(out, columns=list(wb.CHUNK_SCHEMA))


def test_aggregation_is_unweighted_at_every_step():
    """
    One large-cap firm, two filings: a 1-window filing worth 1.0 and a 3-window
    filing worth (0, 0, 0).  Unweighted window -> filing -> firm gives
    (1.0 + 0.0) / 2 = 0.5.  A window-weighted mean would give 0.25.
    """
    rows = []
    rows.append({"ticker": "A", "tier": "large", "filing_type": "10-K",
                 "filing_date": "20250101", "section": "mda", "chunk_index": 0,
                 "n_in_section": 1, "embedding_disagreement": 1.0})
    for i, v in enumerate((0.0, 0.0, 0.0)):
        rows.append({"ticker": "A", "tier": "large", "filing_type": "10-Q",
                     "filing_date": "20250501", "section": "mda", "chunk_index": i,
                     "n_in_section": 3, "embedding_disagreement": v})
    # a small-cap firm so the gradient is defined
    rows.append({"ticker": "Z", "tier": "small", "filing_type": "10-K",
                 "filing_date": "20250101", "section": "mda", "chunk_index": 0,
                 "n_in_section": 2, "embedding_disagreement": 0.8})
    rows.append({"ticker": "Z", "tier": "small", "filing_type": "10-K",
                 "filing_date": "20250101", "section": "mda", "chunk_index": 1,
                 "n_in_section": 2, "embedding_disagreement": 0.6})
    df = _tiny_frame(rows)

    firm = wb._fallback_aggregate_to_firm(df, section="mda")
    got = dict(zip(firm["ticker"], firm["value"]))
    assert math.isclose(got["A"], 0.5)              # NOT 0.25
    assert math.isclose(got["Z"], 0.7)
    assert math.isclose(wb.gradient_small_minus_large(firm), 0.7 - 0.5)


def test_sections_are_never_pooled():
    rows = [
        {"ticker": "A", "tier": "large", "filing_type": "10-K", "filing_date": "20250101",
         "section": "mda", "chunk_index": 0, "n_in_section": 1,
         "embedding_disagreement": 0.2},
        {"ticker": "A", "tier": "large", "filing_type": "10-K", "filing_date": "20250101",
         "section": "risk", "chunk_index": 0, "n_in_section": 1,
         "embedding_disagreement": 0.9},
    ]
    df = _tiny_frame(rows)
    assert math.isclose(
        wb.surviving_windows(df, "mda")["embedding_disagreement"].iloc[0], 0.2)
    assert math.isclose(
        wb.surviving_windows(df, "risk")["embedding_disagreement"].iloc[0], 0.9)
    assert len(wb.surviving_windows(df, "mda")) == 1


# ===========================================================================
# 4. Jonckheere-Terpstra against a worked example
# ===========================================================================

def test_jonckheere_terpstra_worked_example():
    """
    Perfectly separated groups: every value in a later group exceeds every value
    in an earlier one, so J attains its maximum, sum over i<j of n_i * n_j.
    With three groups of 3: 3*3 + 3*3 + 3*3 = 27.
    E[J] = (N^2 - sum n_i^2) / 4 = (81 - 27) / 4 = 13.5.
    Var[J] = (N^2 (2N+3) - sum n_i^2 (2 n_i + 3)) / 72
           = (81*21 - 3*9*9) / 72 = (1701 - 243) / 72 = 20.25, sd = 4.5.
    z = (27 - 13.5) / 4.5 = 3.0.
    """
    groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    res = wb._fallback_jonckheere_terpstra(groups, alternative="increasing")
    assert math.isclose(res["statistic"], 27.0)
    assert math.isclose(res["expected"], 13.5)
    assert math.isclose(res["variance"], 20.25)
    assert math.isclose(res["z"], 3.0)
    assert res["p_value"] < 0.002

    # reversing the groups puts J at its minimum
    rev = wb._fallback_jonckheere_terpstra(groups[::-1], alternative="increasing")
    assert math.isclose(rev["statistic"], 0.0)
    assert rev["p_value"] > 0.99

    # all ties: J sits exactly at its expectation
    tied = wb._fallback_jonckheere_terpstra([[5, 5], [5, 5], [5, 5]])
    assert math.isclose(tied["statistic"], tied["expected"])
    assert math.isclose(tied["z"], 0.0)


def test_jonckheere_terpstra_wrapper_agrees_with_the_fallback():
    groups = [[0.1, 0.2, 0.15], [0.2, 0.25, 0.22], [0.3, 0.31, 0.28]]
    a = wb.jonckheere_terpstra(groups)
    b = wb._fallback_jonckheere_terpstra(groups)
    assert math.isclose(float(a["statistic"]), float(b["statistic"]))
    assert abs(float(a["p_value"]) - float(b["p_value"])) < 0.02


def test_fast_and_generic_permutation_paths_agree():
    firm = pd.DataFrame({
        "ticker": [f"F{i}" for i in range(9)],
        "tier": ["large"] * 3 + ["mid"] * 3 + ["small"] * 3,
        "value": [0.2, 0.25, 0.22, 0.3, 0.28, 0.31, 0.4, 0.38, 0.42],
    })
    v = firm["value"].to_numpy(float)
    lab = firm["tier"].to_numpy()
    try:
        wb._ALLOW_FAST_PERMUTATION = True
        fast = wb._fallback_label_permutation_test(v, lab, wb._gradient_stat,
                                                   n_draws=4000, seed=7)
        wb._ALLOW_FAST_PERMUTATION = False
        slow = wb._fallback_label_permutation_test(v, lab, wb._gradient_stat,
                                                   n_draws=4000, seed=7)
    finally:
        wb._ALLOW_FAST_PERMUTATION = True
    assert math.isclose(fast["observed"], slow["observed"])
    # different RNG consumption order, so compare the null's shape, not row by row
    assert abs(fast["null_mean"] - slow["null_mean"]) < 0.01
    assert abs(fast["null_sd"] - slow["null_sd"]) < 0.01
    assert abs(fast["p_value"] - slow["p_value"]) < 0.02


# ===========================================================================
# 5. W3 - the decisive test, on both fixtures
# ===========================================================================

def test_w3_recovers_a_planted_tier_effect(planted):
    """
    The effect lives in every window's expected value, so equalising the window
    count cannot remove it.  W3 must pass with the pre-registered 2,000 draws.
    """
    df, spec = planted
    assert spec.tier_effect > 0
    primary = wb.primary_result(df, "mda", "embedding_disagreement", SEED,
                                n_permutations=FAST_PERM)
    assert primary["gradient_sign"] == 1
    assert primary["monotone_small_gt_mid_gt_large"]

    w3 = wb.w3_equal_k(df, "mda", "embedding_disagreement", SEED,
                       n_iterations=wb.W3_ITERATIONS, primary_sign=1)
    k2 = w3["by_k"]["k=2"]
    assert k2["criterion_applies"] is True
    assert k2["share_primary_sign_holds"] >= wb.W3_SIGN_SHARE_CRITERION
    assert w3["criterion_met"] is True
    # and the k=2 gradient distribution is centred on the real effect
    assert k2["gradient_distribution"]["mean"] > 0.5 * spec.tier_effect
    # every filing is used at k=2, because 2 is the sample minimum
    assert k2["n_filings_dropped"] == 0


def test_w3_fails_when_the_gradient_is_only_a_window_count_artifact(artifact):
    """
    tier_effect is exactly 0: the expected window-level disagreement is identical
    in every tier.  The sample nonetheless shows an apparent Small > Mid > Large
    gradient (the fixture's seed search guarantees there is something to be
    fooled by).  W3 must NOT certify it.
    """
    df, spec, search = artifact
    assert spec.tier_effect == 0.0
    assert search["apparent_gradient"] > 0
    assert search["apparent_monotone"] is True

    primary = wb.primary_result(df, "mda", "embedding_disagreement", SEED,
                                n_permutations=FAST_PERM)
    assert primary["gradient_sign"] == 1               # there IS an apparent gradient
    assert primary["monotone_small_gt_mid_gt_large"]   # and it looks monotone

    w3 = wb.w3_equal_k(df, "mda", "embedding_disagreement", SEED,
                       n_iterations=wb.W3_ITERATIONS, primary_sign=1)
    assert w3["by_k"]["k=2"]["share_primary_sign_holds"] < wb.W3_SIGN_SHARE_CRITERION
    assert w3["criterion_met"] is False


def test_w3_distinguishes_the_two_fixtures(planted, artifact):
    """
    The explicit requirement: if W3 cannot separate a real effect from a
    count-only artifact, it does not work.  This asserts the separation, and
    that the pre-declared 90% line falls between the two shares.
    """
    df_p, _ = planted
    df_a, _, _ = artifact
    share_p = wb.w3_equal_k(df_p, "mda", "embedding_disagreement", SEED,
                            n_iterations=wb.W3_ITERATIONS,
                            primary_sign=1)["share_primary_sign_holds_k2"]
    share_a = wb.w3_equal_k(df_a, "mda", "embedding_disagreement", SEED,
                            n_iterations=wb.W3_ITERATIONS,
                            primary_sign=1)["share_primary_sign_holds_k2"]
    assert share_a < wb.W3_SIGN_SHARE_CRITERION <= share_p, (
        f"W3 cannot separate the fixtures: planted share {share_p:.4f}, "
        f"artifact share {share_a:.4f}, criterion {wb.W3_SIGN_SHARE_CRITERION}")
    assert share_p - share_a > 0.15


def test_w3_k4_restriction_is_reported_not_absorbed(planted):
    df, _ = planted
    w3 = wb.w3_equal_k(df, "mda", "embedding_disagreement", SEED,
                       n_iterations=200, primary_sign=1)
    k4 = w3["by_k"]["k=4"]
    counts = (df[df["section"] == "mda"]
              .groupby(["ticker", "filing_type", "filing_date"])["chunk_index"].size())
    assert k4["n_filings_used"] == int((counts >= 4).sum())
    assert k4["n_filings_dropped"] == int((counts < 4).sum())
    assert len(k4["filings_dropped"]) == k4["n_filings_dropped"]
    assert k4["criterion_applies"] is False      # the 90% line is a k=2 rule
    assert k4["criterion_met"] is None


def test_w3_is_deterministic_given_the_seed(planted):
    df, _ = planted
    a = wb.w3_equal_k(df, "mda", "embedding_disagreement", 123, n_iterations=300,
                      primary_sign=1)
    b = wb.w3_equal_k(df, "mda", "embedding_disagreement", 123, n_iterations=300,
                      primary_sign=1)
    c = wb.w3_equal_k(df, "mda", "embedding_disagreement", 124, n_iterations=300,
                      primary_sign=1)
    assert a["by_k"]["k=2"]["gradient_distribution"]["mean"] == \
        b["by_k"]["k=2"]["gradient_distribution"]["mean"]
    assert a["by_k"]["k=2"]["gradient_distribution"]["mean"] != \
        c["by_k"]["k=2"]["gradient_distribution"]["mean"]
    assert a["seed"] == 123


# ===========================================================================
# 6. W6 - the null simulation, on both fixtures
# ===========================================================================

def test_w6_rejects_the_null_for_a_planted_effect(planted):
    df, _ = planted
    w6 = wb.w6_null_simulation(df, "mda", "embedding_disagreement", SEED,
                               n_iterations=FAST_W6)
    assert w6["p_value_one_sided"] < 0.01
    assert w6["observed_percentile_in_null"] > 99.0
    # the null is centred on zero by construction: no tier effect is simulated
    assert abs(w6["null_gradient"]["mean"]) < 3 * w6["null_gradient"]["sd"]
    assert abs(w6["null_gradient"]["mean"]) < 0.01


def test_w6_does_not_reject_the_null_for_a_count_artifact(artifact):
    df, _, _ = artifact
    w6 = wb.w6_null_simulation(df, "mda", "embedding_disagreement", SEED,
                               n_iterations=FAST_W6)
    assert w6["p_value_one_sided"] > 0.05, (
        "W6 called a pure window-count artifact significant")
    assert 5.0 < w6["observed_percentile_in_null"] < 95.0
    # and it quantifies how much gradient averaging alone manufactures
    assert w6["null_abs_gradient_p95"] > abs(w6["observed_gradient"])


def test_w6_holds_window_counts_fixed_and_randomises_only_values(planted):
    df, _ = planted
    w6 = wb.w6_null_simulation(df, "mda", "embedding_disagreement", SEED,
                               n_iterations=50)
    real = (df[(df["section"] == "mda") & ~df["any_failed"].astype(bool)]
            .groupby(["ticker", "filing_type", "filing_date"])["chunk_index"].size())
    reported = w6["window_counts_by_filing"]
    assert len(reported) == len(real)
    assert sorted(reported.values()) == sorted(real.to_list())
    pooled_n = w6["pooled_distribution"]["n"]
    assert pooled_n == int(real.sum())
    assert "each filing's real surviving-window count" in w6["held_fixed"][0]


def test_w6_is_deterministic_given_the_seed(planted):
    df, _ = planted
    a = wb.w6_null_simulation(df, "mda", "embedding_disagreement", 77, n_iterations=400)
    b = wb.w6_null_simulation(df, "mda", "embedding_disagreement", 77, n_iterations=400)
    c = wb.w6_null_simulation(df, "mda", "embedding_disagreement", 78, n_iterations=400)
    assert a["null_gradient"]["sd"] == b["null_gradient"]["sd"]
    assert a["null_gradient"]["sd"] != c["null_gradient"]["sd"]


# ===========================================================================
# 7. W1 descriptives
# ===========================================================================

def test_w1_reports_n_and_p_for_every_spearman(planted):
    df, _ = planted
    w1 = wb.w1_descriptives(df, "mda", "embedding_disagreement")
    for level in ("filing_level", "firm_level"):
        assert len(w1["spearman"][level]) == 6
        for name, s in w1["spearman"][level].items():
            assert set(s) == {"rho", "p_value", "n"}, name
            assert s["n"] > 0, name
            assert -1.0 <= s["rho"] <= 1.0, name
            assert 0.0 <= s["p_value"] <= 1.0, name
    assert w1["spearman"]["filing_level"]["window_count_vs_tier_rank"]["n"] == 59
    assert w1["spearman"]["firm_level"]["window_count_vs_tier_rank"]["n"] == 30
    assert w1["criterion"] is None                   # descriptive, no pass/fail


def test_w1_window_count_vs_tier_rank_is_positive_when_large_filings_are_longer(planted):
    df, _ = planted
    w1 = wb.w1_descriptives(df, "mda", "embedding_disagreement")
    s = w1["spearman"]["filing_level"]["window_count_vs_tier_rank"]
    assert s["rho"] > 0                              # large = 3 and files longer
    med = w1["windows_per_filing_by_tier"]
    assert med["small"]["median"] < med["large"]["median"]
    assert med["small"]["min"] >= 2


def test_w1_descriptive_counts_match_the_frame(planted):
    df, _ = planted
    w1 = wb.w1_descriptives(df, "mda", "embedding_disagreement")
    tot = sum(w1["windows_per_filing_by_tier"][t]["n"] for t in wb.TIER_ORDER)
    assert tot == 59
    assert w1["windows_per_filing_overall"]["n"] == 59
    assert sum(w1["windows_per_firm_by_tier"]["total"][t]["n"]
               for t in wb.TIER_ORDER) == 30
    assert w1["window_words_overall"]["n"] == int((df["section"] == "mda").sum())


# ===========================================================================
# 8. W2 matched comparison
# ===========================================================================

def test_w2_bins_partition_the_filings_and_report_composition(planted):
    df, _ = planted
    w2 = wb.w2_matched(df, "mda", "embedding_disagreement", primary_sign=1)
    assert set(w2["bins"]) == {"2-3", "4-5", "6-8", "9+"}
    assert (sum(d["n_filings"] for d in w2["bins"].values())
            + w2["filings_outside_bins"]["n"]) == 59
    for label, d in w2["bins"].items():
        assert sum(d["tier_composition"].values()) == d["n_filings"], label
        assert d["identified_for_three_tier"] == all(
            v > 0 for v in d["tier_composition"].values()), label
        if not d["identified_for_three_tier"]:
            assert "NOT IDENTIFIED" in d["identification_note"], label
    # a bin that is not identified can never be counted as contradicting
    for label, d in w2["bins"].items():
        if not d["identified_for_three_tier"]:
            assert d["contradicts_primary"] is False, label


def test_w2_criterion_not_contradicted_under_a_planted_effect(planted):
    df, _ = planted
    w2 = wb.w2_matched(df, "mda", "embedding_disagreement", primary_sign=1)
    assert w2["criterion_met"] is True
    assert w2["contradicting_bins"] == []
    assert w2["bins_identified_for_three_tier"]       # non-empty: real evidence
    assert w2["criterion_vacuous"] is False


def test_w2_criterion_is_flagged_vacuous_when_no_bin_has_all_three_tiers(artifact):
    df, _, _ = artifact
    w2 = wb.w2_matched(df, "mda", "embedding_disagreement", primary_sign=1)
    assert w2["bins_identified_for_three_tier"] == []
    assert w2["criterion_vacuous"] is True
    assert "VACUOUSLY" in w2["criterion_note"]


def test_w2_detects_a_contradicting_bin():
    """
    A bin in which the small-cap filings sit clearly BELOW the large-cap ones,
    with a tight CI, must be flagged as contradicting a positive primary sign.
    """
    rows = []
    for i in range(8):
        for tier, val in (("large", 0.60), ("mid", 0.50), ("small", 0.20)):
            tk = f"{tier[:2].upper()}{i}"
            for ci in range(4):
                rows.append({
                    "ticker": tk, "tier": tier, "filing_type": "10-K",
                    "filing_date": "20250101", "section": "mda", "chunk_index": ci,
                    "n_in_section": 4,
                    "embedding_disagreement": val + 0.001 * ((i + ci) % 3),
                })
    df = _tiny_frame(rows)
    w2 = wb.w2_matched(df, "mda", "embedding_disagreement", primary_sign=+1)
    b = w2["bins"]["4-5"]
    assert b["identified_for_three_tier"] is True
    assert b["gradient_small_minus_large"] < 0
    assert b["gradient_ci"]["excludes_zero"] is True
    assert b["contradicts_primary"] is True
    assert w2["criterion_met"] is False
    assert w2["contradicting_bins"] == ["4-5"]


def test_w2_cem_accounting_adds_up(planted):
    df, _ = planted
    w2 = wb.w2_matched(df, "mda", "embedding_disagreement", primary_sign=1)
    cem = w2["coarsened_exact_matching"]
    kept = sum(s["n_small"] + s["n_large"] for s in cem["strata_retained"])
    discarded = cem["n_filings_discarded_no_overlap"]
    n_small_large = int(
        pd.DataFrame(w2["filing_table"])["tier"].isin(["small", "large"]).sum())
    assert kept + discarded == n_small_large
    assert cem["n_mid_filings_outside_contrast"] == int(
        (pd.DataFrame(w2["filing_table"])["tier"] == "mid").sum())
    assert cem["n_strata_retained"] == len(cem["strata_retained"])
    for s in cem["strata_discarded"]:
        assert s["n_small"] == 0 or s["n_large"] == 0
        assert "no overlap" in s["reason"]
    assert np.isfinite(cem["weighted_small_minus_large"])
    assert cem["sign_agrees_with_primary"] is True


# ===========================================================================
# 9. W4 position control
# ===========================================================================

def test_w4_position_decile_formula():
    rows = [
        # a 3-window section: 0 -> 0.0, 1 -> 0.5, 2 -> 1.0
        {"ticker": "A", "tier": "large", "filing_type": "10-K", "filing_date": "20250101",
         "section": "mda", "chunk_index": 0, "n_in_section": 3,
         "embedding_disagreement": 0.1},
        {"ticker": "A", "tier": "large", "filing_type": "10-K", "filing_date": "20250101",
         "section": "mda", "chunk_index": 1, "n_in_section": 3,
         "embedding_disagreement": 0.2},
        {"ticker": "A", "tier": "large", "filing_type": "10-K", "filing_date": "20250101",
         "section": "mda", "chunk_index": 2, "n_in_section": 3,
         "embedding_disagreement": 0.3},
        # a 1-window section: denominator floored at 1, so 0.0 not a zero-division
        {"ticker": "B", "tier": "small", "filing_type": "10-K", "filing_date": "20250101",
         "section": "mda", "chunk_index": 0, "n_in_section": 1,
         "embedding_disagreement": 0.4},
    ]
    pos = wb.position_decile_frame(_tiny_frame(rows), section="mda")
    got = dict(zip(zip(pos["ticker"], pos["chunk_index"].astype(int)),
                   pos["position_decile"]))
    assert math.isclose(got[("A", 0)], 0.0)
    assert math.isclose(got[("A", 1)], 0.5)
    assert math.isclose(got[("A", 2)], 1.0)
    assert math.isclose(got[("B", 0)], 0.0)
    assert set(pos["position_decile_bin"]) <= set(range(1, 11))


def test_w4_dropout_from_excluding_the_first_window_is_reported(planted):
    df, _ = planted
    w4 = wb.w4_position(df, "mda", "embedding_disagreement", primary_sign=1)
    v = w4["variants"]["excluding_first_window"]
    counts = (df[df["section"] == "mda"]
              .groupby(["ticker", "filing_type", "filing_date"])["chunk_index"].size())
    assert v["n_windows"] == int((counts - 1).sum())
    assert v["n_filings_reduced_to_one_window"] == int((counts == 2).sum())
    assert v["n_one_window_sections_contributing_zero"] == int((counts == 1).sum())
    assert v["n_filings_dropped"] == int((counts <= 1).sum())
    first = w4["variants"]["first_2_windows"]
    assert first["n_windows"] == int(counts.clip(upper=2).sum())
    assert first["n_filings_dropped"] == 0


def test_w4_directional_agreement_under_a_planted_effect(planted):
    df, _ = planted
    w4 = wb.w4_position(df, "mda", "embedding_disagreement", primary_sign=1)
    for name, v in w4["variants"].items():
        assert v["sign_agrees_with_primary"] is True, name
    assert w4["criterion_met"] is True


# ===========================================================================
# 10. W7 window-size check
# ===========================================================================

def test_w7_exact_window_accounting_and_gate_reporting(planted):
    df, _ = planted
    w7 = wb.w7_window_size(df, "mda", "embedding_disagreement", primary_sign=1)
    e = w7["exact_windows_only"]
    mda = df[(df["section"] == "mda") & ~df["any_failed"].astype(bool)]
    n_exact = int((mda["word_count"] == 1200).sum())
    assert e["n_windows"] == n_exact
    assert e["n_windows_dropped"] == len(mda) - n_exact
    # tails really are excluded
    assert "any_filing_below_gate" in e
    counts = mda[mda["word_count"] == 1200].groupby(
        ["ticker", "filing_type", "filing_date"])["chunk_index"].size()
    assert e["n_filings_below_min_usable_chunks_gate"] == int((counts < 2).sum())
    if e["any_filing_below_gate"]:
        assert "gradient_if_gate_applied" in e


def test_w7_ols_recovers_the_planted_tier_coefficient(planted):
    df, spec = planted
    w7 = wb.w7_window_size(df, "mda", "embedding_disagreement", primary_sign=1)
    o = w7["word_count_covariate_ols"]
    small = o["coefficients"]["tier_small"]
    # the planted window-level effect is 2 * tier_effect between small and large
    assert small["coef"] > 0
    assert abs(small["coef"] - 2 * spec.tier_effect) < 0.02
    assert small["ci_excludes_zero"] is True
    assert small["ci_excludes_zero_cluster_firm"] in (True, False)
    assert o["n_observations"] == int(
        ((df["section"] == "mda") & ~df["any_failed"].astype(bool)).sum())
    assert o["n_clusters_firm"] == 30
    assert o["sign_agrees_with_primary"] is True
    assert w7["criterion_met"] is True


def test_w7_ols_finds_no_tier_effect_in_the_artifact_fixture(artifact):
    df, _, _ = artifact
    w7 = wb.w7_window_size(df, "mda", "embedding_disagreement", primary_sign=1)
    small = w7["word_count_covariate_ols"]["coefficients"]["tier_small"]
    assert small["ci_excludes_zero_cluster_firm"] is False, (
        "a firm-clustered CI excluded 0 although the true tier effect is 0")


def test_ols_matches_a_closed_form_regression():
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=n)
    y = 1.5 + 2.0 * x + rng.normal(scale=0.1, size=n)
    X = np.column_stack([np.ones(n), x])
    res = wb._ols(y, X, ["intercept", "x"])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    assert math.isclose(res["coefficients"]["intercept"]["coef"], beta[0], rel_tol=1e-9)
    assert math.isclose(res["coefficients"]["x"]["coef"], beta[1], rel_tol=1e-9)
    assert abs(res["coefficients"]["x"]["coef"] - 2.0) < 0.02
    assert res["df_residual"] == n - 2


# ===========================================================================
# 11. Orchestration, determinism, reporting, offline
# ===========================================================================

def test_run_battery_is_deterministic(planted):
    df, _ = planted
    a = wb.run_battery(df, seed=42, w3_iterations=200, w6_iterations=200,
                       n_permutations=200)
    b = wb.run_battery(df, seed=42, w3_iterations=200, w6_iterations=200,
                       n_permutations=200)
    assert json.dumps(wb._to_jsonable(a), sort_keys=True) == \
        json.dumps(wb._to_jsonable(b), sort_keys=True)
    c = wb.run_battery(df, seed=43, w3_iterations=200, w6_iterations=200,
                       n_permutations=200)
    assert (a["W3"]["share_primary_sign_holds_k2"]
            != c["W3"]["share_primary_sign_holds_k2"]
            or a["W6"]["null_gradient"]["sd"] != c["W6"]["null_gradient"]["sd"])
    assert a["meta"]["seed"] == 42


def test_run_battery_records_the_seed_and_the_provenance(planted_battery):
    m = planted_battery["meta"]
    assert m["seed"] == SEED
    assert m["w3_iterations"] == wb.W3_ITERATIONS
    assert m["w6_iterations"] == wb.W6_ITERATIONS
    assert m["network_calls"] == 0 and m["model_calls"] == 0
    assert set(m["tests"]) == {"W1", "W2", "W3", "W4", "W6", "W7"}
    assert "W5" not in m["tests"]
    assert set(m["sibling_provenance"]) == {
        "aggregate_to_firm", "jonckheere_terpstra", "label_permutation_test",
        "tier_trend_statistic"}


def test_gate_reports_w5_as_outstanding(planted_battery, artifact_battery):
    assert planted_battery["gate"]["W5_passed"] is None
    assert planted_battery["gate"]["battery_half_of_gate_passed"] is True
    assert "requires W5" in planted_battery["gate"]["verdict_pending_W5"]
    assert artifact_battery["gate"]["battery_half_of_gate_passed"] is False
    assert "NOT ROBUST" in artifact_battery["gate"]["verdict_pending_W5"]


def test_write_report_emits_txt_and_machine_readable_json(planted_battery):
    with tempfile.TemporaryDirectory() as d:
        paths = wb.write_report(planted_battery, out_dir=d)
        assert paths["report_txt"].name == "report.txt"
        assert paths["report_json"].name == "report.json"
        txt = paths["report_txt"].read_text()
        js = json.loads(paths["report_json"].read_text())
    for marker in ("W1  DESCRIPTIVES", "W2  MATCHED COMPARISON",
                   "W3  EQUAL-k SUBSAMPLING", "W4  POSITION CONTROL",
                   "W6  NULL SIMULATION", "W7  WINDOW-SIZE CHECK",
                   "GATE (ANALYSIS_PLAN.md section 5)"):
        assert marker in txt, marker
    assert "W5" in txt
    assert set(js) >= {"meta", "primary", "W1", "W2", "W3", "W4", "W6", "W7", "gate"}
    assert js["meta"]["seed"] == SEED
    assert js["W3"]["share_primary_sign_holds_k2"] == \
        planted_battery["W3"]["share_primary_sign_holds_k2"]
    # write_report uses allow_nan=False, so a non-finite number would have raised
    # rather than emitting the non-standard NaN/Infinity literals; json.loads
    # above therefore proves the file is portable JSON.
    assert js["W1"]["criterion"] is None


def test_missing_schema_columns_raise(planted):
    df, _ = planted
    broken = df.drop(columns=["word_count"])
    assert wb.check_schema(broken) == ["word_count"]
    with pytest.raises(ValueError):
        wb.run_battery(broken)


def test_battery_makes_no_network_calls(planted):
    """
    Every analysis here must run on cached output.  Block the socket layer and
    confirm the whole battery still completes.
    """
    import socket

    class _Blocked(socket.socket):
        def __init__(self, *a, **k):
            raise AssertionError("the window battery attempted a network call")

    df, _ = planted
    real_socket, real_create = socket.socket, socket.create_connection
    try:
        socket.socket = _Blocked
        socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("the window battery attempted a network call"))
        res = wb.run_battery(df, seed=1, w3_iterations=100, w6_iterations=100,
                             n_permutations=100)
    finally:
        socket.socket, socket.create_connection = real_socket, real_create
    assert res["meta"]["network_calls"] == 0


def test_tensorflow_is_blocked_before_any_transformers_import():
    assert sys.modules.get("tensorflow", "absent") is None


# ===========================================================================
# Plain-python runner
# ===========================================================================

def _run_all() -> int:
    import inspect
    mod = sys.modules[__name__]
    df_p, spec_p = fx.planted_effect_fixture(seed=11)
    df_a, spec_a, search_a = fx.count_artifact_fixture(seed=101)
    cache = {
        "planted": (df_p, spec_p),
        "artifact": (df_a, spec_a, search_a),
        "planted_battery": wb.run_battery(df_p, seed=SEED, n_permutations=FAST_PERM),
        "artifact_battery": wb.run_battery(df_a, seed=SEED, n_permutations=FAST_PERM),
    }
    failures = 0
    for name, fn in sorted(vars(mod).items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        kwargs = {p: cache[p] for p in inspect.signature(fn).parameters}
        try:
            fn(**kwargs)
            print(f"PASS {name}")
        except Exception as exc:                               # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
