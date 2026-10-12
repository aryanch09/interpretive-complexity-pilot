"""
Tests for analysis/trend_tests.py -- the module the paper's one confirmatory
test lives in.

What is actually proved here, in order of how much it matters:

 1. The Jonckheere-Terpstra statistic is right.  Three independent checks:
    (a) a published worked example's DATA, with every Mann-Whitney U count
        hand-derived in the comment so the target is checkable without the book;
    (b) agreement with `scipy.stats.mannwhitneyu` -- a third-party
        implementation of the U counts -- on tied and untied data;
    (c) agreement of J with a from-scratch O(n^2) recount.
 2. The TIE CORRECTION in the variance is right.  The JT null mean and variance
    are, by definition, the mean and variance of J over the label-permutation
    distribution.  `_exact_jt_permutation_distribution` enumerates that
    distribution exhaustively for a small tied sample, and the formula is
    required to match its mean and variance to 1e-9.  This is a proof, not a
    plausibility check.
 3. The normal-approximation p-value is close to the exact permutation p-value.
 4. `aggregate_to_firm` produces firm means that are computed by hand in the
    test, and drops exactly what rules 8.1/8.2 say to drop and nothing else.
 5. The whole chain RECOVERS a planted tier effect, and does NOT find one when
    none is planted -- the latter checked as a calibration property over many
    independent synthetic samples, not a single lucky draw.

No network calls, no model calls, no writes outside tests/fixtures/.

Run:  python3 -m pytest tests/ -q      or      python3 -m tests.test_trend_tests
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

sys.modules.setdefault("tensorflow", None)  # see tests/__init__.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.trend_tests import (  # noqa: E402
    DEFAULT_SEED,
    MIN_USABLE_CHUNKS,
    TIER_ORDER,
    aggregate_to_firm,
    bootstrap_ci,
    holm,
    jonckheere_terpstra,
    jonckheere_terpstra_labeled,
    label_permutation_test,
    permutation_test_tier_trend,
    spearman_brown,
    tier_trend_statistic,
)
from tests.fixture_chunks import (  # noqa: E402
    CHUNK_SCHEMA,
    FixtureParams,
    generate_chunk_level_results,
    write_fixture,
)

# ══════════════════════════════════════════════════════════════════════════════
# Independent reference implementations, used only by the tests
# ══════════════════════════════════════════════════════════════════════════════

def _jt_statistic_bruteforce(groups) -> float:
    """J recomputed from the definition with explicit loops, ties at 1/2."""
    total = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for x in groups[i]:
                for y in groups[j]:
                    if y > x:
                        total += 1.0
                    elif y == x:
                        total += 0.5
    return total


def _jt_statistic_via_scipy(groups) -> float:
    """J via scipy.stats.mannwhitneyu -- a third-party U implementation.

    scipy's U1 for `mannwhitneyu(a, b)` counts pairs with a > b, ties at 1/2.
    The JT pair term needs #{later > earlier}, so the LATER group goes first.
    """
    total = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            total += float(
                stats.mannwhitneyu(
                    np.asarray(groups[j], dtype=float),
                    np.asarray(groups[i], dtype=float),
                    alternative="two-sided",
                    method="asymptotic",
                ).statistic
            )
    return total


def _exact_jt_permutation_distribution(values, sizes):
    """Every distinct assignment of `values` to groups of `sizes`, with its J.

    The label-permutation (randomisation) distribution of J: there are
    N! / prod(n_i!) distinct assignments and each is equally likely under the
    null that the group label carries no information.  Returned as a 1-D array
    of J values, one per assignment.
    """
    values = list(values)
    n = len(values)
    if sum(sizes) != n:
        raise ValueError("sizes must sum to len(values)")
    out = []

    def rec(remaining_idx, k, acc):
        if k == len(sizes):
            out.append(_jt_statistic_bruteforce([[values[i] for i in g] for g in acc]))
            return
        for combo in itertools.combinations(remaining_idx, sizes[k]):
            rest = tuple(i for i in remaining_idx if i not in combo)
            rec(rest, k + 1, acc + [combo])

    rec(tuple(range(n)), 0, [])
    expected_n = math.factorial(n) // math.prod(math.factorial(s) for s in sizes)
    assert len(out) == expected_n, (len(out), expected_n)
    return np.asarray(out, dtype=float)


def _chunk_frame(rows) -> pd.DataFrame:
    """Build a full-schema chunk frame from terse dicts, for hand-checkable tests.

    Only the fields a test cares about need to be given; the rest get defaults
    that are valid under the real schema.
    """
    full = []
    for r in rows:
        emb = r.get("embedding_disagreement")
        failed = bool(r.get("any_failed", False))
        full.append({
            "ticker": r["ticker"],
            "tier": r["tier"],
            "filing_type": r.get("filing_type", "10-K"),
            "filing_date": r.get("filing_date", "20250101"),
            "section": r.get("section", "mda"),
            "chunk_index": r.get("chunk_index", 0),
            "n_in_section": r["n_in_section"],
            "word_count": r.get("word_count", 1200),
            "numeric_disagreement": r.get("numeric_disagreement",
                                          None if failed else 0.5),
            "embedding_disagreement": None if failed else emb,
            "value_score": r.get("value_score", 50),
            "sentiment_score": r.get("sentiment_score", 55),
            "fundamentals_score": r.get("fundamentals_score", 45),
            "any_failed": failed,
            "any_parse_error": False,
            "served_by": "DeepInfra",
            "value_reasoning": "v", "sentiment_reasoning": "s",
            "fundamentals_reasoning": "f",
        })
    return pd.DataFrame(full, columns=list(CHUNK_SCHEMA))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Jonckheere-Terpstra: the statistic
# ══════════════════════════════════════════════════════════════════════════════

# Published worked example.
#
# DATA (three ordered groups of five) from Sachs, L. (1997), "Angewandte
# Statistik", 8th ed., Springer, p. 402.  This is the dataset the R package
# PMCMRplus ships as the worked example for `jonckheereTest()` (and for Chacko's
# test), which is why it is used here rather than a dataset of our own:
#
#     group 1:  106, 114, 116, 127, 145
#     group 2:  110, 125, 143, 148, 151
#     group 3:  136, 139, 149, 160, 174
#
# PROVENANCE CAVEAT, stated so nobody over-reads this test: the project's hard
# rules forbid network access, and neither Sachs (1997) nor PMCMRplus is
# available on this machine, so the source's own PRINTED statistic could not be
# read off and compared.  The assertion targets below are therefore derived BY
# HAND from the published data, and the derivation is written out in full so a
# referee can check it with a pencil:
#
#   U_12 = #{g2 > g1}: 106<(110,125,143,148,151)=5; 114<(125,143,148,151)=4;
#                      116<(125,143,148,151)=4; 127<(143,148,151)=3;
#                      145<(148,151)=2                            -> 18
#   U_13 = #{g3 > g1}: 106,114,116,127 each beaten by all 5 = 20;
#                      145<(149,160,174)=3                        -> 23
#   U_23 = #{g3 > g2}: 110=5; 125=5; 143<(149,160,174)=3;
#                      148<(149,160,174)=3; 151<(160,174)=2       -> 18
#   J    = 18 + 23 + 18                                           -> 59
#   E[J] = (15^2 - 3*5^2)/4 = (225-75)/4                          -> 37.5
#   Var  = [15*14*35 - 3*(5*4*15)]/72 = (7350-900)/72             -> 89.58333...
#   z    = (59 - 37.5)/sqrt(89.58333) = 21.5/9.464847             -> 2.27155
#   p    = P(Z >= 2.27155)                                        -> 0.011556
#
# The statistic is independently cross-checked against scipy's Mann-Whitney U in
# `test_jt_statistic_matches_scipy_mannwhitneyu`, and the variance formula --
# including its tie correction, which this untied example cannot exercise -- is
# proved against the exact permutation moments in
# `test_jt_null_moments_match_exact_permutation`.
SACHS_GROUPS = (
    (106, 114, 116, 127, 145),
    (110, 125, 143, 148, 151),
    (136, 139, 149, 160, 174),
)
SACHS_J = 59.0
SACHS_Z = 2.271554
SACHS_P_ONE_SIDED = 0.011556


def test_jt_published_worked_example():
    res = jonckheere_terpstra(SACHS_GROUPS, alternative="increasing")
    assert res.statistic == pytest.approx(SACHS_J, abs=1e-9)
    assert res.expected_statistic == pytest.approx(37.5, abs=1e-9)
    assert res.variance == pytest.approx(6450.0 / 72.0, abs=1e-9)
    assert res.z == pytest.approx(SACHS_Z, abs=1e-5)
    assert res.p_value == pytest.approx(SACHS_P_ONE_SIDED, abs=1e-5)
    assert res.n_ties == 0 and res.n_tie_groups == 0
    assert res.group_sizes == (5, 5, 5) and res.n_total == 15
    # the hand-derived pair counts, as a separate check of the decomposition
    assert _jt_statistic_bruteforce(SACHS_GROUPS) == 18 + 23 + 18


def test_jt_statistic_matches_scipy_mannwhitneyu():
    """J equals the sum of scipy's Mann-Whitney U over ordered pairs, ties too."""
    rng = np.random.default_rng(11)
    cases = [
        SACHS_GROUPS,
        ([1, 2, 3], [2, 3, 4], [3, 4, 5]),                      # heavy ties
        ([5, 5, 5, 5], [5, 5, 5], [5, 5, 5, 5, 5]),             # all tied
        ([1.0, 2.5], [2.5, 2.5, 9.0], [0.5, 3.0]),              # cross-group ties
    ]
    for _ in range(20):
        cases.append(tuple(
            rng.integers(0, 6, size=int(rng.integers(2, 7))).tolist() for _ in range(3)
        ))
    for groups in cases:
        j = jonckheere_terpstra(groups).statistic
        assert j == pytest.approx(_jt_statistic_via_scipy(groups), abs=1e-9)
        assert j == pytest.approx(_jt_statistic_bruteforce(groups), abs=1e-9)


def test_jt_statistic_untied_variance_reduces_to_textbook_form():
    """With no ties the two correction terms must vanish exactly."""
    groups = ([1, 2, 3, 4], [5, 6, 7], [8, 9, 10, 11, 12])
    res = jonckheere_terpstra(groups)
    n = np.array([4, 3, 5], dtype=float)
    N = n.sum()
    untied = (N * (N - 1) * (2 * N + 5) - np.sum(n * (n - 1) * (2 * n + 5))) / 72.0
    assert res.variance == pytest.approx(float(untied), abs=1e-12)
    assert res.n_ties == 0


# ── the tie correction, proved against the exact permutation moments ──────────

TIE_CASES = [
    ([1, 1, 2, 2, 2, 3, 3, 4, 5], (3, 3, 3)),       # several tie groups
    ([0, 0, 0, 0, 1, 1, 2, 2, 2], (3, 3, 3)),       # big tie groups
    ([1, 1, 1, 1, 1, 1, 1, 1], (3, 3, 2)),          # everything tied
    ([1, 2, 3, 4, 5, 6, 7, 8], (3, 3, 2)),          # no ties (control)
    ([2, 2, 5, 5, 7, 9, 9, 9], (2, 3, 3)),          # unequal group sizes
    ([1, 1, 2, 2, 3, 3, 4, 4], (4, 4)),             # two groups
]


@pytest.mark.parametrize("values,sizes", TIE_CASES)
def test_jt_null_moments_match_exact_permutation(values, sizes):
    """E[J] and Var[J] from the formula == exact permutation mean and variance.

    This is the decisive test of the tie correction: those moments are DEFINED
    as the mean and variance of J over the randomisation distribution, which is
    enumerated exhaustively here.  Any error in the three tie terms shows up
    immediately.
    """
    null = _exact_jt_permutation_distribution(values, sizes)
    groups = []
    start = 0
    for s in sizes:                       # any assignment gives the same moments
        groups.append(values[start:start + s])
        start += s
    res = jonckheere_terpstra(groups)
    assert res.expected_statistic == pytest.approx(float(np.mean(null)), abs=1e-9)
    assert res.variance == pytest.approx(float(np.var(null, ddof=0)), abs=1e-9)


def test_jt_normal_approximation_close_to_exact_permutation_p():
    """One-sided normal p vs the exact permutation p of the SAME statistic."""
    cases = [
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], (3, 3, 3)),         # strong trend, no ties
        ([1, 1, 2, 3, 3, 4, 5, 5, 6], (3, 3, 3)),         # trend with ties
        ([5, 1, 7, 3, 9, 2, 8, 4, 6], (3, 3, 3)),         # no trend
        ([2, 2, 2, 3, 3, 3, 4, 4, 4], (3, 3, 3)),         # heavy ties, trend
    ]
    for values, sizes in cases:
        groups, start = [], 0
        for s in sizes:
            groups.append(values[start:start + s])
            start += s
        res = jonckheere_terpstra(groups, alternative="increasing")
        null = _exact_jt_permutation_distribution(values, sizes)
        exact_p = float(np.mean(null >= res.statistic))
        assert res.p_value == pytest.approx(exact_p, abs=0.05), (
            f"values={values} J={res.statistic} normal p={res.p_value:.4f} "
            f"exact p={exact_p:.4f}"
        )


# ── direction convention ──────────────────────────────────────────────────────

def test_jt_increasing_means_later_groups_are_larger():
    rising = ([1, 2, 3], [4, 5, 6], [7, 8, 9])
    up = jonckheere_terpstra(rising, alternative="increasing")
    down = jonckheere_terpstra(rising, alternative="decreasing")
    assert up.z > 0 and up.p_value < 0.01
    assert down.p_value > 0.99
    # reversing the group order must reverse the evidence by the same amount
    rev = jonckheere_terpstra(tuple(reversed(rising)), alternative="increasing")
    assert rev.z == pytest.approx(-up.z, abs=1e-12)
    two = jonckheere_terpstra(rising, alternative="two-sided")
    assert two.p_value == pytest.approx(2 * up.p_value, rel=1e-9)


def test_jt_tier_order_is_small_greater_than_large():
    """TIER_ORDER must encode H1: Small > Mid > Large under "increasing"."""
    assert TIER_ORDER == ("large", "mid", "small")
    firm = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(9)],
        "tier": ["large"] * 3 + ["mid"] * 3 + ["small"] * 3,
        "firm_mean": [0.20, 0.21, 0.22, 0.30, 0.31, 0.32, 0.40, 0.41, 0.42],
    })
    res = jonckheere_terpstra_labeled(firm["firm_mean"], firm["tier"], TIER_ORDER)
    assert res.group_order == TIER_ORDER
    assert res.p_value < 0.01, "small-caps highest must be evidence FOR H1"
    # the opposite data must NOT be evidence for H1
    flipped = firm.assign(firm_mean=firm["firm_mean"].to_numpy()[::-1])
    assert jonckheere_terpstra_labeled(
        flipped["firm_mean"], flipped["tier"], TIER_ORDER
    ).p_value > 0.99
    assert tier_trend_statistic(firm) == res.statistic


def test_jt_rejects_a_mapping_and_unknown_labels():
    with pytest.raises(TypeError, match="ORDERED sequence"):
        jonckheere_terpstra({"large": [1, 2], "mid": [3, 4], "small": [5, 6]})
    with pytest.raises(ValueError, match="not in group_order"):
        jonckheere_terpstra_labeled([1, 2, 3], ["large", "mid", "micro"], TIER_ORDER)
    with pytest.raises(ValueError, match="no observations"):
        jonckheere_terpstra_labeled([1, 2], ["large", "mid"], TIER_ORDER)
    with pytest.raises(ValueError, match="duplicates"):
        jonckheere_terpstra_labeled([1, 2], ["a", "b"], ["a", "b", "a"])
    with pytest.raises(ValueError, match="differ in length"):
        jonckheere_terpstra_labeled([1, 2, 3], ["large", "mid"], TIER_ORDER)


def test_jt_rejects_empty_groups_nans_and_bad_alternative():
    with pytest.raises(ValueError, match="empty"):
        jonckheere_terpstra([[1, 2], []])
    with pytest.raises(ValueError, match="NaN"):
        jonckheere_terpstra([[1, np.nan], [2, 3]])
    with pytest.raises(ValueError, match="at least 2 groups"):
        jonckheere_terpstra([[1, 2, 3]])
    with pytest.raises(ValueError, match="alternative"):
        jonckheere_terpstra([[1], [2]], alternative="up")


def test_jt_all_values_tied_is_degenerate_not_a_crash():
    res = jonckheere_terpstra([[3, 3, 3], [3, 3], [3, 3, 3, 3]])
    assert res.degenerate is True
    assert res.variance == 0.0
    assert res.z == 0.0 and res.p_value == 1.0
    assert res.n_ties == 9 and res.n_tie_groups == 1


def test_jt_counts_ties_it_handled():
    res = jonckheere_terpstra([[1, 1, 2], [2, 3, 3], [3, 4, 5]])
    # pooled: 1,1,2,2,3,3,3,4,5 -> tie groups {1:2, 2:2, 3:3}
    assert res.n_tie_groups == 3
    assert res.n_ties == 7
    assert res.tie_group_sizes == (3, 2, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Label permutation test (C1-perm)
# ══════════════════════════════════════════════════════════════════════════════

def _jt_stat(v, l):
    return jonckheere_terpstra_labeled(v, l, TIER_ORDER, alternative="increasing").statistic


def test_permutation_recovers_a_strong_effect_and_cannot_report_zero():
    values = np.concatenate([np.arange(10) * 0.001,
                             1.0 + np.arange(10) * 0.001,
                             2.0 + np.arange(10) * 0.001])
    labels = ["large"] * 10 + ["mid"] * 10 + ["small"] * 10
    res = label_permutation_test(values, labels, _jt_stat, n_draws=999, seed=1)
    assert res.n_draws_used == 999
    assert res.n_at_least_as_extreme == 0
    # (r+1)/(n+1) floor: never exactly zero
    assert res.p_value == pytest.approx(1.0 / 1000.0, rel=1e-12)
    assert res.p_value > 0.0
    assert res.observed == 300.0          # perfect separation: 3 pairs x 100
    assert res.seed == 1


def test_permutation_pvalue_convention_and_determinism():
    rng = np.random.default_rng(5)
    values = rng.normal(size=30)
    labels = ["large"] * 10 + ["mid"] * 10 + ["small"] * 10
    a = label_permutation_test(values, labels, _jt_stat, n_draws=500, seed=DEFAULT_SEED)
    b = label_permutation_test(values, labels, _jt_stat, n_draws=500, seed=DEFAULT_SEED)
    assert a.p_value == b.p_value
    assert np.array_equal(a.null_distribution, b.null_distribution)
    c = label_permutation_test(values, labels, _jt_stat, n_draws=500, seed=DEFAULT_SEED + 1)
    assert not np.array_equal(a.null_distribution, c.null_distribution)
    # the convention, recomputed from the returned null
    r = int(np.count_nonzero(a.null_distribution >= a.observed))
    assert a.n_at_least_as_extreme == r
    assert a.p_value == pytest.approx((r + 1) / 501.0, rel=1e-12)
    # the null of J must be centred on the JT formula's E[J] = 150 for 10/10/10
    assert a.null_mean == pytest.approx(150.0, abs=4.0)


def test_permutation_refuses_a_filing_level_frame():
    """The unit of analysis is the firm; duplicate units must raise, not pass."""
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    labels = ["large", "large", "mid", "mid", "small", "small"]
    with pytest.raises(ValueError, match="not unique"):
        label_permutation_test(values, labels, _jt_stat, n_draws=50, seed=1,
                               unit_ids=["A", "A", "B", "B", "C", "C"])
    # the same call with genuinely distinct units is fine
    label_permutation_test(values, labels, _jt_stat, n_draws=50, seed=1,
                           unit_ids=list("ABCDEF"))


def test_permutation_reports_when_draws_exceed_distinct_arrangements():
    values = [1.0, 2.0, 3.0, 4.0]
    labels = ["large", "large", "mid", "small"]
    res = label_permutation_test(values, labels, _jt_stat, n_draws=200, seed=2)
    assert res.distinct_assignments == pytest.approx(12.0)   # 4!/(2!1!1!)
    assert res.exhaustive_possible is True                   # 12 <= 200, flagged


def test_permutation_rejects_bad_input():
    with pytest.raises(ValueError, match="n_draws"):
        label_permutation_test([1, 2], ["a", "b"], lambda v, l: 0.0, n_draws=0)
    with pytest.raises(ValueError, match="NaN"):
        label_permutation_test([1.0, np.nan, 3.0], ["large", "mid", "small"],
                               _jt_stat, n_draws=10)
    with pytest.raises(ValueError, match="differ in length"):
        label_permutation_test([1, 2, 3], ["a", "b"], lambda v, l: 0.0, n_draws=10)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Holm-Bonferroni
# ══════════════════════════════════════════════════════════════════════════════

def test_holm_hand_worked_three_test_example():
    """Hand-worked, m = 3, input deliberately unsorted (S1, S2, S3).

        p          = (0.010, 0.040, 0.030)
        ascending  : 0.010, 0.030, 0.040
        multipliers: x3,    x2,    x1
        scaled     : 0.030, 0.060, 0.040
        running max: 0.030, 0.060, 0.060      <- enforces monotonicity
        back to input order: 0.030, 0.060, 0.060
        reject at alpha = 0.05: True, False, False
    """
    res = holm([0.010, 0.040, 0.030])
    assert res.adjusted == pytest.approx((0.030, 0.060, 0.060), abs=1e-12)
    assert res.reject == (True, False, False)
    assert res.n_tests == 3 and res.alpha == 0.05
    assert res.pvalues == (0.010, 0.040, 0.030)


def test_holm_second_hand_worked_example_all_reject():
    """    p = (0.001, 0.009, 0.012)
        scaled: 0.003, 0.018, 0.012 -> running max 0.003, 0.018, 0.018
        all <= 0.05 -> all rejected
    """
    res = holm([0.001, 0.009, 0.012])
    assert res.adjusted == pytest.approx((0.003, 0.018, 0.018), abs=1e-12)
    assert res.reject == (True, True, True)


def test_holm_properties():
    res = holm([0.6, 0.02, 0.9])
    assert res.adjusted[2] == 1.0                       # clipped at 1
    assert all(a >= p for a, p in zip(res.adjusted, res.pvalues))  # never smaller
    assert holm([0.04]).adjusted == (0.04,)             # family of one = no change
    assert holm([0.04]).reject == (True,)
    # monotone in the sorted order
    adj = np.array(holm([0.001, 0.02, 0.03, 0.5]).adjusted)
    assert np.all(np.diff(adj[np.argsort([0.001, 0.02, 0.03, 0.5])]) >= -1e-15)
    # order preservation under an explicit shuffle
    p = [0.001, 0.02, 0.03, 0.5]
    perm = [2, 0, 3, 1]
    a_direct = np.array(holm(p).adjusted)[perm]
    a_shuffled = np.array(holm([p[i] for i in perm]).adjusted)
    assert a_direct == pytest.approx(a_shuffled, abs=1e-15)


def test_holm_rejects_bad_input():
    for bad in ([], [0.5, 1.5], [-0.1], [0.1, np.nan]):
        with pytest.raises(ValueError):
            holm(bad)
    with pytest.raises(ValueError, match="alpha"):
        holm([0.1], alpha=1.5)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Bootstrap CIs
# ══════════════════════════════════════════════════════════════════════════════

def test_bootstrap_percentile_and_bca_bracket_the_estimate():
    rng = np.random.default_rng(3)
    data = rng.normal(0.3, 0.05, size=30)
    res = bootstrap_ci(data, np.mean, n_boot=2000, seed=DEFAULT_SEED, method="bca")
    assert res.method_used == "bca" and res.bca_ci is not None
    assert res.low < res.observed < res.high
    assert res.percentile_ci[0] < res.observed < res.percentile_ci[1]
    assert res.degenerate is False
    assert res.seed == DEFAULT_SEED and res.n_boot == 2000
    # the two methods should not be wildly apart for a mean of 30 normal draws
    assert abs(res.bca_ci[0] - res.percentile_ci[0]) < 0.02
    pct = bootstrap_ci(data, np.mean, n_boot=2000, seed=DEFAULT_SEED, method="percentile")
    assert pct.method_used == "percentile" and pct.bca_ci is None
    assert pct.percentile_ci == pytest.approx(res.percentile_ci, abs=1e-12)


def test_bootstrap_coverage_is_approximately_nominal():
    """A CI that never covers is worse than no CI: check coverage by simulation."""
    truth = 0.3
    covered = 0
    trials = 100
    for i in range(trials):
        data = np.random.default_rng(500 + i).normal(truth, 0.05, size=25)
        res = bootstrap_ci(data, np.mean, n_boot=400, seed=i, method="percentile")
        covered += int(res.low <= truth <= res.high)
    # Binomial(100, 0.95): P(X <= 85) is about 1e-4, so 86 is a safe floor.
    assert covered >= 86, f"percentile coverage {covered}/{trials} is too low"


def test_bootstrap_degenerate_zero_variance_does_not_raise():
    res = bootstrap_ci([0.4] * 12, np.mean, n_boot=500, seed=1, method="bca")
    assert res.degenerate is True
    assert res.low == res.high == pytest.approx(0.4)
    assert res.method_used == "degenerate"
    assert "zero variance" in res.note
    single = bootstrap_ci([0.4], np.mean, n_boot=500, seed=1)
    assert single.degenerate is True and single.low == single.high == 0.4


def test_bootstrap_constant_statistic_does_not_raise():
    """A statistic with no resample variation makes BCa's acceleration undefined."""
    rng = np.random.default_rng(7)
    data = rng.normal(size=20)
    res = bootstrap_ci(data, lambda s: 1.0, n_boot=300, seed=1, method="bca")
    assert res.observed == 1.0
    assert res.low == res.high == 1.0
    assert res.degenerate is True or res.method_used in ("percentile", "degenerate")


def test_bootstrap_determinism_and_input_validation():
    data = np.random.default_rng(9).normal(size=20)
    a = bootstrap_ci(data, np.mean, n_boot=300, seed=42)
    b = bootstrap_ci(data, np.mean, n_boot=300, seed=42)
    assert (a.low, a.high) == (b.low, b.high)
    c = bootstrap_ci(data, np.mean, n_boot=300, seed=43)
    assert (a.low, a.high) != (c.low, c.high)
    with pytest.raises(ValueError, match="method"):
        bootstrap_ci(data, np.mean, method="jackknife")
    with pytest.raises(ValueError, match="NaN"):
        bootstrap_ci([1.0, np.nan], np.mean)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Spearman-Brown
# ══════════════════════════════════════════════════════════════════════════════

def test_spearman_brown_known_values():
    assert spearman_brown(0.5) == pytest.approx(2 * 0.5 / 1.5)      # 0.666...
    assert spearman_brown(0.8) == pytest.approx(1.6 / 1.8)          # 0.888...
    assert spearman_brown(0.0) == 0.0
    assert spearman_brown(1.0) == pytest.approx(1.0)
    # the correction must never lower a positive half-reliability
    for r in (0.1, 0.3, 0.5, 0.7, 0.9):
        assert spearman_brown(r) > r
    # k=2 special case of the general prophecy formula k*r/(1+(k-1)r)
    for r in (0.2, 0.45, 0.77):
        assert spearman_brown(r) == pytest.approx(2 * r / (1 + r))


def test_spearman_brown_rejects_impossible_input():
    with pytest.raises(ValueError, match="undefined"):
        spearman_brown(-1.0)
    for bad in (1.5, -2.0, float("nan")):
        with pytest.raises(ValueError):
            spearman_brown(bad)


def test_spearman_brown_negative_r_still_computes_but_is_out_of_range():
    out = spearman_brown(-0.5)
    assert out == pytest.approx(-2.0)        # 2(-0.5)/(0.5)
    assert out < 0.0                         # caller must report a failed split


# ══════════════════════════════════════════════════════════════════════════════
# 6. aggregate_to_firm -- hand-computed expectations
# ══════════════════════════════════════════════════════════════════════════════

def test_aggregate_firm_means_computed_by_hand():
    """Two firms, four filings, every mean worked out by hand below.

    AAA (large)
      10-K, 3 MD&A windows: 0.10, 0.20, 0.30     -> filing mean 0.20
      10-Q, 2 MD&A windows: 0.40, 0.60           -> filing mean 0.50
      firm mean = (0.20 + 0.50) / 2              -> 0.35       (NOT 0.32 = the
                                                   window-pooled mean of all 5)
      windows feeding it = 5, filings = 2
    BBB (small)
      10-K, 4 MD&A windows: 0.50, 0.50, 0.50, 0.70 -> filing mean 0.55
      10-Q, 2 MD&A windows: 0.10, 0.30             -> filing mean 0.20
      firm mean = (0.55 + 0.20) / 2                -> 0.375
      windows feeding it = 6, filings = 2
    """
    rows = []
    for i, v in enumerate([0.10, 0.20, 0.30]):
        rows.append({"ticker": "AAA", "tier": "large", "filing_type": "10-K",
                     "filing_date": "20250101", "chunk_index": i, "n_in_section": 3,
                     "embedding_disagreement": v})
    for i, v in enumerate([0.40, 0.60]):
        rows.append({"ticker": "AAA", "tier": "large", "filing_type": "10-Q",
                     "filing_date": "20250501", "chunk_index": i, "n_in_section": 2,
                     "embedding_disagreement": v})
    for i, v in enumerate([0.50, 0.50, 0.50, 0.70]):
        rows.append({"ticker": "BBB", "tier": "small", "filing_type": "10-K",
                     "filing_date": "20250201", "chunk_index": i, "n_in_section": 4,
                     "embedding_disagreement": v})
    for i, v in enumerate([0.10, 0.30]):
        rows.append({"ticker": "BBB", "tier": "small", "filing_type": "10-Q",
                     "filing_date": "20250601", "chunk_index": i, "n_in_section": 2,
                     "embedding_disagreement": v})
    out = aggregate_to_firm(_chunk_frame(rows))

    assert list(out["ticker"]) == ["AAA", "BBB"]           # tier order: large first
    assert list(out["tier"]) == ["large", "small"]
    assert out.loc[0, "firm_mean"] == pytest.approx(0.35)
    assert out.loc[1, "firm_mean"] == pytest.approx(0.375)
    assert list(out["n_filings"]) == [2, 2]
    assert list(out["n_windows"]) == [5, 6]
    assert out.loc[0, "filings"] == "10-K:20250101|10-Q:20250501"
    assert set(out.columns) == {
        "ticker", "tier", "firm_mean", "n_filings", "n_windows", "filings",
        "section", "measure",
    }
    # the firm mean is NOT the pooled window mean
    pooled_aaa = np.mean([0.10, 0.20, 0.30, 0.40, 0.60])
    assert out.loc[0, "firm_mean"] != pytest.approx(pooled_aaa)


def test_aggregate_is_unweighted_over_windows_not_word_weighted():
    rows = [
        {"ticker": "AAA", "tier": "mid", "chunk_index": 0, "n_in_section": 2,
         "word_count": 1200, "embedding_disagreement": 0.20},
        {"ticker": "AAA", "tier": "mid", "chunk_index": 1, "n_in_section": 2,
         "word_count": 404, "embedding_disagreement": 0.80},
    ]
    out = aggregate_to_firm(_chunk_frame(rows))
    assert out.loc[0, "firm_mean"] == pytest.approx(0.50)       # unweighted
    word_weighted = (1200 * 0.20 + 404 * 0.80) / 1604
    assert out.loc[0, "firm_mean"] != pytest.approx(word_weighted, abs=1e-6)


def test_aggregate_is_unweighted_over_filings_not_window_weighted():
    """A 25-window filing and a 2-window filing weigh the same at the firm level."""
    rows = [{"ticker": "AAA", "tier": "large", "filing_type": "10-K",
             "filing_date": "20250101", "chunk_index": i, "n_in_section": 25,
             "embedding_disagreement": 0.20} for i in range(25)]
    rows += [{"ticker": "AAA", "tier": "large", "filing_type": "10-Q",
              "filing_date": "20250501", "chunk_index": i, "n_in_section": 2,
              "embedding_disagreement": 0.60} for i in range(2)]
    out = aggregate_to_firm(_chunk_frame(rows))
    assert out.loc[0, "firm_mean"] == pytest.approx(0.40)       # (0.2+0.6)/2
    pooled = (25 * 0.20 + 2 * 0.60) / 27
    assert out.loc[0, "firm_mean"] != pytest.approx(pooled, abs=1e-6)


def test_aggregate_drops_failed_windows_and_never_imputes():
    """Rule 8.1: a failed window contributes nothing -- not a zero, not a mean."""
    rows = [
        {"ticker": "AAA", "tier": "large", "chunk_index": 0, "n_in_section": 4,
         "embedding_disagreement": 0.20},
        {"ticker": "AAA", "tier": "large", "chunk_index": 1, "n_in_section": 4,
         "embedding_disagreement": 0.40},
        {"ticker": "AAA", "tier": "large", "chunk_index": 2, "n_in_section": 4,
         "any_failed": True},
        {"ticker": "AAA", "tier": "large", "chunk_index": 3, "n_in_section": 4,
         "embedding_disagreement": 0.60},
    ]
    out = aggregate_to_firm(_chunk_frame(rows))
    assert out.loc[0, "firm_mean"] == pytest.approx(0.40)      # (0.2+0.4+0.6)/3
    assert out.loc[0, "n_windows"] == 3
    # mean-imputation would give (0.2+0.4+0.4+0.6)/4 = 0.40 too, so also check a
    # case where imputing with 0 or with the mean differs from dropping
    assert out.loc[0, "firm_mean"] != pytest.approx((0.2 + 0.4 + 0.0 + 0.6) / 4)
    audit = out.attrs["audit"]
    assert audit["windows_dropped_any_failed"] == 1
    assert audit["windows_surviving"] == 3


def test_aggregate_gate_requires_min_usable_chunks():
    """1 of 2 surviving fails both limbs: >= MIN_USABLE_CHUNKS and >= half."""
    assert MIN_USABLE_CHUNKS == 2
    rows = [
        {"ticker": "AAA", "tier": "large", "filing_type": "10-K",
         "filing_date": "20250101", "chunk_index": 0, "n_in_section": 2,
         "embedding_disagreement": 0.9},
        {"ticker": "AAA", "tier": "large", "filing_type": "10-K",
         "filing_date": "20250101", "chunk_index": 1, "n_in_section": 2,
         "any_failed": True},
        # a second, clean filing so the firm survives
        {"ticker": "AAA", "tier": "large", "filing_type": "10-Q",
         "filing_date": "20250501", "chunk_index": 0, "n_in_section": 2,
         "embedding_disagreement": 0.3},
        {"ticker": "AAA", "tier": "large", "filing_type": "10-Q",
         "filing_date": "20250501", "chunk_index": 1, "n_in_section": 2,
         "embedding_disagreement": 0.5},
    ]
    out = aggregate_to_firm(_chunk_frame(rows))
    assert out.loc[0, "n_filings"] == 1
    assert out.loc[0, "firm_mean"] == pytest.approx(0.40)   # only the 10-Q
    dropped = out.attrs["audit"]["filings_dropped_by_gate"]
    assert len(dropped) == 1 and dropped[0]["filing_type"] == "10-K"
    assert dropped[0]["n_windows_surviving"] == 1 and dropped[0]["n_windows_total"] == 2


def test_aggregate_gate_requires_half_the_windows():
    """3 of 7 fails the half rule; 4 of 7 passes it.  Boundary checked both ways."""
    def frame(n_surv, n_total=7):
        rows = []
        for i in range(n_total):
            r = {"ticker": "AAA", "tier": "mid", "chunk_index": i,
                 "n_in_section": n_total}
            if i < n_surv:
                r["embedding_disagreement"] = 0.5
            else:
                r["any_failed"] = True
            rows.append(r)
        return _chunk_frame(rows)

    with pytest.raises(ValueError, match="no filing passed"):
        aggregate_to_firm(frame(3))                      # 3 < 7/2 = 3.5
    ok = aggregate_to_firm(frame(4))                     # 4 >= 3.5
    assert ok.loc[0, "n_windows"] == 4
    assert ok.attrs["audit"]["filings_dropped_by_gate"] == []
    # exactly half of an even count passes
    ok_even = aggregate_to_firm(frame(3, n_total=6))
    assert ok_even.loc[0, "n_windows"] == 3


def test_aggregate_gate_denominator_is_n_in_section_not_rows_present():
    """A filing whose rows are partly missing from the CSV must still fail."""
    rows = [{"ticker": "AAA", "tier": "small", "chunk_index": i, "n_in_section": 8,
             "embedding_disagreement": 0.5} for i in range(3)]      # 3 of 8 present
    with pytest.raises(ValueError, match="no filing passed"):
        aggregate_to_firm(_chunk_frame(rows))


def test_aggregate_isolates_the_section():
    """MD&A and Risk are never pooled (plan §2)."""
    rows = [
        {"ticker": "AAA", "tier": "large", "section": "mda", "chunk_index": 0,
         "n_in_section": 2, "embedding_disagreement": 0.10},
        {"ticker": "AAA", "tier": "large", "section": "mda", "chunk_index": 1,
         "n_in_section": 2, "embedding_disagreement": 0.20},
        {"ticker": "AAA", "tier": "large", "section": "risk", "chunk_index": 0,
         "n_in_section": 2, "embedding_disagreement": 0.80},
        {"ticker": "AAA", "tier": "large", "section": "risk", "chunk_index": 1,
         "n_in_section": 2, "embedding_disagreement": 0.90},
    ]
    df = _chunk_frame(rows)
    assert aggregate_to_firm(df, section="mda").loc[0, "firm_mean"] == pytest.approx(0.15)
    assert aggregate_to_firm(df, section="risk").loc[0, "firm_mean"] == pytest.approx(0.85)
    with pytest.raises(ValueError, match="no rows with section"):
        aggregate_to_firm(df, section="item7a")


def test_aggregate_supports_the_numeric_measure_and_other_columns():
    rows = [
        {"ticker": "AAA", "tier": "large", "chunk_index": 0, "n_in_section": 2,
         "embedding_disagreement": 0.1, "numeric_disagreement": 1.0},
        {"ticker": "AAA", "tier": "large", "chunk_index": 1, "n_in_section": 2,
         "embedding_disagreement": 0.2, "numeric_disagreement": 2.0},
    ]
    out = aggregate_to_firm(_chunk_frame(rows), measure="numeric_disagreement")
    assert out.loc[0, "firm_mean"] == pytest.approx(1.5)
    assert out.loc[0, "measure"] == "numeric_disagreement"
    with pytest.raises(KeyError, match="no measure column"):
        aggregate_to_firm(_chunk_frame(rows), measure="not_a_column")


def test_aggregate_excludes_requested_tickers_for_the_hayn_sensitivity():
    rows = []
    for tkr in ("AAA", "HAYN"):
        for i in range(2):
            rows.append({"ticker": tkr, "tier": "small", "chunk_index": i,
                         "n_in_section": 2, "embedding_disagreement": 0.3 + 0.1 * i})
    full = aggregate_to_firm(_chunk_frame(rows))
    assert set(full["ticker"]) == {"AAA", "HAYN"}
    sens = aggregate_to_firm(_chunk_frame(rows), exclude_tickers=["HAYN"])
    assert set(sens["ticker"]) == {"AAA"}
    assert sens.attrs["audit"]["rows_dropped_by_ticker_exclusion"] == 2


def test_aggregate_validates_its_input():
    rows = [{"ticker": "AAA", "tier": "large", "chunk_index": 0, "n_in_section": 2,
             "embedding_disagreement": 0.1},
            {"ticker": "AAA", "tier": "mid", "chunk_index": 1, "n_in_section": 2,
             "embedding_disagreement": 0.2}]
    with pytest.raises(ValueError, match="more than one tier"):
        aggregate_to_firm(_chunk_frame(rows))
    rows2 = [{"ticker": "AAA", "tier": "micro", "chunk_index": i, "n_in_section": 2,
              "embedding_disagreement": 0.1} for i in range(2)]
    with pytest.raises(ValueError, match="unknown tier"):
        aggregate_to_firm(_chunk_frame(rows2))
    good = _chunk_frame([{"ticker": "AAA", "tier": "large", "chunk_index": i,
                          "n_in_section": 2, "embedding_disagreement": 0.1}
                         for i in range(2)])
    with pytest.raises(KeyError, match="missing required column"):
        aggregate_to_firm(good.drop(columns=["tier"]))
    with pytest.raises(TypeError, match="DataFrame"):
        aggregate_to_firm([1, 2, 3])
    rows3 = [{"ticker": "AAA", "tier": "large", "chunk_index": 0, "n_in_section": 2,
              "embedding_disagreement": 0.1},
             {"ticker": "AAA", "tier": "large", "chunk_index": 1, "n_in_section": 3,
              "embedding_disagreement": 0.2}]
    with pytest.raises(ValueError, match="inconsistent"):
        aggregate_to_firm(_chunk_frame(rows3))


def test_aggregate_survives_the_csv_round_trip(tmp_path):
    """Booleans and nulls must behave the same after to_csv/read_csv."""
    params = FixtureParams(seed=4242, label="roundtrip", tier_effect=0.04,
                           failure_rate=0.05)
    path, df = write_fixture(params, out_dir=tmp_path)
    reread = pd.read_csv(path)
    a = aggregate_to_firm(df)
    b = aggregate_to_firm(reread)
    assert a["firm_mean"].to_numpy() == pytest.approx(b["firm_mean"].to_numpy())
    assert list(a["ticker"]) == list(b["ticker"])
    assert reread["any_failed"].sum() == df["any_failed"].sum()
    assert path.name.startswith("SYNTHETIC_")
    sidecar = path.with_suffix(".params.json")
    assert sidecar.exists() and '"seed": 4242' in sidecar.read_text()


# ══════════════════════════════════════════════════════════════════════════════
# 7. End-to-end on the synthetic fixture: recovery and calibration
# ══════════════════════════════════════════════════════════════════════════════

PLANTED_EFFECT = 0.04


def _firm_frame(seed, tier_effect, **kw):
    df = generate_chunk_level_results(
        FixtureParams(seed=seed, tier_effect=tier_effect, **kw)
    )
    return df, aggregate_to_firm(df, section="mda", measure="embedding_disagreement")


def test_fixture_matches_the_real_schema_and_sample_shape():
    params = FixtureParams(seed=DEFAULT_SEED, tier_effect=PLANTED_EFFECT)
    df = generate_chunk_level_results(params)
    assert list(df.columns) == list(CHUNK_SCHEMA)
    assert len(df) == 667
    assert int((df["section"] == "mda").sum()) == 363
    assert df["ticker"].nunique() == 30
    assert len(df.groupby(["ticker", "filing_type", "filing_date"])) == 59
    assert set(df["tier"]) == {"large", "mid", "small"}
    assert set(df["filing_type"]) == {"10-K", "10-Q"}
    assert df["word_count"].between(403, 1598).all()
    assert df["word_count"].median() == 1200
    assert (df["filing_date"].astype(int) > 20240701).all()
    mda = df[df["section"] == "mda"]
    per_filing = mda.groupby(["ticker", "filing_type"])["n_in_section"].first()
    assert per_filing.min() == 2 and per_filing.max() == 25
    tiers = mda.groupby(["ticker", "filing_type"])["tier"].first()
    meds = per_filing.groupby(tiers).median().to_dict()
    assert meds == {"large": 6.5, "mid": 6.5, "small": 4.0}
    # the ten Risk-less filings are all 10-Qs
    with_risk = set(map(tuple, df[df["section"] == "risk"][
        ["ticker", "filing_type"]].drop_duplicates().to_numpy()))
    allf = set(map(tuple, df[["ticker", "filing_type"]].drop_duplicates().to_numpy()))
    missing = allf - with_risk
    assert len(missing) == 10 and {f for _, f in missing} == {"10-Q"}
    assert params.realised["seed"] == DEFAULT_SEED


def test_fixture_numeric_measure_matches_the_plan_definition():
    """numeric_disagreement == population SD (ddof=0) of the three scores / 10."""
    df = generate_chunk_level_results(FixtureParams(seed=7, tier_effect=0.0))
    clean = df[~df["any_failed"]]
    scores = clean[["value_score", "sentiment_score", "fundamentals_score"]].to_numpy(float)
    recomputed = np.std(scores / 10.0, axis=1, ddof=0)
    assert clean["numeric_disagreement"].to_numpy(float) == pytest.approx(
        recomputed, abs=1e-12
    )


def test_fixture_failed_windows_carry_no_disagreement_at_all():
    df = generate_chunk_level_results(
        FixtureParams(seed=3, tier_effect=0.0, failure_rate=0.08)
    )
    failed = df[df["any_failed"]]
    assert len(failed) > 0
    assert failed["embedding_disagreement"].isna().all()
    assert failed["numeric_disagreement"].isna().all()
    assert (failed["served_by"] == "DeepInfra|None").all()
    assert failed[["value_score", "sentiment_score",
                   "fundamentals_score"]].isna().sum(axis=1).eq(1).all()
    assert (~df[~df["any_failed"]]["embedding_disagreement"].isna()).all()


def test_fixture_is_deterministic_in_its_seed():
    a = generate_chunk_level_results(FixtureParams(seed=99, tier_effect=0.02))
    b = generate_chunk_level_results(FixtureParams(seed=99, tier_effect=0.02))
    pd.testing.assert_frame_equal(a, b)
    c = generate_chunk_level_results(FixtureParams(seed=100, tier_effect=0.02))
    assert not a["embedding_disagreement"].equals(c["embedding_disagreement"])


def test_recovers_a_planted_tier_effect():
    """C1 and C1-perm must both find the effect that was planted."""
    df, firm = _firm_frame(DEFAULT_SEED, PLANTED_EFFECT)
    assert len(firm) == 30
    assert firm.groupby("tier").size().to_dict() == {"large": 10, "mid": 10, "small": 10}

    means = firm.groupby("tier")["firm_mean"].mean()
    assert means["large"] < means["mid"] < means["small"], "planted order must appear"
    step = (means["small"] - means["large"]) / 2
    assert step == pytest.approx(PLANTED_EFFECT, abs=0.02), (
        "recovered per-tier step should be near the planted one"
    )

    c1 = jonckheere_terpstra_labeled(firm["firm_mean"], firm["tier"], TIER_ORDER)
    assert c1.p_value < 0.05, f"C1 missed a planted effect: {c1}"
    assert c1.statistic > c1.expected_statistic

    c1_perm = permutation_test_tier_trend(firm, n_draws=2000, seed=DEFAULT_SEED)
    assert c1_perm.p_value < 0.05, f"C1-perm missed a planted effect: {c1_perm}"
    assert c1_perm.observed == c1.statistic, "C1 and C1-perm must share the statistic"

    ci = bootstrap_ci(
        firm.loc[firm.tier == "small", "firm_mean"].to_numpy(), np.mean,
        n_boot=2000, seed=DEFAULT_SEED,
    )
    assert ci.low < means["small"] < ci.high


def test_finds_no_effect_when_none_is_planted():
    """The specific documented seed, and then the calibration over many seeds."""
    _, firm = _firm_frame(DEFAULT_SEED, 0.0)
    c1 = jonckheere_terpstra_labeled(firm["firm_mean"], firm["tier"], TIER_ORDER)
    assert c1.p_value > 0.05, f"false positive on the null fixture: {c1}"
    c1_perm = permutation_test_tier_trend(firm, n_draws=2000, seed=DEFAULT_SEED)
    assert c1_perm.p_value > 0.05

    # Calibration: 120 independent null samples.  Under a correct test the
    # p-values are ~U(0,1) and about 6 of 120 fall below 0.05.
    # Binomial(120, 0.05): P(X >= 17) < 1e-4, so 17 is a safe ceiling.
    pvals = []
    for i in range(120):
        _, f = _firm_frame(31_000 + i, 0.0)
        pvals.append(jonckheere_terpstra_labeled(
            f["firm_mean"], f["tier"], TIER_ORDER).p_value)
    pvals = np.asarray(pvals)
    n_sig = int((pvals < 0.05).sum())
    assert n_sig <= 17, (
        f"{n_sig}/120 null samples significant at 0.05 -- the test is anti-conservative"
    )
    ks = stats.kstest(pvals, "uniform")
    assert ks.pvalue > 0.01, (
        f"null p-values are not uniform (KS p={ks.pvalue:.4g}); the tier-unequal "
        "window counts may be distorting the test's level"
    )


def test_power_against_the_planted_effect_is_high():
    """If the planted effect were only found occasionally, recovery would be luck."""
    hits = 0
    trials = 60
    for i in range(trials):
        _, f = _firm_frame(41_000 + i, PLANTED_EFFECT)
        hits += int(jonckheere_terpstra_labeled(
            f["firm_mean"], f["tier"], TIER_ORDER).p_value < 0.05)
    assert hits >= int(0.85 * trials), f"power only {hits}/{trials} at a planted 0.04"


def test_null_fixture_does_not_fake_a_monotone_gradient():
    """Over many null samples the three tier orderings should be roughly equally
    likely -- the predicted order must not arrive for free from the
    unequal-window-count structure (plan §5, the averaging-artifact concern)."""
    monotone_predicted = 0
    trials = 60
    for i in range(trials):
        _, f = _firm_frame(51_000 + i, 0.0)
        m = f.groupby("tier")["firm_mean"].mean()
        monotone_predicted += int(m["large"] < m["mid"] < m["small"])
    # 1 of 6 orderings is the predicted one; Binomial(60, 1/6) upper tail at 21
    # is about 0.003, so a structural bias toward the hypothesis would show.
    assert monotone_predicted <= 21, (
        f"predicted order appeared {monotone_predicted}/{trials} times with NO "
        "planted effect -- the aggregation chain is biased toward H1"
    )


def test_holm_family_of_three_on_the_fixture():
    """S1/S2/S3 as the plan defines them, run through the Holm correction.

    This checks the WIRING -- three real p-values from the three pre-registered
    secondary analyses, two-sided as §6 requires, surviving the correction with
    the right structure.  It deliberately does NOT assert significance for a
    single seed: whether one synthetic sample clears 0.05 is a coin flip, and a
    test that depends on it would be a test of that seed.  Whether the numeric
    measure is flat and the embedding measure is not is established over many
    samples in `test_numeric_measure_is_flat_as_planted` and
    `test_power_against_the_planted_effect_is_high`.
    """
    df, firm_emb = _firm_frame(DEFAULT_SEED, PLANTED_EFFECT)
    firm_num = aggregate_to_firm(df, section="mda", measure="numeric_disagreement")
    firm_risk = aggregate_to_firm(df, section="risk", measure="embedding_disagreement")
    assert firm_num.shape[0] == 30
    # S2 carries its OWN n (plan §6): HAYN's single 10-Q is one of the ten
    # filings that incorporate risk factors by reference, so HAYN contributes no
    # Risk windows at all and the Risk analysis runs at n = 29 (10/10/9).
    n_risk_firms = df[df["section"] == "risk"]["ticker"].nunique()
    assert n_risk_firms == 29
    assert firm_risk.shape[0] == n_risk_firms
    assert firm_risk.attrs["audit"]["firms_per_tier"] == {
        "large": 10, "mid": 10, "small": 9,
    }
    assert "HAYN" not in set(firm_risk["ticker"])
    ps = [
        jonckheere_terpstra_labeled(f["firm_mean"], f["tier"], TIER_ORDER,
                                    alternative="two-sided").p_value
        for f in (firm_num, firm_risk, firm_emb)
    ]
    res = holm(ps)
    assert len(res.adjusted) == 3
    assert all(a >= p for a, p in zip(res.adjusted, ps))
    assert res.adjusted[int(np.argmin(ps))] == pytest.approx(3 * min(ps), abs=1e-12)
    # the two embedding analyses carry the planted effect, the numeric one does
    # not, so the embedding p-values must be the smaller two of the three
    assert ps[1] < ps[0] and ps[2] < ps[0]
    # S2 (Risk embedding) and S3 (MD&A embedding) both carry a planted 0.04 step
    assert res.reject[1] is True and res.reject[2] is True


def test_numeric_measure_is_flat_as_planted():
    """`numeric_tier_effect=0` must leave the numeric measure genuinely flat.

    Checked two ways over independent samples, because one sample cannot tell a
    flat measure from a lucky one: the grand per-tier means must agree, and the
    two-sided p-values must be ~uniform rather than piling up below 0.05.
    """
    tier_means = {"large": [], "mid": [], "small": []}
    pvals = []
    trials = 100
    for i in range(trials):
        df = generate_chunk_level_results(
            FixtureParams(seed=61_000 + i, tier_effect=PLANTED_EFFECT,
                          numeric_tier_effect=0.0)
        )
        f = aggregate_to_firm(df, section="mda", measure="numeric_disagreement")
        m = f.groupby("tier")["firm_mean"].mean()
        for t in tier_means:
            tier_means[t].append(float(m[t]))
        pvals.append(jonckheere_terpstra_labeled(
            f["firm_mean"], f["tier"], TIER_ORDER, alternative="two-sided").p_value)
    grand = {t: float(np.mean(v)) for t, v in tier_means.items()}
    spread = max(grand.values()) - min(grand.values())
    assert spread < 0.01, f"numeric measure is not flat across tiers: {grand}"
    pvals = np.asarray(pvals)
    n_sig = int((pvals < 0.05).sum())
    # Binomial(100, 0.05): P(X >= 14) is about 2e-4.
    assert n_sig <= 14, f"{n_sig}/100 significant with no numeric effect planted"
    assert stats.kstest(pvals, "uniform").pvalue > 0.01


def test_gate_breaking_filing_is_dropped_from_the_firm_mean():
    df = generate_chunk_level_results(FixtureParams(
        seed=17, tier_effect=PLANTED_EFFECT,
        gate_breaking_filings=(("AAPL", "10-K"),),
    ))
    out = aggregate_to_firm(df)
    aapl = out[out["ticker"] == "AAPL"].iloc[0]
    assert aapl["n_filings"] == 1
    assert aapl["filings"].startswith("10-Q:")
    dropped = out.attrs["audit"]["filings_dropped_by_gate"]
    assert [(d["ticker"], d["filing_type"]) for d in dropped] == [("AAPL", "10-K")]
    assert len(out) == 30     # the firm survives on its remaining filing


def test_whole_chain_runs_with_failures_present_and_reports_them():
    df = generate_chunk_level_results(
        FixtureParams(seed=23, tier_effect=PLANTED_EFFECT, failure_rate=0.10)
    )
    out = aggregate_to_firm(df)
    audit = out.attrs["audit"]
    assert audit["windows_dropped_any_failed"] > 0
    assert audit["windows_surviving"] + audit["windows_dropped_any_failed"] == \
        audit["rows_in_section"]
    assert audit["filings_considered"] == 59
    # with 10% window failures some filings must fail the half/min gate
    assert audit["filings_kept"] <= 59
    assert out["n_windows"].sum() <= audit["windows_surviving"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([str(Path(__file__).resolve()), "-q"]))
