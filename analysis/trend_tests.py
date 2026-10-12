"""
Statistical primitives for the Round 8 confirmatory analysis.

This module holds the tests the pre-registered plan (`ANALYSIS_PLAN.md`) names,
and nothing else.  Every other analysis module is expected to call these rather
than re-deriving them, so there is exactly one implementation of each rule.

What lives here
---------------
    jonckheere_terpstra      C1: ordered-trend test, normal approximation with
                             the tie correction for the variance.  scipy has no
                             implementation.
    jonckheere_terpstra_labeled
                             The same test, entered by (values, labels, explicit
                             ordered label list) so the direction cannot be
                             transposed at the call site.
    tier_trend_statistic     The single scalar C1-perm permutes.  Identical
                             statistic to C1, so the two test one hypothesis.
    label_permutation_test   C1-perm: tier labels permuted across FIRMS.
    holm                     Holm-Bonferroni for the §6 family of three.
    bootstrap_ci             Percentile and BCa intervals.
    spearman_brown           Split-half reliability correction.
    aggregate_to_firm        The §1 aggregation chain: window -> filing -> firm.

Direction convention (read this before calling anything)
--------------------------------------------------------
`jonckheere_terpstra` takes an ORDERED SEQUENCE of groups, never a dict, and
`alternative="increasing"` means **values tend to be LARGER in groups that come
LATER in the sequence you passed**.  H1 is Small > Mid > Large, so the plan's
one-sided test is

    jonckheere_terpstra([large_values, mid_values, small_values],
                        alternative="increasing")

which is what `TIER_ORDER = ("large", "mid", "small")` encodes.  Call sites
should go through `jonckheere_terpstra_labeled(...)` or
`tier_trend_statistic(...)`, which read the tier column by name and build the
group order from `TIER_ORDER`; neither can be silently reversed by argument
order.

Unit of analysis
----------------
The plan fixes the FIRM (n = 30) as the confirmatory unit.  `aggregate_to_firm`
is the only sanctioned route from window-level rows to that unit, and
`label_permutation_test` permutes over whatever rows it is handed — pass it
firm-level rows, and pass `unit_ids` so it can refuse duplicates.

Dependencies: numpy, pandas, scipy only.  Nothing here imports transformers,
torch or tensorflow, so the abseil mutex deadlock documented in CLAUDE.md
cannot be triggered by importing this module.

Determinism: every random procedure takes an explicit `seed` and records it in
its result object.  No wall-clock or process-entropy seeding anywhere.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ── Pre-registered constants ──────────────────────────────────────────────────

#: Tier order for H1.  Values are expected to INCREASE along this sequence
#: (Small > Mid > Large).  Defined once; every call site reads it from here.
TIER_ORDER: tuple[str, ...] = ("large", "mid", "small")

#: Default seed for every random procedure in this module.  Fixed so that a
#: reported p-value is reproducible from the CSV alone.  Chosen as the date the
#: analysis plan was committed (2026-10-04); carries no other meaning.
DEFAULT_SEED: int = 20261004

try:  # single source of truth for the usability gate
    from analysis.disagreement import MIN_USABLE_CHUNKS
except Exception:  # pragma: no cover - keeps this module importable standalone
    MIN_USABLE_CHUNKS = 2

_ALTERNATIVES = ("increasing", "decreasing", "two-sided")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Jonckheere-Terpstra ordered-trend test
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class JTResult:
    """Result of a Jonckheere-Terpstra test.

    statistic        The JT statistic J = sum over ordered group pairs (i<j) of
                     the Mann-Whitney U count, ties contributing 1/2 each.
    z                (J - E[J]) / sqrt(Var[J]), tie-corrected variance.
    p_value          One- or two-sided normal-approximation p-value.
    n_ties           Number of OBSERVATIONS that share their value with at
                     least one other observation in the pooled sample (i.e. the
                     observations the tie correction acts on).
    n_tie_groups     Number of distinct values with multiplicity > 1.
    tie_group_sizes  Multiplicities of those values, descending.
    """

    statistic: float
    z: float
    p_value: float
    n_ties: int
    n_tie_groups: int
    tie_group_sizes: tuple[int, ...]
    alternative: str
    group_order: tuple[str, ...]
    group_sizes: tuple[int, ...]
    n_total: int
    expected_statistic: float
    variance: float
    degenerate: bool = False

    def __str__(self) -> str:  # pragma: no cover - human-readable logging only
        return (
            f"JT({'->'.join(self.group_order)}; n={self.group_sizes}) "
            f"J={self.statistic:.1f} E[J]={self.expected_statistic:.2f} "
            f"z={self.z:.4f} p_{self.alternative}={self.p_value:.5g} "
            f"ties={self.n_ties} in {self.n_tie_groups} group(s)"
        )


def _u_count(x: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney U count of (x_a < y_b) pairs, ties scored 1/2.

    This is the 'how often does the later group exceed the earlier group'
    count, which is what the JT statistic sums.  O(len(x)*len(y)); at n=30 that
    is 300 comparisons, so there is no reason to be clever.
    """
    diff = y[None, :] - x[:, None]
    return float(np.count_nonzero(diff > 0) + 0.5 * np.count_nonzero(diff == 0))


def jonckheere_terpstra(
    groups: Sequence[Sequence[float]],
    alternative: str = "increasing",
    group_labels: Optional[Sequence[str]] = None,
) -> JTResult:
    """Jonckheere-Terpstra test for ordered differences across k >= 2 groups.

    Parameters
    ----------
    groups
        An ORDERED sequence of samples.  Order is the hypothesised order of the
        group medians.  A mapping is rejected: dict ordering is not a
        hypothesis, and silently relying on it is how a one-sided test gets
        reported backwards.
    alternative
        "increasing"  values tend to be LARGER in LATER groups  (p = P(Z >= z))
        "decreasing"  values tend to be SMALLER in LATER groups (p = P(Z <= z))
        "two-sided"   any ordered departure                     (p = 2P(Z >= |z|))
    group_labels
        Optional names, carried into the result for provenance only.

    Returns
    -------
    JTResult

    Notes
    -----
    Statistic:   J = sum_{i<j} U_ij,  U_ij = #{x_i < x_j} + 0.5 #{x_i = x_j}
    Null mean:   E[J] = (N^2 - sum_i n_i^2) / 4
    Null variance, WITH the tie correction (d_t = multiplicities of the distinct
    values in the pooled sample):

        Var[J] = [ N(N-1)(2N+5) - sum_i n_i(n_i-1)(2n_i+5)
                                - sum_t d_t(d_t-1)(2d_t+5) ] / 72
               + [ sum_i n_i(n_i-1)(n_i-2) * sum_t d_t(d_t-1)(d_t-2) ]
                 / [ 36 N(N-1)(N-2) ]
               + [ sum_i n_i(n_i-1) * sum_t d_t(d_t-1) ] / [ 8 N(N-1) ]

    With no ties every d_t = 1, the two correction terms vanish and this reduces
    to the textbook untied variance [N(N-1)(2N+5) - sum n_i(n_i-1)(2n_i+5)]/72.
    `tests/test_trend_tests.py` checks the tied form against the exact
    permutation mean and variance of this same statistic, which is the
    definition those moments are derived from, so the correction is verified
    rather than asserted.

    No continuity correction is applied; the statistic is compared to the normal
    curve directly, as in the standard large-sample form.  For small samples use
    the exact permutation route (`label_permutation_test`).
    """
    if isinstance(groups, Mapping):
        raise TypeError(
            "jonckheere_terpstra() takes an ORDERED sequence of samples, not a "
            "mapping. Pass [groups[g] for g in TIER_ORDER] so the hypothesised "
            "order is explicit at the call site."
        )
    if alternative not in _ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_ALTERNATIVES}, got {alternative!r}")

    arrays: list[np.ndarray] = []
    for i, g in enumerate(groups):
        a = np.asarray(list(g), dtype=float).ravel()
        if a.size == 0:
            raise ValueError(f"group {i} is empty; JT needs every group non-empty")
        if not np.all(np.isfinite(a)):
            raise ValueError(
                f"group {i} contains NaN/inf. Missing values must be DROPPED "
                "upstream (pre-registered rule 8.1: never imputed), not passed here."
            )
        arrays.append(a)
    k = len(arrays)
    if k < 2:
        raise ValueError(f"JT needs at least 2 groups, got {k}")

    labels = tuple(group_labels) if group_labels is not None else tuple(
        f"g{i}" for i in range(k)
    )
    if len(labels) != k:
        raise ValueError("group_labels length must match the number of groups")

    sizes = np.array([a.size for a in arrays], dtype=float)
    n_total = float(sizes.sum())

    # ── statistic ─────────────────────────────────────────────────────────────
    statistic = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            statistic += _u_count(arrays[i], arrays[j])

    # ── null mean ─────────────────────────────────────────────────────────────
    expected = (n_total ** 2 - float(np.sum(sizes ** 2))) / 4.0

    # ── tie structure of the pooled sample ────────────────────────────────────
    pooled = np.concatenate(arrays)
    _, counts = np.unique(pooled, return_counts=True)
    d = counts.astype(float)
    tied = d[d > 1]
    n_tie_groups = int(tied.size)
    n_ties = int(tied.sum())
    tie_group_sizes = tuple(sorted((int(v) for v in tied), reverse=True))

    # ── tie-corrected null variance ───────────────────────────────────────────
    N = n_total
    s_n2 = float(np.sum(sizes * (sizes - 1.0) * (2.0 * sizes + 5.0)))
    s_d2 = float(np.sum(d * (d - 1.0) * (2.0 * d + 5.0)))
    term1 = (N * (N - 1.0) * (2.0 * N + 5.0) - s_n2 - s_d2) / 72.0

    if N >= 3:
        s_n3 = float(np.sum(sizes * (sizes - 1.0) * (sizes - 2.0)))
        s_d3 = float(np.sum(d * (d - 1.0) * (d - 2.0)))
        term2 = (s_n3 * s_d3) / (36.0 * N * (N - 1.0) * (N - 2.0))
    else:
        term2 = 0.0

    s_n1 = float(np.sum(sizes * (sizes - 1.0)))
    s_d1 = float(np.sum(d * (d - 1.0)))
    term3 = (s_n1 * s_d1) / (8.0 * N * (N - 1.0))

    variance = term1 + term2 + term3
    if -1e-9 < variance < 0.0:
        variance = 0.0  # floating-point dust around an exactly-zero variance

    degenerate = variance <= 0.0
    if degenerate:
        # Every observation tied: J equals its null mean identically and the
        # data carry no information about order.  Report no evidence rather
        # than dividing by zero.
        logger.warning(
            "jonckheere_terpstra: null variance is 0 (all %d observations tied); "
            "reporting z=0, p=1. No ordered evidence is available.", int(N),
        )
        z = 0.0
        p = 1.0
    else:
        z = (statistic - expected) / math.sqrt(variance)
        if alternative == "increasing":
            p = float(stats.norm.sf(z))
        elif alternative == "decreasing":
            p = float(stats.norm.cdf(z))
        else:
            p = float(2.0 * stats.norm.sf(abs(z)))
            p = min(p, 1.0)

    return JTResult(
        statistic=float(statistic),
        z=float(z),
        p_value=float(p),
        n_ties=n_ties,
        n_tie_groups=n_tie_groups,
        tie_group_sizes=tie_group_sizes,
        alternative=alternative,
        group_order=labels,
        group_sizes=tuple(int(s) for s in sizes),
        n_total=int(N),
        expected_statistic=float(expected),
        variance=float(variance),
        degenerate=bool(degenerate),
    )


def jonckheere_terpstra_labeled(
    values: Sequence[float],
    labels: Sequence[str],
    group_order: Sequence[str],
    alternative: str = "increasing",
) -> JTResult:
    """JT entered by (values, group labels, EXPLICIT ordered label list).

    `group_order` is required and positional-by-intent: it states, in one place,
    which direction "increasing" means.  For the plan's C1 pass
    `group_order=TIER_ORDER` -> ("large", "mid", "small") -> increasing means
    Small > Mid > Large.

    Every label present in `labels` must appear in `group_order`; an unexpected
    tier is an error, never a silently dropped row.
    """
    v = np.asarray(list(values), dtype=float).ravel()
    lab = np.asarray([str(x) for x in labels]).ravel()
    if v.size != lab.size:
        raise ValueError(f"values ({v.size}) and labels ({lab.size}) differ in length")
    order = [str(g) for g in group_order]
    if len(set(order)) != len(order):
        raise ValueError(f"group_order contains duplicates: {order}")

    unknown = sorted(set(lab.tolist()) - set(order))
    if unknown:
        raise ValueError(
            f"labels contain values not in group_order: {unknown}. "
            f"group_order={order}. Refusing to drop rows implicitly."
        )
    missing = [g for g in order if not np.any(lab == g)]
    if missing:
        raise ValueError(
            f"group_order names group(s) with no observations: {missing}. "
            "JT needs every named group non-empty."
        )
    groups = [v[lab == g] for g in order]
    return jonckheere_terpstra(groups, alternative=alternative, group_labels=order)


def tier_trend_statistic(
    firm_df: pd.DataFrame,
    measure_col: str = "firm_mean",
    tier_col: str = "tier",
    tier_order: Sequence[str] = TIER_ORDER,
) -> float:
    """The single scalar C1-perm permutes: the JT statistic J itself.

    Returning J (not a recoded rank sum, not -J) is what makes C1 and C1-perm
    the identical hypothesis: C1 refers J to the normal approximation, C1-perm
    refers the same J to its label-permutation null.  Larger J = more evidence
    for values increasing along `tier_order` = more evidence for Small > Mid >
    Large.  Under label permutation the group sizes and the pooled tie structure
    are both fixed, so E[J] and Var[J] are constants and J and z are monotone
    equivalent: permuting either gives the same p-value.
    """
    return jonckheere_terpstra_labeled(
        firm_df[measure_col].to_numpy(dtype=float),
        firm_df[tier_col].to_numpy(),
        tier_order,
        alternative="increasing",
    ).statistic


# ══════════════════════════════════════════════════════════════════════════════
# 2. Label permutation test (C1-perm)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PermutationResult:
    observed: float
    p_value: float
    n_draws_used: int
    seed: int
    alternative: str
    null_distribution: np.ndarray = field(repr=False)
    n_at_least_as_extreme: int = 0
    n_units: int = 0
    distinct_assignments: Optional[float] = None
    exhaustive_possible: bool = False

    @property
    def null_mean(self) -> float:
        return float(np.mean(self.null_distribution))

    @property
    def null_sd(self) -> float:
        return float(np.std(self.null_distribution, ddof=1))

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"perm(n_units={self.n_units}, draws={self.n_draws_used}, seed={self.seed}) "
            f"obs={self.observed:.4f} null={self.null_mean:.4f}+-{self.null_sd:.4f} "
            f"p_{self.alternative}={self.p_value:.5g}"
        )


def label_permutation_test(
    values: Sequence[float],
    labels: Sequence[str],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_draws: int = 10_000,
    seed: int = DEFAULT_SEED,
    alternative: str = "greater",
    unit_ids: Optional[Sequence] = None,
) -> PermutationResult:
    """Permute the GROUP LABELS across units and rebuild the null of `statistic`.

    The plan fixes the firm as the unit of analysis, so the rows handed to this
    function must be FIRMS -- one row per firm, not per filing and not per
    window.  Pass `unit_ids` (e.g. the ticker column) and this function will
    refuse duplicates rather than quietly permuting labels across 59 filings and
    reporting a firm-level p-value.

    Parameters
    ----------
    statistic
        Called as `statistic(values, labels)` and must return a float.  Pass
        `lambda v, l: jonckheere_terpstra_labeled(v, l, TIER_ORDER).statistic`
        (or use `permutation_test_tier_trend`) so the permuted statistic is the
        same one C1 reports.
    alternative
        "greater"   evidence is LARGE values of the statistic (the plan's
                    one-sided case, since `tier_trend_statistic` grows with
                    Small > Mid > Large)
        "less"      evidence is SMALL values
        "two-sided" |stat - null mean| as extreme or more

    p-value
        (r + 1) / (n_draws + 1), where r counts permutation draws at least as
        extreme as the observed statistic.  The +1 convention includes the
        observed labelling as one admissible arrangement, so the p-value can
        never be reported as exactly 0 -- with 10,000 draws the floor is
        1/10001 = 9.999e-05.

    Only the labels move.  The values keep their order, so a statistic that
    depends on the pooled values (its tie structure, its mean) sees exactly the
    same pooled sample in every draw.
    """
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError(f"unknown alternative {alternative!r}")
    if n_draws < 1:
        raise ValueError(f"n_draws must be >= 1, got {n_draws}")

    v = np.asarray(list(values), dtype=float).ravel()
    lab = np.asarray([str(x) for x in labels]).ravel()
    if v.size != lab.size:
        raise ValueError(f"values ({v.size}) and labels ({lab.size}) differ in length")
    if v.size < 2:
        raise ValueError("need at least 2 units to permute")
    if not np.all(np.isfinite(v)):
        raise ValueError(
            "values contain NaN/inf; drop unusable units upstream (rule 8.1) "
            "rather than permuting around them"
        )
    if unit_ids is not None:
        ids = [str(u) for u in unit_ids]
        if len(ids) != v.size:
            raise ValueError("unit_ids length must match values length")
        dupes = sorted({u for u in ids if ids.count(u) > 1})
        if dupes:
            raise ValueError(
                f"unit_ids are not unique: {dupes}. The pre-registered unit of "
                "analysis is the FIRM; aggregate with aggregate_to_firm() before "
                "permuting."
            )

    observed = float(statistic(v, lab))
    if not np.isfinite(observed):
        raise ValueError("statistic(values, labels) returned a non-finite value")

    # How many distinct label assignments exist at all?  Reported so a tiny
    # sample cannot be presented as if it had 10,000 independent arrangements.
    _, label_counts = np.unique(lab, return_counts=True)
    log_distinct = math.lgamma(v.size + 1) - sum(
        math.lgamma(int(c) + 1) for c in label_counts
    )
    distinct = math.exp(log_distinct) if log_distinct < 700 else math.inf
    exhaustive_possible = distinct <= n_draws
    if exhaustive_possible:
        logger.warning(
            "label_permutation_test: only %.0f distinct label assignments exist but "
            "%d draws were requested; draws are sampled WITH replacement from the "
            "%.0f arrangements, so they are not %d independent arrangements.",
            distinct, n_draws, distinct, n_draws,
        )

    rng = np.random.default_rng(seed)
    null = np.empty(n_draws, dtype=float)
    permuted = lab.copy()
    for i in range(n_draws):
        rng.shuffle(permuted)
        null[i] = float(statistic(v, permuted))

    if not np.all(np.isfinite(null)):
        n_bad = int(np.count_nonzero(~np.isfinite(null)))
        raise ValueError(
            f"statistic returned non-finite values on {n_bad} of {n_draws} draws; "
            "refusing to compute a p-value from a partially broken null"
        )

    if alternative == "greater":
        r = int(np.count_nonzero(null >= observed))
    elif alternative == "less":
        r = int(np.count_nonzero(null <= observed))
    else:
        centre = float(np.mean(null))
        r = int(np.count_nonzero(np.abs(null - centre) >= abs(observed - centre)))

    p = (r + 1.0) / (n_draws + 1.0)

    return PermutationResult(
        observed=observed,
        p_value=float(p),
        n_draws_used=int(n_draws),
        seed=int(seed),
        alternative=alternative,
        null_distribution=null,
        n_at_least_as_extreme=r,
        n_units=int(v.size),
        distinct_assignments=distinct,
        exhaustive_possible=bool(exhaustive_possible),
    )


def permutation_test_tier_trend(
    firm_df: pd.DataFrame,
    n_draws: int = 10_000,
    seed: int = DEFAULT_SEED,
    measure_col: str = "firm_mean",
    tier_col: str = "tier",
    ticker_col: str = "ticker",
    tier_order: Sequence[str] = TIER_ORDER,
) -> PermutationResult:
    """C1-perm exactly as pre-registered: `tier_trend_statistic`, labels permuted
    across firms, one-sided in the Small > Mid > Large direction.

    Refuses a frame with duplicate tickers, which is what a filing-level frame
    would look like.
    """
    def stat(v: np.ndarray, l: np.ndarray) -> float:
        return jonckheere_terpstra_labeled(
            v, l, tier_order, alternative="increasing"
        ).statistic

    return label_permutation_test(
        firm_df[measure_col].to_numpy(dtype=float),
        firm_df[tier_col].to_numpy(),
        stat,
        n_draws=n_draws,
        seed=seed,
        alternative="greater",
        unit_ids=firm_df[ticker_col].to_numpy() if ticker_col in firm_df else None,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. Holm-Bonferroni
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HolmResult:
    pvalues: tuple[float, ...]
    adjusted: tuple[float, ...]
    reject: tuple[bool, ...]
    alpha: float
    n_tests: int


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> HolmResult:
    """Holm-Bonferroni step-down correction.  Input order is preserved.

    Adjusted p-values are the standard monotone-enforced form: sort ascending,
    multiply the i-th smallest (0-based) by (m - i), take a running maximum so
    the sequence is non-decreasing, clip at 1, then unsort.  `reject[i]` is
    `adjusted[i] <= alpha`, which is equivalent to the step-down procedure that
    stops at the first failure.

    Applies to the §6 secondary family of three (S1, S2, S3).  It does NOT
    apply to C1/C1-perm: the confirmatory family contains one test, and
    C1-perm is the same hypothesis by another route.
    """
    p = np.asarray(list(pvalues), dtype=float).ravel()
    if p.size == 0:
        raise ValueError("holm() needs at least one p-value")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"p-values must be finite, got {p.tolist()}")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(f"p-values must lie in [0, 1], got {p.tolist()}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1), got {alpha}")

    m = p.size
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    scaled = (m - np.arange(m)) * sorted_p
    running = np.maximum.accumulate(scaled)
    adj_sorted = np.minimum(running, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adj_sorted

    return HolmResult(
        pvalues=tuple(float(x) for x in p),
        adjusted=tuple(float(x) for x in adjusted),
        reject=tuple(bool(x <= alpha) for x in adjusted),
        alpha=float(alpha),
        n_tests=int(m),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Bootstrap confidence intervals
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BootstrapResult:
    observed: float
    low: float
    high: float
    method_used: str
    method_requested: str
    n_boot: int
    seed: int
    confidence_level: float
    percentile_ci: tuple[float, float]
    bca_ci: Optional[tuple[float, float]]
    degenerate: bool
    note: str = ""


def bootstrap_ci(
    data: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 10_000,
    seed: int = DEFAULT_SEED,
    method: str = "bca",
    confidence_level: float = 0.95,
) -> BootstrapResult:
    """Bootstrap CI for `statistic(data)`.  Percentile always; BCa on request.

    BCa is delegated to `scipy.stats.bootstrap(method="BCa")` rather than
    reimplemented -- scipy already handles the acceleration via jackknife and
    the bias correction via the resample proportion below the observed value.

    Degenerate data (n < 2, or zero variance, or a statistic that is constant
    across resamples) does NOT raise.  BCa's acceleration is undefined when the
    jackknife values are all equal; in that case the interval collapses to the
    observed point value, `degenerate=True` is set, and the reason is in
    `note`.  Callers must report the collapse, not hide it.

    `statistic` is called as `statistic(1-D array)` (scipy `vectorized=False`),
    so plain `np.mean` / custom scalar functions work unchanged.
    """
    if method not in ("bca", "BCa", "percentile"):
        raise ValueError(f"method must be 'bca' or 'percentile', got {method!r}")
    requested = "bca" if method.lower() == "bca" else "percentile"
    if n_boot < 2:
        raise ValueError(f"n_boot must be >= 2, got {n_boot}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must lie in (0,1), got {confidence_level}")

    x = np.asarray(list(data), dtype=float).ravel()
    if x.size == 0:
        raise ValueError("bootstrap_ci() needs at least one observation")
    if not np.all(np.isfinite(x)):
        raise ValueError("data contains NaN/inf; drop unusable rows upstream (rule 8.1)")

    observed = float(statistic(x))

    def _point(reason: str) -> BootstrapResult:
        logger.warning("bootstrap_ci: %s - interval collapsed to the point estimate.", reason)
        return BootstrapResult(
            observed=observed, low=observed, high=observed,
            method_used="degenerate", method_requested=requested,
            n_boot=int(n_boot), seed=int(seed),
            confidence_level=float(confidence_level),
            percentile_ci=(observed, observed),
            bca_ci=(observed, observed) if requested == "bca" else None,
            degenerate=True, note=reason,
        )

    if x.size < 2:
        return _point(f"only {x.size} observation(s)")
    if float(np.ptp(x)) == 0.0:
        return _point("zero variance in the input data (all values identical)")

    rng = np.random.default_rng(seed)
    # Percentile first: it is defined whenever the resample statistics are, and
    # it is the fallback if BCa degenerates.
    with warnings.catch_warnings():
        warnings.simplefilter("error", stats.DegenerateDataWarning)
        try:
            pct = stats.bootstrap(
                (x,), statistic, n_resamples=n_boot, vectorized=False,
                method="percentile", confidence_level=confidence_level,
                random_state=rng,
            )
            percentile_ci = (float(pct.confidence_interval.low),
                             float(pct.confidence_interval.high))
        except stats.DegenerateDataWarning as exc:
            return _point(f"percentile bootstrap degenerate: {exc}")

    if not all(np.isfinite(percentile_ci)):
        return _point("percentile bootstrap returned a non-finite endpoint")

    bca_ci: Optional[tuple[float, float]] = None
    note = ""
    method_used = "percentile"
    low, high = percentile_ci

    if requested == "bca":
        with warnings.catch_warnings():
            warnings.simplefilter("error", stats.DegenerateDataWarning)
            # scipy's BCa acceleration is a 0/0 when the jackknife values are
            # all equal; it emits a RuntimeWarning and returns nan endpoints,
            # which the finiteness check below turns into a logged fallback.
            warnings.simplefilter("ignore", RuntimeWarning)
            try:
                bca = stats.bootstrap(
                    (x,), statistic, n_resamples=n_boot, vectorized=False,
                    method="BCa", confidence_level=confidence_level,
                    random_state=np.random.default_rng(seed),
                )
                cand = (float(bca.confidence_interval.low),
                        float(bca.confidence_interval.high))
                if all(np.isfinite(cand)):
                    bca_ci = cand
                    low, high = cand
                    method_used = "bca"
                else:
                    note = ("BCa returned a non-finite endpoint (acceleration "
                            "undefined); percentile interval reported instead")
                    logger.warning("bootstrap_ci: %s", note)
            except (stats.DegenerateDataWarning, ValueError) as exc:
                note = (f"BCa unavailable ({type(exc).__name__}: {exc}); "
                        "percentile interval reported instead")
                logger.warning("bootstrap_ci: %s", note)

    return BootstrapResult(
        observed=observed, low=float(low), high=float(high),
        method_used=method_used, method_requested=requested,
        n_boot=int(n_boot), seed=int(seed),
        confidence_level=float(confidence_level),
        percentile_ci=percentile_ci, bca_ci=bca_ci,
        degenerate=False, note=note,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. Spearman-Brown split-half correction
# ══════════════════════════════════════════════════════════════════════════════

def spearman_brown(r_half: float) -> float:
    """Spearman-Brown correction for a split-half reliability: 2r / (1 + r).

    `r_half` is the correlation between the two halves of a measure (here: the
    firm-level measure computed from one half of each filing's windows against
    the other half).  The return value estimates the reliability of the FULL
    measure, i.e. of both halves combined.

    ASSUMES EQUAL-LENGTH HALVES (and equal variances / parallel halves).  This
    is the k=2 special case of the general Spearman-Brown prophecy formula
    r_k = k*r / (1 + (k-1)*r).  A filing with an odd number of windows cannot
    be split into equal halves, so the split used upstream must be stated: the
    correction is only exact for the even-split case, and with unequal halves it
    is an approximation that slightly overstates reliability for the shorter
    half.  Nothing in this function can detect that -- it sees one number.
    """
    r = float(r_half)
    if not np.isfinite(r):
        raise ValueError(f"r_half must be finite, got {r_half!r}")
    if not (-1.0 <= r <= 1.0):
        raise ValueError(f"r_half must be a correlation in [-1, 1], got {r}")
    if r == -1.0:
        raise ValueError(
            "r_half = -1 makes the Spearman-Brown correction undefined "
            "(denominator 1 + r = 0)"
        )
    if r < 0.0:
        logger.warning(
            "spearman_brown: r_half=%.4f is negative; 2r/(1+r)=%.4f is outside the "
            "[0,1] reliability range and should be reported as a failed split-half, "
            "not as a reliability.", r, 2.0 * r / (1.0 + r),
        )
    return 2.0 * r / (1.0 + r)


# ══════════════════════════════════════════════════════════════════════════════
# 6. The pre-registered aggregation chain: window -> filing -> firm
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_CHUNK_COLUMNS = (
    "ticker", "tier", "filing_type", "filing_date", "section",
    "chunk_index", "n_in_section", "any_failed",
)

#: Filing identity.  A firm contributes at most one 10-K and one 10-Q, but the
#: date is part of the key so a manifest change cannot silently merge two.
FILING_KEY = ("ticker", "filing_type", "filing_date")


def _as_bool(series: pd.Series) -> np.ndarray:
    """Coerce an `any_failed` column to bool, accepting the CSV round-trip forms.

    pandas reads a bool column written by `to_csv` back as the strings "True"/
    "False"; a column with nulls comes back as object.  A null `any_failed` is
    treated as True (unknown provenance = not usable), and that is logged.
    """
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    mapped = series.map(
        lambda v: (
            True if v is True else
            False if v is False else
            None if (v is None or (isinstance(v, float) and math.isnan(v))) else
            True if str(v).strip().lower() in ("true", "1", "1.0", "t", "yes") else
            False if str(v).strip().lower() in ("false", "0", "0.0", "f", "no") else
            None
        )
    )
    n_unknown = int(mapped.isna().sum())
    if n_unknown:
        logger.warning(
            "aggregate_to_firm: %d row(s) have an unparseable/missing any_failed "
            "value; treating them as FAILED (not usable). Values seen: %s",
            n_unknown, sorted({str(v) for v in series[mapped.isna()].unique()})[:5],
        )
    return mapped.fillna(True).to_numpy(dtype=bool)


def aggregate_to_firm(
    chunk_df: pd.DataFrame,
    section: str = "mda",
    measure: str = "embedding_disagreement",
    min_usable_chunks: int = MIN_USABLE_CHUNKS,
    exclude_tickers: Sequence[str] = (),
    tier_order: Sequence[str] = TIER_ORDER,
) -> pd.DataFrame:
    """Window -> filing -> firm, exactly as §1 of the analysis plan fixes it.

    Steps, in order, none of them optional:

    1. Keep only rows of `section` (the plan never pools MD&A with Risk).
    2. DROP every window with `any_failed=True` or a null measure.  Rule 8.1:
       never imputed.  A 2-persona distance is not the 3-persona statistic.
    3. Usability gate (rule 8.2), applied to the analysed section: a filing is
       kept only if it has >= `min_usable_chunks` surviving windows AND at
       least half of its windows survive.  The denominator is the section's own
       window count `n_in_section`, so a filing whose rows are partly absent
       from the CSV still fails the gate rather than passing on the rows present.
    4. Filing mean = UNWEIGHTED mean of the measure over surviving windows.
       Unweighted because windows are equal-sized by construction; weighting by
       `word_count` would restore length as a driver.
    5. Firm mean = UNWEIGHTED mean over that firm's surviving filings.  A firm
       with one filing (HAYN) contributes its single filing's mean, which is why
       the plan also requires the HAYN-excluded sensitivity -- pass
       `exclude_tickers=["HAYN"]` for it.

    Returns
    -------
    A tidy per-firm frame, one row per firm, sorted by `tier_order` then ticker:

        ticker, tier, firm_mean, n_filings, n_windows, filings, section, measure

      n_windows : surviving windows that fed the firm mean (sum over filings)
      filings   : "|"-joined filing keys that fed it, e.g. "10-K:20250213|10-Q:20250501"

    Everything dropped is logged at WARNING and recorded in
    `result.attrs["audit"]` (a dict, also returned via `.attrs` so a caller can
    print the drop ledger).  No silent caps anywhere.
    """
    if not isinstance(chunk_df, pd.DataFrame):
        raise TypeError(f"chunk_df must be a DataFrame, got {type(chunk_df)!r}")
    missing = [c for c in REQUIRED_CHUNK_COLUMNS if c not in chunk_df.columns]
    if missing:
        raise KeyError(f"chunk_df is missing required column(s): {missing}")
    if measure not in chunk_df.columns:
        raise KeyError(f"chunk_df has no measure column {measure!r}")
    if min_usable_chunks < 1:
        raise ValueError(f"min_usable_chunks must be >= 1, got {min_usable_chunks}")

    audit: dict = {
        "section": section,
        "measure": measure,
        "min_usable_chunks": int(min_usable_chunks),
        "rows_in": int(len(chunk_df)),
        "excluded_tickers_requested": [str(t) for t in exclude_tickers],
    }

    df = chunk_df.copy()
    df["ticker"] = df["ticker"].astype(str)
    df["tier"] = df["tier"].astype(str)
    df["section"] = df["section"].astype(str)
    df["filing_type"] = df["filing_type"].astype(str)
    df["filing_date"] = df["filing_date"].astype(str)

    # Tier sanity: one tier per ticker, every tier known.
    tier_map = df.groupby("ticker")["tier"].unique()
    conflicting = {t: sorted(v.tolist()) for t, v in tier_map.items() if len(v) > 1}
    if conflicting:
        raise ValueError(f"ticker(s) carry more than one tier: {conflicting}")
    unknown_tiers = sorted(set(df["tier"].unique()) - set(str(t) for t in tier_order))
    if unknown_tiers:
        raise ValueError(
            f"unknown tier label(s) {unknown_tiers}; tier_order={list(tier_order)}"
        )

    # Requested ticker exclusions (e.g. the HAYN sensitivity).
    if len(exclude_tickers):
        excl = {str(t) for t in exclude_tickers}
        absent = sorted(excl - set(df["ticker"].unique()))
        if absent:
            logger.warning(
                "aggregate_to_firm: exclude_tickers names %s, not present in the data",
                absent,
            )
        n_before = len(df)
        df = df[~df["ticker"].isin(excl)]
        audit["rows_dropped_by_ticker_exclusion"] = int(n_before - len(df))
        logger.info(
            "aggregate_to_firm: excluded ticker(s) %s -> %d window row(s) dropped",
            sorted(excl & set(chunk_df['ticker'].astype(str).unique())),
            n_before - len(df),
        )

    # 1. section filter
    sec = df[df["section"] == section].copy()
    audit["rows_in_section"] = int(len(sec))
    audit["rows_other_sections"] = int(len(df) - len(sec))
    if sec.empty:
        raise ValueError(
            f"no rows with section=={section!r}; sections present: "
            f"{sorted(df['section'].unique())}"
        )

    # 2. drop failed / null-measure windows (rule 8.1)
    failed = _as_bool(sec["any_failed"])
    measure_null = sec[measure].isna().to_numpy()
    usable_mask = (~failed) & (~measure_null)
    audit["windows_dropped_any_failed"] = int(np.count_nonzero(failed))
    audit["windows_dropped_null_measure_not_flagged"] = int(
        np.count_nonzero(measure_null & ~failed)
    )
    audit["windows_surviving"] = int(np.count_nonzero(usable_mask))
    if audit["windows_dropped_any_failed"]:
        logger.warning(
            "aggregate_to_firm: dropped %d of %d %s window(s) with any_failed=True "
            "(pre-registered rule 8.1: dropped, never imputed)",
            audit["windows_dropped_any_failed"], len(sec), section,
        )
    if audit["windows_dropped_null_measure_not_flagged"]:
        logger.warning(
            "aggregate_to_firm: dropped %d %s window(s) whose %s is null although "
            "any_failed is False - investigate, this should not happen",
            audit["windows_dropped_null_measure_not_flagged"], section, measure,
        )
    sec["_usable"] = usable_mask

    # 3. per-filing gate (rule 8.2)
    filing_records = []
    dropped_filings = []
    for key, grp in sec.groupby(list(FILING_KEY), sort=True):
        n_rows_present = int(len(grp))
        declared = pd.to_numeric(grp["n_in_section"], errors="coerce").dropna().unique()
        if len(declared) > 1:
            raise ValueError(
                f"filing {key} section {section!r} reports inconsistent "
                f"n_in_section values {sorted(declared.tolist())}"
            )
        n_declared = int(declared[0]) if len(declared) else n_rows_present
        n_total = max(n_declared, n_rows_present)
        if n_rows_present != n_declared:
            logger.warning(
                "aggregate_to_firm: filing %s section %s has %d row(s) in the CSV "
                "but declares n_in_section=%d; using %d as the gate denominator",
                key, section, n_rows_present, n_declared, n_total,
            )
        n_surv = int(grp["_usable"].sum())
        passes = (n_surv >= min_usable_chunks) and (n_surv >= n_total / 2.0)
        rec = {
            "ticker": key[0], "filing_type": key[1], "filing_date": key[2],
            "tier": str(grp["tier"].iloc[0]),
            "n_windows_total": n_total,
            "n_windows_rows_present": n_rows_present,
            "n_windows_surviving": n_surv,
            "passes_gate": bool(passes),
        }
        if passes:
            rec["filing_mean"] = float(grp.loc[grp["_usable"], measure].astype(float).mean())
            filing_records.append(rec)
        else:
            rec["filing_mean"] = float("nan")
            dropped_filings.append(rec)
            logger.warning(
                "aggregate_to_firm: filing %s %s %s DROPPED - %d of %d %s window(s) "
                "survived (gate: >=%d and >=half)",
                key[0], key[1], key[2], n_surv, n_total, section, min_usable_chunks,
            )

    audit["filings_considered"] = len(filing_records) + len(dropped_filings)
    audit["filings_kept"] = len(filing_records)
    audit["filings_dropped_by_gate"] = [
        {k: r[k] for k in ("ticker", "filing_type", "filing_date",
                           "n_windows_surviving", "n_windows_total")}
        for r in dropped_filings
    ]
    if not filing_records:
        raise ValueError(
            f"no filing passed the usability gate for section={section!r}, "
            f"measure={measure!r}: {len(dropped_filings)} filing(s) dropped"
        )

    fil = pd.DataFrame(filing_records)

    # 4/5. filing -> firm, unweighted
    rows = []
    for ticker, grp in fil.groupby("ticker", sort=True):
        rows.append({
            "ticker": ticker,
            "tier": str(grp["tier"].iloc[0]),
            "firm_mean": float(grp["filing_mean"].mean()),
            "n_filings": int(len(grp)),
            "n_windows": int(grp["n_windows_surviving"].sum()),
            "filings": "|".join(
                f"{ft}:{fd}" for ft, fd in sorted(
                    zip(grp["filing_type"], grp["filing_date"])
                )
            ),
            "section": section,
            "measure": measure,
        })
    out = pd.DataFrame(rows)

    rank = {str(t): i for i, t in enumerate(tier_order)}
    out["_rank"] = out["tier"].map(rank)
    out = out.sort_values(["_rank", "ticker"]).drop(columns="_rank").reset_index(drop=True)

    firms_in_section = sorted(sec["ticker"].unique())
    firms_lost = sorted(set(firms_in_section) - set(out["ticker"]))
    audit["firms_in_section"] = len(firms_in_section)
    audit["firms_out"] = int(len(out))
    audit["firms_lost_entirely"] = firms_lost
    audit["firms_per_tier"] = {
        str(t): int((out["tier"] == str(t)).sum()) for t in tier_order
    }
    if firms_lost:
        logger.warning(
            "aggregate_to_firm: %d firm(s) contribute no usable filing and are "
            "absent from the output: %s", len(firms_lost), firms_lost,
        )
    logger.info(
        "aggregate_to_firm(section=%s, measure=%s): %d window rows -> %d usable "
        "windows -> %d/%d filings -> %d firms %s",
        section, measure, audit["rows_in_section"], audit["windows_surviving"],
        audit["filings_kept"], audit["filings_considered"], len(out),
        audit["firms_per_tier"],
    )

    out.attrs["audit"] = audit
    out.attrs["filing_level"] = fil
    return out
