"""Direct unit tests for `scripts/analysis/explain_stats.py`'s pure
statistics functions -- extracted from `test_explain.py` alongside
`explain_stats.py`'s own extraction from `explain.py` (see that module's
doc comment). Every test here constructs its own small synthetic input; none
reads the real (gitignored) `training/runs/null/` artifacts.

`quantile_index`/`null_range_summary`/`rank_statistics`'s claimed parity
with `scripts/null/null-stats.ts`'s `quantileIndex`/`nullSummary`/
`rankStatistics` is cross-checked for real (via `npx tsx`) in
`tests_python/test_null_stats_cross_check.py`, not here -- the tests below
only pin Python's own documented convention.
"""

from __future__ import annotations

import numpy as np
import pytest

import explain_stats


def test_quantile_index_matches_documented_convention():
    # n=500 -> floor(0.025*500)=12, min(499, ceil(0.975*500)-1)=487 -- the
    # low-tail-floor / high-tail-ceil-minus-one convention this module's own
    # doc comment (and `scripts/null/null-stats.ts`'s `quantileIndex`)
    # documents. Real cross-language parity is checked in
    # `test_null_stats_cross_check.py`, not here.
    assert explain_stats.quantile_index(500, 0.025) == 12
    assert explain_stats.quantile_index(500, 0.975) == 487
    # p == 0.5 uses the low-tail floor branch.
    assert explain_stats.quantile_index(500, 0.5) == 250
    # High-tail branch never exceeds n - 1.
    assert explain_stats.quantile_index(4, 0.975) == 3


def test_null_range_summary_median_and_quantiles():
    values = [float(v) for v in range(1, 501)]  # 1..500, evenly spaced
    summary = explain_stats.null_range_summary(values)
    assert summary["median"] == pytest.approx(250.5)
    assert summary["p2_5"] == pytest.approx(values[12])
    assert summary["p97_5"] == pytest.approx(values[487])


def test_rank_statistics_tie_rule():
    # 3 below, 2 equal, 5 total -> bioPercentile = (3 + 0.5*2)/5 = 0.8.
    null_values = [1.0, 2.0, 3.0, 5.0, 5.0]
    stats = explain_stats.rank_statistics(null_values, 5.0)
    assert stats["kBelow"] == 3
    assert stats["kEqual"] == 2
    assert stats["bioPercentile"] == pytest.approx(0.8)


def test_outside_range():
    assert explain_stats.outside_range(0.0, 1.0, 2.0) is True
    assert explain_stats.outside_range(3.0, 1.0, 2.0) is True
    assert explain_stats.outside_range(1.5, 1.0, 2.0) is False
    assert explain_stats.outside_range(1.0, 1.0, 2.0) is False  # boundary is inside


def test_format_pct_spells_out_0th_and_100th_exactly():
    assert explain_stats.format_pct(0.0) == "0th percentile"
    assert explain_stats.format_pct(1.0) == "100th percentile"
    assert explain_stats.format_pct(0.372) == "37.2th percentile"


# ---------------------------------------------------------------------------
# Spearman correlation
# ---------------------------------------------------------------------------


def test_spearman_rho_perfect_monotonic():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert explain_stats.spearman_rho(x, y) == pytest.approx(1.0)
    assert explain_stats.spearman_rho(x, y[::-1]) == pytest.approx(-1.0)


def test_spearman_rho_degenerate_constant_input_is_zero_not_nan():
    x = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert explain_stats.spearman_rho(x, y) == 0.0


def test_spearman_rho_ties_use_average_rank():
    # x has a tie at rank (1,2) -> average rank 1.5 each; verified against
    # a hand-computed Spearman value (equivalent to scipy.stats.spearmanr
    # with tie correction).
    x = np.array([1.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    rho = explain_stats.spearman_rho(x, y)
    assert rho == pytest.approx(0.9486832980505138, rel=1e-9)


# ---------------------------------------------------------------------------
# Deterministic RNG / bootstrap / permutation calibration
# ---------------------------------------------------------------------------


def test_metric_rng_is_deterministic_and_label_dependent():
    rng_a1 = explain_stats.metric_rng(42, "foo")
    rng_a2 = explain_stats.metric_rng(42, "foo")
    rng_b = explain_stats.metric_rng(42, "bar")
    assert rng_a1.integers(0, 1_000_000, size=5).tolist() == rng_a2.integers(0, 1_000_000, size=5).tolist()
    # A different label gives a different stream (not a hard mathematical
    # guarantee, but true with overwhelming probability for a sha256-derived
    # seed -- a collision here would indicate a bug, not bad luck).
    rng_a3 = explain_stats.metric_rng(42, "foo")
    assert rng_a3.integers(0, 1_000_000, size=5).tolist() != rng_b.integers(0, 1_000_000, size=5).tolist()


def test_bootstrap_spearman_ci_deterministic_with_fixed_seed():
    rank_x = np.arange(50.0)
    rank_y = np.arange(50.0) + np.array([0.0, 1.0] * 25)
    ci_a = explain_stats.bootstrap_spearman_ci(rank_x, rank_y, 2_000, explain_stats.metric_rng(7, "m"))
    ci_b = explain_stats.bootstrap_spearman_ci(rank_x, rank_y, 2_000, explain_stats.metric_rng(7, "m"))
    assert ci_a == ci_b


def test_joint_permutation_chance_rate_is_deterministic_and_at_least_max_of_families():
    rng = np.random.default_rng(0)
    n = 100
    scores = rng.standard_normal(n)
    # Family A: one metric strongly correlated with score.
    metric_a = scores + rng.standard_normal(n) * 0.01
    rank_a = explain_stats._rank(metric_a).reshape(-1, 1)
    # Family B: pure noise.
    metric_b = rng.standard_normal(n)
    rank_b = explain_stats._rank(metric_b).reshape(-1, 1)
    seed_indices = np.arange(n)

    joint_rate_a = explain_stats.joint_permutation_chance_rate(
        [(rank_a, seed_indices)], scores, 0.3, 300, np.random.default_rng(99)
    )
    joint_rate_both = explain_stats.joint_permutation_chance_rate(
        [(rank_a, seed_indices), (rank_b, seed_indices)], scores, 0.3, 300, np.random.default_rng(99)
    )
    # Adding a second family (evaluated under the *same* permutation draws)
    # can only add hits, never remove them -- the union rate must be at
    # least as large as any single family's rate.
    assert joint_rate_both >= joint_rate_a

    rate_repeat = explain_stats.joint_permutation_chance_rate(
        [(rank_a, seed_indices), (rank_b, seed_indices)], scores, 0.3, 300, np.random.default_rng(99)
    )
    assert joint_rate_both == rate_repeat


def test_joint_permutation_chance_rate_empty_families_returns_zero():
    assert (
        explain_stats.joint_permutation_chance_rate([], np.array([1.0, 2.0, 3.0]), 0.3, 10, np.random.default_rng(0))
        == 0.0
    )


def test_build_family_rank_matrix_raises_on_missing_value():
    graphs = {f"rewired-{i}": {"weightBalance": {"thrust": None if i == 1 else float(i)}} for i in range(3)}
    metrics = [{"name": "weightBalance:thrust"}]

    def extractor(graph_result, name):
        return graph_result["weightBalance"][name.split(":", 1)[1]]

    with pytest.raises(ValueError, match="missing value"):
        explain_stats.build_family_rank_matrix(metrics, extractor, graphs, frozenset(), 3)
