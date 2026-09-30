"""Real Python<->TypeScript cross-check for `explain_stats.null_range_summary`/
`rank_statistics` (which internally use `quantile_index`) against
`scripts/null/null-stats.ts`'s `nullSummary`/`rankStatistics`.

A thermo-maintainability review finding (M2): the previous
`test_quantile_index_matches_null_stats_ts_convention` did not actually run
or compare against the TypeScript implementation -- it asserted Python's own
output against hand-copied expected values, so a future change to either
side's tie/quantile convention could drift silently. This test runs the
*real* TS module (via `npx tsx`, no build step) on a set of shared fixture
cases and asserts Python's output matches it exactly, so a change to either
implementation's convention that isn't mirrored on the other side fails
here.

Skipped (not failed) with a clear message if Node/tsx isn't available in
this environment -- this repo's Python test suite must still run standalone
without a JS toolchain (see `tests_python/ts_cross_check.py`'s module doc
comment for the CI-vs-local distinction).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import explain_stats
from ts_cross_check import run_ts_cross_check

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCRIPT = Path(__file__).resolve().parent / "fixtures" / "null_stats_cross_check.ts"


def _run_ts_cross_check(payload: dict) -> dict:
    return run_ts_cross_check(FIXTURE_SCRIPT, payload)


#: Shared fixture cases -- deliberately include odd/even n, ties, and a
#: small n (the trained-section-like n=20 case `null-stats.ts`'s own test
#: suite specifically calls out), matching what both languages' own test
#: suites already exercise independently.
NULL_SUMMARY_CASES = [
    list(range(1, 501)),  # 1..500, evenly spaced -- this study's real n
    [1.0, 2.0, 3.0, 4.0, 5.0],  # odd n, no ties
    [10.0, 20.0, 30.0, 40.0],  # even n, no ties
    [5.0, 5.0, 5.0, 5.0, 5.0],  # constant (degenerate)
    list(range(1, 21)),  # n=20, the trained-section-scale case
]

RANK_STATISTICS_CASES = [
    {"nullValues": [1.0, 2.0, 3.0, 5.0, 5.0], "bioValue": 5.0},  # ties
    {"nullValues": [10.0, 20.0, 30.0], "bioValue": 1.0},  # below every value
    {"nullValues": [10.0, 20.0, 30.0], "bioValue": 100.0},  # above every value
    {"nullValues": [5.0, 5.0, 5.0, 5.0], "bioValue": 5.0},  # all-tied
    {"nullValues": [float(v) for v in range(1, 21)], "bioValue": 7.0},  # n=20
]


def test_null_range_summary_matches_real_typescript_null_summary():
    ts_results = _run_ts_cross_check(
        {"nullSummaryCases": NULL_SUMMARY_CASES, "rankStatisticsCases": []}
    )["nullSummaryResults"]

    for values, ts_summary in zip(NULL_SUMMARY_CASES, ts_results):
        py_summary = explain_stats.null_range_summary(values)
        assert py_summary["median"] == pytest.approx(ts_summary["median"]), values
        assert py_summary["p2_5"] == pytest.approx(ts_summary["p2_5"]), values
        assert py_summary["p97_5"] == pytest.approx(ts_summary["p97_5"]), values


def test_rank_statistics_matches_real_typescript_rank_statistics():
    ts_results = _run_ts_cross_check(
        {"nullSummaryCases": [], "rankStatisticsCases": RANK_STATISTICS_CASES}
    )["rankStatisticsResults"]

    for case, ts_stats in zip(RANK_STATISTICS_CASES, ts_results):
        py_stats = explain_stats.rank_statistics(case["nullValues"], case["bioValue"])
        assert py_stats["kBelow"] == ts_stats["kBelow"], case
        assert py_stats["kEqual"] == ts_stats["kEqual"], case
        assert py_stats["bioPercentile"] == pytest.approx(ts_stats["bioPercentile"]), case
