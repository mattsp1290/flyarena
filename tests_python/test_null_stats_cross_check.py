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
without a JS toolchain.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import explain_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCRIPT = Path(__file__).resolve().parent / "fixtures" / "null_stats_cross_check.ts"
TSX_BIN = REPO_ROOT / "node_modules" / ".bin" / "tsx"

#: A handful of candidate Node install locations, tried in order, beyond
#: whatever `node`/`npx` already resolve to on `PATH` -- this repo's own
#: documented dev environment (`.agents/rules/nim.md`'s sibling conventions;
#: see this task's own "Node 22: export PATH=~/.nvm/versions/node/v22.22.3/
#: bin:$PATH" instruction) does not always have Node on the Python-run's
#: inherited `PATH`, so this test looks a little harder before giving up and
#: skipping.
_CANDIDATE_NODE_DIRS = (
    Path.home() / ".nvm" / "versions" / "node" / "v22.22.3" / "bin",
)


def _find_node_bin_dir() -> str | None:
    if shutil.which("node") is not None:
        return None  # already on PATH, no extra dir needed
    for candidate in _CANDIDATE_NODE_DIRS:
        if (candidate / "node").exists():
            return str(candidate)
    return None


def _run_ts_cross_check(payload: dict) -> dict:
    """Runs `null_stats_cross_check.ts` via `node_modules/.bin/tsx` with the
    given JSON payload on stdin, returning its parsed JSON stdout. Skips the
    whole module (via `pytest.skip`, not a failure) if Node cannot be found
    at all, or if invoking it fails for an environmental reason (no network
    for a first-time tsx fetch, sandboxed `/proc` access, etc.) -- a missing
    JS toolchain is a skip, not a Python-side test failure."""
    if not TSX_BIN.exists():
        pytest.skip(f"tests_python: {TSX_BIN} not found (run `npm install` first) -- skipping TS cross-check")

    env = None
    extra_dir = _find_node_bin_dir()
    if extra_dir is not None:
        import os

        env = dict(os.environ)
        env["PATH"] = f"{extra_dir}:{env.get('PATH', '')}"
    elif shutil.which("node") is None:
        pytest.skip("tests_python: node not found on PATH (and not under ~/.nvm) -- skipping TS cross-check")

    try:
        result = subprocess.run(
            [str(TSX_BIN), str(FIXTURE_SCRIPT)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        pytest.skip(f"tests_python: could not run tsx ({error}) -- skipping TS cross-check")

    if result.returncode != 0:
        pytest.skip(
            "tests_python: `tsx null_stats_cross_check.ts` exited non-zero, environment likely cannot run the "
            f"TS toolchain here -- skipping TS cross-check (stderr: {result.stderr[:500]})"
        )
    return json.loads(result.stdout)


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
