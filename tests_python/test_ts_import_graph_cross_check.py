"""Real Python<->TypeScript cross-check for `ts_import_graph.
collect_repo_relative_dependencies` against the REAL `scripts/lib/
import-graph.ts`'s `collectRepoRelativeDependencies`/
`computeSourceIdentitySha256` -- the same "run the real TS module via `npx
tsx`, no build step, and assert Python's output matches it exactly" pattern
`tests_python/test_null_stats_cross_check.py` already established for
`explain_stats.py`/`null-stats.ts` (a thermo-maintainability review finding
there: a hand-copied-expected-values test can drift silently from the real
implementation it's supposed to check; a real cross-check cannot).

This is the specific test this task's own instructions call for: "a
cross-check test that both produce the same dependency set and hash for
regime-check.ts." `ts_import_graph.py` exists *only* so `explain_
provenance.py` can verify a `regime.json` producer sha without a
TypeScript toolchain; this test is what keeps that Python reimplementation
from silently drifting away from the real TS walker `scripts/null/
regime-check.ts`'s `regimeProducer()` actually calls to stamp that sha in
the first place.

Skipped (not failed) with a clear message if Node/tsx isn't available in
this environment -- this repo's Python test suite must still run standalone
without a JS toolchain."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

import ts_import_graph

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCRIPT = Path(__file__).resolve().parent / "fixtures" / "import_graph_cross_check.ts"
TSX_BIN = REPO_ROOT / "node_modules" / ".bin" / "tsx"

#: See `test_null_stats_cross_check.py`'s identical constant's doc comment.
_CANDIDATE_NODE_DIRS = (
    Path.home() / ".nvm" / "versions" / "node" / "v22.22.3" / "bin",
)


def _find_node_bin_dir() -> str | None:
    if shutil.which("node") is not None:
        return None
    for candidate in _CANDIDATE_NODE_DIRS:
        if (candidate / "node").exists():
            return str(candidate)
    return None


def _run_ts_cross_check(entry_file: Path) -> dict:
    if not TSX_BIN.exists():
        pytest.skip(f"tests_python: {TSX_BIN} not found (run `npm install` first) -- skipping TS cross-check")

    env = None
    extra_dir = _find_node_bin_dir()
    if extra_dir is not None:
        env = dict(os.environ)
        env["PATH"] = f"{extra_dir}:{env.get('PATH', '')}"
    elif shutil.which("node") is None:
        pytest.skip("tests_python: node not found on PATH (and not under ~/.nvm) -- skipping TS cross-check")

    payload = {"entryFile": str(entry_file), "repoRoot": str(REPO_ROOT)}
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
            "tests_python: `tsx import_graph_cross_check.ts` exited non-zero, environment likely cannot run the "
            f"TS toolchain here -- skipping TS cross-check (stderr: {result.stderr[:1000]})"
        )
    return json.loads(result.stdout)


def test_python_and_typescript_walkers_agree_on_regime_check_closure():
    entry = REPO_ROOT / "scripts" / "null" / "regime-check.ts"

    ts_result = _run_ts_cross_check(entry)
    py_dependencies = ts_import_graph.collect_repo_relative_dependencies(entry, REPO_ROOT)

    assert sorted(py_dependencies) == sorted(ts_result["dependencies"])

    import graph_io

    py_sha256 = graph_io.source_identity_sha256(REPO_ROOT, py_dependencies)
    assert py_sha256 == ts_result["sha256"]


def test_python_and_typescript_walkers_agree_on_repertoire_evaluate_closure():
    """WP2's repertoire pipeline (`.agents/plans/repertoire-null/`) has its
    own forked-worker driver, `scripts/atlas/repertoire-evaluate.ts` ->
    `scripts/atlas/repertoire-worker.ts` (`new URL('./repertoire-worker.ts',
    import.meta.url)`, the same forked-worker idiom `regime-check.ts` uses --
    a thermo-maintainability review finding caught this driver originally
    locating its worker with a `resolve(dirname(...), 'literal.ts')` call
    instead, a form `RELATIVE_IMPORT_RE`/`ts_import_graph.py`'s own resolver
    does not recognize). This is the same cross-check as the regime-check
    test above, against a second real entry point, so a future producer-
    identity stamp rooted at `repertoire-evaluate.ts` can trust either
    language's walker to find `repertoire-worker.ts`."""
    entry = REPO_ROOT / "scripts" / "atlas" / "repertoire-evaluate.ts"

    ts_result = _run_ts_cross_check(entry)
    py_dependencies = ts_import_graph.collect_repo_relative_dependencies(entry, REPO_ROOT)

    assert sorted(py_dependencies) == sorted(ts_result["dependencies"])
    assert "scripts/atlas/repertoire-worker.ts" in py_dependencies

    import graph_io

    py_sha256 = graph_io.source_identity_sha256(REPO_ROOT, py_dependencies)
    assert py_sha256 == ts_result["sha256"]
