"""Shared helper for this repo's "run the REAL TypeScript implementation via
`npx tsx` (no build step) and assert Python's output matches it exactly"
test pattern, used by `test_null_stats_cross_check.py`,
`test_ts_import_graph_cross_check.py`, and
`test_arena_config_fingerprint_cross_check.py`.

A dual thermo review on an earlier revision flagged that the "find node,
invoke tsx, skip cleanly on a missing/broken JS toolchain" logic had
drifted into three near-identical copies across those files. This module
holds the one copy.

Local runs SKIP (not fail) when Node/tsx is unavailable -- this repo's
Python test suite must still run standalone without a JS toolchain. In CI
(the `CI` env var set -- GitHub Actions' own convention; see
`.github/workflows/ci.yml`'s `python` job, which now installs Node
specifically so these cross-checks run there) a missing toolchain instead
FAILS outright: CI is expected to always have Node available, so a silent
skip there would mean these cross-checks never actually run in the one
place that matters most, with nothing to flag it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TSX_BIN = REPO_ROOT / "node_modules" / ".bin" / "tsx"

#: A handful of candidate Node install locations, tried in order, beyond
#: whatever `node`/`npx` already resolve to on `PATH` -- this repo's
#: documented dev environment does not always have Node on the Python-run's
#: inherited `PATH`, so this looks a little harder before giving up.
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


def _unavailable(message: str) -> None:
    """SKIPs the test locally; FAILs it outright when `CI` is set (see this
    module's own doc comment for why the two environments differ here)."""
    if os.environ.get("CI"):
        pytest.fail(f"{message} (CI env var is set: a missing JS toolchain in CI must fail, not skip)")
    pytest.skip(message)


def run_ts_cross_check(fixture_script: Path, payload: dict | None = None, *, timeout: int = 60) -> dict:
    """Runs `fixture_script` via `node_modules/.bin/tsx`, piping `payload` as
    JSON on stdin when given (omitted entirely for a fixture that reads no
    stdin), and returns its parsed JSON stdout.

    SKIPs (locally) or FAILs (in CI, see `_unavailable`) with a clear
    message if Node/tsx cannot be found at all, or if invoking it fails for
    an environmental reason (no network for a first-time tsx fetch, a
    sandboxed `/proc`, etc.) -- outside of CI, a missing JS toolchain is an
    environment gap, not a Python-side test failure.
    """
    if not TSX_BIN.exists():
        _unavailable(f"tests_python: {TSX_BIN} not found (run `npm install` first) -- skipping TS cross-check")

    env = None
    extra_dir = _find_node_bin_dir()
    if extra_dir is not None:
        env = dict(os.environ)
        env["PATH"] = f"{extra_dir}:{env.get('PATH', '')}"
    elif shutil.which("node") is None:
        _unavailable("tests_python: node not found on PATH (and not under ~/.nvm) -- skipping TS cross-check")

    try:
        result = subprocess.run(
            [str(TSX_BIN), str(fixture_script)],
            input=json.dumps(payload) if payload is not None else None,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        _unavailable(f"tests_python: could not run tsx ({error}) -- skipping TS cross-check")

    if result.returncode != 0:
        _unavailable(
            f"tests_python: `tsx {fixture_script.name}` exited non-zero, environment likely cannot run the "
            f"TS toolchain here -- skipping TS cross-check (stderr: {result.stderr[:1000]})"
        )
    return json.loads(result.stdout)
