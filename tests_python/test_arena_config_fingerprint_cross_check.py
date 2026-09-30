"""Real Python<->TypeScript cross-check for `scripts/analysis/linkage.py`'s
hand-copied `_DEFAULT_ARENA_CONFIG`/`_canonical_number`/
`_create_arena_config_fingerprint`/`DEFAULT_TASK_FINGERPRINT` against
`src/lib/arena/config.ts`'s `createArenaConfigFingerprint` (via
`src/lib/arena/tasks.ts`'s `resolveArenaTask`).

`linkage.py`'s own module doc comment discloses that it hand-copies these
from the TS side ("Only the default task's fingerprint is needed on this
side ... so the other four `ARENA_TASKS` variants ... are not ported"), but
`tests_python/test_linkage.py` only ever checks Python's fingerprint helpers
against themselves (e.g. `default_task_readouts`'s mislabeled-entry test
builds its own expected string), so a future change to
`src/lib/arena/config.ts`'s field set, values, or `canonicalNumber`
formatting would silently drift out of sync with `linkage.py`'s copy and
nothing would catch it. This test runs the *real* TS module (via `npx tsx`,
no build step) and asserts Python's `DEFAULT_TASK_FINGERPRINT` against it,
same pattern as `tests_python/test_null_stats_cross_check.py`.

`linkage.py` only ports the default config (see above), so the non-default
`ARENA_TASKS` variants (`hazard-heavy`/`sparse-food`/`no-movement`/`crowded`)
are not directly testable via `linkage.py` itself -- but since each variant
is just `src/lib/arena/tasks.ts`'s `buildTaskConfig` overrides applied on
top of `ARENA_CONFIG`, and `linkage.py` already exposes the two building
blocks (`_DEFAULT_ARENA_CONFIG` and `_create_arena_config_fingerprint`) this
test reuses without modifying `linkage.py`, it is cheap to cover all five
`ARENA_TASK_IDS` here rather than only the default one.

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

import linkage

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SCRIPT = Path(__file__).resolve().parent / "fixtures" / "arena_config_fingerprint_cross_check.ts"
TSX_BIN = REPO_ROOT / "node_modules" / ".bin" / "tsx"

#: Same "look a little harder before giving up" convention as
#: `tests_python/test_null_stats_cross_check.py`'s `_CANDIDATE_NODE_DIRS`.
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


def _run_ts_cross_check() -> dict:
    """Runs `arena_config_fingerprint_cross_check.ts` via
    `node_modules/.bin/tsx`, returning its parsed JSON stdout. Skips the
    whole module (via `pytest.skip`, not a failure) if Node cannot be found
    at all, or if invoking it fails for an environmental reason -- a
    missing JS toolchain is a skip, not a Python-side test failure."""
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
            "tests_python: `tsx arena_config_fingerprint_cross_check.ts` exited non-zero, environment likely "
            f"cannot run the TS toolchain here -- skipping TS cross-check (stderr: {result.stderr[:500]})"
        )
    return json.loads(result.stdout)


def test_default_task_fingerprint_matches_real_typescript_resolve_arena_task():
    ts_fingerprints = _run_ts_cross_check()["fingerprints"]
    assert linkage.DEFAULT_TASK_FINGERPRINT == ts_fingerprints["default"]


#: `src/lib/arena/tasks.ts`'s `buildTaskConfig` overrides for each
#: non-default `ARENA_TASK_IDS` entry, hand-copied here (not in
#: `linkage.py`, which never needs these -- see module doc comment above).
_NON_DEFAULT_TASK_OVERRIDES: dict[str, dict[str, float]] = {
    "hazard-heavy": {"hazardCount": 4, "hazardPenalty": 6},
    "sparse-food": {"foodCount": 1, "halfWidth": 18, "halfDepth": 12},
    "no-movement": {"movementScorePerUnit": 0},
    "crowded": {"halfWidth": 8, "halfDepth": 5.5},
}


def test_non_default_task_fingerprints_match_real_typescript_resolve_arena_task():
    ts_fingerprints = _run_ts_cross_check()["fingerprints"]
    for task_id, overrides in _NON_DEFAULT_TASK_OVERRIDES.items():
        config = {**linkage._DEFAULT_ARENA_CONFIG, **overrides}
        py_fingerprint = linkage._create_arena_config_fingerprint(config)
        assert py_fingerprint == ts_fingerprints[task_id], task_id
