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
no build step) and asserts Python's output against it, same pattern as
`tests_python/test_null_stats_cross_check.py` (shared "run the real TS,
skip cleanly without a JS toolchain" logic lives in
`tests_python/ts_cross_check.py`).

The fixture (`tests_python/fixtures/arena_config_fingerprint_cross_check.ts`)
emits every `ARENA_TASK_IDS` entry's fully RESOLVED config alongside its
fingerprint -- not just the fingerprint -- so this test never has to
hand-copy `tasks.ts`'s `buildTaskConfig` overrides for the four non-default
tasks to reconstruct their configs itself (a dual thermo review finding on
an earlier revision: a Python-side override dict duplicated those
overrides a second time and would silently miss a 6th `ARENA_TASK_IDS`
entry if one were ever added).

One consequence of that design, worth stating explicitly: for the four
NON-DEFAULT tasks, this test only cross-checks the FINGERPRINT ALGORITHM
(`_create_arena_config_fingerprint`/`_canonical_number` vs
`createArenaConfigFingerprint`/`canonicalNumber`) -- both the `config` and
the `fingerprint` in the TS fixture's output come from the same live
`resolveArenaTask` call, so re-fingerprinting that same config in Python
and comparing to that same TS-computed fingerprint cannot, by construction,
detect a change to a non-default task's OVERRIDE VALUES in `tasks.ts` (e.g.
`crowded`'s `halfWidth`) -- changing an override changes both sides of the
comparison together. Only the DEFAULT task has an independent Python-side
value to compare against (`linkage._DEFAULT_ARENA_CONFIG`,
`DEFAULT_TASK_FINGERPRINT`) -- which is also `linkage.py`'s own claimed
scope (see its module doc comment) -- so that is exactly where this test's
value-level (not just formatting-level) assertion lives.

Skipped (not failed) with a clear message if Node/tsx isn't available in
this environment -- this repo's Python test suite must still run standalone
without a JS toolchain.
"""

from __future__ import annotations

from pathlib import Path

import linkage
from ts_cross_check import run_ts_cross_check

FIXTURE_SCRIPT = Path(__file__).resolve().parent / "fixtures" / "arena_config_fingerprint_cross_check.ts"


def test_task_fingerprints_and_default_config_match_real_typescript():
    ts_tasks = run_ts_cross_check(FIXTURE_SCRIPT)
    assert ts_tasks, "expected at least one ARENA_TASK_IDS entry from the TS fixture"

    # Every task (including default): Python's fingerprint algorithm,
    # applied to the config TS itself resolved, must match TS's own
    # fingerprint for that exact config.
    for task_id, entry in ts_tasks.items():
        py_fingerprint = linkage._create_arena_config_fingerprint(entry["config"])
        assert py_fingerprint == entry["fingerprint"], task_id

    # Default only: independently verify the hand-copied VALUES themselves
    # (not just the fingerprint algorithm) against the real TS config --
    # see module doc comment for why this can only be done for "default".
    ts_default = ts_tasks["default"]
    assert linkage.DEFAULT_TASK_FINGERPRINT == ts_default["fingerprint"]
    assert linkage._DEFAULT_ARENA_CONFIG == ts_default["config"]
    # Key order feeds `_create_arena_config_fingerprint`'s field ordering
    # directly (it walks `config.items()` in insertion order), so value
    # equality above isn't sufficient on its own -- order must match too.
    assert list(linkage._DEFAULT_ARENA_CONFIG.keys()) == list(ts_default["config"].keys())
