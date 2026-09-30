// Tiny CLI shim for
// `tests_python/test_arena_config_fingerprint_cross_check.py`: runs the REAL
// `src/lib/arena/tasks.ts` `resolveArenaTask` (not a reimplementation) for
// every `ARENA_TASK_IDS` entry, and writes `{ [taskId]: { config,
// fingerprint } }` as JSON to stdout -- the task's fully RESOLVED config
// object (not just its fingerprint), so the Python side never has to
// hand-copy each non-default task's `buildTaskConfig` overrides to
// reconstruct it (a dual thermo review finding on an earlier revision of
// this test: a Python-side override dict duplicated `tasks.ts`'s own
// overrides and would silently miss a 6th `ARENA_TASK_IDS` entry). Run via
// `node_modules/.bin/tsx` -- no build step, no test framework, just this
// file's own process boundary. Same pattern as
// `tests_python/fixtures/null_stats_cross_check.ts` (this repo's existing
// "cross-check the real TS implementation from Python" convention).
//
// This exists so `tests_python/test_arena_config_fingerprint_cross_check.py`
// can assert `scripts/analysis/linkage.py`'s hand-copied
// `_DEFAULT_ARENA_CONFIG`/`_canonical_number`/`_create_arena_config_fingerprint`
// against `src/lib/arena/config.ts`'s `createArenaConfigFingerprint` /
// `src/lib/arena/tasks.ts`'s `resolveArenaTask`'s *actual* output, not
// hand-copied expected values -- `linkage.py`'s own module doc comment
// discloses the hand-copy, and nothing previously caught a drift between
// the two sides.

import { ARENA_TASK_IDS, resolveArenaTask } from '../../src/lib/arena/tasks';

interface TaskResult {
  readonly config: Record<string, number>;
  readonly fingerprint: string;
}

const main = (): void => {
  const results: Record<string, TaskResult> = {};
  for (const id of ARENA_TASK_IDS) {
    const resolved = resolveArenaTask(id);
    results[id] = { config: resolved.config, fingerprint: resolved.fingerprint };
  }
  process.stdout.write(JSON.stringify(results));
};

main();
