import { runTask } from './regime-task';
import { runWorkerMain } from './null-worker-shared';

/**
 * `regime-check.ts`'s child process: the WP2 counterpart of
 * `null-worker.ts`, forked once per shard exactly the same way (see that
 * file's own module doc comment for why `fork` + `execArgv` inheritance,
 * not `worker_threads`). A thin entry point only -- the actual task logic
 * lives in the side-effect-free `scripts/null/regime-task.ts` (`runTask`),
 * which `tests/unit/regime-check.test.ts` imports directly instead of this
 * file, so importing the test module never registers an IPC listener on the
 * test process's own message channel (see `regime-task.ts`'s module doc
 * comment for why that split exists -- a dual-review finding).
 *
 * Wire protocol: reuses `null-worker-shared.ts`'s `runWorkerMain` directly
 * (a thermo-maintainability review finding) rather than hand-rolling the
 * same `process.on('message', ...)`/`process.send` dispatch a second time
 * -- `runWorkerMain` was generalized to be generic over the `Result`/
 * `Message` types too (previously hard-coded to `NullSeedResult`/
 * `NullWorkerMessage`, which this module's own `RegimeSeedResult`/
 * `RegimeWorkerMessage` didn't fit), with `Task`/`Result` inferred here
 * from `runTask`'s own signature and `Message`'s default matching
 * `RegimeWorkerMessage`'s shape exactly -- see that function's doc comment.
 */
runWorkerMain(runTask);
