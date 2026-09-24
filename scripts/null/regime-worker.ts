import { runTask } from './regime-task';
import type { RegimeWorkerMessage, RegimeWorkerTask } from './regime-task';

/**
 * `regime-check.ts`'s child process: the WP2 counterpart of
 * `null-worker.ts`, forked once per shard exactly the same way (see that
 * file's own module doc comment for why `fork` + `execArgv` inheritance,
 * not `worker_threads`). A thin `process.on('message', ...)` entry point
 * only -- the actual task logic lives in the side-effect-free
 * `scripts/null/regime-task.ts` (`runTask`), which
 * `tests/unit/regime-check.test.ts` imports directly instead of this file,
 * so importing the test module never registers an IPC listener on the test
 * process's own message channel (see `regime-task.ts`'s module doc comment
 * for why that split exists -- a dual-review finding).
 *
 * Wire protocol: byte-for-byte the same shape as `null-worker-shared.ts`'s
 * `runWorkerMain` (not reused directly -- that helper is generic only over
 * `Task`, hard-coding `NullSeedResult`/`NullWorkerMessage` as its result/
 * message types, which do not fit this module's own result shape).
 */
process.on('message', (task: RegimeWorkerTask) => {
  try {
    const results = runTask(task);
    const message: RegimeWorkerMessage = { type: 'result', graphId: task.graphId, results };
    process.send?.(message);
  } catch (error) {
    const message: RegimeWorkerMessage = {
      type: 'error',
      graphId: task.graphId,
      message: error instanceof Error ? error.message : String(error)
    };
    process.send?.(message);
  }
});
