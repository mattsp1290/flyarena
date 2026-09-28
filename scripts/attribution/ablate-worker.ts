import { runWorkerMain } from '../null/null-worker-shared';
import { runTask, type AblateSeedResult, type AblateWorkerMessage, type AblateWorkerTask } from './ablate-task';

/**
 * `node:child_process.fork`ed leaf for `ablate.ts`. Deliberately separate
 * from `ablate-task.ts` (the pure `runTask`), matching `regime-worker.ts`'s
 * own split from `regime-task.ts`: `runWorkerMain` registers a real
 * `process.on('message', ...)` listener as a side effect of being called at
 * all, so `ablate-task.ts` stays importable (e.g. by a future unit test)
 * without also wiring up a listener on the importing process's own IPC
 * channel.
 */
runWorkerMain<AblateWorkerTask, AblateSeedResult, AblateWorkerMessage>(runTask);
