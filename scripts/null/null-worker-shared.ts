import type { NullSeedResult, NullWorkerMessage } from './null-worker';

/**
 * Shared between `null-worker.ts` and `null-trained-worker.ts`'s
 * `node:child_process.fork`ed worker leaves: the non-finite-score guard and
 * the `process.on('message', ...)`/`process.send` IPC wrapper were
 * duplicated near-verbatim between the two files (a thermo-maintainability
 * review finding — `null-trained-worker.ts`'s own doc comment even said "See
 * null-worker.ts's identical check" instead of importing it). Only
 * type-only imports come back from `./null-worker` (`NullSeedResult`/
 * `NullWorkerMessage`), so there is no runtime circularity even though
 * `null-worker.ts` itself imports the runtime helpers below from here.
 */

/** The three score fields every worker's per-seed result carries, checked for finiteness before it is ever reported back to the parent. */
export interface NullWorkerScoreResult {
  readonly movementScore: number;
  readonly foodPickups: number;
  readonly hazardContacts: number;
}

/**
 * Refuse to report a non-finite `movementScore`/`foodPickups`/
 * `hazardContacts` — a NaN/Infinity score would silently become `null`
 * under `JSON.stringify` and then `0` wherever the published artifact's
 * statistics sum it later (`conditionStats`'s sums, the bootstrap resample
 * sums, even `Array.prototype.sort`'s comparator), shifting a graph's mean,
 * CI, and null rank with no error anywhere downstream. Fail here instead,
 * where the graph/run and seed that produced it are still known (a
 * dual-review finding, originally caught in `null-worker.ts` and later
 * replicated by hand into `null-trained-worker.ts`). `source` is the
 * calling worker's own name (`"null-worker"`/`"null-trained-worker"`), so
 * the thrown message still identifies which worker raised it.
 */
export const assertFiniteScores = (
  source: string,
  graphId: string,
  seed: number,
  result: Readonly<NullWorkerScoreResult>
): void => {
  const { movementScore, foodPickups, hazardContacts } = result;
  if (![movementScore, foodPickups, hazardContacts].every(Number.isFinite)) {
    throw new Error(
      `${source}: ${graphId} seed ${seed} produced a non-finite score ` +
        `(movementScore=${movementScore}, foodPickups=${foodPickups}, hazardContacts=${hazardContacts})`
    );
  }
};

/**
 * Wire up this worker process's `process.on('message', ...)` handler: run
 * `runTask` against the incoming task, `process.send` a `result` message on
 * success or an `error` message (never throwing back out of the handler)
 * on failure. Byte-identical to what `null-worker.ts` and
 * `null-trained-worker.ts` each hand-wrote before this extraction — generic
 * over `Task` only so each file can keep its own task shape
 * (`NullWorkerTask`/`NullTrainedWorkerTask`) without a shared task type.
 */
export const runWorkerMain = <Task extends { readonly graphId: string }>(
  runTask: (task: Task) => readonly NullSeedResult[]
): void => {
  process.on('message', (task: Task) => {
    try {
      const results = runTask(task);
      const message: NullWorkerMessage = { type: 'result', graphId: task.graphId, results };
      process.send?.(message);
    } catch (error) {
      const message: NullWorkerMessage = {
        type: 'error',
        graphId: task.graphId,
        message: error instanceof Error ? error.message : String(error)
      };
      process.send?.(message);
    }
  });
};
