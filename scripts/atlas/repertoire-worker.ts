import { runRepertoireTask, type RepertoireWorkerMessage, type RepertoireWorkerTask } from './repertoire-task';

/**
 * `repertoire-evaluate.ts`'s child process: `node:child_process.fork`s this
 * file once per shard, following `scripts/null/null-worker.ts`'s IPC
 * convention (`process.send`/`process.on('message')`, one task in flight
 * per child at a time). Unlike `null-worker.ts` -- whose `runTask` is
 * synchronous (`runEpisode` is a plain function call) -- this worker's task
 * (`verifyAndEvaluateSearchGraph`, via `repertoire-task.ts`'s
 * `runRepertoireTask`) is `async` (it reads the search file off disk with
 * `node:fs/promises`), so it cannot reuse
 * `scripts/null/null-worker-shared.ts`'s `runWorkerMain`, which assumes a
 * synchronous `runTask: (task) => readonly Result[]` and would silently
 * ship a pending `Promise` over IPC instead of its resolved value. This
 * file's own `process.on('message', ...)` handler awaits the task directly
 * instead. All types/logic live in `repertoire-task.ts`, which has no
 * top-level side effects -- see that file's own doc comment for why.
 */
process.on('message', (task: RepertoireWorkerTask) => {
  runRepertoireTask(task)
    .then((entry) => {
      const message: RepertoireWorkerMessage = { type: 'result', graphId: task.graphId, results: [entry] };
      process.send?.(message);
    })
    .catch((error: unknown) => {
      const message: RepertoireWorkerMessage = {
        type: 'error',
        graphId: task.graphId,
        message: error instanceof Error ? error.message : String(error)
      };
      process.send?.(message);
    });
});
