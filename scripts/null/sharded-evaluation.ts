import { fork } from 'node:child_process';

/**
 * Extracted from `null-evaluate.ts` (a thermo-maintainability review finding,
 * mirroring the same extraction `rewire-index.ts`/`graph-list-index.ts`
 * already underwent when that file first crossed the 1000-line threshold —
 * this WP's own `--arena-task` addition is what pushed it there again):
 * `runShardedEvaluation` has no dependency on this file's own task-list
 * building or output assembly, so it moves here wholesale, self-contained.
 * `null-evaluate.ts` re-exports it so every existing importer
 * (`null-trained-evaluate.ts`, `null-trained-evaluate-graph-list.ts`, this
 * branch's own tests) keeps working unchanged.
 */

/** Node's `--inspect*` flags carry a fixed debug port; forking `shardCount` children with all of them inheriting the same port would collide with `EADDRINUSE` at startup instead of doing any real work. */
const execArgvForChildren = (): string[] => process.execArgv.filter((flag) => !flag.startsWith('--inspect'));

/**
 * Fork `shardCount` copies of a worker script (`null-worker.ts` for
 * `null-evaluate.ts`'s own callers; `null-trained-evaluate.ts` reuses this
 * same function with `null-trained-worker.ts` — see that file), hand each
 * one tasks one at a time (a worker that finishes gets the next queued task,
 * so a slow biological/disconnected task never blocks idle shards), and
 * collect every task's raw results keyed by `graphId`. Deterministic
 * regardless of which shard executes which task or in what order: the
 * caller reassembles output by iterating `tasks` (the canonical, seed-sorted
 * order), not by collection order.
 *
 * Generic over the task/result/message shapes so `null-trained-evaluate.ts`
 * (WP3) can reuse this exact sharding/failure-handling mechanism against its
 * own worker protocol (weights-bearing tasks, not gzip-graph-bearing ones)
 * without a second, drifting copy of it — the only structural requirements
 * are that every task carries a `graphId` and the `heldOutSeeds` it was
 * assigned, and every result carries the `seed` it was scored on, so this
 * function can still verify a worker's reply matches what it was asked to
 * do (see the self-checking protocol below). Every call site pins all three
 * type parameters explicitly (TypeScript can't infer `Result`/`Message` from
 * `tasks` alone, since neither appears in an argument position, and a
 * default for `Message` can't itself reference `Result`'s default — TS
 * checks default type-argument expressions against the *unsubstituted*
 * constraint, not other parameters' defaults). `null-evaluate.ts`'s own
 * `runNullEvaluate` pins `<NullWorkerTask, NullSeedResult, NullWorkerMessage>`;
 * `null-trained-evaluate.ts` (WP3) pins its own equivalent types.
 *
 * Failure handling (a dual-review pass caught two real gaps in an earlier
 * version): the moment *any* task reports an error, or any child exits
 * abnormally (a non-zero code, or a signal this function didn't itself ask
 * for — an OOM kill, an external `kill`, a native crash — previously
 * mistaken for a clean exit whenever `code === null`), every other worker
 * is killed immediately (`abortAll`) rather than being left to keep
 * draining the queue until `Promise.all` happens to settle. Every child is
 * tracked in `children` and swept in a `finally`, so no worker outlives
 * this function on any exit path. `errors` (not promise rejection) is the
 * single source of truth for failure, so a killed sibling's own `exit`
 * event never itself throws — only the thing that caused the abort does.
 */
export const runShardedEvaluation = async <
  Task extends { readonly graphId: string; readonly heldOutSeeds: readonly number[] },
  Result extends { readonly seed: number },
  Message extends
    | { readonly type: 'result'; readonly graphId: string; readonly results: readonly Result[] }
    | { readonly type: 'error'; readonly graphId: string; readonly message: string }
>(
  tasks: readonly Task[],
  shardCount: number,
  workerPath: string
): Promise<Map<string, readonly Result[]>> => {
  const results = new Map<string, readonly Result[]>();
  const errors: string[] = [];
  const children = new Set<ReturnType<typeof fork>>();
  let nextTaskIndex = 0;
  let aborted = false;

  const abortAll = (): void => {
    aborted = true;
    for (const child of children) child.kill();
  };

  const runWorker = (): Promise<void> =>
    new Promise((resolveWorker) => {
      const child = fork(workerPath, [], { execArgv: execArgvForChildren() });
      children.add(child);
      let settled = false;
      let inFlight: Task | undefined;

      const finish = (): void => {
        if (settled) return;
        settled = true;
        children.delete(child);
        resolveWorker();
      };

      const assignNext = (): void => {
        if (aborted || nextTaskIndex >= tasks.length) {
          child.disconnect();
          return;
        }
        const task = tasks[nextTaskIndex];
        nextTaskIndex += 1;
        inFlight = task;
        child.send(task);
      };

      child.on('message', (message: Message) => {
        const expectedTask = inFlight;
        inFlight = undefined;
        // Self-checking protocol: a worker replying about a task this
        // parent never sent it (or replying with a seed list that doesn't
        // match what it was asked to score) indicates a wire-protocol bug,
        // not a legitimate result — treat it as a failure rather than
        // silently trusting whatever came back over IPC.
        if (!expectedTask || message.graphId !== expectedTask.graphId) {
          errors.push(
            `null-evaluate: received a message for "${message.graphId}" but no matching task was in flight ` +
              `(expected "${expectedTask?.graphId ?? 'none'}")`
          );
          abortAll();
          return;
        }
        if (message.type === 'result') {
          const seedsMatch =
            message.results.length === expectedTask.heldOutSeeds.length &&
            message.results.every((result, i) => result.seed === expectedTask.heldOutSeeds[i]);
          if (!seedsMatch) {
            errors.push(`${message.graphId}: result shape does not match the task's held-out seeds`);
            abortAll();
            return;
          }
          if (!results.has(message.graphId)) results.set(message.graphId, message.results);
          assignNext();
        } else {
          errors.push(`${message.graphId}: ${message.message}`);
          abortAll();
        }
      });
      child.on('error', (error) => {
        errors.push(`worker process error: ${error.message}`);
        abortAll();
        finish();
      });
      child.on('exit', (code, signal) => {
        // `code === 0` alone isn't enough: a worker that exits cleanly
        // while a task is still `inFlight` (no `result`/`error` message
        // ever arrived for it) means that task's outcome is simply unknown
        // — worth failing loudly on, not silently treating as "this shard
        // is just done" (a dual-review finding).
        const cleanExit = (code === 0 && !inFlight) || (code === null && aborted);
        if (!cleanExit) {
          errors.push(
            `worker exited unexpectedly (code ${String(code)}, signal ${String(signal)})` +
              (inFlight ? ` while running "${inFlight.graphId}"` : '')
          );
          abortAll();
        }
        finish();
      });

      assignNext();
    });

  const workerCount = Math.max(1, Math.min(shardCount, tasks.length));
  try {
    await Promise.all(Array.from({ length: workerCount }, runWorker));
  } finally {
    // Safety net: `abortAll` already kills every tracked child as soon as a
    // failure is detected, but this catches anything still alive on any
    // other exit path (including a successful run, where every child has
    // already disconnected and is exiting on its own).
    for (const child of children) child.kill();
  }

  if (errors.length > 0) {
    throw new Error(`null-evaluate: ${errors.length} task(s) failed:\n${errors.join('\n')}`);
  }
  const missing = tasks.filter((task) => !results.has(task.graphId)).map((task) => task.graphId);
  if (missing.length > 0) {
    throw new Error(`null-evaluate: no result for ${missing.length} task(s): ${missing.join(', ')}`);
  }
  return results;
};
