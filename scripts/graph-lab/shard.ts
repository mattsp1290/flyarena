import { fork } from 'node:child_process';

/**
 * A small, self-contained sharded-worker runner for the graph-lab entries.
 *
 * This is deliberately its own implementation rather than an import of
 * `scripts/null/null-evaluate.ts`'s `runShardedEvaluation` (which does the
 * same thing and is more battle-tested): that module ends with a top-level
 * `if (process.argv[1] === fileURLToPath(import.meta.url)) void main();`
 * guard that invokes *its own* CLI parser
 * (`parseNullEvaluateArgs(process.argv.slice(2))`) against whatever argv
 * the importing script was run with. Once esbuild bundles an entry (e.g.
 * `entry-lesion.ts`) together with everything it imports into one output
 * file, every inlined module's `import.meta.url` collapses to the *same*
 * URL -- the bundle's own -- because there is only one real ES module left
 * at runtime. Confirmed empirically (a scratch two-file esbuild bundle
 * with the same guard shape): `process.argv[1] === fileURLToPath(import.meta.url)`
 * evaluates `true` for the *inlined* library code too, not just the real
 * entry, because both sides of that comparison now resolve to the bundled
 * output file's own path. Importing `null-evaluate.ts` into a bundled
 * graph-lab entry would therefore make that entry crash on startup calling
 * `null-evaluate`'s own `main()` against the entry's argv. Reported as a
 * plan deviation rather than editing `null-evaluate.ts` (out of this WP's
 * change surface, and shared with unrelated concurrent work) -- see this
 * bean's final report.
 *
 * Behavior mirrors `runShardedEvaluation`'s important properties: forked
 * children join the parent's process group (no `detached`, so a
 * `killpg` from `jobs.py` reaches them too), a failing task or an
 * abnormally-exited child aborts every other worker immediately rather
 * than draining the queue, and results are returned keyed by `graphId` so
 * the caller can reassemble them in its own canonical order regardless of
 * completion timing.
 */
export const runOnWorkers = async <
  Task extends { readonly graphId: string },
  Result,
  Message extends
    | { readonly type: 'result'; readonly graphId: string; readonly results: readonly Result[] }
    | { readonly type: 'error'; readonly graphId: string; readonly message: string }
>(
  tasks: readonly Task[],
  shardCount: number,
  workerPath: string,
  onTaskComplete?: (completed: number, total: number) => void
): Promise<Map<string, readonly Result[]>> => {
  const results = new Map<string, readonly Result[]>();
  const errors: string[] = [];
  const children = new Set<ReturnType<typeof fork>>();
  let nextTaskIndex = 0;
  let completedCount = 0;
  let aborted = false;

  const abortAll = (): void => {
    aborted = true;
    for (const child of children) child.kill();
  };

  const runWorker = (): Promise<void> =>
    new Promise((resolveWorker) => {
      // Plain Node runs the bundled `.mjs` output directly -- no `tsx`
      // loader, so children need no inherited loader flags either.
      const child = fork(workerPath, [], { execArgv: [] });
      children.add(child);
      let settled = false;
      let inFlight: Task | undefined;

      const finish = (): void => {
        if (settled) return;
        settled = true;
        children.delete(child);
        resolveWorker();
      };

      const dispatchNext = (): void => {
        if (aborted) {
          child.kill();
          finish();
          return;
        }
        if (nextTaskIndex >= tasks.length) {
          child.kill();
          finish();
          return;
        }
        inFlight = tasks[nextTaskIndex];
        nextTaskIndex += 1;
        child.send(inFlight);
      };

      child.on('message', (message: Message) => {
        inFlight = undefined;
        if (message.type === 'error') {
          errors.push(`${message.graphId}: ${message.message}`);
          abortAll();
          finish();
          return;
        }
        results.set(message.graphId, message.results);
        completedCount += 1;
        onTaskComplete?.(completedCount, tasks.length);
        dispatchNext();
      });

      child.on('exit', (code, signalName) => {
        if (!settled && (code !== 0 || signalName) && !aborted) {
          const label = inFlight ? inFlight.graphId : '<unknown task>';
          errors.push(
            `${label}: worker exited unexpectedly (code=${String(code)}, signal=${String(signalName)})`
          );
          abortAll();
        }
        finish();
      });

      dispatchNext();
    });

  try {
    await Promise.all(Array.from({ length: Math.max(1, Math.min(shardCount, tasks.length || 1)) }, runWorker));
  } finally {
    for (const child of children) child.kill();
  }

  if (errors.length > 0) {
    throw new Error(`graph-lab shard.ts: ${errors.join('; ')}`);
  }
  return results;
};
