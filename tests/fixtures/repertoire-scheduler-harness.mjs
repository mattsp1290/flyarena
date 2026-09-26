// Spawned via `node --import tsx` (see
// tests/unit/repertoire.test.ts's "real forked worker" section) so this
// process's own `process.execArgv` carries the `tsx` loader -- which
// `runWorkerScheduler`'s forked children then inherit via its
// `execArgvForChildren` filter, letting them resolve `repertoire-worker.ts`'s
// own extensionless `.ts` imports. This is exactly the loader-inheritance
// mechanism `scripts/null/null-evaluate.ts`'s own CLI relies on for its real
// shard-determinism test; a plain `vitest run` process has no such loader in
// its own `execArgv`, so a test that calls `runWorkerScheduler` in-process
// against a real `.ts` worker would fail to fork it at all. Deliberately
// minimal: no test framework, no assertions -- it just runs the scheduler
// and prints the result as JSON for the parent test process to parse and
// compare across shard counts.
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [, , evaluateModulePath, workerPath, shardsArg, tasksPath] = process.argv;
if (!evaluateModulePath || !workerPath || !shardsArg || !tasksPath) {
  throw new Error(
    'usage: repertoire-scheduler-harness.mjs <evaluateModulePath> <workerPath> <shards> <tasksPath>'
  );
}

const { runWorkerScheduler } = await import(pathToFileURL(evaluateModulePath).href);
const tasks = JSON.parse(readFileSync(tasksPath, 'utf8'));
const results = await runWorkerScheduler(tasks, Number(shardsArg), workerPath);
// Sorted by key so the parent test's byte-identical comparison is never
// sensitive to the scheduler's own (already order-independent) internal
// Map insertion order.
const sorted = [...results.entries()].sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
process.stdout.write(JSON.stringify(sorted));
