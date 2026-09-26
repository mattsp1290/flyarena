import { fork } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync } from '../training/fsio';
import { buildRepertoirePlan, loadPlanInputs, planEntryKey, type RepertoirePlanEntry } from './repertoire-plan';
import type { RepertoireEvaluatedEntry, RepertoireWorkerMessage, RepertoireWorkerTask } from './repertoire-task';

/**
 * WP2's sharded TS re-evaluation driver
 * (`.agents/plans/repertoire-null/02-runs-and-reevaluation.md`): re-verify
 * and re-evaluate every one of the planned 46 `(graph, seed)` search files
 * (`repertoire-plan.ts`'s `buildRepertoirePlan`, the same enumeration
 * `repertoire-search.sh` used to produce them) on `node:child_process.fork`ed
 * copies of `repertoire-worker.ts`, following `scripts/null/null-evaluate.ts`'s
 * fork-with-inherited-`execArgv` pattern (filtering `--inspect*` flags so
 * `shardCount` forked children never collide on one fixed debug port).
 *
 * Deliberately does **not** call that file's own exported
 * `runShardedEvaluation`: a thermo-maintainability review correctly pointed
 * out that this file's *earlier* doc comment misdiagnosed the reason --
 * "one result per held-out seed" is not actually an obstacle (a one-element
 * `heldOutSeeds: [task.searchSeed]` / matching `results` pair would satisfy
 * that scheduler's own check exactly). The real, and only, blocker is that
 * `runShardedEvaluation`'s generic constraints hardcode the wire-matching/
 * reporting field name to `graphId`
 * (`Task extends { readonly graphId: string; ... }`,
 * `Message extends { readonly graphId: string; ... }`), and in this
 * pipeline `graphId` is **not** unique per task: `biological` alone has 5
 * tasks (one per search seed 1729-1733) that all share `graphId:
 * 'biological'` (`repertoire-task.ts`'s own doc comment) -- which is exactly
 * why `repertoire-plan.ts` needed a separate composite `key` field
 * (`planEntryKey`, e.g. `"rewired-3@1730"`) in the first place. Generalizing
 * `runShardedEvaluation`/`runWorkerMain` so the wire-matching field is
 * parameterized (rather than hardcoded to `graphId`) -- the same move
 * `null-worker-shared.ts`'s `runWorkerMain` already made once, from a
 * `NullSeedResult`-only signature to `<Task, Result, Message>`, so
 * `regime-worker.ts` could reuse it -- is a real option and is deliberately
 * **deferred**, not rejected: `scripts/null/null-evaluate.ts` is being
 * edited concurrently by another workstream (which also shifts that file's
 * own producer-identity stamps), so this fix round only corrects this doc
 * comment's stated reason and leaves the duplication as a tracked follow-up
 * rather than touching that file here. `runWorkerScheduler` below mirrors
 * the same fork/IPC shape instead, for now: a self-checking `key` match on
 * every reply (`key`, not `graphId`, precisely because `graphId` can't
 * serve that role here), `abortAll` on the first error or unexpected exit
 * (never left to keep draining the queue), every child tracked and swept on
 * every exit path, and a result set that is independent of shard count or
 * completion order, because the caller re-orders by the plan's own
 * canonical order afterward, not by arrival order.
 */
const execArgvForChildren = (): string[] => process.execArgv.filter((flag) => !flag.startsWith('--inspect'));

export const runWorkerScheduler = async (
  tasks: readonly RepertoireWorkerTask[],
  shardCount: number,
  workerPath: string
): Promise<Map<string, RepertoireEvaluatedEntry>> => {
  const results = new Map<string, RepertoireEvaluatedEntry>();
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
      let inFlight: RepertoireWorkerTask | undefined;

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

      child.on('message', (message: RepertoireWorkerMessage) => {
        const expectedTask = inFlight;
        inFlight = undefined;
        if (!expectedTask || message.key !== expectedTask.key) {
          errors.push(
            `repertoire-evaluate: received a message for "${message.key}" but no matching task was in flight ` +
              `(expected "${expectedTask?.key ?? 'none'}")`
          );
          abortAll();
          return;
        }
        if (message.type === 'result') {
          if (message.results.length !== 1) {
            errors.push(`${message.key}: expected exactly one result, got ${message.results.length}`);
            abortAll();
            return;
          }
          if (!results.has(message.key)) results.set(message.key, message.results[0]);
          assignNext();
        } else {
          errors.push(`${message.key}: ${message.message}`);
          abortAll();
        }
      });
      child.on('error', (error) => {
        errors.push(`worker process error: ${error.message}`);
        abortAll();
        finish();
      });
      child.on('exit', (code, signal) => {
        const cleanExit = (code === 0 && !inFlight) || (code === null && aborted);
        if (!cleanExit) {
          errors.push(
            `worker exited unexpectedly (code ${String(code)}, signal ${String(signal)})` +
              (inFlight ? ` while running "${inFlight.key}"` : '')
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
    for (const child of children) child.kill();
  }

  if (errors.length > 0) throw new Error(`repertoire-evaluate: ${errors.length} task(s) failed:\n${errors.join('\n')}`);
  if (results.size !== tasks.length) {
    throw new Error(`repertoire-evaluate: expected ${tasks.length} results, got ${results.size}`);
  }
  return results;
};

/** One task per planned search file -- `repertoire-worker.ts`'s wire-protocol input. */
export const buildRepertoireTasks = (plan: readonly RepertoirePlanEntry[]): readonly RepertoireWorkerTask[] =>
  plan.map((entry) => ({
    key: planEntryKey(entry),
    graphId: entry.graphId,
    searchPath: entry.searchOutputPath,
    expected: entry.expected,
    // `seed` spread last: `entry.expectedSearchOptions` only ever carries
    // population/generations/ticks (`loadPlanInputs` destructures just
    // those three), but ordering it this way makes a stray `seed` field
    // inside it structurally unable to silently override the plan's own
    // `searchSeed`, rather than merely happening not to today.
    expectedOptions: { ...entry.expectedSearchOptions, seed: entry.searchSeed },
    arm: entry.arm,
    rewiringSeed: entry.rewiringSeed,
    searchSeed: entry.searchSeed
  }));

export interface RepertoireEvaluatedArtifact {
  readonly schemaVersion: 1;
  readonly graphs: readonly RepertoireEvaluatedEntry[];
}

/**
 * Assemble the final artifact in the plan's own canonical `(graph, seed)`
 * order (`repertoire-plan.ts`'s `buildRepertoirePlan` doc comment) --
 * independent of shard count or completion order, since `results` is keyed
 * by `planEntryKey(entry)` and looked up per plan entry, never iterated in
 * arrival order. Uses the same `planEntryKey` helper `buildRepertoireTasks`
 * does, rather than rebuilding the composite string a second, independently
 * maintained way (a dual-review finding).
 */
export const assembleEvaluatedArtifact = (
  plan: readonly RepertoirePlanEntry[],
  results: ReadonlyMap<string, RepertoireEvaluatedEntry>
): RepertoireEvaluatedArtifact => ({
  schemaVersion: 1,
  graphs: plan.map((entry) => {
    const key = planEntryKey(entry);
    const entryResult = results.get(key);
    if (!entryResult) throw new Error(`repertoire-evaluate: missing result for ${key}`);
    return entryResult;
  })
});

interface CliArgs {
  readonly searches: string;
  readonly graphsIndex: string;
  readonly data: string;
  readonly shards: number;
  readonly out: string;
}

const DEFAULT_SHARDS = 18;
const DEFAULT_ARMS_DIR = 'training/runs/repertoire/arms';

const parseCliArgs = (argv: readonly string[]): CliArgs => {
  let searches = 'training/runs/repertoire/search';
  let graphsIndex = 'training/runs/repertoire/graphs/index.json';
  let data = 'public/data';
  let shards = DEFAULT_SHARDS;
  let out = 'training/runs/repertoire/evaluated.json';
  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    if (flag === '--searches') {
      searches = requireValue(flag, argv[index + 1]);
      index += 1;
    } else if (flag === '--graphs-index') {
      graphsIndex = requireValue(flag, argv[index + 1]);
      index += 1;
    } else if (flag === '--data') {
      data = requireValue(flag, argv[index + 1]);
      index += 1;
    } else if (flag === '--shards') {
      shards = requirePositiveInt(flag, argv[index + 1]);
      index += 1;
    } else if (flag === '--out') {
      out = requireValue(flag, argv[index + 1]);
      index += 1;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }
  return { searches, graphsIndex, data, shards, out };
};

export const runCliMain = async (args: Readonly<CliArgs>): Promise<void> => {
  const inputs = await loadPlanInputs(args.data, args.graphsIndex, DEFAULT_ARMS_DIR, args.searches);
  const plan = buildRepertoirePlan(inputs);

  const missing = plan.filter((entry) => !existsSync(entry.searchOutputPath));
  if (missing.length > 0) {
    throw new Error(
      `repertoire-evaluate: ${missing.length} planned search file(s) are missing:\n` +
        missing.map((entry) => `  ${planEntryKey(entry)}: ${entry.searchOutputPath}`).join('\n')
    );
  }

  const tasks = buildRepertoireTasks(plan);
  // Computed lazily here (not as a module-top-level constant) so importing
  // this module alone -- e.g. `tests/unit/repertoire.test.ts` pulling in
  // `runWorkerScheduler`/`buildRepertoireTasks` -- never evaluates
  // `import.meta.url` outside of an actual CLI run, matching
  // `null-evaluate.ts`'s own `runEvaluationMode`, which computes its
  // identically-built `workerPath` inside the function that uses it rather
  // than at module scope.
  //
  // `new URL('./repertoire-worker.ts', import.meta.url)`, not
  // `resolve(dirname(...), 'repertoire-worker.ts')`: `scripts/lib/import-graph.ts`'s
  // `RELATIVE_IMPORT_RE` walker recognizes exactly three ways a producer
  // names a relative dependency (`from '...'`, dynamic `import('...')`, and
  // `new URL('...', import.meta.url)` -- the last one specifically for a
  // forked-worker script, spawned by path/URL rather than a static `import`).
  // A `resolve(dirname(...), 'literal.ts')` call matches none of the three,
  // so a future producer-identity walk rooted here would silently miss
  // `repertoire-worker.ts` even though its bytes genuinely execute as part of
  // this driver's output -- matches `null-evaluate.ts`'s own
  // `fileURLToPath(new URL('./null-worker.ts', import.meta.url))` idiom.
  const workerPath = fileURLToPath(new URL('./repertoire-worker.ts', import.meta.url));
  const started = Date.now();
  const results = await runWorkerScheduler(tasks, args.shards, workerPath);
  const elapsedMs = Date.now() - started;
  const artifact = assembleEvaluatedArtifact(plan, results);

  atomicWriteFileSync(args.out, JSON.stringify(artifact) + '\n');
  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
  console.log(
    `repertoire-evaluate: wrote ${artifact.graphs.length} graph(s) to ${args.out} ` +
      `(${args.shards} shard(s), ${(elapsedMs / 1000).toFixed(1)}s)`
  );
};

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  runCliMain(parseCliArgs(process.argv.slice(2))).catch((error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(message);
    process.exit(1);
  });
}
