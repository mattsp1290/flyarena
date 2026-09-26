import { fork } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
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
 * `runShardedEvaluation`: that scheduler's self-checking protocol assumes
 * one task produces *one result per held-out seed*
 * (`Result extends { readonly seed: number }`, checked against the task's
 * own `heldOutSeeds` array) -- here, one task (one search file) produces
 * exactly *one* whole-graph evaluation record, not a per-seed list, so
 * reusing it would mean forcing a one-element `heldOutSeeds`/`results` pair
 * through a check that means something different. `runWorkerScheduler`
 * below mirrors the same shape instead: a self-checking `key` match on
 * every reply, `abortAll` on the first error or unexpected exit (never left
 * to keep draining the queue), every child tracked and swept on every exit
 * path, and a result set that is independent of shard count or completion
 * order, because the caller re-orders by the plan's own canonical order
 * afterward, not by arrival order.
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

const WORKER_PATH = resolve(dirname(fileURLToPath(import.meta.url)), 'repertoire-worker.ts');

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
  const started = Date.now();
  const results = await runWorkerScheduler(tasks, args.shards, WORKER_PATH);
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
