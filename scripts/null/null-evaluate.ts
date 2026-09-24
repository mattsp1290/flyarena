import { fork } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import type { NullSeedResult, NullTaskMode, NullWorkerMessage, NullWorkerTask } from './null-worker';

/**
 * `.agents/plans/rewiring-null/02-authored-null-evaluation.md`'s WP2 driver:
 * scores biological, disconnected, and every rewired graph in a
 * `rewire_batch.py` `index.json` on the same held-out seeds with the
 * authored decoder against a parked opponent, sharded across
 * `node:child_process.fork`ed copies of `null-worker.ts`.
 *
 * This script owns everything the plan calls out for `null-evaluate.ts`:
 * building the task list (biological, disconnected, one task per rewired
 * seed), verifying every rewired file's sha256 against `index.json` before
 * any scoring starts, dispatching tasks to shards, and writing results in
 * **canonical task order** (biological, disconnected, then rewired sorted
 * by numeric seed — never a sort over collected results or over `graphId`
 * strings, which would put `rewired-10` before `rewired-2`) so the output
 * is independent of shard count and completion timing
 * (`tests/unit/null-evaluate.test.ts` proves `--shards 1` and `--shards 3`
 * are byte-identical). Statistics (mean/CI/percentile/
 * histogram) are deliberately not computed here — they live in
 * `null-stats.ts`/`null-report.ts`, which run cheaply against this script's
 * raw-per-seed `authored.json` output without re-simulating anything.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');
const PUBLIC_DATA_DIR = resolve(repoRoot, 'public/data');

const DEFAULT_HELD_OUT_START = 30001;
const DEFAULT_HELD_OUT_COUNT = 100;
const DEFAULT_TICKS = 1800;
const DEFAULT_SHARDS = 18;
const DEFAULT_OUT = resolve(repoRoot, 'training/runs/null/authored.json');

const sha256Hex = (data: Uint8Array): string => createHash('sha256').update(data).digest('hex');

// ---------------------------------------------------------------------------
// rewire_batch.py index.json
// ---------------------------------------------------------------------------

export interface RewireIndexSeedEntry {
  readonly seed: number;
  readonly artifact: string;
  readonly binarySha256: string;
  readonly binaryBytes: number;
  readonly gzipSha256: string;
  readonly gzipBytes: number;
  readonly stats: {
    readonly acceptedSwaps: number;
    readonly attempts: number;
  };
}

export interface RewireIndex {
  readonly sourceArtifact: string;
  readonly sourceSha256: string;
  readonly rewireSourceSha256: string;
  readonly seeds: readonly RewireIndexSeedEntry[];
}

/**
 * Parse and lightly validate `rewire_batch.py`'s `index.json`. Deliberately
 * tolerant of extra/missing provenance fields beyond the ones this script
 * reads (`binfmtSourceSha256`/`numpyVersion`/`params` are recorded by
 * `rewire_batch.py` but not consumed here) — this script only depends on
 * the fields it actually uses.
 */
export const readRewireIndex = (path: string): RewireIndex => {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as Partial<RewireIndex>;
  if (typeof parsed.sourceArtifact !== 'string' || typeof parsed.sourceSha256 !== 'string') {
    throw new Error(`null-evaluate: ${path} is missing sourceArtifact/sourceSha256`);
  }
  if (typeof parsed.rewireSourceSha256 !== 'string') {
    throw new Error(`null-evaluate: ${path} is missing rewireSourceSha256`);
  }
  if (!Array.isArray(parsed.seeds) || parsed.seeds.length === 0) {
    throw new Error(`null-evaluate: ${path} has no seeds`);
  }
  const seenSeeds = new Set<number>();
  for (const entry of parsed.seeds) {
    if (
      typeof entry.seed !== 'number' ||
      !Number.isInteger(entry.seed) ||
      entry.seed < 0 ||
      typeof entry.artifact !== 'string' ||
      typeof entry.binarySha256 !== 'string' ||
      typeof entry.gzipSha256 !== 'string' ||
      typeof entry.gzipBytes !== 'number' ||
      typeof entry.stats?.acceptedSwaps !== 'number' ||
      typeof entry.stats?.attempts !== 'number'
    ) {
      throw new Error(`null-evaluate: ${path} has a malformed seed entry: ${JSON.stringify(entry)}`);
    }
    // A duplicate seed would create two tasks with the same graphId
    // (`rewired-${seed}`); which result "wins" then depends on completion
    // order, which is exactly what the shard byte-identity guarantee
    // promises can never happen — reject it here instead (a dual-review
    // finding; `rewire_batch.py` itself can't produce this, since it
    // iterates a `range`, but a hand-merged or hand-edited index.json can).
    if (seenSeeds.has(entry.seed)) {
      throw new Error(`null-evaluate: ${path} lists rewiring seed ${entry.seed} more than once`);
    }
    seenSeeds.add(entry.seed);
  }
  return parsed as RewireIndex;
};

/**
 * Verify every rewired file's raw gzip bytes against `index.json` before
 * any shard is forked, so a corrupted or stale batch (a smaller local disk
 * problem, a partially-copied directory) fails in seconds rather than
 * after however much of an 8-hour run has already completed. `null-worker.ts`
 * independently re-verifies the *decompressed* sha256 of whatever file it
 * actually loads, closer to where scoring happens.
 */
export const verifyRewiredFiles = (index: Readonly<RewireIndex>, graphsDir: string): void => {
  const mismatches: string[] = [];
  for (const entry of index.seeds) {
    const path = resolve(graphsDir, entry.artifact);
    let bytes: Buffer;
    try {
      bytes = readFileSync(path);
    } catch (error) {
      mismatches.push(`seed ${entry.seed}: cannot read ${path} (${error instanceof Error ? error.message : String(error)})`);
      continue;
    }
    if (bytes.byteLength !== entry.gzipBytes) {
      mismatches.push(`seed ${entry.seed}: ${path} is ${bytes.byteLength} bytes, index.json expects ${entry.gzipBytes}`);
      continue;
    }
    const actual = sha256Hex(bytes);
    if (actual !== entry.gzipSha256) {
      mismatches.push(`seed ${entry.seed}: ${path} gzip sha256 ${actual} does not match index.json (${entry.gzipSha256})`);
    }
  }
  if (mismatches.length > 0) {
    throw new Error(`null-evaluate: ${mismatches.length} rewired file(s) failed verification:\n${mismatches.join('\n')}`);
  }
};

/** Verify the biological source graph's decompressed sha256 against `index.json`'s own `sourceSha256`. */
const verifyBiologicalSource = (path: string, expectedSha256: string): void => {
  const gzipBytes = readFileSync(path);
  const binary = gunzipSync(gzipBytes);
  const actual = sha256Hex(binary);
  if (actual !== expectedSha256) {
    throw new Error(`null-evaluate: ${path} decompressed sha256 ${actual} does not match index.json's sourceSha256 (${expectedSha256})`);
  }
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface NullEvaluateArgs {
  readonly biological: boolean;
  readonly graph?: string;
  readonly rewiredIndex: string;
  readonly graphsDir: string;
  readonly heldOutStart: number;
  readonly heldOutCount: number;
  readonly ticks: number;
  readonly shards: number;
  readonly out: string;
}

export const parseNullEvaluateArgs = (argv: readonly string[]): NullEvaluateArgs => {
  let biological = false;
  let graph: string | undefined;
  let rewiredIndex: string | undefined;
  let graphsDir: string | undefined;
  let heldOutStart = DEFAULT_HELD_OUT_START;
  let heldOutCount = DEFAULT_HELD_OUT_COUNT;
  let ticks = DEFAULT_TICKS;
  let shards = DEFAULT_SHARDS;
  let out = DEFAULT_OUT;

  let index = 0;
  while (index < argv.length) {
    const flag = argv[index];
    if (flag === '--biological') {
      biological = true;
      index += 1;
    } else if (flag === '--graph') {
      graph = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--rewired-index') {
      rewiredIndex = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--graphs-dir') {
      graphsDir = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--held-out-start') {
      heldOutStart = requireNonNegativeInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--held-out-count') {
      heldOutCount = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--ticks') {
      ticks = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--shards') {
      shards = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--out') {
      out = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (!rewiredIndex) throw new Error('--rewired-index is required');
  if (!graphsDir) throw new Error('--graphs-dir is required');
  // `graph` is only ever read when `biological` is set (see `runNullEvaluate`)
  // — silently accepting it otherwise would let an operator believe an
  // override took effect when it didn't (a dual-review finding).
  if (graph !== undefined && !biological) throw new Error('--graph requires --biological');
  // `runNullEvaluate` derives its run-meta sidecar path from `out` by
  // replacing a trailing ".json" — enforced here so that never silently
  // degrades into overwriting `out` itself (see `runMetaPathFor`'s doc
  // comment, a dual-review finding).
  if (!out.endsWith('.json')) throw new Error(`--out must end with ".json" (got "${out}")`);

  return { biological, graph, rewiredIndex, graphsDir, heldOutStart, heldOutCount, ticks, shards, out };
};

// ---------------------------------------------------------------------------
// Task list
// ---------------------------------------------------------------------------

export const buildTasks = (
  index: Readonly<RewireIndex>,
  args: Readonly<NullEvaluateArgs>,
  biologicalPath: string
): NullWorkerTask[] => {
  const heldOutSeeds = Array.from({ length: args.heldOutCount }, (_, i) => args.heldOutStart + i);
  const tasks: NullWorkerTask[] = [];

  if (args.biological) {
    const commonBio = { path: biologicalPath, expectedSha256: index.sourceSha256, heldOutSeeds, ticks: args.ticks };
    tasks.push({ graphId: 'biological', mode: 'biological' as NullTaskMode, ...commonBio });
    tasks.push({ graphId: 'disconnected', mode: 'disconnected' as NullTaskMode, ...commonBio });
  }

  const sortedSeeds = [...index.seeds].sort((a, b) => a.seed - b.seed);
  for (const entry of sortedSeeds) {
    tasks.push({
      graphId: `rewired-${entry.seed}`,
      mode: 'rewired' as NullTaskMode,
      path: resolve(args.graphsDir, entry.artifact),
      expectedSha256: entry.binarySha256,
      heldOutSeeds,
      ticks: args.ticks
    });
  }
  return tasks;
};

// ---------------------------------------------------------------------------
// Sharded execution
// ---------------------------------------------------------------------------

/** Node's `--inspect*` flags carry a fixed debug port; forking `shardCount` children with all of them inheriting the same port would collide with `EADDRINUSE` at startup instead of doing any real work. */
const execArgvForChildren = (): string[] => process.execArgv.filter((flag) => !flag.startsWith('--inspect'));

/**
 * Fork `shardCount` copies of `null-worker.ts`, hand each one tasks one at
 * a time (a worker that finishes gets the next queued task, so a slow
 * biological/disconnected task never blocks idle shards), and collect
 * every task's raw results keyed by `graphId`. Deterministic regardless of
 * which shard executes which task or in what order: the caller reassembles
 * output by iterating `tasks` (the canonical, seed-sorted order), not by
 * collection order.
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
export const runShardedEvaluation = async (
  tasks: readonly NullWorkerTask[],
  shardCount: number,
  workerPath: string
): Promise<Map<string, readonly NullSeedResult[]>> => {
  const results = new Map<string, readonly NullSeedResult[]>();
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
      let inFlight: NullWorkerTask | undefined;

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

      child.on('message', (message: NullWorkerMessage) => {
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

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

export interface NullGraphRaw {
  readonly heldOutSeeds: readonly number[];
  readonly movementScore: readonly number[];
  readonly foodPickups: readonly number[];
  readonly hazardContacts: readonly number[];
}

export interface NullRewiredGraphRaw extends NullGraphRaw {
  readonly seed: number;
  readonly gzipSha256: string;
  readonly acceptedSwaps: number;
  readonly attempts: number;
}

/**
 * Deliberately does *not* record `--shards`: it is an operational parameter
 * of *this run* (how many worker processes happened to execute it), not a
 * property of the graphs or scores, and the whole point of
 * `runShardedEvaluation`'s design is that it must not affect the result —
 * recording it here would make `--shards 1` and `--shards 3` runs differ by
 * exactly that one byte, defeating `tests/unit/null-evaluate.test.ts`'s
 * shard-determinism check. This script instead writes it (alongside wall
 * time) to a `<out>.run.json` sidecar next to `authored.json`; `null-report.ts`
 * reads `shards`/timing from that sidecar by default, with an explicit
 * `--shards` flag available to override it (see `null-report.ts`'s
 * `resolveRunMeta`).
 */
export interface NullEvaluationRaw {
  readonly version: 1;
  readonly sourceGraphSha256: string;
  readonly rewireSourceSha256: string;
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
  readonly biological?: NullGraphRaw;
  readonly disconnected?: NullGraphRaw;
  readonly rewired: readonly NullRewiredGraphRaw[];
  readonly host: { readonly arch: string; readonly node: string };
}

const toGraphRaw = (results: readonly NullSeedResult[]): NullGraphRaw => ({
  heldOutSeeds: results.map((r) => r.seed),
  movementScore: results.map((r) => r.movementScore),
  foodPickups: results.map((r) => r.foodPickups),
  hazardContacts: results.map((r) => r.hazardContacts)
});

export const assembleRaw = (
  index: Readonly<RewireIndex>,
  args: Readonly<NullEvaluateArgs>,
  results: ReadonlyMap<string, readonly NullSeedResult[]>
): NullEvaluationRaw => {
  const require = (graphId: string): readonly NullSeedResult[] => {
    const found = results.get(graphId);
    if (!found) throw new Error(`null-evaluate: missing results for "${graphId}"`);
    return found;
  };

  const rewired: NullRewiredGraphRaw[] = [...index.seeds]
    .sort((a, b) => a.seed - b.seed)
    .map((entry) => ({
      seed: entry.seed,
      gzipSha256: entry.gzipSha256,
      acceptedSwaps: entry.stats.acceptedSwaps,
      attempts: entry.stats.attempts,
      ...toGraphRaw(require(`rewired-${entry.seed}`))
    }));

  return {
    version: 1,
    sourceGraphSha256: index.sourceSha256,
    rewireSourceSha256: index.rewireSourceSha256,
    seeds: { start: args.heldOutStart, count: args.heldOutCount },
    ticks: args.ticks,
    substeps: NEURAL_SUBSTEPS_PER_TICK,
    ...(args.biological
      ? { biological: toGraphRaw(require('biological')), disconnected: toGraphRaw(require('disconnected')) }
      : {}),
    rewired,
    host: { arch: process.arch, node: process.version }
  };
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

/**
 * `<out>.run.json` — see `null-report.ts`'s `RunMeta` doc comment for why
 * this is a separate file from `authored.json` itself.
 *
 * Throws rather than falling back to a regex `.replace` that silently
 * no-ops on a mismatch: `"foo".replace(/\.json$/, '.run.json')` returns
 * `"foo"` unchanged when `outPath` doesn't end in `.json` (a dual-review
 * finding), which would make this function return `outPath` itself —
 * so the very next `writeFileSync` below would silently overwrite the
 * multi-hour `authored.json` this function just wrote with the tiny
 * run-meta sidecar. `--out` is validated to end in `.json` at parse time
 * (`parseNullEvaluateArgs`), but this function stays self-checking for any
 * other caller (a test, a future script) that might not go through the CLI.
 */
const runMetaPathFor = (outPath: string): string => {
  if (!outPath.endsWith('.json')) {
    throw new Error(`null-evaluate: expected a ".json" output path, got "${outPath}"`);
  }
  return `${outPath.slice(0, -'.json'.length)}.run.json`;
};

export const runNullEvaluate = async (
  args: Readonly<NullEvaluateArgs>
): Promise<{ out: string; runMetaOut: string; taskCount: number; elapsedMs: number }> => {
  const index = readRewireIndex(args.rewiredIndex);
  verifyRewiredFiles(index, args.graphsDir);

  const biologicalPath = args.biological ? args.graph ?? resolve(PUBLIC_DATA_DIR, index.sourceArtifact) : '';
  if (args.biological) verifyBiologicalSource(biologicalPath, index.sourceSha256);

  const tasks = buildTasks(index, args, biologicalPath);
  const workerPath = fileURLToPath(new URL('./null-worker.ts', import.meta.url));

  const started = performance.now();
  const results = await runShardedEvaluation(tasks, args.shards, workerPath);
  const elapsedMs = performance.now() - started;
  const perEpisodeMs = elapsedMs / (tasks.length * args.heldOutCount);

  const raw = assembleRaw(index, args, results);
  mkdirSync(dirname(args.out), { recursive: true });
  writeFileSync(args.out, JSON.stringify(raw));

  // Operational metadata `authored.json` deliberately excludes (see
  // `NullEvaluationRaw`'s doc comment) — `null-report.ts` reads this
  // sidecar for its published `shards`/timing fields.
  const runMetaOut = runMetaPathFor(args.out);
  writeFileSync(runMetaOut, `${JSON.stringify({ shards: args.shards, elapsedMs, perEpisodeMs }, null, 2)}\n`);

  return { out: args.out, runMetaOut, taskCount: tasks.length, elapsedMs };
};

const main = async (): Promise<void> => {
  try {
    const args = parseNullEvaluateArgs(process.argv.slice(2));
    const { out, runMetaOut, taskCount, elapsedMs } = await runNullEvaluate(args);
    const totalEpisodes = taskCount * args.heldOutCount;
    const perEpisodeMs = elapsedMs / totalEpisodes;
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `null-evaluate: wrote ${out} and ${runMetaOut} (${taskCount} graphs x ${args.heldOutCount} seeds = ` +
        `${totalEpisodes} episodes) in ${(elapsedMs / 1000).toFixed(1)}s (${perEpisodeMs.toFixed(1)} ms/episode, ` +
        `${args.shards} shards)`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`null-evaluate failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) void main();
