import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import {
  readRewireIndex,
  runCliMain,
  runShardedEvaluation,
  verifyRewiredFiles,
  type CliRunResult,
  type RewireIndex
} from './null-evaluate';
import type { RegimeSeedResult, RegimeWorkerMessage, RegimeWorkerTask } from './regime-worker';

/**
 * `.agents/plans/null-explanation/02-transfer-and-features.md`'s WP2
 * "Regime check" driver: runs authored episodes (opponent parked) on
 * held-out seeds `30001..30010` for biological, disconnected, and all 500
 * rewirings, and records each seed's clamp-active fraction and linear
 * steady-state distance (`scripts/null/regime-worker.ts`'s doc comment has
 * the exact metrics and the algebra behind them).
 *
 * Deliberately its own driver, not an edit to `null-evaluate.ts` (owned by
 * a concurrent bean -- see `regime-worker.ts`'s module doc comment). Reuses
 * `null-evaluate.ts`'s already-generic, already-exported
 * `runShardedEvaluation<Task, Result, Message>` (per the plan's "Make
 * `runShardedEvaluation` generic" instruction -- it was already generic on
 * `main` before this bean started, so no edit to that file was needed to
 * satisfy it) and its exported `readRewireIndex`/`verifyRewiredFiles`/
 * `runCliMain`, so this file adds no second copy of index parsing, gzip
 * sha256 verification, or the parse-args/run/log-summary CLI shape.
 *
 * Requires `scripts/analysis/transfer.py` to have already run (with the
 * same `--graphs-dir`/rewired index) and written its per-graph steady-state
 * sidecars under `--steady-state-dir`: this driver verifies every required
 * sidecar exists before forking any shard, exactly as `null-evaluate.ts`
 * verifies every rewired file's gzip sha256 up front.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');
const PUBLIC_DATA_DIR = resolve(repoRoot, 'public/data');

const DEFAULT_HELD_OUT_START = 30001;
const DEFAULT_HELD_OUT_COUNT = 10;
const DEFAULT_TICKS = 1800;
const DEFAULT_SHARDS = 8;
const DEFAULT_OUT = resolve(repoRoot, 'training/runs/null/regime.json');

export interface RegimeCheckArgs {
  readonly biological: boolean;
  readonly graph?: string;
  readonly rewiredIndex: string;
  readonly graphsDir: string;
  readonly steadyStateDir: string;
  readonly heldOutStart: number;
  readonly heldOutCount: number;
  readonly ticks: number;
  readonly shards: number;
  readonly out: string;
}

export const parseRegimeCheckArgs = (argv: readonly string[]): RegimeCheckArgs => {
  let biological = false;
  let graph: string | undefined;
  let rewiredIndex: string | undefined;
  let graphsDir: string | undefined;
  let steadyStateDir: string | undefined;
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
    } else if (flag === '--steady-state-dir') {
      steadyStateDir = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
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
  if (!steadyStateDir) throw new Error('--steady-state-dir is required');
  if (graph !== undefined && !biological) throw new Error('--graph requires --biological');
  if (!out.endsWith('.json')) throw new Error(`--out must end with ".json" (got "${out}")`);

  return {
    biological,
    graph,
    rewiredIndex,
    graphsDir,
    steadyStateDir,
    heldOutStart,
    heldOutCount,
    ticks,
    shards,
    out
  };
};

// ---------------------------------------------------------------------------
// Task list
// ---------------------------------------------------------------------------

const steadyStatePathFor = (steadyStateDir: string, graphId: string): string =>
  resolve(steadyStateDir, `${graphId}.steadystate.f64`);

/** Fail fast (before forking any shard) if a required steady-state sidecar is missing -- `transfer.py` must run first. */
const verifySteadyStateSidecars = (steadyStateDir: string, graphIds: readonly string[]): void => {
  const missing = graphIds.filter((graphId) => !existsSync(steadyStatePathFor(steadyStateDir, graphId)));
  if (missing.length > 0) {
    throw new Error(
      `regime-check: ${missing.length} steady-state sidecar(s) missing under ${steadyStateDir} ` +
        `(run scripts/analysis/transfer.py first): ${missing.slice(0, 10).join(', ')}` +
        (missing.length > 10 ? ', …' : '')
    );
  }
};

const verifyBiologicalSource = (path: string, expectedSha256: string): void => {
  const gzipBytes = readFileSync(path);
  const binary = gunzipSync(gzipBytes);
  const actual = sha256Hex(binary);
  if (actual !== expectedSha256) {
    throw new Error(
      `regime-check: ${path} decompressed sha256 ${actual} does not match index.json's sourceSha256 (${expectedSha256})`
    );
  }
};

export const buildRegimeTasks = (
  index: Readonly<RewireIndex>,
  args: Readonly<RegimeCheckArgs>,
  biologicalPath: string
): RegimeWorkerTask[] => {
  const heldOutSeeds = Array.from({ length: args.heldOutCount }, (_, i) => args.heldOutStart + i);
  const tasks: RegimeWorkerTask[] = [];

  if (args.biological) {
    const commonBio = { path: biologicalPath, expectedSha256: index.sourceSha256, heldOutSeeds, ticks: args.ticks };
    tasks.push({
      graphId: 'biological',
      mode: 'biological',
      steadyStatePath: steadyStatePathFor(args.steadyStateDir, 'biological'),
      ...commonBio
    });
    tasks.push({
      graphId: 'disconnected',
      mode: 'disconnected',
      steadyStatePath: steadyStatePathFor(args.steadyStateDir, 'disconnected'),
      ...commonBio
    });
  }

  const sortedSeeds = [...index.seeds].sort((a, b) => a.seed - b.seed);
  for (const entry of sortedSeeds) {
    const graphId = `rewired-${entry.seed}`;
    tasks.push({
      graphId,
      mode: 'rewired',
      path: resolve(args.graphsDir, entry.artifact),
      expectedSha256: entry.binarySha256,
      steadyStatePath: steadyStatePathFor(args.steadyStateDir, graphId),
      heldOutSeeds,
      ticks: args.ticks
    });
  }
  return tasks;
};

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

export interface RegimeGraphRaw {
  readonly heldOutSeeds: readonly number[];
  readonly clampFraction: readonly number[];
  readonly steadyStateDistance: readonly number[];
}

export interface RegimeRewiredGraphRaw extends RegimeGraphRaw {
  readonly seed: number;
}

export interface RegimeEvaluationRaw {
  readonly version: 1;
  readonly sourceGraphSha256: string;
  readonly rewireSourceSha256: string;
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
  readonly biological?: RegimeGraphRaw;
  readonly disconnected?: RegimeGraphRaw;
  readonly rewired: readonly RegimeRewiredGraphRaw[];
  readonly host: { readonly arch: string; readonly node: string };
}

const toGraphRaw = (results: readonly RegimeSeedResult[]): RegimeGraphRaw => ({
  heldOutSeeds: results.map((r) => r.seed),
  clampFraction: results.map((r) => r.clampFraction),
  steadyStateDistance: results.map((r) => r.steadyStateDistance)
});

export const assembleRegimeRaw = (
  index: Readonly<RewireIndex>,
  args: Readonly<RegimeCheckArgs>,
  results: ReadonlyMap<string, readonly RegimeSeedResult[]>
): RegimeEvaluationRaw => {
  const require = (graphId: string): readonly RegimeSeedResult[] => {
    const found = results.get(graphId);
    if (!found) throw new Error(`regime-check: missing results for "${graphId}"`);
    return found;
  };

  const sortedSeeds = [...index.seeds].sort((a, b) => a.seed - b.seed);
  const rewired: RegimeRewiredGraphRaw[] = sortedSeeds.map((entry) => ({
    seed: entry.seed,
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

export const runRegimeCheck = async (args: Readonly<RegimeCheckArgs>): Promise<CliRunResult> => {
  const index = readRewireIndex(args.rewiredIndex);
  verifyRewiredFiles(index, args.graphsDir);

  const biologicalPath = args.biological ? args.graph ?? resolve(PUBLIC_DATA_DIR, index.sourceArtifact) : '';
  if (args.biological) verifyBiologicalSource(biologicalPath, index.sourceSha256);

  const tasks = buildRegimeTasks(index, args, biologicalPath);
  verifySteadyStateSidecars(
    args.steadyStateDir,
    tasks.map((task) => task.graphId)
  );
  const workerPath = fileURLToPath(new URL('./regime-worker.ts', import.meta.url));

  const started = performance.now();
  const results = await runShardedEvaluation<RegimeWorkerTask, RegimeSeedResult, RegimeWorkerMessage>(
    tasks,
    args.shards,
    workerPath
  );
  const elapsedMs = performance.now() - started;

  const raw = assembleRegimeRaw(index, args, results);
  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, JSON.stringify(raw));

  const runMetaOut = `${args.out.slice(0, -'.json'.length)}.run.json`;
  atomicWriteFileSync(
    runMetaOut,
    `${JSON.stringify({ shards: args.shards, elapsedMs, perEpisodeMs: elapsedMs / (tasks.length * args.heldOutCount) }, null, 2)}\n`
  );

  return { out: args.out, runMetaOut, taskCount: tasks.length, elapsedMs };
};

const main = runCliMain('regime-check', 'graphs', parseRegimeCheckArgs, runRegimeCheck);

if (process.argv[1] === fileURLToPath(import.meta.url)) void main();
