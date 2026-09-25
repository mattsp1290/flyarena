import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync } from '../training/fsio';
import {
  readRewireIndex,
  runCliMain,
  runShardedEvaluation,
  verifyBiologicalSource,
  verifyRewiredFiles,
  type CliRunResult,
  type RewireIndex
} from './null-evaluate';
import type { RegimeSeedResult, RegimeWorkerMessage, RegimeWorkerTask } from './regime-task';

/**
 * `.agents/plans/null-explanation/02-transfer-and-features.md`'s WP2
 * "Regime check" driver: runs authored episodes (opponent parked) on
 * held-out seeds `30001..30010` for biological, disconnected, and all 500
 * rewirings, and records each seed's clamp-active fraction and linear
 * steady-state distance (`scripts/null/regime-task.ts`'s doc comment has
 * the exact metrics and the algebra behind them).
 *
 * Deliberately its own driver, not an edit to `null-evaluate.ts` (originally
 * owned by a concurrent, in-flight bean -- see `regime-task.ts`'s module doc
 * comment). Reuses `null-evaluate.ts`'s already-generic, already-exported
 * `runShardedEvaluation<Task, Result, Message>` (per the plan's "Make
 * `runShardedEvaluation` generic" instruction -- it was already generic on
 * `main` before this bean started, so no edit to that file was needed to
 * satisfy it) and its exported `readRewireIndex`/`verifyRewiredFiles`/
 * `verifyBiologicalSource`/`runCliMain` (the last of these exported after
 * the concurrent bean merged -- a thermo-maintainability review finding;
 * this file previously carried its own hand-duplicated copy), so this file
 * adds no second copy of index parsing, gzip sha256 verification, or the
 * parse-args/run/log-summary CLI shape.
 *
 * Requires `scripts/analysis/transfer.py` to have already run (with the
 * same `--graphs-dir`/rewired index) and written its per-graph steady-state
 * sidecars, plus a `steady-state/manifest.json` tying each sidecar to the
 * exact graph bytes it was solved from, under `--steady-state-dir`: this
 * driver verifies every required sidecar against that manifest (not merely
 * that the file exists -- a bare existence check cannot catch a stale or
 * partial sidecar left over from a different `transfer.py` run, since every
 * rewiring's sidecar has the same length; a dual-review finding) before
 * forking any shard, exactly as `null-evaluate.ts` verifies every rewired
 * file's gzip sha256 up front. See `readSteadyStateManifest`/
 * `verifySteadyStateManifest` below.
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

interface SteadyStateManifestEntry {
  readonly graphBinarySha256: string;
  readonly sidecarSha256: string;
}

interface SteadyStateManifest {
  readonly version: 1;
  readonly rewireSourceSha256: string;
  readonly graphs: Readonly<Record<string, SteadyStateManifestEntry>>;
}

/**
 * Read and lightly validate `scripts/analysis/transfer.py`'s
 * `steady-state/manifest.json`. Ties every steady-state sidecar to the
 * exact graph bytes `transfer.py` solved it from -- a bare "does the file
 * exist" check (this function's earlier form) cannot detect a sidecar left
 * over from a different `transfer.py` run: every rewiring has the same
 * `neuronCount * inputChannelCount` sidecar length, so a stale one from an
 * older `index.json`/`--graphs-dir` would otherwise pass silently and
 * corrupt `steadyStateDistance` for that graph (a dual-review finding, both
 * reviewers independently). `transfer.py` deletes any manifest already
 * present at the start of its own `main()` (before writing anything) and
 * writes a fresh one only after every graph in its own run has succeeded --
 * so its mere presence rules out a run that crashed partway through *and*
 * an earlier run's stale manifest surviving a crashed rerun into the same
 * `--steady-state-dir` (a round-2 dual-review finding: without the
 * delete-at-start step, a crashed rerun could leave the *previous* run's
 * manifest sitting next to sidecars it no longer describes). The per-task
 * `steadyStateSha256` check inside `regime-task.ts`'s `loadSteadyStateMap`
 * is a second, independent layer on top of this one, not a redundant one:
 * it catches on-disk corruption of a sidecar *after* a valid manifest was
 * written, which this manifest check alone cannot.
 */
const readSteadyStateManifest = (steadyStateDir: string): SteadyStateManifest => {
  const path = resolve(steadyStateDir, 'manifest.json');
  let raw: string;
  try {
    raw = readFileSync(path, 'utf8');
  } catch (error) {
    throw new Error(
      `regime-check: cannot read ${path} (run scripts/analysis/transfer.py first): ` +
        (error instanceof Error ? error.message : String(error))
    );
  }
  const parsed = JSON.parse(raw) as Partial<SteadyStateManifest>;
  if (parsed.version !== 1 || typeof parsed.rewireSourceSha256 !== 'string' || typeof parsed.graphs !== 'object' || parsed.graphs === null) {
    throw new Error(`regime-check: ${path} is not a valid steady-state manifest`);
  }
  return parsed as SteadyStateManifest;
};

/**
 * Fail fast (before forking any shard) if the steady-state manifest is
 * missing, from a different rewire batch, or missing/mismatched for any
 * required task -- see `readSteadyStateManifest`'s doc comment for why a
 * bare existence check is not enough. Returns each task's `graphId ->
 * sidecarSha256` so `buildRegimeTasks` can pass it to the worker for a
 * second, independent verification (mirroring the graph-binary sha256
 * check's own two-layer convention elsewhere in this pipeline).
 */
export const verifySteadyStateManifest = (
  steadyStateDir: string,
  rewireSourceSha256: string,
  tasks: ReadonlyArray<{ readonly graphId: string; readonly expectedSha256: string }>
): ReadonlyMap<string, string> => {
  const manifest = readSteadyStateManifest(steadyStateDir);
  if (manifest.rewireSourceSha256 !== rewireSourceSha256) {
    throw new Error(
      `regime-check: ${steadyStateDir}/manifest.json is from a different rewire batch ` +
        `(rewireSourceSha256 ${manifest.rewireSourceSha256} vs this run's ${rewireSourceSha256}) -- ` +
        're-run scripts/analysis/transfer.py against the current --rewired-index'
    );
  }
  const problems: string[] = [];
  const sidecarShaByGraphId = new Map<string, string>();
  for (const task of tasks) {
    const entry = manifest.graphs[task.graphId];
    if (!entry) {
      problems.push(`${task.graphId}: no steady-state manifest entry (missing sidecar, or transfer.py flagged it singular)`);
      continue;
    }
    if (entry.graphBinarySha256 !== task.expectedSha256) {
      problems.push(
        `${task.graphId}: steady-state sidecar was computed from a different graph ` +
          `(manifest sha256 ${entry.graphBinarySha256} vs this run's ${task.expectedSha256})`
      );
      continue;
    }
    sidecarShaByGraphId.set(task.graphId, entry.sidecarSha256);
  }
  if (problems.length > 0) {
    throw new Error(
      `regime-check: ${problems.length} graph(s) failed steady-state manifest verification ` +
        `(run scripts/analysis/transfer.py first): ${problems.slice(0, 10).join('; ')}` +
        (problems.length > 10 ? '; …' : '')
    );
  }
  return sidecarShaByGraphId;
};

/**
 * Builds every task with `steadyStateSha256: ''` -- a placeholder, not yet
 * verified. `runRegimeCheck` passes this list straight to
 * `verifySteadyStateManifest` (which only reads `graphId`/`expectedSha256`)
 * and then maps the verified `sidecarSha256` back onto each task before any
 * shard is forked; no task with a real sidecar sha unset ever reaches
 * `runShardedEvaluation`.
 */
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
      steadyStateSha256: '',
      ...commonBio
    });
    tasks.push({
      graphId: 'disconnected',
      mode: 'disconnected',
      steadyStatePath: steadyStatePathFor(args.steadyStateDir, 'disconnected'),
      steadyStateSha256: '',
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
      steadyStateSha256: '',
      heldOutSeeds,
      ticks: args.ticks
    });
  }
  return tasks;
};

/** Attaches each task's verified `sidecarSha256` (from `verifySteadyStateManifest`) in place of the `''` placeholder `buildRegimeTasks` sets. Throws if a task's graphId is somehow missing from the map -- `verifySteadyStateManifest` already guarantees every task it was given a graphId for is present, so this is an internal-consistency check, not a user-facing one. */
const withVerifiedSteadyStateSha = (
  tasks: readonly RegimeWorkerTask[],
  sidecarShaByGraphId: ReadonlyMap<string, string>
): RegimeWorkerTask[] =>
  tasks.map((task) => {
    const steadyStateSha256 = sidecarShaByGraphId.get(task.graphId);
    if (steadyStateSha256 === undefined) {
      throw new Error(`regime-check: internal error -- no verified steady-state sha256 for "${task.graphId}"`);
    }
    return { ...task, steadyStateSha256 };
  });

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
  if (args.biological) verifyBiologicalSource('regime-check', biologicalPath, index.sourceSha256);

  const tasksPendingSteadyStateSha = buildRegimeTasks(index, args, biologicalPath);
  const sidecarShaByGraphId = verifySteadyStateManifest(
    args.steadyStateDir,
    index.rewireSourceSha256,
    tasksPendingSteadyStateSha
  );
  const tasks = withVerifiedSteadyStateSha(tasksPendingSteadyStateSha, sidecarShaByGraphId);
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
