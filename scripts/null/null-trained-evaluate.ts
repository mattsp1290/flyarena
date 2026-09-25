import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync } from '../training/fsio';
import { CEM_CONFIG_FIELDS, isEmptyCemConfig, readRunDir } from '../training/run-dir';
import { runCliMain, runMetaPathFor, runShardedEvaluation, toGraphRaw, type NullGraphRaw } from './null-evaluate';
import type { NullSeedResult, NullWorkerMessage } from './null-worker';
import type { NullTrainedWorkerTask } from './null-trained-worker';

/**
 * The `flyarena-bigq` merge commit this study's re-grounding
 * (`.agents/plans/rewiring-null/03-trained-sample.md`'s "Re-grounding"
 * section) read `public/data/trained-readout-v1.manifest.json` and the
 * biological run directories from -- recorded in every `trained.json`
 * output for provenance, alongside the CEM config reconciled from the
 * actual run directories below (not hard-coded, since the config itself is
 * copied from that manifest, not from this constant).
 */
export const BIGQ_MERGE_COMMIT = '69b610d4a9da11b12a7ac180997e702cf9fd2a4f';

/**
 * `.agents/plans/rewiring-null/03-trained-sample.md`'s WP3 driver: rescores
 * every `scripts/null/train-sample.sh` run directory (20 rewired seeds) plus
 * the three `flyarena-bigq` biological replicas (trainer seeds 101/202/303)
 * in TS, using `runEpisode`'s `trained` decoder against a parked opponent,
 * on the same held-out seeds the authored null and the trained-readout
 * report both use (30001..30100 by default) -- so every published number in
 * this study shares one evaluator revision (this file's own git rev,
 * recorded alongside the output), never a PyTorch-side validation fitness.
 *
 * Reuses `null-evaluate.ts`'s `runShardedEvaluation` (the exact fork-based
 * sharding/failure-handling mechanism the authored null uses) against
 * `null-trained-worker.ts`, rather than re-simulating anything itself. Task
 * list construction here only ever *locates and validates* existing run
 * directories/arm bundles -- it never trains, exports, or rewires.
 *
 * Output ordering (like `null-evaluate.ts`'s `assembleRaw`): rewired sorted
 * by numeric seed ascending, biological sorted by numeric trainer seed
 * ascending -- never insertion/collection order, so the published
 * `trained.json` is independent of shard count and completion timing.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

const DEFAULT_HELD_OUT_START = 30001;
const DEFAULT_HELD_OUT_COUNT = 100;
const DEFAULT_TICKS = 1800;
const DEFAULT_SHARDS = 8;
const DEFAULT_REWIRED_SEED_START = 0;
const DEFAULT_REWIRED_SEED_COUNT = 20;
const DEFAULT_REPLICA_SEED = 101;
const DEFAULT_BIOLOGICAL_TRAINER_SEEDS = [101, 202, 303] as const;
/** Matches `training/src/flyarena_training/cli.py`'s `DEFAULT_HIDDEN_SIZE` / `train-sample.sh`'s own `--hidden-size` default. */
const DEFAULT_HIDDEN_SIZE = 16;
const DEFAULT_REWIRED_TRAINED_DIR = resolve(repoRoot, 'training/runs/null/trained');
const DEFAULT_REWIRED_ARMS_DIR = resolve(repoRoot, 'training/runs/null/arms');
const DEFAULT_BIOLOGICAL_RUNS_DIR = resolve(repoRoot, 'training/runs/production');
const DEFAULT_BIOLOGICAL_ARMS_DIR = resolve(repoRoot, 'training/runs/arms');
const DEFAULT_OUT = resolve(repoRoot, 'training/runs/null/trained.json');

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface NullTrainedEvaluateArgs {
  readonly rewiredSeedStart: number;
  readonly rewiredSeedCount: number;
  readonly replicaSeed: number;
  readonly biologicalTrainerSeeds: readonly number[];
  readonly rewiredTrainedDir: string;
  readonly rewiredArmsDir: string;
  readonly biologicalRunsDir: string;
  readonly biologicalArmsDir: string;
  readonly heldOutStart: number;
  readonly heldOutCount: number;
  readonly ticks: number;
  /** H -- must match every scored run's own `config.json` `H` (see `null-trained-worker.ts`'s `assertRunMatchesExpectedIdentity`). */
  readonly hiddenSize: number;
  readonly shards: number;
  readonly out: string;
}

const parseTrainerSeeds = (flag: string, value: string): readonly number[] => {
  const seeds = value.split(',').map((entry) => {
    const parsed = Number(entry.trim());
    if (!Number.isInteger(parsed) || parsed <= 0) {
      throw new Error(`${flag} must be a comma-separated list of positive integers, got "${value}"`);
    }
    return parsed;
  });
  if (seeds.length === 0) throw new Error(`${flag} must list at least one trainer seed`);
  return seeds;
};

export const parseNullTrainedEvaluateArgs = (argv: readonly string[]): NullTrainedEvaluateArgs => {
  let rewiredSeedStart = DEFAULT_REWIRED_SEED_START;
  let rewiredSeedCount = DEFAULT_REWIRED_SEED_COUNT;
  let replicaSeed = DEFAULT_REPLICA_SEED;
  let biologicalTrainerSeeds: readonly number[] = DEFAULT_BIOLOGICAL_TRAINER_SEEDS;
  let rewiredTrainedDir = DEFAULT_REWIRED_TRAINED_DIR;
  let rewiredArmsDir = DEFAULT_REWIRED_ARMS_DIR;
  let biologicalRunsDir = DEFAULT_BIOLOGICAL_RUNS_DIR;
  let biologicalArmsDir = DEFAULT_BIOLOGICAL_ARMS_DIR;
  let heldOutStart = DEFAULT_HELD_OUT_START;
  let heldOutCount = DEFAULT_HELD_OUT_COUNT;
  let ticks = DEFAULT_TICKS;
  let hiddenSize = DEFAULT_HIDDEN_SIZE;
  let shards = DEFAULT_SHARDS;
  let out = DEFAULT_OUT;

  let index = 0;
  while (index < argv.length) {
    const flag = argv[index];
    if (flag === '--rewired-seed-start') {
      rewiredSeedStart = requireNonNegativeInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--rewired-seed-count') {
      rewiredSeedCount = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--replica-seed') {
      replicaSeed = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--biological-trainer-seeds') {
      biologicalTrainerSeeds = parseTrainerSeeds(flag, requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--rewired-trained-dir') {
      rewiredTrainedDir = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--rewired-arms-dir') {
      rewiredArmsDir = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--biological-runs-dir') {
      biologicalRunsDir = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--biological-arms-dir') {
      biologicalArmsDir = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
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
    } else if (flag === '--hidden-size') {
      hiddenSize = requirePositiveInt(flag, argv[index + 1]);
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

  if (!out.endsWith('.json')) throw new Error(`--out must end with ".json" (got "${out}")`);

  return {
    rewiredSeedStart,
    rewiredSeedCount,
    replicaSeed,
    biologicalTrainerSeeds,
    rewiredTrainedDir,
    rewiredArmsDir,
    biologicalRunsDir,
    biologicalArmsDir,
    heldOutStart,
    heldOutCount,
    ticks,
    hiddenSize,
    shards,
    out
  };
};

// ---------------------------------------------------------------------------
// Bundle/run-directory discovery
// ---------------------------------------------------------------------------

/**
 * `export-arms.ts` writes each invocation's bundles to
 * `<outDir>/<graphArtifactSha256>/`, where `graphArtifactSha256` is a hash
 * of the *biological* source graph (`--graph`), not the rewired one -- so
 * it is the same hash for every rewiring seed's own `--out` directory (each
 * seed's `train-sample.sh` invocation passes the same `--graph`). Rather
 * than recomputing that hash a second time here (which would require this
 * script to also take `--graph`, purely to reproduce a hash train-sample.sh
 * already computed once), this locates the single hash subdirectory
 * `export-arms.ts` actually created -- which also works unmodified for a
 * `--dry-run-fixture` run, whose `graphArtifactSha256` is a *content* hash
 * with no corresponding file to sha256sum.
 */
const findSingleSubdirectory = (parentDir: string, label: string): string => {
  if (!existsSync(parentDir)) {
    throw new Error(`null-trained-evaluate: ${label} directory does not exist: ${parentDir}`);
  }
  const entries = readdirSync(parentDir, { withFileTypes: true }).filter((entry) => entry.isDirectory());
  if (entries.length !== 1) {
    throw new Error(
      `null-trained-evaluate: expected exactly one subdirectory under ${parentDir} (${label}), found ` +
        `${entries.length}${entries.length > 0 ? ` (${entries.map((e) => e.name).join(', ')})` : ''}`
    );
  }
  return resolve(parentDir, entries[0].name);
};

const requireFile = (path: string, label: string): void => {
  if (!existsSync(path)) throw new Error(`null-trained-evaluate: missing ${label}: ${path}`);
};

/**
 * `export-arms.ts`'s own D-equality gate (`runExportArms`'s `distinctD`
 * check) already guarantees, *within one invocation*, that a seed's
 * biological/disconnected/rewired bundles share one D -- but it says
 * nothing about seed 5's rewired bundle agreeing with seed 12's, or with
 * the separately-exported biological bundle this script rescores against.
 * `03-trained-sample.md`'s own acceptance criterion ("export-arms.ts must
 * report the same D ... for every rewired bundle as for biological ...
 * Stop on a mismatch") is *across* every bundle this run actually uses, so
 * it is re-checked here, at report-build time, over every arm bundle this
 * script located -- not assumed merely because each one individually
 * passed its own seed's gate.
 */
const readBundleD = (armBundlePath: string): number => {
  const bundle = JSON.parse(readFileSync(armBundlePath, 'utf8')) as { readonly D?: unknown };
  if (typeof bundle.D !== 'number' || !Number.isInteger(bundle.D) || bundle.D <= 0) {
    throw new Error(`null-trained-evaluate: ${armBundlePath} has no valid "D" field`);
  }
  return bundle.D;
};

const assertMatchingD = (bundlePathsByGraphId: ReadonlyMap<string, string>): void => {
  const dByGraphId = new Map<string, number>();
  for (const [graphId, path] of bundlePathsByGraphId) dByGraphId.set(graphId, readBundleD(path));
  const distinctD = new Set(dByGraphId.values());
  if (distinctD.size > 1) {
    const detail = [...dByGraphId.entries()].map(([graphId, d]) => `${graphId}=${d}`).join(', ');
    throw new Error(`null-trained-evaluate: D (output-neuron count) differs across arm bundles: ${detail}`);
  }
};

// ---------------------------------------------------------------------------
// Task list
// ---------------------------------------------------------------------------

export const buildTasks = (args: Readonly<NullTrainedEvaluateArgs>): NullTrainedWorkerTask[] => {
  const heldOutSeeds = Array.from({ length: args.heldOutCount }, (_, i) => args.heldOutStart + i);
  const tasks: NullTrainedWorkerTask[] = [];
  const bundlePathsByGraphId = new Map<string, string>();

  for (let seed = args.rewiredSeedStart; seed < args.rewiredSeedStart + args.rewiredSeedCount; seed += 1) {
    const runDir = resolve(args.rewiredTrainedDir, `seed${seed}`);
    requireFile(resolve(runDir, 'config.json'), `rewired seed ${seed} config.json`);
    requireFile(resolve(runDir, 'theta_final.npy'), `rewired seed ${seed} theta_final.npy`);
    const bundleDir = findSingleSubdirectory(resolve(args.rewiredArmsDir, `seed${seed}`), `rewired seed ${seed} arms`);
    const armBundlePath = resolve(bundleDir, 'rewired.json');
    requireFile(armBundlePath, `rewired seed ${seed} arm bundle`);
    const graphId = `rewired-${seed}`;
    tasks.push({
      graphId,
      runDir,
      armBundlePath,
      heldOutSeeds,
      ticks: args.ticks,
      expectedArm: 'rewired',
      expectedTrainerSeed: args.replicaSeed,
      expectedSubsteps: NEURAL_SUBSTEPS_PER_TICK,
      expectedHiddenSize: args.hiddenSize
    });
    bundlePathsByGraphId.set(graphId, armBundlePath);
  }

  const biologicalBundleDir = findSingleSubdirectory(args.biologicalArmsDir, 'biological arms');
  const biologicalArmBundlePath = resolve(biologicalBundleDir, 'biological.json');
  requireFile(biologicalArmBundlePath, 'biological arm bundle');

  for (const trainerSeed of [...args.biologicalTrainerSeeds].sort((a, b) => a - b)) {
    const runDir = resolve(args.biologicalRunsDir, `biological-${trainerSeed}`);
    requireFile(resolve(runDir, 'config.json'), `biological trainer seed ${trainerSeed} config.json`);
    requireFile(resolve(runDir, 'theta_final.npy'), `biological trainer seed ${trainerSeed} theta_final.npy`);
    const graphId = `biological-${trainerSeed}`;
    tasks.push({
      graphId,
      runDir,
      armBundlePath: biologicalArmBundlePath,
      heldOutSeeds,
      ticks: args.ticks,
      expectedArm: 'biological',
      expectedTrainerSeed: trainerSeed,
      expectedSubsteps: NEURAL_SUBSTEPS_PER_TICK,
      expectedHiddenSize: args.hiddenSize
    });
    bundlePathsByGraphId.set(graphId, biologicalArmBundlePath);
  }

  assertMatchingD(bundlePathsByGraphId);

  return tasks;
};

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/**
 * Identical shape to `null-evaluate.ts`'s `NullGraphRaw` (both scripts
 * report the same per-graph `heldOutSeeds`/`movementScore`/`foodPickups`/
 * `hazardContacts` array set); aliased rather than redeclared so the two
 * scripts share one type and `toGraphRaw` (a thermo-maintainability review
 * finding).
 */
export type NullTrainedGraphRaw = NullGraphRaw;

export interface NullTrainedRewiredGraphRaw extends NullTrainedGraphRaw {
  readonly seed: number;
}

export interface NullTrainedBiologicalGraphRaw extends NullTrainedGraphRaw {
  readonly trainerSeed: number;
}

/**
 * `replicaSeed` records the single trainer seed used for every rewired run
 * (the plan's "one trainer seed (101) ... isolates topology from
 * trainer-seed variance" key decision) -- distinct from `biological`'s three
 * `trainerSeed`-keyed entries, which exist precisely to show trainer-seed
 * variance at fixed (biological) topology as a separate, explicitly-labeled
 * context (see `null-report.ts`'s `bioTrainerSeedSpread`).
 */
export interface NullTrainedEvaluationRaw {
  readonly version: 1;
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
  readonly replicaSeed: number;
  /** Sorted by seed ascending. */
  readonly rewired: readonly NullTrainedRewiredGraphRaw[];
  /** Sorted by trainerSeed ascending. */
  readonly biological: readonly NullTrainedBiologicalGraphRaw[];
  readonly host: { readonly arch: string; readonly node: string };
  /** Output-neuron count, verified equal across every rewired/biological arm bundle this run used (see `assertMatchingD`). */
  readonly d: number;
  /** The `flyarena-bigq` merge commit this study's config/biological runs were re-grounded against. */
  readonly bigqMergeCommit: string;
  /** This script's own git rev at evaluation time (`git rev-parse HEAD`), `null` when not a git checkout. */
  readonly evaluatorGitRev: string | null;
  /**
   * `run-dir.ts`'s `CEM_CONFIG_FIELDS`, reconciled from every task's own
   * `config.json` (the first task's value is the baseline; see
   * `reconcileCemConfig`) -- copied here, not hard-coded, so this script can
   * never silently drift from what the run directories it actually scored
   * were trained with. `null` only if not a single task carried a recorded
   * CEM config (never expected for a real `flyarena-train` run directory).
   */
  readonly cemConfig: Record<string, unknown> | null;
  /**
   * Always `[]` when this script's output was actually written:
   * `reconcileCemConfig` now throws (rather than warning) on any CEM-config
   * disagreement, per the plan's "the CEM config must match bigq's, so it
   * cannot be silently shrunk" -- see that function's own doc comment. Kept
   * in the schema (not removed) for forward compatibility with a possible
   * future explicit override flag, and because `null-report.ts`'s
   * `TrainedSection`/report-markdown already surface it.
   */
  readonly cemConfigWarnings: readonly string[];
}

/** `git rev-parse HEAD`, `null` on any failure (not a git checkout, `git` missing) -- informational, matches `training/src/flyarena_training/cli.py`'s `_git_rev` convention. */
const gitRev = (): string | null => {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repoRoot, encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
};

/**
 * `run-dir.ts`'s `CEM_CONFIG_FIELDS` plus `ticks` (T -- written by
 * `flyarena-train` but, unlike the other CEM hyperparameters, not part of
 * `RunConfig`'s declared TS shape; read via an explicit untyped cast here,
 * the same way `CEM_CONFIG_FIELDS` fields are already read off the parsed
 * config). A run trained at a different tick count is just as invalid a
 * comparison as one trained at a different population/generations/etc., so
 * it is checked alongside them (a round-2 dual-review finding: an earlier
 * version compared only `CEM_CONFIG_FIELDS`, silently missing `ticks`).
 */
const TICKS_FIELD = 'ticks' as const;
const CONFIG_RECONCILE_FIELDS: readonly string[] = [...CEM_CONFIG_FIELDS, TICKS_FIELD];

/**
 * Every task's `config.json`, reduced to `CONFIG_RECONCILE_FIELDS`.
 * `.agents/plans/rewiring-null/03-trained-sample.md`'s own words: "The CEM
 * config must match bigq's, so it cannot be silently shrunk" -- so, unlike
 * `run-dir.ts`'s `deriveTrainingBlock` (which only *warns* on a mismatch,
 * because it is building a purely informational manifest block for
 * `evaluate.ts`), ANY disagreement here throws. This study's every
 * published number depends on every run sharing one config except
 * arm/trainer-seed, so a silently-published mismatch would invalidate the
 * whole comparison, not just one field of a report (a round-2 dual-review
 * finding: an earlier version of this function only warned, matching
 * `deriveTrainingBlock`'s precedent, which does not actually apply here).
 *
 * The baseline is the FIRST biological task (sorted by trainerSeed
 * ascending -- `buildTasks` always builds biological tasks in that order),
 * never "the first task in `tasks` order" (which would usually be
 * `rewired-<rewiredSeedStart>`): a biological run directory is copied
 * directly from the already bigq-manifest-verified production runs, never
 * produced by this study's own `train-sample.sh`, so it is the more
 * trustworthy reference. A rewired run, by contrast, could in principle be
 * a stale/small calibration run that `train-sample.sh`'s resumability check
 * (keyed only on `config.json` existing) would treat as "done" and never
 * regenerate -- using it as the baseline would silently make every OTHER
 * (correctly-configured) run look like the outlier instead.
 *
 * A task whose `config.json` has no recorded CEM hyperparameters at all
 * (every `CONFIG_RECONCILE_FIELDS` value `undefined` -- only ever expected
 * from a tiny/older test fixture, never a real `flyarena-train` run) is
 * tolerated only when EVERY task is like that (`cemConfig: null`,
 * `warnings: []`); any other combination -- some tasks recorded, some not,
 * or recorded-but-different -- throws.
 */
const reconcileCemConfig = (
  tasks: readonly NullTrainedWorkerTask[]
): { readonly cemConfig: Record<string, unknown> | null; readonly warnings: readonly string[] } => {
  const candidates: Array<{ graphId: string; candidate: Record<string, unknown> }> = tasks.map((task) => {
    const { config } = readRunDir(task.runDir);
    const candidate: Record<string, unknown> = {};
    for (const field of CONFIG_RECONCILE_FIELDS) candidate[field] = (config as unknown as Record<string, unknown>)[field];
    return { graphId: task.graphId, candidate };
  });

  if (candidates.every(({ candidate }) => isEmptyCemConfig(candidate))) {
    return { cemConfig: null, warnings: [] };
  }

  const biologicalTask = tasks.find((task) => task.expectedArm === 'biological');
  const baselineGraphId = biologicalTask?.graphId ?? tasks[0].graphId;
  const baselineEntry = candidates.find(({ graphId }) => graphId === baselineGraphId)!;
  if (isEmptyCemConfig(baselineEntry.candidate)) {
    throw new Error(
      `null-trained-evaluate: "${baselineGraphId}" (the CEM-config baseline) has no recorded CEM hyperparameters, ` +
        'but at least one other scored run does -- refusing to publish an inconsistent trained.json'
    );
  }

  for (const { graphId, candidate } of candidates) {
    if (graphId === baselineGraphId) continue;
    if (JSON.stringify(candidate) !== JSON.stringify(baselineEntry.candidate)) {
      throw new Error(
        `null-trained-evaluate: "${graphId}"'s CEM config/training-seed policy (including ticks) differs from ` +
          `"${baselineGraphId}"'s -- ${JSON.stringify(candidate)} vs ${JSON.stringify(baselineEntry.candidate)} -- ` +
          "this study's config must match bigq's exactly (03-trained-sample.md's \"Time budget\" section), so it " +
          'cannot be silently published as a warning'
      );
    }
  }

  return { cemConfig: baselineEntry.candidate, warnings: [] };
};

export const assembleRaw = (
  args: Readonly<NullTrainedEvaluateArgs>,
  tasks: readonly NullTrainedWorkerTask[],
  results: ReadonlyMap<string, readonly NullSeedResult[]>
): NullTrainedEvaluationRaw => {
  const require = (graphId: string): readonly NullSeedResult[] => {
    const found = results.get(graphId);
    if (!found) throw new Error(`null-trained-evaluate: missing results for "${graphId}"`);
    return found;
  };

  const rewiredSeeds = Array.from(
    { length: args.rewiredSeedCount },
    (_, i) => args.rewiredSeedStart + i
  ).sort((a, b) => a - b);
  const rewired: NullTrainedRewiredGraphRaw[] = rewiredSeeds.map((seed) => ({
    seed,
    ...toGraphRaw(require(`rewired-${seed}`))
  }));

  const biological: NullTrainedBiologicalGraphRaw[] = [...args.biologicalTrainerSeeds]
    .sort((a, b) => a - b)
    .map((trainerSeed) => ({ trainerSeed, ...toGraphRaw(require(`biological-${trainerSeed}`)) }));

  if (tasks.length === 0) throw new Error('null-trained-evaluate: assembleRaw requires at least one task');
  const { cemConfig, warnings: cemConfigWarnings } = reconcileCemConfig(tasks);

  return {
    version: 1,
    seeds: { start: args.heldOutStart, count: args.heldOutCount },
    ticks: args.ticks,
    substeps: NEURAL_SUBSTEPS_PER_TICK,
    replicaSeed: args.replicaSeed,
    rewired,
    biological,
    host: { arch: process.arch, node: process.version },
    d: readBundleD(tasks[0].armBundlePath),
    bigqMergeCommit: BIGQ_MERGE_COMMIT,
    evaluatorGitRev: gitRev(),
    cemConfig,
    cemConfigWarnings
  };
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

export const runNullTrainedEvaluate = async (
  args: Readonly<NullTrainedEvaluateArgs>
): Promise<{ out: string; runMetaOut: string; taskCount: number; elapsedMs: number }> => {
  const tasks = buildTasks(args);
  const workerPath = fileURLToPath(new URL('./null-trained-worker.ts', import.meta.url));

  const started = performance.now();
  const results = await runShardedEvaluation<NullTrainedWorkerTask, NullSeedResult, NullWorkerMessage>(
    tasks,
    args.shards,
    workerPath
  );
  const elapsedMs = performance.now() - started;
  const perEpisodeMs = elapsedMs / (tasks.length * args.heldOutCount);

  const raw = assembleRaw(args, tasks, results);
  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, JSON.stringify(raw));

  const runMetaOut = runMetaPathFor('null-trained-evaluate', args.out);
  atomicWriteFileSync(runMetaOut, `${JSON.stringify({ shards: args.shards, elapsedMs, perEpisodeMs }, null, 2)}\n`);

  return { out: args.out, runMetaOut, taskCount: tasks.length, elapsedMs };
};

const main = runCliMain('null-trained-evaluate', 'runs', parseNullTrainedEvaluateArgs, runNullTrainedEvaluate);

if (process.argv[1] === fileURLToPath(import.meta.url)) void main();
