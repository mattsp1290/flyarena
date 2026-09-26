import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { MANIFEST_TRACKED_FIELDS, readRunDir } from '../training/run-dir';
import { readGraphListIndex, verifyGraphListFiles } from './graph-list-index';
import { runCliMain, runMetaPathFor, runShardedEvaluation, toGraphRaw } from './null-evaluate';
import {
  assertMatchingD,
  DEFAULT_HELD_OUT_COUNT,
  DEFAULT_HELD_OUT_START,
  DEFAULT_HIDDEN_SIZE,
  DEFAULT_SHARDS,
  DEFAULT_TICKS,
  findSingleSubdirectory,
  gitRev,
  readBundleD,
  reconcileCemConfig,
  repoRoot,
  requireFile,
  type NullTrainedGraphRaw
} from './null-trained-evaluate';
import type { NullSeedResult, NullWorkerMessage } from './null-worker';
import type { NullTrainedWorkerTask } from './null-trained-worker';

/**
 * `.agents/plans/pathway-interventions/03-evaluation.md`'s WP3:
 * `--graph-list`/`--trained-dir` mode. Rescores `scripts/null/train-sample.sh`'s
 * `--graph-list`/`--ids` run directories (`<trained-dir>/<id>-seed<trainerSeed>/`,
 * one per `(intervention-or-control id, trainer seed)` pair — P at trainer
 * seeds 101/202/303, C000..C004 and M1000..M1004 at trainer seed 101, this
 * study's predeclared 13 runs) — decoder `trained`, opponent parked, same
 * held-out seeds (30001..30100 by default) `null-trained-evaluate.ts`'s
 * rewired-seed mode and the authored null both use.
 *
 * Extracted from `null-trained-evaluate.ts` into its own module (a
 * thermo-maintainability review finding: the addition pushed that file to
 * exactly 1000 lines, the same threshold `null-evaluate.ts` crossed and was
 * split for one commit earlier in this same plan — see `graph-list-index.ts`'s
 * own doc comment for that precedent). A wholly separate CLI mode/output
 * shape from `null-trained-evaluate.ts`'s rewired-seed mode
 * (`NullTrainedEvaluationRaw`'s numeric `rewired`/`biological` seed lists
 * don't fit "arbitrary string ids, 1-3 trainer seeds each"), but it reuses
 * every reusable piece of that pipeline: `null-trained-worker.ts` (the SAME
 * worker script — `NullTrainedWorkerTask`'s shape already covers "a run
 * directory + arm bundle path scored with `--arm rewired`", which this
 * mode's tasks are, exactly), `runShardedEvaluation`, and, imported from
 * `null-trained-evaluate.ts` (exported specifically so this module could be
 * split out without duplicating them): `findSingleSubdirectory`,
 * `readBundleD`/`assertMatchingD`, `reconcileCemConfig`, `gitRev`, and the
 * shared `DEFAULT_HELD_OUT_START`/etc. defaults.
 *
 * The `index.json` this mode reads (`scripts/analysis/interventions.py`'s
 * id-keyed intervention/control graph list) is the exact same format
 * `null-evaluate.ts`'s own `--graph-list` mode (WP2) reads — this module
 * reuses that mode's already-hardened, path-traversal-safe reader/verifier
 * (`graph-list-index.ts`'s `readGraphListIndex`/`verifyGraphListFiles`)
 * rather than a second, independently-drifting parser (a review finding: an
 * earlier version of this module had its own parallel `index.json` reader
 * with a weaker verification contract — see `git log` on this file for that
 * history).
 */

interface InterventionRunSpec {
  readonly id: string;
  readonly trainerSeed: number;
}

/** `--runs id:trainerSeed[,id:trainerSeed...]` — mirrors `null-trained-evaluate.ts`'s own `parseTrainerSeeds` validation style, generalized to an `id:seed` pair per entry (an intervention/control id has no fixed numeric range the way a rewiring seed does). */
const parseRunSpecs = (flag: string, value: string): readonly InterventionRunSpec[] => {
  const specs = value.split(',').map((raw) => {
    const trimmed = raw.trim();
    // Restricted to letters/digits/underscore/hyphen (a review finding,
    // defense in depth): `id` feeds directly into `--trained-dir`/
    // `--arms-dir` lookups below (`resolve(args.trainedDir, "${id}-seed${trainerSeed}")`,
    // `resolve(args.armsDir, id)`), and matches `train-sample.sh`'s own
    // identical restriction on the ids it writes those same paths from.
    const match = /^([A-Za-z0-9_-]+):(\d+)$/.exec(trimmed);
    if (!match) {
      throw new Error(`${flag} must be a comma-separated list of id:trainerSeed pairs (got "${value}")`);
    }
    const trainerSeed = Number(match[2]);
    if (!Number.isInteger(trainerSeed) || trainerSeed <= 0) {
      throw new Error(`${flag}: trainer seed for id "${match[1]}" must be a positive integer (got "${match[2]}")`);
    }
    return { id: match[1], trainerSeed };
  });
  if (specs.length === 0) throw new Error(`${flag} must list at least one id:trainerSeed pair`);
  const seen = new Set<string>();
  for (const spec of specs) {
    const key = `${spec.id}:${spec.trainerSeed}`;
    // A duplicate (id, trainerSeed) pair would create two tasks with the
    // same graphId (see `buildInterventionTasks`'s own `graphId`), which
    // result "wins" then depends on completion order -- exactly what the
    // shard byte-identity guarantee promises can never happen.
    if (seen.has(key)) throw new Error(`${flag} lists "${key}" more than once`);
    seen.add(key);
  }
  return specs;
};

export interface NullTrainedInterventionEvaluateArgs {
  readonly graphList: string;
  readonly runs: readonly InterventionRunSpec[];
  readonly trainedDir: string;
  readonly armsDir: string;
  readonly heldOutStart: number;
  readonly heldOutCount: number;
  readonly ticks: number;
  readonly hiddenSize: number;
  readonly shards: number;
  readonly out: string;
  /** See `assertConfigsMatchManifest`'s doc comment. */
  readonly manifestPath: string;
}

const DEFAULT_INTERVENTION_TRAINED_DIR = resolve(repoRoot, 'training/runs/interventions/trained');
const DEFAULT_INTERVENTION_ARMS_DIR = resolve(repoRoot, 'training/runs/interventions/arms');
const DEFAULT_INTERVENTION_OUT = resolve(repoRoot, 'training/runs/interventions/trained.json');
const DEFAULT_INTERVENTION_MANIFEST_PATH = resolve(repoRoot, 'public/data/trained-readout-v1.manifest.json');

export const parseNullTrainedInterventionEvaluateArgs = (argv: readonly string[]): NullTrainedInterventionEvaluateArgs => {
  let graphList: string | undefined;
  let runs: readonly InterventionRunSpec[] | undefined;
  let trainedDir = DEFAULT_INTERVENTION_TRAINED_DIR;
  let armsDir = DEFAULT_INTERVENTION_ARMS_DIR;
  let heldOutStart = DEFAULT_HELD_OUT_START;
  let heldOutCount = DEFAULT_HELD_OUT_COUNT;
  let ticks = DEFAULT_TICKS;
  let hiddenSize = DEFAULT_HIDDEN_SIZE;
  let shards = DEFAULT_SHARDS;
  let out = DEFAULT_INTERVENTION_OUT;
  let manifestPath = DEFAULT_INTERVENTION_MANIFEST_PATH;

  let index = 0;
  while (index < argv.length) {
    const flag = argv[index];
    if (flag === '--graph-list') {
      graphList = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--runs') {
      runs = parseRunSpecs(flag, requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--trained-dir') {
      trainedDir = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--arms-dir') {
      armsDir = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
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
    } else if (flag === '--manifest') {
      manifestPath = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (!graphList) throw new Error('--graph-list is required');
  if (!runs) throw new Error('--runs is required');
  if (!out.endsWith('.json')) throw new Error(`--out must end with ".json" (got "${out}")`);

  return { graphList, runs, trainedDir, armsDir, heldOutStart, heldOutCount, ticks, hiddenSize, shards, out, manifestPath };
};

/**
 * Every requested `(id, trainerSeed)` pair's arm bundle is checked against
 * the graph-list's own recorded `gzipSha256` for that id
 * (`provenance.kind === 'rewired-artifact'` and `artifactSha256` equal) --
 * proof that this run directory was actually trained against WP1's `id`
 * graph, not merely a directory that happens to be named `<id>-seed<seed>`
 * (a stale/mismatched rename, or an operator error). `assertMatchingD`
 * (imported from `null-trained-evaluate.ts`) separately checks every bundle
 * in this run shares one output-neuron count.
 */
const assertBundleMatchesInterventionGraph = (
  armBundlePath: string,
  graphListPath: string,
  id: string,
  entry: { readonly gzipSha256: string }
): void => {
  const bundle = JSON.parse(readFileSync(armBundlePath, 'utf8')) as {
    readonly provenance?: { readonly kind?: string; readonly artifactSha256?: string };
  };
  if (bundle.provenance?.kind !== 'rewired-artifact' || bundle.provenance.artifactSha256 !== entry.gzipSha256) {
    throw new Error(
      `null-trained-evaluate: arm bundle for id "${id}" (${armBundlePath}) does not match --graph-list ` +
        `${graphListPath}'s "${id}" entry (provenance ${JSON.stringify(bundle.provenance)} vs expected ` +
        `rewired-artifact with artifactSha256 ${entry.gzipSha256})`
    );
  }
};

/**
 * `train-sample.sh`'s CEM hyperparameters (population/elites/generations/
 * alpha/stdFloor/initStd/trainingSeedsPerGeneration) plus top-level `H` --
 * the exact fields `train-sample.sh`'s own `read_manifest_cem_config`
 * reads from `public/data/trained-readout-v1.manifest.json` before any
 * export-arms/flyarena-train call. `MANIFEST_TRACKED_FIELDS` (imported from
 * `run-dir.ts`) is `CEM_CONFIG_FIELDS` minus the per-run-directory-only
 * fields the manifest doesn't carry, rather than a second hand-typed
 * literal array that could silently drift from it (a thermo-maintainability
 * review suggestion).
 */
const readManifestExpectedConfig = (manifestPath: string): Record<string, unknown> => {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as {
    readonly training?: Record<string, unknown>;
    readonly H?: unknown;
  };
  if (!manifest.training) throw new Error(`null-trained-evaluate: ${manifestPath} has no "training" object`);
  const expected: Record<string, unknown> = { H: manifest.H };
  for (const field of MANIFEST_TRACKED_FIELDS) expected[field] = manifest.training[field];
  return expected;
};

/**
 * `reconcileCemConfig` (imported from `null-trained-evaluate.ts`) only
 * checks this mode's own runs against EACH OTHER, using the first
 * biological-or-`tasks[0]` task as its baseline -- a baseline every
 * intervention-mode task fails to provide (there is no biological task in
 * this mode), so it silently falls back to `tasks[0]`, i.e. whichever run
 * happened to be listed FIRST in `--runs`. If every run in a given
 * invocation shared the same override (e.g. a `--generations 10` left over
 * from a calibration wrapper script, or the manifest was regenerated
 * between this study and `flyarena-bigq`), every task would "agree" with
 * every other and `reconcileCemConfig` would raise nothing -- exactly the
 * "CEM config is never shrunk silently" guarantee (`03-evaluation.md`'s WP3
 * section) failing right at publication. This checks every task's recorded
 * CEM config directly against the shipped manifest's `training` block (the
 * same source of truth `train-sample.sh`'s own preflight reads),
 * independent of `--runs`' argument order. A task with NO recorded CEM
 * fields at all is tolerated (matches `isEmptyCemConfig`'s own precedent --
 * an older/tiny test-fixture run dir predates these fields); a task with
 * SOME fields recorded must match the manifest on every one of them.
 */
export const assertConfigsMatchManifest = (
  tasks: readonly NullTrainedWorkerTask[],
  manifestPath: string
): void => {
  const expected = readManifestExpectedConfig(manifestPath);
  const fields = Object.keys(expected);
  for (const task of tasks) {
    const { config } = readRunDir(task.runDir);
    const candidate = config as unknown as Record<string, unknown>;
    if (fields.every((field) => candidate[field] === undefined)) continue;
    for (const field of fields) {
      if (candidate[field] === undefined) continue;
      if (JSON.stringify(candidate[field]) !== JSON.stringify(expected[field])) {
        throw new Error(
          `null-trained-evaluate: "${task.graphId}"'s ${field}=${JSON.stringify(candidate[field])} does not match ` +
            `${manifestPath}'s ${field}=${JSON.stringify(expected[field])} -- the CEM config must never be ` +
            'silently shrunk or otherwise drift from the shipped manifest'
        );
      }
    }
  }
};

export const buildInterventionTasks = (
  args: Readonly<NullTrainedInterventionEvaluateArgs>
): NullTrainedWorkerTask[] => {
  // Read and verify the graph-list ONCE (mirrors `null-evaluate.ts`'s own
  // `--graph-list` mode: `readGraphListIndex` then `verifyGraphListFiles`
  // before any task is built), not once per requested id -- `verifyGraphListFiles`
  // checks every entry in the index, so calling it per-id would needlessly
  // re-hash every graph in the list once per requested run.
  const index = readGraphListIndex(args.graphList);
  verifyGraphListFiles(index, dirname(args.graphList));
  const entryById = new Map(index.entries.map((entry) => [entry.id, entry]));

  const heldOutSeeds = Array.from({ length: args.heldOutCount }, (_, i) => args.heldOutStart + i);
  const bundlePathsByGraphId = new Map<string, string>();

  const tasks = args.runs.map((run) => {
    const entry = entryById.get(run.id);
    if (!entry) throw new Error(`null-trained-evaluate: id "${run.id}" not found in --graph-list ${args.graphList}`);
    const graphId = `${run.id}-seed${run.trainerSeed}`;
    const runDir = resolve(args.trainedDir, graphId);
    requireFile(resolve(runDir, 'config.json'), `intervention "${run.id}" trainer seed ${run.trainerSeed} config.json`);
    requireFile(
      resolve(runDir, 'theta_final.npy'),
      `intervention "${run.id}" trainer seed ${run.trainerSeed} theta_final.npy`
    );
    const bundleDir = findSingleSubdirectory(resolve(args.armsDir, run.id), `intervention "${run.id}" arms`);
    const armBundlePath = resolve(bundleDir, 'rewired.json');
    requireFile(armBundlePath, `intervention "${run.id}" arm bundle`);
    assertBundleMatchesInterventionGraph(armBundlePath, args.graphList, run.id, entry);
    bundlePathsByGraphId.set(graphId, armBundlePath);
    const task: NullTrainedWorkerTask = {
      graphId,
      runDir,
      armBundlePath,
      heldOutSeeds,
      ticks: args.ticks,
      expectedArm: 'rewired',
      expectedTrainerSeed: run.trainerSeed,
      expectedSubsteps: NEURAL_SUBSTEPS_PER_TICK,
      expectedHiddenSize: args.hiddenSize
    };
    return task;
  });

  assertMatchingD(bundlePathsByGraphId);
  return tasks;
};

export interface NullTrainedInterventionGraphRaw extends NullTrainedGraphRaw {
  readonly id: string;
  readonly trainerSeed: number;
  /** `id`'s `--graph-list` entry `gzipSha256`, re-verified (not merely copied) by `buildInterventionTasks` before this row was scored -- a review finding: the row itself should carry the graph it was scored against, not require a reader to cross-reference the archived index separately. */
  readonly gzipSha256: string;
  /** The arm bundle's own self-certifying `sha256` (`export-arms.ts`'s `computeArmBundleSha256`), already proven to equal `config.armBundleSha256` (`null-trained-worker.ts`'s `runTask`) and tied to `gzipSha256` above (`assertBundleMatchesInterventionGraph`). */
  readonly armBundleSha256: string;
}

/**
 * `runs`, sorted by `id` ascending then `trainerSeed` ascending -- never
 * `args.runs`' own (caller-supplied, and possibly shard-timing-adjacent)
 * order, so this output is independent of `--runs`' argument order and of
 * `--shards`/completion timing, matching `null-evaluate.ts`'s "canonical
 * task order" convention.
 */
export interface NullTrainedInterventionEvaluationRaw {
  readonly version: 1;
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
  readonly graphListSha256: string;
  readonly runs: readonly NullTrainedInterventionGraphRaw[];
  readonly host: { readonly arch: string; readonly node: string };
  readonly d: number;
  readonly evaluatorGitRev: string | null;
  /** See `reconcileCemConfig`'s doc comment in `null-trained-evaluate.ts` -- reused verbatim for this mode's own tasks. */
  readonly cemConfig: Record<string, unknown> | null;
  readonly cemConfigWarnings: readonly string[];
}

export const assembleInterventionRaw = (
  args: Readonly<NullTrainedInterventionEvaluateArgs>,
  tasks: readonly NullTrainedWorkerTask[],
  results: ReadonlyMap<string, readonly NullSeedResult[]>
): NullTrainedInterventionEvaluationRaw => {
  const require = (graphId: string): readonly NullSeedResult[] => {
    const found = results.get(graphId);
    if (!found) throw new Error(`null-trained-evaluate: missing results for "${graphId}"`);
    return found;
  };

  // Each output row carries the exact graph and arm bundle hashes it was
  // scored against, rather than requiring a reader to separately
  // cross-reference the archived `--graph-list`/arms tree. `gzipSha256`
  // comes from the SAME verified index entry `buildInterventionTasks`
  // already checked this task's bundle against
  // (`assertBundleMatchesInterventionGraph`); `armBundleSha256` is the
  // bundle's own self-certifying hash, already proven equal to
  // `config.armBundleSha256` by the worker (`null-trained-worker.ts`'s
  // `runTask`) before this task was ever scored.
  const graphListBytes = readFileSync(args.graphList);
  const index = readGraphListIndex(args.graphList);
  const gzipShaById = new Map(index.entries.map((entry) => [entry.id, entry.gzipSha256]));
  const armBundleSha256ByGraphId = new Map<string, string>();
  for (const task of tasks) {
    const bundle = JSON.parse(readFileSync(task.armBundlePath, 'utf8')) as { readonly sha256?: unknown };
    if (typeof bundle.sha256 !== 'string') {
      throw new Error(`null-trained-evaluate: ${task.armBundlePath} has no "sha256" field`);
    }
    armBundleSha256ByGraphId.set(task.graphId, bundle.sha256);
  }

  const runs: NullTrainedInterventionGraphRaw[] = [...args.runs]
    .sort((a, b) => (a.id === b.id ? a.trainerSeed - b.trainerSeed : a.id < b.id ? -1 : 1))
    .map((run) => {
      const graphId = `${run.id}-seed${run.trainerSeed}`;
      const gzipSha256 = gzipShaById.get(run.id);
      if (!gzipSha256) {
        throw new Error(`null-trained-evaluate: id "${run.id}" not found in --graph-list ${args.graphList}`);
      }
      const armBundleSha256 = armBundleSha256ByGraphId.get(graphId);
      if (!armBundleSha256) throw new Error(`null-trained-evaluate: missing arm bundle sha256 for "${graphId}"`);
      return {
        id: run.id,
        trainerSeed: run.trainerSeed,
        gzipSha256,
        armBundleSha256,
        ...toGraphRaw(require(graphId))
      };
    });

  if (tasks.length === 0) throw new Error('null-trained-evaluate: assembleInterventionRaw requires at least one task');
  const { cemConfig, warnings: cemConfigWarnings } = reconcileCemConfig(tasks);

  return {
    version: 1,
    seeds: { start: args.heldOutStart, count: args.heldOutCount },
    ticks: args.ticks,
    substeps: NEURAL_SUBSTEPS_PER_TICK,
    graphListSha256: sha256Hex(graphListBytes),
    runs,
    host: { arch: process.arch, node: process.version },
    d: readBundleD(tasks[0].armBundlePath),
    evaluatorGitRev: gitRev(),
    cemConfig,
    cemConfigWarnings
  };
};

export const runNullTrainedInterventionEvaluate = async (
  args: Readonly<NullTrainedInterventionEvaluateArgs>
): Promise<{ out: string; runMetaOut: string; taskCount: number; elapsedMs: number }> => {
  const tasks = buildInterventionTasks(args);
  // Fails fast, before the (potentially long) sharded rescoring pass below,
  // and independent of `--runs`' argument order -- see
  // `assertConfigsMatchManifest`'s own doc comment.
  assertConfigsMatchManifest(tasks, args.manifestPath);
  const workerPath = fileURLToPath(new URL('./null-trained-worker.ts', import.meta.url));

  const started = performance.now();
  const results = await runShardedEvaluation<NullTrainedWorkerTask, NullSeedResult, NullWorkerMessage>(
    tasks,
    args.shards,
    workerPath
  );
  const elapsedMs = performance.now() - started;
  const perEpisodeMs = elapsedMs / (tasks.length * args.heldOutCount);

  const raw = assembleInterventionRaw(args, tasks, results);
  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, JSON.stringify(raw));

  const runMetaOut = runMetaPathFor('null-trained-evaluate', args.out);
  atomicWriteFileSync(runMetaOut, `${JSON.stringify({ shards: args.shards, elapsedMs, perEpisodeMs }, null, 2)}\n`);

  return { out: args.out, runMetaOut, taskCount: tasks.length, elapsedMs };
};

/** This module's own CLI entrypoint, invoked by `null-trained-evaluate.ts`'s `dispatchMain` when `--graph-list` is present. */
export const runInterventionMain = runCliMain(
  'null-trained-evaluate',
  'runs',
  parseNullTrainedInterventionEvaluateArgs,
  runNullTrainedInterventionEvaluate
);
