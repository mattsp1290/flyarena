import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync, gitRev, sha256Hex } from '../training/fsio';
import { arenaTaskOutputFields, parseArenaTaskArg, type ArenaTaskOutputFields } from './arena-task-fields';
import {
  readGraphListIndex,
  sortedGraphListEntries,
  verifyGraphListFiles,
  type GraphListEntry,
  type GraphListIndex
} from './graph-list-index';
import {
  NULL_DECODER_KINDS,
  type NullDecoderKind,
  type NullSeedResult,
  type NullTaskMode,
  type NullWorkerMessage,
  type NullWorkerTask
} from './null-worker';
import {
  readRewireIndex,
  sortedRewireSeeds,
  verifyBiologicalSource,
  verifyRewiredFiles,
  type RewireIndex,
  type RewireIndexSeedEntry
} from './rewire-index';
import { runShardedEvaluation } from './sharded-evaluation';

/**
 * Re-exported so every existing importer of these symbols from
 * `./null-evaluate` (`regime-check.ts`, `lookup-rewired-artifact.ts`,
 * `null-report.ts`, this branch's own tests, and the concurrent
 * `feat/hbru-trained-interventions` branch) keeps working unchanged --
 * `rewire-index.ts`, `graph-list-index.ts`, and `sharded-evaluation.ts` are
 * where these are now defined and documented (a thermo-maintainability
 * review finding: this file crossed the 1000-line threshold twice; each
 * self-contained section, mirroring the others in shape, was an extraction).
 */
export {
  readGraphListIndex,
  sortedGraphListEntries,
  verifyGraphListFiles,
  type GraphListEntry,
  type GraphListIndex,
  readRewireIndex,
  verifyBiologicalSource,
  verifyRewiredFiles,
  type RewireIndex,
  type RewireIndexSeedEntry,
  runShardedEvaluation
};

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
const DEFAULT_DECODER: NullDecoderKind = 'authored';

/**
 * The only values `--decoder` accepts — the authored decoder family
 * (`.agents/plans/null-explanation/01-decoder-variants.md` WP1). `trained`/
 * `silenced`/`parked` are never valid here: this evaluator always drives the
 * left agent through the authored path (with an optional sign flip) against
 * a parked opponent. Derived from `null-worker.ts`'s `NULL_DECODER_KINDS`,
 * the single source of truth for the four kinds.
 */
const isNullDecoderKind = (value: string): value is NullDecoderKind =>
  (NULL_DECODER_KINDS as readonly string[]).includes(value);

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface NullEvaluateArgs {
  readonly biological: boolean;
  readonly graph?: string;
  /** Required unless `graphList` is set — mutually exclusive with it. */
  readonly rewiredIndex?: string;
  /** Required unless `graphList` is set — mutually exclusive with it. */
  readonly graphsDir?: string;
  /**
   * `scripts/analysis/interventions.py`'s `index.json` path
   * (`.agents/plans/pathway-interventions/03-evaluation.md` WP2's
   * `--graph-list` mode). Mutually exclusive with `rewiredIndex`/`graphsDir`
   * — this mode scores an explicit, arbitrary list of graphs (P, Q, the
   * C/M/MQ control arms) rather than a `rewire_batch.py` seed-indexed batch.
   */
  readonly graphList?: string;
  readonly heldOutStart: number;
  readonly heldOutCount: number;
  readonly ticks: number;
  readonly shards: number;
  readonly out: string;
  /** Left-agent decoder for every task this run builds. Defaults to `'authored'`. */
  readonly decoder: NullDecoderKind;
  /**
   * Restricts `buildTasks`'s rewired-graph tasks to `index.json` seeds in
   * `[start, end)` (half-open, matching `--rewired-seeds START:END`'s CLI
   * spelling to Python slice semantics) — used by the WP1 reproduction gate
   * (`--rewired-seeds 0:5` re-scores only seeds 0..4) so a full 500-graph
   * rerun isn't needed just to check that the new code path reproduces a
   * handful of published scores exactly. Leaving it unset evaluates every
   * seed in the index, unchanged from before this flag existed.
   */
  readonly rewiredSeeds?: { readonly start: number; readonly end: number };
  readonly arenaTask?: string;
}

export const parseNullEvaluateArgs = (argv: readonly string[]): NullEvaluateArgs => {
  let biological = false;
  let graph: string | undefined;
  let rewiredIndex: string | undefined;
  let graphsDir: string | undefined;
  let graphList: string | undefined;
  let heldOutStart = DEFAULT_HELD_OUT_START;
  let heldOutCount = DEFAULT_HELD_OUT_COUNT;
  let ticks = DEFAULT_TICKS;
  let shards = DEFAULT_SHARDS;
  let out = DEFAULT_OUT;
  let decoder: NullDecoderKind = DEFAULT_DECODER;
  let rewiredSeeds: { start: number; end: number } | undefined;
  let arenaTask: string | undefined;

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
    } else if (flag === '--graph-list') {
      graphList = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
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
    } else if (flag === '--decoder') {
      const value = requireValue(flag, argv[index + 1]);
      if (!isNullDecoderKind(value)) {
        throw new Error(`--decoder must be one of ${NULL_DECODER_KINDS.join(', ')} (got "${value}")`);
      }
      decoder = value;
      index += 2;
    } else if (flag === '--rewired-seeds') {
      const value = requireValue(flag, argv[index + 1]);
      const match = /^(\d+):(\d+)$/.exec(value);
      if (!match) {
        throw new Error(`--rewired-seeds must be START:END (got "${value}")`);
      }
      const start = Number(match[1]);
      const end = Number(match[2]);
      if (end <= start) {
        throw new Error(`--rewired-seeds end must be greater than start (got "${value}")`);
      }
      rewiredSeeds = { start, end };
      index += 2;
    } else if (flag === '--arena-task') {
      arenaTask = parseArenaTaskArg(requireValue(flag, argv[index + 1]));
      index += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  // `--graph-list` is mutually exclusive with `--rewired-index`/`--graphs-dir`
  // (`03-evaluation.md`'s own wording) — checked before either "one is
  // required" branch below, so a caller who passes all three gets this
  // specific message rather than sailing past the missing-flag checks.
  if (graphList !== undefined && (rewiredIndex !== undefined || graphsDir !== undefined)) {
    throw new Error('--graph-list is mutually exclusive with --rewired-index/--graphs-dir');
  }
  if (graphList === undefined) {
    if (!rewiredIndex) throw new Error('--rewired-index is required (or use --graph-list)');
    if (!graphsDir) throw new Error('--graphs-dir is required (or use --graph-list)');
  } else if (rewiredSeeds !== undefined) {
    // `--rewired-seeds` only means something against a `rewire_batch.py`
    // seed-indexed index (it narrows `index.seeds` to a numeric range) --
    // `--graph-list`'s entries have no numeric seed to narrow by, and
    // `buildGraphListTasks` never reads it, so every entry would silently
    // be scored anyway. This file already rejects the same class of silent
    // no-op for `--graph` without `--biological` just above (a dual-review
    // finding).
    throw new Error('--rewired-seeds only applies to --rewired-index mode (not --graph-list)');
  }
  // `graph` is only ever read when `biological` is set (see `runNullEvaluate`)
  // — silently accepting it otherwise would let an operator believe an
  // override took effect when it didn't (a dual-review finding).
  if (graph !== undefined && !biological) throw new Error('--graph requires --biological');
  // `runNullEvaluate` derives its run-meta sidecar path from `out` by
  // replacing a trailing ".json" — enforced here so that never silently
  // degrades into overwriting `out` itself (see `runMetaPathFor`'s doc
  // comment, a dual-review finding).
  if (!out.endsWith('.json')) throw new Error(`--out must end with ".json" (got "${out}")`);

  return {
    biological,
    graph,
    rewiredIndex,
    graphsDir,
    graphList,
    heldOutStart,
    heldOutCount,
    ticks,
    shards,
    out,
    decoder,
    rewiredSeeds,
    arenaTask
  };
};

// ---------------------------------------------------------------------------
// Task list
// ---------------------------------------------------------------------------

/**
 * `sortedRewireSeeds(index)` narrowed to `args.rewiredSeeds`'s half-open
 * `[start, end)` range when given (see `NullEvaluateArgs.rewiredSeeds`'s doc
 * comment), otherwise every seed in the index — unchanged from before the
 * flag existed. Throws if any seed in the requested range is missing from
 * the index (a typo'd range, or a range wider than the index actually
 * contains, must fail loudly rather than silently scoring fewer seeds than
 * requested -- a dual-review finding: an earlier version let an empty or
 * partial match through silently, which for an empty match also drove
 * `runCliMain`'s per-episode-ms summary to `Infinity`/`NaN`).
 */
const selectedRewireSeeds = (
  index: Readonly<RewireIndex>,
  args: Readonly<NullEvaluateArgs>
): readonly RewireIndexSeedEntry[] => {
  const seeds = sortedRewireSeeds(index);
  if (!args.rewiredSeeds) return seeds;
  const { start, end } = args.rewiredSeeds;
  const selected = seeds.filter((entry) => entry.seed >= start && entry.seed < end);
  const expectedCount = end - start;
  if (selected.length !== expectedCount) {
    // Sample-and-count rather than materializing every missing seed: a
    // pathological range (e.g. a typo'd `--rewired-seeds 0:99999999999`)
    // must fail fast, not hang or exhaust memory scanning billions of
    // integers (a reviewer finding). `missingCount` is exact (arithmetic,
    // not scan-dependent); the listed sample is capped and may be partial.
    const foundSeeds = new Set(selected.map((entry) => entry.seed));
    const missingCount = expectedCount - selected.length;
    const SAMPLE_LIMIT = 10;
    const SCAN_LIMIT = 1_000_000;
    const sample: number[] = [];
    for (let seed = start; seed < end && sample.length < SAMPLE_LIMIT && seed - start < SCAN_LIMIT; seed += 1) {
      if (!foundSeeds.has(seed)) sample.push(seed);
    }
    const sampleText = sample.length < missingCount ? `${sample.join(', ')}, ...` : sample.join(', ');
    throw new Error(
      `null-evaluate: --rewired-seeds ${start}:${end} requested ${expectedCount} seed(s), but the index is ` +
        `missing ${missingCount}: ${sampleText}`
    );
  }
  return selected;
};

/** `Task.heldOutSeeds` is identical for every task in a run — built once from `args`, shared by both `buildTasks` (`--rewired-index`) and `buildGraphListTasks` (`--graph-list`). */
const heldOutSeedsFor = (args: Readonly<NullEvaluateArgs>): number[] =>
  Array.from({ length: args.heldOutCount }, (_, i) => args.heldOutStart + i);

/** Biological+disconnected tasks (shared shape between `buildTasks` and `buildGraphListTasks`; both modes support `--biological` the same way). */
const biologicalAndDisconnectedTasks = (
  biologicalPath: string,
  sourceSha256: string,
  heldOutSeeds: readonly number[],
  args: Readonly<NullEvaluateArgs>
): NullWorkerTask[] => {
  const commonBio = {
    path: biologicalPath,
    expectedSha256: sourceSha256,
    heldOutSeeds,
    ticks: args.ticks,
    decoder: args.decoder,
    arenaTask: args.arenaTask
  };
  return [
    { graphId: 'biological', mode: 'biological' as NullTaskMode, ...commonBio },
    { graphId: 'disconnected', mode: 'disconnected' as NullTaskMode, ...commonBio }
  ];
};

export const buildTasks = (
  index: Readonly<RewireIndex>,
  args: Readonly<NullEvaluateArgs>,
  biologicalPath: string
): NullWorkerTask[] => {
  const heldOutSeeds = heldOutSeedsFor(args);
  const tasks: NullWorkerTask[] = [];

  if (args.biological) {
    tasks.push(...biologicalAndDisconnectedTasks(biologicalPath, index.sourceSha256, heldOutSeeds, args));
  }

  // `buildTasks` is only ever called from `runNullEvaluate`'s
  // `--rewired-index` path, which validates `args.graphsDir` is defined
  // before calling this (`parseNullEvaluateArgs` requires it when
  // `--graph-list` is absent) -- this guard protects a hand-built `args`
  // object (every existing test constructs one directly) from a confusing
  // `resolve(undefined, ...)` throw instead of this actionable message.
  if (args.graphsDir === undefined) {
    throw new Error('null-evaluate: buildTasks requires args.graphsDir (--rewired-index mode)');
  }
  const graphsDir = args.graphsDir;

  for (const entry of selectedRewireSeeds(index, args)) {
    tasks.push({
      graphId: `rewired-${entry.seed}`,
      mode: 'rewired' as NullTaskMode,
      path: resolve(graphsDir, entry.artifact),
      expectedSha256: entry.binarySha256,
      heldOutSeeds,
      ticks: args.ticks,
      decoder: args.decoder,
      arenaTask: args.arenaTask
    });
  }
  return tasks;
};

/**
 * `--graph-list` mode's task builder (`03-evaluation.md`'s WP2: "Accepts a
 * task with an explicit `graphId` and path, which the existing verified-load
 * path already supports"). Every graph-list entry becomes a `mode: 'rewired'`
 * task — from `null-worker.ts`'s point of view, a P/Q/C000/M1000/MQ2000
 * graph binary is loaded and parsed exactly like a `rewire_batch.py` rewired
 * seed's; the only thing that differs is where the task list and its
 * `graphId`s come from. `graphId` is the entry's own `id` (e.g. `"P"`,
 * `"C007"`), not a synthesized `rewired-${n}` label, so `assembleGraphListRaw`
 * can key its output by that same `id`.
 */
export const buildGraphListTasks = (
  index: Readonly<GraphListIndex>,
  args: Readonly<NullEvaluateArgs>,
  indexDir: string,
  biologicalPath: string
): NullWorkerTask[] => {
  const heldOutSeeds = heldOutSeedsFor(args);
  const tasks: NullWorkerTask[] = [];

  if (args.biological) {
    tasks.push(...biologicalAndDisconnectedTasks(biologicalPath, index.sourceSha256, heldOutSeeds, args));
  }

  for (const entry of sortedGraphListEntries(index)) {
    tasks.push({
      graphId: entry.id,
      mode: 'rewired' as NullTaskMode,
      path: resolve(indexDir, entry.path),
      expectedSha256: entry.binarySha256,
      heldOutSeeds,
      ticks: args.ticks,
      decoder: args.decoder,
      arenaTask: args.arenaTask
    });
  }
  return tasks;
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
export interface NullEvaluationRaw extends ArenaTaskOutputFields {
  readonly version: 1;
  readonly sourceGraphSha256: string;
  readonly rewireSourceSha256: string;
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
  /**
   * The left-agent decoder every task in this run used (`--decoder`,
   * default `'authored'`). `null-report.ts` reads this to pick the
   * published artifact's `condition` label and to enforce that a
   * non-authored run is never written to a shipped path (see that file's
   * `runNullReport`). Absent on any `authored.json` produced before this
   * field existed — every reader treats a missing value as `'authored'`.
   */
  readonly decoder: NullDecoderKind;
  readonly biological?: NullGraphRaw;
  readonly disconnected?: NullGraphRaw;
  readonly rewired: readonly NullRewiredGraphRaw[];
  readonly host: { readonly arch: string; readonly node: string };
}

/**
 * Reshape one task's raw per-seed worker results into its `NullGraphRaw`
 * output shape. Shared with `null-trained-evaluate.ts` (WP3), whose own
 * `NullTrainedGraphRaw` is a type alias for `NullGraphRaw` (the two scripts'
 * per-graph output shape is identical — `heldOutSeeds`/`movementScore`/
 * `foodPickups`/`hazardContacts` — only the *enclosing* raw-evaluation shape
 * differs), rather than each redeclaring an identical function (a
 * thermo-maintainability review finding).
 */
export const toGraphRaw = (results: readonly NullSeedResult[]): NullGraphRaw => ({
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

  const rewired: NullRewiredGraphRaw[] = selectedRewireSeeds(index, args).map((entry) => ({
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
    decoder: args.decoder,
    ...(args.biological
      ? { biological: toGraphRaw(require('biological')), disconnected: toGraphRaw(require('disconnected')) }
      : {}),
    rewired,
    host: { arch: process.arch, node: process.version },
    ...arenaTaskOutputFields(args.arenaTask)
  };
};

/** `--graph-list` mode's per-graph output entry, keyed by the graph-list entry's own `id` (not a seed). */
export interface NullGraphListGraphRaw extends NullGraphRaw {
  readonly id: string;
  readonly gzipSha256: string;
  readonly binarySha256: string;
}

/**
 * `--graph-list` mode's raw output shape. Deliberately its own interface
 * rather than folded into `NullEvaluationRaw` (whose `rewired` field is
 * keyed by numeric `seed`, per `rewire_batch.py`'s index): this mode's
 * graphs are keyed by an arbitrary string `id` (`"P"`, `"C007"`, ...), and
 * `03-evaluation.md`'s WP2 acceptance criterion is exactly "results keyed by
 * id, sorted" — `graphs` below is always `sortedGraphListEntries`-ordered, so
 * this shape is independent of shard count or completion timing the same way
 * `NullEvaluationRaw.rewired` is. No `rewireSourceSha256` field: that
 * describes `rewire_batch.py`'s own provenance, which has no counterpart
 * here (`scripts/analysis/interventions.py`'s equivalent producer-identity
 * hash lives in `index.json`'s own `producer.sourceSha256`, read and
 * verified by `scripts/analysis/interventions.py`'s callers, not by this
 * evaluator).
 */
export interface NullGraphListEvaluationRaw extends ArenaTaskOutputFields {
  readonly version: 1;
  readonly sourceGraphSha256: string;
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
  readonly decoder: NullDecoderKind;
  readonly biological?: NullGraphRaw;
  readonly disconnected?: NullGraphRaw;
  /** Sorted by `id` ascending — see `sortedGraphListEntries`. */
  readonly graphs: readonly NullGraphListGraphRaw[];
  readonly host: { readonly arch: string; readonly node: string };
  /**
   * `git rev-parse HEAD` at the moment this run executed, `null` on any
   * failure (not a git checkout, `git` missing) — additive (WP4 of
   * `.agents/plans/pathway-interventions`, thermo-methodology review I2):
   * this `--graph-list` mode previously stamped no code-identity at all,
   * unlike `null-trained-evaluate.ts`'s own `evaluatorGitRev`, which is
   * exactly the gap `intervention-artifact.ts`'s own `producer.sourceSha256`
   * (a build-time snapshot of the dependency files on disk, not a stamp
   * from the moment the 30,300-episode authored run itself executed) cannot
   * substitute for. Scoped to `--graph-list` mode only — `NullEvaluationRaw`
   * (`--rewired-index` mode, `rewiring-null-v1.json`'s own producer) is
   * deliberately left unchanged, so no already-published artifact's shape
   * or bytes are affected by this addition.
   */
  readonly evaluatorGitRev: string | null;
}

export const assembleGraphListRaw = (
  index: Readonly<GraphListIndex>,
  args: Readonly<NullEvaluateArgs>,
  results: ReadonlyMap<string, readonly NullSeedResult[]>
): NullGraphListEvaluationRaw => {
  const require = (graphId: string): readonly NullSeedResult[] => {
    const found = results.get(graphId);
    if (!found) throw new Error(`null-evaluate: missing results for "${graphId}"`);
    return found;
  };

  const graphs: NullGraphListGraphRaw[] = sortedGraphListEntries(index).map((entry) => ({
    id: entry.id,
    gzipSha256: entry.gzipSha256,
    binarySha256: entry.binarySha256,
    ...toGraphRaw(require(entry.id))
  }));

  return {
    version: 1,
    sourceGraphSha256: index.sourceSha256,
    seeds: { start: args.heldOutStart, count: args.heldOutCount },
    ticks: args.ticks,
    substeps: NEURAL_SUBSTEPS_PER_TICK,
    decoder: args.decoder,
    ...(args.biological
      ? { biological: toGraphRaw(require('biological')), disconnected: toGraphRaw(require('disconnected')) }
      : {}),
    graphs,
    host: { arch: process.arch, node: process.version },
    evaluatorGitRev: gitRev(repoRoot),
    ...arenaTaskOutputFields(args.arenaTask)
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
 * so the very next `atomicWriteFileSync` below would silently overwrite the
 * multi-hour `authored.json` this function just wrote with the tiny
 * run-meta sidecar. `--out` (both scripts' own, identically-named, flag) is
 * validated to end in `.json` at parse time (`parseNullEvaluateArgs`/
 * `parseNullTrainedEvaluateArgs`), but this function stays self-checking
 * for any other caller (a test, a future
 * script) that might not go through either CLI. `source` names the calling
 * script (`"null-evaluate"`/`"null-trained-evaluate"`) so the thrown message
 * still identifies which one raised it -- shared between the two rather
 * than each redeclaring an identical function (a thermo-maintainability
 * review finding).
 */
export const runMetaPathFor = (source: string, outPath: string): string => {
  if (!outPath.endsWith('.json')) {
    throw new Error(`${source}: expected a ".json" output path, got "${outPath}"`);
  }
  return `${outPath.slice(0, -'.json'.length)}.run.json`;
};

/**
 * `DEFAULT_OUT` (`training/runs/null/authored.json`) is the canonical,
 * hours-long, full-index authored run -- not shipped, but still the single
 * input every other WP1/WP2/WP3 script and the reproduction gate itself
 * reads by default. A non-canonical run (a decoder variant, or a
 * `--rewired-seeds`-restricted subset such as the reproduction gate's own
 * `--rewired-seeds 0:5 --decoder authored`) must never silently overwrite it
 * just because the caller forgot an explicit `--out` -- a dual-review
 * finding: the plan's own reproduction-gate example command omits `--out`.
 * Checked before any file is read or any shard forked.
 */
const guardCanonicalOutDefault = (args: Readonly<NullEvaluateArgs>): void => {
  const isNonCanonical =
    args.decoder !== 'authored' || args.rewiredSeeds !== undefined || args.graphList !== undefined || args.arenaTask !== undefined;
  if (isNonCanonical && resolve(args.out) === resolve(DEFAULT_OUT)) {
    throw new Error(
      `null-evaluate: refusing to write a non-canonical run (decoder="${args.decoder}"` +
        `${args.rewiredSeeds ? `, --rewired-seeds ${args.rewiredSeeds.start}:${args.rewiredSeeds.end}` : ''}` +
        `${args.graphList ? `, --graph-list ${args.graphList}` : ''}` +
        `${args.arenaTask ? `, --arena-task ${args.arenaTask}` : ''}) ` +
        `to the default --out (${DEFAULT_OUT}) -- that path is the canonical full-index authored run every other ` +
        'script reads by default. Pass an explicit --out for this run.'
    );
  }
};

/**
 * Shared write tail for both `--rewired-index` and `--graph-list` modes:
 * atomically write `raw` to `args.out`, then its `<out>.run.json` sidecar
 * (shard count + wall time — deliberately excluded from `raw` itself, see
 * `NullEvaluationRaw`'s doc comment). Both writes use the same atomic
 * temp-file-plus-rename treatment so a process killed mid-write (OOM,
 * external `kill -9`, host preemption) never leaves a truncated, multi-hour
 * result with no partial-recovery path.
 */
const writeEvaluationOutput = (
  args: Readonly<NullEvaluateArgs>,
  raw: unknown,
  elapsedMs: number,
  perEpisodeMs: number
): { out: string; runMetaOut: string } => {
  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, JSON.stringify(raw));

  const runMetaOut = runMetaPathFor('null-evaluate', args.out);
  atomicWriteFileSync(runMetaOut, `${JSON.stringify({ shards: args.shards, elapsedMs, perEpisodeMs }, null, 2)}\n`);

  return { out: args.out, runMetaOut };
};

/**
 * The shared run shape both `--rewired-index` and `--graph-list` modes
 * reduce to, once each has read and verified its own index format
 * (a thermo-maintainability review finding: round 1 asked for exactly this
 * extraction and only `writeEvaluationOutput` was actually shared then).
 * `buildTasksFor`/`assemble` are the only two things that differ between the
 * modes -- everything else (biological path resolution/verification, the
 * worker path, timed sharded execution, the atomic write) is identical
 * control flow, parameterized here rather than duplicated.
 */
const runEvaluationMode = async (
  args: Readonly<NullEvaluateArgs>,
  sourceArtifact: string,
  sourceSha256: string,
  buildTasksFor: (biologicalPath: string) => NullWorkerTask[],
  assemble: (results: ReadonlyMap<string, readonly NullSeedResult[]>) => unknown
): Promise<{ out: string; runMetaOut: string; taskCount: number; elapsedMs: number }> => {
  const biologicalPath = args.biological ? args.graph ?? resolve(PUBLIC_DATA_DIR, sourceArtifact) : '';
  if (args.biological) verifyBiologicalSource('null-evaluate', biologicalPath, sourceSha256);

  const tasks = buildTasksFor(biologicalPath);
  const workerPath = fileURLToPath(new URL('./null-worker.ts', import.meta.url));

  const started = performance.now();
  const results = await runShardedEvaluation<NullWorkerTask, NullSeedResult, NullWorkerMessage>(
    tasks,
    args.shards,
    workerPath
  );
  const elapsedMs = performance.now() - started;
  const perEpisodeMs = elapsedMs / (tasks.length * args.heldOutCount);

  const raw = assemble(results);
  const { out, runMetaOut } = writeEvaluationOutput(args, raw, elapsedMs, perEpisodeMs);
  return { out, runMetaOut, taskCount: tasks.length, elapsedMs };
};

/**
 * `--graph-list` mode (`03-evaluation.md`'s WP2): score every entry in
 * `scripts/analysis/interventions.py`'s `index.json` (P, Q, and the
 * C/M/MQ control arms) instead of a `rewire_batch.py` seed-indexed batch.
 * Reads and verifies this mode's own index format, then hands off to
 * `runEvaluationMode` for the shared run shape -- the only differences from
 * `--rewired-index` mode are the index shape, the task-id source (`entry.id`
 * vs. `rewired-${seed}`), and the output shape (`NullGraphListEvaluationRaw`,
 * keyed by `id`), all captured in `buildGraphListTasks`/`assembleGraphListRaw`.
 */
const runNullEvaluateGraphList = async (
  args: Readonly<NullEvaluateArgs>,
  graphList: string
): Promise<{ out: string; runMetaOut: string; taskCount: number; elapsedMs: number }> => {
  const index = readGraphListIndex(graphList);
  const indexDir = dirname(graphList);
  verifyGraphListFiles(index, indexDir);
  return runEvaluationMode(
    args,
    index.sourceArtifact,
    index.sourceSha256,
    (biologicalPath) => buildGraphListTasks(index, args, indexDir, biologicalPath),
    (results) => assembleGraphListRaw(index, args, results)
  );
};

export const runNullEvaluate = async (
  args: Readonly<NullEvaluateArgs>
): Promise<{ out: string; runMetaOut: string; taskCount: number; elapsedMs: number }> => {
  guardCanonicalOutDefault(args);

  // `parseNullEvaluateArgs` already enforces `--graph-list`'s mutual
  // exclusion with `--rewired-index`/`--graphs-dir` and that one of the two
  // modes is present -- but `runNullEvaluate` is also called directly (every
  // existing test builds its own `NullEvaluateArgs` object literal, bypassing
  // the CLI parser), so this function re-checks rather than trusting that
  // callers always route through `parseNullEvaluateArgs` first.
  const graphList = args.graphList;
  if (graphList !== undefined) {
    if (args.rewiredIndex !== undefined || args.graphsDir !== undefined) {
      throw new Error('null-evaluate: --graph-list is mutually exclusive with --rewired-index/--graphs-dir');
    }
    if (args.rewiredSeeds !== undefined) {
      throw new Error('null-evaluate: --rewired-seeds only applies to --rewired-index mode (not --graph-list)');
    }
    return runNullEvaluateGraphList(args, graphList);
  }
  if (args.rewiredIndex === undefined || args.graphsDir === undefined) {
    throw new Error('null-evaluate: --rewired-index and --graphs-dir are required (or use --graph-list)');
  }

  const index = readRewireIndex(args.rewiredIndex);
  verifyRewiredFiles(index, args.graphsDir);
  return runEvaluationMode(
    args,
    index.sourceArtifact,
    index.sourceSha256,
    (biologicalPath) => buildTasks(index, args, biologicalPath),
    (results) => assembleRaw(index, args, results)
  );
};

/** Every `runNullEvaluate`/`runNullTrainedEvaluate`-shaped CLI driver's return value: what `runCliMain` needs to log its summary line. */
export interface CliRunResult {
  readonly out: string;
  readonly runMetaOut: string;
  readonly taskCount: number;
  readonly elapsedMs: number;
}

/**
 * `main()`'s shared shape: parse argv, await `run`, log a one-line summary
 * (`console.log`) on success, or log the error and `process.exit(1)` on
 * failure — never throwing back out to the caller. `null-evaluate.ts` and
 * `null-trained-evaluate.ts` previously hand-wrote near-identical copies of
 * this (same try/catch/console.log/console.error/`process.exit(1)` shape,
 * differing only in which functions they called and the log wording -- a
 * thermo-maintainability review finding); both now build their own `main`
 * from this generic instead. `taskNoun` fills in the one wording difference
 * ("graphs" for the authored null, "runs" for the trained sample) so the
 * summary line still reads naturally for each script.
 */
export const runCliMain = <Args extends { readonly heldOutCount: number; readonly shards: number }>(
  scriptName: string,
  taskNoun: string,
  parseArgs: (argv: readonly string[]) => Args,
  run: (args: Readonly<Args>) => Promise<CliRunResult>
): (() => Promise<void>) => {
  return async () => {
    try {
      const args = parseArgs(process.argv.slice(2));
      const { out, runMetaOut, taskCount, elapsedMs } = await run(args);
      const totalEpisodes = taskCount * args.heldOutCount;
      const perEpisodeMs = elapsedMs / totalEpisodes;
      // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
      console.log(
        `${scriptName}: wrote ${out} and ${runMetaOut} (${taskCount} ${taskNoun} x ${args.heldOutCount} seeds = ` +
          `${totalEpisodes} episodes) in ${(elapsedMs / 1000).toFixed(1)}s (${perEpisodeMs.toFixed(1)} ms/episode, ` +
          `${args.shards} shards)`
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
      console.error(`${scriptName} failed: ${message}`);
      process.exit(1);
    }
  };
};

const main = runCliMain('null-evaluate', 'graphs', parseNullEvaluateArgs, runNullEvaluate);

if (process.argv[1] === fileURLToPath(import.meta.url)) void main();
