import { fork } from 'node:child_process';
import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import {
  NULL_DECODER_KINDS,
  type NullDecoderKind,
  type NullSeedResult,
  type NullTaskMode,
  type NullWorkerMessage,
  type NullWorkerTask
} from './null-worker';

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
 * `index.seeds` sorted by numeric seed ascending — the single source of
 * this module's "canonical task order" invariant (see the module doc
 * comment above). `buildTasks` and `assembleRaw` both need this exact
 * order for the same reason (byte-identical output independent of shard
 * count/timing); previously each independently re-sorted, which let the
 * two copies drift out of agreement by convention alone. Never a sort over
 * `graphId` strings, which would put `rewired-10` before `rewired-2`.
 */
const sortedRewireSeeds = (index: Readonly<RewireIndex>): readonly RewireIndexSeedEntry[] =>
  [...index.seeds].sort((a, b) => a.seed - b.seed);

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

/**
 * Verify the biological source graph's decompressed sha256 against
 * `index.json`'s own `sourceSha256`. Exported (a thermo-maintainability
 * review finding): `regime-check.ts` (WP2) needs the exact same check and
 * previously carried a hand-duplicated copy because this was private.
 * `source` is the calling CLI's own name (`"null-evaluate"`/
 * `"regime-check"`), matching `null-worker-shared.ts`'s
 * `assertFiniteScores`/`loadVerifiedGraphBinary` `source`-prefixed-message
 * convention, so the thrown message still identifies which CLI raised it.
 */
export const verifyBiologicalSource = (source: string, path: string, expectedSha256: string): void => {
  const gzipBytes = readFileSync(path);
  const binary = gunzipSync(gzipBytes);
  const actual = sha256Hex(binary);
  if (actual !== expectedSha256) {
    throw new Error(`${source}: ${path} decompressed sha256 ${actual} does not match index.json's sourceSha256 (${expectedSha256})`);
  }
};

// ---------------------------------------------------------------------------
// scripts/analysis/interventions.py index.json (--graph-list mode)
// ---------------------------------------------------------------------------

/**
 * `.agents/plans/pathway-interventions/03-evaluation.md`'s WP2 `--graph-list`
 * mode: one entry per predeclared intervention/control graph
 * (`scripts/analysis/interventions.py`'s `index.json` — P, Q, C000..C099,
 * M1000..M1099, MQ2000..MQ2099). The plan states the entry contract as
 * `{ id, path, gzipSha256 }`; `binarySha256` is additionally required here
 * (present on every real `interventions.py` entry) so this mode gets the
 * same two-layer verification `--rewired-index` mode already has: the gzip
 * bytes checked up front by `verifyGraphListFiles` (mirroring
 * `verifyRewiredFiles`), and the *decompressed* bytes independently
 * re-checked inside the worker by `loadVerifiedGraphBinary` (mirroring every
 * `RewireIndexSeedEntry` task's `expectedSha256`) — never trusting a single
 * check across the parent/worker process boundary.
 */
export interface GraphListEntry {
  readonly id: string;
  /** Path to the gzip-compressed graph binary, relative to the graph-list index.json's own directory. */
  readonly path: string;
  readonly gzipSha256: string;
  /** Expected sha256 of the *decompressed* binary — see this interface's doc comment. */
  readonly binarySha256: string;
}

export interface GraphListIndex {
  readonly sourceArtifact: string;
  readonly sourceSha256: string;
  readonly entries: readonly GraphListEntry[];
}

/**
 * `buildGraphListTasks` synthesizes tasks with `graphId: 'biological'`/
 * `'disconnected'` when `--biological` is set (see `biologicalAndDisconnectedTasks`).
 * A graph-list entry using either id would collide with those tasks:
 * `runShardedEvaluation` keeps whichever result arrives first for a given
 * `graphId` and silently discards the other, so which graph's scores end up
 * in the output (and in the reproduction check `intervention-report.ts`
 * depends on) would depend on shard completion timing — the exact
 * determinism failure `readGraphListIndex`'s duplicate-id check exists to
 * prevent, just via a different id source (a dual-review finding).
 */
const RESERVED_GRAPH_IDS: ReadonlySet<string> = new Set(['biological', 'disconnected']);

/**
 * Parse and lightly validate `scripts/analysis/interventions.py`'s
 * `index.json`. Deliberately tolerant of extra fields beyond the ones this
 * script reads (`kind`, `swaps`, `targetReached`, `transfer`, `producer`,
 * `kP`/`kQ`/`maxSwaps`/`controlCount` are all recorded by `interventions.py`
 * but not consumed here) — the same tolerance `readRewireIndex` already
 * applies to `rewire_batch.py`'s own index.json.
 */
export const readGraphListIndex = (path: string): GraphListIndex => {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as Partial<GraphListIndex>;
  if (typeof parsed.sourceArtifact !== 'string' || typeof parsed.sourceSha256 !== 'string') {
    throw new Error(`null-evaluate: ${path} is missing sourceArtifact/sourceSha256`);
  }
  if (!Array.isArray(parsed.entries) || parsed.entries.length === 0) {
    throw new Error(`null-evaluate: ${path} has no entries`);
  }
  const seenIds = new Set<string>();
  for (const entry of parsed.entries) {
    if (
      typeof entry.id !== 'string' ||
      entry.id.length === 0 ||
      typeof entry.path !== 'string' ||
      typeof entry.gzipSha256 !== 'string' ||
      typeof entry.binarySha256 !== 'string'
    ) {
      throw new Error(`null-evaluate: ${path} has a malformed graph-list entry: ${JSON.stringify(entry)}`);
    }
    if (RESERVED_GRAPH_IDS.has(entry.id)) {
      throw new Error(
        `null-evaluate: ${path} uses reserved graph id "${entry.id}" (reserved for --biological's own tasks)`
      );
    }
    // A path escaping index.json's own directory (an absolute path, or a
    // "../" traversal) would break this mode's documented contract ("every
    // entry's path is relative to the graph-list index.json's own
    // directory" — see `GraphListEntry.path`'s doc comment) and let a
    // mis-generated index silently read a file outside the run's own graph
    // set. Content integrity is still independently enforced by both sha
    // layers either way, so this is a fail-fast/contract check, not the
    // primary integrity guard.
    if (entry.path.startsWith('/') || entry.path.split('/').includes('..')) {
      throw new Error(
        `null-evaluate: ${path} entry "${entry.id}" has a path outside index.json's own directory: "${entry.path}"`
      );
    }
    // Two tasks sharing the same graphId would let whichever result arrives
    // last silently win (the same reasoning as readRewireIndex's duplicate-
    // seed rejection above) — reject it here rather than at result-assembly
    // time.
    if (seenIds.has(entry.id)) {
      throw new Error(`null-evaluate: ${path} lists graph id "${entry.id}" more than once`);
    }
    seenIds.add(entry.id);
  }
  return parsed as GraphListIndex;
};

/**
 * `entries` sorted by `id` ascending (plain string comparison) — the single
 * source of this mode's "results keyed by id, sorted" output guarantee
 * (`03-evaluation.md`'s own wording), matching `sortedRewireSeeds`'s role
 * for `--rewired-index` mode. Never a sort over collected results or
 * completion order.
 */
export const sortedGraphListEntries = (index: Readonly<GraphListIndex>): readonly GraphListEntry[] =>
  [...index.entries].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));

/**
 * Verify every graph-list file's raw gzip bytes against `index.json` before
 * any shard is forked — mirrors `verifyRewiredFiles`'s reasoning exactly
 * (fail in seconds on a corrupted/stale/partial copy, not hours into a run).
 * `indexDir` is the graph-list index.json's own directory: every entry's
 * `path` is relative to it (e.g. `"graphs/P.bin.gz"`), not to `--graphs-dir`
 * (there is no separate `--graphs-dir` in this mode).
 */
export const verifyGraphListFiles = (index: Readonly<GraphListIndex>, indexDir: string): void => {
  const mismatches: string[] = [];
  for (const entry of index.entries) {
    const path = resolve(indexDir, entry.path);
    let bytes: Buffer;
    try {
      bytes = readFileSync(path);
    } catch (error) {
      mismatches.push(`id ${entry.id}: cannot read ${path} (${error instanceof Error ? error.message : String(error)})`);
      continue;
    }
    const actual = sha256Hex(bytes);
    if (actual !== entry.gzipSha256) {
      mismatches.push(`id ${entry.id}: ${path} gzip sha256 ${actual} does not match index.json (${entry.gzipSha256})`);
    }
  }
  if (mismatches.length > 0) {
    throw new Error(`null-evaluate: ${mismatches.length} graph-list file(s) failed verification:\n${mismatches.join('\n')}`);
  }
};

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
    rewiredSeeds
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
    decoder: args.decoder
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
      decoder: args.decoder
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
      decoder: args.decoder
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
 * Fork `shardCount` copies of a worker script (`null-worker.ts` for this
 * module's own callers; `null-trained-evaluate.ts` reuses this same
 * function with `null-trained-worker.ts` — see that file), hand each one
 * tasks one at a time (a worker that finishes gets the next queued task, so
 * a slow biological/disconnected task never blocks idle shards), and
 * collect every task's raw results keyed by `graphId`. Deterministic
 * regardless of which shard executes which task or in what order: the
 * caller reassembles output by iterating `tasks` (the canonical, seed-sorted
 * order), not by collection order.
 *
 * Generic over the task/result/message shapes so `null-trained-evaluate.ts`
 * (WP3) can reuse this exact sharding/failure-handling mechanism against its
 * own worker protocol (weights-bearing tasks, not gzip-graph-bearing ones)
 * without a second, drifting copy of it — the only structural requirements
 * are that every task carries a `graphId` and the `heldOutSeeds` it was
 * assigned, and every result carries the `seed` it was scored on, so this
 * function can still verify a worker's reply matches what it was asked to
 * do (see the self-checking protocol below). Every call site pins all three
 * type parameters explicitly (TypeScript can't infer `Result`/`Message` from
 * `tasks` alone, since neither appears in an argument position, and a
 * default for `Message` can't itself reference `Result`'s default — TS
 * checks default type-argument expressions against the *unsubstituted*
 * constraint, not other parameters' defaults). This module's own
 * `runNullEvaluate` pins `<NullWorkerTask, NullSeedResult, NullWorkerMessage>`;
 * `null-trained-evaluate.ts` (WP3) pins its own equivalent types.
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
export const runShardedEvaluation = async <
  Task extends { readonly graphId: string; readonly heldOutSeeds: readonly number[] },
  Result extends { readonly seed: number },
  Message extends
    | { readonly type: 'result'; readonly graphId: string; readonly results: readonly Result[] }
    | { readonly type: 'error'; readonly graphId: string; readonly message: string }
>(
  tasks: readonly Task[],
  shardCount: number,
  workerPath: string
): Promise<Map<string, readonly Result[]>> => {
  const results = new Map<string, readonly Result[]>();
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
      let inFlight: Task | undefined;

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

      child.on('message', (message: Message) => {
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
    host: { arch: process.arch, node: process.version }
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
export interface NullGraphListEvaluationRaw {
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
  const isNonCanonical = args.decoder !== 'authored' || args.rewiredSeeds !== undefined || args.graphList !== undefined;
  if (isNonCanonical && resolve(args.out) === resolve(DEFAULT_OUT)) {
    throw new Error(
      `null-evaluate: refusing to write a non-canonical run (decoder="${args.decoder}"` +
        `${args.rewiredSeeds ? `, --rewired-seeds ${args.rewiredSeeds.start}:${args.rewiredSeeds.end}` : ''}` +
        `${args.graphList ? `, --graph-list ${args.graphList}` : ''}) ` +
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
 * `--graph-list` mode (`03-evaluation.md`'s WP2): score every entry in
 * `scripts/analysis/interventions.py`'s `index.json` (P, Q, and the
 * C/M/MQ control arms) instead of a `rewire_batch.py` seed-indexed batch.
 * Mirrors the `--rewired-index` path in `runNullEvaluate` below step for
 * step (verify-before-fork, sharded evaluation, atomic write) — the only
 * differences are the index shape, the task-id source (`entry.id` vs.
 * `rewired-${seed}`), and the output shape (`NullGraphListEvaluationRaw`,
 * keyed by `id`).
 */
const runNullEvaluateGraphList = async (
  args: Readonly<NullEvaluateArgs>,
  graphList: string
): Promise<{ out: string; runMetaOut: string; taskCount: number; elapsedMs: number }> => {
  const index = readGraphListIndex(graphList);
  const indexDir = dirname(graphList);
  verifyGraphListFiles(index, indexDir);

  const biologicalPath = args.biological ? args.graph ?? resolve(PUBLIC_DATA_DIR, index.sourceArtifact) : '';
  if (args.biological) verifyBiologicalSource('null-evaluate', biologicalPath, index.sourceSha256);

  const tasks = buildGraphListTasks(index, args, indexDir, biologicalPath);
  const workerPath = fileURLToPath(new URL('./null-worker.ts', import.meta.url));

  const started = performance.now();
  const results = await runShardedEvaluation<NullWorkerTask, NullSeedResult, NullWorkerMessage>(
    tasks,
    args.shards,
    workerPath
  );
  const elapsedMs = performance.now() - started;
  const perEpisodeMs = elapsedMs / (tasks.length * args.heldOutCount);

  const raw = assembleGraphListRaw(index, args, results);
  const { out, runMetaOut } = writeEvaluationOutput(args, raw, elapsedMs, perEpisodeMs);
  return { out, runMetaOut, taskCount: tasks.length, elapsedMs };
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

  const biologicalPath = args.biological ? args.graph ?? resolve(PUBLIC_DATA_DIR, index.sourceArtifact) : '';
  if (args.biological) verifyBiologicalSource('null-evaluate', biologicalPath, index.sourceSha256);

  const tasks = buildTasks(index, args, biologicalPath);
  const workerPath = fileURLToPath(new URL('./null-worker.ts', import.meta.url));

  const started = performance.now();
  const results = await runShardedEvaluation<NullWorkerTask, NullSeedResult, NullWorkerMessage>(
    tasks,
    args.shards,
    workerPath
  );
  const elapsedMs = performance.now() - started;
  const perEpisodeMs = elapsedMs / (tasks.length * args.heldOutCount);

  const raw = assembleRaw(index, args, results);
  const { out, runMetaOut } = writeEvaluationOutput(args, raw, elapsedMs, perEpisodeMs);
  return { out, runMetaOut, taskCount: tasks.length, elapsedMs };
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
