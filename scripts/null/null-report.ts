import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { conditionRng, pairedStats, type ConditionStats, type PairedStats } from '../training/stats';
import {
  DEFAULT_HISTOGRAM_BINS,
  DEGENERATE_IQR_THRESHOLD,
  buildHistogram,
  graphStats,
  nullSummary,
  rankStatistics,
  type Histogram,
  type NullSummary
} from './null-stats';
import type { NullEvaluationRaw, NullGraphRaw } from './null-evaluate';

/**
 * `.agents/plans/rewiring-null/02-authored-null-evaluation.md`'s
 * `null-report.ts`: the cheap, pure-statistics half of WP2. Reads
 * `null-evaluate.ts`'s raw per-seed `authored.json` (never re-simulates
 * anything), computes every predeclared statistic via `null-stats.ts`, and
 * writes `public/data/rewiring-null-v1.json`, the `rewiringNull` manifest
 * key, and `docs/rewiring-null-report.md`. Runs twice against the same
 * `authored.json` in a row (`npm run null:report` then again) must produce
 * byte-identical `rewiring-null-v1.json` — nothing here reads the clock or
 * any other non-deterministic source; the `host` field is copied straight
 * through from `authored.json` (recorded once, at evaluation time), not
 * recomputed here.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

const DEFAULT_AUTHORED = resolve(repoRoot, 'training/runs/null/authored.json');
const DEFAULT_TRAINED = resolve(repoRoot, 'training/runs/null/trained.json');
/** Exported so tests can exercise `guardShippedDefault`'s exact-path match without duplicating this resolution logic. */
export const DEFAULT_OUT = resolve(repoRoot, 'public/data/rewiring-null-v1.json');
export const DEFAULT_REPORT_MD = resolve(repoRoot, 'docs/rewiring-null-report.md');
export const DEFAULT_MANIFEST = resolve(repoRoot, 'public/data/malecns-arena-v1.manifest.json');

/** 'N','U','L','L' as a fixed default seed; arbitrary but stable across runs, matching `evaluate.ts`'s `DEFAULT_BOOTSTRAP_SEED` convention. */
const DEFAULT_BOOTSTRAP_SEED = 0x4e554c4c;
const DEFAULT_BOOTSTRAP_RESAMPLES = 10000;

/** This study's committed methodology (the plan's "500 rewired graphs"): the floor below which `--out`/`--report-md`/`--manifest` refuse to write to their default (shipped) paths, so a dev/fixture/partial run can never silently clobber the real published artifacts. */
const MIN_REWIRED_FOR_SHIPPED_DEFAULT = 500;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface NullReportArgs {
  readonly authored: string;
  readonly trained: string;
  readonly out: string;
  readonly reportMd: string;
  readonly manifest: string;
  readonly bootstrapSeed: number;
  readonly bootstrapResamples: number;
  readonly histogramBins: number;
  /**
   * Explicit override for the published artifact's `shards` field. Left
   * `undefined` (the default — no flag passed), `resolveRunMeta` reads it
   * from `null-evaluate.ts`'s `<authored>.run.json` sidecar instead, so the
   * published value reflects what the real run actually used rather than a
   * guessed default. See that function and `RunMeta`'s doc comment.
   */
  readonly shards?: number;
}

export const parseNullReportArgs = (argv: readonly string[]): NullReportArgs => {
  let authored = DEFAULT_AUTHORED;
  let trained = DEFAULT_TRAINED;
  let out = DEFAULT_OUT;
  let reportMd = DEFAULT_REPORT_MD;
  let manifest = DEFAULT_MANIFEST;
  let bootstrapSeed = DEFAULT_BOOTSTRAP_SEED;
  let bootstrapResamples = DEFAULT_BOOTSTRAP_RESAMPLES;
  let histogramBins = DEFAULT_HISTOGRAM_BINS;
  let shards: number | undefined;

  let index = 0;
  while (index < argv.length) {
    const flag = argv[index];
    if (flag === '--authored') {
      authored = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--trained') {
      trained = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--out') {
      out = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--report-md') {
      reportMd = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--manifest') {
      manifest = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--bootstrap-seed') {
      bootstrapSeed = requireNonNegativeInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--bootstrap-resamples') {
      bootstrapResamples = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--histogram-bins') {
      histogramBins = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--shards') {
      shards = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  // `resolveRunMeta` derives its sidecar path from `authored` by replacing a
  // trailing ".json" — enforced here for the same reason `null-evaluate.ts`
  // enforces it on `--out` (a dual-review finding).
  if (!authored.endsWith('.json')) throw new Error(`--authored must end with ".json" (got "${authored}")`);

  return { authored, trained, out, reportMd, manifest, bootstrapSeed, bootstrapResamples, histogramBins, shards };
};

// ---------------------------------------------------------------------------
// Run metadata (shard count, wall time) — the operational counterpart to
// authored.json's deterministic raw data
// ---------------------------------------------------------------------------

/**
 * `null-evaluate.ts` writes `<out>.run.json` (e.g. `authored.run.json`)
 * alongside `authored.json`, since `authored.json` itself deliberately
 * excludes shard count and wall time (operational parameters of *that run*,
 * not properties of the data — see `NullEvaluationRaw`'s doc comment in
 * `null-evaluate.ts`). This is that sidecar's shape.
 */
export interface RunMeta {
  readonly shards: number;
  readonly elapsedMs?: number;
  readonly perEpisodeMs?: number;
}

/**
 * See `null-evaluate.ts`'s identically-named function's doc comment for why
 * this throws instead of using a regex `.replace` that silently no-ops (and
 * so would read the sidecar from `authoredPath` itself) when `authoredPath`
 * doesn't end in `.json`. `--authored` is validated to end in `.json` at
 * parse time; this stays self-checking for any other caller.
 */
const runMetaPathFor = (authoredPath: string): string => {
  if (!authoredPath.endsWith('.json')) {
    throw new Error(`null-report: expected a ".json" authored path, got "${authoredPath}"`);
  }
  return `${authoredPath.slice(0, -'.json'.length)}.run.json`;
};

/**
 * Resolve the published artifact's `shards`/timing fields. `--shards`
 * explicitly overrides the sidecar's shard count (for a caller that knows
 * better, or is regenerating a report for a run whose sidecar was lost);
 * otherwise the sidecar is required — there is no silent "assume 18"
 * fallback, since a wrong-but-plausible-looking shard count would be
 * unverifiable provenance in the published artifact (a dual-review finding:
 * an earlier version defaulted to 18 unconditionally).
 */
export const resolveRunMeta = (args: Readonly<NullReportArgs>): RunMeta => {
  const sidecarPath = runMetaPathFor(args.authored);
  const sidecar: Partial<RunMeta> = existsSync(sidecarPath)
    ? (JSON.parse(readFileSync(sidecarPath, 'utf8')) as Partial<RunMeta>)
    : {};

  if (args.shards !== undefined) {
    return { shards: args.shards, elapsedMs: sidecar.elapsedMs, perEpisodeMs: sidecar.perEpisodeMs };
  }
  if (typeof sidecar.shards !== 'number') {
    throw new Error(
      `null-report: cannot determine shard count -- pass --shards explicitly, or ensure ${sidecarPath} exists ` +
        'with a numeric "shards" field (written by null-evaluate.ts next to its authored.json output)'
    );
  }
  return { shards: sidecar.shards, elapsedMs: sidecar.elapsedMs, perEpisodeMs: sidecar.perEpisodeMs };
};

// ---------------------------------------------------------------------------
// Artifact shape
// ---------------------------------------------------------------------------

export interface ScoredEntry {
  readonly score: number;
  readonly median: number;
  readonly std: number;
  readonly ci: readonly [number, number];
}

export interface RewiredEntry extends ScoredEntry {
  readonly seed: number;
  readonly gzipSha256: string;
  readonly acceptedSwaps: number;
  readonly attempts: number;
}

export interface RewiringNullArtifact {
  readonly version: 1;
  readonly condition: 'authored, opponent parked';
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
  readonly sourceGraphSha256: string;
  readonly rewireSourceSha256: string;
  readonly shards: number;
  readonly biological: ScoredEntry;
  readonly disconnected: ScoredEntry;
  /** Sorted by seed ascending. */
  readonly rewired: readonly RewiredEntry[];
  readonly null: NullSummary;
  readonly bioPercentile: number;
  readonly pLow: number;
  readonly pHigh: number;
  readonly bins: Histogram;
  /** Biological minus rewired-seed-0, same held-out seeds — reconciles with `docs/seed-sweep.md`. */
  readonly pairedBiologicalVsRewiredSeed0: PairedStats;
  readonly bootstrap: { readonly resamples: number; readonly seed: number };
  readonly host: { readonly arch: string; readonly node: string };
  /** Present only when `null-evaluate.ts`'s `.run.json` sidecar recorded it. */
  readonly timing?: { readonly elapsedMs: number; readonly perEpisodeMs: number };
}

const toScoredEntry = (stats: ConditionStats): ScoredEntry => ({
  score: stats.mean,
  median: stats.median,
  std: stats.std,
  ci: stats.ci95
});

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/** Recursively sort object keys, matching `scripts/data/compile.py`'s `json.dumps(..., sort_keys=True)` convention this manifest is otherwise written with. */
const sortKeysDeep = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(sortKeysDeep);
  if (value !== null && typeof value === 'object') {
    const sorted: Record<string, unknown> = {};
    for (const key of Object.keys(value as Record<string, unknown>).sort()) {
      sorted[key] = sortKeysDeep((value as Record<string, unknown>)[key]);
    }
    return sorted;
  }
  return value;
};

/**
 * Add/overwrite the manifest's `rewiringNull` key in place, preserving
 * every other byte-derived field (`compilerSourceSha256`, the graph/positions
 * hashes, `rewiredArms`) untouched — same additive pattern the `positions`
 * key uses (`scripts/data/positions.py`). Re-serialized with sorted keys and
 * 2-space indent, matching the file's existing (Python-written)
 * `json.dumps(..., indent=2, sort_keys=True)` format.
 *
 * That byte-for-byte match is verified, not assumed: this only holds
 * because the manifest currently has no float fields (Python writes `1.0`/
 * `1e-05`, JS writes `1`/`0.00001`) and no non-ASCII text (Python's default
 * `ensure_ascii=True` escapes it, JS does not) — a dual-review finding. If
 * either ever changes, re-serializing the *unmodified* manifest would
 * silently rewrite unrelated bytes. Guarded below by re-serializing the
 * manifest before touching it and refusing to proceed if that round trip
 * isn't byte-identical to the file on disk.
 *
 * That check is exposed separately (`verifyManifestRoundTrips`) so
 * `runNullReport` can run it as a preflight, before the artifact or report
 * are written — a dual-review pass caught that running it only inside this
 * function (called last, after the artifact write) left a half-published
 * state on failure: a new `rewiring-null-v1.json` on disk with no manifest
 * entry pointing at it.
 */
export const verifyManifestRoundTrips = (manifestPath: string): void => {
  const originalText = readFileSync(manifestPath, 'utf8');
  const manifest = JSON.parse(originalText) as Record<string, unknown>;
  const roundTripped = `${JSON.stringify(sortKeysDeep(manifest), null, 2)}\n`;
  if (roundTripped !== originalText) {
    throw new Error(
      `null-report: re-serializing ${manifestPath} without any change produced different bytes than the ` +
        "file on disk (likely a float or non-ASCII field JS formats differently than Python's json.dumps) " +
        '-- refusing to write, to avoid silently rewriting unrelated manifest bytes'
    );
  }
};

/**
 * Does **not** call `verifyManifestRoundTrips` itself — `runNullReport`
 * already runs it as a preflight (before anything is written, so a failure
 * never leaves a half-published state; see that function's own call and
 * `verifyManifestRoundTrips`'s doc comment above). Calling it again here
 * was pure duplicated work on every successful publish (a thermo-nuclear
 * maintainability finding). Any other caller of this function must run
 * `verifyManifestRoundTrips(manifestPath)` itself first if it isn't already
 * guaranteed to hold — this function trusts that contract rather than
 * re-verifying it.
 */
export const updateManifestWithRewiringNull = (
  manifestPath: string,
  entry: { readonly artifact: string; readonly sha256: string }
): void => {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
  manifest.rewiringNull = entry;
  atomicWriteFileSync(manifestPath, `${JSON.stringify(sortKeysDeep(manifest), null, 2)}\n`);
};

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

export interface RunNullReportResult {
  readonly out: string;
  readonly reportMdPath: string;
  readonly artifactSha256: string;
  readonly artifact: RewiringNullArtifact;
}

/** `a` and `b` name the same held-out seeds in the same order. */
const sameSeeds = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((seed, i) => seed === b[i]);

export const buildArtifact = (
  raw: Readonly<NullEvaluationRaw>,
  args: Readonly<NullReportArgs>,
  runMeta: Readonly<RunMeta>
): RewiringNullArtifact => {
  if (raw.version !== 1) {
    throw new Error(`null-report: ${args.authored} has unsupported version ${String(raw.version)}, expected 1`);
  }
  if (!raw.biological || !raw.disconnected) {
    throw new Error(
      `null-report: ${args.authored} has no biological/disconnected section ` +
        '(run null-evaluate with --biological for the real report)'
    );
  }

  // Every graph must have been scored on the same held-out seeds, in the
  // same order — `pairedStats` below pairs biological against rewired-seed-0
  // by array index, and the null set otherwise mixes scores from seed lists
  // that were never actually the same episodes. `null-evaluate.ts` always
  // builds every task's `heldOutSeeds` from one shared array, so this holds
  // by construction for its own output; this check protects a hand-merged
  // or hand-edited `authored.json` from silently producing a nonsensical
  // paired comparison (a dual-review finding).
  const referenceSeeds = raw.biological.heldOutSeeds;
  if (!sameSeeds(raw.disconnected.heldOutSeeds, referenceSeeds)) {
    throw new Error(`null-report: ${args.authored}: disconnected was scored on different held-out seeds than biological`);
  }
  for (const entry of raw.rewired) {
    if (!sameSeeds(entry.heldOutSeeds, referenceSeeds)) {
      throw new Error(
        `null-report: ${args.authored}: rewired seed ${entry.seed} was scored on different held-out seeds than biological`
      );
    }
  }

  // `null-worker.ts` already refuses to produce a non-finite movementScore/
  // foodPickups/hazardContacts, but that only protects a *fresh*
  // `null-evaluate.ts` run — `authored.json` is a plain file on disk that
  // could be hand-edited or merged from an older/buggy evaluator. Without
  // this, a `null`/NaN score would round-trip through `JSON.stringify` as
  // `null`, then get silently summed as `0` in every statistic below (a
  // dual-review finding).
  const assertFiniteScores = (label: string, graph: Readonly<NullGraphRaw>): void => {
    const arrays: readonly (readonly [string, readonly number[]])[] = [
      ['movementScore', graph.movementScore],
      ['foodPickups', graph.foodPickups],
      ['hazardContacts', graph.hazardContacts]
    ];
    for (const [field, values] of arrays) {
      const badIndex = values.findIndex((value) => typeof value !== 'number' || !Number.isFinite(value));
      if (badIndex !== -1) {
        throw new Error(`null-report: ${args.authored}: ${label}.${field}[${badIndex}] is not a finite number`);
      }
    }
  };
  assertFiniteScores('biological', raw.biological);
  assertFiniteScores('disconnected', raw.disconnected);
  for (const entry of raw.rewired) assertFiniteScores(`rewired-${entry.seed}`, entry);

  const { bootstrapSeed, bootstrapResamples } = args;
  const biologicalStats = graphStats(raw.biological.movementScore, bootstrapSeed, 'biological', bootstrapResamples);
  const disconnectedStats = graphStats(raw.disconnected.movementScore, bootstrapSeed, 'disconnected', bootstrapResamples);

  const rewiredStats = raw.rewired.map((entry) => ({
    entry,
    stats: graphStats(entry.movementScore, bootstrapSeed, `rewired-${entry.seed}`, bootstrapResamples)
  }));

  const nullValues = rewiredStats.map(({ stats }) => stats.mean);
  const summary = nullSummary(nullValues);
  const rank = rankStatistics(nullValues, biologicalStats.mean);

  const seed0 = raw.rewired.find((entry) => entry.seed === 0);
  if (!seed0) {
    const seeds = raw.rewired.map((entry) => entry.seed).sort((a, b) => a - b);
    throw new Error(
      `null-report: rewired seed 0 is missing from ${args.authored} ` +
        `(found ${seeds.length} seed(s): ${seeds[0]}..${seeds[seeds.length - 1]}) -- the paired comparison needs ` +
        'the shipped control arm; rerun rewire_batch.py/null-evaluate with a seed range that includes 0'
    );
  }
  const paired = pairedStats(
    raw.biological.movementScore,
    seed0.movementScore,
    bootstrapResamples,
    conditionRng(bootstrapSeed, 'paired|biological-vs-rewired-seed0')
  );

  // Bin *edges* span the union with biological/disconnected (so both land
  // inside the plotted range even if they're outliers relative to N), but
  // bin *counts* are the null set N alone: a dual-review pass caught that
  // counting biological/disconnected into the bars too made
  // `sum(bins.counts) === 502` instead of 500, so the published "null
  // distribution of 500 rewired graphs" histogram silently included two
  // extra bars for graphs that were never part of the rewiring null.
  const unionForRange = [...nullValues, biologicalStats.mean, disconnectedStats.mean];
  const bins = buildHistogram(nullValues, args.histogramBins, [Math.min(...unionForRange), Math.max(...unionForRange)]);

  return {
    version: 1,
    condition: 'authored, opponent parked',
    seeds: raw.seeds,
    ticks: raw.ticks,
    substeps: raw.substeps,
    sourceGraphSha256: raw.sourceGraphSha256,
    rewireSourceSha256: raw.rewireSourceSha256,
    shards: runMeta.shards,
    biological: toScoredEntry(biologicalStats),
    disconnected: toScoredEntry(disconnectedStats),
    rewired: rewiredStats.map(({ entry, stats }) => ({
      seed: entry.seed,
      gzipSha256: entry.gzipSha256,
      acceptedSwaps: entry.acceptedSwaps,
      attempts: entry.attempts,
      ...toScoredEntry(stats)
    })),
    null: summary,
    bioPercentile: rank.bioPercentile,
    pLow: rank.pLow,
    pHigh: rank.pHigh,
    bins,
    pairedBiologicalVsRewiredSeed0: paired,
    bootstrap: { resamples: bootstrapResamples, seed: bootstrapSeed },
    host: raw.host,
    ...(runMeta.elapsedMs !== undefined && runMeta.perEpisodeMs !== undefined
      ? { timing: { elapsedMs: runMeta.elapsedMs, perEpisodeMs: runMeta.perEpisodeMs } }
      : {})
  };
};

// ---------------------------------------------------------------------------
// Markdown report
// ---------------------------------------------------------------------------

const fmt = (value: number, digits = 4): string => value.toFixed(digits);
const pct = (value: number): string => `${(value * 100).toFixed(1)}%`;

/**
 * Describes the rewired seed set from the data itself rather than assuming
 * `0..n-1` — `null-evaluate.ts` sorts `rewired` by seed but never requires
 * it to be a contiguous range starting at 0 (a partial or extended batch,
 * e.g. seeds `0..249` plus `300..549`, is possible). A dual-review pass
 * caught the report text asserting `0..n-1` unconditionally.
 */
const rewiredSeedRangeText = (rewired: readonly RewiredEntry[]): string => {
  const seeds = [...rewired.map((entry) => entry.seed)].sort((a, b) => a - b);
  const contiguousFromZero = seeds.every((seed, i) => seed === i);
  if (contiguousFromZero) return `0..${seeds.length - 1}`;
  return `${seeds.length} seeds, ${seeds[0]}..${seeds[seeds.length - 1]} (not necessarily contiguous)`;
};

const renderHistogramTable = (bins: Histogram): string => {
  const maxCount = Math.max(...bins.counts, 1);
  const barWidth = 40;
  const rows = bins.counts.map((count, i) => {
    const lo = bins.edges[i];
    const hi = bins.edges[i + 1];
    const bar = '#'.repeat(Math.max(0, Math.round((count / maxCount) * barWidth)));
    return `| ${fmt(lo, 3)} to ${fmt(hi, 3)} | ${count} | ${bar} |`;
  });
  return ['| range | count | |', '| --- | --- | --- |', ...rows].join('\n');
};

export const renderReportMarkdown = (artifact: Readonly<RewiringNullArtifact>): string => {
  const seed0 = artifact.rewired.find((entry) => entry.seed === 0);
  const seedRangeText = rewiredSeedRangeText(artifact.rewired);
  const timingRow = artifact.timing
    ? `| Wall time | ${(artifact.timing.elapsedMs / 1000).toFixed(1)}s |\n| Per-episode time | ${artifact.timing.perEpisodeMs.toFixed(1)} ms |\n`
    : '';

  const degenerateSection = artifact.null.degenerate
    ? `\n> **Degenerate null.** This null distribution's IQR (${artifact.null.iqr.toExponential(3)}) is below the ` +
      `${DEGENERATE_IQR_THRESHOLD.toExponential(0)} threshold used to flag a degenerate result. Under this evaluation ` +
      `setup (authored decoder, opponent parked, T=${artifact.ticks}), the authored path is essentially insensitive ` +
      `to rewiring topology: rewired-graph scores cluster too tightly to place the biological graph meaningfully. ` +
      'This is a finding about this specific condition, not a general claim about the MaleCNS connectome.\n'
    : '';

  const markdown = `# Rewiring null — authored-decoder evaluation

Where the measured biological MaleCNS topology falls among ${artifact.rewired.length} degree-preserving
rewirings of the same graph, scored under identical dynamics, encoder, decoder, and held-out seeds.

**What "authored decoder" means.** The authored decoder is a fixed, hand-written mapping from
output-neuron rates to motor actions (\`src/lib/arena/actions.ts\`'s \`decodeAction\`, invoked via
\`scripts/training/episode.ts\`'s \`aggregateOutputs -> decodeAction\` pipeline) — it is not biological,
not trained, and not derived from the connectome beyond reading rates off anatomically-labeled output
neurons (see \`docs/model-ledger.md\`, which labels this **Authored** and separately states the product
"must not... imply that authored behavior is biological"). This
report describes how this specific hand-authored decoder, this rate-model dynamics, and this arena
interact with the biological graph's topology versus ${artifact.rewired.length} rewirings of it — it is
not a claim about the real fly's neural function or behavior, and it makes no claim that any rewiring's
topology is causally "worse," or that any other rewiring is "better," than the biological one.
${degenerateSection}
## Method

Every graph below (biological, disconnected, and each of the ${artifact.rewired.length} rewired graphs)
is scored by \`scripts/training/episode.ts\`'s \`runEpisode\`: the authored decoder drives the left agent,
the right agent is parked (always the zero action), over the same ${artifact.seeds.count} held-out seeds
(\`${artifact.seeds.start}..${artifact.seeds.start + artifact.seeds.count - 1}\`), \`T = ${artifact.ticks}\`
ticks, \`K = ${artifact.substeps}\` neural substeps per tick. Degree-preserving rewiring comes only from
\`scripts/data/rewire.py\`'s \`rewire_graph\` (rewiring seeds \`${seedRangeText}\`, seed 0
being the shipped control arm). A graph's score is the mean \`movementScore\` over its held-out seeds; the
null set \`N\` is the ${artifact.rewired.length} rewired graphs' mean scores.

## Parameters

| Parameter | Value |
| --- | --- |
| Condition | ${artifact.condition} |
| Held-out seeds | ${artifact.seeds.start}..${artifact.seeds.start + artifact.seeds.count - 1} (n=${artifact.seeds.count}) |
| Ticks (T) | ${artifact.ticks} |
| Substeps (K) | ${artifact.substeps} |
| Rewired graphs | ${artifact.rewired.length} (seeds ${seedRangeText}) |
| Evaluation shards | ${artifact.shards} |
${timingRow}| Bootstrap resamples | ${artifact.bootstrap.resamples} |
| Bootstrap seed | ${artifact.bootstrap.seed} |
| Histogram bins | ${artifact.bins.counts.length} |
| Source graph sha256 | \`${artifact.sourceGraphSha256}\` |
| Rewiring source sha256 | \`${artifact.rewireSourceSha256}\` |
| Host | ${artifact.host.arch}, Node ${artifact.host.node} |

## Histogram

Null distribution of graph scores (${artifact.rewired.length} rewired graphs), ${artifact.bins.counts.length}
equal-width bins over \`[min(N ∪ {biological, disconnected}), max(N ∪ {biological, disconnected})]\`:

${renderHistogramTable(artifact.bins)}

## Results

| Quantity | Value |
| --- | --- |
| Biological score (95% CI) | ${fmt(artifact.biological.score)} (${fmt(artifact.biological.ci[0])}, ${fmt(artifact.biological.ci[1])}) |
| Disconnected score (95% CI) | ${fmt(artifact.disconnected.score)} (${fmt(artifact.disconnected.ci[0])}, ${fmt(artifact.disconnected.ci[1])}) |
| Rewired-seed-0 score (95% CI) | ${seed0 ? `${fmt(seed0.score)} (${fmt(seed0.ci[0])}, ${fmt(seed0.ci[1])})` : 'n/a'} |
| Null mean | ${fmt(artifact.null.mean)} |
| Null median | ${fmt(artifact.null.median)} |
| Null std | ${fmt(artifact.null.std)} |
| Null 2.5–97.5% | ${fmt(artifact.null.p2_5)} .. ${fmt(artifact.null.p97_5)} |
| Null IQR | ${fmt(artifact.null.iqr)} |
| Biological empirical percentile in null | ${pct(artifact.bioPercentile)} |
| Rank statistic p_low | ${fmt(artifact.pLow, 4)} |
| Rank statistic p_high | ${fmt(artifact.pHigh, 4)} |
| Paired biological − rewired-seed-0 (mean diff, 95% CI) | ${fmt(artifact.pairedBiologicalVsRewiredSeed0.meanDifference)} (${fmt(artifact.pairedBiologicalVsRewiredSeed0.ci95[0])}, ${fmt(artifact.pairedBiologicalVsRewiredSeed0.ci95[1])}) |

\`p_low\`/\`p_high\` are rank-based descriptive statistics \`(k_below + k_equal + 1)/(|N| + 1)\` and
\`(|N| - k_below + 1)/(|N| + 1)\` — reported without significance language, per this study's non-goals.

**Reconciling with [\`docs/seed-sweep.md\`](seed-sweep.md).** That study reports a 20-seed, two-agent, self-paired
sweep (\`T=2700\`) where rewired-seed-0 outscored biological (mean 0.70 vs. −2.04). This report's paired
biological − rewired-seed-0 comparison above, run under this study's different condition (single-agent,
opponent parked, 100 seeds, \`T=${artifact.ticks}\`), agrees in direction — biological scores lower here too
(mean diff ${fmt(artifact.pairedBiologicalVsRewiredSeed0.meanDifference)}, 95% CI ${fmt(artifact.pairedBiologicalVsRewiredSeed0.ci95[0])} to ${fmt(artifact.pairedBiologicalVsRewiredSeed0.ci95[1])}) —
but the two studies differ in agent/opponent condition, tick count, and seed count (and seed set), so this
is corroborating evidence under a related-but-distinct condition, not a replication of the same measurement.

## Limitations

- Scores come from a **single-agent condition with the opponent parked**, on the same held-out seeds and
  tick count as [the trained-readout report](trained-readout-report.md) — but that report evaluates a
  different condition: it scores trained readouts (and a \`silenced\`-readout control) alongside the
  authored decoder, restricted to the biological/rewired-seed-0/disconnected arms, not a null
  distribution over ${artifact.rewired.length} degree-preserving rewirings. Two-agent competitive
  dynamics are not evaluated in either report.
- **One null model** is used: degree-preserving double-edge swaps (\`scripts/data/rewire.py\`). Other null
  models (weight shuffles within degree, Erdős–Rényi with matched density) are deferred follow-ups.
- **No causal or superiority claim is made.** The percentile and rank statistics above are descriptive: they
  say where the biological graph's score falls among this null model's rewirings under this exact evaluation
  setup, not that biological topology causes or predicts any particular score.
${artifact.null.degenerate ? '- The null distribution is **degenerate** (IQR below threshold) — see the note above.\n' : ''}`;

  // A non-degenerate report otherwise ends with a trailing blank line (the
  // template's own newline before the closing backtick, doubled up with the
  // conditional bullet's own newline) — trimmed here rather than reworking
  // every interpolation site, so the markdown always ends with exactly one
  // newline regardless of which optional sections are present.
  return `${markdown.trimEnd()}\n`;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

/**
 * Refuse to write to a default (shipped) path unless the input looks like a
 * real run — `rewiredCount >= MIN_REWIRED_FOR_SHIPPED_DEFAULT`. Mirrors
 * `scripts/training/evaluate.ts`'s trace-graph-mode `--out`/`--report-md`
 * guards: a dev/fixture/partial run must not silently clobber a shipped
 * artifact just because the caller forgot an explicit `--out`.
 */
const guardShippedDefault = (path: string, defaultPath: string, label: string, rewiredCount: number): void => {
  if (resolve(path) !== resolve(defaultPath)) return;
  if (rewiredCount >= MIN_REWIRED_FOR_SHIPPED_DEFAULT) return;
  throw new Error(
    `null-report: refusing to overwrite the shipped ${label} (${defaultPath}) -- the input has only ` +
      `${rewiredCount} rewired graph(s) (this study's committed methodology uses ` +
      `${MIN_REWIRED_FOR_SHIPPED_DEFAULT}). Pass an explicit --out/--report-md/--manifest scratch path for a ` +
      'dev/test run.'
  );
};

/**
 * The manifest being updated must describe the same biological graph (and,
 * where recorded, the same shipped rewired-seed-0 control) that
 * `authored.json` was actually scored against — otherwise a stale
 * `authored.json` (from before a graph recompile) would attach a
 * `rewiringNull` entry to a manifest describing a different graph, with no
 * warning (a dual-review finding). The seed-0 cross-check is best-effort:
 * `manifest.rewiredArms.seed0` is optional here (a scratch/dev manifest
 * fixture need not carry it) and skipped when absent.
 */
const verifySourceGraphMatchesManifest = (
  manifestPath: string,
  artifact: Readonly<Pick<RewiringNullArtifact, 'sourceGraphSha256' | 'rewired'>>,
  authoredPath: string
): void => {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as {
    binarySha256?: string;
    rewiredArms?: { seed0?: { gzipSha256?: string } };
  };
  if (manifest.binarySha256 !== artifact.sourceGraphSha256) {
    throw new Error(
      `null-report: ${authoredPath} was scored against a graph with sha256 ${artifact.sourceGraphSha256}, but ` +
        `${manifestPath}'s biological graph sha256 is ${String(manifest.binarySha256)} -- refusing to attach a ` +
        "rewiringNull entry that doesn't describe the manifest's own graph"
    );
  }
  const manifestSeed0GzipSha256 = manifest.rewiredArms?.seed0?.gzipSha256;
  const artifactSeed0 = artifact.rewired.find((entry) => entry.seed === 0);
  if (manifestSeed0GzipSha256 && artifactSeed0 && artifactSeed0.gzipSha256 !== manifestSeed0GzipSha256) {
    throw new Error(
      `null-report: rewired seed 0's gzip sha256 (${artifactSeed0.gzipSha256}) does not match ` +
        `${manifestPath}'s shipped rewiredArms.seed0 (${manifestSeed0GzipSha256}) -- the report claims seed 0 ` +
        'is "the shipped control arm", which would no longer be true'
    );
  }
};

export const runNullReport = (args: Readonly<NullReportArgs>): RunNullReportResult => {
  // --trained is plumbed for WP3 ("Reads authored.json (and trained.json
  // from WP3 if present)", 02-authored-null-evaluation.md) but WP3 isn't
  // implemented yet. Rather than silently ignoring a real file an operator
  // pointed --trained at (the flag was dead code otherwise — a dual-review
  // finding), fail loudly if one exists; the common case (no WP3 output
  // yet) hits neither branch.
  if (existsSync(args.trained)) {
    throw new Error(
      `null-report: ${args.trained} exists, but merging a trained-readout section is not implemented yet (WP3). ` +
        'Remove --trained or move/delete that file to publish the authored-only report.'
    );
  }

  const raw = JSON.parse(readFileSync(args.authored, 'utf8')) as NullEvaluationRaw;
  const runMeta = resolveRunMeta(args);
  const artifact = buildArtifact(raw, args, runMeta);

  guardShippedDefault(args.out, DEFAULT_OUT, 'published artifact', artifact.rewired.length);
  guardShippedDefault(args.reportMd, DEFAULT_REPORT_MD, 'report', artifact.rewired.length);
  guardShippedDefault(args.manifest, DEFAULT_MANIFEST, 'manifest', artifact.rewired.length);
  verifySourceGraphMatchesManifest(args.manifest, artifact, args.authored);
  verifyManifestRoundTrips(args.manifest);

  // Rendered before any file is written: if markdown rendering ever threw
  // (a future template bug), a partial publish (artifact + manifest written,
  // report missing) is worse than failing before anything changes on disk.
  const artifactContents = JSON.stringify(artifact);
  const artifactSha256 = sha256Hex(artifactContents);
  const reportMdContents = renderReportMarkdown(artifact);

  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, artifactContents);

  updateManifestWithRewiringNull(args.manifest, { artifact: basename(args.out), sha256: artifactSha256 });

  mkdirSync(dirname(args.reportMd), { recursive: true });
  atomicWriteFileSync(args.reportMd, reportMdContents);

  return { out: args.out, reportMdPath: args.reportMd, artifactSha256, artifact };
};

const main = (): void => {
  try {
    const args = parseNullReportArgs(process.argv.slice(2));
    if (!existsSync(args.authored)) {
      throw new Error(`${args.authored} does not exist; run "npm run null:evaluate" first`);
    }
    const result = runNullReport(args);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `null-report: wrote ${result.out} (sha256 ${result.artifactSha256}) and ${result.reportMdPath}\n` +
        `bioPercentile=${pct(result.artifact.bioPercentile)} null.degenerate=${result.artifact.null.degenerate}`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`null-report failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
