import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { conditionRng, mean, median, pairedStats, type PairedStats } from '../training/stats';
import { verifyManifestRoundTrips } from '../null/null-report';
import {
  DEFAULT_HELD_OUT_COUNT,
  DEFAULT_HELD_OUT_START,
  DEFAULT_TICKS,
  type AtlasEvaluationRaw,
  type AtlasGraphKey,
  type AtlasGraphRaw
} from './atlas-evaluate';

/**
 * `.agents/plans/lesion-atlas/02-atlas-computation.md`'s WP2 report: the
 * cheap, pure-statistics half. Reads `atlas-evaluate.ts`'s raw per-seed
 * `atlas-raw.json` (never re-simulates anything), computes the paired
 * lesion-effect statistic for every neuron of both graphs, and writes
 * `public/data/lesion-atlas-v1.json`, the `lesionAtlas` manifest key, and
 * `docs/lesion-atlas-report.md`. Follows `scripts/null/null-report.ts`'s
 * structure closely (per the plan's own "follows scripts/null/null-report.ts:61-104"),
 * reusing its `verifyManifestRoundTrips` directly (a pure manifest-format
 * utility with nothing null-specific in it) rather than re-copying it.
 *
 * Running this twice against the same `atlas-raw.json` must produce
 * byte-identical `lesion-atlas-v1.json`: nothing here reads the clock or
 * any other non-deterministic source, and every bootstrap draw comes from a
 * `conditionRng(bootstrapSeed, label)` stream keyed by a stable label, so
 * it does not depend on iteration order or on what else this invocation
 * computed.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

const DEFAULT_RAW = resolve(repoRoot, 'training/runs/lesion/atlas-raw.json');
export const DEFAULT_OUT = resolve(repoRoot, 'public/data/lesion-atlas-v1.json');
export const DEFAULT_REPORT_MD = resolve(repoRoot, 'docs/lesion-atlas-report.md');
export const DEFAULT_MANIFEST = resolve(repoRoot, 'public/data/malecns-arena-v1.manifest.json');
const DEFAULT_POSITIONS = resolve(repoRoot, 'public/data/malecns-arena-v1.positions.json');

/**
 * 'L','E','S','I' as a fixed default seed -- distinct from
 * `null-report.ts`'s `DEFAULT_BOOTSTRAP_SEED` ('N','U','L','L') so the two
 * studies' bootstrap streams are never accidentally identical, matching
 * that file's own "arbitrary but stable across runs" convention.
 */
const DEFAULT_BOOTSTRAP_SEED = 0x4c455349;
const DEFAULT_BOOTSTRAP_RESAMPLES = 10000;
/** Benjamini-Hochberg false discovery rate, per neuron, per graph (the plan's "q = 0.05 per graph (1,008 tests)"). */
const FDR_Q = 0.05;
/**
 * `pairedStats`' CI is a 95% CI (`scripts/training/stats.ts`'s `bootstrapCI`),
 * so 5% of per-neuron CIs are expected to exclude 0 by chance alone even
 * under a true null. Kept as its own constant, separate from `FDR_Q` above,
 * even though both happen to be 0.05 today -- they describe two different
 * things (the CI's nominal miss rate vs. the FDR procedure's target rate)
 * that could diverge if either is ever tuned independently (a dual-review
 * finding: `expectedChanceExclusions` was previously computed from `FDR_Q`).
 */
const CI_ALPHA = 0.05;
const TOP_EFFECTS_COUNT = 20;
/** "Numbers are rounded to 6 significant digits for size" (the plan's artifact-shape row). */
const SIGNIFICANT_DIGITS = 6;

// ---------------------------------------------------------------------------
// Rounding
// ---------------------------------------------------------------------------

/** Round `value` to `digits` significant digits. `0`/non-finite pass through unchanged (nothing to round). */
export const roundSignificant = (value: number, digits = SIGNIFICANT_DIGITS): number => {
  if (value === 0 || !Number.isFinite(value)) return value;
  const magnitude = Math.floor(Math.log10(Math.abs(value)));
  const factor = 10 ** (digits - 1 - magnitude);
  return Math.round(value * factor) / factor;
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface AtlasReportArgs {
  readonly raw: string;
  readonly out: string;
  readonly reportMd: string;
  readonly manifest: string;
  readonly positions: string;
  readonly bootstrapSeed: number;
  readonly bootstrapResamples: number;
  readonly shards?: number;
}

export const parseAtlasReportArgs = (argv: readonly string[]): AtlasReportArgs => {
  let raw = DEFAULT_RAW;
  let out = DEFAULT_OUT;
  let reportMd = DEFAULT_REPORT_MD;
  let manifest = DEFAULT_MANIFEST;
  let positions = DEFAULT_POSITIONS;
  let bootstrapSeed = DEFAULT_BOOTSTRAP_SEED;
  let bootstrapResamples = DEFAULT_BOOTSTRAP_RESAMPLES;
  let shards: number | undefined;

  let index = 0;
  while (index < argv.length) {
    const flag = argv[index];
    if (flag === '--raw') {
      raw = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
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
    } else if (flag === '--positions') {
      positions = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--bootstrap-seed') {
      bootstrapSeed = requireNonNegativeInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--bootstrap-resamples') {
      bootstrapResamples = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--shards') {
      shards = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (!raw.endsWith('.json')) throw new Error(`--raw must end with ".json" (got "${raw}")`);

  return { raw, out, reportMd, manifest, positions, bootstrapSeed, bootstrapResamples, shards };
};

// ---------------------------------------------------------------------------
// Run metadata (shard count, wall time)
// ---------------------------------------------------------------------------

export interface RunMeta {
  readonly shards: number;
  readonly elapsedMs?: number;
  readonly perEpisodeMs?: number;
}

const runMetaPathFor = (rawPath: string): string => {
  if (!rawPath.endsWith('.json')) {
    throw new Error(`atlas-report: expected a ".json" raw path, got "${rawPath}"`);
  }
  return `${rawPath.slice(0, -'.json'.length)}.run.json`;
};

export const resolveRunMeta = (args: Readonly<AtlasReportArgs>): RunMeta => {
  const sidecarPath = runMetaPathFor(args.raw);
  const sidecar: Partial<RunMeta> = existsSync(sidecarPath)
    ? (JSON.parse(readFileSync(sidecarPath, 'utf8')) as Partial<RunMeta>)
    : {};

  if (args.shards !== undefined) {
    return { shards: args.shards, elapsedMs: sidecar.elapsedMs, perEpisodeMs: sidecar.perEpisodeMs };
  }
  if (typeof sidecar.shards !== 'number') {
    throw new Error(
      `atlas-report: cannot determine shard count -- pass --shards explicitly, or ensure ${sidecarPath} exists ` +
        'with a numeric "shards" field (written by atlas-evaluate.ts next to its atlas-raw.json output)'
    );
  }
  return { shards: sidecar.shards, elapsedMs: sidecar.elapsedMs, perEpisodeMs: sidecar.perEpisodeMs };
};

// ---------------------------------------------------------------------------
// Positions sidecar (body IDs + roles, node-index-aligned with the
// biological graph; the node set is shared with rewired-seed0 -- WP2's own
// artifact-shape note)
// ---------------------------------------------------------------------------

interface PositionsShape {
  readonly bodyIds: readonly string[];
  readonly role: readonly ('sensory' | 'bridge' | 'descending')[];
  readonly graphSha256: string;
}

/**
 * `biologicalGraphGzipSha256` must be the *gzip* sha256 (`AtlasGraphRaw.graphGzipSha256`,
 * `ArenaManifestShape.gzipSha256`'s doc comment) -- `malecns-arena-v1.positions.json`'s
 * own `graphSha256` field is built against the compiled graph's gzip bytes
 * (`src/lib/experiment/assets.ts`'s `loadPositions`), not the decompressed
 * binary sha `AtlasGraphRaw.graphSha256` carries for provenance elsewhere.
 */
const readPositions = (path: string, neuronCount: number, biologicalGraphGzipSha256: string): PositionsShape => {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as Partial<PositionsShape>;
  if (!Array.isArray(parsed.bodyIds) || !Array.isArray(parsed.role) || typeof parsed.graphSha256 !== 'string') {
    throw new Error(`atlas-report: ${path} is missing bodyIds/role/graphSha256`);
  }
  if (parsed.bodyIds.length !== neuronCount || parsed.role.length !== neuronCount) {
    throw new Error(`atlas-report: ${path} bodyIds/role length does not match neuronCount ${neuronCount}`);
  }
  if (parsed.graphSha256 !== biologicalGraphGzipSha256) {
    throw new Error(
      `atlas-report: ${path} was built against graph gzip sha256 ${parsed.graphSha256}, but the raw evaluation's ` +
        `biological graph gzip sha256 is ${biologicalGraphGzipSha256} (stale positions artifact)`
    );
  }
  return parsed as PositionsShape;
};

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/**
 * The two-sided bootstrap p-value the plan calls for: "the two-sided
 * fraction of resampled mean differences on the opposite side of 0, times
 * 2, capped at 1". Deliberately its own resample loop with its own
 * `conditionRng` label (`${label}|pvalue`) rather than reusing
 * `pairedStats`'s internal draws (not exposed by `stats.ts`, which this WP
 * does not touch -- out of this WP's change surface): per-statistic
 * independent seeding is the same principle `conditionRng`'s own doc
 * comment already establishes for `conditionStats`/`pairedStats` --  this
 * p-value's draws are a pure function of `(bootstrapSeed, its own label,
 * its own data)`, independent of how many other statistics this run
 * computed or in what order.
 */
export const pairedBootstrapPValue = (
  a: readonly number[],
  b: readonly number[],
  resamples: number,
  rng: () => number,
  observedMeanDifference: number
): number => {
  if (observedMeanDifference === 0) return 1;
  const diffs = a.map((value, i) => value - b[i]);
  const n = diffs.length;
  const observedSign = Math.sign(observedMeanDifference);
  let oppositeSideCount = 0;
  for (let r = 0; r < resamples; r += 1) {
    let sum = 0;
    for (let i = 0; i < n; i += 1) sum += diffs[Math.floor(rng() * n)];
    const resampledMean = sum / n;
    if (Math.sign(resampledMean) !== 0 && Math.sign(resampledMean) !== observedSign) oppositeSideCount += 1;
    else if (resampledMean === 0) oppositeSideCount += 0.5; // a resample landing exactly on 0 is equally consistent with either side
  }
  return Math.min(1, (2 * oppositeSideCount) / resamples);
};

/** Benjamini-Hochberg FDR at `q`: the largest rank `k` with `p_(k) <= (k / m) * q` marks ranks `1..k` (by ascending p-value) significant. */
export const benjaminiHochbergSignificant = (pValues: readonly number[], q: number): readonly boolean[] => {
  const m = pValues.length;
  if (m === 0) return [];
  const order = pValues.map((p, i) => ({ p, i })).sort((a, b) => a.p - b.p);
  let maxSignificantRank = -1;
  for (let rankIndex = 0; rankIndex < m; rankIndex += 1) {
    const rank = rankIndex + 1;
    if (order[rankIndex].p <= (rank / m) * q) maxSignificantRank = rank;
  }
  const significant = new Array<boolean>(m).fill(false);
  for (let rankIndex = 0; rankIndex < maxSignificantRank; rankIndex += 1) significant[order[rankIndex].i] = true;
  return significant;
};

const percentile = (sortedAscending: readonly number[], p: number): number => {
  const n = sortedAscending.length;
  const index = Math.min(n - 1, Math.max(0, Math.round(p * (n - 1))));
  return sortedAscending[index];
};

// ---------------------------------------------------------------------------
// Artifact shape
// ---------------------------------------------------------------------------

export interface TopEffectEntry {
  readonly index: number;
  readonly bodyId: string;
  readonly role: 'sensory' | 'bridge' | 'descending';
  readonly effect: number;
  readonly ciLow: number;
  readonly ciHigh: number;
  readonly pValue: number;
  readonly fdrSignificant: boolean;
}

export interface AtlasGraphSummary {
  readonly medianEffect: number;
  readonly p5Effect: number;
  readonly p95Effect: number;
  /** Neurons whose 95% CI excludes 0 (naive, uncorrected). */
  readonly ciExcludesZeroCount: number;
  /** `CI_ALPHA * neuronCount`: the number of chance CI exclusions expected under the null even if every neuron's true effect were 0. */
  readonly expectedChanceExclusions: number;
  /** Neurons surviving Benjamini-Hochberg FDR at `q = 0.05`. */
  readonly fdrSignificantCount: number;
  /** Top 20 by |effect|, descending. */
  readonly topEffects: readonly TopEffectEntry[];
}

export interface AtlasGraphArtifact {
  /** The *decompressed binary* sha256 (manifest `binarySha256`/`rewiredArms.seed0.binarySha256`), not the gzip sha (`AtlasGraphRaw.graphGzipSha256`, which only the positions-sidecar cross-check uses). WP3's loader should compare this against the graph it parsed, the same field. */
  readonly graphSha256: string;
  readonly baseline: number;
  readonly effect: readonly number[];
  /**
   * `ciLow`/`ciHigh` bracket the paired bootstrap 95% CI. The plan's
   * statistics section also names a per-neuron `ciCrossesZero`, which is
   * intentionally not stored here as its own array (kept out of the
   * artifact shape to stay within the plan's size budget): a consumer
   * derives it as `ciLow[i] <= 0 && ciHigh[i] >= 0` from these two arrays --
   * safe because `roundSignificant` preserves sign and never rounds a
   * nonzero bound down to exactly 0 in this artifact's normal value range.
   */
  readonly ciLow: readonly number[];
  readonly ciHigh: readonly number[];
  readonly fdrSignificant: readonly boolean[];
  readonly summary: AtlasGraphSummary;
}

export interface LesionAtlasArtifact {
  readonly version: 1;
  readonly condition: 'authored, opponent parked, single-neuron lesion';
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
  readonly neuronCount: number;
  readonly bodyIds: readonly string[];
  readonly graphs: { readonly biological: AtlasGraphArtifact; readonly rewiredSeed0: AtlasGraphArtifact };
  readonly shards: number;
  readonly bootstrap: { readonly resamples: number; readonly seed: number };
  readonly host: { readonly arch: string; readonly node: string };
  readonly timing?: { readonly elapsedMs: number; readonly perEpisodeMs: number };
}

const buildGraphArtifact = (
  graphKey: AtlasGraphKey,
  raw: Readonly<AtlasGraphRaw>,
  positions: PositionsShape,
  bootstrapSeed: number,
  bootstrapResamples: number
): AtlasGraphArtifact => {
  const baselineMean = mean(raw.baselineMovementScore);
  const n = raw.lesion.length;

  const effect = new Array<number>(n);
  const ciLow = new Array<number>(n);
  const ciHigh = new Array<number>(n);
  const pValue = new Array<number>(n);

  for (let i = 0; i < n; i += 1) {
    const entry = raw.lesion[i];
    if (entry.index !== i) {
      throw new Error(`atlas-report: ${graphKey} lesion entries are not sorted/contiguous by index (expected ${i}, got ${entry.index})`);
    }
    const label = `${graphKey}|${entry.index}`;
    const stats: PairedStats = pairedStats(
      entry.movementScore,
      raw.baselineMovementScore,
      bootstrapResamples,
      conditionRng(bootstrapSeed, label)
    );
    effect[i] = stats.meanDifference;
    ciLow[i] = stats.ci95[0];
    ciHigh[i] = stats.ci95[1];
    pValue[i] = pairedBootstrapPValue(
      entry.movementScore,
      raw.baselineMovementScore,
      bootstrapResamples,
      conditionRng(bootstrapSeed, `${label}|pvalue`),
      stats.meanDifference
    );
  }

  const fdrSignificant = benjaminiHochbergSignificant(pValue, FDR_Q);
  const ciExcludesZeroCount = ciLow.filter((low, i) => low > 0 || ciHigh[i] < 0).length;
  const fdrSignificantCount = fdrSignificant.filter(Boolean).length;

  const sortedEffect = [...effect].sort((a, b) => a - b);
  const rankedByAbsEffect = effect
    .map((value, index) => ({ index, absEffect: Math.abs(value) }))
    .sort((a, b) => b.absEffect - a.absEffect)
    .slice(0, TOP_EFFECTS_COUNT);

  const topEffects: TopEffectEntry[] = rankedByAbsEffect.map(({ index }) => ({
    index,
    bodyId: positions.bodyIds[index],
    role: positions.role[index],
    effect: roundSignificant(effect[index]),
    ciLow: roundSignificant(ciLow[index]),
    ciHigh: roundSignificant(ciHigh[index]),
    pValue: roundSignificant(pValue[index]),
    fdrSignificant: fdrSignificant[index]
  }));

  return {
    graphSha256: raw.graphSha256,
    baseline: roundSignificant(baselineMean),
    effect: effect.map((v) => roundSignificant(v)),
    ciLow: ciLow.map((v) => roundSignificant(v)),
    ciHigh: ciHigh.map((v) => roundSignificant(v)),
    fdrSignificant,
    summary: {
      // `median` (scripts/training/stats.ts) averages the two middle
      // elements for an even-length array, unlike `percentile`'s
      // nearest-rank method below (kept for p5Effect/p95Effect, where the
      // plan does not call for interpolation) -- a dual-review finding: this
      // previously used `percentile(sortedEffect, 0.5)`, which returns the
      // upper-middle element instead of the true median for an even n.
      medianEffect: roundSignificant(median(effect)),
      p5Effect: roundSignificant(percentile(sortedEffect, 0.05)),
      p95Effect: roundSignificant(percentile(sortedEffect, 0.95)),
      ciExcludesZeroCount,
      expectedChanceExclusions: roundSignificant(CI_ALPHA * n),
      fdrSignificantCount,
      topEffects
    }
  };
};

export const buildArtifact = (
  raw: Readonly<AtlasEvaluationRaw>,
  positions: PositionsShape,
  runMeta: Readonly<RunMeta>,
  bootstrapSeed: number,
  bootstrapResamples: number
): LesionAtlasArtifact => {
  if (raw.version !== 1) {
    throw new Error(`atlas-report: unsupported raw version ${String(raw.version)}, expected 1`);
  }
  if (!raw.graphs.biological || !raw.graphs.rewiredSeed0) {
    throw new Error('atlas-report: raw evaluation is missing biological/rewiredSeed0 (run atlas-evaluate for both graphs)');
  }

  // The artifact's own `neuronCount` is the number of neurons actually
  // covered by the lesion sweep (`raw.graphs.*.lesion.length`), never
  // `raw.neuronCount` (the manifest's full 1008, recorded on every raw
  // evaluation regardless of `atlas-evaluate.ts --max-lesions`). Using
  // `raw.neuronCount` here would make a calibration run's report claim
  // "1008 simultaneous tests" when only e.g. 10 were run, and -- more
  // seriously -- would let `guardShippedDefault` below wrongly treat a
  // partial run as large enough to safely overwrite the shipped
  // 1008-neuron artifact (a bug caught while calibrating this WP on the
  // real artifacts).
  if (raw.graphs.biological.lesion.length !== raw.graphs.rewiredSeed0.lesion.length) {
    throw new Error(
      `atlas-report: biological covers ${raw.graphs.biological.lesion.length} neuron(s) but rewiredSeed0 covers ` +
        `${raw.graphs.rewiredSeed0.lesion.length} -- both graphs must cover the same neuron set`
    );
  }
  const neuronCount = raw.graphs.biological.lesion.length;

  const biological = buildGraphArtifact('biological', raw.graphs.biological, positions, bootstrapSeed, bootstrapResamples);
  const rewiredSeed0 = buildGraphArtifact('rewiredSeed0', raw.graphs.rewiredSeed0, positions, bootstrapSeed, bootstrapResamples);

  return {
    version: 1,
    condition: 'authored, opponent parked, single-neuron lesion',
    seeds: raw.seeds,
    ticks: raw.ticks,
    substeps: raw.substeps,
    neuronCount,
    // Sliced to `neuronCount` so `bodyIds.length` always matches
    // `effect.length` -- a no-op for a full production run (`neuronCount
    // === positions.bodyIds.length === 1008`), but keeps a calibration
    // run's artifact internally consistent too.
    bodyIds: positions.bodyIds.slice(0, neuronCount),
    graphs: { biological, rewiredSeed0 },
    shards: runMeta.shards,
    bootstrap: { resamples: bootstrapResamples, seed: bootstrapSeed },
    host: raw.host,
    ...(runMeta.elapsedMs !== undefined && runMeta.perEpisodeMs !== undefined
      ? { timing: { elapsedMs: runMeta.elapsedMs, perEpisodeMs: runMeta.perEpisodeMs } }
      : {})
  };
};

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/** Recursively sort object keys, matching the manifest's existing `json.dumps(..., sort_keys=True)` convention -- a local copy of `null-report.ts`'s (not exported) `sortKeysDeep`, same rationale. */
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

export const updateManifestWithLesionAtlas = (
  manifestPath: string,
  entry: { readonly artifact: string; readonly sha256: string }
): void => {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
  manifest.lesionAtlas = entry;
  atomicWriteFileSync(manifestPath, `${JSON.stringify(sortKeysDeep(manifest), null, 2)}\n`);
};

// ---------------------------------------------------------------------------
// Markdown report
// ---------------------------------------------------------------------------

const fmt = (value: number, digits = 4): string => value.toFixed(digits);

/**
 * `pairedBootstrapPValue` accumulates `oppositeSideCount` in steps of `1`
 * (a full opposite-side resample) or `0.5` (an exact-zero tie, weighted
 * evenly between the two sides) -- so its smallest achievable *nonzero*
 * output is `2 * 0.5 / resamples = 1/resamples`, from a single tie, not
 * `2/resamples` (a dual-review round-2 finding: a full opposite-side
 * resample alone gives `2/resamples`, but a lone tie gives a *smaller*
 * nonzero value, so `1/resamples` is the true floor). `p === 0` means no
 * resample landed on the opposite side and no resample tied at exactly
 * 0 -- not that the true p-value is exactly zero. Printing `0.0000` would
 * read as the stronger claim -- shown as an explicit resolution floor
 * instead, so a reader never mistakes Monte Carlo resolution for exact
 * significance.
 */
const fmtP = (p: number, resamples: number): string => (p === 0 ? `< ${(1 / resamples).toExponential(0)}` : fmt(p, 4));

const renderGraphSection = (title: string, graph: Readonly<AtlasGraphArtifact>, bootstrapResamples: number): string => {
  const topRows = graph.summary.topEffects
    .map(
      (entry) =>
        `| ${entry.index} | ${entry.bodyId} | ${entry.role} | ${fmt(entry.effect)} | (${fmt(entry.ciLow)}, ${fmt(entry.ciHigh)}) | ${fmtP(entry.pValue, bootstrapResamples)} | ${entry.fdrSignificant ? 'yes' : 'no'} |`
    )
    .join('\n');

  return `### ${title}

| Quantity | Value |
| --- | --- |
| Baseline score | ${fmt(graph.baseline)} |
| Median effect | ${fmt(graph.summary.medianEffect)} |
| 5th percentile effect | ${fmt(graph.summary.p5Effect)} |
| 95th percentile effect | ${fmt(graph.summary.p95Effect)} |
| CIs excluding 0 (uncorrected) | ${graph.summary.ciExcludesZeroCount} (expected by chance: ${fmt(graph.summary.expectedChanceExclusions, 1)}) |
| FDR-surviving neurons (q=${FDR_Q}) | ${graph.summary.fdrSignificantCount} |

Top ${graph.summary.topEffects.length} neurons by \\|effect\\|:

| Index | Body ID | Role | Effect | 95% CI | p (bootstrap) | FDR significant |
| --- | --- | --- | --- | --- | --- | --- |
${topRows}
`;
};

export const renderReportMarkdown = (artifact: Readonly<LesionAtlasArtifact>): string => {
  const timingRow = artifact.timing
    ? `| Wall time | ${(artifact.timing.elapsedMs / 1000).toFixed(1)}s |\n| Per-episode time | ${artifact.timing.perEpisodeMs.toFixed(1)} ms |\n`
    : '';

  const markdown = `# Single-neuron lesion atlas

For each of the ${artifact.neuronCount} neurons in the biological MaleCNS graph and in the shipped
rewired-seed-0 control, the paired change in \`movementScore\` when that one neuron is silenced for the
whole episode, versus the same episode unlesioned, averaged over ${artifact.seeds.count} held-out seeds.

**What this measures, and what it does not.** Every number below is a property of *this model* --
this hand-authored decoder (\`src/lib/arena/actions.ts\`'s \`decodeAction\`), this rate-model dynamics, this
arena, with the opponent parked and evaluated only on held-out seeds -- not a claim about the real fly's
neural function. A neuron's effect being near zero does not mean it is biologically unimportant; it means
silencing it does not change this specific decoder's score under this specific evaluation. See "Limitations"
below.

## Method

Every neuron is lesioned alone (the counterfactual workbench's semantics exactly: rates zeroed before the
substep loop and after every substep, \`src/lib/counterfactual/engine.ts\`'s \`stepBranch\`, reproduced by
\`scripts/training/episode.ts\`'s \`lesion\` option -- see \`.agents/plans/lesion-atlas/01-lesion-episodes.md\`),
for the whole episode from tick 0 (no warmup fork, unlike the interactive workbench). The left agent runs the
authored decoder; the right agent is parked (always the zero action). Held-out seeds
\`${artifact.seeds.start}..${artifact.seeds.start + artifact.seeds.count - 1}\` (n=${artifact.seeds.count}),
\`T=${artifact.ticks}\` ticks, \`K=${artifact.substeps}\` neural substeps per tick -- the same condition the
rewiring-null study uses. The lesion effect for neuron \`i\` is the mean over seeds of
\`movementScore(lesioned i) - movementScore(baseline)\`, paired by seed, with a 95% bootstrap CI
(\`scripts/training/stats.ts\`'s \`pairedStats\`, ${artifact.bootstrap.resamples} resamples, seed
\`${artifact.bootstrap.seed}\`, one independent RNG stream per neuron per graph). The \`p (bootstrap)\` column in
the top-effects tables below is a separate two-sided percentile-bootstrap p-value (twice the fraction of
resampled mean differences on the opposite side of 0 from the observed effect, capped at 1), drawn from its
own RNG stream independent of the CI's -- a neuron's CI and p-value can therefore disagree near the
boundary (a CI that just excludes 0 while p is just above 0.05, or the reverse). With
${artifact.bootstrap.resamples} resamples the smallest achievable nonzero p-value is
\`1/${artifact.bootstrap.resamples}\` (from a single resample landing exactly on 0, weighted evenly between
the two sides); \`p < ...\` below means no resample crossed to, or landed on, the other side, not that the
true p-value is exactly zero.

**Multiple comparisons.** With ${artifact.neuronCount} simultaneous per-neuron CIs per graph, about
${fmt(artifact.neuronCount * CI_ALPHA, 0)} are expected to exclude 0 by chance alone even if every neuron's true
effect were exactly 0. Benjamini-Hochberg FDR control at \`q = ${FDR_Q}\` (per graph, ${artifact.neuronCount}
tests) marks which neurons' effects survive that correction (\`fdrSignificant\`); only FDR-surviving neurons
should be read as reliable effects, not every neuron whose raw CI happens to exclude 0.

## Parameters

| Parameter | Value |
| --- | --- |
| Condition | ${artifact.condition} |
| Held-out seeds | ${artifact.seeds.start}..${artifact.seeds.start + artifact.seeds.count - 1} (n=${artifact.seeds.count}) |
| Ticks (T) | ${artifact.ticks} |
| Substeps (K) | ${artifact.substeps} |
| Neurons per graph | ${artifact.neuronCount} |
| Evaluation shards | ${artifact.shards} |
${timingRow}| Bootstrap resamples | ${artifact.bootstrap.resamples} |
| Bootstrap seed | ${artifact.bootstrap.seed} |
| FDR q | ${FDR_Q} |
| Host | ${artifact.host.arch}, Node ${artifact.host.node} |

## Results

${renderGraphSection('Biological', artifact.graphs.biological, artifact.bootstrap.resamples)}
${renderGraphSection('Rewired seed 0', artifact.graphs.rewiredSeed0, artifact.bootstrap.resamples)}
## Limitations

- **This model only.** No claim is made that any lesioned neuron "controls" a behavior in the real fly --
  only that silencing it changes this authored decoder's score under this evaluation.
- **Single lesions only.** Redundant neurons can each show a near-zero effect alone while a joint lesion of
  both would not; this atlas does not probe pairs or groups (a stated non-goal, follow-up work).
- **Full-episode lesions from tick 0**, unlike the interactive counterfactual workbench's warmup fork -- the
  two are not directly comparable measurements.
- **About ${fmt(artifact.neuronCount * CI_ALPHA, 0)} of the per-neuron 95% CIs are expected to exclude 0 by
  chance per graph** (${artifact.neuronCount} simultaneous tests at the nominal 5% rate) -- only
  FDR-surviving neurons (\`fdrSignificant\`) should be treated as reliable effects.
- **The opponent is parked** for every episode, matching the null studies' single-agent condition, not a
  competitive one.
- **No biological claim.** This describes how this specific hand-authored decoder and rate-model dynamics
  interact with the measured topology, not a measurement of the real fly's neural function.
`;

  return `${markdown.trimEnd()}\n`;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

/**
 * Every reason `raw` is not "shipped-grade" -- not just under-covered on
 * neurons (the calibration-run bug this WP already fixed once), but also a
 * shorter/cheaper condition than the shipped atlas's. A dual-review finding:
 * checking neuron coverage alone would still let a full-neuron but
 * short-seed/short-tick smoke run (`--held-out-count 3 --ticks 100`, which
 * covers all 1008 neurons in a couple of minutes) pass the guard and
 * overwrite the shipped artifact with drastically noisier numbers. Computed
 * directly from `raw` -- before `buildArtifact`'s bootstrap resampling runs
 * at all -- so an unshippable run is rejected before paying for ~2 x 1008 x
 * (10000 CI resamples + 10000 p-value resamples) draws, matching this
 * file's other pre-`buildArtifact` checks below.
 */
const rawShippedGradeProblems = (raw: Readonly<AtlasEvaluationRaw>): readonly string[] => {
  const problems: string[] = [];
  const biologicalCount = raw.graphs.biological?.lesion.length ?? 0;
  const rewiredCount = raw.graphs.rewiredSeed0?.lesion.length ?? 0;
  // Compared against `raw.neuronCount` (cross-checked against the live
  // manifest by `verifyRawGraphsMatchManifest`, called before this), not a
  // hardcoded constant -- a round-1 dual-review suggestion, so this never
  // goes stale if the compiled graph's neuron count ever changes.
  if (biologicalCount < raw.neuronCount || rewiredCount < raw.neuronCount) {
    problems.push(
      `covers ${biologicalCount}/${rewiredCount} neurons (biological/rewiredSeed0), shipped requires ` +
        `${raw.neuronCount} for both`
    );
  }
  if (raw.seeds.start !== DEFAULT_HELD_OUT_START || raw.seeds.count !== DEFAULT_HELD_OUT_COUNT) {
    problems.push(`seeds ${raw.seeds.start}+${raw.seeds.count} (shipped: ${DEFAULT_HELD_OUT_START}+${DEFAULT_HELD_OUT_COUNT})`);
  }
  if (raw.ticks !== DEFAULT_TICKS) problems.push(`ticks ${raw.ticks} (shipped: ${DEFAULT_TICKS})`);
  if (raw.substeps !== NEURAL_SUBSTEPS_PER_TICK) {
    problems.push(`substeps ${raw.substeps} (shipped: ${NEURAL_SUBSTEPS_PER_TICK})`);
  }
  return problems;
};

const guardShippedDefault = (path: string, defaultPath: string, label: string, problems: readonly string[]): void => {
  if (resolve(path) !== resolve(defaultPath)) return;
  if (problems.length === 0) return;
  throw new Error(
    `atlas-report: refusing to overwrite the shipped ${label} (${defaultPath}) -- this run is not shipped-grade: ` +
      `${problems.join('; ')}. Pass explicit scratch --out/--report-md/--manifest paths for a dev/calibration run.`
  );
};

/**
 * `updateManifestWithLesionAtlas` records the artifact as `basename(args.out)`
 * -- resolved by both the browser and `atlas-evaluate.ts` relative to the
 * *manifest's own directory* (`dirname(args.manifest)`), never `--out`'s
 * directory. Without this check, an otherwise shipped-grade run with a
 * scratch `--out` but the default `--manifest` would pass every other
 * guard, then leave the shipped manifest pointing at a `public/data/`
 * filename that does not actually exist there (a dual-review finding).
 */
export const requireOutBesideManifest = (outPath: string, manifestPath: string): void => {
  if (resolve(dirname(outPath)) === resolve(dirname(manifestPath))) return;
  throw new Error(
    `atlas-report: --out (${outPath}) must be in the same directory as --manifest (${manifestPath}) -- the ` +
      "manifest records the artifact by basename, resolved relative to its own directory"
  );
};

/**
 * Defense in depth: `pairedStats` throws if a lesioned/baseline pair has
 * mismatched *lengths*, but says nothing about whether they are actually
 * the *same seeds in the same order* -- a hand-edited or merged
 * `atlas-raw.json` with a graph's `heldOutSeeds` reordered or drawn from a
 * different range than `raw.seeds` would silently pair the wrong episodes
 * together and publish a nonsensical effect with no error anywhere (a
 * dual-review finding). `atlas-evaluate.ts` always builds every task's
 * `heldOutSeeds` from one shared array, so this holds by construction for
 * its own output -- this only protects a file that reached this point some
 * other way.
 */
export const verifyRawSeedsConsistent = (raw: Readonly<AtlasEvaluationRaw>): void => {
  const expected = Array.from({ length: raw.seeds.count }, (_, i) => raw.seeds.start + i);
  const sameSeeds = (seeds: readonly number[]): boolean =>
    seeds.length === expected.length && seeds.every((seed, i) => seed === expected[i]);
  for (const key of ['biological', 'rewiredSeed0'] as const) {
    const graph = raw.graphs[key];
    if (!graph) continue;
    if (!sameSeeds(graph.heldOutSeeds)) {
      throw new Error(
        `atlas-report: ${key}'s heldOutSeeds do not match raw.seeds (${raw.seeds.start}..+${raw.seeds.count})`
      );
    }
    if (graph.baselineMovementScore.length !== expected.length) {
      throw new Error(`atlas-report: ${key}'s baselineMovementScore length does not match raw.seeds.count`);
    }
    for (const entry of graph.lesion) {
      if (entry.movementScore.length !== expected.length) {
        throw new Error(`atlas-report: ${key} lesion index ${entry.index}'s movementScore length does not match raw.seeds.count`);
      }
    }
  }
};

/**
 * `atlas-evaluate.ts` verifies every graph file's sha256 against the
 * manifest before scoring anything, but `atlas-raw.json` is a plain file on
 * disk that can go stale (the manifest recompiled or re-rewired since that
 * run) or be hand-edited/merged. Without this check, a stale raw evaluation
 * would publish an artifact whose `graphSha256` no longer matches the
 * manifest it was just written next to -- caught only later, in the
 * browser, by WP3's loader (a dual-review finding: this mirrors
 * `null-report.ts`'s `verifySourceGraphMatchesManifest`, but that function
 * is specific to the null study's `NullReportArgs`/single source graph, not
 * reused here). Skips a graph key `raw.graphs` doesn't have -- `buildArtifact`
 * already gives the clearer "missing biological/rewiredSeed0" error for that.
 */
export const verifyRawGraphsMatchManifest = (manifestPath: string, raw: Readonly<AtlasEvaluationRaw>): void => {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as {
    readonly neuronCount?: number;
    readonly binarySha256?: string;
    readonly rewiredArms?: { readonly seed0?: { readonly binarySha256?: string } };
  };
  // `rawShippedGradeProblems` below compares neuron coverage against
  // `raw.neuronCount` rather than a hardcoded constant (a dual-review
  // round-1 suggestion this round-2 pass adopts) -- that comparison is only
  // meaningful if `raw.neuronCount` itself is checked against the live
  // manifest here, or a hand-edited raw file could claim any neuronCount
  // and trivially "cover" it, defeating the guard entirely.
  if (raw.neuronCount !== manifest.neuronCount) {
    throw new Error(
      `atlas-report: raw evaluation's neuronCount (${raw.neuronCount}) does not match ${manifestPath} ` +
        `(${String(manifest.neuronCount)}) -- stale atlas-raw.json?`
    );
  }
  const expected: Record<AtlasGraphKey, string | undefined> = {
    biological: manifest.binarySha256,
    rewiredSeed0: manifest.rewiredArms?.seed0?.binarySha256
  };
  for (const key of ['biological', 'rewiredSeed0'] as const) {
    const graph = raw.graphs[key];
    if (!graph) continue;
    if (graph.graphSha256 !== expected[key]) {
      throw new Error(
        `atlas-report: raw evaluation's ${key} graphSha256 (${graph.graphSha256}) does not match ${manifestPath} ` +
          `(${String(expected[key])}) -- stale atlas-raw.json?`
      );
    }
  }
};

export interface RunAtlasReportResult {
  readonly out: string;
  readonly reportMdPath: string;
  readonly artifactSha256: string;
  readonly artifact: LesionAtlasArtifact;
}

/**
 * The minimum shape every check below assumes (`raw.seeds.start`/`.count`
 * in particular -- `verifyRawSeedsConsistent`/`rawShippedGradeProblems`
 * both read them directly with no shape check of their own). A malformed
 * or wrong-version `atlas-raw.json` would otherwise fail with a raw
 * `TypeError: Cannot read properties of undefined` from deep inside one of
 * those checks instead of a clear, actionable message (a dual-review
 * round-2 finding). `buildArtifact` has its own, later `raw.version !== 1`
 * check for the same reason `null-report.ts`'s `buildArtifact` does --
 * this one exists so every *earlier* check in `runAtlasReport` can assume
 * this shape holds without re-checking it itself.
 */
export const validateRawShape = (rawPath: string, raw: Readonly<AtlasEvaluationRaw>): void => {
  if (raw.version !== 1) {
    throw new Error(`atlas-report: ${rawPath} has unsupported version ${String(raw.version)}, expected 1`);
  }
  if (
    typeof raw.seeds !== 'object' ||
    raw.seeds === null ||
    !Number.isInteger(raw.seeds.start) ||
    !Number.isInteger(raw.seeds.count) ||
    raw.seeds.count <= 0
  ) {
    throw new Error(`atlas-report: ${rawPath} is missing a valid seeds.start/seeds.count`);
  }
  if (typeof raw.graphs !== 'object' || raw.graphs === null) {
    throw new Error(`atlas-report: ${rawPath} is missing a graphs object`);
  }
};

export const runAtlasReport = (args: Readonly<AtlasReportArgs>): RunAtlasReportResult => {
  const raw = JSON.parse(readFileSync(args.raw, 'utf8')) as AtlasEvaluationRaw;
  validateRawShape(args.raw, raw);
  const runMeta = resolveRunMeta(args);

  // Every check in this block is cheap (file reads and field comparisons,
  // no bootstrap resampling) and runs before `buildArtifact`'s ~2 x
  // neuronCount x (bootstrapResamples x 2) draws below, so an unshippable
  // or stale run is rejected in milliseconds, not after paying for the
  // full statistics pass (a dual-review finding).
  requireOutBesideManifest(args.out, args.manifest);
  verifyRawSeedsConsistent(raw);
  verifyRawGraphsMatchManifest(args.manifest, raw);
  const shippedGradeProblems = rawShippedGradeProblems(raw);
  guardShippedDefault(args.out, DEFAULT_OUT, 'published artifact', shippedGradeProblems);
  guardShippedDefault(args.reportMd, DEFAULT_REPORT_MD, 'report', shippedGradeProblems);
  guardShippedDefault(args.manifest, DEFAULT_MANIFEST, 'manifest', shippedGradeProblems);
  verifyManifestRoundTrips(args.manifest);

  if (!raw.graphs.biological) {
    throw new Error(`atlas-report: ${args.raw} has no biological section`);
  }
  const positions = readPositions(args.positions, raw.neuronCount, raw.graphs.biological.graphGzipSha256);

  const artifact = buildArtifact(raw, positions, runMeta, args.bootstrapSeed, args.bootstrapResamples);

  const artifactContents = JSON.stringify(artifact);
  const artifactSha256 = sha256Hex(artifactContents);
  const reportMdContents = renderReportMarkdown(artifact);

  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, artifactContents);

  updateManifestWithLesionAtlas(args.manifest, { artifact: basename(args.out), sha256: artifactSha256 });

  mkdirSync(dirname(args.reportMd), { recursive: true });
  atomicWriteFileSync(args.reportMd, reportMdContents);

  return { out: args.out, reportMdPath: args.reportMd, artifactSha256, artifact };
};

const main = (): void => {
  try {
    const args = parseAtlasReportArgs(process.argv.slice(2));
    if (!existsSync(args.raw)) {
      throw new Error(`${args.raw} does not exist; run "npm run lesion:evaluate" first`);
    }
    const result = runAtlasReport(args);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `atlas-report: wrote ${result.out} (sha256 ${result.artifactSha256}) and ${result.reportMdPath}\n` +
        `biological fdrSignificant=${result.artifact.graphs.biological.summary.fdrSignificantCount} ` +
        `rewiredSeed0 fdrSignificant=${result.artifact.graphs.rewiredSeed0.summary.fdrSignificantCount}`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`atlas-report failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
