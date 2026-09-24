import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
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
import type { NullEvaluationRaw } from './null-evaluate';

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
const DEFAULT_OUT = resolve(repoRoot, 'public/data/rewiring-null-v1.json');
const DEFAULT_REPORT_MD = resolve(repoRoot, 'docs/rewiring-null-report.md');
const DEFAULT_MANIFEST = resolve(repoRoot, 'public/data/malecns-arena-v1.manifest.json');

/** 'N','U','L','L' as a fixed default seed; arbitrary but stable across runs, matching `evaluate.ts`'s `DEFAULT_BOOTSTRAP_SEED` convention. */
const DEFAULT_BOOTSTRAP_SEED = 0x4e554c4c;
const DEFAULT_BOOTSTRAP_RESAMPLES = 10000;
/** Matches `null-evaluate.ts`'s own `DEFAULT_SHARDS` — the plan's "Default 18 (leaves 2 cores)" implementer decision. */
const DEFAULT_SHARDS = 18;

const sha256Hex = (data: string): string => createHash('sha256').update(data, 'utf8').digest('hex');

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
   * Provenance only, for the published artifact's `shards` field — the
   * number of shards actually used for the real evaluation run. Not read
   * from `authored.json`: see that file's `NullEvaluationRaw` doc comment
   * for why shard count deliberately isn't part of the deterministic raw
   * output.
   */
  readonly shards: number;
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
  let shards = DEFAULT_SHARDS;

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

  return { authored, trained, out, reportMd, manifest, bootstrapSeed, bootstrapResamples, histogramBins, shards };
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
 * 2-space indent, matching the file's existing (Python-written) format.
 */
export const updateManifestWithRewiringNull = (
  manifestPath: string,
  entry: { readonly artifact: string; readonly sha256: string }
): void => {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
  manifest.rewiringNull = entry;
  writeFileSync(manifestPath, `${JSON.stringify(sortKeysDeep(manifest), null, 2)}\n`);
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

export const buildArtifact = (raw: Readonly<NullEvaluationRaw>, args: Readonly<NullReportArgs>): RewiringNullArtifact => {
  if (!raw.biological || !raw.disconnected) {
    throw new Error(
      `null-report: ${args.authored} has no biological/disconnected section ` +
        '(run null-evaluate with --biological for the real report)'
    );
  }

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
    throw new Error('null-report: rewired seed 0 is missing (required for the paired biological-vs-rewired-seed0 comparison)');
  }
  const paired = pairedStats(
    raw.biological.movementScore,
    seed0.movementScore,
    bootstrapResamples,
    conditionRng(bootstrapSeed, 'paired|biological-vs-rewired-seed0')
  );

  const bins = buildHistogram([...nullValues, biologicalStats.mean, disconnectedStats.mean], args.histogramBins);

  return {
    version: 1,
    condition: 'authored, opponent parked',
    seeds: raw.seeds,
    ticks: raw.ticks,
    substeps: raw.substeps,
    sourceGraphSha256: raw.sourceGraphSha256,
    rewireSourceSha256: raw.rewireSourceSha256,
    shards: args.shards,
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
    host: raw.host
  };
};

// ---------------------------------------------------------------------------
// Markdown report
// ---------------------------------------------------------------------------

const fmt = (value: number, digits = 4): string => value.toFixed(digits);
const pct = (value: number): string => `${(value * 100).toFixed(1)}%`;

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

  const degenerateSection = artifact.null.degenerate
    ? `\n> **Degenerate null.** This null distribution's IQR (${artifact.null.iqr.toExponential(3)}) is below the ` +
      `${DEGENERATE_IQR_THRESHOLD.toExponential(0)} threshold used to flag a degenerate result. Under this evaluation ` +
      `setup (authored decoder, opponent parked, T=${artifact.ticks}), the authored path is essentially insensitive ` +
      `to rewiring topology: rewired-graph scores cluster too tightly to place the biological graph meaningfully. ` +
      'This is a finding about this specific condition, not a general claim about the MaleCNS connectome.\n'
    : '';

  return `# Rewiring null — authored-decoder evaluation

Where the measured biological MaleCNS topology falls among ${artifact.rewired.length} degree-preserving
rewirings of the same graph, scored under identical dynamics, encoder, decoder, and held-out seeds.
${degenerateSection}
## Method

Every graph below (biological, disconnected, and each of the ${artifact.rewired.length} rewired graphs)
is scored by \`scripts/training/episode.ts\`'s \`runEpisode\`: the authored decoder drives the left agent,
the right agent is parked (always the zero action), over the same ${artifact.seeds.count} held-out seeds
(\`${artifact.seeds.start}..${artifact.seeds.start + artifact.seeds.count - 1}\`), \`T = ${artifact.ticks}\`
ticks, \`K = ${artifact.substeps}\` neural substeps per tick. Degree-preserving rewiring comes only from
\`scripts/data/rewire.py\`'s \`rewire_graph\` (rewiring seeds \`0..${artifact.rewired.length - 1}\`, seed 0
being the shipped control arm). A graph's score is the mean \`movementScore\` over its held-out seeds; the
null set \`N\` is the ${artifact.rewired.length} rewired graphs' mean scores.

## Parameters

| Parameter | Value |
| --- | --- |
| Condition | ${artifact.condition} |
| Held-out seeds | ${artifact.seeds.start}..${artifact.seeds.start + artifact.seeds.count - 1} (n=${artifact.seeds.count}) |
| Ticks (T) | ${artifact.ticks} |
| Substeps (K) | ${artifact.substeps} |
| Rewired graphs | ${artifact.rewired.length} (seeds 0..${artifact.rewired.length - 1}) |
| Evaluation shards | ${artifact.shards} |
| Bootstrap resamples | ${artifact.bootstrap.resamples} |
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

## Limitations

- Scores come from a **single-agent condition with the opponent parked** — the same headline condition the
  trained-readout report uses. Two-agent competitive dynamics are not evaluated here.
- **One null model** is used: degree-preserving double-edge swaps (\`scripts/data/rewire.py\`). Other null
  models (weight shuffles within degree, Erdős–Rényi with matched density) are deferred follow-ups.
- **No causal or superiority claim is made.** The percentile and rank statistics above are descriptive: they
  say where the biological graph's score falls among this null model's rewirings under this exact evaluation
  setup, not that biological topology causes or predicts any particular score.
${artifact.null.degenerate ? '- The null distribution is **degenerate** (IQR below threshold) — see the note above.\n' : ''}
`;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

export const runNullReport = (args: Readonly<NullReportArgs>): RunNullReportResult => {
  const raw = JSON.parse(readFileSync(args.authored, 'utf8')) as NullEvaluationRaw;
  const artifact = buildArtifact(raw, args);

  const artifactContents = JSON.stringify(artifact);
  mkdirSync(dirname(args.out), { recursive: true });
  writeFileSync(args.out, artifactContents);
  const artifactSha256 = sha256Hex(artifactContents);

  updateManifestWithRewiringNull(args.manifest, { artifact: basename(args.out), sha256: artifactSha256 });

  const reportMdContents = renderReportMarkdown(artifact);
  mkdirSync(dirname(args.reportMd), { recursive: true });
  writeFileSync(args.reportMd, reportMdContents);

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
