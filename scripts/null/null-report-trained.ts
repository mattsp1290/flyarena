import { existsSync, readFileSync } from 'node:fs';

import type { ConditionStats } from '../training/stats';
import {
  graphStats,
  nullSummary,
  percentileResolution,
  rankStatistics,
  trainerSeedSpread,
  type NullSummary,
  type TrainerSeedSpread
} from './null-stats';
import type { NullTrainedEvaluationRaw } from './null-trained-evaluate';

/**
 * `.agents/plans/rewiring-null/03-trained-sample.md`'s WP3 report section:
 * the trained-readout sample (20 CEM-trained rewired readouts vs. the three
 * `flyarena-bigq` biological replicas), extracted out of `null-report.ts`
 * (a thermo-architecture/thermo-maintainability review finding: the WP3
 * addition pushed `null-report.ts` from 728 to 1084 lines, past this
 * repo's "do not let a file cross 1000 lines without a very strong reason"
 * rule, and the trained-section code was already a clean, self-contained
 * unit depending only on `null-stats.ts` and `null-trained-evaluate.ts`'s
 * types). `null-report.ts` imports `TrainedSection`/`buildTrainedSection`/
 * `renderTrainedSection` from here the same way it already imports from
 * `null-evaluate.ts`/`null-trained-evaluate.ts` -- a pure extraction, no
 * behavior change. This module deliberately does NOT import anything back
 * from `null-report.ts` (a small `ScoredEntry`/`toScoredEntry`/`fmt`/`pct`
 * are duplicated below rather than shared) to keep the import boundary
 * one-directional.
 */

interface ScoredEntry {
  readonly score: number;
  readonly median: number;
  readonly std: number;
  readonly ci: readonly [number, number];
}

const toScoredEntry = (stats: ConditionStats): ScoredEntry => ({
  score: stats.mean,
  median: stats.median,
  std: stats.std,
  ci: stats.ci95
});

const fmt = (value: number, digits = 4): string => value.toFixed(digits);
const pct = (value: number): string => `${(value * 100).toFixed(1)}%`;

export interface TrainedRewiredEntry extends ScoredEntry {
  readonly seed: number;
}

export interface TrainedBiologicalEntry extends ScoredEntry {
  readonly trainerSeed: number;
}

/**
 * The trained-readout sample: 20 rewired graphs, each CEM-trained with the
 * exact production config `flyarena-bigq` used for its biological replicas
 * (`replicaSeed`, one trainer seed shared by every rewired run -- "isolates
 * topology from trainer-seed variance", `00-overview.md`'s key decisions),
 * compared against the three biological replicas bigq itself trained
 * (`trainerSeed` 101/202/303).
 *
 * `bioTrainerSeedSpread` and `bioPercentile` measure two *different* kinds
 * of variance and must never be read as comparable: `bioPercentile` places
 * biological trainer-seed-101 among the 20 rewired scores -- **topology
 * variance at a fixed trainer seed**. `bioTrainerSeedSpread` is the min/max
 * spread across the three biological replicas' own trained scores, all at
 * the *same* (biological) topology -- **trainer-noise variance at fixed
 * topology**. Whether one spread happens to be larger or smaller than the
 * other says nothing about whether topology "matters more" than trainer
 * noise; see `bioTrainerSeedSpread.label` and `renderTrainedSection`, which
 * restates this in prose next to every place either number is printed.
 */
export interface TrainedSection {
  readonly condition: 'trained, opponent parked';
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
  readonly replicaSeed: number;
  readonly d: number;
  /** Sorted by seed ascending; length 20 in the real study. */
  readonly rewired: readonly TrainedRewiredEntry[];
  /** Sorted by trainerSeed ascending; the three bigq replicas (101/202/303). */
  readonly biological: readonly TrainedBiologicalEntry[];
  /** Statistics over the 20 rewired trained scores. */
  readonly null: NullSummary;
  /** Biological trainer-seed-101's empirical percentile among the 20 rewired trained scores. */
  readonly bioPercentile: number;
  readonly pLow: number;
  readonly pHigh: number;
  /** `1/|rewired|` -- 5% at n=20. See `null-stats.ts`'s `percentileResolution`. */
  readonly percentileResolution: number;
  readonly bioTrainerSeedSpread: TrainerSeedSpread & { readonly label: string };
  /** `trained-readout-v1.manifest.json`'s `gpuRerunFitnessDelta` (a CUDA rerun of the shipped biological replica moved held-out `trained` fitness by this much) -- cited here as evidence that trainer-seed/run-to-run noise at fixed topology can be large, motivating `bioTrainerSeedSpread`'s own existence. */
  readonly bigqGpuRerunFitnessDelta: number;
  readonly bigqMergeCommit: string;
  readonly evaluatorGitRev: string | null;
  readonly cemConfig: Record<string, unknown> | null;
  readonly cemConfigWarnings: readonly string[];
  readonly bootstrap: { readonly resamples: number; readonly seed: number };
  /** Present only when `null-trained-evaluate.ts`'s `.run.json` sidecar recorded it. */
  readonly timing?: { readonly elapsedMs: number; readonly perEpisodeMs: number };
}

/** Explicit label carried on `TrainedSection.bioTrainerSeedSpread` -- see that field's and `TrainedSection`'s own doc comments for the full explanation this is a condensed restatement of. */
const BIO_TRAINER_SEED_SPREAD_LABEL =
  "trainer-noise variance at fixed (biological) topology -- NOT comparable to the null's topology " +
  'variance at a fixed trainer seed (bioPercentile, above); no overlap-based conclusion may be drawn ' +
  'from comparing the two.';

/** `trained-readout-v1.manifest.json`'s `gpuRerunFitnessDelta` field -- see `TrainedSection.bigqGpuRerunFitnessDelta`'s doc comment. */
const readBigqGpuRerunFitnessDelta = (manifestPath: string): number => {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as { gpuRerunFitnessDelta?: unknown };
  if (typeof manifest.gpuRerunFitnessDelta !== 'number' || !Number.isFinite(manifest.gpuRerunFitnessDelta)) {
    throw new Error(`null-report: ${manifestPath} has no numeric "gpuRerunFitnessDelta" field`);
  }
  return manifest.gpuRerunFitnessDelta;
};

/** `<trainedPath>.run.json`'s optional `elapsedMs`/`perEpisodeMs`, mirroring `null-report.ts`'s `resolveRunMeta` sidecar convention for `authored.json` -- see `null-trained-evaluate.ts`'s own `<out>.run.json` sidecar. Unlike `resolveRunMeta`, a missing sidecar is not an error: `TrainedSection.timing` is simply omitted (this study's own acceptance criteria don't require it, unlike the authored null's shard count). */
const resolveTrainedTiming = (
  trainedPath: string
): { readonly elapsedMs: number; readonly perEpisodeMs: number } | undefined => {
  if (!trainedPath.endsWith('.json')) {
    throw new Error(`null-report: expected a ".json" trained path, got "${trainedPath}"`);
  }
  const sidecarPath = `${trainedPath.slice(0, -'.json'.length)}.run.json`;
  if (!existsSync(sidecarPath)) return undefined;
  const sidecar = JSON.parse(readFileSync(sidecarPath, 'utf8')) as {
    elapsedMs?: unknown;
    perEpisodeMs?: unknown;
  };
  if (typeof sidecar.elapsedMs !== 'number' || typeof sidecar.perEpisodeMs !== 'number') return undefined;
  return { elapsedMs: sidecar.elapsedMs, perEpisodeMs: sidecar.perEpisodeMs };
};

/**
 * Pure function of WP3's `trained.json` (never re-simulates anything, same
 * convention as `null-report.ts`'s `buildArtifact`). Every bootstrap label
 * is prefixed `trained|` so this section's resamples are independent of,
 * and never collide with, the authored section's own per-graph labels
 * (`graphStats` in `null-stats.ts`, keyed by `(bootstrapSeed, label)` — see
 * `conditionRng`'s doc comment in `../training/stats.ts`).
 *
 * `authoredSeeds`/`authoredTicks`/`authoredSubsteps` come from the SAME
 * `authored.json` the artifact's own authored sections were built from
 * (`runNullReport` passes `raw.seeds`/`raw.ticks`/`raw.substeps` from its
 * own `buildArtifact` call). The generated report's own prose states the
 * trained section uses "the same held-out seeds ... T ... K" as the
 * authored null above — this asserts that claim is actually true of the
 * two input files being merged, rather than only being true by convention
 * (a dual-review finding: nothing previously compared the two).
 *
 * `trainedReadoutManifestPath`/`trainedJsonPath` are read here (not
 * pre-resolved by the caller) so this function is `null-report.ts`'s single
 * entry point for everything WP3's trained section needs from disk —
 * `readBigqGpuRerunFitnessDelta`/`resolveTrainedTiming` stay module-private
 * helpers instead of being re-exported just to be called from the other
 * file (a thermo-architecture review finding).
 */
export const buildTrainedSection = (
  raw: Readonly<NullTrainedEvaluationRaw>,
  bootstrapSeed: number,
  bootstrapResamples: number,
  trainedReadoutManifestPath: string,
  trainedJsonPath: string,
  authoredSeeds: { readonly start: number; readonly count: number },
  authoredTicks: number,
  authoredSubsteps: number
): TrainedSection => {
  const bigqGpuRerunFitnessDelta = readBigqGpuRerunFitnessDelta(trainedReadoutManifestPath);

  if (raw.version !== 1) {
    throw new Error(`null-report: trained.json has unsupported version ${String(raw.version)}, expected 1`);
  }
  if (raw.rewired.length === 0) throw new Error('null-report: trained.json has no rewired entries');
  if (raw.biological.length === 0) throw new Error('null-report: trained.json has no biological entries');
  if (raw.seeds.start !== authoredSeeds.start || raw.seeds.count !== authoredSeeds.count) {
    throw new Error(
      `null-report: trained.json's held-out seeds (${raw.seeds.start}..${raw.seeds.start + raw.seeds.count - 1}) ` +
        `do not match authored.json's (${authoredSeeds.start}..${authoredSeeds.start + authoredSeeds.count - 1})`
    );
  }
  if (raw.ticks !== authoredTicks) {
    throw new Error(`null-report: trained.json's ticks (${raw.ticks}) do not match authored.json's (${authoredTicks})`);
  }
  if (raw.substeps !== authoredSubsteps) {
    throw new Error(
      `null-report: trained.json's substeps (${raw.substeps}) do not match authored.json's (${authoredSubsteps})`
    );
  }

  const rewiredStats = raw.rewired.map((entry) => ({
    entry,
    stats: graphStats(entry.movementScore, bootstrapSeed, `trained|rewired-${entry.seed}`, bootstrapResamples)
  }));
  const biologicalStats = raw.biological.map((entry) => ({
    entry,
    stats: graphStats(
      entry.movementScore,
      bootstrapSeed,
      `trained|biological-${entry.trainerSeed}`,
      bootstrapResamples
    )
  }));

  const nullValues = rewiredStats.map(({ stats }) => stats.mean);
  const summary = nullSummary(nullValues);

  // The percentile below compares biological against the null at the SAME
  // trainer seed the null's 20 rewired readouts were themselves trained at
  // (`raw.replicaSeed`) — not a separately hard-coded "101" convention that
  // could silently drift from `--replica-seed` (a dual-review finding: an
  // earlier version used a module-level `SHIPPED_REPLICA_TRAINER_SEED = 101`
  // constant here, disconnected from `raw.replicaSeed`).
  const shippedBio = biologicalStats.find(({ entry }) => entry.trainerSeed === raw.replicaSeed);
  if (!shippedBio) {
    throw new Error(
      `null-report: trained.json has no biological trainer-seed-${raw.replicaSeed} entry (raw.replicaSeed, the ` +
        'trainer seed every rewired readout was trained at)'
    );
  }
  const rank = rankStatistics(nullValues, shippedBio.stats.mean);

  const spread = trainerSeedSpread(biologicalStats.map(({ stats }) => stats.mean));

  const timing = resolveTrainedTiming(trainedJsonPath);

  return {
    condition: 'trained, opponent parked',
    seeds: raw.seeds,
    ticks: raw.ticks,
    substeps: raw.substeps,
    replicaSeed: raw.replicaSeed,
    d: raw.d,
    rewired: rewiredStats.map(({ entry, stats }) => ({ seed: entry.seed, ...toScoredEntry(stats) })),
    biological: biologicalStats.map(({ entry, stats }) => ({ trainerSeed: entry.trainerSeed, ...toScoredEntry(stats) })),
    null: summary,
    bioPercentile: rank.bioPercentile,
    pLow: rank.pLow,
    pHigh: rank.pHigh,
    percentileResolution: percentileResolution(nullValues.length),
    bioTrainerSeedSpread: { ...spread, label: BIO_TRAINER_SEED_SPREAD_LABEL },
    bigqGpuRerunFitnessDelta,
    bigqMergeCommit: raw.bigqMergeCommit,
    evaluatorGitRev: raw.evaluatorGitRev,
    cemConfig: raw.cemConfig,
    cemConfigWarnings: raw.cemConfigWarnings,
    bootstrap: { resamples: bootstrapResamples, seed: bootstrapSeed },
    ...(timing ? { timing } : {})
  };
};

const renderTrainedRewiredTable = (trained: Readonly<TrainedSection>): string => {
  const rows = trained.rewired.map(
    (entry) => `| ${entry.seed} | ${fmt(entry.score)} (${fmt(entry.ci[0])}, ${fmt(entry.ci[1])}) |`
  );
  return ['| seed | trained score (95% CI) |', '| --- | --- |', ...rows].join('\n');
};

const renderTrainedBiologicalTable = (trained: Readonly<TrainedSection>): string => {
  const rows = trained.biological.map(
    (entry) => `| ${entry.trainerSeed} | ${fmt(entry.score)} (${fmt(entry.ci[0])}, ${fmt(entry.ci[1])}) |`
  );
  return ['| trainer seed | trained score (95% CI) |', '| --- | --- |', ...rows].join('\n');
};

/**
 * Typed summary of `TrainedSection.cemConfig` (`run-dir.ts`'s
 * `CEM_CONFIG_FIELDS`, reconciled at evaluation time) -- narrows the seven
 * fields this section's markdown actually interpolates out of the raw
 * `Record<string, unknown> | null` it is read from, so a typo'd field name
 * is a compile error instead of silently interpolating `undefined` into the
 * published report (a thermo-maintainability review finding).
 */
interface CemConfigSummary {
  readonly population: number;
  readonly elites: number;
  readonly generations: number;
  readonly alpha: number;
  readonly stdFloor: number;
  readonly initStd: number;
  readonly trainingSeedsPerGeneration: number;
}

const CEM_CONFIG_SUMMARY_FIELDS = [
  'population',
  'elites',
  'generations',
  'alpha',
  'stdFloor',
  'initStd',
  'trainingSeedsPerGeneration'
] as const satisfies readonly (keyof CemConfigSummary)[];

/**
 * Narrow `TrainedSection.cemConfig` to `CemConfigSummary`, or `null` if any
 * of the seven required fields is missing/non-numeric -- an all-or-nothing
 * fallback, unlike the old code this replaces (which read `cem.population`
 * etc. straight off a `Record<string, unknown>` and would have printed
 * `undefined` for just the missing field(s) of a partially-populated
 * object, while still rendering the rest of the summary). `cemConfig` is
 * reconciled by `null-trained-evaluate.ts`'s `reconcileCemConfig` (which
 * throws rather than publishing a `trained.json` with an inconsistent
 * config), so a real `trained.json` always has every field here -- a
 * `null`/incomplete result only happens for an older or hand-edited
 * `trained.json` fixture, in which case the report falls back to the same
 * "no CEM config was recorded" prose a fully-`null` `cemConfig` always has.
 */
const toCemConfigSummary = (cemConfig: Record<string, unknown> | null): CemConfigSummary | null => {
  if (!cemConfig) return null;
  for (const field of CEM_CONFIG_SUMMARY_FIELDS) {
    if (typeof cemConfig[field] !== 'number') return null;
  }
  return cemConfig as unknown as CemConfigSummary;
};

/**
 * WP3's trained-readout sample section, appended after the authored null's
 * own sections when `artifact.trained` is present (`''` otherwise, so an
 * authored-only report's markdown is byte-identical to before WP3
 * existed). Every number here is labeled with the evaluator revision that
 * produced it and the exact CEM config it was trained with, and the
 * trainer-seed-variance/percentile-resolution caveats
 * (`.agents/plans/rewiring-null/03-trained-sample.md`'s own required
 * disclosures) are stated directly next to the numbers they qualify, not
 * only in the Limitations section.
 */
export const renderTrainedSection = (
  trained: Readonly<TrainedSection> | undefined,
  authoredNullSize: number
): string => {
  if (!trained) return '';

  const cem = toCemConfigSummary(trained.cemConfig);
  const cemSummary = cem
    ? `: population ${cem.population}, elites ${cem.elites}, generations ${cem.generations}, alpha ${cem.alpha}, ` +
      `stdFloor ${cem.stdFloor}, initStd ${cem.initStd}, E=${cem.trainingSeedsPerGeneration}`
    : ' (no CEM config was recorded on any scored run directory)';
  const cemWarning =
    trained.cemConfigWarnings.length > 0
      ? `\n\n> **CEM config warning(s):** ${trained.cemConfigWarnings.join('; ')}\n`
      : '';
  const timingRow = trained.timing
    ? `\n| Wall time | ${(trained.timing.elapsedMs / 1000).toFixed(1)}s |\n| Per-episode time | ${trained.timing.perEpisodeMs.toFixed(1)} ms |`
    : '';

  return `
## Trained-readout sample

Where the biological MaleCNS topology's *trained* score — a readout CEM-trained specifically for that
topology, not the fixed hand-authored mapping the sections above use — falls among ${trained.rewired.length}
rewired topologies, each given its own readout trained with the identical production CEM configuration
\`flyarena-bigq\` used for its own biological replicas.

**Condition and config.** ${trained.condition}. \`T = ${trained.ticks}\`, \`K = ${trained.substeps}\`,
\`D = ${trained.d}\`, held-out seeds \`${trained.seeds.start}..${trained.seeds.start + trained.seeds.count - 1}\`
(n=${trained.seeds.count}) — the same held-out seeds the authored null above and
[the trained-readout report](trained-readout-report.md) both use. Every rewired readout is trained at the
single trainer seed \`replicaSeed = ${trained.replicaSeed}\` (isolating topology from trainer-seed
variance, per this study's key decisions), with the exact CEM config copied from the merged
\`flyarena-bigq\` manifest (commit \`${trained.bigqMergeCommit}\`)${cemSummary}. Rescored by evaluator git rev
\`${trained.evaluatorGitRev ?? 'unknown'}\` — the same TS \`runEpisode\` authoritative path the authored null
above uses, so every number in this section shares one evaluator revision with every other number in this
section, never a PyTorch-side validation fitness.${cemWarning}

### Rewired trained scores (n=${trained.rewired.length})

${renderTrainedRewiredTable(trained)}

### Biological trained scores (per trainer seed)

${renderTrainedBiologicalTable(trained)}

### Results

| Quantity | Value |
| --- | --- |
| Null (rewired trained) mean | ${fmt(trained.null.mean)} |
| Null median | ${fmt(trained.null.median)} |
| Null std | ${fmt(trained.null.std)} |
| Null 2.5–97.5% | ${fmt(trained.null.p2_5)} .. ${fmt(trained.null.p97_5)} |
| Null IQR | ${fmt(trained.null.iqr)} |
| Biological (trainer seed ${trained.replicaSeed}) percentile among the ${trained.rewired.length} rewired trained scores | ${pct(trained.bioPercentile)} |
| Percentile resolution (1/n) | ${pct(trained.percentileResolution)} |
| Rank statistic p_low | ${fmt(trained.pLow, 4)} |
| Rank statistic p_high | ${fmt(trained.pHigh, 4)} |${timingRow}

With only ${trained.rewired.length} rewired replicas, the percentile above has a resolution of only
${pct(trained.percentileResolution)}: one more or fewer rewired replica scoring below biological shifts it
by a full ${pct(trained.percentileResolution)} step. This is a much coarser distribution than the authored
null's ${authoredNullSize}-replica, ${pct(percentileResolution(authoredNullSize))}-resolution percentile
above, and percentile differences finer than ${pct(trained.percentileResolution)} are not meaningfully
distinguishable at this sample size.

### Trainer-seed variance context

Biological trained scores across the ${trained.biological.length} \`flyarena-bigq\` replicas (trainer seeds
${[...trained.biological].map((e) => e.trainerSeed).sort((a, b) => a - b).join('/')}) span
\`${fmt(trained.bioTrainerSeedSpread.min)}\` to \`${fmt(trained.bioTrainerSeedSpread.max)}\`
(range \`${fmt(trained.bioTrainerSeedSpread.range)}\`) — **${trained.bioTrainerSeedSpread.label}** — at the
*same* biological topology. For context on how large this kind of noise alone can be: the merged
[\`trained-readout-v1.manifest.json\`](../public/data/trained-readout-v1.manifest.json)'s recorded CUDA
rerun of the shipped biological replica moved TS held-out \`trained\` fitness by
\`${fmt(trained.bigqGpuRerunFitnessDelta)}\` (rerun minus original, that report's sign convention) —
${
  trained.bioTrainerSeedSpread.range > 0
    ? `${(Math.abs(trained.bigqGpuRerunFitnessDelta) / trained.bioTrainerSeedSpread.range).toFixed(2)}x the ` +
      'trainer-seed spread recorded above (see that report\'s own Limitations for the CI comparison)'
    : "the trainer-seed spread recorded above is zero in this run's data, so no ratio is computed"
} — a magnitude comparison offered only as context for how large trainer-seed/run-to-run noise can be, not
a claim that the two numbers should match. **This spread is not comparable to the ${pct(trained.bioPercentile)}
percentile above**: the spread measures trainer-seed/run-to-run noise at *fixed* topology; the percentile
measures where one topology (biological, at trainer seed ${trained.replicaSeed}) falls among
${trained.rewired.length} different topologies, each at the *same* one trainer seed. Whether these two
numbers happen to overlap, and neither's size relative to the other, supports any conclusion about topology
"mattering more or less" than trainer-seed noise.
`;
};
