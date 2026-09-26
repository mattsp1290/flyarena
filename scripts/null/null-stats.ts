import { conditionRng, conditionStats, type ConditionStats } from '../training/stats';

/**
 * Statistics for `.agents/plans/rewiring-null/02-authored-null-evaluation.md`'s
 * "Statistics (predeclared)" section: the authored-decoder null distribution
 * over 500 degree-preserving rewirings, plus where the measured biological
 * graph falls in it. Pure functions only — everything here operates on
 * already-computed per-graph scores (the expensive part, one TS episode per
 * seed per graph, lives in `null-evaluate.ts`/`null-worker.ts`), so
 * `null-report.ts` can rerun this module against a cached `authored.json`
 * and reproduce byte-identical output without re-simulating anything.
 *
 * Per-graph mean/median/std/bootstrap-CI reuses `../training/stats.ts`'s
 * `conditionStats`/`conditionRng` directly rather than re-implementing them
 * — see `graphStats` below.
 */

/**
 * `a` and `b` name the same held-out seeds in the same order. Exported (a
 * thermo-maintainability review finding): `null-report.ts` and
 * `intervention-report.ts` each carried a byte-identical hand-duplicated
 * copy of this exact function, with `intervention-report.ts`'s own doc
 * comment even noting the duplication ("matches `null-report.ts`'s
 * identically-named/-shaped helper") instead of removing it — the same
 * pattern `quantileIndex`'s own export below already fixed for that
 * function.
 */
export const sameSeeds = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((seed, i) => seed === b[i]);

/** `IQR < DEGENERATE_IQR_THRESHOLD` marks the null distribution as degenerate (the plan's "authored path insensitive to topology" case). */
export const DEGENERATE_IQR_THRESHOLD = 1e-9;

export const DEFAULT_HISTOGRAM_BINS = 30;

/**
 * Per-graph descriptive statistics and a seeded 95% bootstrap CI, keyed by
 * a stable label (e.g. `"rewired-7"`) so every graph's CI is an independent
 * function of `(bootstrapSeed, its own label, its own data)` — see
 * `conditionRng`'s doc comment in `../training/stats.ts`.
 */
export const graphStats = (
  values: readonly number[],
  bootstrapSeed: number,
  label: string,
  resamples: number
): ConditionStats => conditionStats(values, resamples, conditionRng(bootstrapSeed, label));

export interface NullSummary {
  readonly n: number;
  readonly mean: number;
  readonly median: number;
  readonly std: number;
  readonly p2_5: number;
  readonly p97_5: number;
  readonly iqr: number;
  readonly degenerate: boolean;
}

/**
 * The empirical quantile index into a length-`n` sorted array for
 * proportion `p`, using the same low-tail-floor / high-tail-ceil-minus-one
 * (clamped) convention `bootstrapCI` (`../training/stats.ts`) already uses
 * for its own resampled-draws quantiles — applied here directly to the
 * sorted null set itself rather than to bootstrap resamples, since "null
 * mean, median, std, 2.5/97.5 percentiles" (the plan's own wording) are
 * empirical statistics of the null set, not a bootstrap of it.
 *
 * Exported (a dual-review finding on `scripts/null/intervention-report.ts`,
 * WP2 of the pathway-interventions study): that module needs the exact same
 * low-tail-floor / high-tail-ceil-minus-one quantile convention for its own
 * `armDistribution`/null-floor computations, and previously carried a
 * hand-duplicated copy whose correctness depended on a doc comment claiming
 * "the same convention" rather than the type system/a shared import
 * guaranteeing it.
 */
export const quantileIndex = (n: number, p: number): number =>
  p <= 0.5 ? Math.floor(p * n) : Math.min(n - 1, Math.ceil(p * n) - 1);

/**
 * Mean/median/std/2.5-97.5 percentiles/IQR of the null set `N` (one mean
 * score per rewired graph), plus the `degenerate` flag the plan's
 * "authored path may be insensitive to topology" risk calls for.
 */
export const nullSummary = (values: readonly number[]): NullSummary => {
  if (values.length === 0) throw new Error('null-stats: nullSummary requires at least one value');
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const meanValue = sorted.reduce((sum, value) => sum + value, 0) / n;
  const medianValue = n % 2 === 1 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  const variance = sorted.reduce((sum, value) => sum + (value - meanValue) ** 2, 0) / n;
  const q1 = sorted[quantileIndex(n, 0.25)];
  const q3 = sorted[quantileIndex(n, 0.75)];
  const iqr = q3 - q1;
  return {
    n,
    mean: meanValue,
    median: medianValue,
    std: Math.sqrt(variance),
    p2_5: sorted[quantileIndex(n, 0.025)],
    p97_5: sorted[quantileIndex(n, 0.975)],
    iqr,
    degenerate: iqr < DEGENERATE_IQR_THRESHOLD
  };
};

export interface RankStatistics {
  readonly kBelow: number;
  readonly kEqual: number;
  /** `(k_below + 0.5 * k_equal) / |N|` — the empirical percentile of the biological score within the null set. */
  readonly bioPercentile: number;
  /** `(k_below + k_equal + 1) / (|N| + 1)` — a rank-based statistic, reported descriptively (no significance language). */
  readonly pLow: number;
  /** `(|N| - k_below + 1) / (|N| + 1)`. */
  readonly pHigh: number;
}

/** Where `bioScore` falls within the null set `nullValues`, per the plan's predeclared rank statistics. */
export const rankStatistics = (nullValues: readonly number[], bioScore: number): RankStatistics => {
  const n = nullValues.length;
  if (n === 0) throw new Error('null-stats: rankStatistics requires a non-empty null set');
  let kBelow = 0;
  let kEqual = 0;
  for (const value of nullValues) {
    if (value < bioScore) kBelow += 1;
    else if (value === bioScore) kEqual += 1;
  }
  return {
    kBelow,
    kEqual,
    bioPercentile: (kBelow + 0.5 * kEqual) / n,
    pLow: (kBelow + kEqual + 1) / (n + 1),
    pHigh: (n - kBelow + 1) / (n + 1)
  };
};

export interface Histogram {
  /** Length `binCount + 1`: bin `i` spans `[edges[i], edges[i + 1])`, except the last bin, which is closed on both ends. */
  readonly edges: readonly number[];
  readonly counts: readonly number[];
}

/**
 * The finest percentile granularity an `n`-point null distribution can
 * express: with `n = 20` rewired trained replicas
 * (`.agents/plans/rewiring-null/03-trained-sample.md`'s "with n = 20 the
 * percentile resolution is 5%"), each additional/fewer null point below the
 * biological score shifts `bioPercentile` by exactly `1/n = 5%` -- a much
 * coarser granularity than the authored null's `n = 500` (0.2%), which is
 * why the trained section's percentile is reported alongside this value
 * rather than read with the same implied precision.
 */
export const percentileResolution = (n: number): number => {
  if (!Number.isInteger(n) || n <= 0) {
    throw new Error(`null-stats: percentileResolution requires a positive integer n, got ${n}`);
  }
  return 1 / n;
};

export interface TrainerSeedSpread {
  readonly min: number;
  readonly max: number;
  readonly range: number;
}

/**
 * Min/max/range across a small set of same-topology, different-trainer-seed
 * scores -- the trained section's `bioTrainerSeedSpread` (the plan's
 * "trainer-seed variance context" line): **trainer-noise variance at fixed
 * (biological) topology**, explicitly not comparable to the null's
 * **topology variance at fixed trainer seed** (the 20 rewired scores, all
 * trained at `replicaSeed`). No overlap-based conclusion between the two
 * may be drawn from this value alone -- see `null-report.ts`'s
 * `renderReportMarkdown` trained section, which states this in prose next
 * to every place this number is printed.
 */
export const trainerSeedSpread = (values: readonly number[]): TrainerSeedSpread => {
  if (values.length === 0) throw new Error('null-stats: trainerSeedSpread requires at least one value');
  const min = Math.min(...values);
  const max = Math.max(...values);
  return { min, max, range: max - min };
};

/**
 * `binCount` equal-width bins spanning `[min(values), max(values)]`
 * (the plan's "30 equal-width histogram bins over
 * `[min(N ∪ {bio, disc}), max(N ∪ {bio, disc})]`" — the caller passes that
 * union in as `range`, but `values` (what gets *counted*) should normally be
 * `N` alone: a dual-review pass caught that an earlier version counted the
 * union too, so the published "null distribution of 500 rewired graphs"
 * histogram silently included the biological and disconnected scores as
 * extra bars (`sum(counts) === 502`, not 500) — see `null-report.ts`'s
 * `buildArtifact` for how `range` and `values` are now passed separately.
 * A single-point span (every value identical) puts every value in the
 * first bin rather than dividing by zero.
 */
export const buildHistogram = (
  values: readonly number[],
  binCount = DEFAULT_HISTOGRAM_BINS,
  range?: readonly [number, number]
): Histogram => {
  if (values.length === 0) throw new Error('null-stats: buildHistogram requires at least one value');
  if (!Number.isInteger(binCount) || binCount <= 0) {
    throw new Error(`null-stats: binCount must be a positive integer, got ${binCount}`);
  }
  const [min, max] = range ?? [Math.min(...values), Math.max(...values)];
  if (!(min <= max)) throw new Error(`null-stats: range [${min}, ${max}] is not a valid ascending range`);
  const span = max - min;
  const edges = Array.from({ length: binCount + 1 }, (_, i) => min + (span * i) / binCount);
  const counts = new Array<number>(binCount).fill(0);
  for (const value of values) {
    if (span === 0) {
      counts[0] += 1;
      continue;
    }
    let bin = Math.floor(((value - min) / span) * binCount);
    if (bin >= binCount) bin = binCount - 1; // value === max lands in the last (closed) bin
    if (bin < 0) bin = 0;
    counts[bin] += 1;
  }
  return { edges, counts };
};
