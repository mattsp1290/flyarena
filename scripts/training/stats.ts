import { createHash } from 'node:crypto';

import { mulberry32 } from '../../src/lib/random/mulberry32';

/**
 * Statistics primitives for `scripts/training/evaluate.ts`: mean/median/std,
 * seeded bootstrap CIs, and paired differences, plus `conditionRng`, the
 * per-label-independent bootstrap RNG derivation. Extracted from
 * `evaluate.ts` (round-1's `graph-provenance` S9, unaddressed across two fix
 * rounds while `evaluate.ts` grew) as a pure, self-contained module: nothing
 * here reads a run directory or writes a file.
 */

export interface ConditionStats {
  readonly n: number;
  readonly mean: number;
  readonly median: number;
  /** Population standard deviation (divide-by-n): descriptive, not inferential (the CI is). */
  readonly std: number;
  readonly ci95: readonly [number, number];
}

export interface PairedStats {
  readonly n: number;
  readonly meanDifference: number;
  readonly ci95: readonly [number, number];
}

export const mean = (values: readonly number[]): number =>
  values.reduce((sum, value) => sum + value, 0) / values.length;

export const median = (values: readonly number[]): number => {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  return n % 2 === 1 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
};

export const std = (values: readonly number[]): number => {
  const m = mean(values);
  const variance = values.reduce((sum, value) => sum + (value - m) ** 2, 0) / values.length;
  return Math.sqrt(variance);
};

/** Bootstrap the 95% CI of `statistic(resample)` over `resamples` seeded, with-replacement resamples. */
export const bootstrapCI = (
  n: number,
  resamples: number,
  rng: () => number,
  statistic: (pickIndex: () => number) => number
): readonly [number, number] => {
  const draws = new Array<number>(resamples);
  for (let r = 0; r < resamples; r += 1) {
    draws[r] = statistic(() => Math.floor(rng() * n));
  }
  draws.sort((a, b) => a - b);
  const lowIndex = Math.floor(0.025 * resamples);
  const highIndex = Math.min(resamples - 1, Math.ceil(0.975 * resamples) - 1);
  return [draws[lowIndex], draws[highIndex]];
};

export const conditionStats = (
  values: readonly number[],
  resamples: number,
  rng: () => number
): ConditionStats => ({
  n: values.length,
  mean: mean(values),
  median: median(values),
  std: std(values),
  ci95: bootstrapCI(values.length, resamples, rng, (pickIndex) => {
    let sum = 0;
    for (let i = 0; i < values.length; i += 1) sum += values[pickIndex()];
    return sum / values.length;
  })
});

/** Paired difference `a[i] - b[i]` for same-seed pairs, with a seeded bootstrap CI on the mean difference. */
export const pairedStats = (
  a: readonly number[],
  b: readonly number[],
  resamples: number,
  rng: () => number
): PairedStats => {
  if (a.length !== b.length) throw new Error('evaluate: paired series must have equal length');
  const diffs = a.map((value, index) => value - b[index]);
  return {
    n: diffs.length,
    meanDifference: mean(diffs),
    ci95: bootstrapCI(diffs.length, resamples, rng, (pickIndex) => {
      let sum = 0;
      for (let i = 0; i < diffs.length; i += 1) sum += diffs[pickIndex()];
      return sum / diffs.length;
    })
  };
};

/**
 * A fresh, independently-seeded bootstrap RNG for one statistic, keyed by a
 * stable label (e.g. `"biological|trained|101"`). Each `conditionStats`/
 * `pairedStats` call gets its own stream derived from `bootstrapSeed` +
 * its label (sha256, first 4 bytes as a uint32) rather than all of them
 * sharing one sequentially-consumed `mulberry32` stream: with a shared
 * stream, a statistic's resample draws — and therefore its CI bounds —
 * depended on how many other statistics happened to be computed before it,
 * so adding or removing an unrelated arm/replica from `--runs` would
 * silently shift every later CI even though that arm/replica's own data
 * never changed. Per-label seeding makes every statistic's CI a pure
 * function of (`bootstrapSeed`, its own label, its own data) — independent
 * of what else this invocation evaluated, not merely independent of `--runs`
 * argument order.
 */
export const conditionRng = (bootstrapSeed: number, label: string): (() => number) => {
  const digest = createHash('sha256').update(`${bootstrapSeed}|${label}`).digest();
  return mulberry32(digest.readUInt32LE(0));
};
