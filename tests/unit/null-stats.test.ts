import { describe, expect, it } from 'vitest';

import {
  DEGENERATE_IQR_THRESHOLD,
  buildHistogram,
  graphStats,
  nullSummary,
  rankStatistics
} from '../../scripts/null/null-stats';

/**
 * Direct coverage for `scripts/null/null-stats.ts`'s predeclared statistics
 * (`.agents/plans/rewiring-null/02-authored-null-evaluation.md`): the
 * percentile/rank formulas, tie handling, histogram bin edges/counts, and
 * the degenerate-IQR flag, all against hand-computable inputs.
 */

describe('graphStats', () => {
  it('delegates to conditionStats/conditionRng and is deterministic for a fixed seed+label', () => {
    const values = [1, 2, 3, 4, 5];
    const a = graphStats(values, 42, 'rewired-3', 200);
    const b = graphStats(values, 42, 'rewired-3', 200);
    expect(a).toEqual(b);
    expect(a.mean).toBe(3);
    expect(a.median).toBe(3);
    expect(a.ci95[0]).toBeLessThanOrEqual(a.ci95[1]);
  });

  it('different labels give different (independent) bootstrap draws', () => {
    const values = [1, 2, 3, 4, 5, 100];
    const a = graphStats(values, 42, 'rewired-3', 200);
    const b = graphStats(values, 42, 'rewired-4', 200);
    expect(a.ci95).not.toEqual(b.ci95);
  });
});

describe('nullSummary', () => {
  it('computes mean/median/std/percentiles/IQR on a known small array', () => {
    // 1..10: mean 5.5, median 5.5, population std sqrt(8.25).
    const values = Array.from({ length: 10 }, (_, i) => i + 1);
    const summary = nullSummary(values);
    expect(summary.n).toBe(10);
    expect(summary.mean).toBe(5.5);
    expect(summary.median).toBe(5.5);
    expect(summary.std).toBeCloseTo(Math.sqrt(8.25), 10);
    // quantileIndex(10, 0.025) = floor(0.25) = 0 -> sorted[0] = 1
    expect(summary.p2_5).toBe(1);
    // quantileIndex(10, 0.975) = min(9, ceil(9.75) - 1) = min(9, 9) = 9 -> sorted[9] = 10
    expect(summary.p97_5).toBe(10);
    // q1 index = floor(2.5) = 2 -> sorted[2] = 3; q3 index = min(9, ceil(7.5)-1) = min(9,7) = 7 -> sorted[7] = 8
    expect(summary.iqr).toBe(5);
    expect(summary.degenerate).toBe(false);
  });

  it('flags a degenerate (near-zero-IQR) null distribution', () => {
    const values = Array.from({ length: 20 }, () => 1);
    const summary = nullSummary(values);
    expect(summary.iqr).toBe(0);
    expect(summary.iqr).toBeLessThan(DEGENERATE_IQR_THRESHOLD);
    expect(summary.degenerate).toBe(true);
  });

  it('does not flag an IQR just above the threshold as degenerate', () => {
    const epsilon = DEGENERATE_IQR_THRESHOLD * 10;
    // Values chosen so q1/q3 land exactly on the low/high halves and differ by `epsilon`.
    const values = [0, 0, 0, 0, epsilon, epsilon, epsilon, epsilon];
    const summary = nullSummary(values);
    expect(summary.iqr).toBeCloseTo(epsilon, 15);
    expect(summary.degenerate).toBe(false);
  });

  it('throws on an empty array', () => {
    expect(() => nullSummary([])).toThrow(/at least one value/);
  });
});

describe('rankStatistics', () => {
  it('computes k_below/k_equal, bioPercentile, and the rank statistics on a known set', () => {
    // N = [1, 2, 3, 4, 5], bioScore = 3: k_below = 2 (1, 2), k_equal = 1 (3).
    const nullValues = [1, 2, 3, 4, 5];
    const result = rankStatistics(nullValues, 3);
    expect(result.kBelow).toBe(2);
    expect(result.kEqual).toBe(1);
    // bioPercentile = (2 + 0.5*1) / 5 = 0.5
    expect(result.bioPercentile).toBeCloseTo(0.5, 12);
    // pLow = (2 + 1 + 1) / (5 + 1) = 4/6
    expect(result.pLow).toBeCloseTo(4 / 6, 12);
    // pHigh = (5 - 2 + 1) / (5 + 1) = 4/6
    expect(result.pHigh).toBeCloseTo(4 / 6, 12);
  });

  it('bioScore below every null value: kBelow = 0, kEqual = 0', () => {
    const result = rankStatistics([10, 20, 30], 1);
    expect(result.kBelow).toBe(0);
    expect(result.kEqual).toBe(0);
    expect(result.bioPercentile).toBe(0);
    expect(result.pLow).toBeCloseTo(1 / 4, 12);
    expect(result.pHigh).toBeCloseTo(4 / 4, 12);
  });

  it('bioScore above every null value: kBelow = n', () => {
    const result = rankStatistics([10, 20, 30], 100);
    expect(result.kBelow).toBe(3);
    expect(result.kEqual).toBe(0);
    expect(result.bioPercentile).toBe(1);
    expect(result.pLow).toBeCloseTo(4 / 4, 12);
    expect(result.pHigh).toBeCloseTo(1 / 4, 12);
  });

  it('every null value tied with bioScore', () => {
    const result = rankStatistics([5, 5, 5, 5], 5);
    expect(result.kBelow).toBe(0);
    expect(result.kEqual).toBe(4);
    expect(result.bioPercentile).toBe(0.5);
  });

  it('throws on an empty null set', () => {
    expect(() => rankStatistics([], 1)).toThrow(/non-empty null set/);
  });
});

describe('buildHistogram', () => {
  it('bins span [min, max] and counts sum to the input length', () => {
    const values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const hist = buildHistogram(values, 5);
    expect(hist.edges).toHaveLength(6);
    expect(hist.edges[0]).toBe(0);
    expect(hist.edges[5]).toBe(10);
    expect(hist.counts).toHaveLength(5);
    expect(hist.counts.reduce((sum, c) => sum + c, 0)).toBe(values.length);
  });

  it('the maximum value lands in the last (closed) bin, not spilling past it', () => {
    const hist = buildHistogram([0, 10], 5);
    expect(hist.counts[4]).toBe(1); // the value 10 (== max)
    expect(hist.counts[0]).toBe(1); // the value 0 (== min)
  });

  it('a single-point span (every value identical) puts everything in the first bin', () => {
    const hist = buildHistogram([7, 7, 7], 4);
    expect(hist.counts[0]).toBe(3);
    expect(hist.counts.slice(1)).toEqual([0, 0, 0]);
  });

  it('throws on an empty array', () => {
    expect(() => buildHistogram([])).toThrow(/at least one value/);
  });

  it('throws on a non-positive bin count', () => {
    expect(() => buildHistogram([1, 2], 0)).toThrow(/positive integer/);
  });
});
