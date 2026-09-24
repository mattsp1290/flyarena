import { describe, expect, it } from 'vitest';

import { bootstrapCI, conditionRng, conditionStats, mean, median, pairedStats, std } from '../../scripts/training/stats';

/**
 * Direct coverage for `scripts/training/stats.ts` — extracted from
 * `evaluate.ts` (round-3 review) as a pure statistics module, previously
 * exercised only indirectly through `evaluate.test.ts`'s full-pipeline
 * fixture runs. These pin the published statistics (mean/median/std,
 * bootstrap CI quantile indices, `conditionRng`'s per-label seeding)
 * directly, against small hand-computable inputs.
 */

describe('mean / median / std', () => {
  it('mean of a known small array', () => {
    expect(mean([1, 2, 3, 4])).toBe(2.5);
  });

  it('median: odd-length array is the middle element', () => {
    expect(median([5, 1, 3])).toBe(3);
  });

  it('median: even-length array is the average of the two middle elements', () => {
    expect(median([1, 2, 3, 4])).toBe(2.5);
  });

  it('median does not mutate its input array (sorts a copy)', () => {
    const values = [3, 1, 2];
    median(values);
    expect(values).toEqual([3, 1, 2]);
  });

  it('std: population standard deviation of a known small array', () => {
    // [2, 4, 4, 4, 5, 5, 7, 9]: mean 5, population variance 4, std 2 (textbook example).
    expect(std([2, 4, 4, 4, 5, 5, 7, 9])).toBeCloseTo(2, 10);
  });

  it('std of a constant array is exactly 0', () => {
    expect(std([7, 7, 7, 7])).toBe(0);
  });
});

describe('bootstrapCI', () => {
  it('is deterministic for a fixed rng and reproduces known quantile indices', () => {
    // A trivial rng that cycles through fixed values in [0, 1); with n = 4,
    // Math.floor(rng() * 4) is fully predictable, so the resampled
    // statistic sequence (and therefore the sorted-quantile CI) can be
    // hand-verified rather than merely "doesn't crash".
    const sequence = [0, 0.24, 0.5, 0.99];
    let i = 0;
    const rng = (): number => sequence[i++ % sequence.length];

    const values = [10, 20, 30, 40];
    const resamples = 8;
    const draws: number[] = [];
    const ci = bootstrapCI(values.length, resamples, rng, (pickIndex) => {
      const picked = values[pickIndex()];
      draws.push(picked);
      return picked;
    });

    // lowIndex = floor(0.025 * 8) = 0; highIndex = min(7, ceil(0.975*8)-1) = min(7, 8-1) = 7.
    const sortedDraws = [...draws].sort((a, b) => a - b);
    expect(ci).toEqual([sortedDraws[0], sortedDraws[7]]);
  });

  it('a zero-variance statistic collapses the CI to a single point', () => {
    const rng = (): number => Math.random();
    const ci = bootstrapCI(5, 100, rng, () => 42);
    expect(ci).toEqual([42, 42]);
  });
});

describe('conditionStats / pairedStats', () => {
  const fixedRng = (): (() => number) => {
    let state = 0.1234;
    return () => {
      // Deterministic pseudo-sequence, not cryptographically anything —
      // just needs to be a valid () => number in [0, 1) for bootstrapCI.
      state = (state * 9301 + 49297) % 233280;
      return state / 233280;
    };
  };

  it('conditionStats reports n/mean/median/std matching the plain functions', () => {
    const values = [1, 2, 3, 4, 5];
    const result = conditionStats(values, 200, fixedRng());
    expect(result.n).toBe(5);
    expect(result.mean).toBe(mean(values));
    expect(result.median).toBe(median(values));
    expect(result.std).toBe(std(values));
    expect(result.ci95[0]).toBeLessThanOrEqual(result.ci95[1]);
  });

  it('pairedStats computes elementwise a[i] - b[i], mean-of-differences', () => {
    const a = [10, 20, 30];
    const b = [1, 2, 3];
    const result = pairedStats(a, b, 200, fixedRng());
    expect(result.n).toBe(3);
    expect(result.meanDifference).toBeCloseTo((9 + 18 + 27) / 3, 10);
  });

  it('pairedStats throws on mismatched-length series', () => {
    expect(() => pairedStats([1, 2], [1], 10, fixedRng())).toThrow(/equal length/);
  });
});

describe('conditionRng', () => {
  it('is deterministic: the same (seed, label) always produces the same stream', () => {
    const a = conditionRng(12345, 'biological|trained|101');
    const b = conditionRng(12345, 'biological|trained|101');
    const drawsA = Array.from({ length: 5 }, () => a());
    const drawsB = Array.from({ length: 5 }, () => b());
    expect(drawsA).toEqual(drawsB);
  });

  it('different labels give different (independent) streams for the same seed', () => {
    const a = conditionRng(12345, 'biological|trained|101');
    const b = conditionRng(12345, 'biological|silenced|101');
    expect(a()).not.toBe(b());
  });

  it('different bootstrap seeds give different streams for the same label', () => {
    const a = conditionRng(1, 'biological|trained|101');
    const b = conditionRng(2, 'biological|trained|101');
    expect(a()).not.toBe(b());
  });

  it('every draw is in [0, 1)', () => {
    const rng = conditionRng(999, 'some-label');
    for (let i = 0; i < 50; i += 1) {
      const value = rng();
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    }
  });
});
