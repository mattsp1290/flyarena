import { describe, expect, it } from 'vitest';

import { computeIndependence } from '../../scripts/attribution/independence';

describe('computeIndependence', () => {
  it('reports a defined ratio of means when the trained mean exceeds 1', () => {
    const result = computeIndependence([10, 20, 30], [5, 5, 5]);
    expect(result.trainedMean).toBeCloseTo(20, 10);
    expect(result.silencedMean).toBeCloseTo(5, 10);
    expect(result.defined).toBe(true);
    expect(result.ratio).toBeCloseTo(0.25, 10);
  });

  it('is a ratio of means, not a mean of per-episode ratios', () => {
    // Per-episode ratios [10/1, 1/10] would average to 5.05; the ratio of
    // means is mean([1,10])/mean([10,1]) = 1.
    const result = computeIndependence([10, 1], [1, 10]);
    expect(result.ratio).toBeCloseTo(1, 10);
  });

  it('reports the share as undefined when the trained mean is <= 1', () => {
    const atOne = computeIndependence([1, 1, 1], [2, 2, 2]);
    expect(atOne.defined).toBe(false);
    expect(atOne.ratio).toBeNull();

    const belowOne = computeIndependence([0.5, 0.2, -1], [2, 2, 2]);
    expect(belowOne.defined).toBe(false);
    expect(belowOne.ratio).toBeNull();
  });

  it('throws on empty score arrays', () => {
    expect(() => computeIndependence([], [1])).toThrow();
    expect(() => computeIndependence([1], [])).toThrow();
  });
});
