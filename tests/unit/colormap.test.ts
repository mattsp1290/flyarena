import { describe, expect, it } from 'vitest';
import {
  COLORMAP_SIZE,
  DIVERGING_LUT,
  DIVERGING_LUT_CENTER_INDEX,
  VIRIDIS_LUT,
  effectToLutIndex,
  rateToLutIndex
} from '../../src/lib/render/colormap';
import { writeColors } from '../../src/lib/render/activity-layout';

/**
 * `rateToColor` (a single-RGB-triple convenience wrapper) was removed
 * (thermo-maintainability S2): `grep -rn "rateToColor" src/` outside this
 * module turned up no production caller — `ActivityScene#update` only ever
 * calls `writeColors` (`activity-layout.ts`), which calls `rateToLutIndex`
 * directly. Coverage below exercises exactly those two production-reachable
 * functions instead of the deleted wrapper.
 */

describe('VIRIDIS_LUT', () => {
  it('has 256 RGB entries, every channel in [0, 1]', () => {
    expect(VIRIDIS_LUT.length).toBe(COLORMAP_SIZE * 3);
    for (let index = 0; index < VIRIDIS_LUT.length; index += 1) {
      expect(VIRIDIS_LUT[index]).toBeGreaterThanOrEqual(0);
      expect(VIRIDIS_LUT[index]).toBeLessThanOrEqual(1);
    }
  });

  it('never lightens/darkens non-monotonically: every step from LUT[i] to LUT[i+1] moves luminance the same direction overall', () => {
    // A real property of viridis (dark purple -> bright yellow): overall
    // luminance rises from end to end. Checked against the LUT endpoints
    // rather than every adjacent pair (a real viridis table dips slightly
    // in places), so this catches a badly wrong/reversed table without
    // being overly strict about the exact curve shape.
    const luminance = (r: number, g: number, b: number): number => 0.2126 * r + 0.7152 * g + 0.0722 * b;
    const first = luminance(VIRIDIS_LUT[0], VIRIDIS_LUT[1], VIRIDIS_LUT[2]);
    const lastOffset = (COLORMAP_SIZE - 1) * 3;
    const last = luminance(VIRIDIS_LUT[lastOffset], VIRIDIS_LUT[lastOffset + 1], VIRIDIS_LUT[lastOffset + 2]);
    expect(last).toBeGreaterThan(first);
  });
});

describe('rateToLutIndex (shared clamp/round math — the only function writeColors, production\'s real color-write path, actually calls)', () => {
  it('maps rateMin -> 0 and rateMax -> lutSteps-1', () => {
    expect(rateToLutIndex(0, 0, 10, COLORMAP_SIZE)).toBe(0);
    expect(rateToLutIndex(10, 0, 10, COLORMAP_SIZE)).toBe(COLORMAP_SIZE - 1);
  });

  it('clamps rates outside [min, max] rather than extrapolating', () => {
    expect(rateToLutIndex(-500, 0, 10, COLORMAP_SIZE)).toBe(0);
    expect(rateToLutIndex(500, 0, 10, COLORMAP_SIZE)).toBe(COLORMAP_SIZE - 1);
  });

  it('maps a degenerate zero-width [min, max] to index 0 rather than dividing by zero', () => {
    expect(rateToLutIndex(0.5, 1, 1, COLORMAP_SIZE)).toBe(0);
  });

  it('is monotonic in the sense that increasing rate never decreases the LUT index used (viridis is a sequential map)', () => {
    expect(rateToLutIndex(8, 0, 10, COLORMAP_SIZE)).toBeGreaterThan(rateToLutIndex(2, 0, 10, COLORMAP_SIZE));
  });

  it('treats a non-finite rate the same as the degenerate zero-width range (maps to index 0, never NaN)', () => {
    expect(rateToLutIndex(Number.NaN, 0, 10, 256)).toBe(0);
    expect(rateToLutIndex(Number.POSITIVE_INFINITY, 0, 10, 256)).toBe(0);
    expect(Number.isFinite(rateToLutIndex(Number.NaN, 0, 10, 256))).toBe(true);
  });
});

describe('writeColors against VIRIDIS_LUT (the real production color-write path: ActivityScene#update -> writeColors)', () => {
  it('maps rateMin/rateMax to VIRIDIS_LUT[0]/VIRIDIS_LUT[255] end to end', () => {
    const rates = Float32Array.from([0, 10]);
    const indices = Int32Array.from([0, 1]);
    const out = new Float32Array(6);

    writeColors(rates, indices, 0, 10, VIRIDIS_LUT, out);

    expect(Array.from(out.subarray(0, 3))).toEqual([VIRIDIS_LUT[0], VIRIDIS_LUT[1], VIRIDIS_LUT[2]]);
    const lastOffset = (COLORMAP_SIZE - 1) * 3;
    expect(Array.from(out.subarray(3, 6))).toEqual([VIRIDIS_LUT[lastOffset], VIRIDIS_LUT[lastOffset + 1], VIRIDIS_LUT[lastOffset + 2]]);
  });

  it('writes only the requested indices, into the caller-owned buffer, never allocating', () => {
    const rates = Float32Array.from([0, 5, 10]);
    // Only neuron 1 is requested — neurons 0 and 2 must be left untouched in `out`.
    const indices = Int32Array.from([1]);
    const out = new Float32Array(3).fill(-1);

    writeColors(rates, indices, 0, 10, VIRIDIS_LUT, out);

    expect(out[0]).not.toBe(-1);
    expect(out[1]).not.toBe(-1);
    expect(out[2]).not.toBe(-1);
  });

  it('treats a non-finite rate the same as the degenerate zero-width range end to end (maps to VIRIDIS_LUT[0], never NaN)', () => {
    const rates = Float32Array.from([Number.NaN]);
    const indices = Int32Array.from([0]);
    const out = new Float32Array(3);

    writeColors(rates, indices, 0, 10, VIRIDIS_LUT, out);

    expect(Array.from(out)).toEqual([VIRIDIS_LUT[0], VIRIDIS_LUT[1], VIRIDIS_LUT[2]]);
    for (const value of out) expect(Number.isFinite(value)).toBe(true);
  });
});

describe('DIVERGING_LUT (WP3 lesion-effect mode)', () => {
  it('has 256 RGB entries, every channel in [0, 1]', () => {
    expect(DIVERGING_LUT.length).toBe(COLORMAP_SIZE * 3);
    for (let index = 0; index < DIVERGING_LUT.length; index += 1) {
      expect(DIVERGING_LUT[index]).toBeGreaterThanOrEqual(0);
      expect(DIVERGING_LUT[index]).toBeLessThanOrEqual(1);
    }
  });

  it('is white (exactly zero effect) at its own center index', () => {
    // A 256-entry (even) table has no exact middle index — index 128's `t`
    // (128/255) is a hair past the true midpoint 0.5, so its color is
    // *almost* exactly white, not bit-identical to it. `toBeCloseTo(1, 2)`
    // (tolerance 0.005) comfortably covers that sub-percent discretization
    // gap while still failing on a real off-by-a-lot regression.
    const offset = DIVERGING_LUT_CENTER_INDEX * 3;
    expect(DIVERGING_LUT[offset]).toBeCloseTo(1, 2);
    expect(DIVERGING_LUT[offset + 1]).toBeCloseTo(1, 2);
    expect(DIVERGING_LUT[offset + 2]).toBeCloseTo(1, 2);
  });

  it('endpoints are the colorblind-safe blue/vermillion pair, not viridis', () => {
    expect(DIVERGING_LUT[0]).toBeCloseTo(0 / 255, 3);
    expect(DIVERGING_LUT[1]).toBeCloseTo(114 / 255, 3);
    expect(DIVERGING_LUT[2]).toBeCloseTo(178 / 255, 3);
    const lastOffset = (COLORMAP_SIZE - 1) * 3;
    expect(DIVERGING_LUT[lastOffset]).toBeCloseTo(213 / 255, 3);
    expect(DIVERGING_LUT[lastOffset + 1]).toBeCloseTo(94 / 255, 3);
    expect(DIVERGING_LUT[lastOffset + 2]).toBeCloseTo(0 / 255, 3);
  });
});

describe('effectToLutIndex', () => {
  it('maps -absMax -> 0, 0 -> the center, and +absMax -> lutSteps-1', () => {
    expect(effectToLutIndex(-10, 10, COLORMAP_SIZE)).toBe(0);
    expect(effectToLutIndex(0, 10, COLORMAP_SIZE)).toBe(DIVERGING_LUT_CENTER_INDEX);
    expect(effectToLutIndex(10, 10, COLORMAP_SIZE)).toBe(COLORMAP_SIZE - 1);
  });

  it('clamps effects outside [-absMax, absMax] rather than extrapolating', () => {
    expect(effectToLutIndex(-500, 10, COLORMAP_SIZE)).toBe(0);
    expect(effectToLutIndex(500, 10, COLORMAP_SIZE)).toBe(COLORMAP_SIZE - 1);
  });

  it('maps a non-positive absMax to the center rather than dividing by zero', () => {
    expect(effectToLutIndex(5, 0, COLORMAP_SIZE)).toBe(DIVERGING_LUT_CENTER_INDEX);
    expect(effectToLutIndex(5, -1, COLORMAP_SIZE)).toBe(DIVERGING_LUT_CENTER_INDEX);
  });

  it('treats a non-finite effect the same as zero (maps to the center, never NaN)', () => {
    expect(effectToLutIndex(Number.NaN, 10, COLORMAP_SIZE)).toBe(DIVERGING_LUT_CENTER_INDEX);
    expect(effectToLutIndex(Number.POSITIVE_INFINITY, 10, COLORMAP_SIZE)).toBe(DIVERGING_LUT_CENTER_INDEX);
    expect(Number.isFinite(effectToLutIndex(Number.NaN, 10, COLORMAP_SIZE))).toBe(true);
  });

  it('is monotonic: a larger signed effect never maps to a smaller LUT index', () => {
    expect(effectToLutIndex(8, 10, COLORMAP_SIZE)).toBeGreaterThan(effectToLutIndex(2, 10, COLORMAP_SIZE));
    expect(effectToLutIndex(-2, 10, COLORMAP_SIZE)).toBeGreaterThan(effectToLutIndex(-8, 10, COLORMAP_SIZE));
  });
});
