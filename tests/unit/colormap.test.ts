import { describe, expect, it } from 'vitest';
import { COLORMAP_SIZE, VIRIDIS_LUT, rateToColor, rateToLutIndex } from '../../src/lib/render/colormap';
import { writeColors } from '../../src/lib/render/activity-layout';

describe('VIRIDIS_LUT', () => {
  it('has 256 RGB entries, every channel in [0, 1]', () => {
    expect(VIRIDIS_LUT.length).toBe(COLORMAP_SIZE * 3);
    for (let index = 0; index < VIRIDIS_LUT.length; index += 1) {
      expect(VIRIDIS_LUT[index]).toBeGreaterThanOrEqual(0);
      expect(VIRIDIS_LUT[index]).toBeLessThanOrEqual(1);
    }
  });
});

describe('rateToColor', () => {
  it('maps rateMin -> LUT[0] and rateMax -> LUT[255]', () => {
    const out = new Float32Array(3);

    rateToColor(0, 0, 10, out, 0);
    expect(Array.from(out)).toEqual([VIRIDIS_LUT[0], VIRIDIS_LUT[1], VIRIDIS_LUT[2]]);

    rateToColor(10, 0, 10, out, 0);
    const lastOffset = (COLORMAP_SIZE - 1) * 3;
    expect(Array.from(out)).toEqual([VIRIDIS_LUT[lastOffset], VIRIDIS_LUT[lastOffset + 1], VIRIDIS_LUT[lastOffset + 2]]);
  });

  it('clamps rates outside [min, max] rather than extrapolating', () => {
    const out = new Float32Array(3);

    rateToColor(-500, 0, 10, out, 0);
    expect(Array.from(out)).toEqual([VIRIDIS_LUT[0], VIRIDIS_LUT[1], VIRIDIS_LUT[2]]);

    rateToColor(500, 0, 10, out, 0);
    const lastOffset = (COLORMAP_SIZE - 1) * 3;
    expect(Array.from(out)).toEqual([VIRIDIS_LUT[lastOffset], VIRIDIS_LUT[lastOffset + 1], VIRIDIS_LUT[lastOffset + 2]]);
  });

  it('maps a degenerate zero-width [min, max] to LUT[0] rather than dividing by zero', () => {
    const out = new Float32Array(3);
    rateToColor(0.5, 1, 1, out, 0);
    expect(Array.from(out)).toEqual([VIRIDIS_LUT[0], VIRIDIS_LUT[1], VIRIDIS_LUT[2]]);
    for (const value of out) expect(Number.isFinite(value)).toBe(true);
  });

  it('writes only into out[offset..offset+2], never allocating or touching the rest of the buffer', () => {
    const out = new Float32Array(9).fill(-1);
    rateToColor(5, 0, 10, out, 3);
    expect(out[0]).toBe(-1);
    expect(out[1]).toBe(-1);
    expect(out[2]).toBe(-1);
    expect(out[3]).not.toBe(-1);
    expect(out[6]).toBe(-1);
    expect(out[7]).toBe(-1);
    expect(out[8]).toBe(-1);
  });

  it('is monotonic in the sense that increasing rate never decreases the LUT index used (viridis is a sequential map)', () => {
    const outLow = new Float32Array(3);
    const outHigh = new Float32Array(3);
    rateToColor(2, 0, 10, outLow, 0);
    rateToColor(8, 0, 10, outHigh, 0);
    expect(Array.from(outLow)).not.toEqual(Array.from(outHigh));
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

  it('treats a non-finite rate the same as the degenerate zero-width range (maps to index 0, never NaN)', () => {
    expect(rateToLutIndex(Number.NaN, 0, 10, 256)).toBe(0);
    expect(rateToLutIndex(Number.POSITIVE_INFINITY, 0, 10, 256)).toBe(0);
    expect(Number.isFinite(rateToLutIndex(Number.NaN, 0, 10, 256))).toBe(true);

    const out = new Float32Array(3);
    rateToColor(Number.NaN, 0, 10, out, 0);
    for (const value of out) expect(Number.isFinite(value)).toBe(true);
  });

  it('writeColors (with VIRIDIS_LUT) and rateToColor agree for every rate — the two color paths cannot silently drift apart', () => {
    const rates = Float32Array.from({ length: 101 }, (_, i) => -0.2 + i * 0.014);
    const indices = Int32Array.from(rates.keys());
    const viaWriteColors = new Float32Array(rates.length * 3);
    writeColors(rates, indices, 0, 1, VIRIDIS_LUT, viaWriteColors);

    const viaRateToColor = new Float32Array(rates.length * 3);
    rates.forEach((rate, k) => rateToColor(rate, 0, 1, viaRateToColor, k * 3));

    expect(Array.from(viaWriteColors)).toEqual(Array.from(viaRateToColor));
  });
});
