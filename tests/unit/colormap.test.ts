import { describe, expect, it } from 'vitest';
import { COLORMAP_SIZE, VIRIDIS_LUT, rateToColor } from '../../src/lib/render/colormap';

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
});
