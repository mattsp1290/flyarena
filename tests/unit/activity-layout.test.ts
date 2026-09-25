import { describe, expect, it } from 'vitest';
import {
  layoutPositions,
  partitionByRole,
  writeColors,
  writeEffectColors,
  writeOutlinePositions,
  type NeuronRole,
  type PositionSource
} from '../../src/lib/render/activity-layout';
import {
  COLORMAP_SIZE,
  DIVERGING_FADE_TARGET,
  DIVERGING_LUT,
  DIVERGING_LUT_CENTER_INDEX,
  effectToLutIndex
} from '../../src/lib/render/colormap';
import { POINT_SIZE } from '../../src/lib/render/activity-constants';

describe('layoutPositions', () => {
  it('centers annotated (soma/tosoma) neurons on their centroid and uniformly scales to unit extent', () => {
    const xyz: ReadonlyArray<readonly [number, number, number] | null> = [
      [0, 0, 0],
      [10, 0, 0],
      [-10, 0, 0],
      [0, 20, 0]
    ];
    const positionSource: readonly PositionSource[] = ['soma', 'soma', 'soma', 'soma'];

    const { points, unavailableIdx } = layoutPositions(xyz, positionSource);

    expect(unavailableIdx.length).toBe(0);
    // Centroid is (0, 5, 0); the largest centered-axis magnitude is |20-5|=15,
    // so the uniform scale factor is 1/15 (never per-axis — proportions are
    // preserved, not stretched to fill a box).
    expect(points[0]).toBeCloseTo(0);
    expect(points[1]).toBeCloseTo(-5 / 15);
    expect(points[2]).toBeCloseTo(0);
    expect(points[3 * 1]).toBeCloseTo(10 / 15);
    expect(points[3 * 3 + 1]).toBeCloseTo(15 / 15);
    for (const value of points) {
      expect(Math.abs(value)).toBeLessThanOrEqual(1 + 1e-9);
    }
  });

  it('treats tosoma coordinates identically to soma ones for centering/scaling — both are real annotated positions', () => {
    const xyz: ReadonlyArray<readonly [number, number, number] | null> = [
      [1, 1, 1],
      [3, 3, 3]
    ];
    const positionSource: readonly PositionSource[] = ['soma', 'tosoma'];
    const { unavailableIdx } = layoutPositions(xyz, positionSource);
    expect(unavailableIdx.length).toBe(0);
  });

  it('places "none" neurons on a separate unavailable strip, never inventing a real coordinate for them', () => {
    const xyz: ReadonlyArray<readonly [number, number, number] | null> = [[0, 0, 0], null, [5, 0, 0], null];
    const positionSource: readonly PositionSource[] = ['soma', 'none', 'soma', 'none'];

    const { points, unavailableIdx } = layoutPositions(xyz, positionSource);

    expect(Array.from(unavailableIdx)).toEqual([1, 3]);
    // Strip neurons share a fixed Y distinct from (and below) the main
    // cloud's [-1, 1] extent, and never both land on the exact same X unless
    // there is only one of them.
    const stripY1 = points[1 * 3 + 1];
    const stripY3 = points[3 * 3 + 1];
    expect(stripY1).toBeLessThan(-1);
    expect(stripY1).toBe(stripY3);
    expect(points[1 * 3]).not.toBe(points[3 * 3]);
  });

  it('an xyz of null is treated as unavailable even if positionSource disagrees (fails safe, never fabricates a position)', () => {
    const xyz: ReadonlyArray<readonly [number, number, number] | null> = [[1, 1, 1], null];
    // Deliberately inconsistent input: positionSource claims 'soma' but xyz is null.
    const positionSource: readonly PositionSource[] = ['soma', 'soma'];
    const { unavailableIdx } = layoutPositions(xyz, positionSource);
    expect(Array.from(unavailableIdx)).toEqual([1]);
  });

  it('wraps a large unavailable set into multiple rows so points do not overlap (regression: 165 real neurons used to cram into one unreadable row)', () => {
    const count = 165;
    const xyz: Array<readonly [number, number, number] | null> = new Array(count).fill(null);
    const positionSource: PositionSource[] = new Array(count).fill('none');
    const { points, unavailableIdx } = layoutPositions(xyz, positionSource);

    expect(unavailableIdx.length).toBe(count);
    const ys = new Set<number>();
    for (let index = 0; index < count; index += 1) ys.add(points[index * 3 + 1]);
    // More than one distinct row (Y level) — the strip actually wrapped.
    expect(ys.size).toBeGreaterThan(1);

    // No two points within the same row sit closer than the minimum spacing
    // (grouping by Y level, then checking sorted X gaps).
    const byRow = new Map<number, number[]>();
    for (let index = 0; index < count; index += 1) {
      const y = points[index * 3 + 1];
      const xs = byRow.get(y) ?? [];
      xs.push(points[index * 3]);
      byRow.set(y, xs);
    }
    for (const xs of byRow.values()) {
      xs.sort((a, b) => a - b);
      for (let i = 1; i < xs.length; i += 1) {
        // Assert against the real `POINT_SIZE` (imported, not a third
        // hardcoded copy of the threshold — thermo-maintainability S1 fix)
        // so this regression test actually fails if a future `POINT_SIZE`
        // change ever shrinks the real point-vs-spacing margin below zero.
        expect(xs[i] - xs[i - 1]).toBeGreaterThan(POINT_SIZE);
      }
    }
  });

  it('handles an all-unavailable input without producing NaN/Infinity', () => {
    const xyz: ReadonlyArray<readonly [number, number, number] | null> = [null, null, null];
    const positionSource: readonly PositionSource[] = ['none', 'none', 'none'];
    const { points, unavailableIdx } = layoutPositions(xyz, positionSource);
    expect(unavailableIdx.length).toBe(3);
    for (const value of points) expect(Number.isFinite(value)).toBe(true);
  });
});

describe('partitionByRole', () => {
  it('groups every neuron index by role, and the three partitions sum to the full positioned neuron count', () => {
    const role: readonly NeuronRole[] = ['sensory', 'bridge', 'bridge', 'descending', 'sensory'];
    const { sensoryIdx, bridgeIdx, descendingIdx } = partitionByRole(role);

    expect(Array.from(sensoryIdx)).toEqual([0, 4]);
    expect(Array.from(bridgeIdx)).toEqual([1, 2]);
    expect(Array.from(descendingIdx)).toEqual([3]);
    expect(sensoryIdx.length + bridgeIdx.length + descendingIdx.length).toBe(role.length);
  });

  it('throws on an unrecognized role rather than silently dropping a neuron', () => {
    const role = ['sensory', 'unknown'] as unknown as readonly NeuronRole[];
    expect(() => partitionByRole(role)).toThrow(/unknown role/i);
  });
});

describe('writeColors', () => {
  // A synthetic, easily-asserted-against 256-entry LUT: index i -> (i/255, 0, 1 - i/255).
  const lut = (() => {
    const table = new Float32Array(256 * 3);
    for (let index = 0; index < 256; index += 1) {
      table[index * 3] = index / 255;
      table[index * 3 + 1] = 0;
      table[index * 3 + 2] = 1 - index / 255;
    }
    return table;
  })();

  it('maps rateMin -> LUT[0] and rateMax -> LUT[255] at the right output offsets for a given index array', () => {
    const rates = new Float32Array([0, 10, 5]);
    const indices = Int32Array.from([1, 2, 0]); // rates 10 (max), 5 (mid), 0 (min)
    const out = new Float32Array(indices.length * 3);

    writeColors(rates, indices, 0, 10, lut, out);

    // k=0 -> neuron 1, rate 10 == max -> LUT[255]
    expect(out[0]).toBeCloseTo(lut[255 * 3]);
    expect(out[1]).toBeCloseTo(lut[255 * 3 + 1]);
    expect(out[2]).toBeCloseTo(lut[255 * 3 + 2]);
    // k=2 -> neuron 0, rate 0 == min -> LUT[0]
    expect(out[6]).toBeCloseTo(lut[0]);
    expect(out[7]).toBeCloseTo(lut[1]);
    expect(out[8]).toBeCloseTo(lut[2]);
  });

  it('clamps rates outside [min, max]', () => {
    const rates = new Float32Array([-1000, 1000]);
    const indices = Int32Array.from([0, 1]);
    const out = new Float32Array(6);

    writeColors(rates, indices, 0, 10, lut, out);

    expect(out[0]).toBeCloseTo(lut[0]);
    expect(out[3]).toBeCloseTo(lut[255 * 3]);
  });

  it('writes only for indices present in the given array, in the array’s own order', () => {
    const rates = new Float32Array([1, 2, 3, 4]);
    const indices = Int32Array.from([3, 0]);
    const out = new Float32Array(6);

    writeColors(rates, indices, 1, 4, lut, out);

    // k=0 -> neuron 3, rate 4 == max
    expect(out[0]).toBeCloseTo(lut[255 * 3]);
    // k=1 -> neuron 0, rate 1 == min
    expect(out[3]).toBeCloseTo(lut[0]);
  });
});

describe('writeEffectColors (WP3 lesion-effect color mode)', () => {
  it('maps -absMax/0/+absMax to DIVERGING_LUT[0]/center/last for an FDR-significant (emphasized) neuron', () => {
    const effect = Float32Array.from([-10, 0, 10]);
    const emphasize = [true, true, true];
    const indices = Int32Array.from([0, 1, 2]);
    const out = new Float32Array(9);

    writeEffectColors(effect, emphasize, indices, 10, DIVERGING_LUT, out);

    expect(Array.from(out.subarray(0, 3))).toEqual([DIVERGING_LUT[0], DIVERGING_LUT[1], DIVERGING_LUT[2]]);
    const centerOffset = DIVERGING_LUT_CENTER_INDEX * 3;
    expect(Array.from(out.subarray(3, 6))).toEqual([
      DIVERGING_LUT[centerOffset],
      DIVERGING_LUT[centerOffset + 1],
      DIVERGING_LUT[centerOffset + 2]
    ]);
    const lastOffset = (COLORMAP_SIZE - 1) * 3;
    expect(Array.from(out.subarray(6, 9))).toEqual([
      DIVERGING_LUT[lastOffset],
      DIVERGING_LUT[lastOffset + 1],
      DIVERGING_LUT[lastOffset + 2]
    ]);
  });

  it('blends a non-FDR-significant neuron 60% toward DIVERGING_FADE_TARGET (a dim neutral, not the LUT\'s bright white center) rather than showing it at full saturation', () => {
    const effect = Float32Array.from([10]);
    const emphasize = [false];
    const indices = Int32Array.from([0]);
    const out = new Float32Array(3);

    writeEffectColors(effect, emphasize, indices, 10, DIVERGING_LUT, out);

    const lutIndex = effectToLutIndex(10, 10, COLORMAP_SIZE) * 3; // the raw (unblended) color this neuron would have gotten
    const expected = [0, 1, 2].map((c) => DIVERGING_LUT[lutIndex + c] * 0.4 + DIVERGING_FADE_TARGET[c] * 0.6);
    expect(out[0]).toBeCloseTo(expected[0], 5);
    expect(out[1]).toBeCloseTo(expected[1], 5);
    expect(out[2]).toBeCloseTo(expected[2], 5);
    // Not the full-saturation color a significant neuron with the same
    // effect would have gotten (the whole point of the blend).
    expect(out[0]).not.toBeCloseTo(DIVERGING_LUT[lutIndex], 5);
    // Not the LUT's own bright white center either — round-2 dual review:
    // blending toward white made a "not reliable" neuron the highest-
    // contrast point on the near-black canvas, the opposite of de-emphasis.
    const centerOffset = DIVERGING_LUT_CENTER_INDEX * 3;
    expect(out[0]).not.toBeCloseTo(DIVERGING_LUT[centerOffset], 2);
  });

  it('a non-significant neuron with exactly zero effect is still dimmed toward DIVERGING_FADE_TARGET, not left at the LUT\'s bright white', () => {
    const effect = Float32Array.from([0]);
    const emphasize = [false];
    const indices = Int32Array.from([0]);
    const out = new Float32Array(3);

    writeEffectColors(effect, emphasize, indices, 10, DIVERGING_LUT, out);

    const centerOffset = DIVERGING_LUT_CENTER_INDEX * 3;
    const expected = [0, 1, 2].map((c) => DIVERGING_LUT[centerOffset + c] * 0.4 + DIVERGING_FADE_TARGET[c] * 0.6);
    expect(out[0]).toBeCloseTo(expected[0], 5);
    expect(out[1]).toBeCloseTo(expected[1], 5);
    expect(out[2]).toBeCloseTo(expected[2], 5);
    // Even a truly zero-effect non-significant neuron reads as de-emphasized
    // (dim), not as the LUT's own near-white zero-effect color.
    expect(out[0]).toBeLessThan(0.9);
  });

  it('writes only the requested indices, into the caller-owned buffer, never allocating, and uses each slot\'s own neuron data for BOTH effect and emphasize (not positional/identity indexing)', () => {
    // Round-2 dual review (Important — Architecture Maintainer): a prior
    // version of this test used `emphasize = [true, true, true]` (uniform),
    // so even though the *effect* indirection was exercised, an
    // `emphasize[k]` (positional) vs `emphasize[indices[k]]` (correct)
    // regression could never be distinguished — every entry was `true`
    // either way. `indices = [1, 0]` (swapped) with genuinely different
    // per-neuron `effect`/`emphasize` values makes *both* a positional-
    // effect bug and a positional-emphasize bug independently detectable at
    // every slot: slot 0 (neuron 1: effect +10, non-significant) would read
    // as neuron 0's data (effect -10, significant) under either bug, and
    // vice versa for slot 1.
    const effect = Float32Array.from([-10, 10, 0]);
    const emphasize = [true, false, true];
    const indices = Int32Array.from([1, 0]);
    const out = new Float32Array(6).fill(-1);

    writeEffectColors(effect, emphasize, indices, 10, DIVERGING_LUT, out);

    const lastOffset = (COLORMAP_SIZE - 1) * 3;
    // slot 0 <- neuron 1 (effect +10, NOT significant): blended toward the fade target, not full saturation.
    const expectedSlot0 = [0, 1, 2].map((c) => DIVERGING_LUT[lastOffset + c] * 0.4 + DIVERGING_FADE_TARGET[c] * 0.6);
    expect(out[0]).toBeCloseTo(expectedSlot0[0], 5);
    expect(out[1]).toBeCloseTo(expectedSlot0[1], 5);
    expect(out[2]).toBeCloseTo(expectedSlot0[2], 5);
    // slot 1 <- neuron 0 (effect -10, significant): full saturation, unblended.
    expect(Array.from(out.subarray(3, 6))).toEqual([DIVERGING_LUT[0], DIVERGING_LUT[1], DIVERGING_LUT[2]]);
    // Neuron 2 (effect 0) is never requested — nothing here matches its color.
    const centerOffset = DIVERGING_LUT_CENTER_INDEX * 3;
    expect(out[0]).not.toBeCloseTo(DIVERGING_LUT[centerOffset], 2);
    expect(out[3]).not.toBeCloseTo(DIVERGING_LUT[centerOffset], 2);
  });

  it('significant and non-significant neurons with the same effect get visibly different colors', () => {
    const effect = Float32Array.from([10, 10]);
    const emphasize = [true, false];
    const indices = Int32Array.from([0, 1]);
    const out = new Float32Array(6);

    writeEffectColors(effect, emphasize, indices, 10, DIVERGING_LUT, out);

    expect(out[0]).not.toBeCloseTo(out[3], 5);
  });
});

describe('writeOutlinePositions (WP3 lesion-effect mode outline-ring overlay — thermo-maintainability I2)', () => {
  // A reversed `emphasize` check here — outlining significant neurons
  // instead of non-significant ones — was mutation-checked by hand: flipping
  // `if (emphasize[neuron]) continue;` to `if (!emphasize[neuron]) continue;`
  // in `activity-layout.ts` and re-running this describe block fails every
  // test below (confirmed during review; reverted afterward).

  it('selects only non-FDR-significant neurons\' positions, packed contiguously in ascending neuron order', () => {
    const basePositions = Float32Array.from([
      0, 0, 0, // neuron 0 (significant)
      1, 1, 1, // neuron 1 (NOT significant)
      2, 2, 2, // neuron 2 (significant)
      3, 3, 3 // neuron 3 (NOT significant)
    ]);
    const emphasize = [true, false, true, false];
    const out = new Float32Array(basePositions.length);

    const count = writeOutlinePositions(basePositions, emphasize, out);

    expect(count).toBe(2);
    expect(Array.from(out.subarray(0, 3))).toEqual([1, 1, 1]);
    expect(Array.from(out.subarray(3, 6))).toEqual([3, 3, 3]);
  });

  it('writes nothing (count 0) when every neuron is FDR-significant', () => {
    const basePositions = Float32Array.from([0, 0, 0, 1, 1, 1]);
    const emphasize = [true, true];
    const out = new Float32Array(basePositions.length);

    expect(writeOutlinePositions(basePositions, emphasize, out)).toBe(0);
  });

  it('writes every neuron, in order, when none are FDR-significant', () => {
    const basePositions = Float32Array.from([0, 0, 0, 1, 1, 1]);
    const emphasize = [false, false];
    const out = new Float32Array(basePositions.length);

    const count = writeOutlinePositions(basePositions, emphasize, out);

    expect(count).toBe(2);
    expect(Array.from(out.subarray(0, 6))).toEqual([0, 0, 0, 1, 1, 1]);
  });

  it('never writes a significant neuron\'s position anywhere in the output, even interleaved with non-significant ones', () => {
    // Neuron 2 (significant, position [9, 9, 9]) must not appear anywhere in
    // `out` — this is the assertion a reversed `emphasize[neuron]` check
    // (outlining significant neurons instead) would fail.
    const basePositions = Float32Array.from([5, 5, 5, 7, 7, 7, 9, 9, 9, 11, 11, 11]);
    const emphasize = [false, true, false, true];
    const out = new Float32Array(basePositions.length);

    const count = writeOutlinePositions(basePositions, emphasize, out);

    expect(count).toBe(2);
    const written = Array.from(out.subarray(0, count * 3));
    expect(written).toEqual([5, 5, 5, 9, 9, 9]);
    expect(written).not.toContain(7);
    expect(written).not.toContain(11);
  });
});
