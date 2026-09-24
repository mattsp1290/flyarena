import { describe, expect, it } from 'vitest';
import {
  diffCloseEnough,
  FLOAT_ABS_TOLERANCE,
  FLOAT_REL_TOLERANCE,
  LEAF_NOISE_FLOOR_ABS,
  LEAF_NOISE_FLOOR_REL,
  MAX_INEXACT_LEAVES,
  numbersCloseEnough
} from '../fixtures/cross-arch-tolerance';

/**
 * `tests/fixtures/cross-arch-tolerance.ts` is the only regression gate
 * `golden-traces.test.ts`/`episode.test.ts` run on any architecture other
 * than `GOLDEN_GENERATING_ARCH` (arm64) — its tolerant branch never
 * executes on an arm64 dev machine, since the byte-exact branch is taken
 * there instead. Pin its contract directly so a change to the helper (a
 * dropped integer rule, a loosened tolerance, an unbounded leaf count) is
 * caught here, on every architecture, rather than silently widening what
 * counts as "matches" on x86 CI only.
 */
describe('numbersCloseEnough', () => {
  it('accepts identical values, including -0 vs 0', () => {
    expect(numbersCloseEnough(0.5, 0.5)).toBe(true);
    expect(numbersCloseEnough(0, -0)).toBe(true);
    expect(numbersCloseEnough(-0, 0)).toBe(true);
  });

  it('accepts a sub-tolerance float perturbation (the measured cross-arch scale)', () => {
    // Measured cross-arch divergence was ~2.8e-17 absolute / ~3.1e-16
    // relative -- about one float64 ULP. This is roughly 9 orders of
    // magnitude inside FLOAT_ABS_TOLERANCE/FLOAT_REL_TOLERANCE.
    expect(numbersCloseEnough(0.08954558536141353, 0.08954558536141351)).toBe(true);
    expect(numbersCloseEnough(1.5, 1.5 + Number.EPSILON)).toBe(true);
  });

  it('rejects a float difference over both tolerances', () => {
    expect(numbersCloseEnough(0.5, 0.5 + FLOAT_ABS_TOLERANCE * 10)).toBe(false);
    expect(numbersCloseEnough(1000, 1000 * (1 + FLOAT_REL_TOLERANCE * 10))).toBe(false);
  });

  it('requires integers to match exactly, even a large rngState-scale value off by one', () => {
    expect(numbersCloseEnough(3_100_882_058, 3_100_882_059)).toBe(false);
    expect(numbersCloseEnough(0, 1)).toBe(false);
  });

  it('does not misroute a committed-integer-valued float with tiny drift into the exact-integer rule', () => {
    // expected is an integer (0), actual has drifted by a sub-tolerance
    // amount and is therefore NOT an integer -- must fall through to the
    // tolerance check, not the integer-exact rule (which would wrongly
    // reject it).
    expect(numbersCloseEnough(0, 1e-16)).toBe(true);
    expect(numbersCloseEnough(1e-16, 0)).toBe(true);
  });
});

describe('diffCloseEnough', () => {
  it('reports no mismatches and no inexact leaves for an exact match', () => {
    const result = diffCloseEnough({ a: [0.25, { b: 'c' }], n: 3 }, { a: [0.25, { b: 'c' }], n: 3 }, 'root');
    expect(result.mismatches).toEqual([]);
    expect(result.inexactLeaves).toBe(0);
  });

  it('counts a sub-tolerance, above-noise-floor float leaf as inexact but not a mismatch', () => {
    // Relative nudge (1e-9) chosen to sit strictly between LEAF_NOISE_FLOOR_REL
    // (1e-13) and FLOAT_REL_TOLERANCE (1e-6): above the floor (counted as an
    // inexact leaf) but under the tolerance (not a mismatch).
    const result = diffCloseEnough({ x: 0.5 }, { x: 0.5 * (1 + 1e-9) }, 'root');
    expect(result.mismatches).toEqual([]);
    expect(result.inexactLeaves).toBe(1);
  });

  it('does not count a leaf within the float64 noise floor, even though it differs bit-for-bit', () => {
    // A single float64 ULP -- the scale of the measured real cross-arch
    // drift (~3e-16 relative) -- must stay under LEAF_NOISE_FLOOR_REL/ABS
    // (1e-13) and therefore not count toward MAX_INEXACT_LEAVES. This is
    // what stops world.ts's float64 feedback loop (measured to spread a
    // single perturbed Math.sin call's noise across up to 28 leaves in one
    // profiled case, `fix/golden-cross-arch` PR round 2) from tripping the
    // dense-regression budget on pure noise.
    const x = 0.08954558536141353;
    const result = diffCloseEnough({ x }, { x: x + Number.EPSILON * x }, 'root');
    expect(result.mismatches).toEqual([]);
    expect(result.inexactLeaves).toBe(0);
  });

  it('counts a leaf just above the noise floor', () => {
    const x = 1.5;
    const perturbed = x * (1 + LEAF_NOISE_FLOOR_REL * 10);
    const result = diffCloseEnough({ x }, { x: perturbed }, 'root');
    expect(result.mismatches).toEqual([]);
    expect(result.inexactLeaves).toBe(1);
  });

  it('does not count a near-zero leaf with a large relative but tiny absolute difference', () => {
    // expected is (near) zero: even a "100% relative" difference must stay
    // uncounted if it is absolutely tiny -- otherwise a leaf that legitimately
    // rounds to ~0 on one architecture and to a denormal-scale value on
    // another would falsely count as a dense-regression signal.
    const result = diffCloseEnough({ x: 0 }, { x: LEAF_NOISE_FLOOR_ABS / 10 }, 'root');
    expect(result.inexactLeaves).toBe(0);
  });

  it('reports an over-tolerance float leaf as a mismatch', () => {
    const result = diffCloseEnough({ x: 0.5 }, { x: 0.5001 }, 'root');
    expect(result.mismatches).toHaveLength(1);
    expect(result.mismatches[0]).toContain('root.x');
    expect(result.inexactLeaves).toBe(1);
  });

  it('reports an integer mismatch even though it would pass the relative tolerance', () => {
    // 4_000_000_001 is within FLOAT_REL_TOLERANCE of 4_000_000_000, but
    // both are integers, so the exact-match rule must still reject it.
    const result = diffCloseEnough({ rngState: 4_000_000_000 }, { rngState: 4_000_000_001 }, 'root');
    expect(result.mismatches).toHaveLength(1);
  });

  it('reports array length drift', () => {
    const result = diffCloseEnough({ a: [1, 2] }, { a: [1, 2, 3] }, 'root');
    expect(result.mismatches).toEqual(['root.a: expected length 2, got 3']);
  });

  it('reports a key missing from actual and a key unexpected in actual, separately', () => {
    const result = diffCloseEnough({ a: 1, b: 2 }, { a: 1, c: 3 }, 'root');
    expect(result.mismatches).toHaveLength(2);
    expect(result.mismatches.some((m) => m.includes('root.b') && m.includes('missing'))).toBe(true);
    expect(result.mismatches.some((m) => m.includes('root.c') && m.includes('unexpected'))).toBe(true);
  });

  it('reports a string mismatch exactly, with no tolerance', () => {
    const result = diffCloseEnough({ configFingerprint: 'v1|a=1' }, { configFingerprint: 'v1|a=2' }, 'root');
    expect(result.mismatches).toHaveLength(1);
  });

  it('surfaces a dense sub-tolerance regression as a large inexactLeaves count, not as mismatches', () => {
    // Simulates the empirically-verified failure mode this file exists to
    // catch: many numeric leaves each shift by a sub-tolerance amount (a
    // uniform float32-ULP-scale weight change), which individually pass
    // numbersCloseEnough but collectively indicate a real behavior change.
    // diffCloseEnough itself only counts; the MAX_INEXACT_LEAVES budget
    // check (applied by the tests that use this helper) is what turns the
    // count into a failure -- verified here directly against the exported
    // budget so a change to either drifts loudly.
    const width = MAX_INEXACT_LEAVES + 5;
    // A relative nudge (not a fixed absolute Number.EPSILON) so every leaf
    // actually changes bit-for-bit regardless of magnitude: adding a fixed
    // tiny absolute value can round back to the original float once the
    // value's own ULP exceeds that constant, which would silently under-count.
    const expected = { weights: Array.from({ length: width }, (_, i) => i + 0.5) };
    const actual = { weights: Array.from({ length: width }, (_, i) => (i + 0.5) * (1 + 1e-10)) };
    const result = diffCloseEnough(expected, actual, 'root');
    expect(result.mismatches).toEqual([]);
    expect(result.inexactLeaves).toBeGreaterThan(MAX_INEXACT_LEAVES);
  });
});
