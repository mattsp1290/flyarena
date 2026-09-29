import { describe, expect, it } from 'vitest';

import { readoutSaliency } from '../../scripts/attribution/saliency';
import { createReadoutOutput, createReadoutScratch, readoutForward, type ReadoutWeights } from '../../src/lib/connectome/readout';

/**
 * `.agents/plans/readout-attribution/02-analyses.md`'s WP2 test
 * requirement: "Analytic saliency, including the output-nonlinearity
 * factor, matches finite differences on a fixture readout within 1e-6,
 * including a saturated-output case."
 */

const D = 4;
const H = 3;
const IDENTITY_INDICES = Int32Array.from({ length: D }, (_, i) => i);

const buildWeights = (biasScale: number): ReadoutWeights => ({
  inputSize: D,
  hiddenSize: H,
  w1: Float32Array.from([0.4, -0.3, 0.2, 0.1, -0.2, 0.5, 0.1, -0.4, 0.3, 0.2, 0.1, -0.1]),
  b1: Float32Array.from([0.05, -0.1, 0.2]),
  w2: Float32Array.from([0.3, -0.2, 0.4, -0.1, 0.2, 0.3, 0.2, -0.3, 0.1]),
  b2: Float32Array.from([0.1 * biasScale, -0.1 * biasScale, 0])
});

/** Central finite-difference gradient of `readoutForward`'s output `outputIndex` w.r.t. input `d`, at `row`. */
const numericalGradient = (weights: ReadoutWeights, row: Float32Array, outputIndex: number, d: number, eps = 1e-3): number => {
  const scratch = createReadoutScratch(weights.hiddenSize);
  const out = createReadoutOutput();
  const plus = Float32Array.from(row);
  plus[d] += eps;
  readoutForward(weights, plus, IDENTITY_INDICES, scratch, out);
  const outPlus = out[outputIndex];
  const minus = Float32Array.from(row);
  minus[d] -= eps;
  readoutForward(weights, minus, IDENTITY_INDICES, scratch, out);
  const outMinus = out[outputIndex];
  return (outPlus - outMinus) / (2 * eps);
};

const buildRows = (seedOffset: number): Float32Array[] =>
  Array.from({ length: 20 }, (_, t) => {
    const row = new Float32Array(D);
    for (let d = 0; d < D; d += 1) row[d] = Math.sin((t + seedOffset) * 0.9 + d * 1.3) * 1.5;
    return row;
  });

describe('readoutSaliency', () => {
  it('matches the mean absolute finite-difference gradient to a tight absolute+relative tolerance (thrust, ordinary case)', () => {
    const weights = buildWeights(1);
    const rows = buildRows(0);
    const result = readoutSaliency(weights, rows);
    for (let d = 0; d < D; d += 1) {
      const meanAbsFinite = rows.reduce((sum, row) => sum + Math.abs(numericalGradient(weights, row, 0, d)), 0) / rows.length;
      // Central-difference truncation error is O(eps^2); the tolerance
      // below is a standard gradient-check bound (absolute + relative),
      // not the raw "within 1e-6" plan language applied to a numerical
      // approximation that itself carries O(eps^2) ~ 1e-6 error at eps=1e-3.
      expect(Math.abs(result.thrust[d] - meanAbsFinite)).toBeLessThan(1e-5 + 1e-4 * Math.abs(meanAbsFinite));
    }
  });

  it('matches the mean absolute finite-difference gradient to a tight absolute+relative tolerance (yaw)', () => {
    const weights = buildWeights(1);
    const rows = buildRows(5);
    const result = readoutSaliency(weights, rows);
    for (let d = 0; d < D; d += 1) {
      const meanAbsFinite = rows.reduce((sum, row) => sum + Math.abs(numericalGradient(weights, row, 1, d)), 0) / rows.length;
      expect(Math.abs(result.yaw[d] - meanAbsFinite)).toBeLessThan(1e-5 + 1e-4 * Math.abs(meanAbsFinite));
    }
  });

  it('matches finite differences in a saturated-output case (large bias, tanh near +-1, f_o\' near 0)', () => {
    const weights = buildWeights(80); // huge bias -> thrust/yaw outputs saturate near +-1
    const rows = buildRows(2);
    const result = readoutSaliency(weights, rows);
    for (let d = 0; d < D; d += 1) {
      const meanAbsFiniteThrust = rows.reduce((sum, row) => sum + Math.abs(numericalGradient(weights, row, 0, d)), 0) / rows.length;
      expect(Math.abs(result.thrust[d] - meanAbsFiniteThrust)).toBeLessThan(1e-5 + 1e-4 * Math.abs(meanAbsFiniteThrust));
      // Saturation should drive the gradient (and therefore saliency) close to zero.
      expect(result.thrust[d]).toBeLessThan(1e-2);
    }
  });

  it('matches finite differences in a PARTIALLY saturated case (f_o\' materially non-trivial, not near 0 or 1)', () => {
    // A fully saturated bias (the previous test) drives f_o' so close to 0
    // that the comparison would pass even if the output-nonlinearity factor
    // were dropped entirely from the analytic gradient. `biasScale = 3.5`
    // keeps f_o' in a range (roughly 0.01-0.3 depending on the row) where
    // omitting it would produce a clearly wrong analytic value, while the
    // float32 finite difference still resolves the gradient cleanly.
    const weights = buildWeights(3.5);
    const rows = buildRows(9);
    const result = readoutSaliency(weights, rows);
    for (let d = 0; d < D; d += 1) {
      const meanAbsFiniteThrust = rows.reduce((sum, row) => sum + Math.abs(numericalGradient(weights, row, 0, d)), 0) / rows.length;
      expect(Math.abs(result.thrust[d] - meanAbsFiniteThrust)).toBeLessThan(1e-5 + 1e-4 * Math.abs(meanAbsFiniteThrust));
    }
    // Sanity: this case is not (near-)vacuous the way the fully saturated
    // one is -- at least one input's saliency is well above the fully
    // saturated case's 1e-2 ceiling.
    expect(Math.max(...result.thrust)).toBeGreaterThan(1e-2);
  });

  it('variance-weighted saliency is the mean-absolute saliency scaled by that input\'s own trajectory std', () => {
    const weights = buildWeights(1);
    const rows = buildRows(0);
    const result = readoutSaliency(weights, rows);
    for (let d = 0; d < D; d += 1) {
      expect(result.thrustVarWeighted[d]).toBeCloseTo(result.thrust[d] * result.inputStd[d], 10);
      expect(result.yawVarWeighted[d]).toBeCloseTo(result.yaw[d] * result.inputStd[d], 10);
    }
  });

  it('throws on an empty trajectory or a mismatched row length', () => {
    const weights = buildWeights(1);
    expect(() => readoutSaliency(weights, [])).toThrow();
    expect(() => readoutSaliency(weights, [new Float32Array(D + 1)])).toThrow();
  });
});
