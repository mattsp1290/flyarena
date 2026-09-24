/**
 * A perceptually-uniform sequential colormap (viridis-like: dark purple at
 * low values, through blue/green, to bright yellow at high values) for the
 * anatomical activity view's rate coloring (WP3, `docs/model-ledger.md`'s
 * "Displayed neural activity" row — **Computed**, never a measured value).
 *
 * The lookup table is generated once at module load from a compact six-term
 * polynomial approximation of the viridis colormap (the same kind of
 * analytic approximation widely used in real-time shaders, where shipping a
 * full sampled table is unnecessary) rather than a `256 x 3` literal —
 * cheaper to keep correct and just as fast once baked into `VIRIDIS_LUT`.
 * Only the resulting `Float32Array` values matter to callers; the exact
 * polynomial coefficients are an implementation detail. This module never
 * claims measured colors — `rateToColor`/`writeColors` (`activity-layout.ts`)
 * both operate on `ConnectomeGraph.metadata.rateMin`/`rateMax`, a *declared*
 * dynamics bound, never a per-frame auto-normalized range (see the plan's
 * "no per-frame auto-normalization" decision — that would exaggerate tiny
 * activity into a misleadingly full-range color).
 */

/** Number of entries in `VIRIDIS_LUT`; also the number of distinguishable color steps `rateToColor` can produce. */
export const COLORMAP_SIZE = 256;

const viridisPolynomial = (t: number): readonly [number, number, number] => {
  const c0: readonly [number, number, number] = [0.2777273272, 0.0054073445, 0.3340998053];
  const c1: readonly [number, number, number] = [0.1050930431, 1.4046135299, 1.3845901626];
  const c2: readonly [number, number, number] = [-0.3308618287, 0.2148475595, 0.095095163];
  const c3: readonly [number, number, number] = [-4.6342304990, -5.7991009734, -19.3324409563];
  const c4: readonly [number, number, number] = [6.2282699363, 14.1799333668, 56.6905526007];
  const c5: readonly [number, number, number] = [4.7763849977, -13.7451453777, -65.3530326334];
  const c6: readonly [number, number, number] = [-5.4354558559, 4.6458526122, 26.3124352496];

  const channel = (i: 0 | 1 | 2): number =>
    c0[i] + t * (c1[i] + t * (c2[i] + t * (c3[i] + t * (c4[i] + t * (c5[i] + t * c6[i])))));

  const clamp01 = (value: number): number => Math.min(1, Math.max(0, value));
  return [clamp01(channel(0)), clamp01(channel(1)), clamp01(channel(2))];
};

const buildViridisLut = (): Float32Array => {
  const lut = new Float32Array(COLORMAP_SIZE * 3);
  for (let index = 0; index < COLORMAP_SIZE; index += 1) {
    const t = index / (COLORMAP_SIZE - 1);
    const [r, g, b] = viridisPolynomial(t);
    lut[index * 3] = r;
    lut[index * 3 + 1] = g;
    lut[index * 3 + 2] = b;
  }
  return lut;
};

/** `256 x 3` RGB lookup table, values in `[0, 1]`. Built once at module load; never mutated. */
export const VIRIDIS_LUT: Float32Array = buildViridisLut();

/**
 * Map `rate` (clamped to `[min, max]`) to an index into an LUT with
 * `lutSteps` entries. Shared by `rateToColor` below (against
 * `VIRIDIS_LUT`/`COLORMAP_SIZE`) and `activity-layout.ts#writeColors`
 * (against a caller-supplied `lut`), so the clamp/round math cannot drift
 * between the two call sites — a real risk dual review flagged, since
 * `writeColors` previously reimplemented this same math independently, with
 * nothing to catch the two copies disagreeing after a future edit to just
 * one of them.
 *
 * `min === max` (a degenerate/zero-width declared range) maps every rate to
 * the bottom of the scale (index 0) rather than dividing by zero. A
 * non-finite `rate` (`NaN`, `+-Infinity` — should never happen given a
 * validated graph, but a single bad neuron must not corrupt its point's
 * color into `NaN`) is treated the same way as `min === max`: both
 * comparisons below are false for `NaN`, which `Number.isFinite` here
 * forces to the safe `t = 0` branch instead.
 */
export const rateToLutIndex = (rate: number, min: number, max: number, lutSteps: number): number => {
  const t = max > min && Number.isFinite(rate) ? (rate - min) / (max - min) : 0;
  const clamped = t < 0 ? 0 : t > 1 ? 1 : t;
  return Math.round(clamped * (lutSteps - 1));
};

/**
 * Map `rate` to an RGB triple from `VIRIDIS_LUT`, writing it into
 * `out[offset..offset+2]`. No allocation: `out` is a caller-owned buffer
 * (typically a `THREE.BufferAttribute`'s backing `Float32Array`), reused
 * every call. See `rateToLutIndex` for the clamping/degenerate-range rules.
 */
export const rateToColor = (rate: number, min: number, max: number, out: Float32Array, offset: number): void => {
  const lutOffset = rateToLutIndex(rate, min, max, COLORMAP_SIZE) * 3;
  out[offset] = VIRIDIS_LUT[lutOffset];
  out[offset + 1] = VIRIDIS_LUT[lutOffset + 1];
  out[offset + 2] = VIRIDIS_LUT[lutOffset + 2];
};
