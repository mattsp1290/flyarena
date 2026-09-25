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
 * claims measured colors — `writeColors` (`activity-layout.ts`, the only
 * color-write path production actually calls — see `ActivityScene#update`)
 * operates on `ConnectomeGraph.metadata.rateMin`/`rateMax`, a *declared*
 * dynamics bound, never a per-frame auto-normalized range (see the plan's
 * "no per-frame auto-normalization" decision — that would exaggerate tiny
 * activity into a misleadingly full-range color).
 */

/** Number of entries in `VIRIDIS_LUT`; also the number of distinguishable color steps `rateToLutIndex` can produce. */
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
 * `lutSteps` entries. Used by `activity-layout.ts#writeColors` (against a
 * caller-supplied `lut`) — the only color-write path production actually
 * calls (`ActivityScene#update`). Pulled out as its own pure function so
 * that math has one home rather than being reimplemented independently
 * wherever a rate needs to become a LUT index — a real risk dual review
 * flagged, since `writeColors` previously duplicated this same math inline,
 * with nothing to catch it drifting from any other copy after a future edit.
 *
 * (`rateToColor`, an earlier single-RGB-triple convenience wrapper around
 * this function, was removed — thermo-maintainability S2 — once it became
 * unreachable from any production code path: `writeColors` is the only
 * caller of this function that ships.)
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
 * Diverging colormap for the anatomical activity view's lesion-effect color
 * mode (WP3, `.agents/plans/lesion-atlas/03-activity-lesion-mode.md`):
 * colorblind-safe blue (negative effect) -> white (zero effect) -> vermillion
 * (positive effect), the two endpoint hues taken from the Okabe-Ito
 * colorblind-safe palette (`#0072B2`/`#D55E00`) rather than the red/green
 * pair a diverging map would naively reach for. Built the same
 * generate-once-at-module-load way as `VIRIDIS_LUT` above.
 *
 * This is a *sequential-looking* table read divergingly by `effectToLutIndex`
 * below: index 0 is the most-negative effect, the entry nearest the middle is
 * (almost exactly — see `DIVERGING_LUT_CENTER_INDEX`'s own doc comment on the
 * one-index discretization gap an even `COLORMAP_SIZE` leaves) zero effect,
 * and the last index is the most-positive effect — the same "caller-supplied
 * LUT, shared clamp/index math" shape `VIRIDIS_LUT`/`rateToLutIndex` already
 * establish for `writeColors`, so `writeEffectColors` (`activity-layout.ts`)
 * can reuse that pattern rather than inventing a second one.
 */
const DIVERGING_LOW: readonly [number, number, number] = [0 / 255, 114 / 255, 178 / 255]; // Okabe-Ito blue, #0072B2
const DIVERGING_MID: readonly [number, number, number] = [1, 1, 1]; // white: exactly zero effect
const DIVERGING_HIGH: readonly [number, number, number] = [213 / 255, 94 / 255, 0 / 255]; // Okabe-Ito vermillion, #D55E00

/**
 * The neutral color non-FDR-significant neurons are blended toward
 * (`activity-layout.ts#writeEffectColors`) — deliberately *not* `DIVERGING_MID`
 * (pure white). Round-2 dual review (Important): the activity view's canvas
 * background is near-black (`ActivityScene.ts`'s `#05070c`), so blending
 * toward white makes an "unreliable, de-emphasized" neuron the *highest*-
 * contrast, most visually prominent point on screen — the opposite of the
 * intended effect, and especially severe for the biological graph, where the
 * shared `absMax` (dominated by the rewired graph's larger effects) leaves
 * even FDR-significant biological neurons only faintly tinted to begin with.
 * This dim, low-saturation neutral instead reads as visually quiet against
 * the dark canvas, the same way `ActivityScene.ts#NO_DATA_COLOR` is a muted
 * (not bright) grey for the same reason — but deliberately a different,
 * darker value than `NO_DATA_COLOR` so "unreliable effect, still has atlas
 * coverage" is never visually confused with "no lesion data at all" (a real
 * risk a review pass flagged: the two states mean different things and must
 * stay visually distinguishable).
 */
export const DIVERGING_FADE_TARGET: readonly [number, number, number] = [0.14, 0.15, 0.17];

const buildDivergingLut = (): Float32Array => {
  const lut = new Float32Array(COLORMAP_SIZE * 3);
  for (let index = 0; index < COLORMAP_SIZE; index += 1) {
    const t = index / (COLORMAP_SIZE - 1);
    const [from, to, localT] =
      t < 0.5 ? ([DIVERGING_LOW, DIVERGING_MID, t / 0.5] as const) : ([DIVERGING_MID, DIVERGING_HIGH, (t - 0.5) / 0.5] as const);
    lut[index * 3] = from[0] + (to[0] - from[0]) * localT;
    lut[index * 3 + 1] = from[1] + (to[1] - from[1]) * localT;
    lut[index * 3 + 2] = from[2] + (to[2] - from[2]) * localT;
  }
  return lut;
};

/** `256 x 3` RGB lookup table, values in `[0, 1]`, centered at zero effect. Built once at module load; never mutated. */
export const DIVERGING_LUT: Float32Array = buildDivergingLut();

/**
 * Map a signed `effect` (clamped to `[-absMax, absMax]`) to an index into a
 * diverging LUT with `lutSteps` entries, centered at zero. Mirrors
 * `rateToLutIndex`'s shape/degenerate-input handling exactly (a
 * non-positive `absMax`, or a non-finite `effect`, both map to the LUT's
 * own center — never NaN, never a division by zero) so a caller can reuse
 * the same "shared clamp/round math, real color-write path" testing
 * discipline this module already established for the sequential viridis
 * path.
 */
export const effectToLutIndex = (effect: number, absMax: number, lutSteps: number): number => {
  const t = absMax > 0 && Number.isFinite(effect) ? effect / absMax : 0;
  const clamped = t < -1 ? -1 : t > 1 ? 1 : t;
  return Math.round(((clamped + 1) / 2) * (lutSteps - 1));
};

/**
 * The LUT index `effectToLutIndex` maps zero effect to, for `DIVERGING_LUT`'s
 * own `COLORMAP_SIZE` step count. Reference/test constant only — no
 * production caller reads this index at runtime (`writeEffectColors` blends
 * non-FDR-significant neurons toward `DIVERGING_FADE_TARGET` above, not
 * toward this index's own LUT color; see that constant's doc comment for
 * why). `COLORMAP_SIZE` (256) is even, so there is no single index whose `t`
 * is exactly `0.5`: `effectToLutIndex(0, ...)` rounds `127.5` up to `128`,
 * which is why the exported value below (and the LUT's color at that index)
 * is the entry *nearest* zero effect, not a mathematically exact center.
 */
export const DIVERGING_LUT_CENTER_INDEX = effectToLutIndex(0, 1, COLORMAP_SIZE);
