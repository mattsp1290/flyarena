/**
 * Pure, framework-agnostic layout/coloring helpers for the anatomical
 * activity view (WP3). Nothing here imports Three.js or touches the DOM, so
 * the honesty-critical mapping from source data to what gets drawn is
 * unit-testable under jsdom, the same reasoning `render/transforms.ts`
 * documents for the arena renderer. `ActivityScene.ts` only wires these
 * results into scene objects.
 *
 * Neuron identity throughout this module is the shared, arm-agnostic index
 * `0..neuronCount-1` from `malecns-arena-v1.positions.json` (see
 * `experiment/assets.ts#loadPositions`) — both arms are built from the same
 * `xyz`/`positionSource`/`role` arrays (degree-preserving rewiring keeps the
 * node set; only connections differ), so one layout/partition serves both.
 */
import { rateToLutIndex } from './colormap';

export type PositionSource = 'soma' | 'tosoma' | 'none';
export type NeuronRole = 'sensory' | 'bridge' | 'descending';

export interface PositionLayout {
  /** Length `xyz.length * 3`, one centered/scaled (or unavailable-strip) `[x, y, z]` per neuron, in input order. */
  points: Float32Array;
  /** Indices (into the same `0..xyz.length-1` order) of neurons with `positionSource === 'none'` — never invented coordinates, only strip-placed. */
  unavailableIdx: Int32Array;
}

/**
 * Vertical offset of the "position unavailable" strip below the centered,
 * unit-scaled soma point cloud (whose extent is `[-1, 1]` on every axis
 * after scaling — see below). Large enough that the strip never visually
 * merges with the main cloud even at a shallow camera angle.
 */
const STRIP_Y = -1.35;
const STRIP_Z = 0;
/**
 * Minimum on-screen gap between adjacent strip points, along both the
 * strip's X spread and its row spacing. Kept at least as large as
 * `ActivityScene.ts`'s point sprite size (0.045 normalized units) so a
 * real-sized unavailable set (165 of 1,008 neurons in the shipped MaleCNS
 * data) wraps into multiple rows instead of cramming into one row where
 * points overlap into an unreadable smear — a real, reproduced problem with
 * an earlier single-row version of this layout.
 */
const STRIP_MIN_SPACING = 0.06;
/** Matches the main cloud's `[-1, 1]` centered/scaled extent (see below). */
const STRIP_WIDTH = 2;

/**
 * Center soma-annotated neurons (`positionSource` is `'soma'` or `'tosoma'`)
 * on their own centroid and uniformly scale them so every axis's extent fits
 * in `[-1, 1]` — never per-axis, so relative shape/proportion of the real
 * annotated geometry is preserved, not stretched to fill a box. Neurons with
 * `positionSource === 'none'` (no real coordinate to show) are never given a
 * fabricated position from this computation; they are placed on a separate,
 * evenly-spaced horizontal strip below the main cloud instead (see
 * `STRIP_Y`), and returned in `unavailableIdx` so a caller can render/label
 * that strip distinctly. A `null` `xyz` entry (the JSON encoding of "no
 * coordinate") is treated identically to `positionSource === 'none'`
 * regardless of what `positionSource` itself says, so a malformed/
 * inconsistent artifact still degrades safely rather than crashing or
 * inventing a `(0, 0, 0)` position.
 */
export const layoutPositions = (
  xyz: ReadonlyArray<readonly [number, number, number] | null>,
  positionSource: readonly PositionSource[]
): PositionLayout => {
  const count = xyz.length;
  const points = new Float32Array(count * 3);
  const unavailable: number[] = [];

  const isAvailable = (index: number): boolean => positionSource[index] !== 'none' && xyz[index] !== null;

  let sumX = 0;
  let sumY = 0;
  let sumZ = 0;
  let availableCount = 0;
  for (let index = 0; index < count; index += 1) {
    if (!isAvailable(index)) {
      unavailable.push(index);
      continue;
    }
    const point = xyz[index] as readonly [number, number, number];
    sumX += point[0];
    sumY += point[1];
    sumZ += point[2];
    availableCount += 1;
  }
  const centerX = availableCount > 0 ? sumX / availableCount : 0;
  const centerY = availableCount > 0 ? sumY / availableCount : 0;
  const centerZ = availableCount > 0 ? sumZ / availableCount : 0;

  let maxExtent = 0;
  for (let index = 0; index < count; index += 1) {
    if (!isAvailable(index)) continue;
    const point = xyz[index] as readonly [number, number, number];
    maxExtent = Math.max(
      maxExtent,
      Math.abs(point[0] - centerX),
      Math.abs(point[1] - centerY),
      Math.abs(point[2] - centerZ)
    );
  }
  const scale = maxExtent > 0 ? 1 / maxExtent : 1;

  for (let index = 0; index < count; index += 1) {
    if (!isAvailable(index)) continue;
    const point = xyz[index] as readonly [number, number, number];
    const offset = index * 3;
    points[offset] = (point[0] - centerX) * scale;
    points[offset + 1] = (point[1] - centerY) * scale;
    points[offset + 2] = (point[2] - centerZ) * scale;
  }

  // Wrap the strip into multiple rows (rather than one long row) once there
  // are more unavailable neurons than fit at `STRIP_MIN_SPACING` across
  // `STRIP_WIDTH` — see that constant's doc comment.
  const unavailableCount = unavailable.length;
  const perRow = Math.max(1, Math.floor(STRIP_WIDTH / STRIP_MIN_SPACING) + 1);
  const rowCount = Math.max(1, Math.ceil(unavailableCount / perRow));
  for (let slot = 0; slot < unavailableCount; slot += 1) {
    const index = unavailable[slot];
    const offset = index * 3;
    const row = Math.floor(slot / perRow);
    const col = slot % perRow;
    const countInThisRow = row === rowCount - 1 ? unavailableCount - row * perRow : perRow;
    const t = countInThisRow > 1 ? col / (countInThisRow - 1) : 0.5;
    points[offset] = -1 + STRIP_WIDTH * t;
    points[offset + 1] = STRIP_Y - row * STRIP_MIN_SPACING;
    points[offset + 2] = STRIP_Z;
  }

  return { points, unavailableIdx: Int32Array.from(unavailable) };
};

export interface RolePartition {
  sensoryIdx: Int32Array;
  bridgeIdx: Int32Array;
  descendingIdx: Int32Array;
}

/**
 * Group every neuron's shared index by its annotated role. Covers *every*
 * neuron (main cloud and unavailable strip alike — every neuron has a role
 * regardless of whether it has a real soma position), so
 * `sensoryIdx.length + bridgeIdx.length + descendingIdx.length === role.length`
 * always holds.
 */
export const partitionByRole = (role: readonly NeuronRole[]): RolePartition => {
  const sensory: number[] = [];
  const bridge: number[] = [];
  const descending: number[] = [];
  for (let index = 0; index < role.length; index += 1) {
    switch (role[index]) {
      case 'sensory':
        sensory.push(index);
        break;
      case 'bridge':
        bridge.push(index);
        break;
      case 'descending':
        descending.push(index);
        break;
      default:
        throw new Error(`partitionByRole: unknown role "${String(role[index])}" at index ${index}`);
    }
  }
  return {
    sensoryIdx: Int32Array.from(sensory),
    bridgeIdx: Int32Array.from(bridge),
    descendingIdx: Int32Array.from(descending)
  };
};

/**
 * Write RGB colors for `indices` (each a neuron index into `rates`) into
 * `out`, in the same order as `indices` — `out[k*3..k*3+2]` for
 * `indices[k]`. No allocation: `lut` and `out` are both caller-owned
 * buffers, and `lut` can be any table shaped `steps * 3` (not necessarily
 * `colormap.ts#VIRIDIS_LUT`) — accepting it as a parameter rather than
 * hard-coding the concrete colormap keeps this function testable against a
 * synthetic table. It shares its clamp/rounding math with
 * `colormap.ts#rateToColor` via `rateToLutIndex` (a pure `(rate, min, max,
 * steps) -> index` function with no dependency on any concrete LUT data),
 * so the two color paths cannot silently disagree after a future edit to
 * just one of them — a real risk an earlier, independently-duplicated
 * version of this math had.
 */
export const writeColors = (
  rates: Float32Array,
  indices: Int32Array,
  min: number,
  max: number,
  lut: Float32Array,
  out: Float32Array
): void => {
  const lutSteps = lut.length / 3;
  for (let k = 0; k < indices.length; k += 1) {
    const neuron = indices[k];
    const lutIndex = rateToLutIndex(rates[neuron], min, max, lutSteps) * 3;
    const outOffset = k * 3;
    out[outOffset] = lut[lutIndex];
    out[outOffset + 1] = lut[lutIndex + 1];
    out[outOffset + 2] = lut[lutIndex + 2];
  }
};
