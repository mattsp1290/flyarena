/**
 * Shared normalized-unit constants for the anatomical activity view (WP3).
 * Split into their own module (thermo-maintainability S1 fix) so
 * `ActivityScene.ts` (WebGL point rendering) and `activity-layout.ts` (pure
 * layout math, no Three.js/DOM dependency — see that module's own doc
 * comment on why it stays that way) can both import one source of truth
 * instead of `activity-layout.ts`'s `STRIP_MIN_SPACING` being linked to
 * `ActivityScene.ts`'s `POINT_SIZE` only by a comment, with no import
 * connecting the two source values (round-2 review S2, previously still
 * open) — a future edit to one could silently desync from the other.
 */

/**
 * Three.js `PointsMaterial#size`, in the same normalized units the main
 * soma point cloud and the "position unavailable" strip are laid out into
 * (`[-1, 1]` centered/scaled extent — see `activity-layout.ts#layoutPositions`).
 */
export const POINT_SIZE = 0.045;
