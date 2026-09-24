/**
 * The experimental-arm vocabulary (`ArmName`, `ARM_NAMES`), factored into
 * its own leaf module with no imports of its own. Originally hoisted into
 * `export-arms.ts` (round-3 review fix), but `run-dir.ts` documents itself
 * as a lightweight contract module a future WP3 driver can depend on
 * "without importing all of `evaluate.ts`'s CLI/report-writing code" — and
 * `export-arms.ts` is itself a CLI script that imports `export-traces.ts`
 * (another CLI) plus `tests/fixtures/trace-graph`/`trace-graph-rewire`.
 * Depending on `export-arms.ts` just to get a three-element constant broke
 * that promise. This module has zero imports, so anything that needs only
 * the arm vocabulary (`run-dir.ts`, `report.ts`, `evaluate.ts`) can depend
 * on it without pulling in CLI parsing or test fixtures.
 *
 * `ArmName` is derived from `ARM_NAMES` (not declared independently), so
 * the type and the runtime list cannot drift apart: adding a fourth arm
 * means editing exactly one array literal, here.
 */
export const ARM_NAMES = ['biological', 'rewired', 'disconnected'] as const;

export type ArmName = (typeof ARM_NAMES)[number];
