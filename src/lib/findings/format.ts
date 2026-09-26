/**
 * WP1 of `.agents/plans/findings-tour`: the shared number formatters every
 * Findings-panel step sentence (`./steps.ts`) is required to route through
 * (`01-findings-panel.md`'s "Numbers are rendered only through
 * `src/lib/findings/format.ts`" non-negotiable — a template-lint unit test
 * fails on a numeric literal appearing directly in a template string).
 *
 * `formatPercentile` is extracted verbatim from the formatter that used to
 * be a private local inside `NullExplanationNote.svelte` (`${(value *
 * 100).toFixed(1)}th percentile`) — that component now imports it from
 * here instead of keeping its own copy, so the ledger panel and the
 * Findings panel read the exact same percentile field identically.
 * `NullHistogram.svelte`'s own `bioPercentile` rendering switches to this
 * formatter too (WP1's change surface), replacing its previous bare `…%`
 * label.
 *
 * `formatRho` is new: `.toFixed(3)`, matching the inline ρ formatting
 * `NullExplanationNote.svelte`'s `qualifyingMetricLines` already uses
 * (`metric.spearman.toFixed(3)`) — this module is the one place that
 * precision is now defined, so a future edit to one can't silently drift
 * from the other.
 */

/** `${(value * 100).toFixed(1)}th percentile` — `value` is a `[0, 1]` fraction (e.g. an artifact's `bioPercentile` field). */
export const formatPercentile = (value: number): string => `${(value * 100).toFixed(1)}th percentile`;

/** `value.toFixed(3)` — a Spearman rank correlation (ρ), matching `NullExplanationNote.svelte`'s existing inline precision. */
export const formatRho = (value: number): string => value.toFixed(3);
