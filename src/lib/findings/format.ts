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
 *
 * Thermo review (methodology I1): `PATHWAY_AUTHORED_CLAUSE_TEXT`/
 * `PATHWAY_TRAINED_CLAUSE_TEXT` below are moved here verbatim from what
 * used to be `NullExplanationNote.svelte`'s own private, unexported
 * `PATHWAY_AUTHORED_CLAUSE`/`PATHWAY_TRAINED_CLAUSE` maps — that component
 * now imports them from here instead of keeping its own copy (its rendered
 * ledger text is unchanged), so `src/lib/findings/steps.ts`'s step 6 can
 * reuse the same canonical category facts instead of independently
 * rendering the raw enum slug (`describeTrainedCategory` below).
 */

import type { PathwayInterventionsAuthoredCategory, PathwayInterventionsTrainedCategory } from '../experiment/pathwayInterventions';

/** `${(value * 100).toFixed(1)}th percentile` — `value` is a `[0, 1]` fraction (e.g. an artifact's `bioPercentile` field). */
export const formatPercentile = (value: number): string => `${(value * 100).toFixed(1)}th percentile`;

/** `value.toFixed(3)` — a Spearman rank correlation (ρ), matching `NullExplanationNote.svelte`'s existing inline precision. */
export const formatRho = (value: number): string => value.toFixed(3);

/**
 * One template per authored pathway-intervention category (`00-overview.md`'s
 * vocabulary, stated mechanically — never a stronger or weaker claim than
 * the category itself licenses), each carrying the "net effect of the
 * accepted swap set, not a single-edge effect" framing in its own wording.
 * Moved verbatim from `NullExplanationNote.svelte`'s former private
 * `PATHWAY_AUTHORED_CLAUSE` (thermo review, methodology I1) — text
 * unchanged, so that component's rendered output stays byte-identical.
 */
export const PATHWAY_AUTHORED_CLAUSE_TEXT: Record<PathwayInterventionsAuthoredCategory, string> = {
  'pathway-supported':
    "the pathway-supported category holds: the accepted swap set's net effect outperforms both the unrestricted (C) and class-matched (M) random controls",
  'edge-class-effect':
    "the edge-class-effect category holds: the accepted swap set's net effect outperforms the unrestricted (C) control but not the class-matched (M) control — any edge of this class helps about equally",
  'generic-rewiring-effect':
    "the generic-rewiring-effect category holds: the accepted swap set's net effect does not outperform the unrestricted (C) control — any perturbation of this size helps about equally",
  'not-supported': "the not-supported category holds: the accepted swap set's net effect does not clear the null's 25th percentile"
};

/**
 * One template per trained category — see
 * `scripts/null/intervention-report-trained.ts`'s own doc comment for why
 * `'no-specific-effect'` is not a renamed authored category (it is the
 * deliberate merge of `'generic-rewiring-effect'`/`'not-supported'` for the
 * trained side, where this study's predeclared rules cannot decide that
 * finer split). Used only when `trainedRobust` is true. Moved verbatim from
 * `NullExplanationNote.svelte`'s former private `PATHWAY_TRAINED_CLAUSE`
 * (thermo review, methodology I1) — text unchanged.
 */
export const PATHWAY_TRAINED_CLAUSE_TEXT: Record<PathwayInterventionsTrainedCategory, string> = {
  'pathway-supported': 'P also outperforms both freshly-trained control arms across all three trainer seeds tested',
  'edge-class-effect': 'P outperforms the freshly-trained unrestricted (C) arm but not the class-matched (M) arm, across all three trainer seeds tested',
  'no-specific-effect': 'P shows no advantage over either freshly-trained control arm, across all three trainer seeds tested'
};

/**
 * Short, human-readable label for a trained pathway-intervention category —
 * `src/lib/findings/steps.ts`'s step 6, which (unlike `NullExplanationNote.svelte`'s
 * long explanatory clauses above) needs a short label to drop into its own
 * fixed sentence pattern. `'pathway-supported'`/`'edge-class-effect'` are
 * already self-descriptive compound words and pass through unchanged;
 * `'no-specific-effect'` is genuinely easy to misread as "no effect" (it
 * means the trained side's predeclared rules cannot decide between
 * `generic-rewiring-effect` and `not-supported` — a *resolution* limit, not
 * a null-result claim — see `docs/pathway-interventions-report.md`), so it
 * gets its own plain-English gloss (thermo review, methodology I1).
 *
 * `withCaveat` (thermo review, methodology I2) appends a short, honest
 * parenthetical noting that `'no-specific-effect'` is itself a reporting
 * convention, not a predeclared category — only when the caller has
 * confirmed the artifact's own `trained.note` field actually discloses this
 * (never unconditionally hard-coded; see `steps.ts`'s call site). Kept out
 * of the per-seed "seeds disagree" listing (each seed's own call omits it)
 * so that branch doesn't repeat the same caveat once per dissenting seed.
 */
export const describeTrainedCategory = (
  category: PathwayInterventionsTrainedCategory,
  options?: { readonly withCaveat?: boolean }
): string => {
  if (category !== 'no-specific-effect') return category;
  const caveat = options?.withCaveat
    ? '; a reporting convention adopted after the trained scores were known, not a predeclared category'
    : '';
  return `no specific effect (neither pathway-supported nor edge-class${caveat})`;
};
