import { metricVerdictLabel } from '../../atlas/repertoireStrip';
import { CELL_COUNT } from '../../atlas/types';
import { provenanceFor, reasonFor, sidecarStepStatus, type BuildFindingStepsInputs, type FindingStep } from './shared';

// ---------------------------------------------------------------------------
// Behavior repertoire
// ---------------------------------------------------------------------------

/**
 * WP3 of `.agents/plans/repertoire-null` (wired per `findings-tour`'s own
 * `01-findings-panel.md`): the behavior-repertoire comparison between the
 * biological topology and its degree-preserving rewirings, under the
 * shipped MAP-Elites search. `FindingsPanel.svelte` links out to `#atlas`
 * next to this step regardless of status (the atlas is a separate hash
 * route, not an artifact this step loads — see `00-overview.md`'s "the
 * repertoire step links to #atlas rather than loading the atlas").
 *
 * Not decoder-specific (`condition: 'both'`, unchanged from before this WP):
 * the search itself always uses the authored readout family, but the
 * finding is about topology, not about which decoder is currently selected.
 */
export const buildBehaviorRepertoireStep = (inputs: BuildFindingStepsInputs): FindingStep => {
  const provenance = provenanceFor(
    inputs.manifest,
    inputs.manifest?.behaviorRepertoireNull,
    'Behavior-repertoire result (behavior-repertoire-null-v1.json)',
    'behavior-repertoire-null-report.md',
    inputs.dataBaseUrl
  );
  const status = sidecarStepStatus(inputs.repertoireNull);
  const base = {
    id: 'behavior-repertoire',
    title: 'Behavior repertoire',
    condition: 'both' as const,
    provenance: provenance ? [provenance] : []
  };
  if (status !== 'ok' || inputs.repertoireNull?.status !== 'ok') {
    return { ...base, status, reason: reasonFor(status, inputs.repertoireNull) };
  }
  const { primary, search, robustness: repertoireRobustness } = inputs.repertoireNull.data;
  const seeds = [search.primarySearchSeed, ...search.extraSearchSeeds];
  const robustnessClause = repertoireRobustness.robust
    ? `robust across all ${seeds.length} search seeds`
    : `not robust across search seeds (${seeds.map((seed) => `seed ${seed}: ${repertoireRobustness.perSeed[seed]}`).join(', ')})`;
  // `primary.tie` (a maintainability review, Suggestion): validated by the
  // loader but previously never surfaced anywhere -- a future run whose
  // "typical" category is only "typical" because of a degenerate rewired
  // distribution (the predeclared tie rule) would otherwise read as an
  // ordinary typical result, with no hint that either metric's rewired
  // distribution happened to equal biological exactly.
  const tieSuffix = primary.tie ? ' (tie)' : '';
  // `CELL_COUNT` (never a bare `36` literal — the template-lint test below
  // forbids a hard-coded numeric literal in this file's own source): the
  // atlas's fixed 6x6 coverage x turning grid (`src/lib/atlas/types.ts`),
  // not a field this artifact itself carries.
  //
  // `primary.rewiredDistribution.occupied.n` (a maintainability review,
  // Suggestion), not `search.rewiredCount` — the sample size printed next
  // to a median should be the size of the sample that median was actually
  // taken over, matching `repertoire-report.ts`'s own report (which prints
  // its measured `rewiredGraphCountAtPrimary`, never the study-wide count,
  // in this exact spot). Only one trailing "under this model." (a
  // maintainability review, Suggestion: an earlier version also opened
  // with "this model's", stating the same disclosure twice).
  //
  // A thermo-methodology review (Important) found this sentence stated the
  // category next to only `occupied`, even though the predeclared rule
  // decides it jointly from *both* `occupied` and `qd` -- see
  // `../../atlas/repertoireStrip.ts#buildRepertoireStripText`'s identical
  // fix and doc comment for the full reasoning; `metricVerdictLabel`
  // (shared from that same module, not re-implemented here) recomputes
  // each metric's own verdict from data already on `primary`.
  const qdDist = primary.rewiredDistribution.qd;
  const basisClause =
    primary.category !== 'typical'
      ? 'on both metrics'
      : `occupied ${metricVerdictLabel(primary.bio.occupied, primary.rewiredDistribution.occupied)}, qd ${metricVerdictLabel(primary.bio.qd, qdDist)}`;
  const sentence =
    `Under the shipped MAP-Elites search, biological occupies ${primary.bio.occupied} of ${CELL_COUNT} ` +
    `behavior cells against a rewired median of ${primary.rewiredDistribution.occupied.p50} ` +
    `(n=${primary.rewiredDistribution.occupied.n}), qd ${Math.round(primary.bio.qd)} vs rewired median ${Math.round(qdDist.p50)} — ` +
    `${primary.category}${tieSuffix} ${basisClause} at search seed ` +
    `${search.primarySearchSeed}, ${robustnessClause}, under this model.`;
  return { ...base, status: 'ok', sentence };
};
