import type { RobustnessVerdict, SelectionResult } from '../../experiment/selectionRobustness';
import { formatPercentile } from '../format';
import { provenanceFor, reasonFor, sidecarStepStatus, type BuildFindingStepsInputs, type FindingStep } from './shared';

// ---------------------------------------------------------------------------
// Selection robustness
// ---------------------------------------------------------------------------

/** `RobustnessVerdict['verdict']` rendered as plain text -- never a raw `${verdict}` interpolation, so a future fourth verdict value fails to compile here instead of silently rendering `"true"`/`"false"`/`"indeterminate"`'s JS-native stringification. */
const verdictWord = (verdict: RobustnessVerdict['verdict']): string => (verdict === true ? 'true' : verdict === false ? 'false' : 'indeterminate');

/**
 * One selection's null/explanation/pathway result in plain English --
 * `details`'s own per-item detail (`./shared.ts`'s `FindingStep.details`
 * doc comment), not folded into the main `sentence`. Every clause states
 * the mechanism, never a bare status tag: a degenerate pathway names *why*
 * (`degenerateMechanism`, already a full sentence fragment from
 * `scripts/selections/selection-report.ts`'s own producer -- see that
 * file's top doc comment for the exact wording convention), and
 * `search-limited` is distinguished from `not-supported` per `00-overview.md`'s
 * predeclared "Search-budget disclosure" rule.
 */
const selectionSummary = (selection: Readonly<SelectionResult>): string => {
  const nullClause = `null ${selection.null.holds ? 'holds' : 'does not hold'} (biological at the ${formatPercentile(selection.null.bioPercentile)} of the rewired null)`;
  const explanationClause = `explanation ${selection.explanation.replicates ? 'replicates' : 'does not replicate'}`;
  const pathwayClause =
    selection.pathway.cDegenerate || selection.pathway.mDegenerate
      ? `pathway P/Q degenerate, not categorized -- ${selection.pathway.degenerateMechanism}`
      : selection.pathway.searchLimited
        ? "pathway search-limited (the swap search reached its cap before the transfer target)"
        : `pathway ${selection.pathway.supported ? 'supported' : 'not-supported'}`;
  return `${nullClause}; ${explanationClause}; ${pathwayClause}.`;
};

/**
 * WP3 of `.agents/plans/selection-robustness` (`03-artifact-and-findings.md`):
 * whether the rewiring-null/explanation/pathway findings above hold across
 * four predeclared alternative subgraph selections (bridge-population size,
 * bridge-selection method, and the authored sensory-channel mapping).
 * `condition: 'authored'`: the whole study runs the authored decoder only,
 * on the default arena task (a trained arm is a predeclared follow-up, not
 * part of this WP -- `00-overview.md`'s non-goals).
 */
export const buildSelectionRobustnessStep = (inputs: BuildFindingStepsInputs): FindingStep => {
  const provenance = provenanceFor(
    inputs.manifest,
    inputs.manifest?.selectionRobustness,
    'Selection-robustness result (selection-robustness-v1.json)',
    'selection-robustness-report.md',
    inputs.dataBaseUrl
  );
  const status = sidecarStepStatus(inputs.selectionRobustness);
  const base = {
    id: 'selection-robustness',
    title: 'Selection robustness',
    condition: 'authored' as const,
    provenance: provenance ? [provenance] : []
  };
  if (status !== 'ok' || inputs.selectionRobustness?.status !== 'ok') {
    return { ...base, status, reason: reasonFor(status, inputs.selectionRobustness) };
  }
  const { selections, overall } = inputs.selectionRobustness.data;
  const sentence =
    `Across ${selections.length} alternative subgraph selections, under the authored (hand-written) decoder only: ` +
    `robust to size is ${verdictWord(overall.robustToSize.verdict)}; robust to method is ${verdictWord(overall.robustToMethod.verdict)}; ` +
    `the channel-mapping result is ${verdictWord(overall.mapping.verdict)} -- see per-selection detail below, under this model.`;
  const details = selections.map((selection) => ({ id: selection.id, text: selectionSummary(selection) }));
  return { ...base, status: 'ok', sentence, details };
};
