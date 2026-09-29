/**
 * WP1 of `.agents/plans/findings-tour` (`01-findings-panel.md`): pure
 * builder for the Findings panel's (now nine-step, since selection-
 * robustness WP3) evidence chain (`src/lib/ui/FindingsPanel.svelte`).
 * Every sentence is templated exclusively from fields on the already
 * fetched, sha256-verified, shape-validated artifacts
 * `ExperimentController#initialize()` already loads (`rewiringNull`,
 * `nullExplanation`, `pathwayInterventions` — see
 * `.agents/plans/findings-tour/00-overview.md`'s "Repository findings")
 * plus the manifest entries those loaders were verified against — never a
 * hard-coded number, and every rendered number is routed through
 * `./format.ts`'s `formatPercentile`/`formatRho` (unit-tested by
 * `tests/unit/findings-steps.test.ts`'s template-lint check).
 *
 * `buildFindingSteps` takes `dataBaseUrl` explicitly (mirroring
 * `loadRewiringNull`/`loadNullExplanation`/`loadPathwayInterventions`'s own
 * "caller passes the base URL, never reads `import.meta.env` itself"
 * convention) so it stays a plain, synchronous, fully-testable pure
 * function — no dependency on a bundler global.
 *
 * A step whose backing artifact has not yet resolved, was never shipped, or
 * failed verification degrades only that one step (`FindingStepStatus`):
 * the panel still renders every step (`aria-current="step"`/"Step N of M"
 * stay honest and stable, derived from the array length), swapping the templated sentence for a short,
 * honestly-worded status line instead of hiding the step outright.
 *
 * This file is the orchestrator only: the shared `FindingStep`/
 * `BuildFindingStepsInputs` types and cross-step plumbing live in
 * `./steps/shared.ts`, and each study's own `buildXStep` function (plus its
 * private per-step helpers) lives in its own `./steps/<study>.ts` file. A
 * thermo-maintainability review (Important, I2) found this file had grown
 * to 808 lines at ~65-115 lines per study, with a second in-flight step
 * (`flyarena-hd0j`, readout attribution -- see `./sections.ts`'s own doc
 * comment) already queued to land here next; splitting now, before that
 * PR lands, avoids a merge conflict on the exact file both changes would
 * otherwise touch most. This split changes no behavior: every import path
 * outside this directory is unchanged (`buildFindingSteps`,
 * `findingStepStatusLabel`, `FindingStep`, `BuildFindingStepsInputs` are
 * all still exported from this same `./steps` path), and every
 * `buildXStep` function's own body is a byte-identical move.
 */

import { buildRewiringNullStep } from './steps/rewiringNull';
import { buildMirroredDecoderStep } from './steps/mirroredDecoder';
import { buildExplanationStep } from './steps/explanation';
import { buildInterventionStep } from './steps/intervention';
import { buildTrainedNullStep } from './steps/trainedNull';
import { buildTrainedInterventionsStep } from './steps/trainedInterventions';
import { buildTaskGeneralityStep } from './steps/taskGenerality';
import { buildBehaviorRepertoireStep } from './steps/behaviorRepertoire';
import { buildSelectionRobustnessStep } from './steps/selectionRobustness';
import type { BuildFindingStepsInputs, FindingStep } from './steps/shared';

export type { FindingStep, FindingStepProvenance, FindingStepStatus, BuildFindingStepsInputs } from './steps/shared';
export { findingStepStatusLabel } from './steps/shared';

/**
 * Builds all nine Findings-panel steps, in evidence-chain order
 * (`01-findings-panel.md`'s step list, extended by task-generality WP4's
 * "before Behavior repertoire" placement, and by selection-robustness WP3's
 * own "Append the step at the end of the step list present at
 * implementation time"). Pure and synchronous: every input is a value the
 * caller already has in scope (controller callback mirrors), never a fetch
 * performed here.
 */
export const buildFindingSteps = (inputs: BuildFindingStepsInputs): readonly FindingStep[] => [
  buildRewiringNullStep(inputs),
  buildMirroredDecoderStep(inputs),
  buildExplanationStep(inputs),
  buildInterventionStep(inputs),
  buildTrainedNullStep(inputs),
  buildTrainedInterventionsStep(inputs),
  buildTaskGeneralityStep(inputs),
  buildBehaviorRepertoireStep(inputs),
  buildSelectionRobustnessStep(inputs)
];
