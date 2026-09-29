import { formatPercentile } from '../format';
import {
  provenanceFor,
  reasonFor,
  rewiringNullStepStatus,
  sidecarStepStatus,
  worseStatus,
  STATUS_SEVERITY,
  type BuildFindingStepsInputs,
  type FindingStep,
  type FindingStepProvenance
} from './shared';

// ---------------------------------------------------------------------------
// Step 2: Mirrored decoder
// ---------------------------------------------------------------------------

export const buildMirroredDecoderStep = (inputs: BuildFindingStepsInputs): FindingStep => {
  // (dual review, Important) This step's sentence draws on *two* source
  // artifacts (the rewiring-null baseline and the null-explanation mirrored
  // variant) — both get their own provenance entry, per `FindingStep.provenance`'s
  // own "one entry per source artifact" contract, so a reader can always
  // find the artifact behind either number the sentence cites.
  const rewiringProvenance = provenanceFor(
    inputs.manifest,
    inputs.manifest?.rewiringNull,
    'Rewiring-null result, un-mirrored baseline (rewiring-null-v1.json)',
    'rewiring-null-report.md',
    inputs.dataBaseUrl
  );
  const explanationProvenance = provenanceFor(
    inputs.manifest,
    inputs.manifest?.nullExplanation,
    'Null-explanation result, mirrored-decoder variant (null-explanation-v1.json)',
    'null-explanation-report.md',
    inputs.dataBaseUrl
  );
  const base = {
    id: 'mirrored-decoder',
    title: 'Mirrored decoder',
    condition: 'authored' as const,
    provenance: [rewiringProvenance, explanationProvenance].filter((entry): entry is FindingStepProvenance => entry !== undefined)
  };

  // The mirrored-decoder check compares the null-explanation artifact's
  // `variants.flipBoth.bioPercentile` against the *un-mirrored* baseline
  // from the rewiring-null artifact (`NullExplanationNote.svelte`'s own
  // `baselinePercentile` prop) — both must have actually resolved to `'ok'`
  // before this step has anything true to say. When either has not, the
  // *worse* of the two statuses is shown (`worseStatus`'s own doc comment),
  // not always the rewiring-null one — a genuine `'invalid'` on either
  // input must never be hidden behind the other's more benign status.
  const rewiringStatus = rewiringNullStepStatus(inputs.rewiringNull);
  const explanationStatus = sidecarStepStatus(inputs.nullExplanation);
  const combinedStatus = worseStatus(rewiringStatus, explanationStatus);
  if (combinedStatus !== 'ok' || inputs.rewiringNull?.status !== 'ok' || inputs.nullExplanation?.status !== 'ok') {
    const reason =
      STATUS_SEVERITY[explanationStatus] >= STATUS_SEVERITY[rewiringStatus]
        ? reasonFor(explanationStatus, inputs.nullExplanation)
        : reasonFor(rewiringStatus, inputs.rewiringNull);
    return { ...base, status: combinedStatus, reason };
  }

  const baseline = inputs.rewiringNull.data.bioPercentile;
  const mirrored = inputs.nullExplanation.data.variants.flipBoth.bioPercentile;
  const baselineLabel = formatPercentile(baseline);
  const mirroredLabel = formatPercentile(mirrored);
  const persistedClause =
    baseline <= 0 && mirrored <= 0
      ? `still leaves biological at the bottom of the null distribution (${mirroredLabel})`
      : mirroredLabel === baselineLabel
        ? `leaves biological at the same ${mirroredLabel} of the null distribution`
        : `moves biological from the ${baselineLabel} to the ${mirroredLabel} of the null distribution`;
  const sentence =
    `Under the authored (hand-written) decoder, mirroring the decoder's thrust and yaw signs ${persistedClause}, ` +
    `under this model.`;
  return { ...base, status: 'ok', sentence };
};
