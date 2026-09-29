import { formatPercentile } from '../format';
import { provenanceFor, reasonFor, rewiringNullStepStatus, type BuildFindingStepsInputs, type FindingStep } from './shared';

// ---------------------------------------------------------------------------
// Step 1: Rewiring null
// ---------------------------------------------------------------------------

export const buildRewiringNullStep = (inputs: BuildFindingStepsInputs): FindingStep => {
  const provenance = provenanceFor(
    inputs.manifest,
    inputs.manifest?.rewiringNull,
    'Rewiring-null result (rewiring-null-v1.json)',
    'rewiring-null-report.md',
    inputs.dataBaseUrl
  );
  const status = rewiringNullStepStatus(inputs.rewiringNull);
  const base = {
    id: 'rewiring-null',
    title: 'Rewiring null',
    condition: 'authored' as const,
    provenance: provenance ? [provenance] : []
  };
  if (status !== 'ok' || inputs.rewiringNull?.status !== 'ok') {
    return { ...base, status, reason: reasonFor(status, inputs.rewiringNull) };
  }
  const { data } = inputs.rewiringNull;
  const sentence =
    `Under the authored (hand-written) decoder, biological ranks at the ${formatPercentile(data.bioPercentile)} ` +
    `among ${data.null.n} degree-preserving rewirings, under this model.`;
  return { ...base, status: 'ok', sentence };
};
