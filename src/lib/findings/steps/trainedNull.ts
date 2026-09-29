import { validateRewiringNullTrained, type RewiringNullTrainedSection } from '../../experiment/rewiringNull';
import { formatPercentile } from '../format';
import { provenanceFor, reasonFor, rewiringNullStepStatus, type BuildFindingStepsInputs, type FindingStep } from './shared';

// ---------------------------------------------------------------------------
// Step 5: Trained null
// ---------------------------------------------------------------------------

export const buildTrainedNullStep = (inputs: BuildFindingStepsInputs): FindingStep => {
  const provenance = provenanceFor(
    inputs.manifest,
    inputs.manifest?.rewiringNull,
    'Rewiring-null result, trained-sample section (rewiring-null-v1.json)',
    'rewiring-null-report.md',
    inputs.dataBaseUrl
  );
  const base = {
    id: 'trained-null',
    title: 'Trained null',
    condition: 'trained' as const,
    provenance: provenance ? [provenance] : []
  };
  const status = rewiringNullStepStatus(inputs.rewiringNull);
  if (status !== 'ok' || inputs.rewiringNull?.status !== 'ok') {
    return { ...base, status, reason: reasonFor(status, inputs.rewiringNull) };
  }
  const trained: RewiringNullTrainedSection | undefined = validateRewiringNullTrained(inputs.rewiringNull.data.trained);
  if (!trained) {
    return {
      ...base,
      status: 'invalid',
      reason: 'rewiring-null artifact has no valid trained-sample section (RewiringNullArtifact.trained)'
    };
  }
  const percentiles = trained.bioReplicaPercentiles.map((entry) => entry.percentile);
  const minLabel = formatPercentile(Math.min(...percentiles));
  const maxLabel = formatPercentile(Math.max(...percentiles));
  const headlineLabel = formatPercentile(trained.bioPercentile);
  const sensitivityClause =
    minLabel === maxLabel
      ? `stays at the ${headlineLabel} across every trainer seed tested`
      : `ranges from ${minLabel} to ${maxLabel} across the trainer seeds tested (headline: ${headlineLabel} at trainer seed ${trained.replicaSeed})`;
  const sentence =
    `Under trained readouts, biological's percentile among the ${trained.rewiredCount} rewired trained scores ` +
    `${sensitivityClause}, under this model.`;
  return { ...base, status: 'ok', sentence };
};
