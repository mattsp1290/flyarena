import { provenanceFor, reasonFor, sidecarStepStatus, type BuildFindingStepsInputs, type FindingStep } from './shared';

// ---------------------------------------------------------------------------
// Step 4: Intervention (authored)
// ---------------------------------------------------------------------------

export const buildInterventionStep = (inputs: BuildFindingStepsInputs): FindingStep => {
  const provenance = provenanceFor(
    inputs.manifest,
    inputs.manifest?.pathwayInterventions,
    'Pathway-interventions result (pathway-interventions-v1.json)',
    'pathway-interventions-report.md',
    inputs.dataBaseUrl
  );
  const status = sidecarStepStatus(inputs.pathwayInterventions);
  const base = {
    id: 'intervention',
    title: 'Intervention',
    condition: 'authored' as const,
    provenance: provenance ? [provenance] : []
  };
  if (status !== 'ok' || inputs.pathwayInterventions?.status !== 'ok') {
    return { ...base, status, reason: reasonFor(status, inputs.pathwayInterventions) };
  }
  const { authored } = inputs.pathwayInterventions.data;
  const channelClause = authored.channelSpecific
    ? 'with the channel-specific modifier holding'
    : 'without the channel-specific modifier holding';
  const sentence =
    `Under the authored (hand-written) decoder, the tested clearance→thrust intervention falls in the ` +
    `${authored.category} category, ${channelClause}, under this model.`;
  return { ...base, status: 'ok', sentence };
};
