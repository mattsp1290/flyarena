import { formatRho } from '../format';
import { provenanceFor, reasonFor, sidecarStepStatus, type BuildFindingStepsInputs, type FindingStep } from './shared';

// ---------------------------------------------------------------------------
// Step 3: Explanation
// ---------------------------------------------------------------------------

export const buildExplanationStep = (inputs: BuildFindingStepsInputs): FindingStep => {
  const provenance = provenanceFor(
    inputs.manifest,
    inputs.manifest?.nullExplanation,
    'Null-explanation result (null-explanation-v1.json)',
    'null-explanation-report.md',
    inputs.dataBaseUrl
  );
  const status = sidecarStepStatus(inputs.nullExplanation);
  const base = {
    id: 'explanation',
    title: 'Explanation',
    condition: 'authored' as const,
    provenance: provenance ? [provenance] : []
  };
  if (status !== 'ok' || inputs.nullExplanation?.status !== 'ok') {
    return { ...base, status, reason: reasonFor(status, inputs.nullExplanation) };
  }
  const { qualifyingMetrics: metrics, regimeInvalid } = inputs.nullExplanation.data.finding;

  // (dual review, Important) `finding.regimeInvalid` is the loader's own
  // cross-checked field (`nullExplanation.ts` rejects an artifact whose
  // `finding.regimeInvalid` disagrees with `regime.gatePassed`) — a step
  // that stated qualifying metrics "pass both predeclared gates" with no
  // qualifier, on a re-run where the regime gate failed, would silently
  // turn an inconclusive result into a positive one. Stated regardless of
  // `metrics.length`, matching `NullExplanationNote.svelte`'s own
  // regime-clause wording. Inserted before the fixed "under this model."
  // ending (every step's own non-negotiable), never after it.
  const regimeClause = regimeInvalid
    ? '; the linear-regime check failed, so this result is reported as regime-invalid (inconclusive)'
    : '';

  if (metrics.length === 0) {
    const sentence =
      'Under the authored (hand-written) decoder, no metric independently passes both predeclared gates ' +
      `(outside the rewired null's range, and rank-correlated with score across rewirings)${regimeClause}, under this model.`;
    return { ...base, status: 'ok', sentence };
  }
  // (dual review, Important) The two predeclared gates are "outside the
  // null's 2.5-97.5% range" and "|rho| at or above the threshold"
  // (nullExplanation.ts's own doc comment, and the artifact's own
  // qualifyingMetricsNote) — not literally "clearance->thrust gates": a
  // qualifying metric can be a linear transfer entry (T:channel->population)
  // or a structural feature (e.g. weightedInDegree:*), and the real shipped
  // data qualifies one of each. This states the gates accurately instead of
  // misnaming a structural-feature metric as a clearance->thrust one.
  const metricList = metrics.map((metric) => `${metric.name} (rank correlation with score, ρ = ${formatRho(metric.spearman)})`).join('; ');
  const sentence =
    `Under the authored (hand-written) decoder, ${metrics.length} metric${metrics.length === 1 ? '' : 's'} ` +
    `independently pass${metrics.length === 1 ? 'es' : ''} both predeclared gates (outside the rewired null's ` +
    `range, and rank-correlated with score across rewirings): ${metricList} — a descriptive association, not a ` +
    `causal claim${regimeClause}, under this model.`;
  return { ...base, status: 'ok', sentence };
};
