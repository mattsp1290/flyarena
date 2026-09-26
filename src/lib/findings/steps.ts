/**
 * WP1 of `.agents/plans/findings-tour` (`01-findings-panel.md`): pure
 * builder for the Findings panel's seven-step evidence chain
 * (`src/lib/ui/FindingsPanel.svelte`). Every sentence is templated
 * exclusively from fields on the already fetched, sha256-verified,
 * shape-validated artifacts `ExperimentController#initialize()` already
 * loads (`rewiringNull`, `nullExplanation`, `pathwayInterventions` — see
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
 * the panel still renders all seven steps (`aria-current="step"`/"Step N of
 * 7" stay honest and stable), swapping the templated sentence for a short,
 * honestly-worded status line instead of hiding the step outright.
 */

import type { ArenaManifest, SidecarManifestEntry } from '../experiment/assets';
import {
  validateRewiringNullTrained,
  type RewiringNullLoadResult,
  type RewiringNullTrainedSection
} from '../experiment/rewiringNull';
import type { NullExplanationLoadResult } from '../experiment/nullExplanation';
import { P_TRAINER_SEEDS, type PathwayInterventionsLoadResult } from '../experiment/pathwayInterventions';
import { githubDocUrl } from '../ui/links';
import { formatPercentile, formatRho } from './format';

/**
 * The step's own display status — a normalized view over whichever status
 * vocabulary its backing loader(s) actually use (`RewiringNullLoadResult`'s
 * `'absent'`, `NullExplanationLoadResult`'s/`PathwayInterventionsLoadResult`'s
 * `'missing'`, both loaders' shared `'unavailable'`/`'invalid'`), plus
 * `'loading'` for the honest "the load has not resolved yet" state
 * (`ExperimentController#initialize()`'s callbacks fire asynchronously, so
 * every load result prop starts `undefined`) — no loader is renamed; this
 * is the step's own presentation vocabulary layered on top, not a new term
 * in any loader's own status union.
 */
export type FindingStepStatus = 'ok' | 'loading' | 'missing' | 'unavailable' | 'invalid';

export interface FindingStepProvenance {
  readonly label: string;
  readonly artifactPath: string;
  readonly sha256Prefix: string;
  readonly reportPath: string;
}

export interface FindingStep {
  readonly id: string;
  readonly title: string;
  readonly status: FindingStepStatus;
  /** Present only when `status === 'ok'`. */
  readonly sentence?: string;
  readonly condition: 'authored' | 'trained' | 'both';
  /** One entry per source artifact this step's sentence is templated from; empty when the step has never had a real source to cite (e.g. step 7 before `repertoire-null` lands). */
  readonly provenance: readonly FindingStepProvenance[];
  /** The underlying loader's own honest reason string, present only when `status` is `'unavailable'` or `'invalid'`. */
  readonly reason?: string;
}

export interface BuildFindingStepsInputs {
  readonly manifest: ArenaManifest | undefined;
  readonly dataBaseUrl: string;
  readonly rewiringNull: RewiringNullLoadResult | undefined;
  readonly nullExplanation: NullExplanationLoadResult | undefined;
  readonly pathwayInterventions: PathwayInterventionsLoadResult | undefined;
}

const STATUS_LABEL: Record<Exclude<FindingStepStatus, 'ok'>, string> = {
  loading: 'Loading…',
  missing: 'Not yet published',
  unavailable: 'Could not be loaded',
  invalid: 'Failed verification'
};

/** `STATUS_LABEL`'s text for a non-`'ok'` step — exported for the panel component so its own status line never hand-duplicates this vocabulary. */
export const findingStepStatusLabel = (status: Exclude<FindingStepStatus, 'ok'>): string => STATUS_LABEL[status];

const provenanceFor = (
  manifest: ArenaManifest | undefined,
  entry: SidecarManifestEntry | { artifact: string; sha256: string } | undefined,
  label: string,
  reportSlug: string,
  dataBaseUrl: string
): FindingStepProvenance | undefined => {
  if (!manifest || !entry) return undefined;
  return {
    label,
    artifactPath: `${dataBaseUrl}/${entry.artifact}`,
    sha256Prefix: entry.sha256.slice(0, 12),
    reportPath: githubDocUrl(reportSlug)
  };
};

/** Maps `RewiringNullLoadResult`'s `'absent'`/`'unavailable'`/`'invalid'` onto the step vocabulary above — `'ok'` is handled by each step's own builder. */
const rewiringNullStepStatus = (result: RewiringNullLoadResult | undefined): Exclude<FindingStepStatus, 'ok'> | 'ok' => {
  if (result === undefined) return 'loading';
  if (result.status === 'absent') return 'missing';
  return result.status;
};

const sidecarStepStatus = (
  result: NullExplanationLoadResult | PathwayInterventionsLoadResult | undefined
): Exclude<FindingStepStatus, 'ok'> | 'ok' => {
  if (result === undefined) return 'loading';
  return result.status;
};

/**
 * Only `'unavailable'`/`'invalid'` carry a `reason` onto the step (matching
 * `LedgerPanel.svelte`'s/`NullExplanationNote.svelte`'s own precedent: a
 * `'missing'`/`'absent'` artifact — "nothing was ever shipped" — gets only
 * the fixed "Not yet published" label, never an appended reason string,
 * even though the underlying loader result happens to carry one of its own
 * for logging purposes).
 */
const reasonFor = (
  status: FindingStepStatus,
  result: { readonly status: string; readonly reason?: string } | undefined
): string | undefined => (status === 'unavailable' || status === 'invalid' ? result?.reason : undefined);

// ---------------------------------------------------------------------------
// Step 1: Rewiring null
// ---------------------------------------------------------------------------

const buildRewiringNullStep = (inputs: BuildFindingStepsInputs): FindingStep => {
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

// ---------------------------------------------------------------------------
// Step 2: Mirrored decoder
// ---------------------------------------------------------------------------

const buildMirroredDecoderStep = (inputs: BuildFindingStepsInputs): FindingStep => {
  const provenance = provenanceFor(
    inputs.manifest,
    inputs.manifest?.nullExplanation,
    'Null-explanation result (null-explanation-v1.json)',
    'null-explanation-report.md',
    inputs.dataBaseUrl
  );
  const base = {
    id: 'mirrored-decoder',
    title: 'Mirrored decoder',
    condition: 'authored' as const,
    provenance: provenance ? [provenance] : []
  };

  // The mirrored-decoder check compares the null-explanation artifact's
  // `variants.flipBoth.bioPercentile` against the *un-mirrored* baseline
  // from the rewiring-null artifact (`NullExplanationNote.svelte`'s own
  // `baselinePercentile` prop) — both must have actually resolved to `'ok'`
  // before this step has anything true to say.
  const rewiringStatus = rewiringNullStepStatus(inputs.rewiringNull);
  if (rewiringStatus !== 'ok' || inputs.rewiringNull?.status !== 'ok') {
    return { ...base, status: rewiringStatus, reason: reasonFor(rewiringStatus, inputs.rewiringNull) };
  }
  const explanationStatus = sidecarStepStatus(inputs.nullExplanation);
  if (explanationStatus !== 'ok' || inputs.nullExplanation?.status !== 'ok') {
    return { ...base, status: explanationStatus, reason: reasonFor(explanationStatus, inputs.nullExplanation) };
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

// ---------------------------------------------------------------------------
// Step 3: Explanation
// ---------------------------------------------------------------------------

const buildExplanationStep = (inputs: BuildFindingStepsInputs): FindingStep => {
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
  const metrics = inputs.nullExplanation.data.finding.qualifyingMetrics;
  if (metrics.length === 0) {
    const sentence =
      'Under the authored (hand-written) decoder, no clearance→thrust metric independently passes both ' +
      'predeclared gates, under this model.';
    return { ...base, status: 'ok', sentence };
  }
  const metricList = metrics.map((metric) => `${metric.name} (ρ = ${formatRho(metric.spearman)})`).join('; ');
  const sentence =
    `Under the authored (hand-written) decoder, ${metrics.length} metric${metrics.length === 1 ? '' : 's'} ` +
    `independently pass${metrics.length === 1 ? 'es' : ''} both predeclared clearance→thrust gates: ${metricList}, ` +
    `under this model.`;
  return { ...base, status: 'ok', sentence };
};

// ---------------------------------------------------------------------------
// Step 4: Intervention (authored)
// ---------------------------------------------------------------------------

const buildInterventionStep = (inputs: BuildFindingStepsInputs): FindingStep => {
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

// ---------------------------------------------------------------------------
// Step 5: Trained null
// ---------------------------------------------------------------------------

const buildTrainedNullStep = (inputs: BuildFindingStepsInputs): FindingStep => {
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

// ---------------------------------------------------------------------------
// Step 6: Trained interventions
// ---------------------------------------------------------------------------

const buildTrainedInterventionsStep = (inputs: BuildFindingStepsInputs): FindingStep => {
  const provenance = provenanceFor(
    inputs.manifest,
    inputs.manifest?.pathwayInterventions,
    'Pathway-interventions result, trained section (pathway-interventions-v1.json)',
    'pathway-interventions-report.md',
    inputs.dataBaseUrl
  );
  const base = {
    id: 'trained-interventions',
    title: 'Trained interventions',
    condition: 'both' as const,
    provenance: provenance ? [provenance] : []
  };
  const status = sidecarStepStatus(inputs.pathwayInterventions);
  if (status !== 'ok' || inputs.pathwayInterventions?.status !== 'ok') {
    return { ...base, status, reason: reasonFor(status, inputs.pathwayInterventions) };
  }
  const { authored, trained } = inputs.pathwayInterventions.data;

  // Fixed sentence pattern (`01-findings-panel.md`): "With trained readouts,
  // {all three seeds agree: | seeds disagree:} {trained category} — {this
  // does not reproduce | this matches} the authored decoder's {authored
  // category} result, under this model." The category *values* are always
  // shown (never only `trainedRobust`), so a robust *absence* of effect
  // (e.g. `no-specific-effect`) can never read as confirmation — and when
  // the seeds disagree, there is no single robustly-reproduced category, so
  // "matches" is never claimed in that branch either.
  const agreementPrefix = trained.trainedRobust ? 'all three seeds agree:' : 'seeds disagree:';
  const categoryText = trained.trainedRobust
    ? trained.perSeedCategory[P_TRAINER_SEEDS[0]]
    : P_TRAINER_SEEDS.map((seed) => `seed ${seed}: ${trained.perSeedCategory[seed]}`).join(', ');
  const matches = trained.trainedRobust && trained.perSeedCategory[P_TRAINER_SEEDS[0]] === authored.category;
  const reproductionClause = matches ? 'this matches' : 'this does not reproduce';
  const sentence =
    `With trained readouts, ${agreementPrefix} ${categoryText} — ${reproductionClause} the authored decoder's ` +
    `${authored.category} result, under this model.`;
  return { ...base, status: 'ok', sentence };
};

// ---------------------------------------------------------------------------
// Step 7: Behavior repertoire
// ---------------------------------------------------------------------------

/**
 * Always `'missing'` in this WP: `.agents/plans/repertoire-null` has not
 * landed (no `manifest.behaviorRepertoireNull` field exists on
 * `ArenaManifest` yet — see `00-overview.md`'s "Repository findings" and
 * this bean's own follow-up note), so there is no loader to call and no
 * artifact to cite. `FindingsPanel.svelte` links out to `#atlas` next to
 * this step regardless of status (the atlas is a separate hash route, not
 * an artifact this step loads — see `00-overview.md`).
 */
const buildBehaviorRepertoireStep = (): FindingStep => ({
  id: 'behavior-repertoire',
  title: 'Behavior repertoire',
  status: 'missing',
  condition: 'both',
  provenance: []
});

/**
 * Builds all seven Findings-panel steps, in evidence-chain order
 * (`01-findings-panel.md`'s step list). Pure and synchronous: every input
 * is a value the caller already has in scope (controller callback mirrors),
 * never a fetch performed here.
 */
export const buildFindingSteps = (inputs: BuildFindingStepsInputs): readonly FindingStep[] => [
  buildRewiringNullStep(inputs),
  buildMirroredDecoderStep(inputs),
  buildExplanationStep(inputs),
  buildInterventionStep(inputs),
  buildTrainedNullStep(inputs),
  buildTrainedInterventionsStep(inputs),
  buildBehaviorRepertoireStep()
];
