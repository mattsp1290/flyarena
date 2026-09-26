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
import {
  P_TRAINER_SEEDS,
  type PathwayInterventionsLoadResult,
  type PathwayInterventionsTrainedCategory
} from '../experiment/pathwayInterventions';
import { githubDocUrl } from '../ui/links';
import { describeTrainedCategory, formatPercentile, formatRho } from './format';

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
  // `SidecarManifestEntry` (`{ artifact: string; sha256: string }`) alone
  // already covers `manifest.rewiringNull`'s own inline type structurally
  // (thermo review, maintainability Suggestion) — no second union member
  // needed.
  entry: SidecarManifestEntry | undefined,
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

/**
 * (dual review, Important) For a step whose sentence draws on *two* source
 * artifacts (currently only step 2, the mirrored decoder), the displayed
 * status must be whichever input's status is worse, not always the same
 * one — otherwise a genuine verification failure on the "wrong" input could
 * be hidden behind the other input's more benign status (e.g. the rewiring
 * null merely `'missing'` while the null-explanation artifact is actually
 * `'invalid'`). `'invalid'` (a real verification failure) always outranks
 * `'unavailable'` (a retryable fetch failure), which outranks `'missing'`
 * (nothing was ever shipped), which outranks `'loading'` (not yet
 * resolved) — `'ok'` is the least severe and never wins over a degraded
 * status from the other input.
 */
const STATUS_SEVERITY: Record<FindingStepStatus, number> = { invalid: 4, unavailable: 3, missing: 2, loading: 1, ok: 0 };

const worseStatus = (a: FindingStepStatus, b: FindingStepStatus): FindingStepStatus =>
  STATUS_SEVERITY[b] > STATUS_SEVERITY[a] ? b : a;

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
  // {all N seeds agree: | seeds disagree:} {trained category} — {this does
  // not reproduce | this matches | this is consistent with} the authored
  // decoder's {authored category} result, under this model." The category
  // *values* are always shown (never only `trainedRobust`), so a robust
  // *absence* of effect (e.g. `no-specific-effect`) can never read as
  // confirmation — and when the seeds disagree, there is no single
  // robustly-reproduced category, so neither "matches" nor "is consistent
  // with" is ever claimed in that branch.
  const agreementPrefix = trained.trainedRobust ? `all ${P_TRAINER_SEEDS.length} seeds agree:` : 'seeds disagree:';
  // (thermo review, methodology I1/I2) Rendered through `describeTrainedCategory`
  // (`./format.ts`), never the raw enum slug — `'no-specific-effect'` reads
  // as "no effect" to an unprimed reader, when it actually means the
  // trained side's predeclared rules cannot decide between
  // `generic-rewiring-effect`/`not-supported`. The robust branch also
  // carries the short "reporting convention" caveat, but only when the
  // artifact's own `trained.note` field actually discloses it (never
  // unconditionally) — the per-seed "seeds disagree" listing omits it so a
  // dissenting no-specific-effect seed doesn't repeat the same caveat.
  const categoryText = trained.trainedRobust
    ? describeTrainedCategory(trained.perSeedCategory[P_TRAINER_SEEDS[0]], { withCaveat: Boolean(trained.note) })
    : P_TRAINER_SEEDS.map((seed) => `seed ${seed}: ${describeTrainedCategory(trained.perSeedCategory[seed])}`).join(', ');

  // (dual review, Important) The authored and trained categories are two
  // *different* vocabularies, not the same one — `no-specific-effect` is
  // `pathwayInterventions.ts`'s own documented merge of the authored
  // `generic-rewiring-effect`/`not-supported` split (the trained side's
  // predeclared rules cannot decide that finer split). Comparing the raw
  // strings would wrongly read an authored `not-supported` result next to a
  // robust trained `no-specific-effect` as "does not reproduce", when the
  // trained side is actually consistent with it — it just cannot confirm
  // which of the two merged authored categories held. Map the authored
  // category into the trained vocabulary before comparing, and only claim
  // "matches" when both sides literally agree (never merely map-equal).
  const authoredAsTrainedCategory: PathwayInterventionsTrainedCategory =
    authored.category === 'generic-rewiring-effect' || authored.category === 'not-supported'
      ? 'no-specific-effect'
      : (authored.category as PathwayInterventionsTrainedCategory);
  const trainedCategory: PathwayInterventionsTrainedCategory | undefined = trained.trainedRobust
    ? trained.perSeedCategory[P_TRAINER_SEEDS[0]]
    : undefined;
  const reproductionClause =
    trainedCategory === undefined || trainedCategory !== authoredAsTrainedCategory
      ? 'this does not reproduce'
      : (trainedCategory as string) === (authored.category as string)
        ? 'this matches'
        : 'this is consistent with (the trained rules cannot split generic-rewiring-effect from not-supported)';
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
