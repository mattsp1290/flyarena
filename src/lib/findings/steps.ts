/**
 * WP1 of `.agents/plans/findings-tour` (`01-findings-panel.md`): pure
 * builder for the Findings panel's (now eight-step, since task-generality
 * WP4) evidence chain
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
 * the panel still renders every step (`aria-current="step"`/"Step N of M"
 * stay honest and stable, derived from the array length), swapping the templated sentence for a short,
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
import type { RepertoireNullLoadResult } from '../experiment/repertoireNull';
import type { TaskGeneralityLoadResult, TaskGeneralityTask, TaskGeneralityTrainedCategory } from '../experiment/taskGenerality';
import type { RobustnessVerdict, SelectionResult, SelectionRobustnessLoadResult } from '../experiment/selectionRobustness';
import { metricVerdictLabel } from '../atlas/repertoireStrip';
import { CELL_COUNT } from '../atlas/types';
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
  /** One entry per source artifact this step's sentence is templated from; empty when the step has never had a real source to cite. */
  readonly provenance: readonly FindingStepProvenance[];
  /** The underlying loader's own honest reason string, present only when `status` is `'unavailable'` or `'invalid'`. */
  readonly reason?: string;
  /**
   * Optional per-item detail rendered as its own `<ul>/<li>` list under
   * `sentence` (`FindingsPanel.svelte`), for a step whose result is
   * naturally a small collection rather than one flat fact -- currently
   * only the task-generality step (`buildTaskGeneralityStep`). Thermo
   * review (Important): an earlier version crammed this same per-task
   * detail into `sentence` itself, producing a single ~1050-character
   * run-on sentence with no navigable structure for a screen-reader user
   * stepping through the panel. Absent (never an empty array) for every
   * step whose result fits in one sentence.
   */
  readonly perTask?: readonly { readonly id: string; readonly authored: string; readonly trained: string }[];
  /**
   * Same "per-item detail moves out of `sentence`" rationale as `perTask`
   * above, for the selection-robustness step's four selections -- a
   * `{authored, trained}` shape does not fit here (the whole study is
   * authored-decoder-only; see `.agents/plans/selection-robustness/00-overview.md`'s
   * non-goals), so this is its own field rather than overloading `perTask`
   * with an empty `trained` string.
   */
  readonly perSelection?: readonly { readonly id: string; readonly summary: string }[];
}

export interface BuildFindingStepsInputs {
  readonly manifest: ArenaManifest | undefined;
  readonly dataBaseUrl: string;
  readonly rewiringNull: RewiringNullLoadResult | undefined;
  readonly nullExplanation: NullExplanationLoadResult | undefined;
  readonly pathwayInterventions: PathwayInterventionsLoadResult | undefined;
  /** WP3 of `.agents/plans/repertoire-null`, following `findings-tour`'s own `01-findings-panel.md` ("optional `repertoireNull`" input). `undefined` while the repertoire-null load has not yet resolved -- `buildBehaviorRepertoireStep` below reports that as `'loading'`, the same convention every other step's `undefined` input already uses. */
  readonly repertoireNull: RepertoireNullLoadResult | undefined;
  /** WP4 of `.agents/plans/task-generality`. `undefined` while the task-generality load has not yet resolved -- the same `'loading'` convention every other step's `undefined` input already uses. */
  readonly taskGenerality: TaskGeneralityLoadResult | undefined;
  /** WP3 of `.agents/plans/selection-robustness`. `undefined` while the selection-robustness load has not yet resolved -- the same `'loading'` convention every other step's `undefined` input already uses. */
  readonly selectionRobustness: SelectionRobustnessLoadResult | undefined;
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
  result:
    | NullExplanationLoadResult
    | PathwayInterventionsLoadResult
    | RepertoireNullLoadResult
    | TaskGeneralityLoadResult
    | SelectionRobustnessLoadResult
    | undefined
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
// Task generality
// ---------------------------------------------------------------------------

/**
 * WP4 of `.agents/plans/task-generality`: whether the rewiring-null and
 * pathway-intervention findings above hold beyond the default foraging
 * task, across four predeclared `ArenaConfig` variants. Inserted after
 * "Trained interventions" and before "Behavior repertoire" in this array's
 * own order (`04-artifact-and-findings.md`'s own placement) -- this array's
 * order no longer determines the panel's *display* order, which
 * `../findings/sections.ts#groupSteps` computes separately (WP1 of
 * `.agents/plans/consolidated-release`); see that module's own doc comment
 * for the real rendered grouping. `condition: 'both'`: the sentence states
 * both the authored and trained overall verdicts together, mirroring the
 * "Trained interventions" step's own "both decoders in one sentence" shape.
 * Reuses the same `P_TRAINER_SEEDS` (already imported above for that step)
 * -- the trained decoder's fixed trainer-seed set is study-wide, not
 * per-artifact.
 */

/**
 * One task's authored clause, e.g. `"pathway-supported"` or `"degenerate"`.
 * Null-holds is called out only when it does *not* hold (the common case
 * needs no extra clause).
 */
const authoredTaskClause = (task: Readonly<TaskGeneralityTask>): string => {
  if (!task.categorized) return 'degenerate';
  const holdsSuffix = task.null.nullHolds ? '' : ', null does not hold';
  return `${task.pathway.category}${holdsSuffix}`;
};

/**
 * The short per-task category label -- deliberately *not*
 * `describeTrainedCategory` (`./format.ts`), which always expands
 * `'no-specific-effect'` to the full "no specific effect (neither
 * pathway-supported nor edge-class...)" gloss with no way to omit it. That
 * gloss is exactly what made the old per-task-per-seed sentence repeat the
 * same 7-word phrase up to 8 times (thermo review, Important, both
 * reviewers). Here the gloss is stated exactly once, in
 * `buildTaskGeneralityStep`'s own `sentence`; every per-task/per-seed use
 * gets only the plain label.
 */
const shortTrainedCategoryLabel = (category: TaskGeneralityTrainedCategory): string =>
  category === 'no-specific-effect' ? 'no specific effect' : category;

/**
 * One task's trained clause -- "not robust" is always stated inline (never
 * a footnote), naming only the seed(s) that dissent from the headline
 * (representative-seed) category, e.g. `"no specific effect, not robust
 * (seed 202 pathway-supported)"`.
 */
const trainedTaskClauseShort = (task: Readonly<TaskGeneralityTask>): string => {
  const trained = task.trained;
  if (trained.degenerate) return 'degenerate';
  const category = shortTrainedCategoryLabel(trained.category);
  if (trained.trainedRobust) return `${category} (robust)`;
  // Dissenting = disagrees with the headline value (`trained.category`,
  // the representative trainer seed's own category -- already validated by
  // the loader to equal `perSeed[P_TRAINER_SEEDS[0]]`), not a freshly
  // computed "majority": this reuses the one representative-category
  // convention the producer/report already establish, rather than
  // introducing a second, different voting rule for this one surface.
  const dissenting = P_TRAINER_SEEDS.filter((seed) => trained.perSeed[seed] !== trained.category);
  const dissentingText = dissenting.map((seed) => `seed ${seed} ${shortTrainedCategoryLabel(trained.perSeed[seed])}`).join(', ');
  return `${category}, not robust (${dissentingText})`;
};

/** Whether any non-degenerate task's trained result actually uses `'no-specific-effect'` at any seed -- the same condition the report's own `buildResultDependentLimitations`/`buildClearanceLimitations` gate their disclosures on, reused here so the step and the report never disagree about when this needs explaining. */
const anyNoSpecificEffect = (tasks: readonly Readonly<TaskGeneralityTask>[]): boolean =>
  tasks.some((t) => !t.trained.degenerate && Object.values(t.trained.perSeed).includes('no-specific-effect'));

const buildTaskGeneralityStep = (inputs: BuildFindingStepsInputs): FindingStep => {
  const provenance = provenanceFor(
    inputs.manifest,
    inputs.manifest?.taskGenerality,
    'Task-generality result (task-generality-v1.json)',
    'task-generality-report.md',
    inputs.dataBaseUrl
  );
  const status = sidecarStepStatus(inputs.taskGenerality);
  const base = {
    id: 'task-generality',
    title: 'Task generality',
    condition: 'both' as const,
    provenance: provenance ? [provenance] : []
  };
  if (status !== 'ok' || inputs.taskGenerality?.status !== 'ok') {
    return { ...base, status, reason: reasonFor(status, inputs.taskGenerality) };
  }
  const { tasks, overall, trainedCategoryNote } = inputs.taskGenerality.data;

  // Thermo review (Important, both reviewers): a single sentence folding in
  // all 4 tasks' full per-seed detail ran to ~1050 characters with no
  // navigable structure. `sentence` now states only the two overall
  // verdicts (plus the "no specific effect" definition/convention
  // disclosure, stated once here rather than once per task); the per-task
  // breakdown moves to `perTask`, rendered as its own `<ul>/<li>` list by
  // `FindingsPanel.svelte` -- a natural stop for both a sighted skim and a
  // screen-reader user walking the step.
  const authoredSummary =
    overall.authored.verdict === 'general'
      ? `general (${overall.authored.nonDegenerateCount} of ${overall.authored.totalCount} non-degenerate, null holds and the pathway generalizes in every one)`
      : `task-dependent (${overall.authored.nonDegenerateCount} of ${overall.authored.totalCount} non-degenerate; see per-task detail below)`;
  const trainedSummary =
    overall.trained.verdict === 'general'
      ? `general (${overall.trained.nonDegenerateCount} of ${overall.trained.totalCount} non-degenerate)`
      : `task-dependent (${overall.trained.nonDegenerateCount} of ${overall.trained.totalCount} non-degenerate; see per-task detail below)`;

  // Defined once here (never per task/per seed) -- every step's own
  // sentence must still end with "under this model." (the fixed ending
  // every step in this panel shares), so this clause is inserted *before*
  // that ending, not appended after it.
  const definitionClause = anyNoSpecificEffect(tasks)
    ? `; "no specific effect" means neither pathway-supported nor edge-class holds${trainedCategoryNote ? ' (a reporting convention adopted after the trained scores were known, not a predeclared category)' : ''}`
    : '';

  const sentence =
    `Across ${tasks.length} task variants, authored: ${authoredSummary}; trained: ${trainedSummary}${definitionClause}, under this model.`;

  // Every task gets its own list item regardless of the overall verdicts
  // (never conditionally hidden), each stating both decoders' results --
  // "not robust" and the dissenting seed(s) stay inline (never a footnote),
  // and no-movement's degenerate P/C/M category is never merged with any
  // other clause (its authored text is simply "degenerate"; the
  // independently-valid Q-vs-MQ result lives only in the full report, not
  // duplicated here).
  const perTask = tasks.map((task) => ({
    id: task.id,
    authored: authoredTaskClause(task),
    trained: trainedTaskClauseShort(task)
  }));

  return { ...base, status: 'ok', sentence, perTask };
};

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
const buildBehaviorRepertoireStep = (inputs: BuildFindingStepsInputs): FindingStep => {
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
  // `../atlas/repertoireStrip.ts#buildRepertoireStripText`'s identical fix
  // and doc comment for the full reasoning; `metricVerdictLabel` (shared
  // from that same module, not re-implemented here) recomputes each
  // metric's own verdict from data already on `primary`.
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

// ---------------------------------------------------------------------------
// Selection robustness
// ---------------------------------------------------------------------------

/** `RobustnessVerdict['verdict']` rendered as plain text -- never a raw `${verdict}` interpolation, so a future fourth verdict value fails to compile here instead of silently rendering `"true"`/`"false"`/`"indeterminate"`'s JS-native stringification. */
const verdictWord = (verdict: RobustnessVerdict['verdict']): string => (verdict === true ? 'true' : verdict === false ? 'false' : 'indeterminate');

/**
 * One selection's null/explanation/pathway result in plain English --
 * `perSelection`'s own per-item detail (this file's top `FindingStep.perSelection`
 * doc comment), not folded into the main `sentence`. Every clause states the
 * mechanism, never a bare status tag: a degenerate pathway names *why*
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
const buildSelectionRobustnessStep = (inputs: BuildFindingStepsInputs): FindingStep => {
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
  const perSelection = selections.map((selection) => ({ id: selection.id, summary: selectionSummary(selection) }));
  return { ...base, status: 'ok', sentence, perSelection };
};

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
