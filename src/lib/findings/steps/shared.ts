/**
 * `src/lib/findings/steps.ts`'s shared types and cross-step plumbing,
 * extracted into its own module (a thermo-maintainability review, item I2:
 * `steps.ts` had grown to 808 lines, ~65-115 lines per study, with a second
 * in-flight step -- `flyarena-hd0j`, readout attribution -- already queued
 * to land in the same file next). Each `buildXStep` function now lives in
 * its own `src/lib/findings/steps/<study>.ts` file; this module holds
 * exactly what every one of them needs: the `FindingStep`/`FindingStepStatus`/
 * `BuildFindingStepsInputs` types, and the small set of already-shared
 * helpers (`provenanceFor`, `reasonFor`, `sidecarStepStatus`,
 * `rewiringNullStepStatus`, `worseStatus`/`STATUS_SEVERITY`,
 * `findingStepStatusLabel`) no single step builder owns.
 *
 * This is a pure move: every symbol here is byte-identical to what used to
 * live directly in `steps.ts` (only doc comments were adjusted to point at
 * the new file layout), and `steps.ts` itself re-exports the public types/
 * `findingStepStatusLabel` from here unchanged, so no import path outside
 * this directory needs to change.
 */

import type { ArenaManifest, SidecarManifestEntry } from '../../experiment/assets';
import type { RewiringNullLoadResult } from '../../experiment/rewiringNull';
import type { NullExplanationLoadResult } from '../../experiment/nullExplanation';
import type { PathwayInterventionsLoadResult } from '../../experiment/pathwayInterventions';
import type { RepertoireNullLoadResult } from '../../experiment/repertoireNull';
import type { TaskGeneralityLoadResult } from '../../experiment/taskGenerality';
import type { SelectionRobustnessLoadResult } from '../../experiment/selectionRobustness';
import { githubDocUrl } from '../../ui/links';

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
   * naturally a small collection rather than one flat fact. Each item is
   * already fully formatted text -- the panel renders it verbatim, it does
   * not know or care which study produced it.
   *
   * A thermo-maintainability review (Important, I1) found this field used
   * to be two separately-typed, separately-rendered fields --
   * `perTask: {id, authored, trained}[]` (task-generality only) and
   * `perSelection: {id, summary}[]` (selection-robustness only) -- with
   * `FindingsPanel.svelte` carrying two structurally identical `{#if}`
   * render blocks (down to reusing the same `.per-task` CSS class for
   * both, "an early tell the two are the same thing"). Both builders
   * already produced fully-formatted text per item before this merge
   * (`buildTaskGeneralityStep` via `` `authored: ${authoredTaskClause(task)}; trained: ${trainedTaskClauseShort(task)}` ``,
   * `buildSelectionRobustnessStep` via `selectionSummary`) -- this field
   * is that same shape, generalized to any study, so a third per-study
   * step (e.g. `flyarena-hd0j`, readout attribution) has one obvious place
   * to put its own per-item detail instead of copying a third field/render
   * block. Absent (never an empty array) for every step whose result fits
   * in one sentence.
   */
  readonly details?: readonly { readonly id: string; readonly text: string }[];
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

export const provenanceFor = (
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
export const rewiringNullStepStatus = (result: RewiringNullLoadResult | undefined): Exclude<FindingStepStatus, 'ok'> | 'ok' => {
  if (result === undefined) return 'loading';
  if (result.status === 'absent') return 'missing';
  return result.status;
};

export const sidecarStepStatus = (
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
export const reasonFor = (
  status: FindingStepStatus,
  result: { readonly status: string; readonly reason?: string } | undefined
): string | undefined => (status === 'unavailable' || status === 'invalid' ? result?.reason : undefined);

/**
 * (dual review, Important) For a step whose sentence draws on *two* source
 * artifacts (currently only the mirrored-decoder step), the displayed
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
export const STATUS_SEVERITY: Record<FindingStepStatus, number> = { invalid: 4, unavailable: 3, missing: 2, loading: 1, ok: 0 };

export const worseStatus = (a: FindingStepStatus, b: FindingStepStatus): FindingStepStatus =>
  STATUS_SEVERITY[b] > STATUS_SEVERITY[a] ? b : a;
