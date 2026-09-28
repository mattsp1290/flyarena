/**
 * `public/data/task-generality-v1.json` (WP4 of `.agents/plans/task-generality`,
 * `04-artifact-and-findings.md`): whether the rewiring-null and pathway-
 * intervention findings hold beyond the default foraging task, across four
 * predeclared `ArenaConfig` variants (`docs/task-generality-report.md` has
 * the full method, per-task results, and limitations). The real producer's
 * full type is `scripts/null/task-generality-report.ts`'s
 * `TaskGeneralityArtifact` — that module is a Node-only pipeline (not part
 * of the browser bundle), so this file independently authors and validates
 * just the subset the Findings panel's step actually renders, the same
 * "reimplemented, not imported" discipline `./rewiringNull.ts`/
 * `./pathwayInterventions.ts`/`./repertoireNull.ts` already document for
 * their own artifacts. Field *names* and nesting mirror the producer's real
 * JSON one-to-one (a subset of its fields, not a flattened re-shape) so the
 * two can never silently diverge on what a field is called.
 */

import { fetchAndVerifySidecarJson, type ArenaManifest } from './assets';
import { P_TRAINER_SEEDS } from './pathwayInterventions';
import type { SidecarLoadResult } from './sidecarResult';

/** `00-overview.md`'s predeclared authored-decoder categories, plus `'degenerate'` (the predeclared guard). */
export type TaskGeneralityPathwayCategory = 'pathway-supported' | 'edge-class-effect' | 'generic-rewiring-effect' | 'not-supported' | 'degenerate';

/** The trained side's category vocabulary — see `pathwayInterventions.ts`'s own `PathwayInterventionsTrainedCategory` doc comment for why `'no-specific-effect'` is not a renamed authored category. */
export type TaskGeneralityTrainedCategory = 'pathway-supported' | 'edge-class-effect' | 'no-specific-effect';

const isPathwayCategory = (value: unknown): value is TaskGeneralityPathwayCategory =>
  value === 'pathway-supported' || value === 'edge-class-effect' || value === 'generic-rewiring-effect' || value === 'not-supported' || value === 'degenerate';

const isTrainedCategory = (value: unknown): value is TaskGeneralityTrainedCategory =>
  value === 'pathway-supported' || value === 'edge-class-effect' || value === 'no-specific-effect';

export type TaskGeneralityTrained =
  | { readonly degenerate: true }
  | {
      readonly degenerate: false;
      readonly trainedRobust: boolean;
      readonly category: TaskGeneralityTrainedCategory;
      readonly perSeed: Readonly<Record<(typeof P_TRAINER_SEEDS)[number], TaskGeneralityTrainedCategory>>;
    };

export interface TaskGeneralityTask {
  readonly id: string;
  readonly categorized: boolean;
  readonly null: { readonly nullHolds: boolean; readonly degenerate: boolean };
  readonly pathway: { readonly category: TaskGeneralityPathwayCategory; readonly generalizes: boolean };
  readonly trained: TaskGeneralityTrained;
}

export interface TaskGeneralityOverall {
  readonly verdict: 'general' | 'task-dependent';
  readonly nonDegenerateCount: number;
  readonly totalCount: number;
}

export interface TaskGeneralityArtifact {
  readonly version: 1;
  readonly sources: { readonly rewiringNullSha: string; readonly pathwayInterventionsSha: string };
  readonly tasks: readonly TaskGeneralityTask[];
  readonly overall: { readonly authored: TaskGeneralityOverall; readonly trained: TaskGeneralityOverall };
}

export type TaskGeneralityLoadResult = SidecarLoadResult<TaskGeneralityArtifact>;

const isOverall = (value: unknown): value is TaskGeneralityOverall => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    (v.verdict === 'general' || v.verdict === 'task-dependent') &&
    typeof v.nonDegenerateCount === 'number' &&
    typeof v.totalCount === 'number'
  );
};

const isTrained = (value: unknown): value is TaskGeneralityTrained => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  if (typeof v.degenerate !== 'boolean') return false;
  if (v.degenerate) return true;
  if (typeof v.trainedRobust !== 'boolean' || !isTrainedCategory(v.category)) return false;
  const perSeed = v.perSeed as Record<string, unknown> | undefined;
  return Boolean(perSeed) && P_TRAINER_SEEDS.every((seed) => isTrainedCategory((perSeed as Record<string, unknown>)[seed]));
};

const isTask = (value: unknown): value is TaskGeneralityTask => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  const nullField = v.null as Record<string, unknown> | undefined;
  const pathway = v.pathway as Record<string, unknown> | undefined;
  return (
    typeof v.id === 'string' &&
    typeof v.categorized === 'boolean' &&
    Boolean(nullField) &&
    typeof (nullField as Record<string, unknown>).nullHolds === 'boolean' &&
    typeof (nullField as Record<string, unknown>).degenerate === 'boolean' &&
    Boolean(pathway) &&
    isPathwayCategory((pathway as Record<string, unknown>).category) &&
    typeof (pathway as Record<string, unknown>).generalizes === 'boolean' &&
    isTrained(v.trained)
  );
};

/** `00-overview.md`'s overall-verdict rule, recomputed from the per-task data it is supposed to summarize (mirrors `pathwayInterventions.ts`'s/`repertoireNull.ts`'s own "a hash-valid artifact could still ship a summary that disagrees with its own per-item data" cross-checks). */
const recomputeOverall = (
  tasks: readonly TaskGeneralityTask[],
  nonDegenerate: (task: Readonly<TaskGeneralityTask>) => boolean,
  generalizes: (task: Readonly<TaskGeneralityTask>) => boolean
): TaskGeneralityOverall => {
  const eligible = tasks.filter(nonDegenerate);
  const verdict: 'general' | 'task-dependent' = eligible.length >= 3 && eligible.every(generalizes) ? 'general' : 'task-dependent';
  return { verdict, nonDegenerateCount: eligible.length, totalCount: tasks.length };
};

const overallEquals = (a: Readonly<TaskGeneralityOverall>, b: Readonly<TaskGeneralityOverall>): boolean =>
  a.verdict === b.verdict && a.nonDegenerateCount === b.nonDegenerateCount && a.totalCount === b.totalCount;

const validateShape = (value: unknown): { ok: true; data: TaskGeneralityArtifact } | { ok: false; reason: string } => {
  if (typeof value !== 'object' || value === null) {
    return { ok: false, reason: 'task-generality artifact is not a JSON object' };
  }
  const v = value as Record<string, unknown>;
  if (v.version !== 1) return { ok: false, reason: `task-generality artifact has unsupported version ${String(v.version)}` };

  const sources = v.sources as Record<string, unknown> | undefined;
  if (!sources || typeof sources.rewiringNullSha !== 'string' || typeof sources.pathwayInterventionsSha !== 'string') {
    return { ok: false, reason: 'task-generality artifact is missing sources.rewiringNullSha/sources.pathwayInterventionsSha' };
  }

  if (!Array.isArray(v.tasks) || v.tasks.length === 0 || !v.tasks.every(isTask)) {
    return { ok: false, reason: 'task-generality artifact has a malformed "tasks" array' };
  }
  const tasks = v.tasks as TaskGeneralityTask[];

  const overall = v.overall as Record<string, unknown> | undefined;
  if (!overall || !isOverall(overall.authored) || !isOverall(overall.trained)) {
    return { ok: false, reason: 'task-generality artifact has a malformed "overall" field' };
  }
  const authoredOverall = overall.authored as TaskGeneralityOverall;
  const trainedOverall = overall.trained as TaskGeneralityOverall;

  const recomputedAuthored = recomputeOverall(
    tasks,
    (t) => t.categorized,
    (t) => t.null.nullHolds && t.pathway.generalizes
  );
  if (!overallEquals(recomputedAuthored, authoredOverall)) {
    return { ok: false, reason: 'task-generality artifact overall.authored disagrees with its own per-task data' };
  }
  const recomputedTrained = recomputeOverall(
    tasks,
    (t) => !t.trained.degenerate,
    (t) => !t.trained.degenerate && t.trained.trainedRobust && t.trained.category === 'pathway-supported'
  );
  if (!overallEquals(recomputedTrained, trainedOverall)) {
    return { ok: false, reason: 'task-generality artifact overall.trained disagrees with its own per-task data' };
  }

  return {
    ok: true,
    data: {
      version: 1,
      sources: { rewiringNullSha: sources.rewiringNullSha, pathwayInterventionsSha: sources.pathwayInterventionsSha },
      tasks,
      overall: { authored: authoredOverall, trained: trainedOverall }
    }
  };
};

/**
 * Fetch, sha256-verify, and structurally validate `task-generality-v1.json`
 * (`manifest.taskGenerality`), then cross-check it against the manifest's
 * own pinned rewiring-null and pathway-interventions artifacts — never
 * throws, matching every sibling loader's "never throws, always return a
 * reasoned status" contract, so a missing/tampered/malformed artifact only
 * ever hides or degrades the Findings panel's task-generality step, never
 * the rest of the experiment.
 *
 * `dataBaseUrl` must be the same value the caller passes to the other
 * `loadX` functions (`${import.meta.env.BASE_URL}data` in production) so
 * this artifact resolves under the app's real deployment base path too.
 */
export const loadTaskGenerality = async (
  manifest: ArenaManifest,
  dataBaseUrl: string
): Promise<TaskGeneralityLoadResult> => {
  const fetched = await fetchAndVerifySidecarJson(manifest.taskGenerality, dataBaseUrl, 'task-generality artifact');
  if (fetched.status === 'no-entry') {
    return { status: 'missing', reason: 'The manifest has no taskGenerality artifact entry.' };
  }
  if (fetched.status === 'fetch-error') {
    return { status: 'unavailable', reason: fetched.reason };
  }
  if (fetched.status === 'hash-mismatch' || fetched.status === 'parse-error') {
    return { status: 'invalid', reason: fetched.reason };
  }

  const validated = validateShape(fetched.parsed);
  if (!validated.ok) return { status: 'invalid', reason: validated.reason };
  const data = validated.data;

  const shippedRewiringNullSha = manifest.rewiringNull?.sha256;
  if (!shippedRewiringNullSha) {
    return { status: 'invalid', reason: 'manifest is missing rewiringNull.sha256, needed to cross-check the task-generality artifact' };
  }
  if (data.sources.rewiringNullSha !== shippedRewiringNullSha) {
    return {
      status: 'invalid',
      reason: `task-generality sources.rewiringNullSha does not match the manifest's rewiring-null artifact (${shippedRewiringNullSha}) — stale artifact`
    };
  }
  const shippedPathwayInterventionsSha = manifest.pathwayInterventions?.sha256;
  if (!shippedPathwayInterventionsSha) {
    return { status: 'invalid', reason: 'manifest is missing pathwayInterventions.sha256, needed to cross-check the task-generality artifact' };
  }
  if (data.sources.pathwayInterventionsSha !== shippedPathwayInterventionsSha) {
    return {
      status: 'invalid',
      reason: `task-generality sources.pathwayInterventionsSha does not match the manifest's pathway-interventions artifact (${shippedPathwayInterventionsSha}) — stale artifact`
    };
  }

  return { status: 'ok', data };
};
