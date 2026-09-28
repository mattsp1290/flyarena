import { mkdirSync, readFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { ARENA_CONFIG, type ArenaConfig } from '../../src/lib/arena/config';
import { ARENA_TASK_IDS, resolveArenaTask, type ArenaTaskId } from '../../src/lib/arena/tasks';
import { requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { mean } from '../training/stats';
import { collectRepoRelativeDependencies, computeSourceIdentitySha256 } from '../lib/import-graph';
import { sortKeysDeep, verifyManifestRoundTrips, type RewiringNullArtifact } from './null-report';
import { quantileIndex } from './null-stats';
import { parseGraphListIndexInfo, type GraphListIndexInfo, type InterventionStatistics, type OutcomeCategory } from './intervention-report';
import {
  classifyRuns,
  trainedArmDistribution,
  trainedTaskResult,
  P_TRAINER_SEEDS,
  type PTrainerSeed,
  type TrainedOutcomeCategory,
  type TrainedTaskResult
} from './intervention-report-trained';
import type { NullTrainedInterventionEvaluationRaw } from './null-trained-evaluate-graph-list';
import type { PathwayInterventionsArtifact } from './intervention-artifact';

/**
 * WP4 of `.agents/plans/task-generality` (`04-artifact-and-findings.md`):
 * combines WP2's per-task authored null/intervention statistics
 * (`null-report.ts --variant-out`'s `null-summary.json`,
 * `intervention-report.ts --stats-only`'s `intervention-stats.json`) and
 * WP3's per-task rescored `trained.json`
 * (`null-trained-evaluate-graph-list.ts`) into `public/data/task-generality-v1.json`
 * (manifest key `taskGenerality`) and `docs/task-generality-report.md`. A
 * separate CLI from `intervention-report.ts`/`intervention-artifact.ts`, on
 * the same precedent those two set for `pathway-interventions` (WP4
 * combining WP2/WP3 into one artifact+report producer, its own file rather
 * than growing an already-near-1000-line module).
 *
 * Every category/percentile/margin below is computed from the four tasks'
 * already-scored inputs -- this module reuses `evaluateDegenerateGuard`
 * (via the already-applied `armDegeneracy`/`p.category`/`q.channelSpecific`
 * fields `intervention-report.ts --stats-only` wrote) and `trainedTaskResult`
 * (`intervention-report-trained.ts`, which itself wraps `evaluateTrainedCategory`/
 * `isTrainedRobust` unchanged) rather than re-implementing either rule.
 *
 * Deterministic and re-run-safe: every input is read once from disk (never
 * re-simulated), `sortKeysDeep`+`JSON.stringify` never iterate a `Map`/`Set`,
 * and nothing here reads the clock -- running this CLI twice against the
 * same inputs produces byte-identical `task-generality-v1.json` bytes
 * (`04-artifact-and-findings.md`'s "Regeneration on the arm64 Spark is
 * byte-identical" acceptance criterion).
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

export const DEFAULT_TASKS_DIR = resolve(repoRoot, 'training/runs/tasks');
export const DEFAULT_GRAPH_INDEX = resolve(repoRoot, 'training/runs/tasks/graphs/index.json');
export const DEFAULT_INTERVENTION_INDEX = resolve(repoRoot, 'training/runs/tasks/interventions/index.json');
export const DEFAULT_REWIRING_NULL = resolve(repoRoot, 'public/data/rewiring-null-v1.json');
export const DEFAULT_PATHWAY_INTERVENTIONS = resolve(repoRoot, 'public/data/pathway-interventions-v1.json');
export const DEFAULT_MANIFEST = resolve(repoRoot, 'public/data/malecns-arena-v1.manifest.json');
export const DEFAULT_OUT = resolve(repoRoot, 'public/data/task-generality-v1.json');
export const DEFAULT_REPORT_MD = resolve(repoRoot, 'docs/task-generality-report.md');

/** The four predeclared task variants (`00-overview.md`'s table) -- `ARENA_TASK_IDS` minus `'default'`, derived rather than a second hand-typed list so a future task addition/removal to `tasks.ts` can't silently drift out of sync with this study. */
export const STUDY_TASK_IDS: readonly Exclude<ArenaTaskId, 'default'>[] = ARENA_TASK_IDS.filter(
  (id): id is Exclude<ArenaTaskId, 'default'> => id !== 'default'
);

// ---------------------------------------------------------------------------
// Producer code identity
// ---------------------------------------------------------------------------

export interface TaskGeneralityProducer {
  readonly script: string;
  readonly sourceSha256: string;
  readonly dependencies: readonly string[];
}

/** Same real-import-graph code-identity scheme as `intervention-artifact.ts`'s `pathwayInterventionsProducer`/`repertoire-report.ts`'s producer -- `scripts/lib/import-graph.ts`'s `collectRepoRelativeDependencies`, not a hand-maintained filename list. */
export const taskGeneralityProducer = (): TaskGeneralityProducer => {
  const dependencies = collectRepoRelativeDependencies(fileURLToPath(import.meta.url), repoRoot);
  return {
    script: 'scripts/null/task-generality-report.ts',
    sourceSha256: computeSourceIdentitySha256(repoRoot, dependencies),
    dependencies
  };
};

// ---------------------------------------------------------------------------
// Config-diff ("changes")
// ---------------------------------------------------------------------------

/**
 * `00-overview.md`'s "Changes from `ARENA_CONFIG`" column, computed by
 * diffing the task's config against `ARENA_CONFIG` field by field --
 * mechanically derived from `src/lib/arena/tasks.ts`'s own `ARENA_TASKS`,
 * never hand-copied from the plan's table (which this module's report still
 * quotes verbatim for the *narrative* column, cross-checked in tests against
 * this diff so the two can't silently disagree).
 */
export const diffArenaConfig = (config: Readonly<ArenaConfig>): Readonly<Record<string, number>> => {
  const changes: Record<string, number> = {};
  for (const key of Object.keys(ARENA_CONFIG) as (keyof ArenaConfig)[]) {
    if (config[key] !== ARENA_CONFIG[key]) changes[key] = config[key];
  }
  return changes;
};

// ---------------------------------------------------------------------------
// Per-task inputs
// ---------------------------------------------------------------------------

export interface TaskGeneralityInputPaths {
  readonly nullSummary: string;
  readonly interventionStats: string;
  readonly trained: string;
}

export const taskInputPaths = (
  authoredDir: string,
  trainedDir: string,
  id: Exclude<ArenaTaskId, 'default'>
): TaskGeneralityInputPaths => ({
  nullSummary: resolve(authoredDir, id, 'null-summary.json'),
  interventionStats: resolve(authoredDir, id, 'intervention-stats.json'),
  trained: resolve(trainedDir, id, 'trained.json')
});

// ---------------------------------------------------------------------------
// Artifact shape
// ---------------------------------------------------------------------------

export interface TaskGeneralitySources {
  readonly rewiringNullSha: string;
  readonly pathwayInterventionsSha: string;
  readonly graphIndexSha: string;
  readonly interventionIndexSha: string;
  readonly producer: TaskGeneralityProducer;
}

export interface TaskGeneralityNullResult {
  readonly bioScore: number;
  readonly nullMedian: number;
  /** The task's own rewired-null 25th-percentile *value* (`quantileIndex`'s low-tail-floor convention, matching `publishedNullFloorValue`'s), not a percentile-rank statistic. */
  readonly p25: number;
  readonly bioPercentile: number;
  readonly degenerate: boolean;
  /** `00-overview.md`'s "biological is below the task's rewired-null 25th percentile". */
  readonly nullHolds: boolean;
}

export interface TaskGeneralityPathwayResult {
  readonly pScore: number;
  readonly cP95: number;
  readonly mP95: number;
  readonly cDegenerate: boolean;
  readonly mDegenerate: boolean;
  /** The raw authored category (`'degenerate'` when the null/C/M-arm degenerate guard fired) -- `generalizes` below is `category === 'pathway-supported'`, restated as a plain boolean for the overall-verdict computation. */
  readonly category: OutcomeCategory;
  readonly generalizes: boolean;
}

export type TaskGeneralityTrainedResult =
  | { readonly degenerate: true }
  | {
      readonly degenerate: false;
      readonly perSeed: Readonly<Record<PTrainerSeed, TrainedOutcomeCategory>>;
      readonly trainedRobust: boolean;
      /**
       * The representative trainer seed's (`P_TRAINER_SEEDS[0]` = 101)
       * category, always populated when non-degenerate -- the same
       * "representative value when robust, but still present (and readable
       * alongside `trainedRobust: false`) otherwise" convention
       * `pathwayInterventions.ts`'s own `perSeedCategory` doc comment
       * establishes. Never read alone without `trainedRobust`: this study's
       * own reviewed framing (`bn show flyarena-nxfx`'s 2026-09-28 note)
       * requires "not robust" to be stated alongside this value, never
       * omitted -- see `renderTaskGeneralityReportMarkdown`.
       */
      readonly category: TrainedOutcomeCategory;
    };

export interface TaskGeneralityTask {
  readonly id: Exclude<ArenaTaskId, 'default'>;
  readonly fingerprint: string;
  readonly changes: Readonly<Record<string, number>>;
  readonly null: TaskGeneralityNullResult;
  readonly pathway: TaskGeneralityPathwayResult;
  /** `pathway.category !== 'degenerate'`. */
  readonly categorized: boolean;
  readonly trained: TaskGeneralityTrainedResult;
}

export interface TaskGeneralityOverall {
  readonly verdict: 'general' | 'task-dependent';
  readonly nonDegenerateCount: number;
  readonly totalCount: number;
}

export interface TaskGeneralityArtifact {
  readonly version: 1;
  readonly sources: TaskGeneralitySources;
  readonly tasks: readonly TaskGeneralityTask[];
  readonly overall: { readonly authored: TaskGeneralityOverall; readonly trained: TaskGeneralityOverall };
  /** The default (shipped) task's own authored/trained results, quoted for the report's context/disclosure section only -- never a member of `tasks`/`overall` above (`00-overview.md`'s "default-task context" is background, not a fifth study task). */
  readonly defaultContext: {
    readonly authoredCategory: Exclude<OutcomeCategory, 'degenerate'>;
    readonly trainedCategory: TrainedOutcomeCategory;
    readonly trainedRobust: boolean;
  };
  readonly host: { readonly arch: string; readonly node: string };
}

// ---------------------------------------------------------------------------
// Build (pure)
// ---------------------------------------------------------------------------

export interface BuildTaskInputs {
  readonly id: Exclude<ArenaTaskId, 'default'>;
  readonly nullSummaryText: string;
  readonly nullSummaryLabel: string;
  readonly interventionStatsText: string;
  readonly interventionStatsLabel: string;
  readonly trainedText: string;
  readonly trainedLabel: string;
}

const NULL_HOLDS_PERCENTILE = 0.25;

const buildTask = (
  inputs: Readonly<BuildTaskInputs>,
  interventionIndexSha: string,
  interventionIndexInfo: Readonly<GraphListIndexInfo>
): TaskGeneralityTask => {
  const resolved = resolveArenaTask(inputs.id);
  const nullSummary = JSON.parse(inputs.nullSummaryText) as RewiringNullArtifact;
  if (nullSummary.arenaTask?.id !== inputs.id || nullSummary.arenaTask.fingerprint !== resolved.fingerprint) {
    throw new Error(`task-generality-report: ${inputs.nullSummaryLabel} is not recorded under arena task "${inputs.id}"`);
  }
  const nullSummaryBytes = Buffer.from(inputs.nullSummaryText, 'utf8');

  const stats = JSON.parse(inputs.interventionStatsText) as InterventionStatistics;
  if (stats.arenaTask?.id !== inputs.id || stats.arenaTask.fingerprint !== resolved.fingerprint) {
    throw new Error(`task-generality-report: ${inputs.interventionStatsLabel} is not recorded under arena task "${inputs.id}"`);
  }
  if (!stats.statsOnly || !stats.armDegeneracy) {
    throw new Error(`task-generality-report: ${inputs.interventionStatsLabel} is not a --stats-only result with a degenerate guard`);
  }
  // Same guard `intervention-artifact.ts`'s `buildPathwayInterventionsArtifact`
  // applies to the default task's `statistics.json` -- checked independently
  // of `diagnosticOnly`'s mere presence (a `--allow-reproduction-mismatch`
  // run) for the same reason: a hand-edited/older file could carry one
  // without the other, and this study must never publish a per-task result
  // computed against a mismatched biological reproduction.
  if (stats.diagnosticOnly || stats.biologicalReproduction?.matches !== true) {
    throw new Error(
      `task-generality-report: ${inputs.interventionStatsLabel} is diagnosticOnly, or its biologicalReproduction.matches is not true -- refusing to publish a task-generality result built from it`
    );
  }
  if (stats.inputs.indexSha256 !== interventionIndexSha) {
    throw new Error(
      `task-generality-report: ${inputs.interventionStatsLabel}'s inputs.indexSha256 (${stats.inputs.indexSha256}) does not match the shared intervention index sha (${interventionIndexSha})`
    );
  }
  const publishedNullSha = sha256Hex(nullSummaryBytes);
  if (stats.inputs.publishedNullSha256 !== publishedNullSha) {
    throw new Error(
      `task-generality-report: ${inputs.interventionStatsLabel} was scored against a different null than ${inputs.nullSummaryLabel} (sha256 ${publishedNullSha})`
    );
  }

  const rewiredScores = nullSummary.rewired.map((entry) => entry.score);
  const sortedRewired = [...rewiredScores].sort((a, b) => a - b);
  const p25 = sortedRewired[quantileIndex(sortedRewired.length, NULL_HOLDS_PERCENTILE)];
  const bioScore = nullSummary.biological.score;

  const nullResult: TaskGeneralityNullResult = {
    bioScore,
    nullMedian: nullSummary.null.median,
    p25,
    bioPercentile: nullSummary.bioPercentile,
    degenerate: nullSummary.null.degenerate,
    nullHolds: bioScore < p25
  };

  const pathwayCategory = stats.p.category;
  const pathwayResult: TaskGeneralityPathwayResult = {
    pScore: stats.p.score,
    cP95: stats.controls.C.p95,
    mP95: stats.controls.M.p95,
    cDegenerate: stats.armDegeneracy.cArm.degenerate,
    mDegenerate: stats.armDegeneracy.mArm.degenerate,
    category: pathwayCategory,
    generalizes: pathwayCategory === 'pathway-supported'
  };

  const trainedRaw = JSON.parse(inputs.trainedText) as NullTrainedInterventionEvaluationRaw;
  if (trainedRaw.arenaTask !== inputs.id || trainedRaw.arenaTaskFingerprint !== resolved.fingerprint) {
    throw new Error(`task-generality-report: ${inputs.trainedLabel} is not recorded under arena task "${inputs.id}"`);
  }
  if (trainedRaw.graphListSha256 !== interventionIndexSha) {
    throw new Error(
      `task-generality-report: ${inputs.trainedLabel}'s graphListSha256 (${trainedRaw.graphListSha256}) does not match the shared intervention index sha (${interventionIndexSha})`
    );
  }
  const trained = buildTrainedTaskResult(trainedRaw, interventionIndexInfo);

  return {
    id: inputs.id,
    fingerprint: resolved.fingerprint,
    changes: diffArenaConfig(resolved.config),
    null: nullResult,
    pathway: pathwayResult,
    categorized: pathwayCategory !== 'degenerate',
    trained
  };
};

const buildTrainedTaskResult = (
  raw: Readonly<NullTrainedInterventionEvaluationRaw>,
  interventionIndexInfo: Readonly<GraphListIndexInfo>
): TaskGeneralityTrainedResult => {
  const { pBySeed, cRuns, mRuns } = classifyRuns(raw, interventionIndexInfo);
  const scoreOf = (run: { readonly movementScore: readonly number[] }): number => mean(run.movementScore);
  const cArm = trainedArmDistribution(cRuns.map(scoreOf));
  const mArm = trainedArmDistribution(mRuns.map(scoreOf));
  const pScores = Object.fromEntries(
    P_TRAINER_SEEDS.map((seed) => {
      const run = pBySeed.get(seed);
      if (!run) throw new Error(`task-generality-report: missing P run at trainer seed ${seed}`);
      return [seed, scoreOf(run)];
    })
  ) as Record<PTrainerSeed, number>;

  const result: TrainedTaskResult = trainedTaskResult(pScores, cArm, mArm);
  if (result.degenerate) return { degenerate: true };
  return {
    degenerate: false,
    perSeed: result.perSeed,
    trainedRobust: result.trainedRobust,
    category: result.perSeed[P_TRAINER_SEEDS[0]]
  };
};

export interface BuildArtifactInputs {
  readonly tasks: readonly BuildTaskInputs[];
  readonly interventionIndexText: string;
  readonly interventionIndexLabel: string;
  readonly interventionIndexBytes: Buffer;
  readonly graphIndexBytes: Buffer;
  readonly rewiringNullBytes: Buffer;
  readonly pathwayInterventionsBytes: Buffer;
  readonly pathwayInterventionsParsed: Readonly<PathwayInterventionsArtifact>;
}

const MIN_NON_DEGENERATE_TASKS = 3;

const computeOverall = (
  tasks: readonly TaskGeneralityTask[],
  nonDegenerate: (task: Readonly<TaskGeneralityTask>) => boolean,
  generalizes: (task: Readonly<TaskGeneralityTask>) => boolean
): TaskGeneralityOverall => {
  const eligible = tasks.filter(nonDegenerate);
  const verdict: 'general' | 'task-dependent' =
    eligible.length >= MIN_NON_DEGENERATE_TASKS && eligible.every(generalizes) ? 'general' : 'task-dependent';
  return { verdict, nonDegenerateCount: eligible.length, totalCount: tasks.length };
};

export const buildTaskGeneralityArtifact = (inputs: Readonly<BuildArtifactInputs>): TaskGeneralityArtifact => {
  const interventionIndexSha = sha256Hex(inputs.interventionIndexBytes);
  const interventionIndexInfo = parseGraphListIndexInfo(inputs.interventionIndexText, inputs.interventionIndexLabel);
  const tasks = inputs.tasks.map((task) => buildTask(task, interventionIndexSha, interventionIndexInfo));

  const authoredOverall = computeOverall(
    tasks,
    (t) => t.categorized,
    (t) => t.null.nullHolds && t.pathway.generalizes
  );
  const trainedOverall = computeOverall(
    tasks,
    (t) => !t.trained.degenerate,
    (t) => !t.trained.degenerate && t.trained.trainedRobust && t.trained.category === 'pathway-supported'
  );

  const pi = inputs.pathwayInterventionsParsed;
  const defaultAuthoredCategory = pi.authored.category;
  const defaultTrainedRepresentative = pi.trained.perSeed[P_TRAINER_SEEDS[0]];

  return {
    version: 1,
    sources: {
      rewiringNullSha: sha256Hex(inputs.rewiringNullBytes),
      pathwayInterventionsSha: sha256Hex(inputs.pathwayInterventionsBytes),
      graphIndexSha: sha256Hex(inputs.graphIndexBytes),
      interventionIndexSha,
      producer: taskGeneralityProducer()
    },
    tasks,
    overall: { authored: authoredOverall, trained: trainedOverall },
    defaultContext: {
      authoredCategory: defaultAuthoredCategory,
      trainedCategory: defaultTrainedRepresentative.category,
      trainedRobust: pi.trained.trainedRobust
    },
    host: { arch: process.arch, node: process.version }
  };
};

// ---------------------------------------------------------------------------
// Report markdown
// ---------------------------------------------------------------------------

const pct = (value: number): string => `${(value * 100).toFixed(1)}%`;
const fmt = (value: number): string => value.toFixed(2);

const TASK_TABLE = `| id | Changes | Intent |
| --- | --- | --- |
| \`hazard-heavy\` | \`hazardCount 4\`, \`hazardPenalty 6\` | Scoring dominated by avoidance |
| \`sparse-food\` | \`foodCount 1\`, \`halfWidth 18\`, \`halfDepth 12\` | Long-range search |
| \`no-movement\` | \`movementScorePerUnit 0\` | Score only from food and hazards, which removes the distance term thrust feeds |
| \`crowded\` | \`halfWidth 8\`, \`halfDepth 5.5\` | Walls are close and clearance signals dominate |`;

const PREDECLARED_OUTCOME_RULES = `- **Null holds:** biological is below the task's rewired-null 25th percentile (the authored null, 500 rewirings, seeds \`30001-30100\`, T 1800).
- **Pathway generalizes:** that task's P score passes the same "pathway-supported" rule as the original (at or above the task's null 25th percentile and above the 95th percentile of both C and M), using the existing P/C/M graphs. Those graphs were built from the task-independent transfer matrix, so they are reused unchanged.
- **Trained (predeclared taxonomy, made mechanical here):** per P trainer seed, with \`cMax\` and \`mMax\` the maxima of the 5 C and 5 M trained scores: \`pathway-supported\` if P > \`cMax\` and P > \`mMax\`; \`edge-class-effect\` if P > \`cMax\` and P <= \`mMax\`; \`no-specific-effect\` if P <= \`cMax\`. The published trained null is context only and never decides the category. \`trainedRobust\` holds only if all 3 seeds give the same category.
- **Degenerate guard:** a task's authored result is **not categorized** (reported as \`degenerate\`) if the null, the C arm, or the M arm has IQR below the existing \`DEGENERATE_IQR_THRESHOLD\`. The same guard applies to the trained side: if the 5-run trained C or M arm has IQR below the threshold (or all its scores are equal), that task's trained result is \`degenerate\` and gets no category.
- **Overall:** "general" if the null holds and the pathway generalizes in all non-degenerate tasks, with at least 3 of 4 tasks non-degenerate; "task-dependent" otherwise, listing per task. The authored and trained results are reported separately.`;

const renderTaskRow = (task: Readonly<TaskGeneralityTask>): string => {
  const changesText = Object.entries(task.changes)
    .map(([key, value]) => `${key}=${value}`)
    .join(', ');
  const nullLine = `Null: biological ${fmt(task.null.bioScore)} (${pct(task.null.bioPercentile)}) vs 25th percentile ${fmt(task.null.p25)} (median ${fmt(task.null.nullMedian)}) -- ${task.null.nullHolds ? 'holds' : 'does not hold'}${task.null.degenerate ? ' (null distribution degenerate)' : ''}.`;

  let pathwayLine: string;
  if (!task.categorized) {
    pathwayLine = `Pathway: **degenerate** (C arm degenerate: ${task.pathway.cDegenerate}, M arm degenerate: ${task.pathway.mDegenerate}) -- not categorized. P=${fmt(task.pathway.pScore)}, C p95=${fmt(task.pathway.cP95)}, M p95=${fmt(task.pathway.mP95)}.`;
  } else {
    pathwayLine = `Pathway: **${task.pathway.category}** (${task.pathway.generalizes ? 'generalizes' : 'does not generalize'}). P=${fmt(task.pathway.pScore)}, C p95=${fmt(task.pathway.cP95)}, M p95=${fmt(task.pathway.mP95)}.`;
  }

  let trainedLine: string;
  const trained = task.trained;
  if (trained.degenerate) {
    trainedLine = 'Trained: **degenerate** (5-run C or M arm has IQR below threshold) -- not categorized.';
  } else {
    // Narrowed into `trained` above (rather than re-narrowing `task.trained.degenerate`
    // at each use): TypeScript does not carry a nested-property discriminant
    // narrowing across the `.map()`/`.filter()` closures below.
    const perSeedText = P_TRAINER_SEEDS.map((seed) => `seed ${seed}: ${trained.perSeed[seed]}`).join(', ');
    const robustText = trained.trainedRobust ? 'robust' : 'not robust';
    const supportedSeeds = P_TRAINER_SEEDS.filter((seed) => trained.perSeed[seed] === 'pathway-supported');
    const marginNote =
      !trained.trainedRobust && supportedSeeds.length > 0
        ? ` (${supportedSeeds.length} of ${P_TRAINER_SEEDS.length} seeds reach pathway-supported -- this is a single-seed hit, not a robust finding)`
        : '';
    trainedLine = `Trained: **${trained.category}** (${robustText}${marginNote}). Per seed: ${perSeedText}.`;
  }

  return `### \`${task.id}\` (${changesText})\n\n- ${nullLine}\n- ${pathwayLine}\n- ${trainedLine}`;
};

const LIMITATIONS = `- This experiment covers this model only: config variants of the existing arena, no new physics, sensors, or reward code, and no claim about fly behavior.
- \`sensorRange\` is unchanged (24) in every task, so observation scaling is unchanged, but the measured clearance and \`foodDistance\` distributions differ per task -- reported values, not a formula. Wall clearance saturates near its reachable maximum in \`sparse-food\` and is compressed in \`crowded\`; \`foodDistance\` saturates at 1 (uninformative) far more often in \`sparse-food\`'s enlarged arena.
- \`no-movement\` severs the direct channel through which thrust earned score (distance x \`movementScorePerUnit\`, set to 0); any pathway result there reflects only thrust's indirect effect on food and hazard outcomes. Its authored pathway category is \`degenerate\` (the C control arm's IQR is 0, the predeclared guard), while its Q-vs-MQ channel-specific result is independently valid and is not conflated with the degenerate P/C/M category above.
- P/C/M/Q/MQ graphs were selected once on the default task's task-independent transfer matrix and reused unchanged across all four tasks -- this study never re-selects them per task.
- Degenerate tasks (authored or trained) are not categorized at all, and are excluded from the "at least 3 of 4 non-degenerate" overall verdict's eligible set.
- Trained results use only 5 freshly-trained controls per arm (a coarse resolution) and are reported robust only when all 3 P trainer seeds agree; a category reached at only 1 of 3 seeds is disclosed as a single-seed hit next to that category, never presented as if it were the robust finding, and is never treated as generalizing the default task's own separately-published trained result.
- \`no-specific-effect\` (trained side) is a reporting convention adopted after the trained scores were known, not a predeclared category (see \`docs/pathway-interventions-report.md\`).
- The null and pathway comparisons are repeated across 4 tasks x 2 decoders (8 combinations) without correction for multiple comparisons; each is reported and read on its own predeclared terms.
- Everything here is descriptive and bound to this model only; no causal claim is made about the real fly.`;

export const renderTaskGeneralityReportMarkdown = (artifact: Readonly<TaskGeneralityArtifact>): string => {
  const { tasks, overall, defaultContext } = artifact;
  const taskSections = tasks.map(renderTaskRow).join('\n\n');

  return `# Task generality of the null and pathway findings

## Question

\`docs/rewiring-null-report.md\` and \`docs/pathway-interventions-report.md\` (both on the default foraging task,
\`ARENA_CONFIG\`) report that biological scores at the bottom of the 500-rewiring null, and that the
clearance->thrust pathway intervention P is pathway-supported. This experiment tests whether those two findings
hold beyond the default task, using four predeclared \`ArenaConfig\` variants.

## Task variants

${TASK_TABLE}

Each variant changes only the listed \`ArenaConfig\` fields; every other field (including \`sensorRange\`) is
unchanged from \`ARENA_CONFIG\`, and each passes \`retainArenaConfig\`'s validation, including the disk-packing
capacity check.

## Predeclared outcome rules

${PREDECLARED_OUTCOME_RULES}

## Per-task results

${taskSections}

## Overall

- **Authored:** **${overall.authored.verdict}** (${overall.authored.nonDegenerateCount} of ${overall.authored.totalCount} tasks non-degenerate).
- **Trained:** **${overall.trained.verdict}** (${overall.trained.nonDegenerateCount} of ${overall.trained.totalCount} tasks non-degenerate).

The default task's own separately-published results are context only, not a fifth study task: authored category
**${defaultContext.authoredCategory}**; trained category **${defaultContext.trainedCategory}** (${defaultContext.trainedRobust ? 'robust' : 'not robust'} across trainer seeds). A single-seed pathway-supported hit on a task variant above is never framed as generalizing this default-task result.

## Limitations

${LIMITATIONS}
`;
};

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/** Add/overwrite the manifest's `taskGenerality` key in place -- mirrors `intervention-artifact.ts`'s `updateManifestWithPathwayInterventions`. */
export const updateManifestWithTaskGenerality = (
  manifestPath: string,
  entry: { readonly artifact: string; readonly sha256: string }
): void => {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
  manifest.taskGenerality = entry;
  atomicWriteFileSync(manifestPath, `${JSON.stringify(sortKeysDeep(manifest), null, 2)}\n`);
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface TaskGeneralityReportArgs {
  readonly authoredDir: string;
  readonly trainedDir: string;
  readonly graphIndex: string;
  readonly interventionIndex: string;
  readonly rewiringNull: string;
  readonly pathwayInterventions: string;
  readonly manifest: string;
  readonly out: string;
  readonly reportMd: string;
}

export const parseTaskGeneralityReportArgs = (argv: readonly string[]): TaskGeneralityReportArgs => {
  let authoredDir = DEFAULT_TASKS_DIR;
  let trainedDir = DEFAULT_TASKS_DIR;
  let graphIndex = DEFAULT_GRAPH_INDEX;
  let interventionIndex = DEFAULT_INTERVENTION_INDEX;
  let rewiringNull = DEFAULT_REWIRING_NULL;
  let pathwayInterventions = DEFAULT_PATHWAY_INTERVENTIONS;
  let manifest = DEFAULT_MANIFEST;
  let out = DEFAULT_OUT;
  let reportMd = DEFAULT_REPORT_MD;

  let i = 0;
  while (i < argv.length) {
    const flag = argv[i];
    if (flag === '--authored-dir') {
      authoredDir = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--trained-dir') {
      trainedDir = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--graph-index') {
      graphIndex = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--intervention-index') {
      interventionIndex = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--rewiring-null') {
      rewiringNull = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--pathway-interventions') {
      pathwayInterventions = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--manifest') {
      manifest = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--out') {
      out = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--report-md') {
      reportMd = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (resolve(out) === resolve(reportMd)) {
    throw new Error('task-generality-report: --out and --report-md must not be the same path');
  }
  return { authoredDir, trainedDir, graphIndex, interventionIndex, rewiringNull, pathwayInterventions, manifest, out, reportMd };
};

export interface RunTaskGeneralityReportResult {
  readonly out: string;
  readonly reportMdPath: string;
  readonly artifactSha256: string;
  readonly artifact: TaskGeneralityArtifact;
}

export const runTaskGeneralityReport = (args: Readonly<TaskGeneralityReportArgs>): RunTaskGeneralityReportResult => {
  const interventionIndexBytes = readFileSync(args.interventionIndex);
  const graphIndexBytes = readFileSync(args.graphIndex);
  const rewiringNullBytes = readFileSync(args.rewiringNull);
  const pathwayInterventionsBytes = readFileSync(args.pathwayInterventions);
  const pathwayInterventionsParsed = JSON.parse(pathwayInterventionsBytes.toString('utf8')) as PathwayInterventionsArtifact;

  const tasks: BuildTaskInputs[] = STUDY_TASK_IDS.map((id) => {
    const paths = taskInputPaths(args.authoredDir, args.trainedDir, id);
    return {
      id,
      nullSummaryText: readFileSync(paths.nullSummary, 'utf8'),
      nullSummaryLabel: paths.nullSummary,
      interventionStatsText: readFileSync(paths.interventionStats, 'utf8'),
      interventionStatsLabel: paths.interventionStats,
      trainedText: readFileSync(paths.trained, 'utf8'),
      trainedLabel: paths.trained
    };
  });

  const artifact = buildTaskGeneralityArtifact({
    tasks,
    interventionIndexText: interventionIndexBytes.toString('utf8'),
    interventionIndexLabel: args.interventionIndex,
    interventionIndexBytes,
    graphIndexBytes,
    rewiringNullBytes,
    pathwayInterventionsBytes,
    pathwayInterventionsParsed
  });

  verifyManifestRoundTrips(args.manifest);

  const artifactContents = JSON.stringify(artifact);
  const artifactSha256 = sha256Hex(artifactContents);
  const reportMdContents = renderTaskGeneralityReportMarkdown(artifact);

  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, artifactContents);

  updateManifestWithTaskGenerality(args.manifest, { artifact: basename(args.out), sha256: artifactSha256 });

  mkdirSync(dirname(args.reportMd), { recursive: true });
  atomicWriteFileSync(args.reportMd, reportMdContents);

  return { out: args.out, reportMdPath: args.reportMd, artifactSha256, artifact };
};

const main = (): void => {
  try {
    const args = parseTaskGeneralityReportArgs(process.argv.slice(2));
    const result = runTaskGeneralityReport(args);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `task-generality-report: wrote ${result.out} (sha256 ${result.artifactSha256}) and ${result.reportMdPath}\n` +
        `overall.authored=${result.artifact.overall.authored.verdict} overall.trained=${result.artifact.overall.trained.verdict}`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`task-generality-report failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
