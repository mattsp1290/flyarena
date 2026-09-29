import { P_TRAINER_SEEDS } from '../../experiment/pathwayInterventions';
import type { TaskGeneralityTask, TaskGeneralityTrainedCategory } from '../../experiment/taskGenerality';
import { provenanceFor, reasonFor, sidecarStepStatus, type BuildFindingStepsInputs, type FindingStep } from './shared';

// ---------------------------------------------------------------------------
// Task generality
// ---------------------------------------------------------------------------

/**
 * WP4 of `.agents/plans/task-generality`: whether the rewiring-null and
 * pathway-intervention findings above hold beyond the default foraging
 * task, across four predeclared `ArenaConfig` variants. Inserted after
 * "Trained interventions" and before "Behavior repertoire" in
 * `buildFindingSteps`'s own array order (`04-artifact-and-findings.md`'s
 * own placement) -- that array's order no longer determines the panel's
 * *display* order, which `../sections.ts#groupSteps` computes separately
 * (WP1 of `.agents/plans/consolidated-release`); see that module's own doc
 * comment for the real rendered grouping. `condition: 'both'`: the
 * sentence states both the authored and trained overall verdicts together,
 * mirroring the "Trained interventions" step's own "both decoders in one
 * sentence" shape. Reuses the same `P_TRAINER_SEEDS` -- the trained
 * decoder's fixed trainer-seed set is study-wide, not per-artifact.
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
 * `describeTrainedCategory` (`../format.ts`), which always expands
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

export const buildTaskGeneralityStep = (inputs: BuildFindingStepsInputs): FindingStep => {
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
  // breakdown moves to `details`, rendered as its own `<ul>/<li>` list by
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
  const details = tasks.map((task) => ({
    id: task.id,
    text: `authored: ${authoredTaskClause(task)}; trained: ${trainedTaskClauseShort(task)}`
  }));

  return { ...base, status: 'ok', sentence, details };
};
