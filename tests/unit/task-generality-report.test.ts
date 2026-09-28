// @vitest-environment node
import { describe, expect, it } from 'vitest';

import {
  buildTaskGeneralityArtifact,
  diffArenaConfig,
  renderTaskGeneralityReportMarkdown,
  STUDY_TASK_IDS,
  type BuildArtifactInputs,
  type BuildTaskInputs,
  type TaskGeneralityArtifact
} from '../../scripts/null/task-generality-report';
import { P_TRAINER_SEEDS, TRAINED_ARM_SIZE, CONTROL_TRAINER_SEED } from '../../scripts/null/intervention-report-trained';
import { sha256Hex } from '../../scripts/training/fsio';
import { resolveArenaTask } from '../../src/lib/arena/tasks';
import type { PathwayInterventionsArtifact } from '../../scripts/null/intervention-artifact';

/**
 * WP4 of `.agents/plans/task-generality`: coverage for
 * `scripts/null/task-generality-report.ts`'s category/verdict logic.
 * `buildTaskGeneralityArtifact` is exercised directly against synthetic
 * in-memory inputs (no filesystem), mirroring
 * `tests/unit/intervention-artifact.test.ts`'s own "pure builder" style.
 * Every per-task fixture is internally consistent (matching arena-task
 * fingerprints, matching cross-referenced shas, matching gzipSha256 between
 * the intervention index and each trained run) so a test failure means the
 * category/verdict logic itself is wrong, not a fixture mismatch.
 */

const SHA = (label: string): string => sha256Hex(label);

// ---------------------------------------------------------------------------
// Shared "world": one intervention index (P + 5 C + 5 M), reused by every
// task fixture below, since the real study reuses the same P/C/M graphs
// unchanged across all four tasks.
// ---------------------------------------------------------------------------

const C_IDS = Array.from({ length: TRAINED_ARM_SIZE }, (_, i) => `C00${i}`);
const M_IDS = Array.from({ length: TRAINED_ARM_SIZE }, (_, i) => `M100${i}`);

const interventionIndexObject = {
  controlCount: 100,
  entries: [
    { id: 'P', kind: 'P', gzipSha256: SHA('P') },
    ...C_IDS.map((id) => ({ id, kind: 'C', gzipSha256: SHA(id) })),
    ...M_IDS.map((id) => ({ id, kind: 'M', gzipSha256: SHA(id) }))
  ]
};
const interventionIndexText = JSON.stringify(interventionIndexObject);
const interventionIndexBytes = Buffer.from(interventionIndexText, 'utf8');
const interventionIndexSha = sha256Hex(interventionIndexBytes);

const graphIndexBytes = Buffer.from('graph-index-fixture', 'utf8');
const rewiringNullBytes = Buffer.from('rewiring-null-fixture', 'utf8');
const pathwayInterventionsBytes = Buffer.from('pathway-interventions-fixture', 'utf8');

const pathwayInterventionsParsed = {
  authored: { category: 'pathway-supported' },
  trained: {
    trainedRobust: true,
    perSeed: { 101: { category: 'no-specific-effect' }, 202: { category: 'no-specific-effect' }, 303: { category: 'no-specific-effect' } }
  }
} as unknown as PathwayInterventionsArtifact;

interface TaskFixtureOptions {
  readonly id: (typeof STUDY_TASK_IDS)[number];
  readonly bioScore: number;
  readonly rewiredScores: readonly number[];
  readonly nullDegenerate?: boolean;
  readonly pathwayCategory: string;
  readonly cArmDegenerate?: boolean;
  readonly mArmDegenerate?: boolean;
  readonly pScoreAuthored: number;
  readonly cP95: number;
  readonly mP95: number;
  /** Mean `movementScore` per P trainer seed. */
  readonly pTrainedScores: Readonly<Record<(typeof P_TRAINER_SEEDS)[number], number>>;
  readonly cTrainedScores: readonly number[];
  readonly mTrainedScores: readonly number[];
}

/** Builds one task's `BuildTaskInputs`, with every cross-referenced field consistent by construction. */
const taskFixture = (options: TaskFixtureOptions): BuildTaskInputs => {
  const fingerprint = resolveArenaTask(options.id).fingerprint;

  const nullSummary = {
    arenaTask: { id: options.id, fingerprint },
    biological: { score: options.bioScore },
    null: { median: (options.rewiredScores[Math.floor(options.rewiredScores.length / 2)] as number) ?? 0, degenerate: options.nullDegenerate ?? false },
    bioPercentile: 0.01,
    rewired: options.rewiredScores.map((score, i) => ({ seed: i, score }))
  };
  const nullSummaryText = JSON.stringify(nullSummary);
  const publishedNullSha = sha256Hex(Buffer.from(nullSummaryText, 'utf8'));

  const interventionStats = {
    arenaTask: { id: options.id, fingerprint },
    statsOnly: true,
    inputs: { indexSha256: interventionIndexSha, publishedNullSha256: publishedNullSha },
    armDegeneracy: {
      nullArm: { iqr: 1, degenerate: false },
      cArm: { iqr: options.cArmDegenerate ? 0 : 1, degenerate: options.cArmDegenerate ?? false },
      mArm: { iqr: options.mArmDegenerate ? 0 : 1, degenerate: options.mArmDegenerate ?? false },
      mqArm: { iqr: 1, degenerate: false },
      categoryDegenerate: (options.cArmDegenerate ?? false) || (options.mArmDegenerate ?? false),
      channelSpecificDegenerate: false
    },
    p: { score: options.pScoreAuthored, category: options.pathwayCategory },
    controls: { C: { p95: options.cP95 }, M: { p95: options.mP95 } }
  };
  const interventionStatsText = JSON.stringify(interventionStats);

  const runs = [
    ...P_TRAINER_SEEDS.map((seed) => ({
      id: 'P',
      trainerSeed: seed,
      gzipSha256: SHA('P'),
      movementScore: [options.pTrainedScores[seed]]
    })),
    ...C_IDS.map((id, i) => ({ id, trainerSeed: CONTROL_TRAINER_SEED, gzipSha256: SHA(id), movementScore: [options.cTrainedScores[i]] })),
    ...M_IDS.map((id, i) => ({ id, trainerSeed: CONTROL_TRAINER_SEED, gzipSha256: SHA(id), movementScore: [options.mTrainedScores[i]] }))
  ];
  const trained = {
    arenaTask: options.id,
    arenaTaskFingerprint: fingerprint,
    graphListSha256: interventionIndexSha,
    runs
  };
  const trainedText = JSON.stringify(trained);

  return {
    id: options.id,
    nullSummaryText,
    nullSummaryLabel: `${options.id}/null-summary.json`,
    interventionStatsText,
    interventionStatsLabel: `${options.id}/intervention-stats.json`,
    trainedText,
    trainedLabel: `${options.id}/trained.json`
  };
};

/** A "clearly holds/generalizes/robustly-pathway-supported" task, for the all-general scenario. */
const generalTaskOptions = (id: (typeof STUDY_TASK_IDS)[number]): TaskFixtureOptions => ({
  id,
  bioScore: -10,
  rewiredScores: Array.from({ length: 20 }, (_, i) => i - 5), // p25 well above -10
  pathwayCategory: 'pathway-supported',
  pScoreAuthored: 100,
  cP95: 10,
  mP95: 20,
  pTrainedScores: { 101: 100, 202: 100, 303: 100 },
  cTrainedScores: [10, 11, 12, 13, 14],
  mTrainedScores: [20, 21, 22, 23, 24]
});

const buildArtifact = (tasks: readonly BuildTaskInputs[]): TaskGeneralityArtifact =>
  buildTaskGeneralityArtifact({
    tasks,
    interventionIndexText,
    interventionIndexLabel: 'interventions/index.json',
    interventionIndexBytes,
    graphIndexBytes,
    rewiringNullBytes,
    pathwayInterventionsBytes,
    pathwayInterventionsParsed
  } satisfies BuildArtifactInputs);

describe('diffArenaConfig', () => {
  it('reports only the changed fields for each predeclared task, matching src/lib/arena/tasks.ts', () => {
    for (const id of STUDY_TASK_IDS) {
      const diff = diffArenaConfig(resolveArenaTask(id).config);
      expect(Object.keys(diff).length).toBeGreaterThan(0);
      // Every declared change actually differs from ARENA_CONFIG.
      const config = resolveArenaTask(id).config as unknown as Record<string, number>;
      for (const key of Object.keys(diff)) {
        expect(config[key]).toBe(diff[key]);
      }
    }
    // `default` (not a study task, but the same function) has no changes.
    expect(diffArenaConfig(resolveArenaTask('default').config)).toEqual({});
  });
});

describe('buildTaskGeneralityArtifact: overall verdicts', () => {
  it('is "general" for both authored and trained when all four tasks hold/generalize/reach a robust pathway-supported result', () => {
    const tasks = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    const artifact = buildArtifact(tasks);
    expect(artifact.overall.authored).toEqual({ verdict: 'general', nonDegenerateCount: 4, totalCount: 4 });
    expect(artifact.overall.trained).toEqual({ verdict: 'general', nonDegenerateCount: 4, totalCount: 4 });
    for (const task of artifact.tasks) {
      expect(task.categorized).toBe(true);
      expect(task.pathway.generalizes).toBe(true);
      expect(task.trained).toMatchObject({ degenerate: false, trainedRobust: true, category: 'pathway-supported' });
    }
  });

  it('is "task-dependent" when one task\'s pathway does not generalize (a "not-supported" task failing to reproduce the finding)', () => {
    const [firstId, ...restIds] = STUDY_TASK_IDS;
    const failing = taskFixture({
      ...generalTaskOptions(firstId),
      pathwayCategory: 'not-supported',
      pScoreAuthored: -20, // below the null floor -- not-supported
      pTrainedScores: { 101: 1, 202: 1, 303: 1 },
      cTrainedScores: [10, 11, 12, 13, 14],
      mTrainedScores: [20, 21, 22, 23, 24]
    });
    const tasks = [failing, ...restIds.map((id) => taskFixture(generalTaskOptions(id)))];
    const artifact = buildArtifact(tasks);
    expect(artifact.overall.authored.verdict).toBe('task-dependent');
    expect(artifact.overall.authored).toMatchObject({ nonDegenerateCount: 4, totalCount: 4 });
    expect(artifact.overall.trained.verdict).toBe('task-dependent');
    const failingTask = artifact.tasks.find((t) => t.id === firstId);
    expect(failingTask?.pathway.category).toBe('not-supported');
    expect(failingTask?.pathway.generalizes).toBe(false);
    expect(failingTask?.trained.degenerate).toBe(false);
    if (!failingTask || failingTask.trained.degenerate) throw new Error('expected a categorized trained result');
    expect(failingTask.trained.category).not.toBe('pathway-supported');
  });

  it('excludes a degenerate task from the eligible set, and still reports "general" when the remaining 3 of 4 all hold/generalize', () => {
    const [degenerateId, ...restIds] = STUDY_TASK_IDS;
    const degenerate = taskFixture({ ...generalTaskOptions(degenerateId), cArmDegenerate: true, pathwayCategory: 'degenerate' });
    const tasks = [degenerate, ...restIds.map((id) => taskFixture(generalTaskOptions(id)))];
    const artifact = buildArtifact(tasks);
    const degenerateTask = artifact.tasks.find((t) => t.id === degenerateId);
    expect(degenerateTask?.categorized).toBe(false);
    expect(degenerateTask?.pathway.cDegenerate).toBe(true);
    // Authored: 3 of 4 non-degenerate, all hold/generalize -> general.
    expect(artifact.overall.authored).toEqual({ verdict: 'general', nonDegenerateCount: 3, totalCount: 4 });
  });

  it('is "task-dependent" once fewer than 3 of 4 tasks are non-degenerate, even if every eligible one holds/generalizes', () => {
    const [firstId, secondId, ...restIds] = STUDY_TASK_IDS;
    const degenerateA = taskFixture({ ...generalTaskOptions(firstId), cArmDegenerate: true, pathwayCategory: 'degenerate' });
    const degenerateB = taskFixture({ ...generalTaskOptions(secondId), mArmDegenerate: true, pathwayCategory: 'degenerate' });
    const tasks = [degenerateA, degenerateB, ...restIds.map((id) => taskFixture(generalTaskOptions(id)))];
    const artifact = buildArtifact(tasks);
    expect(artifact.overall.authored).toEqual({ verdict: 'task-dependent', nonDegenerateCount: 2, totalCount: 4 });
  });

  it('trained: a category reached at only some seeds (not robust) never counts toward "general", even when it is pathway-supported', () => {
    const tasks = STUDY_TASK_IDS.map((id, i) =>
      i === 0
        ? taskFixture({ ...generalTaskOptions(id), pTrainedScores: { 101: 100, 202: 1, 303: 1 } }) // pathway-supported at 1/3 seeds only
        : taskFixture(generalTaskOptions(id))
    );
    const artifact = buildArtifact(tasks);
    const notRobustTask = artifact.tasks[0];
    if (notRobustTask.trained.degenerate) throw new Error('expected a categorized trained result');
    expect(notRobustTask.trained.trainedRobust).toBe(false);
    expect(notRobustTask.trained.perSeed[101]).toBe('pathway-supported');
    expect(notRobustTask.trained.perSeed[202]).toBe('no-specific-effect');
    // The representative category is seed 101's own (per-seed, not "robust").
    expect(notRobustTask.trained.category).toBe('pathway-supported');
    expect(artifact.overall.trained.verdict).toBe('task-dependent');
  });

  it('trained: a degenerate 5-run C or M arm marks that task degenerate and excludes it from the trained eligible set', () => {
    const tasks = STUDY_TASK_IDS.map((id, i) =>
      i === 0 ? taskFixture({ ...generalTaskOptions(id), mTrainedScores: [5, 5, 5, 5, 5] }) : taskFixture(generalTaskOptions(id))
    );
    const artifact = buildArtifact(tasks);
    expect(artifact.tasks[0].trained).toEqual({ degenerate: true });
    expect(artifact.overall.trained).toEqual({ verdict: 'general', nonDegenerateCount: 3, totalCount: 4 });
  });
});

describe('buildTaskGeneralityArtifact: cross-checks', () => {
  it('throws when a null-summary is not recorded under the requested arena task', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const wrongTask = JSON.parse(bad.nullSummaryText) as { arenaTask: { id: string; fingerprint: string } };
    wrongTask.arenaTask.id = 'crowded';
    const tasks = STUDY_TASK_IDS.map((id) => (id === 'hazard-heavy' ? { ...bad, nullSummaryText: JSON.stringify(wrongTask) } : taskFixture(generalTaskOptions(id))));
    expect(() => buildArtifact(tasks)).toThrow(/not recorded under arena task/);
  });

  it('throws when the intervention-stats published-null sha does not match the paired null-summary bytes', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const stats = JSON.parse(bad.interventionStatsText) as { inputs: { publishedNullSha256: string } };
    stats.inputs.publishedNullSha256 = SHA('stale');
    const tasks = STUDY_TASK_IDS.map((id) =>
      id === 'hazard-heavy' ? { ...bad, interventionStatsText: JSON.stringify(stats) } : taskFixture(generalTaskOptions(id))
    );
    expect(() => buildArtifact(tasks)).toThrow(/scored against a different null/);
  });

  it('throws when a trained.json graphListSha256 does not match the shared intervention index sha', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const trained = JSON.parse(bad.trainedText) as { graphListSha256: string };
    trained.graphListSha256 = SHA('stale-index');
    const tasks = STUDY_TASK_IDS.map((id) => (id === 'hazard-heavy' ? { ...bad, trainedText: JSON.stringify(trained) } : taskFixture(generalTaskOptions(id))));
    expect(() => buildArtifact(tasks)).toThrow(/does not match the shared intervention index sha/);
  });
});

describe('buildTaskGeneralityArtifact: determinism', () => {
  it('produces byte-identical JSON across two calls with the same inputs', () => {
    const tasks = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    const a = JSON.stringify(buildArtifact(tasks));
    const b = JSON.stringify(buildArtifact(tasks));
    expect(a).toBe(b);
  });
});

describe('renderTaskGeneralityReportMarkdown', () => {
  it('quotes the predeclared task table and outcome rules verbatim, and states every task\'s per-line result', () => {
    const tasks = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    const artifact = buildArtifact(tasks);
    const markdown = renderTaskGeneralityReportMarkdown(artifact);
    expect(markdown).toContain('hazardCount 4');
    expect(markdown).toContain('Null holds:');
    expect(markdown).toContain('Pathway generalizes:');
    expect(markdown).toContain('Degenerate guard:');
    for (const id of STUDY_TASK_IDS) {
      expect(markdown).toContain(`### \`${id}\``);
    }
    expect(markdown).toContain('## Limitations');
    expect(markdown).toContain('no causal claim is made about the real fly');
  });

  it('states "degenerate" and "not robust" plainly next to a task\'s own result, never only in a footnote', () => {
    const [degenerateId, ...restIds] = STUDY_TASK_IDS;
    const degenerate = taskFixture({ ...generalTaskOptions(degenerateId), cArmDegenerate: true, pathwayCategory: 'degenerate' });
    const notRobustId = restIds[0];
    const notRobust = taskFixture({ ...generalTaskOptions(notRobustId), pTrainedScores: { 101: 100, 202: 1, 303: 1 } });
    const tasks = [degenerate, notRobust, ...restIds.slice(1).map((id) => taskFixture(generalTaskOptions(id)))];
    const markdown = renderTaskGeneralityReportMarkdown(buildArtifact(tasks));
    expect(markdown).toContain('Pathway: **degenerate**');
    expect(markdown).toContain('not robust');
    expect(markdown).toMatch(/single-seed hit, not a robust finding/);
  });
});
