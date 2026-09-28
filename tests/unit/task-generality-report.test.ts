// @vitest-environment node
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  buildTaskGeneralityArtifact,
  diffArenaConfig,
  parseTaskGeneralityReportArgs,
  renderTaskGeneralityReportMarkdown,
  runTaskGeneralityReport,
  taskInputPaths,
  STUDY_TASK_IDS,
  type BuildArtifactInputs,
  type BuildTaskInputs,
  type TaskGeneralityArtifact
} from '../../scripts/null/task-generality-report';
import { P_TRAINER_SEEDS, TRAINED_ARM_SIZE, CONTROL_TRAINER_SEED } from '../../scripts/null/intervention-report-trained';
import { sortKeysDeep } from '../../scripts/null/null-report';
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
const manifestBiologicalSha = SHA('biological-graph');
const manifestGzipSha = SHA('gzip-graph');
const manifestRewiringNullSha = sha256Hex(rewiringNullBytes);
const manifestPathwayInterventionsSha = sha256Hex(pathwayInterventionsBytes);

const pathwayInterventionsParsed = {
  version: 1,
  sources: { biologicalSha: manifestBiologicalSha, rewiringNullSha: manifestRewiringNullSha, indexSha: interventionIndexSha },
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
  readonly qScore?: number;
  readonly qChannelSpecific?: boolean | 'degenerate';
  /** Mean `movementScore` per P trainer seed. */
  readonly pTrainedScores: Readonly<Record<(typeof P_TRAINER_SEEDS)[number], number>>;
  readonly cTrainedScores: readonly number[];
  readonly mTrainedScores: readonly number[];
  /** `foodDistance`'s own `fractionSaturated`, default 0 (never saturates) -- overridden by the perturbation test below to prove the rendered limitation text tracks this value. */
  readonly foodDistanceFractionSaturated?: number;
  /** `forwardClearance`'s own `fractionSaturated`, default 0. */
  readonly forwardClearanceFractionSaturated?: number;
}

const CLEARANCE_CHANNEL_NAMES = ['foodDistance', 'forwardClearance', 'leftClearance', 'rightClearance'] as const;

/** One channel's `ChannelPercentiles`-shaped fixture -- `fractionSaturated` is the only field any test perturbs; the rest are plausible fixed values. */
const channelFixture = (fractionSaturated: number) => ({
  p5: 0.1,
  p50: 0.2,
  p95: 0.3,
  max: fractionSaturated > 0 ? 1 : 0.35,
  n: 18000,
  fractionSaturated
});

/** Builds one task's `BuildTaskInputs`, with every cross-referenced field consistent by construction. */
const taskFixture = (options: TaskFixtureOptions): BuildTaskInputs => {
  const fingerprint = resolveArenaTask(options.id).fingerprint;

  const nullSummary = {
    arenaTask: { id: options.id, fingerprint },
    sourceGraphSha256: manifestBiologicalSha,
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
    biologicalReproduction: { computedScore: options.bioScore, publishedScore: options.bioScore, matches: true },
    inputs: { indexSha256: interventionIndexSha, publishedNullSha256: publishedNullSha },
    armDegeneracy: {
      nullArm: { iqr: options.nullDegenerate ? 0 : 1, degenerate: options.nullDegenerate ?? false },
      cArm: { iqr: options.cArmDegenerate ? 0 : 1, degenerate: options.cArmDegenerate ?? false },
      mArm: { iqr: options.mArmDegenerate ? 0 : 1, degenerate: options.mArmDegenerate ?? false },
      mqArm: { iqr: 1, degenerate: false },
      categoryDegenerate: (options.nullDegenerate ?? false) || (options.cArmDegenerate ?? false) || (options.mArmDegenerate ?? false),
      channelSpecificDegenerate: false
    },
    p: { score: options.pScoreAuthored, category: options.pathwayCategory },
    q: { score: options.qScore ?? 1, channelSpecific: options.qChannelSpecific ?? true },
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

  const clearance = {
    version: 1,
    arenaTask: { id: options.id, fingerprint },
    sensorRange: 24,
    graph: { path: 'public/data/malecns-arena-v1.bin.gz', sha256: manifestGzipSha },
    seeds: { start: 1, count: 10 },
    ticks: 1800,
    substeps: 4,
    channels: Object.fromEntries(
      CLEARANCE_CHANNEL_NAMES.map((channel) => [
        channel,
        channelFixture(
          channel === 'foodDistance'
            ? (options.foodDistanceFractionSaturated ?? 0)
            : channel === 'forwardClearance'
              ? (options.forwardClearanceFractionSaturated ?? 0)
              : 0
        )
      ])
    )
  };
  const clearanceText = JSON.stringify(clearance);

  return {
    id: options.id,
    nullSummaryText,
    nullSummaryLabel: `${options.id}/null-summary.json`,
    interventionStatsText,
    interventionStatsLabel: `${options.id}/intervention-stats.json`,
    trainedText,
    trainedLabel: `${options.id}/trained.json`,
    clearanceText,
    clearanceLabel: `${options.id}/clearance.json`
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
    pathwayInterventionsParsed,
    manifestBiologicalSha,
    manifestGzipSha,
    manifestRewiringNullSha,
    manifestPathwayInterventionsSha
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

  it('is "task-dependent" (authored) when one task\'s null does not hold, even though its pathway generalizes', () => {
    const [firstId, ...restIds] = STUDY_TASK_IDS;
    // bioScore above every rewired score (well above p25) -- "does not hold".
    const failing = taskFixture({ ...generalTaskOptions(firstId), bioScore: 100 });
    const tasks = [failing, ...restIds.map((id) => taskFixture(generalTaskOptions(id)))];
    const artifact = buildArtifact(tasks);
    const failingTask = artifact.tasks.find((t) => t.id === firstId);
    expect(failingTask?.null.nullHolds).toBe(false);
    expect(failingTask?.pathway.generalizes).toBe(true);
    expect(artifact.overall.authored).toEqual({ verdict: 'task-dependent', nonDegenerateCount: 4, totalCount: 4 });
  });

  it('nullHolds is false exactly at the p25 boundary (the rule is strict "<", not "<=")', () => {
    const options = generalTaskOptions(STUDY_TASK_IDS[0]);
    // p25 of the fixture's rewiredScores (20 values, i-5 for i in 0..19) is
    // the 5th sorted value (quantileIndex(20, 0.25) = 5): value 0.
    const bioScore = 0;
    const boundary = taskFixture({ ...options, bioScore });
    const artifact = buildArtifact([boundary, ...STUDY_TASK_IDS.slice(1).map((id) => taskFixture(generalTaskOptions(id)))]);
    const task = artifact.tasks[0];
    expect(task.null.p25).toBe(bioScore);
    expect(task.null.nullHolds).toBe(false);
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

  it('throws when clearance.json is not recorded under the requested arena task', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const clearance = JSON.parse(bad.clearanceText) as { arenaTask: { id: string; fingerprint: string } };
    clearance.arenaTask.id = 'crowded';
    const tasks = STUDY_TASK_IDS.map((id) => (id === 'hazard-heavy' ? { ...bad, clearanceText: JSON.stringify(clearance) } : taskFixture(generalTaskOptions(id))));
    expect(() => buildArtifact(tasks)).toThrow(/clearance\.json is not recorded under arena task/);
  });

  it('throws when clearance.json\'s graph.sha256 does not match the manifest\'s gzipSha256', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const clearance = JSON.parse(bad.clearanceText) as { graph: { sha256: string } };
    clearance.graph.sha256 = SHA('a-different-gzip');
    const tasks = STUDY_TASK_IDS.map((id) => (id === 'hazard-heavy' ? { ...bad, clearanceText: JSON.stringify(clearance) } : taskFixture(generalTaskOptions(id))));
    expect(() => buildArtifact(tasks)).toThrow(/does not match the manifest's gzipSha256/);
  });

  it('throws when clearance.json is missing a required channel', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const clearance = JSON.parse(bad.clearanceText) as { channels: Record<string, unknown> };
    delete clearance.channels.foodDistance;
    const tasks = STUDY_TASK_IDS.map((id) => (id === 'hazard-heavy' ? { ...bad, clearanceText: JSON.stringify(clearance) } : taskFixture(generalTaskOptions(id))));
    expect(() => buildArtifact(tasks)).toThrow(/is missing channel "foodDistance"/);
  });

  it('records each task\'s clearance.json sha256 in sources.clearanceShas', () => {
    const tasks = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    const artifact = buildArtifact(tasks);
    for (const task of tasks) {
      expect(artifact.sources.clearanceShas[task.id]).toBe(sha256Hex(Buffer.from(task.clearanceText, 'utf8')));
    }
  });

  it('throws when intervention-stats is diagnosticOnly', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const stats = JSON.parse(bad.interventionStatsText) as Record<string, unknown>;
    stats.diagnosticOnly = true;
    const tasks = STUDY_TASK_IDS.map((id) => (id === 'hazard-heavy' ? { ...bad, interventionStatsText: JSON.stringify(stats) } : taskFixture(generalTaskOptions(id))));
    expect(() => buildArtifact(tasks)).toThrow(/diagnosticOnly, or its biologicalReproduction\.matches is not true/);
  });

  it('throws when intervention-stats biologicalReproduction.matches is false', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const stats = JSON.parse(bad.interventionStatsText) as { biologicalReproduction: { matches: boolean } };
    stats.biologicalReproduction.matches = false;
    const tasks = STUDY_TASK_IDS.map((id) => (id === 'hazard-heavy' ? { ...bad, interventionStatsText: JSON.stringify(stats) } : taskFixture(generalTaskOptions(id))));
    expect(() => buildArtifact(tasks)).toThrow(/diagnosticOnly, or its biologicalReproduction\.matches is not true/);
  });

  it('throws when intervention-stats is not a --stats-only result', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const stats = JSON.parse(bad.interventionStatsText) as Record<string, unknown>;
    delete stats.statsOnly;
    const tasks = STUDY_TASK_IDS.map((id) => (id === 'hazard-heavy' ? { ...bad, interventionStatsText: JSON.stringify(stats) } : taskFixture(generalTaskOptions(id))));
    expect(() => buildArtifact(tasks)).toThrow(/is not a --stats-only result/);
  });

  it('throws when intervention-stats inputs.indexSha256 is stale', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const stats = JSON.parse(bad.interventionStatsText) as { inputs: { indexSha256: string } };
    stats.inputs.indexSha256 = SHA('stale');
    const tasks = STUDY_TASK_IDS.map((id) => (id === 'hazard-heavy' ? { ...bad, interventionStatsText: JSON.stringify(stats) } : taskFixture(generalTaskOptions(id))));
    expect(() => buildArtifact(tasks)).toThrow(/does not match the shared intervention index sha/);
  });

  it('throws when trained.json is recorded under a different arena task', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const trained = JSON.parse(bad.trainedText) as { arenaTask: string };
    trained.arenaTask = 'crowded';
    const tasks = STUDY_TASK_IDS.map((id) => (id === 'hazard-heavy' ? { ...bad, trainedText: JSON.stringify(trained) } : taskFixture(generalTaskOptions(id))));
    expect(() => buildArtifact(tasks)).toThrow(/is not recorded under arena task/);
  });

  it('throws when trained.json is missing a P run at one of the three trainer seeds', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const trained = JSON.parse(bad.trainedText) as { runs: readonly { readonly id: string; readonly trainerSeed: number }[] };
    const runs = trained.runs.filter((run) => !(run.id === 'P' && run.trainerSeed === 303));
    const tasks = STUDY_TASK_IDS.map((id) =>
      id === 'hazard-heavy' ? { ...bad, trainedText: JSON.stringify({ ...trained, runs }) } : taskFixture(generalTaskOptions(id))
    );
    expect(() => buildArtifact(tasks)).toThrow(/missing the P run at trainer seed 303/);
  });

  it('throws when a per-task null-summary was scored against a different biological graph than the manifest', () => {
    const bad = taskFixture(generalTaskOptions('hazard-heavy'));
    const nullSummary = JSON.parse(bad.nullSummaryText) as { sourceGraphSha256: string };
    nullSummary.sourceGraphSha256 = SHA('a-different-graph');
    const tasks = STUDY_TASK_IDS.map((id) => (id === 'hazard-heavy' ? { ...bad, nullSummaryText: JSON.stringify(nullSummary) } : taskFixture(generalTaskOptions(id))));
    expect(() => buildArtifact(tasks)).toThrow(/does not match the manifest's biological graph/);
  });

  it('throws when the rewiring-null bytes do not match the manifest\'s pinned rewiringNull.sha256', () => {
    const tasks = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    expect(() =>
      buildTaskGeneralityArtifact({
        tasks,
        interventionIndexText,
        interventionIndexLabel: 'interventions/index.json',
        interventionIndexBytes,
        graphIndexBytes,
        rewiringNullBytes,
        pathwayInterventionsBytes,
        pathwayInterventionsParsed,
        manifestBiologicalSha,
        manifestGzipSha,
        manifestRewiringNullSha: SHA('stale-manifest-pin'),
        manifestPathwayInterventionsSha
      })
    ).toThrow(/does not match the manifest's rewiringNull\.sha256/);
  });

  it('throws when the pathway-interventions bytes do not match the manifest\'s pinned pathwayInterventions.sha256', () => {
    const tasks = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    expect(() =>
      buildTaskGeneralityArtifact({
        tasks,
        interventionIndexText,
        interventionIndexLabel: 'interventions/index.json',
        interventionIndexBytes,
        graphIndexBytes,
        rewiringNullBytes,
        pathwayInterventionsBytes,
        pathwayInterventionsParsed,
        manifestBiologicalSha,
        manifestGzipSha,
        manifestRewiringNullSha,
        manifestPathwayInterventionsSha: SHA('stale-manifest-pin')
      })
    ).toThrow(/does not match the manifest's pathwayInterventions\.sha256/);
  });

  it('throws when the pathway-interventions artifact\'s own sources.indexSha is not the intervention index these tasks share', () => {
    const tasks = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    const staleIndexPi = {
      ...pathwayInterventionsParsed,
      sources: { ...pathwayInterventionsParsed.sources, indexSha: SHA('a-different-index') }
    } as unknown as PathwayInterventionsArtifact;
    expect(() =>
      buildTaskGeneralityArtifact({
        tasks,
        interventionIndexText,
        interventionIndexLabel: 'interventions/index.json',
        interventionIndexBytes,
        graphIndexBytes,
        rewiringNullBytes,
        pathwayInterventionsBytes,
        pathwayInterventionsParsed: staleIndexPi,
        manifestBiologicalSha,
        manifestGzipSha,
        manifestRewiringNullSha,
        manifestPathwayInterventionsSha
      })
    ).toThrow(/the P\/C\/M graphs were not reused unchanged/);
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

  it('says "a 2-of-3-seed split", never "single-seed hit", when 2 of 3 trainer seeds reach pathway-supported', () => {
    const tasks = STUDY_TASK_IDS.map((id, i) =>
      i === 0 ? taskFixture({ ...generalTaskOptions(id), pTrainedScores: { 101: 100, 202: 100, 303: 1 } }) : taskFixture(generalTaskOptions(id))
    );
    const markdown = renderTaskGeneralityReportMarkdown(buildArtifact(tasks));
    expect(markdown).toMatch(/2 of 3 seeds reach pathway-supported -- a 2-of-3-seed split, not a robust finding/);
    expect(markdown).not.toMatch(/2 of 3 seeds reach pathway-supported -- this is a single-seed hit/);
  });

  it('states the Q-vs-MQ channel-specific result for a degenerate task, derived from data, never a hardcoded task name', () => {
    const [degenerateId, ...restIds] = STUDY_TASK_IDS;
    const degenerate = taskFixture({
      ...generalTaskOptions(degenerateId),
      cArmDegenerate: true,
      pathwayCategory: 'degenerate',
      qScore: 1.7,
      qChannelSpecific: true
    });
    const tasks = [degenerate, ...restIds.map((id) => taskFixture(generalTaskOptions(id)))];
    const markdown = renderTaskGeneralityReportMarkdown(buildArtifact(tasks));
    expect(markdown).toContain(`\`${degenerateId}\`'s authored pathway category is \`degenerate\``);
    expect(markdown).toContain('its Q-vs-MQ channel-specific result (holds) is independently valid and is not conflated with the degenerate P/C/M category');
  });

  it('discloses the no-specific-effect reporting-convention note only when some task actually used it', () => {
    const allPathwaySupported = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    const artifact = buildArtifact(allPathwaySupported);
    // Real generalTaskOptions fixtures use pathway-supported at every seed, so the note should not appear.
    expect(renderTaskGeneralityReportMarkdown(artifact)).not.toContain(artifact.trainedCategoryNote);

    const withNoSpecificEffect = STUDY_TASK_IDS.map((id, i) =>
      i === 0 ? taskFixture({ ...generalTaskOptions(id), pTrainedScores: { 101: 1, 202: 1, 303: 1 } }) : taskFixture(generalTaskOptions(id))
    );
    const artifactWithNse = buildArtifact(withNoSpecificEffect);
    expect(renderTaskGeneralityReportMarkdown(artifactWithNse)).toContain(artifactWithNse.trainedCategoryNote);
  });

  it('TASK_TABLE lists exactly the diffArenaConfig changes for every study task (never lets the two silently disagree)', () => {
    const tasks = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    const markdown = renderTaskGeneralityReportMarkdown(buildArtifact(tasks));
    const taskTableSection = markdown.split('## Measured sensor saturation')[0];
    for (const id of STUDY_TASK_IDS) {
      const row = taskTableSection.split('\n').find((line) => line.startsWith(`| \`${id}\``));
      expect(row, `no TASK_TABLE row for "${id}"`).toBeDefined();
      for (const [key, value] of Object.entries(diffArenaConfig(resolveArenaTask(id).config))) {
        expect(row, `${id}: ${key}`).toContain(`\`${key} ${value}\``);
      }
    }
  });

  // ---------------------------------------------------------------------------
  // Thermo-methodology review (Critical): the Limitations section's
  // clearance/foodDistance claim must be derived from WP2's own
  // clearance.json measurement, not a hardcoded a-priori prediction -- and
  // must change when the underlying measured data changes.
  // ---------------------------------------------------------------------------

  it('quotes the real measured clearance/foodDistance percentiles in a "Measured sensor saturation" table', () => {
    const tasks = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    const markdown = renderTaskGeneralityReportMarkdown(buildArtifact(tasks));
    expect(markdown).toContain('## Measured sensor saturation');
    expect(markdown).toContain('| task | channel | p5 | p50 | p95 | max | fraction saturated |');
    for (const id of STUDY_TASK_IDS) {
      expect(markdown).toContain(`| \`${id}\` | foodDistance | 0.100 | 0.200 | 0.300 | 0.350 | 0.00% |`);
    }
  });

  it('says foodDistance never saturates when every task\'s measured fractionSaturated is 0 (the real shipped case)', () => {
    const tasks = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    const markdown = renderTaskGeneralityReportMarkdown(buildArtifact(tasks));
    expect(markdown).toMatch(/Measured `foodDistance` never saturates in any task \(fractionSaturated 0 in all 4 tasks/);
    expect(markdown).not.toMatch(/Measured `foodDistance` saturates/);
  });

  it('changes the foodDistance limitation text to name the saturating task when its measured fractionSaturated becomes nonzero (proves the text tracks the data, not a fixed prediction)', () => {
    const [firstId, ...restIds] = STUDY_TASK_IDS;
    const saturating = taskFixture({ ...generalTaskOptions(firstId), foodDistanceFractionSaturated: 0.05 });
    const tasks = [saturating, ...restIds.map((id) => taskFixture(generalTaskOptions(id)))];
    const markdown = renderTaskGeneralityReportMarkdown(buildArtifact(tasks));
    expect(markdown).toMatch(new RegExp(`Measured \`foodDistance\` saturates \\(fractionSaturated > 0\\) in: \`${firstId}\` \\(5\\.00%\\)`));
    expect(markdown).toContain('every other task shows 0% `foodDistance` saturation');
    expect(markdown).not.toMatch(/Measured `foodDistance` never saturates in any task/);
  });

  it('says no task shows wall-clearance saturation when every measured fractionSaturated is 0', () => {
    const tasks = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    const markdown = renderTaskGeneralityReportMarkdown(buildArtifact(tasks));
    expect(markdown).toContain(
      'No task shows any measured wall-clearance saturation on `forwardClearance`/`leftClearance`/`rightClearance` (fractionSaturated 0 on every channel in every task).'
    );
  });

  it('names the specific task and channel when measured wall-clearance saturation is nonzero', () => {
    const [firstId, ...restIds] = STUDY_TASK_IDS;
    const saturating = taskFixture({ ...generalTaskOptions(firstId), forwardClearanceFractionSaturated: 0.0116 });
    const tasks = [saturating, ...restIds.map((id) => taskFixture(generalTaskOptions(id)))];
    const markdown = renderTaskGeneralityReportMarkdown(buildArtifact(tasks));
    expect(markdown).toMatch(new RegExp(`Measured wall clearance saturates only in: \\\`${firstId}\\\` \\(forwardClearance 1\\.16%\\)`));
    expect(markdown).toContain('every other task shows 0% wall-clearance saturation on every channel');
    expect(markdown).not.toContain('No task shows any measured wall-clearance saturation');
  });

  it('states the trained-side "generalizes" rule explicitly, disclosing that 00-overview.md does not define it', () => {
    const tasks = STUDY_TASK_IDS.map((id) => taskFixture(generalTaskOptions(id)));
    const markdown = renderTaskGeneralityReportMarkdown(buildArtifact(tasks));
    expect(markdown).toContain('**Trained-side "generalizes" (this study\'s own operational rule, not verbatim from `00-overview.md`):**');
    expect(markdown).toMatch(/never\s+states a trained-side predicate in those terms/);
    expect(markdown).toMatch(/has not been ratified in `00-overview\.md` itself/);
  });
});

describe('runTaskGeneralityReport (CLI layer)', () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'task-generality-cli-'));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  // The CLI layer's `runTaskGeneralityReport` genuinely `JSON.parse`s the
  // pathway-interventions bytes it reads from disk (unlike the pure-builder
  // tests above, which pass `pathwayInterventionsParsed` in as an
  // already-parsed object) -- these bytes must be real, valid JSON matching
  // that same fixture, with its own sha threaded through the manifest and
  // `pi.sources.rewiringNullSha`/`indexSha`/`biologicalSha` (which already
  // point at `manifestRewiringNullSha`/`interventionIndexSha`/`manifestBiologicalSha`
  // via `pathwayInterventionsParsed` above).
  const pathwayInterventionsJsonBytes = Buffer.from(JSON.stringify(pathwayInterventionsParsed), 'utf8');
  const cliManifestPathwayInterventionsSha = sha256Hex(pathwayInterventionsJsonBytes);

  /** Writes one complete, internally-consistent world of inputs to `dir`, mirroring the real Spark layout (`--authored-dir`/`--trained-dir` each `<dir>/<task>/...`). */
  const writeWorld = (): void => {
    writeFileSync(join(dir, 'interventions-index.json'), interventionIndexText);
    writeFileSync(join(dir, 'graphs-index.json'), 'graph-index-fixture');
    writeFileSync(join(dir, 'rewiring-null-v1.json'), rewiringNullBytes);
    writeFileSync(join(dir, 'pathway-interventions-v1.json'), pathwayInterventionsJsonBytes);
    // `verifyManifestRoundTrips` (called by `runTaskGeneralityReport`) requires
    // the file on disk to already be `sortKeysDeep`+`JSON.stringify(..., null, 2)`+'\n' --
    // exactly the format `updateManifestWithTaskGenerality` itself writes back.
    const manifestObject = {
      binarySha256: manifestBiologicalSha,
      gzipSha256: manifestGzipSha,
      rewiringNull: { sha256: manifestRewiringNullSha },
      pathwayInterventions: { sha256: cliManifestPathwayInterventionsSha }
    };
    writeFileSync(join(dir, 'manifest.json'), `${JSON.stringify(sortKeysDeep(manifestObject), null, 2)}\n`);
    for (const id of STUDY_TASK_IDS) {
      const fixture = taskFixture(generalTaskOptions(id));
      const paths = taskInputPaths(dir, dir, id);
      mkdirSync(dirname(paths.nullSummary), { recursive: true });
      writeFileSync(paths.nullSummary, fixture.nullSummaryText, { flag: 'wx' });
      writeFileSync(paths.interventionStats, fixture.interventionStatsText, { flag: 'wx' });
      writeFileSync(paths.trained, fixture.trainedText, { flag: 'wx' });
      writeFileSync(paths.clearance, fixture.clearanceText, { flag: 'wx' });
    }
  };

  const runArgs = () =>
    parseTaskGeneralityReportArgs([
      '--authored-dir',
      dir,
      '--trained-dir',
      dir,
      '--graph-index',
      join(dir, 'graphs-index.json'),
      '--intervention-index',
      join(dir, 'interventions-index.json'),
      '--rewiring-null',
      join(dir, 'rewiring-null-v1.json'),
      '--pathway-interventions',
      join(dir, 'pathway-interventions-v1.json'),
      '--manifest',
      join(dir, 'manifest.json'),
      '--out',
      join(dir, 'task-generality-v1.json'),
      '--report-md',
      join(dir, 'task-generality-report.md')
    ]);

  it('writes the artifact, the manifest key, and the report', () => {
    writeWorld();
    const result = runTaskGeneralityReport(runArgs());
    expect(result.artifact.overall.authored.verdict).toBe('general');

    const writtenArtifact = readFileSync(join(dir, 'task-generality-v1.json'), 'utf8');
    expect(sha256Hex(writtenArtifact)).toBe(result.artifactSha256);

    const manifest = JSON.parse(readFileSync(join(dir, 'manifest.json'), 'utf8')) as { taskGenerality: { artifact: string; sha256: string } };
    expect(manifest.taskGenerality).toEqual({ artifact: 'task-generality-v1.json', sha256: result.artifactSha256 });

    const reportMd = readFileSync(join(dir, 'task-generality-report.md'), 'utf8');
    expect(reportMd).toContain('# Task generality of the null and pathway findings');
  });

  it('regenerating twice against the same inputs gives byte-identical output', () => {
    writeWorld();
    const first = runTaskGeneralityReport(runArgs());
    const firstBytes = readFileSync(first.out);
    // The manifest now carries the first run's taskGenerality key -- an
    // exact rerun (no input changed) must still reproduce the same bytes,
    // matching the "regeneration is byte-identical" acceptance criterion.
    const second = runTaskGeneralityReport(runArgs());
    const secondBytes = readFileSync(second.out);
    expect(secondBytes.equals(firstBytes)).toBe(true);
    expect(second.artifactSha256).toBe(first.artifactSha256);
  });

  it('rejects an unknown CLI flag', () => {
    expect(() => parseTaskGeneralityReportArgs(['--bogus', 'x'])).toThrow(/Unknown argument/);
  });

  it('rejects --out and --report-md pointing at the same path', () => {
    expect(() => parseTaskGeneralityReportArgs(['--out', '/tmp/x.json', '--report-md', '/tmp/x.json'])).toThrow(
      /--out and --report-md must not be the same path/
    );
  });
});
