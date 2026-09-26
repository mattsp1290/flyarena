// @vitest-environment node
import { describe, expect, it } from 'vitest';

import {
  buildTrainedStatistics,
  CONTROL_TRAINER_SEED,
  evaluateTrainedCategory,
  isTrainedRobust,
  P_TRAINER_SEEDS,
  TRAINED_ARM_SIZE,
  trainedCategoryFlags,
  trainedTaskResult,
  type PTrainerSeed,
  type TrainedArmDistribution,
  type TrainedSeedResult
} from '../../scripts/null/intervention-report-trained';
import type { GraphKind, GraphListIndexInfo } from '../../scripts/null/intervention-report';
import type { NullTrainedInterventionEvaluationRaw, NullTrainedInterventionGraphRaw } from '../../scripts/null/null-trained-evaluate-graph-list';

/**
 * Coverage for `scripts/null/intervention-report-trained.ts` — WP4's
 * trained-decoder category evaluator (`.agents/plans/pathway-interventions/00-overview.md`'s
 * trained-decoder rules, plus this study's own predeclared "no-specific-effect"
 * disclosure for the generic-vs-not-supported split the trained side cannot
 * mechanically decide). Uses small synthetic inputs throughout, mirroring
 * `tests/unit/intervention-report.test.ts`'s own precedent.
 */

const armDist = (scores: readonly number[]): TrainedArmDistribution => {
  const sorted = [...scores].sort((a, b) => a - b);
  return { n: sorted.length, scores: sorted, max: sorted[sorted.length - 1] };
};

describe('evaluateTrainedCategory / trainedCategoryFlags', () => {
  const cArm = armDist([1, 2, 3, 4, 5]); // max 5
  const mArm = armDist([2, 4, 6, 8, 10]); // max 10

  it('pathway-supported: P above both arm maxima', () => {
    expect(evaluateTrainedCategory(11, cArm, mArm)).toBe('pathway-supported');
    expect(trainedCategoryFlags(11, cArm, mArm)).toEqual({ aboveC: true, aboveM: true });
  });

  it('edge-class-effect: P above C max but not above M max', () => {
    expect(evaluateTrainedCategory(7, cArm, mArm)).toBe('edge-class-effect');
    expect(trainedCategoryFlags(7, cArm, mArm)).toEqual({ aboveC: true, aboveM: false });
  });

  it('edge-class-effect boundary: P exactly at M max does not clear it (strict >)', () => {
    expect(evaluateTrainedCategory(10, cArm, mArm)).toBe('edge-class-effect');
  });

  it('no-specific-effect: P at or below C max, regardless of M', () => {
    expect(evaluateTrainedCategory(5, cArm, mArm)).toBe('no-specific-effect'); // exactly at C max
    expect(evaluateTrainedCategory(3, cArm, mArm)).toBe('no-specific-effect'); // well below
  });
});

describe('isTrainedRobust', () => {
  const seedResult = (category: TrainedSeedResult['category']): TrainedSeedResult => ({
    trainerSeed: 101,
    score: 0,
    stats: { n: 1, mean: 0, median: 0, std: 0, ci95: [0, 0] },
    aboveC: false,
    aboveM: false,
    category,
    context: { publishedTrainedNullP25: 0, percentileResolution: 0.05, abovePublishedNullP25: false }
  });

  it('true when all three seeds agree', () => {
    const perSeed: Record<PTrainerSeed, TrainedSeedResult> = {
      101: seedResult('no-specific-effect'),
      202: seedResult('no-specific-effect'),
      303: seedResult('no-specific-effect')
    };
    expect(isTrainedRobust(perSeed)).toBe(true);
  });

  it('false on a 2-1 split (not merely "no seed disagrees with the majority")', () => {
    const perSeed: Record<PTrainerSeed, TrainedSeedResult> = {
      101: seedResult('no-specific-effect'),
      202: seedResult('no-specific-effect'),
      303: seedResult('pathway-supported')
    };
    expect(isTrainedRobust(perSeed)).toBe(false);
  });
});

describe('buildTrainedStatistics: end-to-end on synthetic data', () => {
  const graphRaw = (id: string, trainerSeed: number, score: number): NullTrainedInterventionGraphRaw => ({
    id,
    trainerSeed,
    gzipSha256: `gz-${id}`.padEnd(64, '0'),
    armBundleSha256: `arm-${id}`.padEnd(64, '0'),
    heldOutSeeds: [30001, 30002],
    movementScore: [score, score],
    foodPickups: [0, 0],
    hazardContacts: [0, 0]
  });

  const infoFor = (idsAndKinds: readonly (readonly [string, GraphKind])[]): GraphListIndexInfo => ({
    entries: new Map(idsAndKinds.map(([id, kind]) => [id, { kind, gzipSha256: `gz-${id}`.padEnd(64, '0') }])),
    controlCount: 100
  });

  const CONTROL_IDS_C = ['C000', 'C001', 'C002', 'C003', 'C004'];
  const CONTROL_IDS_M = ['M1000', 'M1001', 'M1002', 'M1003', 'M1004'];

  const buildRaw = (pScores: Readonly<Record<PTrainerSeed, number>>, cScores: readonly number[], mScores: readonly number[]): NullTrainedInterventionEvaluationRaw => ({
    version: 1,
    seeds: { start: 30001, count: 2 },
    ticks: 20,
    substeps: 4,
    graphListSha256: 'x'.repeat(64),
    host: { arch: 'arm64', node: 'v22.0.0' },
    d: 48,
    evaluatorGitRev: 'deadbeef',
    cemConfig: null,
    cemConfigWarnings: [],
    runs: [
      ...P_TRAINER_SEEDS.map((seed) => graphRaw('P', seed, pScores[seed])),
      ...CONTROL_IDS_C.map((id, i) => graphRaw(id, CONTROL_TRAINER_SEED, cScores[i])),
      ...CONTROL_IDS_M.map((id, i) => graphRaw(id, CONTROL_TRAINER_SEED, mScores[i]))
    ]
  });

  const infoAll = infoFor([
    ['P', 'P'],
    ...CONTROL_IDS_C.map((id) => [id, 'C'] as const),
    ...CONTROL_IDS_M.map((id) => [id, 'M'] as const)
  ]);

  it('reports no-specific-effect at every seed and trainedRobust=true when P never clears the C arm (this study\'s real result)', () => {
    const raw = buildRaw({ 101: 65, 202: 70, 303: 70.5 }, [62, 70, 70, 70.5, 71.5], [72, 73, 66, 75, 70]);
    const stats = buildTrainedStatistics(raw, infoAll, [64, 67, 68, 69, 70, 72, 75, 80, 85, 122], 42, 200);

    expect(stats.controls.C.max).toBe(71.5);
    expect(stats.controls.M.max).toBe(75);
    expect(stats.perSeed[101].category).toBe('no-specific-effect');
    expect(stats.perSeed[202].category).toBe('no-specific-effect');
    expect(stats.perSeed[303].category).toBe('no-specific-effect');
    expect(stats.trainedRobust).toBe(true);
    expect(stats.note.length).toBeGreaterThan(0);
  });

  it('reports pathway-supported when P clears both arms at every seed', () => {
    const raw = buildRaw({ 101: 100, 202: 101, 303: 102 }, [1, 2, 3, 4, 5], [2, 4, 6, 8, 10]);
    const stats = buildTrainedStatistics(raw, infoAll, [1, 2, 3], 42, 200);
    expect(stats.perSeed[101].category).toBe('pathway-supported');
    expect(stats.trainedRobust).toBe(true);
  });

  it('reports edge-class-effect when P clears C but not M at every seed', () => {
    const raw = buildRaw({ 101: 6, 202: 7, 303: 8 }, [1, 2, 3, 4, 5], [20, 21, 22, 23, 24]);
    const stats = buildTrainedStatistics(raw, infoAll, [1, 2, 3], 42, 200);
    expect(stats.perSeed[101].category).toBe('edge-class-effect');
    expect(stats.trainedRobust).toBe(true);
  });

  it('trainedRobust is false when seeds disagree on category', () => {
    // seed 101 stays below C max (no-specific-effect); seeds 202/303 clear both arms (pathway-supported).
    const raw = buildRaw({ 101: 3, 202: 100, 303: 100 }, [1, 2, 3, 4, 5], [2, 4, 6, 8, 10]);
    const stats = buildTrainedStatistics(raw, infoAll, [1, 2, 3], 42, 200);
    expect(stats.perSeed[101].category).toBe('no-specific-effect');
    expect(stats.perSeed[202].category).toBe('pathway-supported');
    expect(stats.trainedRobust).toBe(false);
  });

  it('context.abovePublishedNullP25 is informational only and never affects category', () => {
    // Published trained null is all far below P's score, so P clears the
    // context p25 easily -- but P still does not clear the real C/M arms,
    // so the category must still be no-specific-effect.
    const raw = buildRaw({ 101: 5, 202: 5, 303: 5 }, [10, 11, 12, 13, 14], [10, 11, 12, 13, 14]);
    const stats = buildTrainedStatistics(raw, infoAll, [0, 0.5, 1, 1.5, 2], 42, 200);
    expect(stats.perSeed[101].category).toBe('no-specific-effect');
    expect(stats.perSeed[101].context.abovePublishedNullP25).toBe(true);
  });

  it('running twice on the same input is byte-identical (JSON.stringify)', () => {
    const raw = buildRaw({ 101: 65, 202: 70, 303: 70.5 }, [62, 70, 70, 70.5, 71.5], [72, 73, 66, 75, 70]);
    const first = buildTrainedStatistics(raw, infoAll, [64, 67, 68], 42, 200);
    const second = buildTrainedStatistics(raw, infoAll, [64, 67, 68], 42, 200);
    expect(JSON.stringify(second)).toBe(JSON.stringify(first));
  });

  it('throws on unsupported version', () => {
    const raw = { ...buildRaw({ 101: 1, 202: 1, 303: 1 }, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), version: 2 as unknown as 1 };
    expect(() => buildTrainedStatistics(raw, infoAll, [1], 42, 200)).toThrow(/unsupported version/);
  });

  it('throws when a P trainer seed is missing', () => {
    const raw = buildRaw({ 101: 1, 202: 1, 303: 1 }, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]);
    const withoutSeed303 = { ...raw, runs: raw.runs.filter((r) => !(r.id === 'P' && r.trainerSeed === 303)) };
    expect(() => buildTrainedStatistics(withoutSeed303, infoAll, [1], 42, 200)).toThrow(/missing the P run at trainer seed 303/);
  });

  it('throws when a P run has an unexpected trainer seed', () => {
    const raw = buildRaw({ 101: 1, 202: 1, 303: 1 }, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]);
    const withBadSeed = {
      ...raw,
      runs: raw.runs.map((r) => (r.id === 'P' && r.trainerSeed === 303 ? { ...r, trainerSeed: 999 } : r))
    };
    expect(() => buildTrainedStatistics(withBadSeed, infoAll, [1], 42, 200)).toThrow(/unexpected trainer seed 999/);
  });

  it(`throws when a control arm has fewer than ${TRAINED_ARM_SIZE} runs`, () => {
    const raw = buildRaw({ 101: 1, 202: 1, 303: 1 }, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]);
    const shortC = { ...raw, runs: raw.runs.filter((r) => r.id !== 'C004') };
    expect(() => buildTrainedStatistics(shortC, infoAll, [1], 42, 200)).toThrow(/expected exactly 5 kind-"C" runs, found 4/);
  });

  it('throws when a control-arm run is at the wrong trainer seed', () => {
    const raw = buildRaw({ 101: 1, 202: 1, 303: 1 }, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]);
    const wrongSeed = { ...raw, runs: raw.runs.map((r) => (r.id === 'C000' ? { ...r, trainerSeed: 202 } : r)) };
    expect(() => buildTrainedStatistics(wrongSeed, infoAll, [1], 42, 200)).toThrow(/expected the control trainer seed 101/);
  });

  it('throws when trained.json has a Q/MQ/R run (not evaluated with trained readouts)', () => {
    const raw = buildRaw({ 101: 1, 202: 1, 303: 1 }, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]);
    const withQ = { ...raw, runs: [...raw.runs, graphRaw('Q', CONTROL_TRAINER_SEED, 5)] };
    const infoWithQ = infoFor([...infoAll.entries.entries()].map(([id, e]) => [id, e.kind] as const).concat([['Q', 'Q']]));
    expect(() => buildTrainedStatistics(withQ, infoWithQ, [1], 42, 200)).toThrow(/Q\/MQ\/R are not evaluated/);
  });

  it('throws when publishedTrainedNullScores is empty', () => {
    const raw = buildRaw({ 101: 1, 202: 1, 303: 1 }, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]);
    expect(() => buildTrainedStatistics(raw, infoAll, [], 42, 200)).toThrow(/publishedTrainedNullScores must be a non-empty list of finite numbers/);
  });

  it('throws when publishedTrainedNullScores contains a non-finite value', () => {
    const raw = buildRaw({ 101: 1, 202: 1, 303: 1 }, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]);
    expect(() => buildTrainedStatistics(raw, infoAll, [1, Number.NaN], 42, 200)).toThrow(/publishedTrainedNullScores must be a non-empty list of finite numbers/);
  });

  it('throws when trained.json "runs" is not an array', () => {
    const raw = { ...buildRaw({ 101: 1, 202: 1, 303: 1 }, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), runs: null as unknown as NullTrainedInterventionGraphRaw[] };
    expect(() => buildTrainedStatistics(raw, infoAll, [1], 42, 200)).toThrow(/trained\.json "runs" is not an array/);
  });

  it('throws when a run has an empty movementScore', () => {
    const raw = buildRaw({ 101: 1, 202: 1, 303: 1 }, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]);
    const emptyScore = { ...raw, runs: raw.runs.map((r) => (r.id === 'C000' ? { ...r, movementScore: [] } : r)) };
    expect(() => buildTrainedStatistics(emptyScore, infoAll, [1], 42, 200)).toThrow(/"C000"\.movementScore is missing or empty/);
  });

  it('throws when a run has a non-finite movementScore entry', () => {
    const raw = buildRaw({ 101: 1, 202: 1, 303: 1 }, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]);
    const badScore = { ...raw, runs: raw.runs.map((r) => (r.id === 'C000' ? { ...r, movementScore: [1, Number.NaN] } : r)) };
    expect(() => buildTrainedStatistics(badScore, infoAll, [1], 42, 200)).toThrow(/"C000"\.movementScore\[1\] is not a finite number/);
  });

  it("throws when a run's gzipSha256 disagrees with the index (stale trained.json)", () => {
    const raw = buildRaw({ 101: 1, 202: 1, 303: 1 }, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]);
    const staleGzip = { ...raw, runs: raw.runs.map((r) => (r.id === 'C000' ? { ...r, gzipSha256: 'stale'.padEnd(64, '0') } : r)) };
    expect(() => buildTrainedStatistics(staleGzip, infoAll, [1], 42, 200)).toThrow(/"C000" was scored from a different graph file than index\.json currently lists/);
  });
});

describe('trainedTaskResult', () => {
  const pScores: Readonly<Record<PTrainerSeed, number>> = { 101: 11, 202: 11, 303: 11 };

  it('degenerate: the C arm has IQR below the threshold (an all-equal arm)', () => {
    const cArm = armDist([5, 5, 5, 5, 5]);
    const mArm = armDist([2, 4, 6, 8, 10]);
    expect(trainedTaskResult(pScores, cArm, mArm)).toEqual({ degenerate: true });
  });

  it('degenerate: the M arm has IQR below the threshold', () => {
    const cArm = armDist([1, 2, 3, 4, 5]);
    const mArm = armDist([7, 7, 7, 7, 7]);
    expect(trainedTaskResult(pScores, cArm, mArm)).toEqual({ degenerate: true });
  });

  it('non-degenerate and robust: every seed lands in the same category', () => {
    const cArm = armDist([1, 2, 3, 4, 5]); // max 5
    const mArm = armDist([2, 4, 6, 8, 10]); // max 10
    const result = trainedTaskResult({ 101: 11, 202: 12, 303: 13 }, cArm, mArm);
    expect(result.degenerate).toBe(false);
    if (result.degenerate) throw new Error('unreachable');
    expect(result.perSeed).toEqual({ 101: 'pathway-supported', 202: 'pathway-supported', 303: 'pathway-supported' });
    expect(result.trainedRobust).toBe(true);
  });

  it('non-degenerate and not robust: seeds disagree on category (a tie at the C max counts as no-specific-effect)', () => {
    const cArm = armDist([1, 2, 3, 4, 5]); // max 5
    const mArm = armDist([2, 4, 6, 8, 10]); // max 10
    const result = trainedTaskResult({ 101: 5, 202: 7, 303: 11 }, cArm, mArm);
    expect(result.degenerate).toBe(false);
    if (result.degenerate) throw new Error('unreachable');
    expect(result.perSeed).toEqual({ 101: 'no-specific-effect', 202: 'edge-class-effect', 303: 'pathway-supported' });
    expect(result.trainedRobust).toBe(false);
  });
});
