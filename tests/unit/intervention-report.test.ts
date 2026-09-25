// @vitest-environment node
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeEach, describe, expect, it } from 'vitest';

import {
  armDistribution,
  buildInterventionStatistics,
  checkBiologicalReproduction,
  evaluateCategory,
  evaluateChannelSpecific,
  readGraphKinds,
  readPublishedNull,
  type GraphKind,
  type PublishedNull
} from '../../scripts/null/intervention-report';
import type { NullGraphListEvaluationRaw } from '../../scripts/null/null-evaluate';

/**
 * Coverage for `scripts/null/intervention-report.ts`'s statistics layer
 * (`.agents/plans/pathway-interventions/03-evaluation.md`'s WP2). Uses
 * small synthetic inputs throughout -- the expensive part (one episode per
 * seed per graph) lives in `null-evaluate.ts`/`null-worker.ts`
 * (`tests/unit/null-evaluate.test.ts` covers that); this module is pure
 * statistics over already-computed scores.
 */

describe('armDistribution', () => {
  it('computes n/p5/p50/p95 over a sorted 100-length arm', () => {
    const scores = Array.from({ length: 100 }, (_, i) => i); // 0..99
    const arm = armDistribution(scores);
    expect(arm.n).toBe(100);
    expect(arm.scores).toEqual([...scores].sort((a, b) => a - b));
    expect(arm.p50).toBe(50);
    // quantileIndex(100, 0.95) = min(99, ceil(95)-1) = 94 (0-indexed values 0..99).
    expect(arm.p95).toBe(94);
    expect(arm.p5).toBe(5);
  });

  it('throws on an empty arm', () => {
    expect(() => armDistribution([])).toThrow(/at least one score/);
  });
});

describe('evaluateCategory', () => {
  const cArm = armDistribution(Array.from({ length: 100 }, () => 1));
  const mArm = armDistribution(Array.from({ length: 100 }, () => 2));

  it('not-supported: P below the null floor percentile', () => {
    expect(evaluateCategory(10, 0.1, cArm, mArm)).toBe('not-supported');
  });

  it('generic-rewiring-effect: P at/above the floor but at/below C p95', () => {
    expect(evaluateCategory(1, 0.25, cArm, mArm)).toBe('generic-rewiring-effect');
    expect(evaluateCategory(0.5, 0.9, cArm, mArm)).toBe('generic-rewiring-effect');
  });

  it('edge-class-effect: P above C p95 but at/below M p95', () => {
    expect(evaluateCategory(1.5, 0.9, cArm, mArm)).toBe('edge-class-effect');
  });

  it('pathway-supported: P above both C p95 and M p95', () => {
    expect(evaluateCategory(3, 0.99, cArm, mArm)).toBe('pathway-supported');
  });

  it('boundary: exactly at the null floor percentile counts as reached', () => {
    // 0.25 is not < 0.25, so the floor check passes; P (0.5) is <= C's p95 (1), so generic.
    expect(evaluateCategory(0.5, 0.25, cArm, mArm)).toBe('generic-rewiring-effect');
  });
});

describe('evaluateChannelSpecific', () => {
  const mqArm = armDistribution(Array.from({ length: 100 }, () => 1));

  it('true when Q clears both the null floor and MQ p95', () => {
    expect(evaluateChannelSpecific(2, 0.5, mqArm)).toBe(true);
  });

  it('false when Q is below the null floor percentile', () => {
    expect(evaluateChannelSpecific(2, 0.1, mqArm)).toBe(false);
  });

  it('false when Q does not clear MQ p95', () => {
    expect(evaluateChannelSpecific(1, 0.9, mqArm)).toBe(false);
  });
});

describe('checkBiologicalReproduction', () => {
  it('matches when the computed mean equals the published score exactly', () => {
    const publishedNull: PublishedNull = { biologicalScore: 2, scores: [0, 1, 2, 3] };
    const check = checkBiologicalReproduction([1, 2, 3], publishedNull);
    expect(check.computedScore).toBe(2);
    expect(check.matches).toBe(true);
  });

  it('does not match on any deviation', () => {
    const publishedNull: PublishedNull = { biologicalScore: 2, scores: [0, 1, 2, 3] };
    const check = checkBiologicalReproduction([1, 2, 3.0001], publishedNull);
    expect(check.matches).toBe(false);
  });
});

describe('readGraphKinds', () => {
  let root: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'intervention-report-kinds-'));
  });

  it('parses id -> kind from index.json entries', () => {
    const path = join(root, 'index.json');
    writeFileSync(
      path,
      JSON.stringify({
        entries: [
          { id: 'P', kind: 'P' },
          { id: 'C000', kind: 'C' },
          { id: 'MQ2000', kind: 'MQ' }
        ]
      })
    );
    const kinds = readGraphKinds(path);
    expect(kinds.get('P')).toBe('P');
    expect(kinds.get('C000')).toBe('C');
    expect(kinds.get('MQ2000')).toBe('MQ');
    rmSync(root, { recursive: true, force: true });
  });

  it('rejects an entry with an unrecognized kind', () => {
    const path = join(root, 'index.json');
    writeFileSync(path, JSON.stringify({ entries: [{ id: 'X', kind: 'bogus' }] }));
    expect(() => readGraphKinds(path)).toThrow(/malformed entry/);
    rmSync(root, { recursive: true, force: true });
  });

  it('rejects the same id listed with two different kinds', () => {
    const path = join(root, 'index.json');
    writeFileSync(
      path,
      JSON.stringify({
        entries: [
          { id: 'X', kind: 'C' },
          { id: 'X', kind: 'M' }
        ]
      })
    );
    expect(() => readGraphKinds(path)).toThrow(/two different kinds/);
    rmSync(root, { recursive: true, force: true });
  });

  it('tolerates the same id/kind repeated', () => {
    const path = join(root, 'index.json');
    writeFileSync(
      path,
      JSON.stringify({
        entries: [
          { id: 'X', kind: 'C' },
          { id: 'X', kind: 'C' }
        ]
      })
    );
    expect(() => readGraphKinds(path)).not.toThrow();
    rmSync(root, { recursive: true, force: true });
  });
});

describe('readPublishedNull', () => {
  it('parses biological.score and rewired[].score', () => {
    const root = mkdtempSync(join(tmpdir(), 'intervention-report-null-'));
    const path = join(root, 'null.json');
    writeFileSync(
      path,
      JSON.stringify({ biological: { score: 1.5 }, rewired: [{ score: 1 }, { score: 2 }, { score: 3 }] })
    );
    const parsed = readPublishedNull(path);
    expect(parsed.biologicalScore).toBe(1.5);
    expect(parsed.scores).toEqual([1, 2, 3]);
    rmSync(root, { recursive: true, force: true });
  });

  it('throws when biological.score is missing', () => {
    const root = mkdtempSync(join(tmpdir(), 'intervention-report-null-bad-'));
    const path = join(root, 'null.json');
    writeFileSync(path, JSON.stringify({ rewired: [{ score: 1 }] }));
    expect(() => readPublishedNull(path)).toThrow(/missing biological\.score/);
    rmSync(root, { recursive: true, force: true });
  });

  it('throws when rewired is empty', () => {
    const root = mkdtempSync(join(tmpdir(), 'intervention-report-null-empty-'));
    const path = join(root, 'null.json');
    writeFileSync(path, JSON.stringify({ biological: { score: 1 }, rewired: [] }));
    expect(() => readPublishedNull(path)).toThrow(/no rewired entries/);
    rmSync(root, { recursive: true, force: true });
  });
});

describe('buildInterventionStatistics: end-to-end on synthetic data', () => {
  // A deliberately tiny graph list: P, Q, 4 C, 4 M, 4 MQ graphs. Scores are
  // hand-picked so P clearly beats C and M (pathway-supported) and Q clearly
  // beats MQ (channel-specific), against a published null whose scores put
  // P/Q comfortably above the 25th percentile.
  const seeds = [30001, 30002, 30003];
  const constantMovementScore = (value: number): readonly number[] => seeds.map(() => value);

  const graphList = (idsAndScores: readonly (readonly [string, number])[]): NullGraphListEvaluationRaw => ({
    version: 1,
    sourceGraphSha256: 'x'.repeat(64),
    seeds: { start: 30001, count: seeds.length },
    ticks: 20,
    substeps: 4,
    decoder: 'authored',
    biological: { heldOutSeeds: seeds, movementScore: constantMovementScore(0), foodPickups: [0, 0, 0], hazardContacts: [0, 0, 0] },
    disconnected: { heldOutSeeds: seeds, movementScore: constantMovementScore(-1), foodPickups: [0, 0, 0], hazardContacts: [0, 0, 0] },
    graphs: idsAndScores.map(([id, score]) => ({
      id,
      gzipSha256: 'y'.repeat(64),
      binarySha256: 'z'.repeat(64),
      heldOutSeeds: seeds,
      movementScore: constantMovementScore(score),
      foodPickups: [0, 0, 0],
      hazardContacts: [0, 0, 0]
    })),
    host: { arch: 'arm64', node: 'v22.0.0' }
  });

  const kindsFor = (idsAndKinds: readonly (readonly [string, GraphKind])[]): ReadonlyMap<string, GraphKind> =>
    new Map(idsAndKinds);

  const buildFixture = () => {
    const cIds = ['C000', 'C001', 'C002', 'C003'];
    const mIds = ['M1000', 'M1001', 'M1002', 'M1003'];
    const mqIds = ['MQ2000', 'MQ2001', 'MQ2002', 'MQ2003'];
    const idsAndScores: (readonly [string, number])[] = [
      ['P', 10],
      ['Q', 8],
      ...cIds.map((id) => [id, 1] as const),
      ...mIds.map((id) => [id, 2] as const),
      ...mqIds.map((id) => [id, 1] as const)
    ];
    const raw = graphList(idsAndScores);
    const kinds = kindsFor([
      ['P', 'P'],
      ['Q', 'Q'],
      ...cIds.map((id) => [id, 'C'] as const),
      ...mIds.map((id) => [id, 'M'] as const),
      ...mqIds.map((id) => [id, 'MQ'] as const)
    ]);
    // A published null of 20 values, all well below P/Q's scores, so both
    // land at the 100th percentile (comfortably above the 25% floor).
    const publishedNull: PublishedNull = {
      biologicalScore: 0,
      scores: Array.from({ length: 20 }, (_, i) => -5 + i * 0.1)
    };
    return { raw, kinds, publishedNull };
  };

  it('P is pathway-supported and Q is channel-specific under these scores', () => {
    const { raw, kinds, publishedNull } = buildFixture();
    const stats = buildInterventionStatistics(raw, kinds, publishedNull, 42, 200);

    expect(stats.biologicalReproduction.matches).toBe(true);
    expect(stats.graphs).toHaveLength(14); // P, Q, 4 C, 4 M, 4 MQ
    // Sorted by id ascending.
    expect(stats.graphs.map((g) => g.id)).toEqual([...stats.graphs.map((g) => g.id)].sort());

    expect(stats.controls.C.n).toBe(4);
    expect(stats.controls.C.p95).toBe(1);
    expect(stats.controls.M.n).toBe(4);
    expect(stats.controls.M.p95).toBe(2);
    expect(stats.controls.MQ.n).toBe(4);
    expect(stats.controls.MQ.p95).toBe(1);

    expect(stats.p.score).toBe(10);
    expect(stats.p.percentileInPublishedNull).toBe(1); // above every published-null value
    expect(stats.p.category).toBe('pathway-supported');
    expect(stats.p.pRankAmongC).toBeGreaterThan(0);
    expect(stats.p.pRankAmongC).toBeLessThanOrEqual(1);

    expect(stats.q.score).toBe(8);
    expect(stats.q.channelSpecific).toBe(true);
    expect(stats.q.qRankAmongMQ).toBeGreaterThan(0);
  });

  it('every non-biological graph carries a pairedVsBiological difference', () => {
    const { raw, kinds, publishedNull } = buildFixture();
    const stats = buildInterventionStatistics(raw, kinds, publishedNull, 42, 200);
    for (const graph of stats.graphs) {
      expect(graph.pairedVsBiological).toBeDefined();
      expect(graph.pairedVsBiological?.n).toBe(seeds.length);
    }
  });

  it('running the statistics twice on the same input is byte-identical (JSON.stringify)', () => {
    const { raw, kinds, publishedNull } = buildFixture();
    const first = buildInterventionStatistics(raw, kinds, publishedNull, 42, 200);
    const second = buildInterventionStatistics(raw, kinds, publishedNull, 42, 200);
    expect(JSON.stringify(second)).toBe(JSON.stringify(first));
  });

  it('generic-rewiring-effect when P does not clear the C arm', () => {
    const cIds = ['C000', 'C001'];
    const mIds = ['M1000', 'M1001'];
    const mqIds = ['MQ2000', 'MQ2001'];
    const idsAndScores: (readonly [string, number])[] = [
      ['P', 1],
      ['Q', 1],
      ...cIds.map((id) => [id, 5] as const), // P (1) is well below C's p95 (5)
      ...mIds.map((id) => [id, 5] as const),
      ...mqIds.map((id) => [id, 5] as const)
    ];
    const raw = graphList(idsAndScores);
    const kinds = kindsFor([
      ['P', 'P'],
      ['Q', 'Q'],
      ...cIds.map((id) => [id, 'C'] as const),
      ...mIds.map((id) => [id, 'M'] as const),
      ...mqIds.map((id) => [id, 'MQ'] as const)
    ]);
    const publishedNull: PublishedNull = { biologicalScore: 0, scores: Array.from({ length: 20 }, (_, i) => -5 + i * 0.1) };
    const stats = buildInterventionStatistics(raw, kinds, publishedNull, 42, 200);
    expect(stats.p.category).toBe('generic-rewiring-effect');
  });

  it('not-supported when P never clears the published null floor', () => {
    const cIds = ['C000', 'C001'];
    const mIds = ['M1000', 'M1001'];
    const mqIds = ['MQ2000', 'MQ2001'];
    const idsAndScores: (readonly [string, number])[] = [
      ['P', -100],
      ['Q', -100],
      ...cIds.map((id) => [id, 1] as const),
      ...mIds.map((id) => [id, 1] as const),
      ...mqIds.map((id) => [id, 1] as const)
    ];
    const raw = graphList(idsAndScores);
    const kinds = kindsFor([
      ['P', 'P'],
      ['Q', 'Q'],
      ...cIds.map((id) => [id, 'C'] as const),
      ...mIds.map((id) => [id, 'M'] as const),
      ...mqIds.map((id) => [id, 'MQ'] as const)
    ]);
    const publishedNull: PublishedNull = { biologicalScore: 0, scores: Array.from({ length: 20 }, (_, i) => -5 + i * 0.1) };
    const stats = buildInterventionStatistics(raw, kinds, publishedNull, 42, 200);
    expect(stats.p.category).toBe('not-supported');
    expect(stats.q.channelSpecific).toBe(false);
  });

  it('throws when the raw evaluation has no biological section', () => {
    const { raw, kinds, publishedNull } = buildFixture();
    const { biological: _unused, ...withoutBiological } = raw;
    expect(() =>
      buildInterventionStatistics(withoutBiological as NullGraphListEvaluationRaw, kinds, publishedNull, 42, 200)
    ).toThrow(/no biological section/);
  });

  it('throws when a graph id has no matching kind', () => {
    const { raw, publishedNull } = buildFixture();
    const kinds = new Map<string, GraphKind>([['P', 'P']]); // missing every other id
    expect(() => buildInterventionStatistics(raw, kinds, publishedNull, 42, 200)).toThrow(/no kind found/);
  });
});
