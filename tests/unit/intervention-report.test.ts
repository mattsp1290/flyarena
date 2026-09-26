// @vitest-environment node
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  armDistribution,
  buildInterventionStatistics,
  checkBiologicalReproduction,
  DEFAULT_OUT,
  evaluateCategory,
  evaluateChannelSpecific,
  parseInterventionReportArgs,
  publishedNullFloorValue,
  readGraphListIndexInfo,
  readPublishedNull,
  runInterventionReport,
  type GraphKind,
  type GraphListIndexInfo,
  type InterventionReportArgs,
  type PublishedNull
} from '../../scripts/null/intervention-report';
import type { NullGraphListEvaluationRaw } from '../../scripts/null/null-evaluate';
import { resolveArenaTask } from '../../src/lib/arena/tasks';

/**
 * Coverage for `scripts/null/intervention-report.ts`'s statistics layer
 * (`.agents/plans/pathway-interventions/03-evaluation.md`'s WP2). Uses
 * small synthetic inputs throughout -- the expensive part (one episode per
 * seed per graph) lives in `null-evaluate.ts`/`null-worker.ts`
 * (`tests/unit/null-evaluate.test.ts` covers that); this module is pure
 * statistics over already-computed scores.
 */

const SCRATCH_INPUTS = {
  authoredSha256: 'a'.repeat(64),
  indexSha256: 'b'.repeat(64),
  publishedNullSha256: 'c'.repeat(64)
};

describe('parseInterventionReportArgs', () => {
  it('applies defaults, including allowReproductionMismatch: false', () => {
    const args = parseInterventionReportArgs([]);
    expect(args.allowReproductionMismatch).toBe(false);
    expect(args.bootstrapResamples).toBe(10000);
  });

  it('parses --allow-reproduction-mismatch as a boolean flag (no value consumed)', () => {
    const args = parseInterventionReportArgs(['--allow-reproduction-mismatch', '--authored', 'a.json']);
    expect(args.allowReproductionMismatch).toBe(true);
    expect(args.authored).toContain('a.json');
  });

  it('parses --authored/--index/--null/--out/--bootstrap-seed/--bootstrap-resamples', () => {
    const args = parseInterventionReportArgs([
      '--authored',
      'a.json',
      '--index',
      'i.json',
      '--null',
      'n.json',
      '--out',
      'o.json',
      '--bootstrap-seed',
      '7',
      '--bootstrap-resamples',
      '500'
    ]);
    expect(args.authored).toContain('a.json');
    expect(args.index).toContain('i.json');
    expect(args.publishedNull).toContain('n.json');
    expect(args.out).toContain('o.json');
    expect(args.bootstrapSeed).toBe(7);
    expect(args.bootstrapResamples).toBe(500);
  });

  it('rejects an unknown flag', () => {
    expect(() => parseInterventionReportArgs(['--bogus'])).toThrow(/Unknown argument/);
  });

  it.each(['--authored', '--index', '--null'] as const)(
    'rejects --out pointing at the same path as %s',
    (inputFlag) => {
      expect(() =>
        parseInterventionReportArgs([inputFlag, 'shared.json', '--out', 'shared.json'])
      ).toThrow(/--out must not overwrite an input file/);
    }
  );
});

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

describe('publishedNullFloorValue', () => {
  it('computes sorted(scores)[quantileIndex(n, 0.25)], matching interventions.py\'s own definition', () => {
    // n = 8: quantileIndex(8, 0.25) = floor(0.25 * 8) = 2 -> sorted[2].
    const scores = [7, 3, 1, 8, 2, 6, 5, 4]; // sorted: 1,2,3,4,5,6,7,8
    const publishedNull: PublishedNull = {
      biologicalScore: 0,
      scores,
      sourceGraphSha256: 'x'.repeat(64),
      seeds: { start: 30001, count: 100 },
      ticks: 1800,
      substeps: 4
    };
    expect(publishedNullFloorValue(publishedNull)).toBe(3);
  });
});

describe('evaluateCategory', () => {
  const cArm = armDistribution(Array.from({ length: 100 }, () => 1));
  const mArm = armDistribution(Array.from({ length: 100 }, () => 2));
  const nullFloorScore = 0;

  it('not-supported: P below the null floor value', () => {
    expect(evaluateCategory(-0.001, nullFloorScore, cArm, mArm)).toBe('not-supported');
  });

  it('at-or-above the floor: exactly at the floor value counts as reached ("at or above")', () => {
    // P (5, == floor) is <= C's p95 (1)? No: 5 > 1, so it clears C. Use a
    // floor equal to a low P score to isolate the floor boundary itself.
    expect(evaluateCategory(0.5, 0.5, cArm, mArm)).toBe('generic-rewiring-effect');
  });

  it('generic-rewiring-effect: P at/above the floor but at/below C p95', () => {
    expect(evaluateCategory(1, nullFloorScore, cArm, mArm)).toBe('generic-rewiring-effect');
    expect(evaluateCategory(0.5, 0.5, cArm, mArm)).toBe('generic-rewiring-effect');
  });

  it('edge-class-effect: P above C p95 but at/below M p95', () => {
    expect(evaluateCategory(1.5, nullFloorScore, cArm, mArm)).toBe('edge-class-effect');
  });

  it('pathway-supported: P above both C p95 and M p95', () => {
    expect(evaluateCategory(3, nullFloorScore, cArm, mArm)).toBe('pathway-supported');
  });
});

describe('evaluateChannelSpecific', () => {
  const mqArm = armDistribution(Array.from({ length: 100 }, () => 1));

  it('true when Q clears both the null floor and MQ p95 (both strict >)', () => {
    expect(evaluateChannelSpecific(2, 0.5, mqArm)).toBe(true);
  });

  it('false when Q is below the null floor value', () => {
    expect(evaluateChannelSpecific(2, 3, mqArm)).toBe(false);
  });

  it('false when Q is exactly at the null floor value (plan says "above", strict)', () => {
    expect(evaluateChannelSpecific(2, 2, mqArm)).toBe(false);
  });

  it('false when Q does not clear MQ p95', () => {
    expect(evaluateChannelSpecific(1, 0.5, mqArm)).toBe(false);
  });
});

describe('checkBiologicalReproduction', () => {
  const publishedNull = (biologicalScore: number): PublishedNull => ({
    biologicalScore,
    scores: [0, 1, 2, 3],
    sourceGraphSha256: 'x'.repeat(64),
    seeds: { start: 30001, count: 100 },
    ticks: 1800,
    substeps: 4
  });

  it('matches when the computed mean equals the published score exactly', () => {
    const check = checkBiologicalReproduction([1, 2, 3], publishedNull(2));
    expect(check.computedScore).toBe(2);
    expect(check.matches).toBe(true);
  });

  it('does not match on any deviation', () => {
    const check = checkBiologicalReproduction([1, 2, 3.0001], publishedNull(2));
    expect(check.matches).toBe(false);
  });
});

describe('readGraphListIndexInfo', () => {
  let root: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'intervention-report-info-'));
  });
  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('parses id -> {kind, gzipSha256} plus controlCount from index.json entries', () => {
    const path = join(root, 'index.json');
    writeFileSync(
      path,
      JSON.stringify({
        controlCount: 100,
        entries: [
          { id: 'P', kind: 'P', gzipSha256: 'a'.repeat(64) },
          { id: 'C000', kind: 'C', gzipSha256: 'b'.repeat(64) },
          { id: 'MQ2000', kind: 'MQ', gzipSha256: 'c'.repeat(64) }
        ]
      })
    );
    const info = readGraphListIndexInfo(path);
    expect(info.controlCount).toBe(100);
    expect(info.entries.get('P')).toEqual({ kind: 'P', gzipSha256: 'a'.repeat(64) });
    expect(info.entries.get('C000')).toEqual({ kind: 'C', gzipSha256: 'b'.repeat(64) });
    expect(info.entries.get('MQ2000')).toEqual({ kind: 'MQ', gzipSha256: 'c'.repeat(64) });
  });

  it('rejects an entry with an unrecognized kind', () => {
    const path = join(root, 'index.json');
    writeFileSync(
      path,
      JSON.stringify({ controlCount: 1, entries: [{ id: 'X', kind: 'bogus', gzipSha256: 'a'.repeat(64) }] })
    );
    expect(() => readGraphListIndexInfo(path)).toThrow(/malformed entry/);
  });

  it('rejects an entry missing gzipSha256', () => {
    const path = join(root, 'index.json');
    writeFileSync(path, JSON.stringify({ controlCount: 1, entries: [{ id: 'X', kind: 'C' }] }));
    expect(() => readGraphListIndexInfo(path)).toThrow(/malformed entry/);
  });

  it('rejects a duplicate id (even with the same kind/sha)', () => {
    const path = join(root, 'index.json');
    const entry = { id: 'X', kind: 'C', gzipSha256: 'a'.repeat(64) };
    writeFileSync(path, JSON.stringify({ controlCount: 1, entries: [entry, { ...entry }] }));
    expect(() => readGraphListIndexInfo(path)).toThrow(/more than once/);
  });

  it('rejects an empty entries array', () => {
    const path = join(root, 'index.json');
    writeFileSync(path, JSON.stringify({ controlCount: 1, entries: [] }));
    expect(() => readGraphListIndexInfo(path)).toThrow(/has no entries/);
  });

  it('rejects a missing/non-positive-integer controlCount', () => {
    const path = join(root, 'index.json');
    const entry = { id: 'X', kind: 'C', gzipSha256: 'a'.repeat(64) };
    writeFileSync(path, JSON.stringify({ entries: [entry] })); // no controlCount
    expect(() => readGraphListIndexInfo(path)).toThrow(/missing a positive integer controlCount/);

    writeFileSync(path, JSON.stringify({ controlCount: 0, entries: [entry] }));
    expect(() => readGraphListIndexInfo(path)).toThrow(/missing a positive integer controlCount/);

    writeFileSync(path, JSON.stringify({ controlCount: 1.5, entries: [entry] }));
    expect(() => readGraphListIndexInfo(path)).toThrow(/missing a positive integer controlCount/);
  });
});

describe('readPublishedNull', () => {
  const validBody = {
    biological: { score: 1.5 },
    rewired: [{ score: 1 }, { score: 2 }, { score: 3 }],
    sourceGraphSha256: 'x'.repeat(64),
    seeds: { start: 30001, count: 100 },
    ticks: 1800,
    substeps: 4
  };

  let root: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'intervention-report-null-'));
  });
  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('parses biological.score, rewired[].score, sourceGraphSha256, seeds, ticks, substeps', () => {
    const path = join(root, 'null.json');
    writeFileSync(path, JSON.stringify(validBody));
    const parsed = readPublishedNull(path);
    expect(parsed.biologicalScore).toBe(1.5);
    expect(parsed.scores).toEqual([1, 2, 3]);
    expect(parsed.sourceGraphSha256).toBe('x'.repeat(64));
    expect(parsed.seeds).toEqual({ start: 30001, count: 100 });
    expect(parsed.ticks).toBe(1800);
    expect(parsed.substeps).toBe(4);
  });

  it('throws when biological.score is missing', () => {
    const path = join(root, 'null.json');
    const { biological: _b, ...rest } = validBody;
    writeFileSync(path, JSON.stringify(rest));
    expect(() => readPublishedNull(path)).toThrow(/missing a finite biological\.score/);
  });

  it('throws when biological.score is non-finite (NaN cannot appear in real JSON, but a hand-edited file could smuggle Infinity via a string)', () => {
    const path = join(root, 'null.json');
    writeFileSync(path, JSON.stringify({ ...validBody, biological: { score: 'Infinity' } }));
    expect(() => readPublishedNull(path)).toThrow(/missing a finite biological\.score/);
  });

  it('throws when rewired is empty', () => {
    const path = join(root, 'null.json');
    writeFileSync(path, JSON.stringify({ ...validBody, rewired: [] }));
    expect(() => readPublishedNull(path)).toThrow(/no rewired entries/);
  });

  it('throws when a rewired score is not a finite number', () => {
    const path = join(root, 'null.json');
    writeFileSync(path, JSON.stringify({ ...validBody, rewired: [{ score: 'nope' }] }));
    expect(() => readPublishedNull(path)).toThrow(/not a finite number/);
  });

  it('throws when sourceGraphSha256 is missing', () => {
    const path = join(root, 'null.json');
    const { sourceGraphSha256: _s, ...rest } = validBody;
    writeFileSync(path, JSON.stringify(rest));
    expect(() => readPublishedNull(path)).toThrow(/missing sourceGraphSha256/);
  });

  it('throws when seeds is missing', () => {
    const path = join(root, 'null.json');
    const { seeds: _s, ...rest } = validBody;
    writeFileSync(path, JSON.stringify(rest));
    expect(() => readPublishedNull(path)).toThrow(/missing seeds\.start\/seeds\.count/);
  });

  it('throws when ticks is missing', () => {
    const path = join(root, 'null.json');
    const { ticks: _t, ...rest } = validBody;
    writeFileSync(path, JSON.stringify(rest));
    expect(() => readPublishedNull(path)).toThrow(/missing ticks/);
  });

  it('throws when substeps is missing', () => {
    const path = join(root, 'null.json');
    const { substeps: _s, ...rest } = validBody;
    writeFileSync(path, JSON.stringify(rest));
    expect(() => readPublishedNull(path)).toThrow(/missing substeps/);
  });
});

describe('buildInterventionStatistics: end-to-end on synthetic data', () => {
  const seeds = [30001, 30002, 30003];
  const constantMovementScore = (value: number): readonly number[] => seeds.map(() => value);
  const SOURCE_SHA = 'x'.repeat(64);

  const graphList = (idsAndScores: readonly (readonly [string, number])[]): NullGraphListEvaluationRaw => ({
    version: 1,
    sourceGraphSha256: SOURCE_SHA,
    seeds: { start: 30001, count: seeds.length },
    ticks: 20,
    substeps: 4,
    decoder: 'authored',
    biological: {
      heldOutSeeds: seeds,
      movementScore: constantMovementScore(0),
      foodPickups: [0, 0, 0],
      hazardContacts: [0, 0, 0]
    },
    disconnected: {
      heldOutSeeds: seeds,
      movementScore: constantMovementScore(-1),
      foodPickups: [0, 0, 0],
      hazardContacts: [0, 0, 0]
    },
    graphs: idsAndScores.map(([id, score]) => ({
      id,
      gzipSha256: `gz-${id}`.padEnd(64, '0'),
      binarySha256: 'z'.repeat(64),
      heldOutSeeds: seeds,
      movementScore: constantMovementScore(score),
      foodPickups: [0, 0, 0],
      hazardContacts: [0, 0, 0]
    })),
    host: { arch: 'arm64', node: 'v22.0.0' },
    evaluatorGitRev: null
  });

  const infoFor = (
    idsAndKinds: readonly (readonly [string, GraphKind])[],
    controlCount: number
  ): GraphListIndexInfo => ({
    entries: new Map(idsAndKinds.map(([id, kind]) => [id, { kind, gzipSha256: `gz-${id}`.padEnd(64, '0') }])),
    controlCount
  });

  const nullFor = (scores: readonly number[]): PublishedNull => ({
    biologicalScore: 0,
    scores,
    sourceGraphSha256: SOURCE_SHA,
    seeds: { start: 30001, count: seeds.length },
    ticks: 20,
    substeps: 4
  });

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
    const info = infoFor(
      [
        ['P', 'P'],
        ['Q', 'Q'],
        ...cIds.map((id) => [id, 'C'] as const),
        ...mIds.map((id) => [id, 'M'] as const),
        ...mqIds.map((id) => [id, 'MQ'] as const)
      ],
      4
    );
    // A published null of 20 values, all well below P/Q's scores, so both
    // land above the 25th-percentile floor value comfortably.
    const publishedNull = nullFor(Array.from({ length: 20 }, (_, i) => -5 + i * 0.1));
    return { raw, info, publishedNull };
  };

  it('P is pathway-supported and Q is channel-specific under these scores', () => {
    const { raw, info, publishedNull } = buildFixture();
    const stats = buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS);

    expect(stats.biologicalReproduction.matches).toBe(true);
    expect(stats.inputs).toEqual(SCRATCH_INPUTS);
    expect(stats.graphs).toHaveLength(14); // P, Q, 4 C, 4 M, 4 MQ
    expect(stats.graphs.map((g) => g.id)).toEqual([...stats.graphs.map((g) => g.id)].sort());

    expect(stats.controls.C.n).toBe(4);
    expect(stats.controls.C.p95).toBe(1);
    expect(stats.controls.M.n).toBe(4);
    expect(stats.controls.M.p95).toBe(2);
    expect(stats.controls.MQ.n).toBe(4);
    expect(stats.controls.MQ.p95).toBe(1);

    expect(stats.publishedNullFloor).toBeLessThan(10);
    expect(stats.p.score).toBe(10);
    expect(stats.p.category).toBe('pathway-supported');
    // n=4 controls, P above all of them: pHigh = (4 - 4 + 1)/(4+1) = 1/5.
    expect(stats.p.pRankAmongC).toBe(1 / 5);
    expect(stats.p.pRankAmongM).toBe(1 / 5);

    expect(stats.q.score).toBe(8);
    expect(stats.q.channelSpecific).toBe(true);
    expect(stats.q.qRankAmongMQ).toBe(1 / 5);
  });

  it('every non-biological graph carries a pairedVsBiological difference', () => {
    const { raw, info, publishedNull } = buildFixture();
    const stats = buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS);
    for (const graph of stats.graphs) {
      expect(graph.pairedVsBiological).toBeDefined();
      expect(graph.pairedVsBiological?.n).toBe(seeds.length);
    }
  });

  it('running the statistics twice on the same input is byte-identical (JSON.stringify)', () => {
    const { raw, info, publishedNull } = buildFixture();
    const first = buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS);
    const second = buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS);
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
    const info = infoFor(
      [
        ['P', 'P'],
        ['Q', 'Q'],
        ...cIds.map((id) => [id, 'C'] as const),
        ...mIds.map((id) => [id, 'M'] as const),
        ...mqIds.map((id) => [id, 'MQ'] as const)
      ],
      2
    );
    const publishedNull = nullFor(Array.from({ length: 20 }, (_, i) => -5 + i * 0.1));
    const stats = buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS);
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
    const info = infoFor(
      [
        ['P', 'P'],
        ['Q', 'Q'],
        ...cIds.map((id) => [id, 'C'] as const),
        ...mIds.map((id) => [id, 'M'] as const),
        ...mqIds.map((id) => [id, 'MQ'] as const)
      ],
      2
    );
    const publishedNull = nullFor(Array.from({ length: 20 }, (_, i) => -5 + i * 0.1));
    const stats = buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS);
    expect(stats.p.category).toBe('not-supported');
    expect(stats.q.channelSpecific).toBe(false);
  });

  it('throws when the raw evaluation has no biological section', () => {
    const { raw, info, publishedNull } = buildFixture();
    const { biological: _unused, ...withoutBiological } = raw;
    expect(() =>
      buildInterventionStatistics(
        withoutBiological as NullGraphListEvaluationRaw,
        info,
        publishedNull,
        42,
        200,
        SCRATCH_INPUTS
      )
    ).toThrow(/no biological section/);
  });

  it('throws when the index lists an id that authored.json has no entry for', () => {
    const { raw, publishedNull } = buildFixture();
    // Every real id from buildFixture, plus one the index expects but
    // authored.json never scored.
    const info = infoFor(
      [
        ['P', 'P'],
        ['Q', 'Q'],
        ['C000', 'C'],
        ['C001', 'C'],
        ['C002', 'C'],
        ['C003', 'C'],
        ['M1000', 'M'],
        ['M1001', 'M'],
        ['M1002', 'M'],
        ['M1003', 'M'],
        ['MQ2000', 'MQ'],
        ['MQ2001', 'MQ'],
        ['MQ2002', 'MQ'],
        ['MQ2003', 'MQ'],
        ['MQ2004', 'MQ'] // authored.json has no entry for this one
      ],
      4
    );
    expect(() => buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS)).toThrow(
      /coverage mismatch.*missing 1 index id\(s\): MQ2004/
    );
  });

  it('throws when authored.json has an id the index has no entry for (the reverse-coverage direction)', () => {
    const { raw, publishedNull } = buildFixture();
    const info = infoFor([['P', 'P']], 1); // authored.json's other 13 ids are absent from the index
    expect(() => buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS)).toThrow(
      /coverage mismatch.*id\(s\) not in index\.json/
    );
  });

  it('throws when raw.decoder is not "authored"', () => {
    const { raw, info, publishedNull } = buildFixture();
    const wrongDecoder = { ...raw, decoder: 'authored-flip-both' as const };
    expect(() => buildInterventionStatistics(wrongDecoder, info, publishedNull, 42, 200, SCRATCH_INPUTS)).toThrow(
      /authored-decoder only/
    );
  });

  it('throws when authored.json and the published null were scored against different source graphs', () => {
    const { raw, info, publishedNull } = buildFixture();
    const mismatched = { ...publishedNull, sourceGraphSha256: 'y'.repeat(64) };
    expect(() => buildInterventionStatistics(raw, info, mismatched, 42, 200, SCRATCH_INPUTS)).toThrow(
      /different biological graphs/
    );
  });

  it('throws when authored.json seeds differ from the published null', () => {
    const { raw, info, publishedNull } = buildFixture();
    const mismatched = { ...publishedNull, seeds: { start: 1, count: seeds.length } };
    expect(() => buildInterventionStatistics(raw, info, mismatched, 42, 200, SCRATCH_INPUTS)).toThrow(
      /seeds\/ticks\/substeps differ/
    );
  });

  it('throws when authored.json ticks differ from the published null', () => {
    const { raw, info, publishedNull } = buildFixture();
    const mismatched = { ...publishedNull, ticks: 9999 };
    expect(() => buildInterventionStatistics(raw, info, mismatched, 42, 200, SCRATCH_INPUTS)).toThrow(
      /seeds\/ticks\/substeps differ/
    );
  });

  it('throws when authored.json has a duplicate graph id', () => {
    const { raw, info, publishedNull } = buildFixture();
    const dup = { ...raw, graphs: [...raw.graphs, raw.graphs[0]] };
    expect(() => buildInterventionStatistics(dup, info, publishedNull, 42, 200, SCRATCH_INPUTS)).toThrow(
      /duplicate graph id/
    );
  });

  it('throws when a graph\'s gzipSha256 does not match the index (stale authored.json)', () => {
    const { raw, info, publishedNull } = buildFixture();
    const staleGraphs = raw.graphs.map((g, i) => (i === 0 ? { ...g, gzipSha256: 'stale'.padEnd(64, '0') } : g));
    const stale = { ...raw, graphs: staleGraphs };
    expect(() => buildInterventionStatistics(stale, info, publishedNull, 42, 200, SCRATCH_INPUTS)).toThrow(
      /different graph file than index\.json/
    );
  });

  it('throws when a graph was scored on different held-out seeds than biological', () => {
    const { raw, info, publishedNull } = buildFixture();
    const misaligned = raw.graphs.map((g, i) => (i === 0 ? { ...g, heldOutSeeds: [1, 2, 3] } : g));
    const bad = { ...raw, graphs: misaligned };
    expect(() => buildInterventionStatistics(bad, info, publishedNull, 42, 200, SCRATCH_INPUTS)).toThrow(
      /different held-out seeds than biological/
    );
  });

  it('throws when a graph has a non-finite movementScore', () => {
    const { raw, info, publishedNull } = buildFixture();
    const badGraphs = raw.graphs.map((g, i) => (i === 0 ? { ...g, movementScore: [NaN, 0, 0] } : g));
    const bad = { ...raw, graphs: badGraphs };
    expect(() => buildInterventionStatistics(bad, info, publishedNull, 42, 200, SCRATCH_INPUTS)).toThrow(
      /is not a finite number/
    );
  });

  it('throws when there is not exactly one graph of kind P (zero found)', () => {
    const { raw, publishedNull } = buildFixture();
    const cIds = ['C000', 'C001', 'C002', 'C003'];
    const mIds = ['M1000', 'M1001', 'M1002', 'M1003'];
    const mqIds = ['MQ2000', 'MQ2001', 'MQ2002', 'MQ2003'];
    // Mislabel P's kind as Q (not C/M/MQ, so the controlCount arm-size check
    // above is untouched and this test isolates exactlyOneOfKind itself):
    // zero graphs now have kind 'P'.
    const info = infoFor(
      [
        ['P', 'Q'],
        ['Q', 'Q'],
        ...cIds.map((id) => [id, 'C'] as const),
        ...mIds.map((id) => [id, 'M'] as const),
        ...mqIds.map((id) => [id, 'MQ'] as const)
      ],
      4
    );
    expect(() => buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS)).toThrow(
      /expected exactly one graph of kind "P", found 0/
    );
  });

  it('throws when there is not exactly one graph of kind Q (two found, P still correct)', () => {
    const { publishedNull } = buildFixture();
    const cIds = ['C000', 'C001', 'C002', 'C003'];
    const mIds = ['M1000', 'M1001', 'M1002', 'M1003'];
    const mqIds = ['MQ2000', 'MQ2001', 'MQ2002', 'MQ2003'];
    // An extra "Q2" id, also kind 'Q' -- P and the C/M/MQ arms are all
    // correctly sized (controlCount still holds), isolating exactlyOneOfKind's
    // "found 2" branch for Q specifically.
    const idsAndScores: (readonly [string, number])[] = [
      ['P', 10],
      ['Q', 8],
      ['Q2', 8],
      ...cIds.map((id) => [id, 1] as const),
      ...mIds.map((id) => [id, 2] as const),
      ...mqIds.map((id) => [id, 1] as const)
    ];
    const raw = graphList(idsAndScores);
    const info = infoFor(
      [
        ['P', 'P'],
        ['Q', 'Q'],
        ['Q2', 'Q'],
        ...cIds.map((id) => [id, 'C'] as const),
        ...mIds.map((id) => [id, 'M'] as const),
        ...mqIds.map((id) => [id, 'MQ'] as const)
      ],
      4
    );
    expect(() => buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS)).toThrow(
      /expected exactly one graph of kind "Q", found 2/
    );
  });

  it('reports the specific P-count error (not a masking arm-size error) when P is mislabeled into the C arm', () => {
    // Regression test: assertConsistentInputs checks P/Q counts *before* the
    // C/M/MQ controlCount loop specifically so this case (the most likely
    // hand-edit/corruption: the real "P" id relabeled `kind: 'C'`) reports
    // "expected exactly one graph of kind P, found 0" rather than the
    // technically-true-but-less-helpful "arm C has 5 graph(s), but
    // index.json declares controlCount=4" (a dual-review finding).
    const { raw, publishedNull } = buildFixture();
    const cIds = ['C000', 'C001', 'C002', 'C003'];
    const mIds = ['M1000', 'M1001', 'M1002', 'M1003'];
    const mqIds = ['MQ2000', 'MQ2001', 'MQ2002', 'MQ2003'];
    const info = infoFor(
      [
        ['P', 'C'],
        ['Q', 'Q'],
        ...cIds.map((id) => [id, 'C'] as const),
        ...mIds.map((id) => [id, 'M'] as const),
        ...mqIds.map((id) => [id, 'MQ'] as const)
      ],
      4
    );
    expect(() => buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS)).toThrow(
      /expected exactly one graph of kind "P", found 0/
    );
  });

  it('throws when the kind-"P" graph\'s id is not literally "P" (a mislabeled index passes the "exactly one" check but not this one)', () => {
    const { raw, publishedNull } = buildFixture();
    const cIds = ['C000', 'C001', 'C002', 'C003'];
    const mIds = ['M1000', 'M1001', 'M1002', 'M1003'];
    const mqIds = ['MQ2000', 'MQ2001', 'MQ2002', 'MQ2003'];
    // Swap P's and C000's kinds: exactly one graph has kind 'P' (id "C000")
    // and the arms are still the right size, so exactlyOneOfKind alone would
    // not catch this -- only the id === 'P' assertion does.
    const info = infoFor(
      [
        ['P', 'C'],
        ['C000', 'P'],
        ['Q', 'Q'],
        ...cIds.slice(1).map((id) => [id, 'C'] as const),
        ...mIds.map((id) => [id, 'M'] as const),
        ...mqIds.map((id) => [id, 'MQ'] as const)
      ],
      4
    );
    expect(() => buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS)).toThrow(
      /kind "P"\/"Q" have id\(s\) "C000"\/"Q"/
    );
  });

  it('throws when an arm has fewer graphs than the index\'s own declared controlCount', () => {
    const { publishedNull } = buildFixture();
    const cIds = ['C000', 'C001', 'C002', 'C003'];
    const mIds = ['M1000', 'M1001', 'M1002']; // one short of controlCount
    const mqIds = ['MQ2000', 'MQ2001', 'MQ2002', 'MQ2003'];
    const idsAndScores: (readonly [string, number])[] = [
      ['P', 10],
      ['Q', 8],
      ...cIds.map((id) => [id, 1] as const),
      ...mIds.map((id) => [id, 2] as const),
      ...mqIds.map((id) => [id, 1] as const)
    ];
    const raw = graphList(idsAndScores);
    const info = infoFor(
      [
        ['P', 'P'],
        ['Q', 'Q'],
        ...cIds.map((id) => [id, 'C'] as const),
        ...mIds.map((id) => [id, 'M'] as const),
        ...mqIds.map((id) => [id, 'MQ'] as const)
      ],
      4 // declared controlCount, but M only has 3 entries above
    );
    expect(() => buildInterventionStatistics(raw, info, publishedNull, 42, 200, SCRATCH_INPUTS)).toThrow(
      /arm M has 3 graph\(s\), but index\.json declares controlCount=4/
    );
  });
});

describe('runInterventionReport (CLI layer)', () => {
  const seeds = [30001, 30002, 30003];
  const SOURCE_SHA = 'x'.repeat(64);
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'intervention-report-cli-'));
  });

  afterEach(() => {
    // Exception-safe cleanup (a thermo-maintainability review finding,
    // flagged in rounds 2 and 3): a trailing `rmSync` at the end of each
    // test body skips cleanup whenever an assertion inside that body fails
    // (not just when the code under test throws), leaking the temp
    // directory into `$TMPDIR`. `afterEach` always runs, pass or fail.
    rmSync(root, { recursive: true, force: true });
  });

  const writeFixtureFiles = (biologicalScore: number, publishedBiologicalScore: number) => {
    const graphs = [
      { id: 'P', kind: 'P', score: 10 },
      { id: 'Q', kind: 'Q', score: 8 },
      { id: 'C000', kind: 'C', score: 1 },
      { id: 'M1000', kind: 'M', score: 2 },
      { id: 'MQ2000', kind: 'MQ', score: 1 }
    ];
    const authored: NullGraphListEvaluationRaw = {
      version: 1,
      sourceGraphSha256: SOURCE_SHA,
      seeds: { start: 30001, count: seeds.length },
      ticks: 20,
      substeps: 4,
      decoder: 'authored',
      biological: {
        heldOutSeeds: seeds,
        movementScore: seeds.map(() => biologicalScore),
        foodPickups: [0, 0, 0],
        hazardContacts: [0, 0, 0]
      },
      graphs: graphs.map((g) => ({
        id: g.id,
        gzipSha256: `gz-${g.id}`.padEnd(64, '0'),
        binarySha256: 'z'.repeat(64),
        heldOutSeeds: seeds,
        movementScore: seeds.map(() => g.score),
        foodPickups: [0, 0, 0],
        hazardContacts: [0, 0, 0]
      })),
      host: { arch: 'arm64', node: 'v22.0.0' },
      evaluatorGitRev: null
    };
    writeFileSync(join(root, 'authored.json'), JSON.stringify(authored));

    const index = {
      sourceArtifact: 'src.bin.gz',
      sourceSha256: SOURCE_SHA,
      controlCount: 1,
      entries: graphs.map((g) => ({ id: g.id, kind: g.kind, gzipSha256: `gz-${g.id}`.padEnd(64, '0') }))
    };
    writeFileSync(join(root, 'index.json'), JSON.stringify(index));

    const publishedNull = {
      biological: { score: publishedBiologicalScore },
      rewired: Array.from({ length: 20 }, (_, i) => ({ score: -5 + i * 0.1 })),
      sourceGraphSha256: SOURCE_SHA,
      seeds: { start: 30001, count: seeds.length },
      ticks: 20,
      substeps: 4
    };
    writeFileSync(join(root, 'null.json'), JSON.stringify(publishedNull));
  };

  const argsFor = (overrides: Partial<InterventionReportArgs> = {}): InterventionReportArgs => ({
    authored: join(root, 'authored.json'),
    index: join(root, 'index.json'),
    publishedNull: join(root, 'null.json'),
    out: join(root, 'nested', 'statistics.json'),
    bootstrapSeed: 42,
    bootstrapResamples: 200,
    allowReproductionMismatch: false,
    ...overrides
  });

  it('creates the output directory if it does not exist (mkdir before write)', () => {
    writeFixtureFiles(0, 0);
    const { out } = runInterventionReport(argsFor());
    expect(JSON.parse(readFileSync(out, 'utf8')).biologicalReproduction.matches).toBe(true);
  });

  it('records sha256 of each input file under statistics.inputs', () => {
    writeFixtureFiles(0, 0);
    const { statistics } = runInterventionReport(argsFor());
    expect(statistics.inputs.authoredSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(statistics.inputs.indexSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(statistics.inputs.publishedNullSha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it('refuses to write when the biological reproduction check fails (fail-closed by default)', () => {
    writeFixtureFiles(0, 999); // computed 0 != published 999
    const args = argsFor();
    expect(() => runInterventionReport(args)).toThrow(/biological reproduction check failed/);
    expect(existsSync(args.out)).toBe(false);
  });

  it('writes anyway when --allow-reproduction-mismatch is set, for diagnosis, and marks the output diagnosticOnly', () => {
    writeFixtureFiles(0, 999);
    const { statistics } = runInterventionReport(argsFor({ allowReproductionMismatch: true }));
    expect(statistics.biologicalReproduction.matches).toBe(false);
    expect(statistics.diagnosticOnly).toBe(true);
  });

  it('does not mark diagnosticOnly when --allow-reproduction-mismatch is passed but the check actually passes', () => {
    writeFixtureFiles(0, 0); // matches
    const { statistics } = runInterventionReport(argsFor({ allowReproductionMismatch: true }));
    expect(statistics.biologicalReproduction.matches).toBe(true);
    expect(statistics.diagnosticOnly).toBeUndefined();
  });

  it('refuses to write a diagnosticOnly result to the default --out', () => {
    writeFixtureFiles(0, 999); // reproduction fails
    const args = { ...argsFor({ allowReproductionMismatch: true }), out: DEFAULT_OUT };
    expect(() => runInterventionReport(args)).toThrow(/refusing to write a diagnosticOnly result to the default --out/);
  });

  it('does not block a clean (non-diagnosticOnly) run targeting the default --out for an unrelated reason', () => {
    // Proves the guard is scoped to diagnosticOnly specifically -- a plain
    // run at the default --out is legitimate and must not be blocked by
    // this check. It still fails here, for the unrelated, expected reason
    // that --authored/--index/--null don't exist on disk (writeFixtureFiles
    // was never called in this test).
    const args = { ...argsFor(), out: DEFAULT_OUT };
    expect(() => runInterventionReport(args)).not.toThrow(/refusing to write a diagnosticOnly result/);
  });

  describe('--arena-task (task-generality WP1)', () => {
    const patchAuthoredArenaTask = (id: string): void => {
      const path = join(root, 'authored.json');
      const authored = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>;
      authored.arenaTask = id;
      authored.arenaTaskFingerprint = resolveArenaTask(id).fingerprint;
      writeFileSync(path, JSON.stringify(authored));
    };

    it('labels the output when --arena-task matches authored.json\'s recorded task', () => {
      writeFixtureFiles(0, 0);
      patchAuthoredArenaTask('hazard-heavy');
      const { statistics } = runInterventionReport(argsFor({ arenaTask: 'hazard-heavy' }));
      expect(statistics.arenaTask).toEqual({ id: 'hazard-heavy', fingerprint: resolveArenaTask('hazard-heavy').fingerprint });
    });

    it('throws when --arena-task disagrees with authored.json\'s recorded task', () => {
      writeFixtureFiles(0, 0);
      patchAuthoredArenaTask('hazard-heavy');
      expect(() => runInterventionReport(argsFor({ arenaTask: 'crowded' }))).toThrow(
        /does not match authored\.json's recorded arena task fingerprint/
      );
    });

    it('throws when --arena-task is passed but authored.json was scored under the default task', () => {
      writeFixtureFiles(0, 0);
      expect(() => runInterventionReport(argsFor({ arenaTask: 'hazard-heavy' }))).toThrow(
        /does not match authored\.json's recorded arena task fingerprint/
      );
    });

    it('does not add an arenaTask key when --arena-task is omitted and authored.json is the default task (byte-identity gate)', () => {
      writeFixtureFiles(0, 0);
      const { statistics } = runInterventionReport(argsFor());
      expect(statistics.arenaTask).toBeUndefined();
      expect(JSON.stringify(statistics)).not.toContain('arenaTask');
    });

    it('refuses to write a non-default arena-task result to the default --out', () => {
      writeFixtureFiles(0, 0);
      patchAuthoredArenaTask('hazard-heavy');
      const args = { ...argsFor({ arenaTask: 'hazard-heavy' }), out: DEFAULT_OUT };
      expect(() => runInterventionReport(args)).toThrow(/refusing to write a --arena-task "hazard-heavy" result/);
    });
  });
});
