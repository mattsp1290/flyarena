// @vitest-environment node
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gunzipSync, gzipSync } from 'node:zlib';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';

import { encodeGraphBinary, parseGraphBinary } from '../../src/lib/connectome/format';
import { runEpisode } from '../../scripts/training/episode';
import { createTraceGraph } from '../fixtures/trace-graph';
import { createFixtureRewiredTraceGraph } from '../fixtures/trace-graph-rewire';
import {
  DEFAULT_HELD_OUT_COUNT,
  DEFAULT_HELD_OUT_START,
  DEFAULT_TICKS,
  buildTasks,
  parseAtlasEvaluateArgs,
  type AtlasEvaluateArgs,
  type AtlasGraphKey,
  type GraphSpec
} from '../../scripts/lesion/atlas-evaluate';
import {
  DEFAULT_MANIFEST,
  DEFAULT_OUT,
  DEFAULT_REPORT_MD,
  benjaminiHochbergSignificant,
  buildArtifact,
  pairedBootstrapPValue,
  parseAtlasReportArgs,
  resolveRunMeta,
  roundSignificant,
  runAtlasReport,
  type AtlasReportArgs
} from '../../scripts/lesion/atlas-report';
import { conditionRng } from '../../scripts/training/stats';
import type { AtlasEvaluationRaw, AtlasGraphRaw } from '../../scripts/lesion/atlas-evaluate';

/**
 * Coverage for WP2 (`.agents/plans/lesion-atlas/02-atlas-computation.md`):
 * report statistics on synthetic raw input (no episode simulation), and
 * shard determinism (`--shards 1` vs `--shards 3`) on the trace-graph
 * fixture with 4 lesions and 3 seeds -- the plan's own stated test.
 */

const sha256Hex = (data: Uint8Array): string => createHash('sha256').update(data).digest('hex');

// ---------------------------------------------------------------------------
// parseAtlasEvaluateArgs / buildTasks
// ---------------------------------------------------------------------------

describe('parseAtlasEvaluateArgs', () => {
  it('applies defaults: both graphs, plan-specified numeric defaults', () => {
    const args = parseAtlasEvaluateArgs([]);
    expect(args.graphs).toEqual(['biological', 'rewiredSeed0']);
    expect(args.heldOutStart).toBe(30001);
    expect(args.heldOutCount).toBe(100);
    expect(args.ticks).toBe(1800);
    expect(args.shards).toBe(18);
    expect(args.maxLesions).toBeUndefined();
  });

  it('parses --graphs biological,rewired-seed0 in either order', () => {
    expect(parseAtlasEvaluateArgs(['--graphs', 'rewired-seed0,biological']).graphs).toEqual([
      'rewiredSeed0',
      'biological'
    ]);
    expect(parseAtlasEvaluateArgs(['--graphs', 'biological']).graphs).toEqual(['biological']);
  });

  it('rejects an unknown graph name', () => {
    expect(() => parseAtlasEvaluateArgs(['--graphs', 'bogus'])).toThrow(/unknown graph/);
  });

  it('rejects a duplicate graph name', () => {
    expect(() => parseAtlasEvaluateArgs(['--graphs', 'biological,biological'])).toThrow(/listed more than once/);
  });

  it('rejects an unknown flag', () => {
    expect(() => parseAtlasEvaluateArgs(['--bogus'])).toThrow(/Unknown argument/);
  });

  it('rejects --out without a .json extension', () => {
    expect(() => parseAtlasEvaluateArgs(['--out', 'atlas'])).toThrow(/--out must end with "\.json"/);
  });

  it('parses --max-lesions', () => {
    expect(parseAtlasEvaluateArgs(['--max-lesions', '10']).maxLesions).toBe(10);
  });
});

describe('buildTasks', () => {
  const baseArgs: AtlasEvaluateArgs = {
    graphs: ['biological', 'rewiredSeed0'],
    manifest: 'manifest.json',
    heldOutStart: 30001,
    heldOutCount: 3,
    ticks: 20,
    shards: 1,
    out: 'out.json'
  };
  const specs = new Map<AtlasGraphKey, GraphSpec>([
    ['biological', { key: 'biological', path: 'bio.bin.gz', expectedSha256: 'a'.repeat(64), gzipSha256: 'c'.repeat(64) }],
    ['rewiredSeed0', { key: 'rewiredSeed0', path: 'rewired.bin.gz', expectedSha256: 'b'.repeat(64), gzipSha256: 'd'.repeat(64) }]
  ]);

  it('produces baseline-then-ascending-index tasks per graph, in canonical (graph, index) order', () => {
    const tasks = buildTasks(baseArgs, specs, 4);
    expect(tasks.map((t) => t.graphId)).toEqual([
      'biological|baseline',
      'biological|0',
      'biological|1',
      'biological|2',
      'biological|3',
      'rewiredSeed0|baseline',
      'rewiredSeed0|0',
      'rewiredSeed0|1',
      'rewiredSeed0|2',
      'rewiredSeed0|3'
    ]);
    expect(tasks.every((t) => t.heldOutSeeds.length === 3)).toBe(true);
  });

  it('honors --max-lesions without ever skipping the baseline task', () => {
    const tasks = buildTasks({ ...baseArgs, maxLesions: 2 }, specs, 4);
    expect(tasks.map((t) => t.graphId)).toEqual([
      'biological|baseline',
      'biological|0',
      'biological|1',
      'rewiredSeed0|baseline',
      'rewiredSeed0|0',
      'rewiredSeed0|1'
    ]);
  });

  it('only builds tasks for the requested graphs', () => {
    const tasks = buildTasks({ ...baseArgs, graphs: ['biological'] }, specs, 2);
    expect(tasks.every((t) => t.graphId.startsWith('biological|'))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Shard determinism: --shards 1 vs --shards 3, trace-graph fixture, 4
// lesions, 3 seeds (the plan's own stated gate).
// ---------------------------------------------------------------------------

describe('atlas-evaluate CLI: shard determinism (trace-graph fixture)', () => {
  let root: string;
  let manifestPath: string;
  let bioGraph: ReturnType<typeof createTraceGraph>;
  let rewiredGraph: ReturnType<typeof createTraceGraph>;
  let bioBinarySha256: string;
  let rewiredBinarySha256: string;

  const HELD_OUT_START = 30001;
  const HELD_OUT_COUNT = 3;
  const TICKS = 20;
  const MAX_LESIONS = 4;

  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'atlas-evaluate-fixture-'));

    const bioGraphOriginal = createTraceGraph();
    const bioBinary = Buffer.from(encodeGraphBinary(bioGraphOriginal));
    bioBinarySha256 = sha256Hex(bioBinary);
    const bioGzip = gzipSync(bioBinary);
    writeFileSync(join(root, 'malecns-arena-v1.bin.gz'), bioGzip);

    const rewiredGraphOriginal = createFixtureRewiredTraceGraph(bioGraphOriginal, 0);
    const rewiredBinary = Buffer.from(encodeGraphBinary(rewiredGraphOriginal));
    rewiredBinarySha256 = sha256Hex(rewiredBinary);
    const rewiredGzip = gzipSync(rewiredBinary);
    writeFileSync(join(root, 'malecns-arena-v1-rewired-seed0.bin.gz'), rewiredGzip);

    // Parsed back from the exact gzip bytes on disk, not the original
    // in-memory objects: `GraphMetadata`'s dynamics-relevant floats
    // (timestepSeconds/leakRate/rateMin/rateMax/inputClampMin/Max/globalGain)
    // round-trip through the wire format's Float32 encoding
    // (`format.ts`'s `setFloat32`/`getFloat32`), which is lossy relative to
    // the full-precision JS number literals `createTraceGraph()` starts
    // from -- confirmed empirically: `runEpisode` on the original in-memory
    // graph and on this round-tripped graph differ by ~1e-7 relative, not
    // bit-for-bit, even though every typed array (`contactMagnitudes`,
    // `postsynapticIndices`, etc.) is element-wise equal between the two.
    // `atlas-worker.ts` only ever sees the round-tripped version (it parses
    // the gzip file from disk, the same as the real CLI/production path),
    // so the cross-check below must use these, not the originals, or every
    // assertion would fail on a difference that has nothing to do with
    // whether the lesion was applied correctly.
    bioGraph = parseGraphBinary(gunzipSync(bioGzip).buffer.slice(0));
    rewiredGraph = parseGraphBinary(gunzipSync(rewiredGzip).buffer.slice(0));

    const manifest = {
      artifact: 'malecns-arena-v1.bin.gz',
      binarySha256: bioBinarySha256,
      gzipSha256: sha256Hex(bioGzip),
      neuronCount: bioGraph.metadata.neuronCount,
      rewiredArms: {
        seed0: {
          artifact: 'malecns-arena-v1-rewired-seed0.bin.gz',
          binarySha256: rewiredBinarySha256,
          gzipSha256: sha256Hex(rewiredGzip)
        }
      }
    };
    manifestPath = join(root, 'malecns-arena-v1.manifest.json');
    writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  });

  afterAll(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  const runCli = (shards: number, out: string) =>
    spawnSync(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/lesion/atlas-evaluate.ts',
        '--manifest',
        manifestPath,
        '--held-out-start',
        String(HELD_OUT_START),
        '--held-out-count',
        String(HELD_OUT_COUNT),
        '--ticks',
        String(TICKS),
        '--max-lesions',
        String(MAX_LESIONS),
        '--shards',
        String(shards),
        '--out',
        out
      ],
      { encoding: 'utf8', timeout: 60_000 }
    );

  it('--shards 1 and --shards 3 produce byte-identical atlas-raw.json', () => {
    const out1 = join(root, 'atlas-raw-shards1.json');
    const out3 = join(root, 'atlas-raw-shards3.json');

    const result1 = runCli(1, out1);
    expect(result1.status, result1.stderr).toBe(0);

    const result3 = runCli(3, out3);
    expect(result3.status, result3.stderr).toBe(0);

    const bytes1 = readFileSync(out1);
    const bytes3 = readFileSync(out3);
    expect(bytes3.equals(bytes1)).toBe(true);

    const parsed = JSON.parse(bytes1.toString('utf8')) as AtlasEvaluationRaw;
    expect(parsed.version).toBe(1);
    expect(parsed.ticks).toBe(TICKS);
    expect(parsed.seeds).toEqual({ start: HELD_OUT_START, count: HELD_OUT_COUNT });

    const biological = parsed.graphs.biological as AtlasGraphRaw;
    const rewiredSeed0 = parsed.graphs.rewiredSeed0 as AtlasGraphRaw;

    // Each graph really did load its own file, not e.g. both loading the
    // biological file: sha256 matches the manifest, and the two graphs'
    // baselines (same seeds, different topology) actually differ.
    expect(biological.graphSha256).toBe(bioBinarySha256);
    expect(rewiredSeed0.graphSha256).toBe(rewiredBinarySha256);
    expect(rewiredSeed0.baselineMovementScore).not.toEqual(biological.baselineMovementScore);

    for (const graph of [biological, rewiredSeed0]) {
      expect(graph.baselineMovementScore).toHaveLength(HELD_OUT_COUNT);
      expect(graph.lesion).toHaveLength(MAX_LESIONS);
      expect(graph.lesion.map((l) => l.index)).toEqual([0, 1, 2, 3]);
      for (const entry of graph.lesion) {
        expect(entry.movementScore).toHaveLength(HELD_OUT_COUNT);
        for (const score of entry.movementScore) expect(Number.isFinite(score)).toBe(true);
      }
    }

    // Pin the IPC path to an in-process reference (a dual-review finding:
    // byte-identity across shard counts proves determinism, but says
    // nothing about whether atlas-worker.ts actually applied the specific
    // requested lesion index rather than, say, ignoring `lesionIndex` or
    // always lesioning index 0). runEpisode with the same graph/seed/lesion
    // must reproduce the CLI's own per-seed movementScore exactly.
    const heldOutSeeds = Array.from({ length: HELD_OUT_COUNT }, (_, i) => HELD_OUT_START + i);
    for (const [graphRaw, graph] of [
      [biological, bioGraph],
      [rewiredSeed0, rewiredGraph]
    ] as const) {
      for (const index of [0, 3]) {
        const expected = heldOutSeeds.map(
          (seed) =>
            runEpisode({
              seed,
              ticks: TICKS,
              left: { decoder: 'authored', graph, lesion: Int32Array.of(index) },
              right: { decoder: 'parked' }
            }).left.movementScore
        );
        expect(graphRaw.lesion[index].movementScore).toEqual(expected);
        // And the lesion must actually have changed something versus baseline
        // -- otherwise this fixture wouldn't be exercising the lesion path at all.
        expect(graphRaw.lesion[index].movementScore).not.toEqual(graphRaw.baselineMovementScore);
      }
    }
  });

  it('rejects a graph file whose bytes do not match the manifest', () => {
    const tamperedManifestPath = join(root, 'tampered.manifest.json');
    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    manifest.binarySha256 = 'f'.repeat(64);
    writeFileSync(tamperedManifestPath, JSON.stringify(manifest));

    const out = join(root, 'atlas-raw-tampered.json');
    const result = spawnSync(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/lesion/atlas-evaluate.ts',
        '--manifest',
        tamperedManifestPath,
        '--held-out-count',
        '1',
        '--ticks',
        String(TICKS),
        '--max-lesions',
        '1',
        '--shards',
        '1',
        '--out',
        out
      ],
      { encoding: 'utf8', timeout: 60_000 }
    );
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/failed verification|sha256/);
  });
});

// ---------------------------------------------------------------------------
// Report statistics on synthetic raw input
// ---------------------------------------------------------------------------

describe('roundSignificant', () => {
  it('rounds to 6 significant digits', () => {
    expect(roundSignificant(1.23456789)).toBeCloseTo(1.23457, 5);
    expect(roundSignificant(0.000123456789)).toBeCloseTo(0.000123457, 9);
    expect(roundSignificant(123456789)).toBe(123457000);
  });

  it('leaves 0 and non-finite values unchanged', () => {
    expect(roundSignificant(0)).toBe(0);
    expect(roundSignificant(Number.NaN)).toBeNaN();
    expect(roundSignificant(Number.POSITIVE_INFINITY)).toBe(Number.POSITIVE_INFINITY);
  });
});

describe('benjaminiHochbergSignificant', () => {
  it('marks nothing significant when every p-value is 1', () => {
    expect(benjaminiHochbergSignificant([1, 1, 1, 1], 0.05)).toEqual([false, false, false, false]);
  });

  it('marks every hypothesis significant when every p-value is 0', () => {
    expect(benjaminiHochbergSignificant([0, 0, 0], 0.05)).toEqual([true, true, true]);
  });

  it('applies the standard BH procedure on a known example', () => {
    // p-values (ascending): 0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205 (m=8, q=0.05)
    // thresholds (k/m)*q: 0.00625, 0.0125, 0.01875, 0.025, 0.03125, 0.0375, 0.04375, 0.05
    // p_(k) <= threshold holds only at k=1 (0.001<=0.00625) and k=2 (0.008<=0.0125);
    // every later rank fails (0.039<=0.01875 is false, and so on), so the largest
    // passing k is 2 and only ranks 1..2 are rejected (marked significant).
    const pValues = [0.205, 0.001, 0.074, 0.039, 0.06, 0.042, 0.008, 0.041];
    const significant = benjaminiHochbergSignificant(pValues, 0.05);
    expect(significant).toEqual([false, true, false, false, false, false, true, false]);
  });

  it('is step-up, not step-down: a later passing rank rejects every earlier rank too, even ones that individually failed', () => {
    // m=3, q=0.05: thresholds are (1/3)*0.05=0.01667, (2/3)*0.05=0.03333, (3/3)*0.05=0.05.
    // Sorted p-values: 0.001 (rank1, <=0.01667 true), 0.04 (rank2, <=0.03333 FALSE),
    // 0.045 (rank3, <=0.05 true). The largest passing rank is 3, so BH's step-up rule
    // rejects ranks 1..3 -- ALL THREE, including rank 2 (p=0.04), which individually
    // failed its own threshold. A step-down procedure (stop at the first failure) would
    // instead reject only rank 1. This is the one example in this suite that can tell
    // the two procedures apart -- the "known example" test above cannot, since no later
    // rank passes there.
    expect(benjaminiHochbergSignificant([0.045, 0.001, 0.04], 0.05)).toEqual([true, true, true]);
  });

  it('returns an empty array for an empty input', () => {
    expect(benjaminiHochbergSignificant([], 0.05)).toEqual([]);
  });
});

describe('pairedBootstrapPValue', () => {
  // Direct coverage for the plan's bootstrap p-value formula: "the
  // two-sided fraction of resampled mean differences on the opposite side
  // of 0, times 2, capped at 1". Previously exercised only through
  // zero-variance fixtures (buildArtifact's tests), where p is always
  // exactly 0 or 1 -- the sign logic, the 0.5 tie weight, and the cap were
  // never actually exercised (a dual-review finding).
  const RESAMPLES = 20000;

  it('returns 1 immediately when the observed mean difference is 0, without drawing any resamples', () => {
    // A poisoned rng that throws if ever called -- proves the `=== 0` fast
    // path never touches the resample loop.
    const poisonedRng = (): number => {
      throw new Error('rng should not be called when observedMeanDifference is 0');
    };
    expect(pairedBootstrapPValue([1, -1], [0, 0], RESAMPLES, poisonedRng, 0)).toBe(1);
  });

  it('is sign-symmetric: negating both the diffs and the observed mean gives the identical p-value', () => {
    const diffs = [1, 2, -1, 0, 3, -2, 0.5, -0.5, 4, -3];
    const observed = diffs.reduce((sum, v) => sum + v, 0) / diffs.length;
    const zeros = diffs.map(() => 0);

    // Two independent RNG closures from the same (seed, label) draw the
    // identical pseudorandom sequence -- see conditionRng's own doc
    // comment ("a pure function of (bootstrapSeed, its own label, its own
    // data)"). This is what lets the two calls below be an apples-to-apples
    // comparison: same resample *indices* drawn in both.
    const p1 = pairedBootstrapPValue(diffs, zeros, RESAMPLES, conditionRng(1, 'symmetry-test'), observed);
    const p2 = pairedBootstrapPValue(zeros, diffs, RESAMPLES, conditionRng(1, 'symmetry-test'), -observed);
    expect(p2).toBe(p1);
    expect(p1).toBeGreaterThan(0);
    expect(p1).toBeLessThan(1);
  });

  it('a single nonzero seed among mostly-zero diffs gives p close to the predicted tie-weighted value', () => {
    // n=100 diffs: one at +10, the rest exactly 0. A resample's mean is 0
    // unless the nonzero index is drawn at least once (probability
    // 1-(0.99)^100 ~= 0.634); when it is, the resampled mean is positive,
    // the same side as the observed mean, so it never contributes to
    // oppositeSideCount. A resample that misses the nonzero index entirely
    // ((0.99)^100 ~= 0.366 of draws) lands exactly on 0, each contributing
    // the 0.5 tie weight. Predicted p = 2 * 0.5 * (0.99)^100 ~= 0.366.
    const n = 100;
    const diffs = Array.from({ length: n }, (_, i) => (i === 0 ? 10 : 0));
    const zeros = diffs.map(() => 0);
    const observed = 10 / n;
    const p = pairedBootstrapPValue(diffs, zeros, RESAMPLES, conditionRng(2, 'tie-weight-test'), observed);
    const predicted = 2 * 0.5 * 0.99 ** 100;
    expect(p).toBeGreaterThan(predicted - 0.03);
    expect(p).toBeLessThan(predicted + 0.03);
  });

  it('noise around a true zero effect gives a large (non-significant) p-value', () => {
    const rng = conditionRng(3, 'noise-test');
    // Small symmetric noise around 0 -- no real effect.
    const diffs = Array.from({ length: 50 }, () => (rng() - 0.5) * 0.01);
    const zeros = diffs.map(() => 0);
    const observed = diffs.reduce((sum, v) => sum + v, 0) / diffs.length;
    const p = pairedBootstrapPValue(diffs, zeros, RESAMPLES, conditionRng(4, 'noise-test-resample'), observed);
    expect(p).toBeGreaterThan(0.2);
  });
});

const HEX64 = (fill: string): string => fill.repeat(64);

/** A small, internally-consistent `AtlasEvaluationRaw` -- 4 neurons, enough to exercise every code path without a real evaluation run. */
const buildRaw = (neuronCount: number, effects: readonly number[]): AtlasEvaluationRaw => {
  const heldOutSeeds = [30001, 30002, 30003, 30004, 30005, 30006, 30007, 30008, 30009, 30010];
  const baselineMovementScore = heldOutSeeds.map((_, i) => 10 + i * 0.1);
  const graph = (graphKey: string): AtlasGraphRaw => ({
    graphSha256: HEX64(graphKey === 'biological' ? 'a' : 'b'),
    graphGzipSha256: HEX64(graphKey === 'biological' ? 'g' : 'h'),
    heldOutSeeds,
    baselineMovementScore,
    lesion: effects.map((effect, index) => ({
      index,
      // Every seed shifted by exactly `effect` -- a zero-variance paired
      // difference, so pairedStats' CI collapses to a point at `effect`
      // regardless of bootstrap resample draws (deterministic to assert against).
      movementScore: baselineMovementScore.map((score) => score + effect)
    }))
  });

  return {
    version: 1,
    neuronCount,
    seeds: { start: 30001, count: heldOutSeeds.length },
    ticks: 20,
    substeps: 4,
    graphs: { biological: graph('biological'), rewiredSeed0: graph('rewiredSeed0') },
    host: { arch: 'arm64', node: 'v22.22.3' }
  };
};

// `graphSha256` here mirrors the *gzip* sha (`buildRaw`'s `graphGzipSha256: HEX64('g')`
// for the biological graph) -- `readPositions`'s real cross-check (exercised by the
// CLI shard-determinism test, not by `buildArtifact` directly) compares against that.
const buildPositions = (neuronCount: number) => ({
  bodyIds: Array.from({ length: neuronCount }, (_, i) => String(1000 + i)),
  role: Array.from({ length: neuronCount }, () => 'sensory' as const),
  graphSha256: HEX64('g')
});

describe('buildArtifact', () => {
  const runMeta = { shards: 3, elapsedMs: 1234, perEpisodeMs: 5.6 };

  it('computes a zero-variance paired effect exactly, with a point CI', () => {
    const effects = [0, 5, -3, 0.001];
    const raw = buildRaw(effects.length, effects);
    const positions = buildPositions(effects.length);
    const artifact = buildArtifact(raw, positions, runMeta, 42, 500);

    expect(artifact.graphs.biological.effect).toEqual(effects.map((e) => roundSignificant(e)));
    // Zero-variance diffs: every bootstrap resample mean equals the observed
    // mean exactly, so the CI is a point at the effect itself.
    expect(artifact.graphs.biological.ciLow).toEqual(artifact.graphs.biological.effect);
    expect(artifact.graphs.biological.ciHigh).toEqual(artifact.graphs.biological.effect);
  });

  it('ranks topEffects by |effect| descending and carries body IDs/roles', () => {
    const effects = [0.1, -9, 2, 0.5];
    const raw = buildRaw(effects.length, effects);
    const positions = buildPositions(effects.length);
    const artifact = buildArtifact(raw, positions, runMeta, 42, 500);

    const top = artifact.graphs.biological.summary.topEffects;
    expect(top.map((e) => e.index)).toEqual([1, 2, 3, 0]); // |−9| > |2| > |0.5| > |0.1|
    expect(top[0].bodyId).toBe(positions.bodyIds[1]);
    expect(top[0].role).toBe('sensory');
  });

  it('a non-zero, non-degenerate effect is FDR-significant; a zero effect is not', () => {
    // Ten neurons: one clear, non-zero, low-variance effect and nine
    // exactly-zero effects (point CIs at 0, p-value 1 -- see
    // pairedBootstrapPValue's "observedMeanDifference === 0 -> return 1").
    const effects = [10, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const raw = buildRaw(effects.length, effects);
    const positions = buildPositions(effects.length);
    const artifact = buildArtifact(raw, positions, runMeta, 7, 500);

    expect(artifact.graphs.biological.fdrSignificant[0]).toBe(true);
    expect(artifact.graphs.biological.fdrSignificant.slice(1).every((v) => v === false)).toBe(true);
    expect(artifact.graphs.biological.summary.fdrSignificantCount).toBe(1);
    expect(artifact.graphs.biological.summary.expectedChanceExclusions).toBeCloseTo(0.05 * 10, 6);
  });

  it('records baseline, shards, bootstrap, host, and timing', () => {
    const raw = buildRaw(2, [0, 0]);
    const positions = buildPositions(2);
    const artifact = buildArtifact(raw, positions, runMeta, 42, 500);

    expect(artifact.graphs.biological.baseline).toBeCloseTo(10.45, 5);
    expect(artifact.shards).toBe(3);
    expect(artifact.bootstrap).toEqual({ resamples: 500, seed: 42 });
    expect(artifact.host).toEqual({ arch: 'arm64', node: 'v22.22.3' });
    expect(artifact.timing).toEqual({ elapsedMs: 1234, perEpisodeMs: 5.6 });
  });

  it('is a pure function of its inputs: identical inputs produce byte-identical JSON', () => {
    const raw = buildRaw(6, [1, -2, 0, 3.5, -0.25, 0]);
    const positions = buildPositions(6);
    const a = JSON.stringify(buildArtifact(raw, positions, runMeta, 99, 1000));
    const b = JSON.stringify(buildArtifact(raw, positions, runMeta, 99, 1000));
    expect(a).toBe(b);
  });

  it('throws when either graph section is missing', () => {
    const raw = buildRaw(2, [0, 0]);
    const positions = buildPositions(2);
    expect(() =>
      buildArtifact({ ...raw, graphs: { biological: raw.graphs.biological } }, positions, runMeta, 1, 100)
    ).toThrow(/missing biological\/rewiredSeed0/);
  });

  it('derives neuronCount from the actual lesion sweep, not raw.neuronCount (calibration-run regression)', () => {
    // Regression coverage for a real bug caught while calibrating this WP:
    // atlas-evaluate.ts's raw output always records the manifest's full
    // neuronCount (1008) in raw.neuronCount, even when --max-lesions
    // restricted the sweep to fewer neurons. buildArtifact must use the
    // actual per-graph lesion count, not raw.neuronCount, or a calibration
    // run's report would falsely claim "1008 simultaneous tests" and
    // guardShippedDefault would wrongly treat a 4-neuron calibration run as
    // large enough to overwrite the shipped 1008-neuron artifact.
    const effects = [1, -2, 0, 3.5];
    const raw = buildRaw(1008, effects); // raw.neuronCount=1008, but only 4 lesion entries per graph
    const positions = buildPositions(1008);
    const artifact = buildArtifact(raw, positions, runMeta, 1, 100);

    expect(artifact.neuronCount).toBe(4);
    expect(artifact.graphs.biological.effect).toHaveLength(4);
    expect(artifact.bodyIds).toHaveLength(4);
    expect(artifact.bodyIds).toEqual(positions.bodyIds.slice(0, 4));
  });

  it('throws when the two graphs cover different neuron counts', () => {
    const raw = buildRaw(4, [1, -2, 0, 3.5]);
    const positions = buildPositions(4);
    const truncated = { ...raw.graphs.rewiredSeed0!, lesion: raw.graphs.rewiredSeed0!.lesion.slice(0, 2) };
    expect(() =>
      buildArtifact({ ...raw, graphs: { biological: raw.graphs.biological, rewiredSeed0: truncated } }, positions, runMeta, 1, 100)
    ).toThrow(/both graphs must cover the same neuron set/);
  });
});

describe('resolveRunMeta (lesion atlas)', () => {
  let root: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'atlas-report-runmeta-'));
  });
  afterEach(() => rmSync(root, { recursive: true, force: true }));

  const baseArgs = (raw: string, shards?: number): AtlasReportArgs => ({
    raw,
    out: join(root, 'out.json'),
    reportMd: join(root, 'out.md'),
    manifest: join(root, 'manifest.json'),
    positions: join(root, 'positions.json'),
    bootstrapSeed: 1,
    bootstrapResamples: 200,
    shards
  });

  it('throws when no --shards is given and no sidecar exists', () => {
    const raw = join(root, 'atlas-raw.json');
    writeFileSync(raw, '{}');
    expect(() => resolveRunMeta(baseArgs(raw))).toThrow(/cannot determine shard count/);
  });

  it('reads shards and timing from the <raw>.run.json sidecar', () => {
    const raw = join(root, 'atlas-raw.json');
    writeFileSync(raw, '{}');
    writeFileSync(join(root, 'atlas-raw.run.json'), JSON.stringify({ shards: 9, elapsedMs: 42, perEpisodeMs: 1.1 }));
    expect(resolveRunMeta(baseArgs(raw))).toEqual({ shards: 9, elapsedMs: 42, perEpisodeMs: 1.1 });
  });
});

/**
 * Alphabetically-keyed by construction, matching `sortKeysDeep`'s output --
 * `verifyManifestRoundTrips`'s round-trip safety check requires this (same
 * convention as `null-report.test.ts`'s `writeTestManifest`). Defaults to
 * `buildRaw`'s own `graphSha256` values (`HEX64('a')`/`HEX64('b')`) so
 * `verifyRawGraphsMatchManifest` -- which now runs before every
 * `runAtlasReport` write -- passes for a fixture built from `buildRaw`
 * without every call site having to spell these out.
 */
const writeTestManifest = (
  path: string,
  shas: { biologicalSha256?: string; rewiredSeed0Sha256?: string } = {}
): void => {
  const manifest = {
    binarySha256: shas.biologicalSha256 ?? HEX64('a'),
    note: 'test fixture, not the real manifest',
    rewiredArms: { seed0: { binarySha256: shas.rewiredSeed0Sha256 ?? HEX64('b') } }
  };
  writeFileSync(path, `${JSON.stringify(manifest, null, 2)}\n`);
};

describe('runAtlasReport: positions gzip-sha cross-check (regression)', () => {
  // Regression coverage for a real bug caught while calibrating this WP on
  // the real artifacts: `malecns-arena-v1.positions.json`'s own `graphSha256`
  // field is built against the manifest's *gzip* sha256
  // (`src/lib/experiment/assets.ts`'s `loadPositions`), not the decompressed
  // `binarySha256` `AtlasGraphRaw.graphSha256` carries for provenance --
  // `readPositions` must compare against `graphGzipSha256`, not `graphSha256`.
  let root: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'atlas-report-e2e-'));
  });
  afterEach(() => rmSync(root, { recursive: true, force: true }));

  const writeRaw = (rawPath: string, graphGzipSha256: string): void => {
    const raw = buildRaw(2, [3, -1]);
    const patched: AtlasEvaluationRaw = {
      ...raw,
      graphs: {
        biological: { ...raw.graphs.biological!, graphGzipSha256 },
        rewiredSeed0: raw.graphs.rewiredSeed0!
      }
    };
    writeFileSync(rawPath, JSON.stringify(patched));
    writeFileSync(`${rawPath.slice(0, -'.json'.length)}.run.json`, JSON.stringify({ shards: 1 }));
  };

  const argsFor = (rawPath: string, positionsPath: string): AtlasReportArgs => ({
    raw: rawPath,
    out: join(root, 'atlas.json'),
    reportMd: join(root, 'atlas-report.md'),
    manifest: join(root, 'manifest.json'),
    positions: positionsPath,
    bootstrapSeed: 1,
    bootstrapResamples: 50
  });

  it('succeeds when the positions sidecar graphSha256 matches the raw evaluation gzip sha', () => {
    const rawPath = join(root, 'atlas-raw.json');
    const gzipSha = HEX64('g');
    writeRaw(rawPath, gzipSha);
    const positionsPath = join(root, 'positions.json');
    writeFileSync(positionsPath, JSON.stringify(buildPositions(2)) /* graphSha256: HEX64('g') */);
    writeTestManifest(join(root, 'manifest.json'));

    const result = runAtlasReport(argsFor(rawPath, positionsPath));
    expect(result.artifact.graphs.biological.effect).toHaveLength(2);
  });

  it('records the artifact sha256 in the manifest, and a rerun is byte-identical (two plan acceptance criteria)', () => {
    const rawPath = join(root, 'atlas-raw.json');
    writeRaw(rawPath, HEX64('g'));
    const positionsPath = join(root, 'positions.json');
    writeFileSync(positionsPath, JSON.stringify(buildPositions(2)));
    writeTestManifest(join(root, 'manifest.json'));

    const args = argsFor(rawPath, positionsPath);
    const result1 = runAtlasReport(args);

    // "The manifest sha256 equals the artifact bytes" (the plan's own
    // acceptance criterion).
    const artifactBytes1 = readFileSync(result1.out);
    expect(sha256Hex(artifactBytes1)).toBe(result1.artifactSha256);
    const manifestAfter1 = JSON.parse(readFileSync(args.manifest, 'utf8')) as { lesionAtlas: { sha256: string } };
    expect(manifestAfter1.lesionAtlas.sha256).toBe(result1.artifactSha256);

    // "Running lesion:report twice gives byte-identical output" (the plan's
    // other acceptance criterion) -- rerun against a fresh copy of the same
    // inputs (the manifest was mutated by the first run, so start it over
    // from the same starting bytes rather than reusing the now-different
    // on-disk manifest).
    writeTestManifest(args.manifest);
    const result2 = runAtlasReport(args);
    const artifactBytes2 = readFileSync(result2.out);
    expect(artifactBytes2.equals(artifactBytes1)).toBe(true);
    expect(result2.artifactSha256).toBe(result1.artifactSha256);
  });

  it('throws a clear "stale positions artifact" error on a gzip-sha mismatch', () => {
    const rawPath = join(root, 'atlas-raw.json');
    writeRaw(rawPath, HEX64('different-gzip-sha'));
    const positionsPath = join(root, 'positions.json');
    writeFileSync(positionsPath, JSON.stringify(buildPositions(2)) /* graphSha256: HEX64('g') */);
    writeTestManifest(join(root, 'manifest.json'));

    expect(() => runAtlasReport(argsFor(rawPath, positionsPath))).toThrow(/stale positions artifact/);
  });
});

describe('runAtlasReport: guardShippedDefault (regression)', () => {
  // Direct coverage for I-1/I-4b of the dual review: guardShippedDefault
  // must refuse to overwrite the shipped defaults not only when a run is
  // under-covered on neurons (the original calibration-run bug this WP
  // already fixed once -- tests/unit/lesion-atlas.test.ts's earlier
  // "derives neuronCount..." test covers that at the buildArtifact level),
  // but also when a run covers every neuron yet used a shorter/cheaper
  // condition than the shipped atlas (seeds/ticks/substeps) -- a full-neuron
  // "smoke run" (`--held-out-count 3 --ticks 100`) would otherwise pass a
  // neuron-count-only guard and silently overwrite the shipped artifact
  // with drastically noisier numbers. Asserts only the *throw*, mirroring
  // null-report.test.ts's own `DEFAULT_OUT` guard test -- guardShippedDefault
  // runs before any write, so passing the literal shipped `DEFAULT_OUT`/
  // `DEFAULT_MANIFEST` paths here never actually touches the real
  // public/data files as long as the guard does its job (which is exactly
  // what each `toThrow` below verifies).
  //
  // `requireOutBesideManifest` (I-2, tested separately below) requires
  // `--out` and `--manifest` to share a directory, so any case that uses the
  // real `DEFAULT_OUT` must also use the real `DEFAULT_MANIFEST` -- and
  // `verifyRawGraphsMatchManifest` (I-5) then requires the fixture's
  // `graphSha256` values to match whatever is actually in that real,
  // committed manifest file. Read those (real, public) shas once and reuse
  // them, rather than writing into `public/data/` or guessing values that
  // would drift whenever the compiled graph is regenerated.
  const realManifest = JSON.parse(readFileSync(DEFAULT_MANIFEST, 'utf8')) as {
    binarySha256: string;
    rewiredArms: { seed0: { binarySha256: string } };
  };

  let root: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'atlas-report-guard-'));
  });
  afterEach(() => rmSync(root, { recursive: true, force: true }));

  const FULL_NEURON_COUNT = 1008;

  /**
   * Builds a fully self-consistent `AtlasEvaluationRaw` for an arbitrary
   * `(seeds, ticks)` pair -- unlike `buildRaw` (hardcoded to a fixed
   * 10-seed array), every graph's `heldOutSeeds`/`baselineMovementScore`/
   * `lesion[].movementScore` here is sized and seeded to match `seeds`
   * exactly, so `verifyRawSeedsConsistent` (a dual-review defense-in-depth
   * check) never rejects these guard-focused fixtures for an unrelated
   * reason. `graphSha256` values match the real, committed manifest -- see
   * the describe block's comment.
   */
  const writeFixture = (
    rawPath: string,
    seeds: { readonly start: number; readonly count: number },
    ticks: number
  ): void => {
    const heldOutSeeds = Array.from({ length: seeds.count }, (_, i) => seeds.start + i);
    const baselineMovementScore = heldOutSeeds.map((_, i) => 10 + i * 0.1);
    const graph = (graphSha256: string): AtlasGraphRaw => ({
      graphSha256,
      graphGzipSha256: HEX64('g'),
      heldOutSeeds,
      baselineMovementScore,
      lesion: Array.from({ length: FULL_NEURON_COUNT }, (_, index) => ({
        index,
        movementScore: baselineMovementScore // effect 0: only the guard's own throw is under test
      }))
    });
    const raw: AtlasEvaluationRaw = {
      version: 1,
      neuronCount: FULL_NEURON_COUNT,
      seeds,
      ticks,
      substeps: 4,
      graphs: {
        biological: graph(realManifest.binarySha256),
        rewiredSeed0: graph(realManifest.rewiredArms.seed0.binarySha256)
      },
      host: { arch: 'arm64', node: 'v22.22.3' }
    };
    writeFileSync(rawPath, JSON.stringify(raw));
    writeFileSync(`${rawPath.slice(0, -'.json'.length)}.run.json`, JSON.stringify({ shards: 1 }));
    writeFileSync(join(root, 'positions.json'), JSON.stringify(buildPositions(FULL_NEURON_COUNT)));
  };

  const guardedArgs = (rawPath: string, out: string, reportMd: string, manifest: string): AtlasReportArgs => ({
    raw: rawPath,
    out,
    reportMd,
    manifest,
    positions: join(root, 'positions.json'),
    bootstrapSeed: 1,
    bootstrapResamples: 20 // small: only the guard's own throw is under test, not the statistics
  });

  it('refuses --held-out-count 3/--ticks 100 smoke run (all 1008 neurons, wrong seeds+ticks) at DEFAULT_OUT', () => {
    const rawPath = join(root, 'atlas-raw.json');
    writeFixture(rawPath, { start: 30001, count: 3 }, 100);
    // out+manifest must share a directory (I-2); DEFAULT_OUT and
    // DEFAULT_MANIFEST both live in public/data, so this is the one
    // combination that can use the real DEFAULT_OUT without writing a
    // scratch file into public/data itself.
    expect(() =>
      runAtlasReport(guardedArgs(rawPath, DEFAULT_OUT, join(root, 'r.md'), DEFAULT_MANIFEST))
    ).toThrow(/refusing to overwrite the shipped/);
  });

  it('refuses the same smoke run at DEFAULT_REPORT_MD, independent of --out/--manifest', () => {
    const rawPath = join(root, 'atlas-raw.json');
    writeFixture(rawPath, { start: 30001, count: 3 }, 100);
    const manifestPath = join(root, 'm.json');
    writeTestManifest(manifestPath, {
      biologicalSha256: realManifest.binarySha256,
      rewiredSeed0Sha256: realManifest.rewiredArms.seed0.binarySha256
    });
    expect(() =>
      runAtlasReport(guardedArgs(rawPath, join(root, 'o.json'), DEFAULT_REPORT_MD, manifestPath))
    ).toThrow(/refusing to overwrite the shipped/);
  });

  it('refuses a run with the shipped neuron/seed count but the wrong tick count', () => {
    const rawPath = join(root, 'atlas-raw.json');
    // Neuron coverage and seeds match the shipped condition exactly; only
    // ticks is off -- isolates that guardShippedDefault checks ticks
    // independently of neuron coverage, not merely as a side effect of a
    // smoke run also being short on seeds.
    writeFixture(rawPath, { start: DEFAULT_HELD_OUT_START, count: DEFAULT_HELD_OUT_COUNT }, DEFAULT_TICKS - 1);
    expect(() =>
      runAtlasReport(guardedArgs(rawPath, DEFAULT_OUT, join(root, 'r.md'), DEFAULT_MANIFEST))
    ).toThrow(/refusing to overwrite the shipped/);
  });

  it('does not guard a scratch path: a short run still writes when none of out/report-md/manifest is a shipped default', () => {
    const rawPath = join(root, 'atlas-raw.json');
    writeFixture(rawPath, { start: 30001, count: 3 }, 100);
    const manifestPath = join(root, 'manifest.json');
    writeTestManifest(manifestPath, {
      biologicalSha256: realManifest.binarySha256,
      rewiredSeed0Sha256: realManifest.rewiredArms.seed0.binarySha256
    });
    const result = runAtlasReport(guardedArgs(rawPath, join(root, 'o.json'), join(root, 'r.md'), manifestPath));
    expect(result.artifact.neuronCount).toBe(FULL_NEURON_COUNT);
  });
});

describe('parseAtlasReportArgs', () => {
  it('applies defaults, with shards left undefined', () => {
    const args = parseAtlasReportArgs([]);
    expect(args.shards).toBeUndefined();
    expect(args.bootstrapResamples).toBe(10000);
    expect(args.out).toBe(DEFAULT_OUT);
    expect(args.reportMd).toBe(DEFAULT_REPORT_MD);
    expect(args.manifest).toBe(DEFAULT_MANIFEST);
  });

  it('rejects --raw without a .json extension', () => {
    expect(() => parseAtlasReportArgs(['--raw', 'atlas'])).toThrow(/--raw must end with "\.json"/);
  });

  it('rejects an unknown flag', () => {
    expect(() => parseAtlasReportArgs(['--bogus'])).toThrow(/Unknown argument/);
  });
});
