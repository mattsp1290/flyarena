// @vitest-environment node
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'vitest';

import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { createTraceGraph } from '../fixtures/trace-graph';
import { createFixtureRewiredTraceGraph } from '../fixtures/trace-graph-rewire';
import {
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
  parseAtlasReportArgs,
  resolveRunMeta,
  roundSignificant,
  runAtlasReport,
  type AtlasReportArgs
} from '../../scripts/lesion/atlas-report';
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

  const HELD_OUT_START = 30001;
  const HELD_OUT_COUNT = 3;
  const TICKS = 20;
  const MAX_LESIONS = 4;

  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'atlas-evaluate-fixture-'));

    const bioGraph = createTraceGraph();
    const bioBinary = Buffer.from(encodeGraphBinary(bioGraph));
    const bioBinarySha256 = sha256Hex(bioBinary);
    const bioGzip = gzipSync(bioBinary);
    writeFileSync(join(root, 'malecns-arena-v1.bin.gz'), bioGzip);

    const rewiredGraph = createFixtureRewiredTraceGraph(bioGraph, 0);
    const rewiredBinary = Buffer.from(encodeGraphBinary(rewiredGraph));
    const rewiredBinarySha256 = sha256Hex(rewiredBinary);
    const rewiredGzip = gzipSync(rewiredBinary);
    writeFileSync(join(root, 'malecns-arena-v1-rewired-seed0.bin.gz'), rewiredGzip);

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
    for (const key of ['biological', 'rewiredSeed0'] as const) {
      const graph = parsed.graphs[key] as AtlasGraphRaw;
      expect(graph.baselineMovementScore).toHaveLength(HELD_OUT_COUNT);
      expect(graph.lesion).toHaveLength(MAX_LESIONS);
      expect(graph.lesion.map((l) => l.index)).toEqual([0, 1, 2, 3]);
      for (const entry of graph.lesion) {
        expect(entry.movementScore).toHaveLength(HELD_OUT_COUNT);
        for (const score of entry.movementScore) expect(Number.isFinite(score)).toBe(true);
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

  it('applies the standard BH step-up rule on a known example', () => {
    // p-values (ascending): 0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205 (m=8, q=0.05)
    // thresholds (k/m)*q: 0.00625, 0.0125, 0.01875, 0.025, 0.03125, 0.0375, 0.04375, 0.05
    // largest k with p_(k) <= threshold: k=3 (p=0.041 <= 0.025? no) -- recompute below in-line.
    const pValues = [0.205, 0.001, 0.074, 0.039, 0.06, 0.042, 0.008, 0.041];
    const significant = benjaminiHochbergSignificant(pValues, 0.05);
    // Only the two smallest survive: 0.001 (rank 1, threshold 0.00625) and
    // 0.008 (rank 2, threshold 0.0125). 0.039 (rank 3, threshold 0.01875) fails,
    // and BH's step-up rule stops at the largest passing rank (here, rank 2) --
    // it does not separately re-test later ranks.
    expect(significant).toEqual([false, true, false, false, false, false, true, false]);
  });

  it('returns an empty array for an empty input', () => {
    expect(benjaminiHochbergSignificant([], 0.05)).toEqual([]);
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

/** Alphabetically-keyed by construction, matching `sortKeysDeep`'s output -- `verifyManifestRoundTrips`'s round-trip safety check requires this (same convention as `null-report.test.ts`'s `writeTestManifest`). */
const writeTestManifest = (path: string): void => {
  const manifest = { note: 'test fixture, not the real manifest' };
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

  it('throws a clear "stale positions artifact" error on a gzip-sha mismatch', () => {
    const rawPath = join(root, 'atlas-raw.json');
    writeRaw(rawPath, HEX64('different-gzip-sha'));
    const positionsPath = join(root, 'positions.json');
    writeFileSync(positionsPath, JSON.stringify(buildPositions(2)) /* graphSha256: HEX64('g') */);
    writeTestManifest(join(root, 'manifest.json'));

    expect(() => runAtlasReport(argsFor(rawPath, positionsPath))).toThrow(/stale positions artifact/);
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
