import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  buildRepertoirePlan,
  verifyBundleIdentity,
  type RepertoirePlanInputs
} from '../../scripts/atlas/repertoire-plan';
import { runRepertoireTask, type RepertoireCellResult, type RepertoireWorkerTask } from '../../scripts/atlas/repertoire-task';
import { runWorkerScheduler, buildRepertoireTasks, assembleEvaluatedArtifact } from '../../scripts/atlas/repertoire-evaluate';
import {
  occupied,
  qd,
  span,
  heldoutOwnMedian,
  rewiredDistribution,
  metricVerdict,
  categorize,
  robustness,
  type RepertoireCellMetricInput
} from '../../scripts/atlas/repertoire-metrics';
import type { RewireIndex } from '../../scripts/null/rewire-index';
import {
  computeArmBundleSha256,
  deserializeArmBundle,
  runExportArms,
  type SerializedArmBundle
} from '../../scripts/training/export-arms';
import { sha256Hex } from '../../scripts/training/fsio';
import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { readoutParameterCount } from '../../src/lib/connectome/readout';
import {
  ATLAS_VERSION,
  COVERAGE_EDGES,
  TURN_EDGES,
  DISCOVERY_SEEDS,
  HELDOUT_SEEDS,
  type SearchArtifact
} from '../../src/lib/atlas/types';
import { ARENA_CONFIG } from '../../src/lib/arena/config';

// A synthetic 20-seed RewireIndex, shaped like `rewire_batch.py`'s real
// `index.json` (`readRewireIndex`'s own validation), for `buildRepertoirePlan`
// tests that never touch a real GPU-produced index file.
const fakeRewireIndex = (seedCount: number): RewireIndex => ({
  sourceArtifact: 'malecns-arena-v1.bin.gz',
  sourceSha256: 'a'.repeat(64),
  rewireSourceSha256: 'b'.repeat(64),
  seeds: Array.from({ length: seedCount }, (_, seed) => ({
    seed,
    artifact: `malecns-arena-v1-rewired-seed${seed}.bin.gz`,
    binarySha256: sha256Hex(`seed-${seed}`),
    binaryBytes: 1,
    gzipSha256: sha256Hex(`gzip-${seed}`),
    gzipBytes: 1,
    stats: { acceptedSwaps: 1, attempts: 1 }
  }))
});

describe('repertoire-plan: buildRepertoirePlan', () => {
  const inputsFor = (seedCount: number): RepertoirePlanInputs => ({
    rewireIndex: fakeRewireIndex(seedCount),
    biologicalBinarySha256: 'bio-binary-sha',
    biologicalGzipSha256: 'bio-gzip-sha',
    disconnectedBinarySha256: 'disc-binary-sha',
    armsDir: '/arms',
    searchDir: '/search'
  });

  it('produces exactly the planned 46 (graph, seed) pairs', () => {
    const plan = buildRepertoirePlan(inputsFor(20));
    expect(plan.length).toBe(46);
    expect(plan.filter((e) => e.arm === 'biological').length).toBe(5);
    expect(plan.filter((e) => e.arm === 'disconnected').length).toBe(1);
    expect(plan.filter((e) => e.arm === 'rewired').length).toBe(40);
    // Every rewiring 0-19 is searched at 1729; only 0-4 get the 4 extra seeds.
    for (let seed = 0; seed < 20; seed += 1) {
      const forSeed = plan.filter((e) => e.rewiringSeed === seed);
      expect(forSeed.length).toBe(seed < 5 ? 5 : 1);
      expect(forSeed.map((e) => e.searchSeed)).toContain(1729);
    }
  });

  it('orders entries by (graph, seed), never lexically over graphId (rewired-10 after rewired-9, not before rewired-2)', () => {
    const plan = buildRepertoirePlan(inputsFor(20));
    const rewiredIds = plan.filter((e) => e.arm === 'rewired').map((e) => e.rewiringSeed as number);
    // Non-decreasing rewiring seed across the whole rewired run.
    for (let i = 1; i < rewiredIds.length; i += 1) expect(rewiredIds[i]).toBeGreaterThanOrEqual(rewiredIds[i - 1]);
    const rewired9Index = plan.findIndex((e) => e.graphId === 'rewired-9');
    const rewired10Index = plan.findIndex((e) => e.graphId === 'rewired-10');
    expect(rewired10Index).toBeGreaterThan(rewired9Index);
    expect(plan[0].graphId).toBe('biological');
    expect(plan.filter((e) => e.graphId === 'disconnected').length).toBe(1);
  });

  it('keys each rewired entry to its own seed\'s binarySha256 from index.json, not a shared/parent sha', () => {
    const plan = buildRepertoirePlan(inputsFor(20));
    const rewired3 = plan.find((e) => e.graphId === 'rewired-3' && e.searchSeed === 1729)!;
    const rewired7 = plan.find((e) => e.graphId === 'rewired-7' && e.searchSeed === 1729)!;
    expect(rewired3.expected.binarySha256).toBe(sha256Hex('seed-3'));
    expect(rewired7.expected.binarySha256).toBe(sha256Hex('seed-7'));
    expect(rewired3.expected.binarySha256).not.toBe(rewired7.expected.binarySha256);
    // Parent identity is shared across every arm from one export, by design.
    expect(rewired3.expected.parentGzipSha256).toBe('bio-gzip-sha');
  });

  it('throws if index.json is missing an expected rewiring seed', () => {
    expect(() => buildRepertoirePlan(inputsFor(19))).toThrow('no entry for rewiring seed 19');
  });

  it('bundle paths are distinct per rewiring seed (the export-arms --out overwrite hazard the plan calls out)', () => {
    const plan = buildRepertoirePlan(inputsFor(20));
    const bundlePaths = new Set(plan.filter((e) => e.arm === 'rewired').map((e) => e.bundlePath));
    expect(bundlePaths.size).toBe(20);
  });
});

describe('repertoire-plan: verifyBundleIdentity', () => {
  let root: string;
  let biological: SerializedArmBundle;
  let rewired: SerializedArmBundle;
  let disconnected: SerializedArmBundle;
  let outDir: string;

  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'repertoire-verify-bundle-'));
    const result = runExportArms({ outDir: root, fixtureRewire: true, fixtureRewireSeed: 5 });
    outDir = result.outDir;
    biological = JSON.parse(readFileSync(join(outDir, 'biological.json'), 'utf8'));
    rewired = JSON.parse(readFileSync(join(outDir, 'rewired.json'), 'utf8'));
    disconnected = JSON.parse(readFileSync(join(outDir, 'disconnected.json'), 'utf8'));
  });

  afterAll(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  const binarySha = (bundle: SerializedArmBundle) => sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(bundle))));

  it('accepts a self-consistent rewired bundle whose re-encoded binary matches', () => {
    expect(() =>
      verifyBundleIdentity(join(outDir, 'rewired.json'), {
        arm: 'rewired',
        binarySha256: binarySha(rewired),
        parentGzipSha256: rewired.graphArtifactSha256
      })
    ).not.toThrow();
  });

  it('rejects an arm mismatch against what was expected', () => {
    expect(() =>
      verifyBundleIdentity(join(outDir, 'rewired.json'), {
        arm: 'biological',
        binarySha256: binarySha(rewired),
        parentGzipSha256: rewired.graphArtifactSha256
      })
    ).toThrow('arm mismatch');
  });

  it('rejects a wrong binary sha even with a correct parent sha', () => {
    expect(() =>
      verifyBundleIdentity(join(outDir, 'rewired.json'), {
        arm: 'rewired',
        binarySha256: 'f'.repeat(64),
        parentGzipSha256: rewired.graphArtifactSha256
      })
    ).toThrow('binary identity mismatch');
  });

  it('rejects a wrong parent sha even with a correct binary sha', () => {
    expect(() =>
      verifyBundleIdentity(join(outDir, 'rewired.json'), {
        arm: 'rewired',
        binarySha256: binarySha(rewired),
        parentGzipSha256: 'a'.repeat(64)
      })
    ).toThrow('parent identity mismatch');
  });

  it('rejects a disconnected bundle mislabeled from a non-empty-edge bundle', () => {
    const mislabeledWithoutHash: SerializedArmBundle = { ...biological, arm: 'disconnected' };
    const mislabeled: SerializedArmBundle = { ...mislabeledWithoutHash, sha256: computeArmBundleSha256(mislabeledWithoutHash) };
    const path = join(root, 'mislabeled-disconnected.json');
    writeFileSync(path, JSON.stringify(mislabeled));
    expect(() =>
      verifyBundleIdentity(path, {
        arm: 'disconnected',
        binarySha256: binarySha(mislabeled),
        parentGzipSha256: mislabeled.graphArtifactSha256
      })
    ).toThrow('edgeCount');
  });

  it('accepts the disconnected bundle with its own derived identity', () => {
    expect(() =>
      verifyBundleIdentity(join(outDir, 'disconnected.json'), {
        arm: 'disconnected',
        binarySha256: binarySha(disconnected),
        parentGzipSha256: disconnected.graphArtifactSha256
      })
    ).not.toThrow();
  });

  it('rejects a bundle whose declared sha256 no longer matches its own contents (tampered, not re-hashed)', () => {
    const tampered: SerializedArmBundle = {
      ...rewired,
      presynapticSigns: rewired.presynapticSigns.map((sign) => -sign)
      // sha256 left as the original bundle's -- no longer self-consistent.
    };
    const path = join(root, 'tampered-rewired.json');
    writeFileSync(path, JSON.stringify(tampered));
    expect(() =>
      verifyBundleIdentity(path, {
        arm: 'rewired',
        binarySha256: binarySha(rewired),
        parentGzipSha256: rewired.graphArtifactSha256
      })
    ).toThrow('not self-consistent');
  });
});

describe('repertoire-metrics: pure functions', () => {
  const cell = (over: Partial<RepertoireCellMetricInput> = {}): RepertoireCellMetricInput => ({
    cell: 0,
    quality: 1,
    heldoutOwn: [{ movementScore: 1, foodPickups: 0, hazardContacts: 0, distanceTravelled: 0, coverage: 0, turning: 0 }],
    ...over
  });

  it('occupied counts cells directly', () => {
    expect(occupied([cell(), cell(), cell()])).toBe(3);
    expect(occupied([])).toBe(0);
  });

  it('qd sums max(0, quality), flooring negative-quality cells at zero', () => {
    expect(qd([cell({ quality: 2 }), cell({ quality: -5 }), cell({ quality: 0.5 })])).toBeCloseTo(2.5);
  });

  it('span counts distinct coverage bins (cell % 6) plus distinct turning bins (floor(cell / 6))', () => {
    // cells 0, 1 share turning bin 0, coverage bins 0 and 1 -> 2 coverage + 1 turning = 3.
    expect(span([cell({ cell: 0 }), cell({ cell: 1 })])).toBe(3);
    // cells 0 and 6: same coverage bin (0), different turning bins (0, 1) -> 1 + 2 = 3.
    expect(span([cell({ cell: 0 }), cell({ cell: 6 })])).toBe(3);
  });

  it('heldoutOwnMedian: odd and even cell counts', () => {
    const withScore = (score: number) => cell({ heldoutOwn: [{ movementScore: score, foodPickups: 0, hazardContacts: 0, distanceTravelled: 0, coverage: 0, turning: 0 }] });
    expect(heldoutOwnMedian([withScore(1), withScore(3), withScore(2)])).toBe(2);
    expect(heldoutOwnMedian([withScore(1), withScore(4)])).toBe(2.5);
  });

  it('heldoutOwnMedian averages multiple held-out seeds per cell before taking the median', () => {
    const twoSeedCell = (a: number, b: number) =>
      cell({
        heldoutOwn: [
          { movementScore: a, foodPickups: 0, hazardContacts: 0, distanceTravelled: 0, coverage: 0, turning: 0 },
          { movementScore: b, foodPickups: 0, hazardContacts: 0, distanceTravelled: 0, coverage: 0, turning: 0 }
        ]
      });
    // means: 2, 6 -> median 4.
    expect(heldoutOwnMedian([twoSeedCell(1, 3), twoSeedCell(5, 7)])).toBe(4);
  });

  it('rewiredDistribution computes p25/p50/p75 with the null-stats.ts quantile convention', () => {
    const dist = rewiredDistribution([10, 20, 30, 40]);
    expect(dist.n).toBe(4);
    expect(dist.values).toEqual([10, 20, 30, 40]);
    // quantileIndex(4, .25) = floor(1) = 1 -> values[1] = 20
    expect(dist.p25).toBe(20);
    // quantileIndex(4, .5) = floor(2) = 2 -> values[2] = 30
    expect(dist.p50).toBe(30);
    // quantileIndex(4, .75) = min(3, ceil(3)-1) = min(3,2) = 2 -> values[2] = 30
    expect(dist.p75).toBe(30);
  });

  it('metricVerdict: wider requires >= p75 and > p25', () => {
    const dist = rewiredDistribution([1, 2, 3, 4, 5, 6, 7, 8]);
    const verdict = metricVerdict(dist.p75 + 1, dist);
    expect(verdict.wider).toBe(true);
    expect(verdict.narrower).toBe(false);
    expect(verdict.tie).toBe(false);
  });

  it('metricVerdict: narrower requires <= p25 and < p75', () => {
    const dist = rewiredDistribution([1, 2, 3, 4, 5, 6, 7, 8]);
    const verdict = metricVerdict(dist.p25 - 1, dist);
    expect(verdict.narrower).toBe(true);
    expect(verdict.wider).toBe(false);
  });

  it('metricVerdict: a degenerate distribution (p25 = p75 = bio) is neither wider nor narrower, and is flagged tie', () => {
    const dist = rewiredDistribution([5, 5, 5, 5]);
    const verdict = metricVerdict(5, dist);
    expect(verdict.wider).toBe(false);
    expect(verdict.narrower).toBe(false);
    expect(verdict.tie).toBe(true);
  });

  it('categorize: wider only if both metrics are wider', () => {
    const dist = rewiredDistribution([1, 2, 3, 4, 5, 6, 7, 8]);
    const wideVerdict = metricVerdict(dist.p75 + 1, dist);
    const typicalVerdict = metricVerdict(dist.p50, dist);
    expect(categorize(wideVerdict, wideVerdict).category).toBe('wider');
    expect(categorize(wideVerdict, typicalVerdict).category).toBe('typical');
  });

  it('categorize: narrower only if both metrics are narrower, and wider/narrower can never both hold', () => {
    const dist = rewiredDistribution([1, 2, 3, 4, 5, 6, 7, 8]);
    const narrowVerdict = metricVerdict(dist.p25 - 1, dist);
    expect(categorize(narrowVerdict, narrowVerdict).category).toBe('narrower');
    // No metric pair can be simultaneously wider and narrower for the same category call.
    const wideVerdict = metricVerdict(dist.p75 + 1, dist);
    const result = categorize(wideVerdict, narrowVerdict);
    expect(result.category).toBe('typical');
  });

  it('categorize propagates the tie flag from either metric', () => {
    const tiedDist = rewiredDistribution([5, 5, 5, 5]);
    const tiedVerdict = metricVerdict(5, tiedDist);
    const normalDist = rewiredDistribution([1, 2, 3, 4, 5, 6, 7, 8]);
    const normalVerdict = metricVerdict(normalDist.p50, normalDist);
    expect(categorize(tiedVerdict, normalVerdict).tie).toBe(true);
    expect(categorize(normalVerdict, normalVerdict).tie).toBe(false);
  });

  it('robustness is true only when every seed agrees', () => {
    expect(
      robustness([
        { seed: 1729, category: 'wider' },
        { seed: 1730, category: 'wider' },
        { seed: 1731, category: 'wider' }
      ]).robust
    ).toBe(true);
    const mixed = robustness([
      { seed: 1729, category: 'wider' },
      { seed: 1730, category: 'typical' }
    ]);
    expect(mixed.robust).toBe(false);
    expect(mixed.perSeed[1729]).toBe('wider');
    expect(mixed.perSeed[1730]).toBe('typical');
  });
});

describe('repertoire-task: runRepertoireTask rejects arm/bundle mix-ups', () => {
  let root: string;
  let biological: SerializedArmBundle;
  let rewired: SerializedArmBundle;
  let outDir: string;

  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'repertoire-task-'));
    const result = runExportArms({ outDir: root, fixtureRewire: true, fixtureRewireSeed: 9 });
    outDir = result.outDir;
    biological = JSON.parse(readFileSync(join(outDir, 'biological.json'), 'utf8'));
    rewired = JSON.parse(readFileSync(join(outDir, 'rewired.json'), 'utf8'));
  });

  afterAll(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  const runtimeConfig = Object.fromEntries(
    Object.entries(ARENA_CONFIG).map(([key, value]) => [key.replace(/[A-Z]/g, (c) => '_' + c.toLowerCase()), value])
  );

  const buildSearchArtifact = (bundle: SerializedArmBundle): SearchArtifact => {
    const d = bundle.D;
    const hiddenSize = 8;
    const parameterCount = readoutParameterCount(d, hiddenSize);
    const theta = Array.from({ length: parameterCount }, (_, i) => (((i * 37) % 101) / 101 - 0.5) * 0.2);
    return {
      schemaVersion: 1,
      modelVersion: ATLAS_VERSION,
      options: { seed: 1, population: 4, generations: 1, ticks: 30 },
      inputSize: d,
      hiddenSize,
      substeps: 4,
      discoverySeeds: DISCOVERY_SEEDS,
      heldoutSeeds: HELDOUT_SEEDS,
      coverageEdges: COVERAGE_EDGES,
      turnEdges: TURN_EDGES,
      graphArtifactSha256: bundle.graphArtifactSha256,
      bundleSha256: bundle.sha256,
      bundle: bundle as unknown as Record<string, unknown>,
      candidates: [{ id: 0, theta, quality: 0, coverage: 0, turning: 0 }],
      history: [{ generation: 1, occupied: 1, bestQuality: 0 }],
      searchPolicy: {
        initialStd: 0.5,
        mutationScales: [0.05, 0.15, 0.4],
        freshFraction: 0.25,
        weightBound: 8,
        ties: 'earlier candidate',
        rng: 'torch CPU Generator'
      },
      runtime: {
        device: 'cpu',
        deviceName: 'vitest',
        torch: '0.0.0',
        cuda: null,
        seconds: 1,
        peakTensorBytes: 0,
        config: runtimeConfig
      }
    };
  };

  const binarySha = (bundle: SerializedArmBundle) => sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(bundle))));

  it('rejects a search whose bundle is rewired when the plan expected biological', async () => {
    const searchPath = join(root, 'wrong-arm-biological.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(rewired)));
    await expect(
      runRepertoireTask({
        graphId: 'biological@1729',
        searchPath,
        expected: { arm: 'biological', binarySha256: binarySha(rewired), parentGzipSha256: rewired.graphArtifactSha256 },
        arm: 'biological',
        rewiringSeed: null,
        searchSeed: 1729
      })
    ).rejects.toThrow('arm mismatch');
  });

  it('rejects a search whose bundle is biological when the plan expected rewired', async () => {
    const searchPath = join(root, 'wrong-arm-rewired.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(biological)));
    await expect(
      runRepertoireTask({
        graphId: 'rewired-9@1729',
        searchPath,
        expected: { arm: 'rewired', binarySha256: binarySha(biological), parentGzipSha256: biological.graphArtifactSha256 },
        arm: 'rewired',
        rewiringSeed: 9,
        searchSeed: 1729
      })
    ).rejects.toThrow('arm mismatch');
  });

  it('returns a RepertoireEvaluatedEntry with heldoutOwn (never a "biological"-named field) for a rewired graph', async () => {
    const searchPath = join(root, 'valid-rewired.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(rewired)));
    const entry = await runRepertoireTask({
      graphId: 'rewired-9@1729',
      searchPath,
      expected: { arm: 'rewired', binarySha256: binarySha(rewired), parentGzipSha256: rewired.graphArtifactSha256 },
      arm: 'rewired',
      rewiringSeed: 9,
      searchSeed: 1729
    });
    expect(entry.occupied).toBe(1);
    expect(entry.gpuArchiveSize).toBe(1);
    expect(entry.cells).toHaveLength(1);
    const cellResult: RepertoireCellResult = entry.cells[0];
    expect(cellResult.heldoutOwn).toHaveLength(12);
    expect('biological' in cellResult).toBe(false);
  });
});

describe('repertoire-evaluate: runWorkerScheduler shard determinism / failure paths (stub worker)', () => {
  // `repertoire-worker.ts` itself is a real `.ts` file that needs a `tsx`
  // loader to resolve its own extensionless `./repertoire-task` import when
  // forked -- `scripts/null/null-evaluate.ts`'s own shard-determinism test
  // gets that loader by spawning its *CLI* as a subprocess with `--import
  // tsx` (so the forked children inherit it via `execArgvForChildren`), but
  // a plain `vitest run` invocation's own process has no such loader in its
  // `execArgv` to inherit. `tests/fixtures/repertoire-stub-worker.mjs` (a
  // plain `.mjs` file, no TS/loader involved) exercises the *scheduler*'s
  // own mechanics directly instead -- shard-count/completion-order
  // independence and fail-fast abort -- exactly mirroring
  // `tests/unit/null-evaluate.test.ts`'s own "runShardedEvaluation:
  // failure/abort paths (stub worker)" section. The real per-task pipeline
  // (verify + evaluate + heldoutOwn renaming + arm-mismatch rejection) is
  // covered separately above by calling `runRepertoireTask` directly,
  // in-process, with no forking involved.
  const stubWorkerPath = resolve(dirname(fileURLToPath(import.meta.url)), '../fixtures/repertoire-stub-worker.mjs');

  /** `delayMs` is a stub-worker-only extension of the wire protocol, not part of the real `RepertoireWorkerTask`. */
  const task = (graphId: string, delayMs = 0): RepertoireWorkerTask & { delayMs: number } => ({
    graphId,
    searchPath: 'unused',
    expected: { arm: 'rewired', binarySha256: 'unused', parentGzipSha256: 'unused' },
    arm: 'rewired',
    rewiringSeed: null,
    searchSeed: 1,
    delayMs
  });

  it('results are keyed by graphId and independent of which shard finishes which task first', async () => {
    // Reverse-order completion: task 0 is slowest, task 4 is fastest.
    const tasks = [0, 1, 2, 3, 4].map((i) => task(`t${i}`, (5 - i) * 15));
    const results = await runWorkerScheduler(tasks, 5, stubWorkerPath);
    expect([...results.keys()].sort()).toEqual(['t0', 't1', 't2', 't3', 't4']);
  });

  it('--shards 1 and --shards 3 assemble byte-identical output for the same task list', async () => {
    const plan = [0, 1, 2].map((i) => ({
      graphId: `fixture-${i}`,
      arm: 'rewired' as const,
      rewiringSeed: null,
      searchSeed: 1,
      bundlePath: 'unused',
      searchOutputPath: 'unused',
      expected: { arm: 'rewired' as const, binarySha256: 'unused', parentGzipSha256: 'unused' }
    }));
    const tasks = plan.map((entry, i) => task(`${entry.graphId}@${entry.searchSeed}`, (3 - i) * 10));

    const results1 = await runWorkerScheduler(tasks, 1, stubWorkerPath);
    const results3 = await runWorkerScheduler(tasks, 3, stubWorkerPath);

    const artifact1 = assembleEvaluatedArtifact(plan, results1);
    const artifact3 = assembleEvaluatedArtifact(plan, results3);

    expect(JSON.stringify(artifact1)).toBe(JSON.stringify(artifact3));
    expect(artifact1.graphs.map((g) => g.graphId)).toEqual(['fixture-0@1', 'fixture-1@1', 'fixture-2@1']);
  });

  it('a task-level error aborts the whole run rather than draining the rest of the queue', async () => {
    const tasks = [task('err'), ...Array.from({ length: 12 }, (_, i) => task(`t${i}`, 200))];
    await expect(runWorkerScheduler(tasks, 2, stubWorkerPath)).rejects.toThrow(/stub-induced failure/);
  });

  it('a worker killed by a signal is reported as a failure, not treated as a clean exit', async () => {
    const tasks = [task('kill'), task('t1', 50), task('t2', 50)];
    await expect(runWorkerScheduler(tasks, 3, stubWorkerPath)).rejects.toThrow(/exited unexpectedly/);
  });
});
