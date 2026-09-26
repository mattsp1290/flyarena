import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  buildRepertoirePlan,
  planEntryKey,
  verifyBundleIdentity,
  type RepertoirePlanEntry,
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

const FAKE_SEARCH_OPTIONS = { population: 64, generations: 24, ticks: 900 };

describe('repertoire-plan: buildRepertoirePlan', () => {
  const inputsFor = (seedCount: number): RepertoirePlanInputs => ({
    rewireIndex: fakeRewireIndex(seedCount),
    biologicalBinarySha256: 'bio-binary-sha',
    biologicalGzipSha256: 'bio-gzip-sha',
    disconnectedBinarySha256: 'disc-binary-sha',
    armsDir: '/arms',
    searchDir: '/search',
    searchOptions: FAKE_SEARCH_OPTIONS
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

  it('every entry carries the shipped search budget, unmodified', () => {
    const plan = buildRepertoirePlan(inputsFor(20));
    for (const entry of plan) expect(entry.expectedSearchOptions).toEqual(FAKE_SEARCH_OPTIONS);
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

describe('repertoire-plan: planEntryKey', () => {
  it('builds the composite <graphId>@<searchSeed> key', () => {
    expect(planEntryKey({ graphId: 'rewired-3', searchSeed: 1730 })).toBe('rewired-3@1730');
    expect(planEntryKey({ graphId: 'biological', searchSeed: 1729 })).toBe('biological@1729');
  });
});

describe('repertoire-evaluate: buildRepertoireTasks', () => {
  // Regression coverage for the dual-review finding that a task's wire `key`
  // (composite `<graphId>@<seed>`) and its plain `graphId` were once the
  // same field, and that a task's `expectedOptions.seed` must always equal
  // its own plan entry's `searchSeed` -- both properties this function
  // itself is responsible for, over the real 46-entry plan (not a 3-entry
  // fixture), so a regression on any one of the 46 pairs is caught.
  it('every task has key === planEntryKey(entry), graphId === entry.graphId (never the composite key), and expectedOptions.seed === searchSeed', () => {
    const plan = buildRepertoirePlan({
      rewireIndex: fakeRewireIndex(20),
      biologicalBinarySha256: 'bio-binary-sha',
      biologicalGzipSha256: 'bio-gzip-sha',
      disconnectedBinarySha256: 'disc-binary-sha',
      armsDir: '/arms',
      searchDir: '/search',
      searchOptions: FAKE_SEARCH_OPTIONS
    });
    const tasks = buildRepertoireTasks(plan);
    expect(tasks).toHaveLength(46);
    for (const [i, task] of tasks.entries()) {
      const entry = plan[i];
      expect(task.key).toBe(planEntryKey(entry));
      expect(task.graphId).toBe(entry.graphId);
      expect(task.key).not.toBe(task.graphId); // every entry has a non-empty searchSeed suffix
      expect(task.expectedOptions.seed).toBe(entry.searchSeed);
      expect(task.expectedOptions).toMatchObject(FAKE_SEARCH_OPTIONS);
    }
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

// Shared by both the in-process `runRepertoireTask` tests and the real
// forked-worker harness test below -- a minimal, `validateSearch`-passing
// one-candidate search artifact for `bundle`, matching
// `tests/unit/atlas.test.ts`'s own `buildSearchArtifact` fixture pattern.
const FIXTURE_RUNTIME_CONFIG = Object.fromEntries(
  Object.entries(ARENA_CONFIG).map(([key, value]) => [key.replace(/[A-Z]/g, (c) => '_' + c.toLowerCase()), value])
);
const FIXTURE_SEARCH_OPTIONS = { seed: 1, population: 4, generations: 1, ticks: 30 };

const buildSearchArtifact = (bundle: SerializedArmBundle, candidateCount = 1): SearchArtifact => {
  const d = bundle.D;
  const hiddenSize = 8;
  const parameterCount = readoutParameterCount(d, hiddenSize);
  const candidates = Array.from({ length: candidateCount }, (_, id) => ({
    id,
    theta: Array.from({ length: parameterCount }, (_, i) => (((i * 37 + id * 11) % 101) / 101 - 0.5) * 0.2),
    quality: 0,
    coverage: 0,
    turning: 0
  }));
  return {
    schemaVersion: 1,
    modelVersion: ATLAS_VERSION,
    options: FIXTURE_SEARCH_OPTIONS,
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
    candidates,
    history: [{ generation: 1, occupied: candidateCount, bestQuality: 0 }],
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
      config: FIXTURE_RUNTIME_CONFIG
    }
  };
};

const binarySha = (bundle: SerializedArmBundle) => sha256Hex(new Uint8Array(encodeGraphBinary(deserializeArmBundle(bundle))));

describe('repertoire-task: runRepertoireTask', () => {
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

  it('rejects a search whose bundle is rewired when the plan expected biological', async () => {
    const searchPath = join(root, 'wrong-arm-biological.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(rewired)));
    await expect(
      runRepertoireTask({
        key: 'biological@1729',
        graphId: 'biological',
        searchPath,
        expected: { arm: 'biological', binarySha256: binarySha(rewired), parentGzipSha256: rewired.graphArtifactSha256 },
        expectedOptions: FIXTURE_SEARCH_OPTIONS,
        arm: 'biological',
        rewiringSeed: null,
        searchSeed: 1
      })
    ).rejects.toThrow('arm mismatch');
  });

  it('rejects a search whose bundle is biological when the plan expected rewired', async () => {
    const searchPath = join(root, 'wrong-arm-rewired.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(biological)));
    await expect(
      runRepertoireTask({
        key: 'rewired-9@1729',
        graphId: 'rewired-9',
        searchPath,
        expected: { arm: 'rewired', binarySha256: binarySha(biological), parentGzipSha256: biological.graphArtifactSha256 },
        expectedOptions: FIXTURE_SEARCH_OPTIONS,
        arm: 'rewired',
        rewiringSeed: 9,
        searchSeed: 1
      })
    ).rejects.toThrow('arm mismatch');
  });

  it('rejects a search whose recorded seed does not match the planned search seed, even with correct graph identity', async () => {
    const searchPath = join(root, 'wrong-seed.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(rewired)));
    await expect(
      runRepertoireTask({
        key: 'rewired-9@1730',
        graphId: 'rewired-9',
        searchPath,
        expected: { arm: 'rewired', binarySha256: binarySha(rewired), parentGzipSha256: rewired.graphArtifactSha256 },
        // The fixture search file's own options.seed is 1, not 1730 -- this
        // simulates a copied/renamed search file (a dual-review finding:
        // graph identity alone is shared across every search seed of the
        // same graph, so it cannot catch this on its own).
        expectedOptions: { ...FIXTURE_SEARCH_OPTIONS, seed: 1730 },
        arm: 'rewired',
        rewiringSeed: 9,
        searchSeed: 1730
      })
    ).rejects.toThrow('Search options mismatch');
  });

  it('rejects a search whose recorded budget is smaller than the shipped budget, even with correct graph identity and seed', async () => {
    const searchPath = join(root, 'shrunk-budget.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(rewired)));
    await expect(
      runRepertoireTask({
        key: 'rewired-9@1',
        graphId: 'rewired-9',
        searchPath,
        expected: { arm: 'rewired', binarySha256: binarySha(rewired), parentGzipSha256: rewired.graphArtifactSha256 },
        // Shipped budget is population 64 -- the fixture search file was
        // (legitimately, for test speed) generated at population 4.
        expectedOptions: { ...FIXTURE_SEARCH_OPTIONS, population: 64 },
        arm: 'rewired',
        rewiringSeed: 9,
        searchSeed: 1
      })
    ).rejects.toThrow('Search options mismatch');
  });

  it('returns a RepertoireEvaluatedEntry with the plain graphId, heldoutOwn (never a "biological"-named field), and the verified searchOptions', async () => {
    const searchPath = join(root, 'valid-rewired.json');
    writeFileSync(searchPath, JSON.stringify(buildSearchArtifact(rewired)));
    const entry = await runRepertoireTask({
      key: 'rewired-9@1',
      graphId: 'rewired-9',
      searchPath,
      expected: { arm: 'rewired', binarySha256: binarySha(rewired), parentGzipSha256: rewired.graphArtifactSha256 },
      expectedOptions: FIXTURE_SEARCH_OPTIONS,
      arm: 'rewired',
      rewiringSeed: 9,
      searchSeed: 1
    });
    expect(entry.graphId).toBe('rewired-9');
    expect(entry.occupied).toBe(1);
    expect(entry.gpuArchiveSize).toBe(1);
    expect(entry.searchOptions).toEqual(FIXTURE_SEARCH_OPTIONS);
    expect(entry.cells).toHaveLength(1);
    const cellResult: RepertoireCellResult = entry.cells[0];
    expect(cellResult.heldoutOwn).toHaveLength(12);
    expect('biological' in cellResult).toBe(false);
  });
});

describe('repertoire-evaluate: runWorkerScheduler shard determinism / failure paths (stub worker)', () => {
  // `repertoire-worker.ts` itself is a real `.ts` file that needs a `tsx`
  // loader to resolve its own extensionless `./repertoire-task` import when
  // forked. A plain `vitest run` invocation's own process has no such
  // loader in its `execArgv` to inherit, so these tests exercise the
  // *scheduler*'s own mechanics directly against a plain-`.mjs` stub worker
  // instead -- shard-count/completion-order independence and fail-fast
  // abort -- mirroring `tests/unit/null-evaluate.test.ts`'s own
  // "runShardedEvaluation: failure/abort paths (stub worker)" section. The
  // real forked worker (a genuine `.ts` fork, not a stub) is exercised
  // separately below, via a `tsx`-launched subprocess harness, and the real
  // per-task pipeline (verify + evaluate + heldoutOwn renaming + arm/seed/
  // budget-mismatch rejection) is covered above by calling
  // `runRepertoireTask` directly, in-process, with no forking involved.
  const stubWorkerPath = resolve(dirname(fileURLToPath(import.meta.url)), '../fixtures/repertoire-stub-worker.mjs');

  /** `delayMs` is a stub-worker-only extension of the wire protocol, not part of the real `RepertoireWorkerTask`. Here `key` and `graphId` are deliberately the same string -- these tests exercise the scheduler's own key-matching, not the plain-vs-composite distinction (covered by `planEntryKey`'s and `runRepertoireTask`'s own tests above). */
  const task = (key: string, delayMs = 0): RepertoireWorkerTask & { delayMs: number } => ({
    key,
    graphId: key,
    searchPath: 'unused',
    expected: { arm: 'rewired', binarySha256: 'unused', parentGzipSha256: 'unused' },
    expectedOptions: { seed: 1, population: 4, generations: 1, ticks: 30 },
    arm: 'rewired',
    rewiringSeed: null,
    searchSeed: 1,
    delayMs
  });

  it('results are keyed by key and independent of which shard finishes which task first', async () => {
    // Reverse-order completion: task 0 is slowest, task 4 is fastest.
    const tasks = [0, 1, 2, 3, 4].map((i) => task(`t${i}`, (5 - i) * 15));
    const results = await runWorkerScheduler(tasks, 5, stubWorkerPath);
    expect([...results.keys()].sort()).toEqual(['t0', 't1', 't2', 't3', 't4']);
  });

  it('--shards 1 and --shards 3 assemble byte-identical output for the same task list', async () => {
    const plan: RepertoirePlanEntry[] = [0, 1, 2].map((i) => ({
      graphId: `fixture-${i}`,
      arm: 'rewired' as const,
      rewiringSeed: null,
      searchSeed: 1,
      bundlePath: 'unused',
      searchOutputPath: 'unused',
      expected: { arm: 'rewired' as const, binarySha256: 'unused', parentGzipSha256: 'unused' },
      expectedSearchOptions: { population: 4, generations: 1, ticks: 30 }
    }));
    // `buildRepertoireTasks` -- the real production function, not this
    // section's own ad-hoc `task()` helper -- so this test actually
    // exercises the plain-`graphId`-vs-composite-`key` split it builds,
    // with per-task `delayMs` added only for the stub's own timing
    // extension to the wire protocol.
    const tasks = buildRepertoireTasks(plan).map((t, i) => ({ ...t, delayMs: (3 - i) * 10 }));

    const results1 = await runWorkerScheduler(tasks, 1, stubWorkerPath);
    const results3 = await runWorkerScheduler(tasks, 3, stubWorkerPath);

    const artifact1 = assembleEvaluatedArtifact(plan, results1);
    const artifact3 = assembleEvaluatedArtifact(plan, results3);

    expect(JSON.stringify(artifact1)).toBe(JSON.stringify(artifact3));
    // The persisted `graphId` on each entry is the plain id `buildRepertoireTasks`
    // set (`entry.graphId`), never the composite `key` (`planEntryKey(entry)`)
    // -- pins the fix for the dual-review finding that an earlier version
    // leaked the composite key into this field.
    expect(artifact1.graphs.map((g) => g.graphId)).toEqual(['fixture-0', 'fixture-1', 'fixture-2']);
    expect(tasks.map((t) => t.key)).toEqual(['fixture-0@1', 'fixture-1@1', 'fixture-2@1']);
  });

  it('a task-level error aborts the whole run quickly, not after the full queue drains', async () => {
    // Self-calibrating regression floor (mirrors
    // `tests/unit/null-evaluate.test.ts`'s identically-reasoned stub-worker
    // test): with 2 shards and 'err' dispatched first, a correct abort lets
    // at most one delayed task per shard start before `abortAll` fires, so
    // wall time is dominated by fork/IPC overhead alone, independent of
    // `remainingTaskCount`. A regressed scheduler that keeps draining the
    // queue after the first error would instead split all 12 remaining
    // 200ms tasks across 2 shards and take about 1200ms -- asserting well
    // under that floor catches the regression without a flaky tight bound.
    const delayMs = 200;
    const remainingTaskCount = 12;
    const tasks = [task('err'), ...Array.from({ length: remainingTaskCount }, (_, i) => task(`t${i}`, delayMs))];
    const regressionFloorMs = (remainingTaskCount / 2) * delayMs; // 1200ms
    const started = Date.now();
    await expect(runWorkerScheduler(tasks, 2, stubWorkerPath)).rejects.toThrow(/stub-induced failure/);
    const elapsedMs = Date.now() - started;
    expect(elapsedMs).toBeLessThan(regressionFloorMs / 2); // generous 600ms bound, well below the 1200ms floor
  });

  it('a worker killed by a signal is reported as a failure, not treated as a clean exit', async () => {
    const tasks = [task('kill'), task('t1', 50), task('t2', 50)];
    await expect(runWorkerScheduler(tasks, 3, stubWorkerPath)).rejects.toThrow(/exited unexpectedly/);
  });
});

describe('repertoire-evaluate: real forked worker (tsx subprocess harness)', () => {
  // Unlike the stub-worker section above, this spawns a small harness
  // script (`tests/fixtures/repertoire-scheduler-harness.mjs`) as its OWN
  // subprocess via `node --import tsx`, exactly the way
  // `tests/unit/null-evaluate.test.ts` spawns `null-evaluate.ts`'s CLI to
  // get a real `.ts`-loader-carrying `process.execArgv` for its forked
  // children to inherit. This exercises the REAL `repertoire-worker.ts`
  // (a genuine fork of a `.ts` file, not a stub) end to end, on real
  // fixture search files -- confirmed working during dual review; a
  // shard-determinism test using only the stub worker cannot exercise this
  // path, since the stub's output is identical by construction regardless
  // of what the real worker does.
  let root: string;
  let biological: SerializedArmBundle;
  let disconnected: SerializedArmBundle;
  let outDir: string;
  const searchPaths = new Map<string, string>();

  const harnessPath = resolve(dirname(fileURLToPath(import.meta.url)), '../fixtures/repertoire-scheduler-harness.mjs');
  const evaluateModulePath = resolve(process.cwd(), 'scripts/atlas/repertoire-evaluate.ts');
  const workerPath = resolve(process.cwd(), 'scripts/atlas/repertoire-worker.ts');

  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'repertoire-real-worker-'));
    const result = runExportArms({ outDir: root, fixtureRewire: true, fixtureRewireSeed: 2 });
    outDir = result.outDir;
    biological = JSON.parse(readFileSync(join(outDir, 'biological.json'), 'utf8'));
    disconnected = JSON.parse(readFileSync(join(outDir, 'disconnected.json'), 'utf8'));

    for (const [key, bundle, count] of [
      ['fixture-a@1', biological, 1],
      ['fixture-b@1', disconnected, 2],
      ['fixture-c@1', biological, 1]
    ] as const) {
      const path = join(root, `${key.replace('@', '-')}.json`);
      writeFileSync(path, JSON.stringify(buildSearchArtifact(bundle, count)));
      searchPaths.set(key, path);
    }
  });

  afterAll(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  const buildTasks = (): (RepertoireWorkerTask & Record<string, unknown>)[] => [
    {
      key: 'fixture-a@1',
      graphId: 'fixture-a',
      searchPath: searchPaths.get('fixture-a@1')!,
      expected: { arm: 'biological' as const, binarySha256: binarySha(biological), parentGzipSha256: biological.graphArtifactSha256 },
      expectedOptions: FIXTURE_SEARCH_OPTIONS,
      arm: 'biological' as const,
      rewiringSeed: null,
      searchSeed: 1
    },
    {
      key: 'fixture-b@1',
      graphId: 'fixture-b',
      searchPath: searchPaths.get('fixture-b@1')!,
      expected: { arm: 'disconnected' as const, binarySha256: binarySha(disconnected), parentGzipSha256: disconnected.graphArtifactSha256 },
      expectedOptions: FIXTURE_SEARCH_OPTIONS,
      arm: 'disconnected' as const,
      rewiringSeed: null,
      searchSeed: 1
    },
    {
      key: 'fixture-c@1',
      graphId: 'fixture-c',
      searchPath: searchPaths.get('fixture-c@1')!,
      expected: { arm: 'biological' as const, binarySha256: binarySha(biological), parentGzipSha256: biological.graphArtifactSha256 },
      expectedOptions: FIXTURE_SEARCH_OPTIONS,
      arm: 'biological' as const,
      rewiringSeed: null,
      searchSeed: 1
    }
  ];

  const runHarness = (shards: number, tasksPath: string) =>
    spawnSync(
      process.execPath,
      ['--import', 'tsx', harnessPath, evaluateModulePath, workerPath, String(shards), tasksPath],
      { encoding: 'utf8', timeout: 60_000, cwd: process.cwd() }
    );

  it('forks the real repertoire-worker.ts and produces byte-identical results for --shards 1 and --shards 3', () => {
    const tasksPath = join(root, 'tasks.json');
    writeFileSync(tasksPath, JSON.stringify(buildTasks()));

    const result1 = runHarness(1, tasksPath);
    const result3 = runHarness(3, tasksPath);

    expect(result1.status, result1.stderr).toBe(0);
    expect(result3.status, result3.stderr).toBe(0);

    const entries1 = JSON.parse(result1.stdout) as [string, unknown][];
    const entries3 = JSON.parse(result3.stdout) as [string, unknown][];
    expect(entries1.map(([key]) => key)).toEqual(['fixture-a@1', 'fixture-b@1', 'fixture-c@1']);
    expect(JSON.stringify(entries1)).toBe(JSON.stringify(entries3));

    const [, biologicalEntryA] = entries1[0] as [string, { readonly occupied: number; readonly graphId: string }];
    expect(biologicalEntryA.graphId).toBe('fixture-a');
    expect(biologicalEntryA.occupied).toBe(1);
  }, 60_000);
});
