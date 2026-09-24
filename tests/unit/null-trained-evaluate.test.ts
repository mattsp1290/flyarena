import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { runExportArms } from '../../scripts/training/export-arms';
import {
  assembleRaw,
  buildTasks,
  parseNullTrainedEvaluateArgs,
  type NullTrainedEvaluateArgs
} from '../../scripts/null/null-trained-evaluate';
import type { NullSeedResult } from '../../scripts/null/null-worker';
import { writeTinyRunDir } from '../fixtures/trained-readout-run';

/**
 * Coverage for `scripts/null/null-trained-evaluate.ts`'s pure/file-system
 * functions (`parseNullTrainedEvaluateArgs`, `buildTasks`, `assembleRaw`) —
 * a dual-review finding: the first version of this branch had zero test
 * coverage for this driver beyond a manual `--dry-run-fixture` run. Every
 * fixture here uses `runExportArms`'s real `--fixture-rewire` path (the
 * same one `train-sample.sh --dry-run-fixture` exercises) to produce real,
 * self-certifying arm bundles, and `writeTinyRunDir` (already used by
 * `evaluate.ts`'s own test suite) to produce real run directories —
 * never hand-typed bundle/config JSON that might not satisfy the shapes
 * `readRunDir`/`computeArmBundleSha256` actually expect.
 */

describe('parseNullTrainedEvaluateArgs', () => {
  it('applies defaults', () => {
    const args = parseNullTrainedEvaluateArgs([]);
    expect(args.rewiredSeedStart).toBe(0);
    expect(args.rewiredSeedCount).toBe(20);
    expect(args.replicaSeed).toBe(101);
    expect(args.biologicalTrainerSeeds).toEqual([101, 202, 303]);
    expect(args.heldOutStart).toBe(30001);
    expect(args.heldOutCount).toBe(100);
    expect(args.ticks).toBe(1800);
    expect(args.hiddenSize).toBe(16);
    expect(args.shards).toBe(8);
  });

  it('parses --biological-trainer-seeds as a comma-separated list', () => {
    const args = parseNullTrainedEvaluateArgs(['--biological-trainer-seeds', '11,22,33']);
    expect(args.biologicalTrainerSeeds).toEqual([11, 22, 33]);
  });

  it('rejects a non-integer/non-positive entry in --biological-trainer-seeds', () => {
    expect(() => parseNullTrainedEvaluateArgs(['--biological-trainer-seeds', '1,x,3'])).toThrow(
      /comma-separated list of positive integers/
    );
    expect(() => parseNullTrainedEvaluateArgs(['--biological-trainer-seeds', '1,-2'])).toThrow(
      /comma-separated list of positive integers/
    );
  });

  it('parses --hidden-size and --replica-seed overrides', () => {
    const args = parseNullTrainedEvaluateArgs(['--hidden-size', '32', '--replica-seed', '999']);
    expect(args.hiddenSize).toBe(32);
    expect(args.replicaSeed).toBe(999);
  });

  it('rejects --out without a .json extension', () => {
    expect(() => parseNullTrainedEvaluateArgs(['--out', 'trained'])).toThrow(/--out must end with "\.json"/);
  });

  it('rejects an unknown flag', () => {
    expect(() => parseNullTrainedEvaluateArgs(['--bogus'])).toThrow(/Unknown argument/);
  });
});

describe('buildTasks / assembleRaw (fixture run directories)', () => {
  let root: string;
  let rewiredTrainedDir: string;
  let rewiredArmsDir: string;
  let biologicalRunsDir: string;
  let biologicalArmsDir: string;
  let biologicalBundleSha256: string;
  let rewiredBundleSha256ForSeed: Record<number, string>;
  let d: number;

  const baseArgs = (overrides: Partial<NullTrainedEvaluateArgs> = {}): NullTrainedEvaluateArgs => ({
    rewiredSeedStart: 0,
    rewiredSeedCount: 2,
    replicaSeed: 101,
    biologicalTrainerSeeds: [101],
    rewiredTrainedDir,
    rewiredArmsDir,
    biologicalRunsDir,
    biologicalArmsDir,
    heldOutStart: 30001,
    heldOutCount: 3,
    ticks: 20,
    hiddenSize: 4,
    shards: 2,
    out: join(root, 'trained.json'),
    ...overrides
  });

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'null-trained-evaluate-'));
    rewiredTrainedDir = join(root, 'trained');
    rewiredArmsDir = join(root, 'arms');
    biologicalRunsDir = join(root, 'production');
    biologicalArmsDir = join(root, 'bio-arms');

    // Two rewired seeds' arm bundles (export-arms.ts's real --fixture-rewire
    // path -- same trace-graph fixture train-sample.sh --dry-run-fixture uses).
    // Each fixtureRewireSeed produces a DIFFERENT bundle (and therefore a
    // different self-certifying sha256), so each seed's own hash is kept
    // separately rather than in one shared variable the loop would overwrite.
    rewiredBundleSha256ForSeed = {};
    for (const seed of [0, 1]) {
      const result = runExportArms({
        fixtureRewire: true,
        fixtureRewireSeed: seed,
        outDir: join(rewiredArmsDir, `seed${seed}`)
      });
      const bundle = JSON.parse(readFileSync(resolve(result.outDir, 'rewired.json'), 'utf8')) as { sha256: string; D: number };
      rewiredBundleSha256ForSeed[seed] = bundle.sha256;
      d = bundle.D;
      writeTinyRunDir({
        dir: join(rewiredTrainedDir, `seed${seed}`),
        arm: 'rewired',
        trainerSeed: 101,
        D: bundle.D,
        H: 4,
        substeps: NEURAL_SUBSTEPS_PER_TICK,
        weightSeed: seed + 1,
        armBundleSha256: bundle.sha256
      });
    }

    // The biological arm bundle (a single hash directory, like a real
    // `npm run training:export-arms` invocation against the full graph).
    const bioResult = runExportArms({ fixtureRewire: true, fixtureRewireSeed: 0, outDir: biologicalArmsDir });
    const bioBundle = JSON.parse(readFileSync(resolve(bioResult.outDir, 'biological.json'), 'utf8')) as {
      sha256: string;
      D: number;
    };
    biologicalBundleSha256 = bioBundle.sha256;
    writeTinyRunDir({
      dir: join(biologicalRunsDir, 'biological-101'),
      arm: 'biological',
      trainerSeed: 101,
      D: bioBundle.D,
      H: 4,
      substeps: NEURAL_SUBSTEPS_PER_TICK,
      weightSeed: 42,
      armBundleSha256: bioBundle.sha256
    });
  });

  afterEach(() => rmSync(root, { recursive: true, force: true }));

  it('builds one task per rewired seed plus one per biological trainer seed, with the right expected identity', () => {
    const tasks = buildTasks(baseArgs());
    expect(tasks.map((t) => t.graphId)).toEqual(['rewired-0', 'rewired-1', 'biological-101']);

    const rewired0 = tasks.find((t) => t.graphId === 'rewired-0')!;
    expect(rewired0.expectedArm).toBe('rewired');
    expect(rewired0.expectedTrainerSeed).toBe(101); // args.replicaSeed
    expect(rewired0.expectedSubsteps).toBe(NEURAL_SUBSTEPS_PER_TICK);
    expect(rewired0.expectedHiddenSize).toBe(4);
    expect(rewired0.heldOutSeeds).toEqual([30001, 30002, 30003]);
    expect(rewired0.ticks).toBe(20);

    const biological101 = tasks.find((t) => t.graphId === 'biological-101')!;
    expect(biological101.expectedArm).toBe('biological');
    expect(biological101.expectedTrainerSeed).toBe(101);
  });

  it('throws when a rewired seed has no config.json', () => {
    rmSync(join(rewiredTrainedDir, 'seed0', 'config.json'));
    expect(() => buildTasks(baseArgs())).toThrow(/config\.json/);
  });

  it('throws when a rewired seed has no theta_final.npy', () => {
    rmSync(join(rewiredTrainedDir, 'seed1', 'theta_final.npy'));
    expect(() => buildTasks(baseArgs())).toThrow(/theta_final\.npy/);
  });

  it('throws when a biological trainer seed run directory is missing entirely', () => {
    expect(() => buildTasks(baseArgs({ biologicalTrainerSeeds: [999] }))).toThrow(/config\.json/);
  });

  it('throws when an arms subdirectory has zero hash subdirectories', () => {
    // A run directory must exist for seed 5 too, so buildTasks reaches its
    // arms-subdirectory check rather than failing earlier on a missing
    // config.json/theta_final.npy.
    writeTinyRunDir({
      dir: join(rewiredTrainedDir, 'seed5'),
      arm: 'rewired',
      trainerSeed: 101,
      D: d,
      H: 4,
      substeps: NEURAL_SUBSTEPS_PER_TICK,
      weightSeed: 5
    });
    mkdirSync(join(rewiredArmsDir, 'seed5'), { recursive: true });
    expect(() => buildTasks(baseArgs({ rewiredSeedStart: 5, rewiredSeedCount: 1 }))).toThrow(
      /expected exactly one subdirectory/
    );
  });

  it('throws when an arms subdirectory has more than one hash subdirectory', () => {
    mkdirSync(join(rewiredArmsDir, 'seed0', 'a-second-hash-dir'), { recursive: true });
    expect(() => buildTasks(baseArgs({ rewiredSeedCount: 1 }))).toThrow(/expected exactly one subdirectory/);
  });

  it('throws (D-equality gate) when a bundle reports a different D than the rest', () => {
    // Hand-edit one bundle's D field directly -- buildTasks's D-equality
    // check (assertMatchingD) only reads this field, it never re-verifies
    // the bundle's self-certifying sha256 (that verification happens later,
    // inside null-trained-worker.ts, at actual-scoring time), so this is a
    // valid way to exercise the check without needing a bundle whose sha256
    // still matches a hand-edited D. The hash subdirectory name is looked
    // up directly (not via the shared `rewiredBundleSha256` fixture
    // variable, which `beforeEach`'s per-seed loop overwrites on each
    // iteration and so only ever holds the LAST seed's hash).
    const seed0ArmsDir = join(rewiredArmsDir, 'seed0');
    const seed0HashDir = readdirSync(seed0ArmsDir, { withFileTypes: true }).find((e) => e.isDirectory())!.name;
    const bundlePath = join(seed0ArmsDir, seed0HashDir, 'rewired.json');
    const bundle = JSON.parse(readFileSync(bundlePath, 'utf8')) as Record<string, unknown>;
    writeFileSync(bundlePath, JSON.stringify({ ...bundle, D: (bundle.D as number) + 1 }));
    expect(() => buildTasks(baseArgs())).toThrow(/D \(output-neuron count\) differs/);
  });

  it('assembleRaw: rewired sorted by seed, biological sorted by trainerSeed, d/bigqMergeCommit/evaluatorGitRev present', () => {
    const args = baseArgs();
    const tasks = buildTasks(args);
    const seeds = [30001, 30002, 30003];
    const results = new Map<string, readonly NullSeedResult[]>(
      tasks.map((t) => [t.graphId, seeds.map((seed) => ({ seed, movementScore: 1, foodPickups: 0, hazardContacts: 0 }))])
    );

    const raw = assembleRaw(args, tasks, results);
    expect(raw.version).toBe(1);
    expect(raw.rewired.map((e) => e.seed)).toEqual([0, 1]);
    expect(raw.biological.map((e) => e.trainerSeed)).toEqual([101]);
    expect(raw.d).toBe(d);
    expect(raw.bigqMergeCommit).toBe('69b610d4a9da11b12a7ac180997e702cf9fd2a4f');
    expect(typeof raw.evaluatorGitRev === 'string' || raw.evaluatorGitRev === null).toBe(true);
    expect(raw.cemConfig).toBeNull(); // writeTinyRunDir wasn't given any cemConfig fields in this fixture
    expect(raw.cemConfigWarnings).toEqual([]);
  });

  it('assembleRaw: reconcileCemConfig warns when one run disagrees with the baseline CEM config', () => {
    // Rebuild seed0/seed1's run dirs with an explicit (agreeing) CEM config,
    // and give biological-101 a disagreeing one.
    rmSync(join(rewiredTrainedDir, 'seed0'), { recursive: true, force: true });
    writeTinyRunDir({
      dir: join(rewiredTrainedDir, 'seed0'),
      arm: 'rewired',
      trainerSeed: 101,
      D: d,
      H: 4,
      substeps: NEURAL_SUBSTEPS_PER_TICK,
      weightSeed: 1,
      armBundleSha256: rewiredBundleSha256ForSeed[0],
      cemConfig: { population: 128, elites: 32, generations: 150 }
    });
    rmSync(join(rewiredTrainedDir, 'seed1'), { recursive: true, force: true });
    writeTinyRunDir({
      dir: join(rewiredTrainedDir, 'seed1'),
      arm: 'rewired',
      trainerSeed: 101,
      D: d,
      H: 4,
      substeps: NEURAL_SUBSTEPS_PER_TICK,
      weightSeed: 2,
      armBundleSha256: rewiredBundleSha256ForSeed[1],
      cemConfig: { population: 128, elites: 32, generations: 150 }
    });
    rmSync(join(biologicalRunsDir, 'biological-101'), { recursive: true, force: true });
    writeTinyRunDir({
      dir: join(biologicalRunsDir, 'biological-101'),
      arm: 'biological',
      trainerSeed: 101,
      D: d,
      H: 4,
      substeps: NEURAL_SUBSTEPS_PER_TICK,
      weightSeed: 3,
      armBundleSha256: biologicalBundleSha256,
      cemConfig: { population: 64, elites: 32, generations: 150 } // disagreeing population
    });

    const args = baseArgs();
    const tasks = buildTasks(args);
    const seeds = [30001, 30002, 30003];
    const results = new Map<string, readonly NullSeedResult[]>(
      tasks.map((t) => [t.graphId, seeds.map((seed) => ({ seed, movementScore: 1, foodPickups: 0, hazardContacts: 0 }))])
    );

    const raw = assembleRaw(args, tasks, results);
    expect(raw.cemConfig).toMatchObject({ population: 128, elites: 32, generations: 150 });
    expect(raw.cemConfigWarnings).toHaveLength(1);
    expect(raw.cemConfigWarnings[0]).toMatch(/biological-101/);
  });

  it('assembleRaw throws when a result is missing for a task', () => {
    const args = baseArgs();
    const tasks = buildTasks(args);
    const results = new Map<string, readonly NullSeedResult[]>(); // empty -- every graphId missing
    expect(() => assembleRaw(args, tasks, results)).toThrow(/missing results for/);
  });
});
