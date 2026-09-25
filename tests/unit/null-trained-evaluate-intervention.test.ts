// @vitest-environment node
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { gzipSync } from 'node:zlib';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { createTraceGraph } from '../fixtures/trace-graph';
import { createFixtureRewiredTraceGraph } from '../fixtures/trace-graph-rewire';
import { runExportArms } from '../../scripts/training/export-arms';
import {
  assembleInterventionRaw,
  buildInterventionTasks,
  parseNullTrainedInterventionEvaluateArgs,
  type NullTrainedInterventionEvaluateArgs
} from '../../scripts/null/null-trained-evaluate';
import type { NullSeedResult } from '../../scripts/null/null-worker';
import { writeTinyRunDir } from '../fixtures/trained-readout-run';

/**
 * Coverage for `scripts/null/null-trained-evaluate.ts`'s WP3
 * `--graph-list`/`--trained-dir` intervention mode
 * (`.agents/plans/pathway-interventions/03-evaluation.md`). Mirrors
 * `null-trained-evaluate.test.ts`'s existing fixture style (real
 * `runExportArms` bundles, `writeTinyRunDir` run directories) but builds a
 * REAL biological + rewired `.bin.gz` pair on disk (the trace-graph fixture,
 * gzip-encoded) so `--graph-list`'s own gzip-sha256 verification
 * (`resolveVerifiedInterventionGraphPath`) and the
 * `provenance.kind === 'rewired-artifact'` cross-check
 * (`assertBundleMatchesInterventionGraph`) are exercised against real
 * bytes, not the `--fixture-rewire` short-circuit the rewired-seed mode's
 * own tests use (that path never produces `'rewired-artifact'` provenance
 * at all).
 */

const sha256Hex = (data: Buffer) => createHash('sha256').update(data).digest('hex');

describe('parseNullTrainedInterventionEvaluateArgs', () => {
  it('applies defaults', () => {
    const args = parseNullTrainedInterventionEvaluateArgs(['--graph-list', 'index.json', '--runs', 'P:101']);
    expect(args.runs).toEqual([{ id: 'P', trainerSeed: 101 }]);
    expect(args.heldOutStart).toBe(30001);
    expect(args.heldOutCount).toBe(100);
    expect(args.ticks).toBe(1800);
    expect(args.hiddenSize).toBe(16);
    expect(args.shards).toBe(8);
    expect(args.trainedDir.endsWith('training/runs/interventions/trained')).toBe(true);
    expect(args.armsDir.endsWith('training/runs/interventions/arms')).toBe(true);
    expect(args.out.endsWith('training/runs/interventions/trained.json')).toBe(true);
  });

  it('parses multiple id:trainerSeed pairs, including the same id at different seeds', () => {
    const args = parseNullTrainedInterventionEvaluateArgs([
      '--graph-list',
      'index.json',
      '--runs',
      'P:101,P:202,P:303,C000:101'
    ]);
    expect(args.runs).toEqual([
      { id: 'P', trainerSeed: 101 },
      { id: 'P', trainerSeed: 202 },
      { id: 'P', trainerSeed: 303 },
      { id: 'C000', trainerSeed: 101 }
    ]);
  });

  it('rejects a malformed --runs entry', () => {
    expect(() => parseNullTrainedInterventionEvaluateArgs(['--graph-list', 'i.json', '--runs', 'P-101'])).toThrow(
      /comma-separated list of id:trainerSeed pairs/
    );
  });

  it('rejects a non-positive trainer seed', () => {
    expect(() => parseNullTrainedInterventionEvaluateArgs(['--graph-list', 'i.json', '--runs', 'P:0'])).toThrow(
      /must be a positive integer/
    );
  });

  it('rejects a duplicate id:trainerSeed pair', () => {
    expect(() =>
      parseNullTrainedInterventionEvaluateArgs(['--graph-list', 'i.json', '--runs', 'P:101,P:101'])
    ).toThrow(/lists "P:101" more than once/);
  });

  it('requires --graph-list', () => {
    expect(() => parseNullTrainedInterventionEvaluateArgs(['--runs', 'P:101'])).toThrow(/--graph-list is required/);
  });

  it('requires --runs', () => {
    expect(() => parseNullTrainedInterventionEvaluateArgs(['--graph-list', 'i.json'])).toThrow(/--runs is required/);
  });

  it('rejects --out without a .json extension', () => {
    expect(() =>
      parseNullTrainedInterventionEvaluateArgs(['--graph-list', 'i.json', '--runs', 'P:101', '--out', 'trained'])
    ).toThrow(/--out must end with "\.json"/);
  });
});

describe('buildInterventionTasks / assembleInterventionRaw', () => {
  let root: string;
  let graphListDir: string;
  let indexPath: string;
  let trainedDir: string;
  let armsDir: string;
  let d: number;

  const baseArgs = (overrides: Partial<NullTrainedInterventionEvaluateArgs> = {}): NullTrainedInterventionEvaluateArgs => ({
    graphList: indexPath,
    runs: [
      { id: 'P', trainerSeed: 101 },
      { id: 'P', trainerSeed: 202 },
      { id: 'C000', trainerSeed: 101 }
    ],
    trainedDir,
    armsDir,
    heldOutStart: 30001,
    heldOutCount: 3,
    ticks: 20,
    hiddenSize: 4,
    shards: 2,
    out: join(root, 'trained.json'),
    ...overrides
  });

  /** Writes a real biological + rewired `.bin.gz` pair, exports a real `rewired-artifact` bundle for `id`, and adds `id` to the graph-list index this suite writes to `indexPath`. */
  const addInterventionGraph = (
    entries: Array<{ id: string; path: string; gzipSha256: string }>,
    id: string,
    rewireSeed: number
  ): void => {
    const biological = createTraceGraph();
    const rewired = createFixtureRewiredTraceGraph(biological, rewireSeed);
    const graphPath = join(graphListDir, `${id}-biological.bin.gz`);
    const rewiredPath = join(graphListDir, `graphs`, `${id}.bin.gz`);
    mkdirSync(join(graphListDir, 'graphs'), { recursive: true });
    writeFileSync(graphPath, gzipSync(Buffer.from(encodeGraphBinary(biological))));
    const rewiredBytes = gzipSync(Buffer.from(encodeGraphBinary(rewired)));
    writeFileSync(rewiredPath, rewiredBytes);

    const result = runExportArms({
      graphPath,
      rewiredPath,
      fixtureRewire: false,
      fixtureRewireSeed: 0,
      outDir: join(armsDir, id)
    });
    const bundle = JSON.parse(readFileSync(resolve(result.outDir, 'rewired.json'), 'utf8')) as {
      sha256: string;
      D: number;
    };
    d = bundle.D;

    entries.push({ id, path: `graphs/${id}.bin.gz`, gzipSha256: sha256Hex(rewiredBytes) });

    for (const trainerSeed of id === 'P' ? [101, 202] : [101]) {
      writeTinyRunDir({
        dir: join(trainedDir, `${id}-seed${trainerSeed}`),
        arm: 'rewired',
        trainerSeed,
        D: bundle.D,
        H: 4,
        substeps: NEURAL_SUBSTEPS_PER_TICK,
        weightSeed: trainerSeed,
        armBundleSha256: bundle.sha256
      });
    }
  };

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'null-trained-evaluate-intervention-'));
    graphListDir = join(root, 'graph-list');
    trainedDir = join(root, 'trained');
    armsDir = join(root, 'arms');
    indexPath = join(graphListDir, 'index.json');
    mkdirSync(graphListDir, { recursive: true });

    const entries: Array<{ id: string; path: string; gzipSha256: string }> = [];
    addInterventionGraph(entries, 'P', 0);
    addInterventionGraph(entries, 'C000', 1);
    writeFileSync(indexPath, JSON.stringify({ entries, version: 1 }));
  });

  afterEach(() => rmSync(root, { recursive: true, force: true }));

  it('builds one task per (id, trainerSeed) pair, keyed "id-seedN", with expectedArm "rewired"', () => {
    const tasks = buildInterventionTasks(baseArgs());
    expect(tasks.map((t) => t.graphId).sort()).toEqual(['C000-seed101', 'P-seed101', 'P-seed202']);
    const p101 = tasks.find((t) => t.graphId === 'P-seed101')!;
    expect(p101.expectedArm).toBe('rewired');
    expect(p101.expectedTrainerSeed).toBe(101);
    expect(p101.expectedHiddenSize).toBe(4);
    const p202 = tasks.find((t) => t.graphId === 'P-seed202')!;
    expect(p202.expectedTrainerSeed).toBe(202);
    // Both P trainer seeds share the SAME arm bundle (the rewired arm
    // doesn't depend on trainer seed, only training does).
    expect(p202.armBundlePath).toBe(p101.armBundlePath);
  });

  it('throws when a requested id is not in the graph-list', () => {
    expect(() => buildInterventionTasks(baseArgs({ runs: [{ id: 'M1000', trainerSeed: 101 }] }))).toThrow(
      /id "M1000" not found in --graph-list/
    );
  });

  it('throws when a run directory is missing config.json', () => {
    rmSync(join(trainedDir, 'P-seed101', 'config.json'));
    expect(() => buildInterventionTasks(baseArgs())).toThrow(/config\.json/);
  });

  it('throws when the graph-list entry gzip sha256 no longer matches the on-disk graph (tampered/stale)', () => {
    writeFileSync(join(graphListDir, 'graphs', 'P.bin.gz'), Buffer.from('tampered'));
    expect(() => buildInterventionTasks(baseArgs())).toThrow(/gzip sha256 .* does not match/);
  });

  it('throws when an arm bundle\'s provenance does not match the graph-list entry (assertBundleMatchesInterventionGraph)', () => {
    const pBundleDir = resolve(armsDir, 'P');
    const hashDirName = readdirSync(pBundleDir, { withFileTypes: true }).find((e) => e.isDirectory())!.name;
    const bundlePath = join(pBundleDir, hashDirName, 'rewired.json');
    const bundle = JSON.parse(readFileSync(bundlePath, 'utf8')) as Record<string, unknown>;
    writeFileSync(
      bundlePath,
      JSON.stringify({ ...bundle, provenance: { kind: 'rewired-fixture-only-swap', seed: 0 } })
    );
    expect(() => buildInterventionTasks(baseArgs())).toThrow(/does not match --graph-list/);
  });

  it('assembleInterventionRaw: runs sorted by id then trainerSeed, graphListSha256/d/evaluatorGitRev present', () => {
    const args = baseArgs();
    const tasks = buildInterventionTasks(args);
    const seeds = [30001, 30002, 30003];
    const results = new Map<string, readonly NullSeedResult[]>(
      tasks.map((t) => [t.graphId, seeds.map((seed) => ({ seed, movementScore: 0.5, foodPickups: 1, hazardContacts: 0 }))])
    );
    const raw = assembleInterventionRaw(args, tasks, results);
    expect(raw.runs.map((r) => `${r.id}:${r.trainerSeed}`)).toEqual(['C000:101', 'P:101', 'P:202']);
    expect(raw.d).toBe(d);
    expect(raw.graphListSha256).toBe(sha256Hex(readFileSync(indexPath)));
    expect(raw.evaluatorGitRev === null || typeof raw.evaluatorGitRev === 'string').toBe(true);
    expect(raw.cemConfig).toBeNull(); // writeTinyRunDir wasn't given any cemConfig fields in this fixture
    expect(raw.cemConfigWarnings).toEqual([]);
  });

  it('assembleInterventionRaw throws when a task has no matching result', () => {
    const args = baseArgs();
    const tasks = buildInterventionTasks(args);
    expect(() => assembleInterventionRaw(args, tasks, new Map())).toThrow(/missing results for/);
  });
});
