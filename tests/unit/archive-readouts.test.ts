import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { gzipSync } from 'node:zlib';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { readoutFromFlat } from '../../src/lib/connectome/readout-serialization';
import { readNpyFloat32Array } from '../../scripts/training/npy';
import {
  arenaTaskFingerprintOf,
  assertArmMatchesGraphId,
  assertTaskGraphMatchesDefaultGraph,
  buildArchivedReadout,
  DEFAULT_MANIFEST_PATH,
  kindForEntry,
  loadExistingArchive,
  mergeReadouts,
  parseArgs,
  parseSourceArg,
  parseTaskInterventionRawScoresArg,
  runArchiveReadouts,
  taskInterventionRawScoresOutPath,
  writeArchive,
  type ArchivedReadout,
  type ArchiveReadoutsArgs
} from '../../scripts/attribution/archive-readouts';
import { writeTinyRunDir } from '../fixtures/trained-readout-run';

/**
 * Coverage for `scripts/attribution/archive-readouts.ts`
 * (`.agents/plans/readout-attribution/01-archive-and-types.md`'s WP1). Uses
 * `writeTinyRunDir` (the existing tiny synthetic run-directory fixture --
 * `tests/fixtures/trained-readout-run.ts`), matching
 * `tests/unit/null-trained-evaluate.test.ts`'s own convention.
 *
 * `copyRawInterventionScores`/`assertArmBundlesMatchRawScores`,
 * `copyInterventionIndex`, and `extractInterventionSwaps` moved to their own
 * test files (`raw-intervention-scores.test.ts`, `intervention-index.test.ts`,
 * `intervention-swaps.test.ts`) alongside the modules they now live in (a
 * thermo-maintainability review finding: this file bundled four independent
 * artifact-builders). `resolveBigqGraphIdentity`/`resolveInterventionGraphIdentity`
 * are covered in `graph-identity.test.ts`.
 */

let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'archive-readouts-'));
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe('parseSourceArg', () => {
  it('parses "<graphId>=<run-dir>"', () => {
    const spec = parseSourceArg('--source', 'biological=training/runs/production/biological-101');
    expect(spec.graphId).toBe('biological');
    expect(spec.idSuffix).toBeNull();
    expect(spec.runDir.endsWith('training/runs/production/biological-101')).toBe(true);
  });

  it('parses "<graphId>:<idSuffix>=<run-dir>"', () => {
    const spec = parseSourceArg('--source', 'biological:gpurerun=training/runs/production/biological-101-gpurerun');
    expect(spec.graphId).toBe('biological');
    expect(spec.idSuffix).toBe('gpurerun');
  });

  it('rejects a spec with no "="', () => {
    expect(() => parseSourceArg('--source', 'biological')).toThrow(/must be/);
  });

  it('rejects a spec with an empty graphId before the idSuffix colon', () => {
    expect(() => parseSourceArg('--source', ':gpurerun=some/dir')).toThrow(/missing graphId/);
  });

  it('rejects an empty idSuffix after ":"', () => {
    expect(() => parseSourceArg('--source', 'biological:=some/dir')).toThrow(/empty idSuffix/);
  });

  it('rejects a graphId with an unsafe character', () => {
    expect(() => parseSourceArg('--source', 'bio logical=some/dir')).toThrow(/must match/);
  });

  it('rejects an idSuffix with an unsafe character', () => {
    expect(() => parseSourceArg('--source', 'biological:gpu rerun=some/dir')).toThrow(/must match/);
  });

  it('accepts the real "rewired-seed0" label despite containing "-seed0"', () => {
    // A naive "reject -seed<digits> in labels" defense (considered and
    // rejected -- see ArchivedReadout.id's doc comment) would incorrectly
    // reject this real, plan-mandated graphId.
    const spec = parseSourceArg('--source', 'rewired-seed0=some/dir');
    expect(spec.graphId).toBe('rewired-seed0');
  });
});

describe('parseTaskInterventionRawScoresArg', () => {
  it('parses "<arenaTaskId>=<path>"', () => {
    const spec = parseTaskInterventionRawScoresArg('--task-intervention-raw-scores', 'hazard-heavy=training/runs/tasks/hazard-heavy/trained.json');
    expect(spec.arenaTask).toBe('hazard-heavy');
    expect(spec.path.endsWith('training/runs/tasks/hazard-heavy/trained.json')).toBe(true);
  });

  it('rejects a spec with no "="', () => {
    expect(() => parseTaskInterventionRawScoresArg('--task-intervention-raw-scores', 'hazard-heavy')).toThrow(/must be/);
  });

  it('rejects a spec with an empty path', () => {
    expect(() => parseTaskInterventionRawScoresArg('--task-intervention-raw-scores', 'hazard-heavy=')).toThrow(/must be/);
  });

  it('rejects an unrecognized arena-task id', () => {
    expect(() => parseTaskInterventionRawScoresArg('--task-intervention-raw-scores', 'not-a-real-task=some/path.json')).toThrow();
  });
});

describe('taskInterventionRawScoresOutPath', () => {
  it('points at "<dir>/task-intervention-trained-raw-<task>-v1.json"', () => {
    const path = taskInterventionRawScoresOutPath(resolve(root, 'archive'), 'crowded');
    expect(path).toBe(resolve(root, 'archive', 'task-intervention-trained-raw-crowded-v1.json'));
  });

  it('throws for an unrecognized arena-task id', () => {
    expect(() => taskInterventionRawScoresOutPath(root, 'not-a-real-task')).toThrow();
  });
});

describe('parseArgs', () => {
  it('rejects an unrecognized flag', () => {
    expect(() => parseArgs(['--not-a-real-flag', 'value'])).toThrow(/Unknown argument: --not-a-real-flag/);
  });

  it('requires at least one of --source/--raw-intervention-scores/--intervention-index/--intervention-attribution', () => {
    expect(() => parseArgs([])).toThrow(/at least one of --source/);
  });

  it('does not require all four -- one --source is enough', () => {
    const args = parseArgs(['--source', 'biological=some/dir']);
    expect(args.sources).toHaveLength(1);
  });

  it('does not require --source -- one copy-only flag is enough', () => {
    const args = parseArgs(['--raw-intervention-scores', 'some/trained.json']);
    expect(args.rawInterventionScoresPath?.endsWith('some/trained.json')).toBe(true);
  });

  for (const flag of ['--out', '--raw-intervention-scores-out', '--intervention-index-out', '--intervention-swaps-out']) {
    it(`rejects ${flag} with a non-".json" path`, () => {
      expect(() => parseArgs(['--source', 'biological=some/dir', flag, 'not-json.txt'])).toThrow(
        new RegExp(`${flag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} must end with "\\.json"`)
      );
    });
  }

  it('accepts every output flag ending in ".json"', () => {
    const args = parseArgs([
      '--source',
      'biological=some/dir',
      '--out',
      'a.json',
      '--raw-intervention-scores-out',
      'b.json',
      '--intervention-index-out',
      'c.json',
      '--intervention-swaps-out',
      'd.json'
    ]);
    expect(args.out.endsWith('a.json')).toBe(true);
    expect(args.rawInterventionScoresOut.endsWith('b.json')).toBe(true);
    expect(args.interventionIndexOut.endsWith('c.json')).toBe(true);
    expect(args.interventionSwapsOut.endsWith('d.json')).toBe(true);
  });

  it('rejects a --source with a missing value', () => {
    expect(() => parseArgs(['--source'])).toThrow();
  });

  it('does not require --source -- --task-intervention-raw-scores alone is enough', () => {
    const args = parseArgs(['--task-intervention-raw-scores', 'hazard-heavy=some/trained.json']);
    expect(args.taskInterventionRawScores).toEqual([
      { arenaTask: 'hazard-heavy', path: expect.stringContaining('some/trained.json') }
    ]);
  });

  it('accepts --task-intervention-raw-scores-out-dir', () => {
    const args = parseArgs(['--source', 'biological=some/dir', '--task-intervention-raw-scores-out-dir', 'some/out-dir']);
    expect(args.taskInterventionRawScoresOutDir.endsWith('some/out-dir')).toBe(true);
  });

  it('rejects --task-intervention-raw-scores given more than once for the same arena task (a dual-review finding: two would silently last-write-wins the same output file)', () => {
    expect(() =>
      parseArgs([
        '--task-intervention-raw-scores',
        'hazard-heavy=some/a.json',
        '--task-intervention-raw-scores',
        'hazard-heavy=some/b.json'
      ])
    ).toThrow(/"hazard-heavy" was given more than once/);
  });

  it('allows --task-intervention-raw-scores for two DIFFERENT arena tasks', () => {
    const args = parseArgs([
      '--task-intervention-raw-scores',
      'hazard-heavy=some/a.json',
      '--task-intervention-raw-scores',
      'sparse-food=some/b.json'
    ]);
    expect(args.taskInterventionRawScores).toHaveLength(2);
  });
});

describe('kindForEntry', () => {
  it('is "bigq" for the three bigq graph ids at the default task', () => {
    expect(kindForEntry('biological', 'default')).toBe('bigq');
    expect(kindForEntry('rewired-seed0', 'default')).toBe('bigq');
    expect(kindForEntry('disconnected', 'default')).toBe('bigq');
  });

  it('is "intervention" for a non-bigq graph id at the default task', () => {
    expect(kindForEntry('P', 'default')).toBe('intervention');
    expect(kindForEntry('C000', 'default')).toBe('intervention');
    expect(kindForEntry('M1000', 'default')).toBe('intervention');
  });

  it('is "task-intervention" for any graph id at a non-default task', () => {
    expect(kindForEntry('biological', 'hazard-heavy')).toBe('task-intervention');
    expect(kindForEntry('P', 'hazard-heavy')).toBe('task-intervention');
  });
});

describe('assertArmMatchesGraphId', () => {
  it('accepts every bigq graphId with its expected arm', () => {
    expect(() => assertArmMatchesGraphId('biological', 'biological')).not.toThrow();
    expect(() => assertArmMatchesGraphId('rewired-seed0', 'rewired')).not.toThrow();
    expect(() => assertArmMatchesGraphId('disconnected', 'disconnected')).not.toThrow();
  });

  it('accepts any non-bigq graphId with arm "rewired"', () => {
    expect(() => assertArmMatchesGraphId('P', 'rewired')).not.toThrow();
    expect(() => assertArmMatchesGraphId('C000', 'rewired')).not.toThrow();
  });

  it('rejects a bigq graphId whose run has the wrong arm (the swapped---source-paths case)', () => {
    expect(() => assertArmMatchesGraphId('biological', 'rewired')).toThrow(/config\.arm is "rewired"/);
    expect(() => assertArmMatchesGraphId('rewired-seed0', 'biological')).toThrow(/expected "rewired"/);
  });

  it('rejects a non-bigq graphId whose run is not arm "rewired"', () => {
    expect(() => assertArmMatchesGraphId('P', 'biological')).toThrow(/expected "rewired"/);
  });
});

describe('buildArchivedReadout', () => {
  const D = 6;
  const H = 4;

  const writeRunDir = (dir: string, overrides: Partial<Parameters<typeof writeTinyRunDir>[0]> = {}): void => {
    writeTinyRunDir({
      dir,
      arm: 'biological',
      trainerSeed: 101,
      D,
      H,
      substeps: 4,
      weightSeed: 7,
      armBundleSha256: 'bundle-sha-biological',
      ...overrides
    });
    // writeTinyRunDir doesn't write graphArtifactSha256 (RunConfig doesn't
    // declare it) -- a real flyarena-train run always does, so patch it in
    // directly, matching how this script reads it (a raw cast, not through
    // RunConfig).
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-artifact-sha';
    writeFileSync(configPath, JSON.stringify(config));
  };

  it('round-trips a run directory into an archive entry whose theta matches readoutFromFlat and weightsSha256', () => {
    const dir = resolve(root, 'biological-101');
    writeRunDir(dir);

    const entry = buildArchivedReadout({ graphId: 'biological', idSuffix: null, runDir: dir });

    expect(entry.id).toBe('biological-seed101');
    expect(entry.arm).toBe('biological');
    expect(entry.kind).toBe('bigq');
    expect(entry.graphId).toBe('biological');
    expect(entry.trainerSeed).toBe(101);
    expect(entry.arenaTask).toBe('default');
    expect(entry.graphArtifactSha256).toBe('graph-artifact-sha');
    expect(entry.armBundleSha256).toBe('bundle-sha-biological');
    // buildArchivedReadout never has the cross-file context (the manifest or
    // the intervention index) to resolve these -- that's runArchiveReadouts'
    // attachGraphIdentity's job, done as a separate post-processing pass.
    expect(entry.graphGzipSha256).toBeNull();
    expect(entry.graphBinarySha256).toBeNull();
    expect(entry.D).toBe(D);
    expect(entry.H).toBe(H);
    expect(entry.parameterCount).toBe(H * D + H + 3 * H + 3);
    expect(entry.thetaSha256).toBe(createHash('sha256').update(readFileSync(resolve(dir, 'theta_final.npy'))).digest('hex'));
    expect(entry.sourcePath.length).toBeGreaterThan(0);

    const decoded = Buffer.from(entry.theta, 'base64');
    const theta = new Float32Array(decoded.buffer, decoded.byteOffset, decoded.byteLength / 4);
    // Exact value round-trip against the source run, not merely matching lengths.
    const sourceTheta = readNpyFloat32Array(resolve(dir, 'theta_final.npy'));
    expect(Array.from(theta)).toEqual(Array.from(sourceTheta));
    // weightsSha256 IS recomputable from the archive alone (unlike thetaSha256).
    expect(entry.weightsSha256).toBe(createHash('sha256').update(decoded).digest('hex'));

    const weights = readoutFromFlat(theta, D, H);
    expect(weights.w1.length).toBe(H * D);
    expect(weights.b1.length).toBe(H);
    expect(weights.w2.length).toBe(3 * H);
    expect(weights.b2.length).toBe(3);
  });

  it('applies an idSuffix to the archived id (the GPU-rerun labeling case)', () => {
    const dir = resolve(root, 'biological-101-gpurerun');
    writeRunDir(dir, { weightSeed: 8 });

    const entry = buildArchivedReadout({ graphId: 'biological', idSuffix: 'gpurerun', runDir: dir });
    expect(entry.id).toBe('biological-seed101-gpurerun');
  });

  it('rejects a graphId inconsistent with the run\'s own arm (a mislabeled/swapped --source)', () => {
    const dir = resolve(root, 'rewired-run');
    writeRunDir(dir, { arm: 'rewired', armBundleSha256: 'bundle-sha-rewired' });

    // Labeled "biological", but the run directory is actually "rewired".
    expect(() => buildArchivedReadout({ graphId: 'biological', idSuffix: null, runDir: dir })).toThrow(
      /points at a run whose config\.arm is "rewired"/
    );
  });

  it('accepts an intervention graphId (always arm "rewired")', () => {
    const dir = resolve(root, 'c000-run');
    writeRunDir(dir, { arm: 'rewired', armBundleSha256: 'bundle-sha-c000' });
    const entry = buildArchivedReadout({ graphId: 'C000', idSuffix: null, runDir: dir });
    expect(entry.kind).toBe('intervention');
  });

  it('refuses a run directory with no armBundleSha256', () => {
    const dir = resolve(root, 'no-bundle');
    writeTinyRunDir({ dir, arm: 'biological', trainerSeed: 101, D, H, substeps: 4, weightSeed: 1 });
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-artifact-sha';
    writeFileSync(configPath, JSON.stringify(config));

    expect(() => buildArchivedReadout({ graphId: 'biological', idSuffix: null, runDir: dir })).toThrow(
      /no armBundleSha256/
    );
  });

  it('refuses a run directory with no graphArtifactSha256', () => {
    const dir = resolve(root, 'no-graph-artifact-sha');
    writeTinyRunDir({
      dir,
      arm: 'biological',
      trainerSeed: 101,
      D,
      H,
      substeps: 4,
      weightSeed: 1,
      armBundleSha256: 'bundle-sha'
    });
    expect(() => buildArchivedReadout({ graphId: 'biological', idSuffix: null, runDir: dir })).toThrow(
      /no graphArtifactSha256/
    );
  });

  it('archives a non-default arenaTask under kind "task-intervention"', () => {
    const dir = resolve(root, 'hazard-heavy-run');
    writeTinyRunDir({
      dir,
      arm: 'rewired',
      trainerSeed: 101,
      D,
      H,
      substeps: 4,
      weightSeed: 2,
      armBundleSha256: 'bundle-sha-hazard',
      arenaTask: 'hazard-heavy'
    });
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-artifact-sha';
    writeFileSync(configPath, JSON.stringify(config));

    const entry = buildArchivedReadout({ graphId: 'P', idSuffix: null, runDir: dir });
    expect(entry.arenaTask).toBe('hazard-heavy');
    expect(entry.kind).toBe('task-intervention');
    // WP1b: a non-default arenaTask entry carries its resolved fingerprint.
    expect(entry.arenaTaskFingerprint).toMatch(/^arena-config-v1\|/);
    // WP1b dual-review finding: the arena-task suffix is baked into the id
    // AUTOMATICALLY, with no --source idSuffix supplied -- so id-uniqueness
    // across tasks for the same (graphId, trainerSeed) never depends on the
    // caller remembering to pass one.
    expect(entry.id).toBe('P-seed101-hazard-heavy');
  });

  it('stacks an explicit idSuffix with the automatic arena-task suffix, rather than replacing it', () => {
    const dir = resolve(root, 'hazard-heavy-gpurerun-run');
    writeTinyRunDir({
      dir,
      arm: 'rewired',
      trainerSeed: 101,
      D,
      H,
      substeps: 4,
      weightSeed: 6,
      armBundleSha256: 'bundle-sha-hazard-2',
      arenaTask: 'hazard-heavy'
    });
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-artifact-sha';
    writeFileSync(configPath, JSON.stringify(config));

    const entry = buildArchivedReadout({ graphId: 'P', idSuffix: 'gpurerun', runDir: dir });
    expect(entry.id).toBe('P-seed101-gpurerun-hazard-heavy');
  });

  it('leaves arenaTaskFingerprint undefined for the default task', () => {
    const dir = resolve(root, 'default-task-run');
    writeRunDir(dir, { weightSeed: 9 });
    const entry = buildArchivedReadout({ graphId: 'biological', idSuffix: null, runDir: dir });
    expect(entry.arenaTask).toBe('default');
    expect(entry.arenaTaskFingerprint).toBeUndefined();
  });

  it('throws on an unrecognized arenaTask id', () => {
    const dir = resolve(root, 'bogus-task-run');
    writeTinyRunDir({
      dir,
      arm: 'biological',
      trainerSeed: 101,
      D,
      H,
      substeps: 4,
      weightSeed: 3,
      armBundleSha256: 'bundle-sha'
    });
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-artifact-sha';
    config.arenaTask = 'not-a-real-task';
    writeFileSync(configPath, JSON.stringify(config));

    expect(() => buildArchivedReadout({ graphId: 'biological', idSuffix: null, runDir: dir })).toThrow();
  });
});

const makeEntry = (overrides: Partial<ArchivedReadout> = {}): ArchivedReadout => ({
  id: 'biological-seed101',
  arm: 'biological',
  kind: 'bigq',
  graphId: 'biological',
  trainerSeed: 101,
  arenaTask: 'default',
  graphArtifactSha256: 'graph-sha',
  armBundleSha256: 'bundle-sha',
  graphGzipSha256: 'graph-gzip-sha',
  graphBinarySha256: 'graph-binary-sha',
  D: 6,
  H: 4,
  parameterCount: 43,
  thetaSha256: 'a'.repeat(64),
  weightsSha256: 'b'.repeat(64),
  theta: 'AAAA',
  sourcePath: 'training/runs/production/biological-101',
  ...overrides
});

describe('loadExistingArchive', () => {
  it('returns null when the file does not exist', () => {
    expect(loadExistingArchive(resolve(root, 'missing.json'))).toBeNull();
  });

  it('loads a valid version-1 archive', () => {
    const path = resolve(root, 'archive.json');
    writeArchive(path, [makeEntry()]);
    expect(loadExistingArchive(path)?.readouts).toHaveLength(1);
  });

  it('refuses an unsupported version', () => {
    const path = resolve(root, 'bad-version.json');
    writeFileSync(path, JSON.stringify({ version: 2, readouts: [] }));
    expect(() => loadExistingArchive(path)).toThrow(/unsupported version/);
  });
});

describe('mergeReadouts', () => {
  it('sorts merged entries by id ascending, independent of insertion order', () => {
    const merged = mergeReadouts(
      [],
      [
        makeEntry({ id: 'rewired-seed0-seed101', weightsSha256: 'w-rewired' }),
        makeEntry({ id: 'biological-seed101', weightsSha256: 'w-biological' }),
        makeEntry({ id: 'C000-seed101', weightsSha256: 'w-c000' })
      ]
    );
    expect(merged.map((e) => e.id)).toEqual(['C000-seed101', 'biological-seed101', 'rewired-seed0-seed101']);
  });

  it('is idempotent: re-adding an identical entry does not throw and keeps the prior entry', () => {
    const existing = [makeEntry({ sourcePath: 'original/path' })];
    const merged = mergeReadouts(existing, [makeEntry({ sourcePath: 'different/path' })]);
    expect(merged).toHaveLength(1);
    // The PRIOR entry's sourcePath is kept, not overwritten by the re-run's
    // (which may come from a different checkout) -- keeps the archive's
    // committed bytes independent of who re-runs this CLI.
    expect(merged[0].sourcePath).toBe('original/path');
  });

  it('refuses a duplicate id whose thetaSha256 differs', () => {
    const existing = [makeEntry({ thetaSha256: 'a'.repeat(64) })];
    expect(() => mergeReadouts(existing, [makeEntry({ thetaSha256: 'b'.repeat(64) })])).toThrow(
      /different thetaSha256/
    );
  });

  it('refuses a duplicate id whose armBundleSha256 differs even when thetaSha256 matches', () => {
    const existing = [makeEntry({ armBundleSha256: 'bundle-a' })];
    expect(() => mergeReadouts(existing, [makeEntry({ armBundleSha256: 'bundle-b' })])).toThrow(
      /different armBundleSha256/
    );
  });

  it('refuses a duplicate id whose graphId differs (a relabeled re-run)', () => {
    const existing = [makeEntry({ graphId: 'biological' })];
    expect(() => mergeReadouts(existing, [makeEntry({ graphId: 'rewired-seed0' })])).toThrow(/different graphId/);
  });

  it('refuses a duplicate id whose graphGzipSha256/graphBinarySha256 differ', () => {
    const existing = [makeEntry({ graphGzipSha256: 'gz-a', graphBinarySha256: 'bin-a' })];
    expect(() => mergeReadouts(existing, [makeEntry({ graphGzipSha256: 'gz-b', graphBinarySha256: 'bin-a' })])).toThrow(
      /different graphGzipSha256/
    );
  });

  it('refuses a NEW id whose weightsSha256 duplicates an already-archived DIFFERENT id (the GPU-rerun-never-actually-run case)', () => {
    const existing = [makeEntry({ id: 'biological-seed101', weightsSha256: 'shared-weights' })];
    expect(() =>
      mergeReadouts(existing, [makeEntry({ id: 'biological-seed101-gpurerun', weightsSha256: 'shared-weights' })])
    ).toThrow(/same weightsSha256 as already-archived id "biological-seed101"/);
  });

  it('allows two different ids with different weightsSha256 (the real GPU-rerun case)', () => {
    const existing = [makeEntry({ id: 'biological-seed101', weightsSha256: 'cpu-weights' })];
    const merged = mergeReadouts(existing, [makeEntry({ id: 'biological-seed101-gpurerun', weightsSha256: 'gpu-weights' })]);
    expect(merged.map((e) => e.id)).toEqual(['biological-seed101', 'biological-seed101-gpurerun']);
  });

  it('detects a weightsSha256 collision between two NEW additions in the same invocation, not just against existing entries', () => {
    expect(() =>
      mergeReadouts(
        [],
        [
          makeEntry({ id: 'C000-seed101', graphId: 'C000', weightsSha256: 'dup-weights' }),
          makeEntry({ id: 'C001-seed101', graphId: 'C001', weightsSha256: 'dup-weights' })
        ]
      )
    ).toThrow(/same weightsSha256/);
  });

  it('keeps existing entries not touched by this invocation', () => {
    const existing = [makeEntry({ id: 'P-seed101', weightsSha256: 'w-p' })];
    const merged = mergeReadouts(existing, [makeEntry({ id: 'C000-seed101', weightsSha256: 'w-c000' })]);
    expect(merged.map((e) => e.id)).toEqual(['C000-seed101', 'P-seed101']);
  });
});

describe('arenaTaskFingerprintOf', () => {
  it('returns the stored value for a task-intervention entry', () => {
    const entry = makeEntry({
      kind: 'task-intervention',
      arenaTask: 'hazard-heavy',
      arenaTaskFingerprint: 'stored-hazard-heavy-fingerprint'
    });
    expect(arenaTaskFingerprintOf(entry)).toBe('stored-hazard-heavy-fingerprint');
  });

  it('resolves the DEFAULT task\'s real fingerprint for a default entry with no stored field, never "any"/"unknown"', () => {
    const entry = makeEntry({ kind: 'bigq', arenaTask: 'default', arenaTaskFingerprint: undefined });
    const resolved = arenaTaskFingerprintOf(entry);
    expect(resolved).toMatch(/^arena-config-v1\|/);
    // Matches what a real default-task entry's fingerprint would resolve to
    // (cross-checked against a real task-intervention entry's own resolved
    // fingerprint being DIFFERENT, so this isn't vacuously true for any string).
    const hazardHeavy = arenaTaskFingerprintOf(makeEntry({ kind: 'task-intervention', arenaTask: 'hazard-heavy' }));
    expect(resolved).not.toBe(hazardHeavy);
  });

  it('throws on an entry with an unrecognized arenaTask id (never silently resolves to "any")', () => {
    const entry = makeEntry({ arenaTask: 'not-a-real-task', arenaTaskFingerprint: undefined });
    expect(() => arenaTaskFingerprintOf(entry)).toThrow();
  });
});

describe('assertTaskGraphMatchesDefaultGraph', () => {
  it('is a no-op for a non-task-intervention entry', () => {
    const addition = makeEntry({ kind: 'bigq' });
    expect(() => assertTaskGraphMatchesDefaultGraph(addition, [])).not.toThrow();
  });

  it('is a no-op when there is no existing archive at all', () => {
    const addition = makeEntry({ kind: 'bigq', graphId: 'P' });
    expect(() => assertTaskGraphMatchesDefaultGraph(addition, [])).not.toThrow();
  });

  it('throws when no default-task "intervention" entry exists for this graphId', () => {
    const addition = makeEntry({
      id: 'P-seed101-hazard-heavy',
      kind: 'task-intervention',
      graphId: 'P',
      arenaTask: 'hazard-heavy',
      graphGzipSha256: 'gz-p',
      graphBinarySha256: 'bin-p'
    });
    expect(() => assertTaskGraphMatchesDefaultGraph(addition, [])).toThrow(/has no already-\s*archived default-task/);
  });

  it('throws when the resolved graph disagrees with the default-task entry\'s graph', () => {
    const defaultEntry = makeEntry({
      id: 'P-seed101',
      kind: 'intervention',
      graphId: 'P',
      graphGzipSha256: 'gz-p-default',
      graphBinarySha256: 'bin-p-default'
    });
    const addition = makeEntry({
      id: 'P-seed101-hazard-heavy',
      kind: 'task-intervention',
      graphId: 'P',
      arenaTask: 'hazard-heavy',
      graphGzipSha256: 'gz-p-DIFFERENT',
      graphBinarySha256: 'bin-p-default'
    });
    expect(() => assertTaskGraphMatchesDefaultGraph(addition, [defaultEntry])).toThrow(
      /per-task readouts must use the same intervention graphs as the default task/
    );
  });

  it('does not throw when the resolved graph agrees with the default-task entry\'s graph', () => {
    const defaultEntry = makeEntry({
      id: 'P-seed101',
      kind: 'intervention',
      graphId: 'P',
      graphGzipSha256: 'gz-p-default',
      graphBinarySha256: 'bin-p-default'
    });
    const addition = makeEntry({
      id: 'P-seed101-hazard-heavy',
      kind: 'task-intervention',
      graphId: 'P',
      arenaTask: 'hazard-heavy',
      graphGzipSha256: 'gz-p-default',
      graphBinarySha256: 'bin-p-default'
    });
    expect(() => assertTaskGraphMatchesDefaultGraph(addition, [defaultEntry])).not.toThrow();
  });
});

/** Builds a real, `readGraphListIndex`-valid `index.json` (with a real gzip graph file alongside it) for one intervention id, for `runArchiveReadouts` tests that need `--intervention-index` to resolve graph identity. */
const writeFakeInterventionIndex = (dir: string, id: string, graphBytes: Buffer): string => {
  const graphsDir = resolve(dir, 'graphs');
  mkdirSync(graphsDir, { recursive: true });
  const gzipBytes = gzipSync(graphBytes);
  const relPath = `graphs/${id}.bin.gz`;
  writeFileSync(resolve(dir, relPath), gzipBytes);
  const gzipSha256 = createHash('sha256').update(gzipBytes).digest('hex');
  const binarySha256 = createHash('sha256').update(graphBytes).digest('hex');
  const indexPath = resolve(dir, 'index.json');
  writeFileSync(
    indexPath,
    JSON.stringify({
      sourceArtifact: 'public/data/malecns-arena-v1.bin.gz',
      sourceSha256: 'a'.repeat(64),
      entries: [{ id, path: relPath, gzipSha256, binarySha256 }]
    })
  );
  return indexPath;
};

describe('runArchiveReadouts', () => {
  const baseArgs = (overrides: Partial<ArchiveReadoutsArgs> = {}): ArchiveReadoutsArgs => ({
    sources: [],
    out: resolve(root, 'trained-readouts-v1.json'),
    rawInterventionScoresPath: null,
    rawInterventionScoresOut: resolve(root, 'intervention-trained-raw-v1.json'),
    interventionIndexPath: null,
    interventionIndexOut: resolve(root, 'intervention-index-v1.json'),
    interventionAttributionPath: null,
    interventionSwapsOut: resolve(root, 'intervention-swaps-v1.json'),
    taskInterventionRawScores: [],
    taskInterventionRawScoresOutDir: root,
    ...overrides
  });

  it('writes the readouts archive and reports the actual paths written, resolving bigq graph identity from the real committed manifest', () => {
    const dir = resolve(root, 'biological-101');
    writeTinyRunDir({ dir, arm: 'biological', trainerSeed: 101, D: 6, H: 4, substeps: 4, weightSeed: 1, armBundleSha256: 'sha' });
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-sha';
    writeFileSync(configPath, JSON.stringify(config));

    const args = baseArgs({ sources: [{ graphId: 'biological', idSuffix: null, runDir: dir }] });
    const result = runArchiveReadouts(args);

    expect(result.readoutCount).toBe(1);
    expect(result.wrote).toEqual([args.out]);
    const readouts = loadExistingArchive(args.out)?.readouts ?? [];
    expect(readouts).toHaveLength(1);
    // Resolved from DEFAULT_MANIFEST_PATH (the real, committed manifest) --
    // no --intervention-index needed for a bigq graphId.
    expect(readouts[0].graphGzipSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(readouts[0].graphBinarySha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it('throws when a non-bigq --source is given without --intervention-index', () => {
    const dir = resolve(root, 'c000-run');
    writeTinyRunDir({ dir, arm: 'rewired', trainerSeed: 101, D: 6, H: 4, substeps: 4, weightSeed: 1, armBundleSha256: 'sha' });
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-sha';
    writeFileSync(configPath, JSON.stringify(config));

    const args = baseArgs({ sources: [{ graphId: 'C000', idSuffix: null, runDir: dir }] });
    expect(() => runArchiveReadouts(args)).toThrow(/needs --intervention-index/);
    expect(loadExistingArchive(args.out)).toBeNull();
  });

  it('resolves an intervention --source\'s graph identity from --intervention-index when supplied', () => {
    const dir = resolve(root, 'c000-run');
    writeTinyRunDir({ dir, arm: 'rewired', trainerSeed: 101, D: 6, H: 4, substeps: 4, weightSeed: 1, armBundleSha256: 'sha' });
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-sha';
    writeFileSync(configPath, JSON.stringify(config));

    const indexPath = writeFakeInterventionIndex(root, 'C000', Buffer.from('c000-graph-bytes'));
    const args = baseArgs({
      sources: [{ graphId: 'C000', idSuffix: null, runDir: dir }],
      interventionIndexPath: indexPath
    });
    const result = runArchiveReadouts(args);
    const readouts = loadExistingArchive(args.out)?.readouts ?? [];
    expect(result.wrote).toContain(args.interventionIndexOut);
    expect(readouts[0].graphGzipSha256).toBe(createHash('sha256').update(gzipSync(Buffer.from('c000-graph-bytes'))).digest('hex'));
    expect(readouts[0].graphBinarySha256).toBe(createHash('sha256').update(Buffer.from('c000-graph-bytes')).digest('hex'));
  });

  it('is deterministic: running twice produces byte-identical output', () => {
    const dir = resolve(root, 'biological-101');
    writeTinyRunDir({ dir, arm: 'biological', trainerSeed: 101, D: 6, H: 4, substeps: 4, weightSeed: 1, armBundleSha256: 'sha' });
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-sha';
    writeFileSync(configPath, JSON.stringify(config));

    const args = baseArgs({ sources: [{ graphId: 'biological', idSuffix: null, runDir: dir }] });
    runArchiveReadouts(args);
    const firstBytes = readFileSync(args.out);
    runArchiveReadouts(args);
    const secondBytes = readFileSync(args.out);
    expect(secondBytes).toEqual(firstBytes);
  });

  it('does not write the readouts archive when a later input fails validation', () => {
    const dir = resolve(root, 'biological-101');
    writeTinyRunDir({ dir, arm: 'biological', trainerSeed: 101, D: 6, H: 4, substeps: 4, weightSeed: 1, armBundleSha256: 'sha' });
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-sha';
    writeFileSync(configPath, JSON.stringify(config));

    const badIndexPath = resolve(root, 'bad-index.json');
    writeFileSync(badIndexPath, JSON.stringify({ sourceArtifact: 'x', sourceSha256: 'a'.repeat(64), entries: [] }));

    const args = baseArgs({
      sources: [{ graphId: 'biological', idSuffix: null, runDir: dir }],
      interventionIndexPath: badIndexPath
    });

    expect(() => runArchiveReadouts(args)).toThrow(/has no entries/);
    // The readouts archive must not have been written either, even though
    // it would have succeeded on its own -- validate-before-write ordering.
    expect(loadExistingArchive(args.out)).toBeNull();
  });

  it('throws when an intervention --source is cross-checked against a disagreeing raw scores file', () => {
    const dir = resolve(root, 'c000-run');
    writeTinyRunDir({ dir, arm: 'rewired', trainerSeed: 101, D: 6, H: 4, substeps: 4, weightSeed: 1, armBundleSha256: 'archived-sha' });
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-sha';
    writeFileSync(configPath, JSON.stringify(config));

    const indexPath = writeFakeInterventionIndex(root, 'C000', Buffer.from('c000-graph-bytes'));
    const rawPath = resolve(root, 'raw.json');
    writeFileSync(
      rawPath,
      JSON.stringify({ runs: [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'different-sha', movementScore: [1] }] })
    );

    const args = baseArgs({
      // Mislabeled: this run's armBundleSha256 ("archived-sha") disagrees
      // with the raw file's recorded armBundleSha256 for id "C000".
      sources: [{ graphId: 'C000', idSuffix: null, runDir: dir }],
      interventionIndexPath: indexPath,
      rawInterventionScoresPath: rawPath
    });

    expect(() => runArchiveReadouts(args)).toThrow(/mismatched --source label/);
    expect(loadExistingArchive(args.out)).toBeNull();
  });

  describe('WP1b: --task-intervention-raw-scores / task-intervention --source', () => {
    const writeTaskRunDir = (dir: string, overrides: Partial<Parameters<typeof writeTinyRunDir>[0]> = {}): void => {
      writeTinyRunDir({
        dir,
        arm: 'rewired',
        trainerSeed: 101,
        D: 6,
        H: 4,
        substeps: 4,
        weightSeed: 5,
        armBundleSha256: 'archived-task-sha',
        arenaTask: 'hazard-heavy',
        ...overrides
      });
      const configPath = resolve(dir, 'config.json');
      const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
      config.graphArtifactSha256 = 'graph-artifact-sha';
      writeFileSync(configPath, JSON.stringify(config));
    };

    it('archives a task-intervention entry, verifying its graph against the already-archived default-task entry', () => {
      // Seed an existing archive with the default-task "intervention" entry
      // for graphId "C000" -- the task-intervention addition's graph is
      // verified against THIS entry (assertTaskGraphMatchesDefaultGraph).
      const graphBytes = Buffer.from('c000-graph-bytes');
      const gzipSha256 = createHash('sha256').update(gzipSync(graphBytes)).digest('hex');
      const binarySha256 = createHash('sha256').update(graphBytes).digest('hex');
      const existingDefaultEntry = makeEntry({
        id: 'C000-seed101',
        kind: 'intervention',
        graphId: 'C000',
        graphGzipSha256: gzipSha256,
        graphBinarySha256: binarySha256,
        weightsSha256: 'default-c000-weights'
      });
      const outPath = resolve(root, 'trained-readouts-v1.json');
      writeArchive(outPath, [existingDefaultEntry]);

      const dir = resolve(root, 'c000-hazard-heavy-run');
      writeTaskRunDir(dir);

      const indexPath = writeFakeInterventionIndex(root, 'C000', graphBytes);
      const indexBytes = readFileSync(indexPath);
      const graphListSha256 = createHash('sha256').update(indexBytes).digest('hex');

      const rawPath = resolve(root, 'hazard-heavy-trained.json');
      writeFileSync(
        rawPath,
        JSON.stringify({
          arenaTask: 'hazard-heavy',
          arenaTaskFingerprint: 'fingerprint',
          graphListSha256,
          runs: [
            {
              id: 'C000',
              trainerSeed: 101,
              armBundleSha256: 'archived-task-sha',
              gzipSha256: 'raw-file-gzip-sha',
              movementScore: [1, 2, 3]
            }
          ]
        })
      );

      const args = baseArgs({
        out: outPath,
        // No idSuffix: WP1b auto-derives the id's task suffix from the run's
        // own arenaTask -- an explicit --source idSuffix is no longer needed
        // for task-uniqueness (a dual-review finding).
        sources: [{ graphId: 'C000', idSuffix: null, runDir: dir }],
        interventionIndexPath: indexPath,
        taskInterventionRawScores: [{ arenaTask: 'hazard-heavy', path: rawPath }]
      });

      const result = runArchiveReadouts(args);
      expect(result.readoutCount).toBe(2);
      expect(result.wrote).toContain(taskInterventionRawScoresOutPath(root, 'hazard-heavy'));

      const readouts = loadExistingArchive(outPath)?.readouts ?? [];
      const taskEntry = readouts.find((r) => r.id === 'C000-seed101-hazard-heavy');
      expect(taskEntry).toBeDefined();
      expect(taskEntry?.kind).toBe('task-intervention');
      expect(taskEntry?.arenaTask).toBe('hazard-heavy');
      expect(taskEntry?.graphGzipSha256).toBe(gzipSha256);
      expect(taskEntry?.graphBinarySha256).toBe(binarySha256);

      // The default-task entry itself is untouched.
      expect(readouts.find((r) => r.id === 'C000-seed101')).toEqual(existingDefaultEntry);
    });

    it('throws when --task-intervention-raw-scores is given without --intervention-index', () => {
      const rawPath = resolve(root, 'hazard-heavy-trained.json');
      writeFileSync(
        rawPath,
        JSON.stringify({
          arenaTask: 'hazard-heavy',
          graphListSha256: 'a'.repeat(64),
          runs: [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'x', movementScore: [1] }]
        })
      );
      const args = baseArgs({ taskInterventionRawScores: [{ arenaTask: 'hazard-heavy', path: rawPath }] });
      expect(() => runArchiveReadouts(args)).toThrow(/needs --intervention-index/);
    });

    it('throws when a task raw-scores file\'s graphListSha256 does not match this invocation\'s --intervention-index', () => {
      const graphBytes = Buffer.from('c000-graph-bytes');
      const indexPath = writeFakeInterventionIndex(root, 'C000', graphBytes);
      const rawPath = resolve(root, 'hazard-heavy-trained.json');
      writeFileSync(
        rawPath,
        JSON.stringify({
          arenaTask: 'hazard-heavy',
          graphListSha256: 'f'.repeat(64), // wrong -- does not match indexPath's own sha256
          runs: [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'x', gzipSha256: 'y', movementScore: [1] }]
        })
      );
      const args = baseArgs({
        interventionIndexPath: indexPath,
        taskInterventionRawScores: [{ arenaTask: 'hazard-heavy', path: rawPath }]
      });
      expect(() => runArchiveReadouts(args)).toThrow(/does not match this invocation's --intervention-index sha256/);
    });

    it('throws when a task-intervention --source disagrees with the already-archived default-task graph', () => {
      const existingDefaultEntry = makeEntry({
        id: 'C000-seed101',
        kind: 'intervention',
        graphId: 'C000',
        graphGzipSha256: 'gz-DIFFERENT-from-index',
        graphBinarySha256: 'bin-DIFFERENT-from-index',
        weightsSha256: 'default-c000-weights'
      });
      const outPath = resolve(root, 'trained-readouts-v1.json');
      writeArchive(outPath, [existingDefaultEntry]);

      const dir = resolve(root, 'c000-hazard-heavy-run-2');
      writeTaskRunDir(dir, { armBundleSha256: 'sha-2' });

      const indexPath = writeFakeInterventionIndex(root, 'C000', Buffer.from('c000-graph-bytes'));

      const args = baseArgs({
        out: outPath,
        sources: [{ graphId: 'C000', idSuffix: null, runDir: dir }],
        interventionIndexPath: indexPath
      });

      expect(() => runArchiveReadouts(args)).toThrow(/per-task readouts must use the same intervention graphs as the default task/);
      // The archive must not have been overwritten with a partial/invalid write.
      expect(loadExistingArchive(outPath)?.readouts).toEqual([existingDefaultEntry]);
    });
  });
});

describe('DEFAULT_MANIFEST_PATH', () => {
  it('points at the real, committed manifest', () => {
    expect(DEFAULT_MANIFEST_PATH.endsWith('public/data/malecns-arena-v1.manifest.json')).toBe(true);
  });
});
