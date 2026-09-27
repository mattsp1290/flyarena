import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { readoutFromFlat } from '../../src/lib/connectome/readout-serialization';
import { readNpyFloat32Array } from '../../scripts/training/npy';
import {
  assertArmBundlesMatchRawScores,
  assertArmMatchesGraphId,
  buildArchivedReadout,
  copyInterventionIndex,
  copyRawInterventionScores,
  extractInterventionSwaps,
  kindForEntry,
  loadExistingArchive,
  mergeReadouts,
  parseSourceArg,
  runArchiveReadouts,
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

describe('copyRawInterventionScores', () => {
  it('copies a well-formed raw trained.json', () => {
    const src = resolve(root, 'trained.json');
    const out = resolve(root, 'archive', 'intervention-trained-raw-v1.json');
    const payload = { version: 1, runs: [{ id: 'C000', trainerSeed: 101, movementScore: [1, 2, 3] }] };
    writeFileSync(src, JSON.stringify(payload));

    copyRawInterventionScores(src, out);
    const written = JSON.parse(readFileSync(out, 'utf8'));
    expect(written).toEqual(payload);
  });

  it('copies non-canonical (pretty-printed) input byte-for-byte, not re-encoded', () => {
    const src = resolve(root, 'pretty-trained.json');
    const out = resolve(root, 'archive', 'pretty-out.json');
    const text = `${JSON.stringify({ version: 1, runs: [{ id: 'C000', trainerSeed: 101, movementScore: [1] }] }, null, 2)}\n`;
    writeFileSync(src, text);

    copyRawInterventionScores(src, out);
    // Byte identity, not merely structural equality: a JSON.stringify(parsed)
    // re-encode would silently collapse this back to compact form.
    expect(readFileSync(out)).toEqual(readFileSync(src));
  });

  it('refuses a file with no "runs" array', () => {
    const src = resolve(root, 'bad.json');
    writeFileSync(src, JSON.stringify({ version: 1 }));
    expect(() => copyRawInterventionScores(src, resolve(root, 'out.json'))).toThrow(/no "runs" array/);
  });

  it('refuses a malformed run entry', () => {
    const src = resolve(root, 'bad-run.json');
    writeFileSync(src, JSON.stringify({ runs: [{ id: 'C000' }] }));
    expect(() => copyRawInterventionScores(src, resolve(root, 'out.json'))).toThrow(/malformed run entry/);
  });
});

describe('copyInterventionIndex', () => {
  const validPayload = {
    sourceArtifact: 'public/data/malecns-arena-v1.bin.gz',
    sourceSha256: 'a'.repeat(64),
    entries: [{ id: 'P', path: 'graphs/P.bin.gz', gzipSha256: 'b'.repeat(64), binarySha256: 'c'.repeat(64) }]
  };

  it('copies a well-formed graph-list index.json', () => {
    const src = resolve(root, 'index.json');
    const out = resolve(root, 'archive', 'intervention-index-v1.json');
    writeFileSync(src, JSON.stringify(validPayload));

    copyInterventionIndex(src, out);
    expect(JSON.parse(readFileSync(out, 'utf8'))).toEqual(validPayload);
  });

  it('copies non-canonical (pretty-printed) input byte-for-byte, not re-encoded', () => {
    const src = resolve(root, 'pretty-index.json');
    const out = resolve(root, 'archive', 'pretty-index-out.json');
    const text = `${JSON.stringify(validPayload, null, 2)}\n`;
    writeFileSync(src, text);

    copyInterventionIndex(src, out);
    expect(readFileSync(out)).toEqual(readFileSync(src));
  });

  it('refuses a file with no entries', () => {
    const src = resolve(root, 'bad-index.json');
    writeFileSync(src, JSON.stringify({ sourceSha256: 'a'.repeat(64), entries: [] }));
    expect(() => copyInterventionIndex(src, resolve(root, 'out.json'))).toThrow(/not a valid graph-list index/);
  });
});

const validAttributionPayload = {
  P: {
    kind: 'P',
    swaps: 1,
    steps: [
      {
        step: 0,
        accepted: true,
        addedEdge: { pre: 1, post: 2 },
        addedEdge2: { pre: 3, post: 4 },
        removedEdge: { pre: 5, post: 6 },
        removedEdge2: { pre: 7, post: 8 }
      },
      {
        step: 1,
        accepted: false,
        addedEdge: { pre: 100, post: 200 },
        addedEdge2: { pre: 100, post: 200 },
        removedEdge: { pre: 100, post: 200 },
        removedEdge2: { pre: 100, post: 200 }
      }
    ]
  },
  Q: {
    kind: 'Q',
    swaps: 1,
    steps: [
      {
        step: 0,
        accepted: true,
        addedEdge: { pre: 9, post: 10 },
        addedEdge2: { pre: 11, post: 12 },
        removedEdge: { pre: 13, post: 14 },
        removedEdge2: { pre: 15, post: 16 }
      }
    ]
  }
};

describe('extractInterventionSwaps', () => {
  it('extracts only accepted steps\' addedEdge/addedEdge2/removedEdge/removedEdge2 for P and Q', () => {
    const src = resolve(root, 'attribution.json');
    const out = resolve(root, 'archive', 'intervention-swaps-v1.json');
    writeFileSync(src, JSON.stringify(validAttributionPayload));

    extractInterventionSwaps(src, out);
    const written = JSON.parse(readFileSync(out, 'utf8'));
    expect(written.version).toBe(1);
    expect(written.sourceSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(written.swaps).toEqual([
      {
        id: 'P',
        addedEdges: [
          { pre: 1, post: 2 },
          { pre: 3, post: 4 }
        ],
        removedEdges: [
          { pre: 5, post: 6 },
          { pre: 7, post: 8 }
        ]
      },
      {
        id: 'Q',
        addedEdges: [
          { pre: 9, post: 10 },
          { pre: 11, post: 12 }
        ],
        removedEdges: [
          { pre: 13, post: 14 },
          { pre: 15, post: 16 }
        ]
      }
    ]);
  });

  it('refuses a file with no P entry', () => {
    const src = resolve(root, 'no-p.json');
    writeFileSync(src, JSON.stringify({ Q: { kind: 'Q', steps: [] } }));
    expect(() => extractInterventionSwaps(src, resolve(root, 'out.json'))).toThrow(/no "P" entry/);
  });

  it('refuses a file with no Q entry', () => {
    const src = resolve(root, 'no-q.json');
    writeFileSync(src, JSON.stringify({ P: { kind: 'P', swaps: 0, steps: [] } }));
    expect(() => extractInterventionSwaps(src, resolve(root, 'out.json'))).toThrow(/no "Q" entry/);
  });

  it('refuses when the accepted-step count disagrees with the entry\'s own "swaps" field', () => {
    const src = resolve(root, 'bad-swaps-count.json');
    const payload = {
      ...validAttributionPayload,
      P: { ...validAttributionPayload.P, swaps: 99 }
    };
    writeFileSync(src, JSON.stringify(payload));
    expect(() => extractInterventionSwaps(src, resolve(root, 'out.json'))).toThrow(/accepted step\(s\) but swaps=99/);
  });

  it('refuses a malformed edge', () => {
    const src = resolve(root, 'bad-edge.json');
    const payload = {
      ...validAttributionPayload,
      P: {
        kind: 'P',
        swaps: 1,
        steps: [
          {
            step: 0,
            accepted: true,
            addedEdge: { pre: 1 }, // missing "post"
            addedEdge2: { pre: 3, post: 4 },
            removedEdge: { pre: 5, post: 6 },
            removedEdge2: { pre: 7, post: 8 }
          }
        ]
      }
    };
    writeFileSync(src, JSON.stringify(payload));
    expect(() => extractInterventionSwaps(src, resolve(root, 'out.json'))).toThrow(/malformed edge/);
  });
});

describe('assertArmBundlesMatchRawScores', () => {
  it('does not throw when armBundleSha256 values agree', () => {
    const additions = [makeEntry({ kind: 'intervention', graphId: 'C000', trainerSeed: 101, armBundleSha256: 'shared-sha' })];
    const rawRuns = [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'shared-sha', movementScore: [] }];
    expect(() => assertArmBundlesMatchRawScores(additions, rawRuns)).not.toThrow();
  });

  it('throws when an intervention addition\'s armBundleSha256 disagrees with the raw file (mislabeled --source)', () => {
    const additions = [makeEntry({ kind: 'intervention', graphId: 'C000', trainerSeed: 101, armBundleSha256: 'wrong-sha' })];
    const rawRuns = [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'correct-sha', movementScore: [] }];
    expect(() => assertArmBundlesMatchRawScores(additions, rawRuns)).toThrow(/mismatched --source label/);
  });

  it('ignores a bigq addition (never cross-checked against the intervention raw file)', () => {
    const additions = [makeEntry({ kind: 'bigq', graphId: 'biological', trainerSeed: 101, armBundleSha256: 'anything' })];
    const rawRuns = [{ id: 'biological', trainerSeed: 101, armBundleSha256: 'something-else', movementScore: [] }];
    expect(() => assertArmBundlesMatchRawScores(additions, rawRuns)).not.toThrow();
  });

  it('ignores an addition with no matching raw entry', () => {
    const additions = [makeEntry({ kind: 'intervention', graphId: 'C099', trainerSeed: 101 })];
    expect(() => assertArmBundlesMatchRawScores(additions, [])).not.toThrow();
  });
});

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
    ...overrides
  });

  it('writes the readouts archive and reports the actual paths written', () => {
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
    expect(loadExistingArchive(args.out)?.readouts).toHaveLength(1);
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
    writeFileSync(badIndexPath, JSON.stringify({ sourceSha256: 'a'.repeat(64), entries: [] }));

    const args = baseArgs({
      sources: [{ graphId: 'biological', idSuffix: null, runDir: dir }],
      interventionIndexPath: badIndexPath
    });

    expect(() => runArchiveReadouts(args)).toThrow(/not a valid graph-list index/);
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

    const rawPath = resolve(root, 'raw.json');
    writeFileSync(
      rawPath,
      JSON.stringify({ runs: [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'different-sha', movementScore: [1] }] })
    );

    const args = baseArgs({
      // Mislabeled: this run's armBundleSha256 ("archived-sha") disagrees
      // with the raw file's recorded armBundleSha256 for id "C000".
      sources: [{ graphId: 'C000', idSuffix: null, runDir: dir }],
      rawInterventionScoresPath: rawPath
    });

    expect(() => runArchiveReadouts(args)).toThrow(/mismatched --source label/);
    expect(loadExistingArchive(args.out)).toBeNull();
  });
});
