import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { readoutFromFlat } from '../../src/lib/connectome/readout-serialization';
import {
  buildArchivedReadout,
  copyInterventionIndex,
  copyRawInterventionScores,
  extractInterventionSwaps,
  kindForEntry,
  mergeReadouts,
  parseSourceArg,
  type ArchivedReadout
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

describe('buildArchivedReadout', () => {
  const D = 6;
  const H = 4;

  it('round-trips a run directory into an archive entry whose theta matches readoutFromFlat', () => {
    const dir = resolve(root, 'biological-101');
    writeTinyRunDir({
      dir,
      arm: 'biological',
      trainerSeed: 101,
      D,
      H,
      substeps: 4,
      weightSeed: 7,
      armBundleSha256: 'bundle-sha-biological'
    });
    // writeTinyRunDir doesn't write graphArtifactSha256 (RunConfig doesn't
    // declare it) -- a real flyarena-train run always does, so patch it in
    // directly, matching how this script reads it (a raw cast, not through
    // RunConfig).
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-artifact-sha';
    writeFileSync(configPath, JSON.stringify(config));

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
    expect(entry.thetaSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(entry.sourcePath.length).toBeGreaterThan(0);

    const decoded = Buffer.from(entry.theta, 'base64');
    const theta = new Float32Array(decoded.buffer, decoded.byteOffset, decoded.byteLength / 4);
    const weights = readoutFromFlat(theta, D, H);
    expect(weights.w1.length).toBe(H * D);
    expect(weights.b1.length).toBe(H);
    expect(weights.w2.length).toBe(3 * H);
    expect(weights.b2.length).toBe(3);
  });

  it('applies an idSuffix to the archived id (the GPU-rerun labeling case)', () => {
    const dir = resolve(root, 'biological-101-gpurerun');
    writeTinyRunDir({
      dir,
      arm: 'biological',
      trainerSeed: 101,
      D,
      H,
      substeps: 4,
      weightSeed: 8,
      armBundleSha256: 'bundle-sha-biological'
    });
    const configPath = resolve(dir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.graphArtifactSha256 = 'graph-artifact-sha';
    writeFileSync(configPath, JSON.stringify(config));

    const entry = buildArchivedReadout({ graphId: 'biological', idSuffix: 'gpurerun', runDir: dir });
    expect(entry.id).toBe('biological-seed101-gpurerun');
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
  theta: 'AAAA',
  sourcePath: 'training/runs/production/biological-101',
  ...overrides
});

describe('mergeReadouts', () => {
  it('sorts merged entries by id ascending, independent of insertion order', () => {
    const merged = mergeReadouts(
      [],
      [makeEntry({ id: 'rewired-seed0-seed101' }), makeEntry({ id: 'biological-seed101' }), makeEntry({ id: 'C000-seed101' })]
    );
    expect(merged.map((e) => e.id)).toEqual(['C000-seed101', 'biological-seed101', 'rewired-seed0-seed101']);
  });

  it('is idempotent: re-adding an identical entry (same thetaSha256) does not throw', () => {
    const existing = [makeEntry()];
    const merged = mergeReadouts(existing, [makeEntry()]);
    expect(merged).toHaveLength(1);
  });

  it('refuses a duplicate id whose thetaSha256 differs', () => {
    const existing = [makeEntry({ thetaSha256: 'a'.repeat(64) })];
    expect(() => mergeReadouts(existing, [makeEntry({ thetaSha256: 'b'.repeat(64) })])).toThrow(
      /different thetaSha256/
    );
  });

  it('keeps existing entries not touched by this invocation', () => {
    const existing = [makeEntry({ id: 'P-seed101' })];
    const merged = mergeReadouts(existing, [makeEntry({ id: 'C000-seed101' })]);
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
  it('copies a well-formed graph-list index.json', () => {
    const src = resolve(root, 'index.json');
    const out = resolve(root, 'archive', 'intervention-index-v1.json');
    const payload = {
      sourceArtifact: 'public/data/malecns-arena-v1.bin.gz',
      sourceSha256: 'a'.repeat(64),
      entries: [{ id: 'P', path: 'graphs/P.bin.gz', gzipSha256: 'b'.repeat(64), binarySha256: 'c'.repeat(64) }]
    };
    writeFileSync(src, JSON.stringify(payload));

    copyInterventionIndex(src, out);
    expect(JSON.parse(readFileSync(out, 'utf8'))).toEqual(payload);
  });

  it('refuses a file with no entries', () => {
    const src = resolve(root, 'bad-index.json');
    writeFileSync(src, JSON.stringify({ sourceSha256: 'a'.repeat(64), entries: [] }));
    expect(() => copyInterventionIndex(src, resolve(root, 'out.json'))).toThrow(/not a valid graph-list index/);
  });
});

describe('extractInterventionSwaps', () => {
  it('extracts only accepted steps\' addedEdge/addedEdge2/removedEdge/removedEdge2 for P and Q', () => {
    const src = resolve(root, 'attribution.json');
    const out = resolve(root, 'archive', 'intervention-swaps-v1.json');
    const payload = {
      P: {
        kind: 'P',
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
    writeFileSync(src, JSON.stringify(payload));

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
});
