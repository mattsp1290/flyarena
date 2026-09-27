import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { gzipSync } from 'node:zlib';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { GraphListIndex } from '../../scripts/null/graph-list-index';
import { resolveBigqGraphIdentity, resolveInterventionGraphIdentity } from '../../scripts/attribution/graph-identity';

/**
 * Coverage for `scripts/attribution/graph-identity.ts`
 * (a thermo-provenance review finding on WP1: `armBundleSha256` is not a
 * durable, path-independent graph identity -- see that module's own doc
 * comment). Builds small synthetic gzip fixtures rather than depending on
 * the real `public/data/malecns-arena-v1*.bin.gz` artifacts, so these tests
 * exercise the validation logic (and its failure paths) independently of
 * this repo's actual committed graph bytes.
 */

const sha256Hex = (data: Buffer): string => createHash('sha256').update(data).digest('hex');

let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'graph-identity-'));
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe('resolveBigqGraphIdentity', () => {
  const writeManifest = (biologicalBytes: Buffer, rewiredBytes: Buffer): string => {
    const bioGzip = gzipSync(biologicalBytes);
    const rewiredGzip = gzipSync(rewiredBytes);
    writeFileSync(resolve(root, 'malecns-arena-v1.bin.gz'), bioGzip);
    writeFileSync(resolve(root, 'malecns-arena-v1-rewired-seed0.bin.gz'), rewiredGzip);
    const manifest = {
      artifact: 'malecns-arena-v1.bin.gz',
      gzipSha256: sha256Hex(bioGzip),
      binarySha256: sha256Hex(biologicalBytes),
      rewiredArms: {
        seed0: {
          artifact: 'malecns-arena-v1-rewired-seed0.bin.gz',
          gzipSha256: sha256Hex(rewiredGzip),
          binarySha256: sha256Hex(rewiredBytes)
        }
      }
    };
    const manifestPath = resolve(root, 'malecns-arena-v1.manifest.json');
    writeFileSync(manifestPath, JSON.stringify(manifest));
    return manifestPath;
  };

  it('resolves and validates "biological" against the manifest and the actual artifact bytes', () => {
    const manifestPath = writeManifest(Buffer.from('biological-graph-bytes'), Buffer.from('rewired-graph-bytes'));
    const identity = resolveBigqGraphIdentity('biological', manifestPath);
    expect(identity?.graphGzipSha256).toBe(sha256Hex(gzipSync(Buffer.from('biological-graph-bytes'))));
    expect(identity?.graphBinarySha256).toBe(sha256Hex(Buffer.from('biological-graph-bytes')));
  });

  it('resolves and validates "rewired-seed0" against rewiredArms.seed0', () => {
    const manifestPath = writeManifest(Buffer.from('biological-graph-bytes'), Buffer.from('rewired-graph-bytes'));
    const identity = resolveBigqGraphIdentity('rewired-seed0', manifestPath);
    expect(identity?.graphGzipSha256).toBe(sha256Hex(gzipSync(Buffer.from('rewired-graph-bytes'))));
    expect(identity?.graphBinarySha256).toBe(sha256Hex(Buffer.from('rewired-graph-bytes')));
  });

  it('returns null for "disconnected" (no separate artifact -- derived at runtime)', () => {
    const manifestPath = writeManifest(Buffer.from('bio'), Buffer.from('rewired'));
    expect(resolveBigqGraphIdentity('disconnected', manifestPath)).toBeNull();
  });

  it('throws when the manifest\'s claimed gzipSha256 does not match the actual artifact bytes (tampered manifest or stale artifact)', () => {
    const manifestPath = writeManifest(Buffer.from('biological-graph-bytes'), Buffer.from('rewired-graph-bytes'));
    // Overwrite the artifact after the manifest was written, so its gzip no longer matches.
    writeFileSync(resolve(root, 'malecns-arena-v1.bin.gz'), gzipSync(Buffer.from('DIFFERENT bytes')));
    expect(() => resolveBigqGraphIdentity('biological', manifestPath)).toThrow(/does not match/);
  });

  it('throws for an unknown bigq graphId', () => {
    const manifestPath = writeManifest(Buffer.from('bio'), Buffer.from('rewired'));
    expect(() => resolveBigqGraphIdentity('not-a-real-graph-id', manifestPath)).toThrow(/not a known bigq graphId/);
  });

  it('throws when the manifest has no rewiredArms.seed0', () => {
    const manifestPath = resolve(root, 'no-rewired.manifest.json');
    writeFileSync(
      manifestPath,
      JSON.stringify({ artifact: 'malecns-arena-v1.bin.gz', gzipSha256: 'x'.repeat(64), binarySha256: 'y'.repeat(64) })
    );
    expect(() => resolveBigqGraphIdentity('rewired-seed0', manifestPath)).toThrow(/no "rewiredArms.seed0" entry/);
  });
});

describe('resolveInterventionGraphIdentity', () => {
  const writeIndexWithGraph = (id: string, graphBytes: Buffer): { readonly index: GraphListIndex; readonly indexDir: string } => {
    const graphsDir = resolve(root, 'graphs');
    mkdirSync(graphsDir, { recursive: true });
    const gzipBytes = gzipSync(graphBytes);
    const relPath = `graphs/${id}.bin.gz`;
    writeFileSync(resolve(root, relPath), gzipBytes);
    const index: GraphListIndex = {
      sourceArtifact: 'public/data/malecns-arena-v1.bin.gz',
      sourceSha256: 'a'.repeat(64),
      entries: [{ id, path: relPath, gzipSha256: sha256Hex(gzipBytes), binarySha256: sha256Hex(graphBytes) }]
    };
    return { index, indexDir: root };
  };

  it('resolves and validates an intervention graphId against the index entry and the actual graph bytes', () => {
    const { index, indexDir } = writeIndexWithGraph('C000', Buffer.from('c000-graph-bytes'));
    const identity = resolveInterventionGraphIdentity('C000', index, indexDir);
    expect(identity.graphGzipSha256).toBe(sha256Hex(gzipSync(Buffer.from('c000-graph-bytes'))));
    expect(identity.graphBinarySha256).toBe(sha256Hex(Buffer.from('c000-graph-bytes')));
  });

  it('throws when the index has no entry for the requested graphId', () => {
    const { index, indexDir } = writeIndexWithGraph('C000', Buffer.from('c000-graph-bytes'));
    expect(() => resolveInterventionGraphIdentity('C999', index, indexDir)).toThrow(/no entry for id "C999"/);
  });

  it('throws when the index\'s claimed gzipSha256 does not match the actual graph file bytes', () => {
    const { index, indexDir } = writeIndexWithGraph('C000', Buffer.from('c000-graph-bytes'));
    // Overwrite the graph file after the index was built, so its gzip no longer matches.
    writeFileSync(resolve(indexDir, 'graphs/C000.bin.gz'), gzipSync(Buffer.from('TAMPERED')));
    expect(() => resolveInterventionGraphIdentity('C000', index, indexDir)).toThrow(/does not match/);
  });
});
