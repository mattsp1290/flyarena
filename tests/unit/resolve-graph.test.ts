import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { gzipSync } from 'node:zlib';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { resolveReadoutGraph } from '../../scripts/attribution/resolve-graph';
import { createTraceGraph } from '../fixtures/trace-graph';

/**
 * Coverage for `scripts/attribution/resolve-graph.ts`
 * (`.agents/plans/readout-attribution/02-analyses.md`'s WP2 test
 * requirement: "resolve-graph.ts returns the intervention graph for a
 * P/C/M entry and throws for an unknown one"). Builds a real, wire-format
 * `ConnectomeGraph` fixture (`createTraceGraph`, re-encoded) rather than
 * arbitrary bytes, since `resolveReadoutGraph` parses the resolved bytes as
 * a graph (unlike `graph-identity.test.ts`'s pure sha-comparison fixtures).
 */

const sha256Hex = (data: Buffer): string => createHash('sha256').update(data).digest('hex');

let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'resolve-graph-'));
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

const writeManifest = (): { readonly manifestPath: string; readonly bioBinary: Buffer } => {
  const bioBinary = Buffer.from(encodeGraphBinary(createTraceGraph()));
  const bioGzip = gzipSync(bioBinary);
  writeFileSync(resolve(root, 'malecns-arena-v1.bin.gz'), bioGzip);
  // Reuse the same bytes for "rewired-seed0" -- the fixture only needs to
  // decode as a valid graph, not represent a real rewiring.
  writeFileSync(resolve(root, 'malecns-arena-v1-rewired-seed0.bin.gz'), bioGzip);
  const manifest = {
    artifact: 'malecns-arena-v1.bin.gz',
    gzipSha256: sha256Hex(bioGzip),
    binarySha256: sha256Hex(bioBinary),
    rewiredArms: {
      seed0: { artifact: 'malecns-arena-v1-rewired-seed0.bin.gz', gzipSha256: sha256Hex(bioGzip), binarySha256: sha256Hex(bioBinary) }
    }
  };
  const manifestPath = resolve(root, 'malecns-arena-v1.manifest.json');
  writeFileSync(manifestPath, JSON.stringify(manifest));
  return { manifestPath, bioBinary };
};

const writeInterventionIndex = (): { readonly indexPath: string; readonly archivedIndexPath: string; readonly binary: Buffer } => {
  const binary = Buffer.from(encodeGraphBinary(createTraceGraph()));
  const gzip = gzipSync(binary);
  const graphsDir = resolve(root, 'graphs');
  mkdirSync(graphsDir, { recursive: true });
  writeFileSync(resolve(graphsDir, 'C000.bin.gz'), gzip);
  const index = {
    sourceArtifact: 'malecns-arena-v1.bin.gz',
    sourceSha256: 'a'.repeat(64),
    entries: [{ id: 'C000', path: 'graphs/C000.bin.gz', gzipSha256: sha256Hex(gzip), binarySha256: sha256Hex(binary) }]
  };
  const indexPath = resolve(root, 'index.json');
  writeFileSync(indexPath, JSON.stringify(index));
  const archivedIndexPath = resolve(root, 'archived-index.json');
  writeFileSync(archivedIndexPath, JSON.stringify(index));
  return { indexPath, archivedIndexPath, binary };
};

describe('resolveReadoutGraph', () => {
  it('resolves "biological" from the manifest and parses it as a valid graph', () => {
    const { manifestPath, bioBinary } = writeManifest();
    const graph = resolveReadoutGraph(
      { graphId: 'biological', graphGzipSha256: sha256Hex(gzipSync(bioBinary)), graphBinarySha256: sha256Hex(bioBinary) },
      { manifestPath }
    );
    expect(graph.metadata.neuronCount).toBe(createTraceGraph().metadata.neuronCount);
  });

  it('resolves "disconnected" by deriving it from the sha-verified biological graph', () => {
    const { manifestPath } = writeManifest();
    const graph = resolveReadoutGraph({ graphId: 'disconnected', graphGzipSha256: null, graphBinarySha256: null }, { manifestPath });
    expect(graph.metadata.edgeCount).toBe(0);
    expect(graph.metadata.neuronCount).toBe(createTraceGraph().metadata.neuronCount);
  });

  it('throws if a "disconnected" entry has a non-null graph sha (archive invariant violated)', () => {
    const { manifestPath } = writeManifest();
    expect(() =>
      resolveReadoutGraph({ graphId: 'disconnected', graphGzipSha256: 'deadbeef', graphBinarySha256: null }, { manifestPath })
    ).toThrow(/null/);
  });

  it('resolves an intervention id (e.g. "C000") from the physical index, verified against the archived copy', () => {
    const { manifestPath } = writeManifest();
    const { indexPath, archivedIndexPath, binary } = writeInterventionIndex();
    const graph = resolveReadoutGraph(
      { graphId: 'C000', graphGzipSha256: sha256Hex(gzipSync(binary)), graphBinarySha256: sha256Hex(binary) },
      { manifestPath, interventionIndexPath: indexPath, archivedInterventionIndexPath: archivedIndexPath }
    );
    expect(graph.metadata.neuronCount).toBe(createTraceGraph().metadata.neuronCount);
  });

  it('throws for an unknown graphId', () => {
    const { manifestPath } = writeManifest();
    const { indexPath, archivedIndexPath } = writeInterventionIndex();
    expect(() =>
      resolveReadoutGraph(
        { graphId: 'not-a-real-id', graphGzipSha256: null, graphBinarySha256: null },
        { manifestPath, interventionIndexPath: indexPath, archivedInterventionIndexPath: archivedIndexPath }
      )
    ).toThrow();
  });

  it('throws when the physical intervention index does not byte-match the archived copy', () => {
    const { manifestPath } = writeManifest();
    const { indexPath, binary } = writeInterventionIndex();
    const tamperedArchivePath = resolve(root, 'tampered-archive-index.json');
    writeFileSync(tamperedArchivePath, JSON.stringify({ sourceArtifact: 'x', sourceSha256: 'b'.repeat(64), entries: [] }));
    expect(() =>
      resolveReadoutGraph(
        { graphId: 'C000', graphGzipSha256: sha256Hex(gzipSync(binary)), graphBinarySha256: sha256Hex(binary) },
        { manifestPath, interventionIndexPath: indexPath, archivedInterventionIndexPath: tamperedArchivePath }
      )
    ).toThrow(/does not match/);
  });

  it('throws when the archive\'s recorded graphBinarySha256 does not match the regenerated graph', () => {
    const { manifestPath, bioBinary } = writeManifest();
    expect(() =>
      resolveReadoutGraph(
        { graphId: 'biological', graphGzipSha256: sha256Hex(gzipSync(bioBinary)), graphBinarySha256: 'f'.repeat(64) },
        { manifestPath }
      )
    ).toThrow(/does not match/);
  });
});
