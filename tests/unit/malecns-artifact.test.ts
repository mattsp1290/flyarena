import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { parseGraphBinary, validateGraph } from '../../src/lib/connectome/format';

/**
 * Loads the checked-in, real MaleCNS-derived artifact (compiled offline by
 * scripts/data/compile.py -- see docs/data-provenance.md) through the exact
 * TypeScript parser the production app uses, and cross-checks it against
 * its own manifest/ledger. This runs in CI without downloading any raw
 * MaleCNS data: the artifact and its JSON siblings are small, committed
 * files under public/data/.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const sha256Hex = (data: Uint8Array): string => createHash('sha256').update(data).digest('hex');

interface Manifest {
  formatVersion: number;
  neuronCount: number;
  edgeCount: number;
  inputChannelCount: number;
  outputPopulationCount: number;
  binarySha256: string;
  binaryBytes: number;
  gzipSha256: string;
  gzipBytes: number;
  license: string;
  rewiredArms?: Record<
    string,
    { artifact: string; binarySha256: string; binaryBytes: number; gzipSha256: string; gzipBytes: number }
  >;
}

const loadArtifact = (basename: string) => {
  const gzipBytes = readFileSync(resolve(publicDataDir, `${basename}.bin.gz`));
  const binary = gunzipSync(gzipBytes);
  return { gzipBytes, binary };
};

describe('malecns-arena-v1 artifact (real, pinned MaleCNS-derived graph)', () => {
  const manifest = JSON.parse(
    readFileSync(resolve(publicDataDir, 'malecns-arena-v1.manifest.json'), 'utf-8')
  ) as Manifest;

  it('gzip and decompressed sha256 match the manifest', () => {
    const { gzipBytes, binary } = loadArtifact('malecns-arena-v1');

    expect(gzipBytes.byteLength).toBe(manifest.gzipBytes);
    expect(sha256Hex(gzipBytes)).toBe(manifest.gzipSha256);

    expect(binary.byteLength).toBe(manifest.binaryBytes);
    expect(sha256Hex(binary)).toBe(manifest.binarySha256);
  });

  it('parses and validates through the production TypeScript parser', () => {
    const { binary } = loadArtifact('malecns-arena-v1');
    // Node's Buffer (from gunzipSync) is a view over a larger pooled
    // ArrayBuffer; slice to the exact byte range before handing it to
    // parseGraphBinary, which expects an ArrayBuffer sized to the graph.
    const arrayBuffer = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);

    const graph = parseGraphBinary(arrayBuffer);
    expect(() => validateGraph(graph)).not.toThrow();

    expect(graph.metadata.formatVersion).toBe(manifest.formatVersion);
    expect(graph.metadata.neuronCount).toBe(manifest.neuronCount);
    expect(graph.metadata.edgeCount).toBe(manifest.edgeCount);
    expect(graph.metadata.inputChannelCount).toBe(manifest.inputChannelCount);
    expect(graph.metadata.outputPopulationCount).toBe(manifest.outputPopulationCount);

    // Sanity checks on the declared I/O contract this repo's model relies
    // on: at least one neuron drives every declared input channel and
    // every declared output population (an all-dead channel would make the
    // artifact silently inert for that channel/population).
    const channelsWithInput = new Set<number>();
    const populationsWithOutput = new Set<number>();
    for (let neuron = 0; neuron < graph.metadata.neuronCount; neuron += 1) {
      const channel = graph.inputChannelIndex[neuron];
      if (channel >= 0) channelsWithInput.add(channel);
      const population = graph.outputPopulationIndex[neuron];
      if (population >= 0) populationsWithOutput.add(population);
    }
    expect(channelsWithInput.size).toBe(graph.metadata.inputChannelCount);
    expect(populationsWithOutput.size).toBe(graph.metadata.outputPopulationCount);
  });

  it('license and format version are the expected pinned values', () => {
    expect(manifest.license).toBe('CC-BY-4.0');
    expect(manifest.formatVersion).toBe(1);
  });

  it('the rewired control arm (if present) also parses/validates and matches its manifest hash', () => {
    const rewiredEntry = manifest.rewiredArms?.seed0;
    // The rewired arm is optional at the format level (a bean could ship
    // biological-only), but this compiler always produces one alongside
    // the biological artifact -- assert it is actually present rather than
    // silently skipping the check.
    expect(rewiredEntry).toBeDefined();
    if (!rewiredEntry) return;

    const gzipBytes = readFileSync(resolve(publicDataDir, rewiredEntry.artifact));
    expect(sha256Hex(gzipBytes)).toBe(rewiredEntry.gzipSha256);

    const binary = gunzipSync(gzipBytes);
    expect(sha256Hex(binary)).toBe(rewiredEntry.binarySha256);

    const arrayBuffer = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
    const graph = parseGraphBinary(arrayBuffer);
    expect(() => validateGraph(graph)).not.toThrow();

    // The rewired arm must share the biological arm's node set exactly
    // (same neuron count/IDs) and differ only in topology.
    const { binary: biologicalBinary } = loadArtifact('malecns-arena-v1');
    const biologicalArrayBuffer = biologicalBinary.buffer.slice(
      biologicalBinary.byteOffset,
      biologicalBinary.byteOffset + biologicalBinary.byteLength
    );
    const biologicalGraph = parseGraphBinary(biologicalArrayBuffer);
    expect(Array.from(graph.biologicalIds)).toEqual(Array.from(biologicalGraph.biologicalIds));
    expect(graph.metadata.edgeCount).toBe(biologicalGraph.metadata.edgeCount);
  });
});
