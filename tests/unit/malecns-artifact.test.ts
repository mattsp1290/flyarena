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
const compilerSourceDir = resolve(here, '../../scripts/data');

const sha256Hex = (data: Uint8Array): string => createHash('sha256').update(data).digest('hex');

/**
 * The exact filenames "the compiler" consists of -- kept in lockstep with
 * Python's `COMPILER_SOURCE_FILENAMES` in scripts/data/compile.py. This is
 * deliberately an explicit allowlist, not a `*.py` directory listing: a
 * non-compiler sidecar script that also lives in scripts/data/ (such as
 * positions.py, which joins the pinned annotations' soma columns onto the
 * compiled graph's own biologicalIds but never influences the compiled
 * .bin.gz bytes) must not change this hash.
 */
const COMPILER_SOURCE_FILENAMES = ['binfmt.py', 'compile.py', 'download.py', 'rewire.py'] as const;

/**
 * Recomputes `compilerSourceSha256` from the working tree: must exactly
 * match `compiler_source_sha256()` in scripts/data/compile.py (sorted
 * `COMPILER_SOURCE_FILENAMES`, each contributing filename + NUL byte + raw
 * bytes into one sha256 hasher). Kept in lockstep with the Python
 * implementation by
 * `test_compiler_source_sha256_matches_committed_ledger_and_manifest` in
 * tests_python/test_compile.py, which performs the same check from the
 * Python side.
 */
const computeCompilerSourceSha256 = (): string => {
  const filenames = [...COMPILER_SOURCE_FILENAMES].sort();
  const hash = createHash('sha256');
  for (const name of filenames) {
    hash.update(name, 'utf-8');
    hash.update(Buffer.from([0]));
    hash.update(readFileSync(resolve(compilerSourceDir, name)));
  }
  return hash.digest('hex');
};

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
  compilerSourceSha256: string;
  rewiredArms?: Record<
    string,
    { artifact: string; binarySha256: string; binaryBytes: number; gzipSha256: string; gzipBytes: number }
  >;
  positions?: {
    artifact: string;
    sha256: string;
    coverage: { soma: number; tosoma: number; none: number };
  };
}

interface Ledger {
  compilerSourceSha256: string;
  selectionCounts: {
    sensorySelectedCount: number;
    bridgeSelectedCount: number;
    descendingSelectedCount: number;
  };
  positionsCoverage?: { soma: number; tosoma: number; none: number };
}

interface PositionsDocument {
  version: number;
  sourceFile: string;
  sourceSha256: string;
  graphSha256: string;
  units: string;
  bodyIds: string[];
  role: Array<'sensory' | 'bridge' | 'descending'>;
  positionSource: Array<'soma' | 'tosoma' | 'none'>;
  xyz: Array<[number, number, number] | null>;
  coverage: { soma: number; tosoma: number; none: number };
  roleCounts: { sensory: number; bridge: number; descending: number };
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

  it('compilerSourceSha256, recomputed from the working-tree COMPILER_SOURCE_FILENAMES files, matches the committed manifest and ledger', () => {
    // Guards against the class of bug a prior review flagged in the
    // now-removed self-referential compilerRevision git SHA: a code change
    // to the compiler with no accompanying recompile/recommit of the
    // artifact. See docs/data-provenance.md's "Compiler provenance" section.
    const ledger = JSON.parse(
      readFileSync(resolve(publicDataDir, 'malecns-arena-v1.ledger.json'), 'utf-8')
    ) as Ledger;

    const recomputed = computeCompilerSourceSha256();

    expect(recomputed).toBe(manifest.compilerSourceSha256);
    expect(recomputed).toBe(ledger.compilerSourceSha256);
  });

  it('the soma positions sidecar (if present) is hash-consistent with the manifest, ledger, and compiled graph', () => {
    // scripts/data/positions.py is a separate offline sidecar (see
    // docs/data-provenance.md's "Soma positions sidecar" section) that is
    // not re-run by compile.py, so nothing else guarantees this stays
    // consistent after a recompile -- this is that guard.
    const positionsEntry = manifest.positions;
    expect(positionsEntry).toBeDefined();
    if (!positionsEntry) return;

    const positionsBytes = readFileSync(resolve(publicDataDir, positionsEntry.artifact));
    expect(sha256Hex(positionsBytes)).toBe(positionsEntry.sha256);

    const doc = JSON.parse(positionsBytes.toString('utf-8')) as PositionsDocument;
    expect(doc.graphSha256).toBe(manifest.gzipSha256);
    expect(doc.coverage).toEqual(positionsEntry.coverage);
    expect(doc.bodyIds.length).toBe(manifest.neuronCount);
    expect(doc.coverage.soma + doc.coverage.tosoma + doc.coverage.none).toBe(manifest.neuronCount);

    const { binary } = loadArtifact('malecns-arena-v1');
    const arrayBuffer = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
    const graph = parseGraphBinary(arrayBuffer);
    expect(doc.bodyIds).toEqual(Array.from(graph.biologicalIds, (id) => id.toString()));

    const ledger = JSON.parse(
      readFileSync(resolve(publicDataDir, 'malecns-arena-v1.ledger.json'), 'utf-8')
    ) as Ledger;
    expect(ledger.positionsCoverage).toEqual(doc.coverage);
    expect(doc.roleCounts).toEqual({
      sensory: ledger.selectionCounts.sensorySelectedCount,
      bridge: ledger.selectionCounts.bridgeSelectedCount,
      descending: ledger.selectionCounts.descendingSelectedCount
    });
  });
});
