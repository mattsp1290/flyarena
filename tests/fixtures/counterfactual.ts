import { encodeGraphBinary, parseGraphBinary, type ConnectomeGraph } from '../../src/lib/connectome/format';
import { sha256Hex, type LoadedArenaArtifacts } from '../../src/lib/experiment/assets';
import { createTraceGraph } from './trace-graph';

export async function counterfactualAssets(graph: ConnectomeGraph = createTraceGraph()): Promise<LoadedArenaArtifacts> {
  const biological = encodeGraphBinary(graph);
  const binarySha256 = await sha256Hex(biological);
  const entry = { artifact: 'fixture.bin.gz', binarySha256, binaryBytes: biological.byteLength, gzipBytes: 0, gzipSha256: '' };
  return {
    // `parsedBiological` must be the *decoded* graph (round-tripped through
    // `encodeGraphBinary`/`parseGraphBinary`, same as `biological` itself),
    // not the original in-memory `graph` object — real production code
    // (`loadArenaArtifacts`) only ever gets a parsed graph by decoding the
    // fetched binary artifact, which stores values in `Float32Array`s and so
    // is not bit-identical to a pristine double-precision fixture object.
    // Handing back `graph` directly here previously produced a
    // higher-precision `parsedBiological` than production ever sees, which
    // silently diverged `prepareGraph`'s output from what the encoded
    // `biological`/`rewired` buffers alone would parse to (caught by
    // `tests/unit/counterfactual-engine.test.ts`'s determinism snapshot once
    // `prepareGraph` started reusing `assets.parsedBiological` instead of
    // re-parsing `biological` itself).
    biological, rewired: biological.slice(0), parsedBiological: parseGraphBinary(biological.slice(0)),
    manifest: { ...entry, formatVersion:1, neuronCount:graph.metadata.neuronCount, edgeCount:graph.metadata.edgeCount,
      inputChannelCount:8, outputPopulationCount:3, sourceDataset:'Authored test fixture', license:'Test only',
      rewiredArms:{ seed0: { ...entry, swapStats:{edgeCount:graph.metadata.edgeCount} } } }
  };
}
