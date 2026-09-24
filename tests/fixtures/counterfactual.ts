import { encodeGraphBinary, type ConnectomeGraph } from '../../src/lib/connectome/format';
import { sha256Hex, type LoadedArenaArtifacts } from '../../src/lib/experiment/assets';
import { createTraceGraph } from './trace-graph';

export async function counterfactualAssets(graph: ConnectomeGraph = createTraceGraph()): Promise<LoadedArenaArtifacts> {
  const biological = encodeGraphBinary(graph);
  const binarySha256 = await sha256Hex(biological);
  const entry = { artifact: 'fixture.bin.gz', binarySha256, binaryBytes: biological.byteLength, gzipBytes: 0, gzipSha256: '' };
  return {
    biological, rewired: biological.slice(0),
    manifest: { ...entry, formatVersion:1, neuronCount:graph.metadata.neuronCount, edgeCount:graph.metadata.edgeCount,
      inputChannelCount:8, outputPopulationCount:3, sourceDataset:'Authored test fixture', license:'Test only',
      rewiredArms:{ seed0: { ...entry, swapStats:{edgeCount:graph.metadata.edgeCount} } } }
  };
}
