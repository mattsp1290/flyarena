import { createDisconnectedGraph, encodeGraphBinary, parseGraphBinary, type ConnectomeGraph, type GraphMode } from '../connectome/format';
import { loadArenaArtifacts, sha256Hex, type LoadedArenaArtifacts } from '../experiment/assets';
import { TARGET_LABELS, type Preparation, type TargetId } from './types';

export interface PreparedGraph extends Preparation { graph: ConnectomeGraph }

export async function prepareGraph(assets: LoadedArenaArtifacts, mode: GraphMode): Promise<PreparedGraph> {
  if (!['biological', 'rewired', 'disconnected'].includes(mode)) throw new Error('Unknown topology');
  const source = mode === 'rewired' ? assets.rewired : assets.biological;
  // `rewired` has no parsed-graph counterpart on `LoadedArenaArtifacts` (only
  // the biological arm's parse is threaded through — see `assets.ts`), so it
  // still parses its own source here. `biological`/`disconnected` reuse
  // `assets.parsedBiological` instead of re-parsing `assets.biological` a
  // second time (thermo-architecture I1 fix, code-review follow-up): the
  // caller (`loadArenaArtifacts`, or a CLI/test fixture matching its
  // contract) already parsed it once to cross-check `neuronCount`/
  // `edgeCount`, and that parse would otherwise be silently redone here.
  const biologicalOrRewired = mode === 'rewired' ? parseGraphBinary(source.slice(0)) : assets.parsedBiological;
  const graph = mode === 'disconnected' ? createDisconnectedGraph(biologicalOrRewired) : biologicalOrRewired;
  const { metadata: m } = graph;
  if (m.inputChannelCount !== 8 || m.outputPopulationCount !== 3 || m.rateMin > 0 || m.rateMax < 0) {
    throw new Error('Graph is incompatible with arena observations, actions or zero-rate silencing');
  }
  const entry = mode === 'rewired' ? assets.manifest.rewiredArms.seed0 : assets.manifest;
  const sourceHash = await sha256Hex(source);
  if (sourceHash !== entry.binarySha256) throw new Error('Source graph hash mismatch');
  const identity = {
    topology: mode, binarySha256: mode === 'disconnected' ? await sha256Hex(encodeGraphBinary(graph)) : sourceHash,
    sourceBinarySha256: sourceHash, sourceArtifact: entry.artifact,
    sourceDataset: assets.manifest.sourceDataset, license: assets.manifest.license,
    neuronCount: m.neuronCount, edgeCount: m.edgeCount
  };
  const targets = (Object.keys(TARGET_LABELS) as TargetId[]).map(id => {
    const indices = Array.from({ length: m.neuronCount }, (_, i) => i).filter(i => {
      if (id === 'bridge') return graph.inputChannelIndex[i] < 0 && graph.outputPopulationIndex[i] < 0;
      if (id === 'output') return graph.outputPopulationIndex[i] >= 0;
      return graph.inputChannelIndex[i] === Number(id.slice(6));
    });
    return { id, label: TARGET_LABELS[id], indices, bodyIds: indices.map(i => graph.biologicalIds[i].toString()) };
  });
  return { graph, identity, targets };
}

export const loadPreparedGraph = async (baseUrl: string, topology: GraphMode): Promise<PreparedGraph> =>
  prepareGraph(await loadArenaArtifacts(baseUrl), topology);
