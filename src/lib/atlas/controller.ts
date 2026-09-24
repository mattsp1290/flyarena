import type { PreparedGraph } from '../counterfactual/targets';
import type { AtlasController } from '../counterfactual/decoder';
import { outputNeuronIndices } from '../connectome/readout';
import { sha256Hex } from '../experiment/assets';
import type { LoadedAtlas } from './types';

export async function resolveController(
  loaded: LoadedAtlas,
  prepared: PreparedGraph,
  id: number
): Promise<AtlasController> {
  const { atlas, sha256 } = loaded;
  if (prepared.identity.topology === 'rewired')
    throw new Error('Atlas probes support biological or disconnected topology');
  if (
    prepared.identity.sourceBinarySha256 !== atlas.graphBinarySha256 ||
    JSON.stringify(Array.from(outputNeuronIndices(prepared.graph))) !==
      JSON.stringify(atlas.outputNeuronIndices)
  )
    throw new Error('Atlas and probe graph identities differ');
  if (!atlas.cells.some((cell) => cell.id === id))
    throw new Error('Controller is not in the published atlas');
  const candidate = atlas.source.candidates.find((c) => c.id === id)!;
  const parameters = {
    inputSize: atlas.source.inputSize,
    hiddenSize: atlas.source.hiddenSize,
    theta: candidate.theta
  };
  const weightsSha256 = await sha256Hex(
    new TextEncoder().encode(JSON.stringify(parameters)).buffer
  );
  return { id, atlasSha256: sha256, weightsSha256, ...structuredClone(parameters) };
}
