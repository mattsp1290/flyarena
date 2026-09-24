import type { ConnectomeGraph } from '../connectome/format';
import {
  outputNeuronIndices,
  readoutForward,
  validateReadoutWeights,
  createReadoutScratch,
  createReadoutOutput
} from '../connectome/readout';
import { readoutFromFlat } from '../connectome/readout-serialization';

export interface AtlasController {
  id: number;
  atlasSha256: string;
  weightsSha256: string;
  inputSize: number;
  hiddenSize: number;
  theta: number[];
}
export type Decoder =
  | { decoder: 'authored' }
  | { decoder: 'atlas-trained'; controller: AtlasController };
export const AUTHORED: Decoder = { decoder: 'authored' };

/** Each branch owns scratch; only immutable weights are conceptually shared. */
export function createDecoder(graph: ConnectomeGraph, spec: Decoder) {
  if (spec.decoder === 'authored')
    return (_rates: Float32Array, outputs: Float32Array) => Array.from(outputs);
  const controller = spec.controller;
  if (
    controller.hiddenSize !== 8 ||
    !Number.isInteger(controller.id) ||
    controller.id < 0 ||
    controller.id > 12287 ||
    !/^[a-f0-9]{64}$/.test(controller.atlasSha256) ||
    !/^[a-f0-9]{64}$/.test(controller.weightsSha256) ||
    controller.theta.some((v) => !Number.isFinite(v) || Math.abs(v) > 8)
  )
    throw new Error('Invalid atlas controller');
  const weights = validateReadoutWeights(
    readoutFromFlat(controller.theta, controller.inputSize, controller.hiddenSize),
    graph
  );
  const indices = outputNeuronIndices(graph),
    scratch = createReadoutScratch(weights.hiddenSize),
    out = createReadoutOutput();
  return (rates: Float32Array) => {
    readoutForward(weights, rates, indices, scratch, out);
    return Array.from(out);
  };
}
