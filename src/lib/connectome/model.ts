import type { ConnectomeGraph } from './format';

/**
 * Bounded leaky rate network: the synchronous CPU oracle for the sparse
 * connectome. Dynamics here are authored/literature-derived (see
 * docs/model-ledger.md), not a measurement from the source animal. The same
 * functions back both the direct in-thread oracle and the Worker runtime so
 * the two can never drift apart.
 *
 * Every function that runs once per neural substep is allocation-free: call
 * `createModelState`/`createStepScratch`/`createOutputBuffer` once per graph
 * and reuse the returned buffers across every subsequent `stepModel` call.
 */

/** Mutable float32 network state: the current bounded rate of every neuron. */
export interface NeuralModelState {
  readonly rate: Float32Array;
}

export const createModelState = (graph: Readonly<ConnectomeGraph>): NeuralModelState => ({
  rate: new Float32Array(graph.metadata.neuronCount)
});

/** Zero every neuron's rate. Does not touch input/output buffers. */
export const resetModelState = (state: NeuralModelState): void => {
  state.rate.fill(0);
};

/** Preallocated per-step scratch; reused across `stepModel` calls, never grown. */
export interface StepScratch {
  readonly drive: Float32Array;
}

export const createStepScratch = (graph: Readonly<ConnectomeGraph>): StepScratch => ({
  drive: new Float32Array(graph.metadata.neuronCount)
});

export const createOutputBuffer = (graph: Readonly<ConnectomeGraph>): Float32Array =>
  new Float32Array(graph.metadata.outputPopulationCount);

const clamp = (value: number, minimum: number, maximum: number): number =>
  value < minimum ? minimum : value > maximum ? maximum : value;

/**
 * Advance the network by one substep of `graph.metadata.timestepSeconds`.
 *
 * Per neuron: accumulate recurrent synaptic drive (presynaptic sign x
 * positive contact magnitude x presynaptic rate x global gain, scattered
 * from every presynaptic row onto its postsynaptic targets) plus clamped
 * external drive for input-mapped neurons, then apply explicit-Euler leak
 * integration and clamp the result to `[rateMin, rateMax]`.
 *
 * `channelValues` is read-only and indexed by `graph.inputChannelIndex`;
 * it must have at least `graph.metadata.inputChannelCount` entries.
 */
export const stepModel = (
  graph: Readonly<ConnectomeGraph>,
  state: NeuralModelState,
  scratch: StepScratch,
  channelValues: ArrayLike<number>
): void => {
  const {
    metadata,
    presynapticOffsets,
    postsynapticIndices,
    contactMagnitudes,
    presynapticSigns,
    inputChannelIndex,
    inputWeight
  } = graph;
  const { neuronCount, timestepSeconds, leakRate, rateMin, rateMax, inputClampMin, inputClampMax, globalGain } =
    metadata;
  const { drive } = scratch;
  const { rate } = state;

  drive.fill(0);

  // Recurrent synaptic drive: scatter each presynaptic row onto its targets.
  for (let pre = 0; pre < neuronCount; pre += 1) {
    const presynapticRate = rate[pre];
    if (presynapticRate === 0) continue;
    const signedRate = presynapticSigns[pre] * presynapticRate * globalGain;
    const start = presynapticOffsets[pre];
    const end = presynapticOffsets[pre + 1];
    for (let edge = start; edge < end; edge += 1) {
      drive[postsynapticIndices[edge]] += signedRate * contactMagnitudes[edge];
    }
  }

  // External sensory drive for input-mapped neurons, clamped before injection.
  for (let neuron = 0; neuron < neuronCount; neuron += 1) {
    const channel = inputChannelIndex[neuron];
    if (channel < 0) continue;
    const raw = channelValues[channel] ?? 0;
    drive[neuron] += inputWeight[neuron] * clamp(raw, inputClampMin, inputClampMax);
  }

  // Bounded leaky explicit-Euler integration.
  for (let neuron = 0; neuron < neuronCount; neuron += 1) {
    const next = rate[neuron] + timestepSeconds * (-leakRate * rate[neuron] + drive[neuron]);
    rate[neuron] = clamp(next, rateMin, rateMax);
  }
};

/**
 * Aggregate per-neuron rates into per-population action features:
 * `outputs[population] = sum(outputWeight[neuron] * rate[neuron])` over every
 * neuron assigned to that population. `outputs` must have length
 * `graph.metadata.outputPopulationCount`; it is overwritten, not accumulated.
 *
 * By convention output population 0/1/2 map to thrust/yaw/brake to match
 * `src/lib/arena/actions.ts`'s `decodeAction` order, but this function and
 * the graph format are agnostic to population count and meaning.
 */
export const aggregateOutputs = (
  graph: Readonly<ConnectomeGraph>,
  state: Readonly<NeuralModelState>,
  outputs: Float32Array
): void => {
  outputs.fill(0);
  const { outputPopulationIndex, outputWeight, metadata } = graph;
  const { rate } = state;
  for (let neuron = 0; neuron < metadata.neuronCount; neuron += 1) {
    const population = outputPopulationIndex[neuron];
    if (population < 0) continue;
    outputs[population] += outputWeight[neuron] * rate[neuron];
  }
};

/**
 * Run `substeps` consecutive `stepModel` calls with the same held-constant
 * `channelValues`, then aggregate into `outputs`. This is the one call per
 * world tick the Worker runtime (and the in-thread oracle) actually makes;
 * see `docs/architecture.md`'s closed-loop order.
 */
export const runSubsteps = (
  graph: Readonly<ConnectomeGraph>,
  state: NeuralModelState,
  scratch: StepScratch,
  channelValues: ArrayLike<number>,
  substeps: number,
  outputs: Float32Array
): void => {
  for (let step = 0; step < substeps; step += 1) {
    stepModel(graph, state, scratch, channelValues);
  }
  aggregateOutputs(graph, state, outputs);
};
