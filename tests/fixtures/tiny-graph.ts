import type { ConnectomeGraph, GraphMetadata } from '../../src/lib/connectome/format';
import { SUPPORTED_FORMAT_VERSION } from '../../src/lib/connectome/format';

/**
 * Shared connectome fixtures. `createTinyGraph` is small enough that every
 * expected value in `tests/unit/model.test.ts` and `tests/unit/format.test.ts`
 * can be checked by hand; `createRandomGraph` produces a larger deterministic
 * graph for `tests/integration/worker-parity.test.ts`'s 1,000-step run.
 */

const DEFAULT_METADATA: GraphMetadata = {
  formatVersion: SUPPORTED_FORMAT_VERSION,
  neuronCount: 4,
  edgeCount: 2,
  inputChannelCount: 2,
  outputPopulationCount: 2,
  timestepSeconds: 1,
  leakRate: 0,
  rateMin: -1,
  rateMax: 1,
  inputClampMin: -1,
  inputClampMax: 1,
  globalGain: 1
};

/**
 * Four neurons: 0 and 1 are input neurons (channels 0 and 1); 0 excites 2
 * (sign +1, magnitude 2); 1 inhibits 3 (sign -1, magnitude 1.5). Neuron 2
 * feeds output population 0, neuron 3 feeds output population 1. Neither
 * output neuron has outgoing edges and neither input neuron has an incoming
 * edge, so one `stepModel` call from a known `state.rate` is analytically
 * exact by hand: `drive[2] = presynapticSigns[0] * rate[0] * globalGain *
 * contactMagnitudes[0]`, and symmetrically for `drive[3]` from neuron 1.
 *
 * `metadataOverrides` lets tests isolate one dynamic (leak, clamp, rate
 * bounds) at a time while keeping the same topology.
 */
export const createTinyGraph = (
  metadataOverrides: Partial<GraphMetadata> = {}
): ConnectomeGraph => ({
  metadata: { ...DEFAULT_METADATA, ...metadataOverrides },
  biologicalIds: BigUint64Array.from([1n, 2n, 3n, 4n]),
  // Row 0: [0, 2). Row 1: [1, 2). Rows 2 and 3 are empty.
  presynapticOffsets: Uint32Array.from([0, 1, 2, 2, 2]),
  postsynapticIndices: Uint32Array.from([2, 3]),
  contactMagnitudes: Float32Array.from([2, 1.5]),
  presynapticSigns: Int8Array.from([1, -1, 1, 1]),
  inputChannelIndex: Int32Array.from([0, 1, -1, -1]),
  inputWeight: Float32Array.from([1, 1, 0, 0]),
  outputPopulationIndex: Int32Array.from([-1, -1, 0, 1]),
  outputWeight: Float32Array.from([0, 0, 1, 1])
});

/** Deterministic mulberry32 PRNG, independent of the arena's own RNG. */
const mulberry32 = (seed: number): (() => number) => {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

export interface RandomGraphOptions {
  neuronCount?: number;
  inputChannelCount?: number;
  outputPopulationCount?: number;
  /** Expected outgoing edges per neuron. */
  edgeDensity?: number;
  metadataOverrides?: Partial<GraphMetadata>;
}

/**
 * A larger seeded random sparse graph for stress/parity testing. Every
 * neuron gets a fixed Dale's-law sign; each input channel and each output
 * population is assigned its own distinct neuron (via a seeded shuffle, so
 * none are dropped to a collision), and every input neuron gets a direct
 * edge to every output neuron so no output population can end up starved of
 * drive. Remaining recurrent edges are random but grouped into valid
 * presynaptic CSR rows. Bounded rate clamps mean the network cannot diverge
 * regardless of `globalGain`, so this stays safe to run for many steps.
 * Throws if `neuronCount` is too small to give every channel/population its
 * own neuron.
 */
export const createRandomGraph = (
  seed: number,
  options: RandomGraphOptions = {}
): ConnectomeGraph => {
  const neuronCount = options.neuronCount ?? 48;
  const inputChannelCount = options.inputChannelCount ?? 8;
  const outputPopulationCount = options.outputPopulationCount ?? 3;
  const edgeDensity = options.edgeDensity ?? 4;
  const random = mulberry32(seed);

  if (inputChannelCount + outputPopulationCount > neuronCount) {
    throw new Error(
      'createRandomGraph: neuronCount must be at least inputChannelCount + outputPopulationCount ' +
        'to assign every channel/population a distinct neuron'
    );
  }

  const biologicalIds = new BigUint64Array(neuronCount);
  const presynapticSigns = new Int8Array(neuronCount);
  const inputChannelIndex = new Int32Array(neuronCount).fill(-1);
  const inputWeight = new Float32Array(neuronCount);
  const outputPopulationIndex = new Int32Array(neuronCount).fill(-1);
  const outputWeight = new Float32Array(neuronCount);

  for (let neuron = 0; neuron < neuronCount; neuron += 1) {
    biologicalIds[neuron] = BigInt(1000 + neuron);
    presynapticSigns[neuron] = random() < 0.5 ? -1 : 1;
  }

  // Seeded partial Fisher-Yates: draw `inputChannelCount + outputPopulationCount`
  // distinct neurons so every channel and every population gets its own
  // neuron (an earlier version drew each independently and could silently
  // drop a channel/population to a collision).
  const order = Array.from({ length: neuronCount }, (_, index) => index);
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }

  const inputNeurons: number[] = [];
  for (let channel = 0; channel < inputChannelCount; channel += 1) {
    const neuron = order[channel];
    inputChannelIndex[neuron] = channel;
    inputWeight[neuron] = 0.5 + random();
    inputNeurons.push(neuron);
  }

  const outputNeurons: number[] = [];
  for (let population = 0; population < outputPopulationCount; population += 1) {
    const neuron = order[inputChannelCount + population];
    outputPopulationIndex[neuron] = population;
    outputWeight[neuron] = 0.3 + random() * 0.7;
    outputNeurons.push(neuron);
  }

  const edgesByPre: Array<Array<{ post: number; magnitude: number }>> = Array.from(
    { length: neuronCount },
    () => []
  );
  for (let pre = 0; pre < neuronCount; pre += 1) {
    const outDegree = Math.round(edgeDensity * random() * 2);
    for (let index = 0; index < outDegree; index += 1) {
      const post = Math.floor(random() * neuronCount);
      edgesByPre[pre].push({ post, magnitude: 0.1 + random() * 1.9 });
    }
  }

  // Guarantee every output population actually receives drive whenever any
  // input channel is active: a purely random topology can (and, checked
  // empirically, sometimes does) leave an output neuron with no incoming
  // edges at all, which would make its aggregated output identically zero
  // for every step and make a parity test over it vacuous.
  for (const inputNeuron of inputNeurons) {
    for (const outputNeuron of outputNeurons) {
      edgesByPre[inputNeuron].push({ post: outputNeuron, magnitude: 0.2 + random() * 0.8 });
    }
  }

  const presynapticOffsets = new Uint32Array(neuronCount + 1);
  let edgeCount = 0;
  for (let pre = 0; pre < neuronCount; pre += 1) {
    presynapticOffsets[pre] = edgeCount;
    edgeCount += edgesByPre[pre].length;
  }
  presynapticOffsets[neuronCount] = edgeCount;

  const postsynapticIndices = new Uint32Array(edgeCount);
  const contactMagnitudes = new Float32Array(edgeCount);
  let cursor = 0;
  for (let pre = 0; pre < neuronCount; pre += 1) {
    for (const edge of edgesByPre[pre]) {
      postsynapticIndices[cursor] = edge.post;
      contactMagnitudes[cursor] = edge.magnitude;
      cursor += 1;
    }
  }

  const metadata: GraphMetadata = {
    formatVersion: SUPPORTED_FORMAT_VERSION,
    neuronCount,
    edgeCount,
    inputChannelCount,
    outputPopulationCount,
    timestepSeconds: 1 / 30,
    leakRate: 0.35,
    rateMin: -2,
    rateMax: 2,
    inputClampMin: -1,
    inputClampMax: 1,
    globalGain: 0.5,
    ...options.metadataOverrides
  };

  return {
    metadata,
    biologicalIds,
    presynapticOffsets,
    postsynapticIndices,
    contactMagnitudes,
    presynapticSigns,
    inputChannelIndex,
    inputWeight,
    outputPopulationIndex,
    outputWeight
  };
};
