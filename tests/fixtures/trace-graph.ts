import type { ConnectomeGraph, GraphMetadata } from '../../src/lib/connectome/format';
import { SUPPORTED_FORMAT_VERSION } from '../../src/lib/connectome/format';

/**
 * Deterministic trace-export graph for `scripts/training/export-traces.ts`
 * and `tests/unit/readout.test.ts`/`tests/unit/golden-traces.test.ts`.
 *
 * `tests/fixtures/tiny-graph.ts`'s `createTinyGraph` has
 * `outputPopulationCount: 2` and only 4 neurons (one per output population,
 * none per input channel beyond the bare minimum), which would leave brake
 * permanently 0 in an exported trace. This fixture gives every output
 * population (thrust, yaw, brake) at least two real neurons so a golden
 * trace can exercise all three.
 *
 * 24 neurons: 8 input neurons (one per observation channel, `src/lib/arena/sensors.ts`'s
 * `OBSERVATION_CHANNELS`), 10 hidden neurons carrying recurrent drive, and 6
 * output neurons (2 per population). Roles are fixed constants below (not a
 * seeded shuffle like `tiny-graph.ts`'s `createRandomGraph`), so which
 * neuron is which is readable directly from the source; edge weights/signs
 * are drawn from a fixed-seed deterministic PRNG (the same mulberry32
 * algorithm `createRandomGraph` uses), plus a handful of explicitly authored
 * cycle edges so recurrence is guaranteed rather than merely probable.
 */

const NEURON_COUNT = 24;
const INPUT_CHANNEL_COUNT = 8;
const OUTPUT_POPULATION_COUNT = 3;

/** Neurons 0..7, one per observation channel (channel index === neuron index). */
const INPUT_NEURONS = Array.from({ length: INPUT_CHANNEL_COUNT }, (_, channel) => channel);

/** Neurons 8..17: recurrent hidden layer. */
const HIDDEN_NEURONS = Array.from({ length: 10 }, (_, index) => INPUT_CHANNEL_COUNT + index);

/**
 * Neurons 18..23: two per population, in `OUTPUT_POPULATION` order
 * (thrust, yaw, brake). Exported so tests can hand-verify
 * `outputNeuronIndices(createTraceGraph())` without recomputing this layout.
 */
export const TRACE_GRAPH_OUTPUT_NEURONS: readonly (readonly [number, number])[] = [
  [18, 19],
  [20, 21],
  [22, 23]
];

/** Flattened, ascending: the exact indices `outputNeuronIndices` must return. */
export const TRACE_GRAPH_OUTPUT_NEURON_INDICES = Int32Array.from(
  TRACE_GRAPH_OUTPUT_NEURONS.flat()
);

/**
 * Deterministic mulberry32 PRNG; same algorithm as `tiny-graph.ts`'s
 * `createRandomGraph`. Exported so `scripts/training/export-traces.ts` can
 * reuse it (for its own, differently-seeded weight generator) instead of
 * carrying a third copy.
 */
export const mulberry32 = (seed: number): (() => number) => {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/**
 * This fixture is deterministic for any seed value (same value in, same
 * graph out): what this particular constant controls is not reproducibility
 * but which graph comes out, and therefore whether the exported traces
 * satisfy `tests/unit/golden-traces.test.ts`'s "every decoded action
 * component is non-zero on at least one committed tick" assertion. `10` was
 * chosen by trying small integers against every one of the export script's
 * `TRACE_SEEDS` world seeds; unlike the graph this replaced, all three
 * action components are already non-zero from tick 0 on every committed
 * seed (checked with `TRACE_TICKS = 60`), so there is real margin, not a
 * near-miss. If this file's edge-generation logic changes, re-run that same
 * small-integer search before trusting the non-zero assertion again.
 */
const TRACE_GRAPH_SEED = 10;

/**
 * Build the trace-export graph. Deterministic: every call returns
 * numerically identical arrays (fresh instances, same values).
 */
export const createTraceGraph = (): ConnectomeGraph => {
  const random = mulberry32(TRACE_GRAPH_SEED);

  const biologicalIds = new BigUint64Array(NEURON_COUNT);
  const presynapticSigns = new Int8Array(NEURON_COUNT);
  const inputChannelIndex = new Int32Array(NEURON_COUNT).fill(-1);
  const inputWeight = new Float32Array(NEURON_COUNT);
  const outputPopulationIndex = new Int32Array(NEURON_COUNT).fill(-1);
  const outputWeight = new Float32Array(NEURON_COUNT);

  for (let neuron = 0; neuron < NEURON_COUNT; neuron += 1) {
    biologicalIds[neuron] = BigInt(2000 + neuron);
    // Both signs are used somewhere in the network (mixed signs); which
    // neuron gets which is seeded PRNG output, not a hand-picked pattern.
    presynapticSigns[neuron] = random() < 0.5 ? -1 : 1;
  }

  for (const neuron of INPUT_NEURONS) {
    inputChannelIndex[neuron] = neuron;
    inputWeight[neuron] = 0.6 + random() * 0.8;
  }

  for (let population = 0; population < OUTPUT_POPULATION_COUNT; population += 1) {
    for (const neuron of TRACE_GRAPH_OUTPUT_NEURONS[population]) {
      outputPopulationIndex[neuron] = population;
      outputWeight[neuron] = 0.4 + random() * 0.6;
    }
  }

  const edgesByPre: Array<Map<number, number>> = Array.from(
    { length: NEURON_COUNT },
    () => new Map()
  );
  const addEdge = (pre: number, post: number, magnitude: number): void => {
    const row = edgesByPre[pre];
    row.set(post, (row.get(post) ?? 0) + magnitude);
  };

  // Feedforward: every input neuron drives a handful of hidden neurons.
  for (const inputNeuron of INPUT_NEURONS) {
    const fanOut = 2 + Math.floor(random() * 2); // 2 or 3
    for (let index = 0; index < fanOut; index += 1) {
      const target = HIDDEN_NEURONS[Math.floor(random() * HIDDEN_NEURONS.length)];
      addEdge(inputNeuron, target, 0.2 + random() * 0.8);
    }
  }

  // Recurrent hidden layer: random sparse edges among hidden neurons, which
  // (combined with the explicit cycle edges below) gives the network real
  // recurrent dynamics rather than a purely feedforward pass.
  for (const pre of HIDDEN_NEURONS) {
    const fanOut = 1 + Math.floor(random() * 3);
    for (let index = 0; index < fanOut; index += 1) {
      const target = HIDDEN_NEURONS[Math.floor(random() * HIDDEN_NEURONS.length)];
      if (target === pre) continue; // self-loops are added explicitly below
      addEdge(pre, target, 0.1 + random() * 0.9);
    }
  }

  // Explicit cycles, so recurrence is guaranteed rather than merely likely:
  // a 2-cycle, a 3-cycle, and one self-loop (autapses are permitted; see
  // format.ts's validateGraph comment on self-loops).
  addEdge(8, 9, 0.5);
  addEdge(9, 8, 0.5);
  addEdge(10, 11, 0.4);
  addEdge(11, 12, 0.4);
  addEdge(12, 10, 0.4);
  addEdge(13, 13, 0.3);

  // Hidden -> output: every hidden neuron drives every output neuron so no
  // population is ever starved of drive. Mirrors createRandomGraph's own
  // guarantee for the same reason: an output neuron with no incoming edges
  // would make its population's aggregated output identically zero on every
  // tick, which would make the "every action component is non-zero on some
  // tick" acceptance test vacuous for that population.
  for (const hiddenNeuron of HIDDEN_NEURONS) {
    for (const population of TRACE_GRAPH_OUTPUT_NEURONS) {
      for (const outputNeuron of population) {
        addEdge(hiddenNeuron, outputNeuron, 0.2 + random() * 0.6);
      }
    }
  }

  // Canonicalize: strictly increasing postsynaptic indices per row (see
  // "Canonical row ordering and duplicate edges" in docs/graph-format.md).
  const sortedRows: Array<Array<[number, number]>> = edgesByPre.map((row) =>
    Array.from(row.entries()).sort(([a], [b]) => a - b)
  );
  const presynapticOffsets = new Uint32Array(NEURON_COUNT + 1);
  let edgeCount = 0;
  for (let pre = 0; pre < NEURON_COUNT; pre += 1) {
    presynapticOffsets[pre] = edgeCount;
    edgeCount += sortedRows[pre].length;
  }
  presynapticOffsets[NEURON_COUNT] = edgeCount;

  const postsynapticIndices = new Uint32Array(edgeCount);
  const contactMagnitudes = new Float32Array(edgeCount);
  let cursor = 0;
  for (let pre = 0; pre < NEURON_COUNT; pre += 1) {
    for (const [post, magnitude] of sortedRows[pre]) {
      postsynapticIndices[cursor] = post;
      contactMagnitudes[cursor] = magnitude;
      cursor += 1;
    }
  }

  const metadata: GraphMetadata = {
    formatVersion: SUPPORTED_FORMAT_VERSION,
    neuronCount: NEURON_COUNT,
    edgeCount,
    inputChannelCount: INPUT_CHANNEL_COUNT,
    outputPopulationCount: OUTPUT_POPULATION_COUNT,
    timestepSeconds: 1 / 30,
    leakRate: 0.35,
    rateMin: -2,
    rateMax: 2,
    inputClampMin: -1,
    inputClampMax: 1,
    globalGain: 0.5
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
