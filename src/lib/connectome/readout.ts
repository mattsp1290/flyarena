import { OUTPUT_POPULATION } from '../arena/actions';
import type { ConnectomeGraph, GraphMode } from './format';

/**
 * Shared readout contract: one small MLP (`D -> H -> 3`) reading the
 * per-neuron rates of every output-assigned neuron and producing the three
 * `decodeAction` inputs (thrust, yaw, brake), in `OUTPUT_POPULATION` order.
 * This is the trainable surface of the trained-readout experiment
 * (`.agents/plans/trained-readout/00-overview.md`): the encoder, graph
 * topology, and recurrent dynamics in `model.ts` stay frozen, and this file
 * is the single TypeScript implementation the Worker, the Node evaluator,
 * and (by parity) the PyTorch training port all share.
 *
 * `D` (`inputSize`) is the count of output-assigned neurons in a given
 * graph (`outputPopulationIndex[i] >= 0`), not the 3 population sums that
 * `aggregateOutputs` produces: reading per-neuron rates instead of pre-summed
 * populations is what gives the readout real capacity.
 *
 * `readoutForward` is allocation-free, matching the per-step convention in
 * `model.ts`: call `createReadoutScratch`/`createReadoutOutput` once per
 * `(graph, hiddenSize)` pair and reuse the returned buffers across every
 * subsequent call.
 */

/**
 * Trainable readout weights. Row-major: `w1[h * inputSize + d]` is the
 * weight from input `d` to hidden unit `h`; `w2[o * hiddenSize + h]` is the
 * weight from hidden unit `h` to output `o` (`o` indexed by
 * `OUTPUT_POPULATION`).
 */
export interface ReadoutWeights {
  readonly inputSize: number;
  readonly hiddenSize: number;
  readonly w1: Float32Array;
  readonly b1: Float32Array;
  readonly w2: Float32Array;
  readonly b2: Float32Array;
}

/** Ascending neuron indices where `outputPopulationIndex[i] >= 0`; length is `D`. */
export const outputNeuronIndices = (graph: Readonly<ConnectomeGraph>): Int32Array => {
  const { outputPopulationIndex, metadata } = graph;
  const indices: number[] = [];
  for (let neuron = 0; neuron < metadata.neuronCount; neuron += 1) {
    if (outputPopulationIndex[neuron] >= 0) indices.push(neuron);
  }
  return Int32Array.from(indices);
};

/** Total trainable scalar count for a `D -> H -> 3` readout: `w1 + b1 + w2 + b2`. */
export const readoutParameterCount = (inputSize: number, hiddenSize: number): number =>
  hiddenSize * inputSize + hiddenSize + 3 * hiddenSize + 3;

/** Preallocated hidden-layer scratch; reused across `readoutForward` calls, never grown. */
export const createReadoutScratch = (hiddenSize: number): Float32Array =>
  new Float32Array(hiddenSize);

/** Preallocated 3-element output buffer, in `OUTPUT_POPULATION` order. */
export const createReadoutOutput = (): Float32Array => new Float32Array(3);

const invalidReadout = (message: string): never => {
  throw new Error(`Invalid readout weights: ${message}`);
};

const requireFiniteArray = (array: Float32Array, label: string): void => {
  for (let index = 0; index < array.length; index += 1) {
    if (!Number.isFinite(array[index])) invalidReadout(`${label}[${index}] must be finite`);
  }
};

/**
 * Validate a `ReadoutWeights` against the graph it will read from: `D` must
 * match the graph's output-neuron count, every array must have its expected
 * length, and every weight/bias must be finite. Returns `weights` for
 * chaining; throws on the first violation found.
 */
export const validateReadoutWeights = (
  weights: Readonly<ReadoutWeights>,
  graph: Readonly<ConnectomeGraph>
): Readonly<ReadoutWeights> => {
  const expectedInputSize = outputNeuronIndices(graph).length;
  if (weights.inputSize !== expectedInputSize) {
    invalidReadout(
      `inputSize ${weights.inputSize} does not match the graph's output-neuron count ${expectedInputSize}`
    );
  }
  if (weights.inputSize <= 0) {
    // A graph with no output-assigned neurons would otherwise pass here
    // (0 === 0) with a readout that has no way to read the network at all.
    invalidReadout('inputSize must be positive: the graph has no output-assigned neurons');
  }
  if (!Number.isInteger(weights.hiddenSize) || weights.hiddenSize <= 0) {
    invalidReadout('hiddenSize must be a positive integer');
  }

  const { inputSize, hiddenSize } = weights;
  const lengthChecks: ReadonlyArray<readonly [string, number, number]> = [
    ['w1', weights.w1.length, hiddenSize * inputSize],
    ['b1', weights.b1.length, hiddenSize],
    ['w2', weights.w2.length, 3 * hiddenSize],
    ['b2', weights.b2.length, 3]
  ];
  for (const [label, actual, expected] of lengthChecks) {
    if (actual !== expected) {
      invalidReadout(`${label} length ${actual} does not match expected length ${expected}`);
    }
  }

  requireFiniteArray(weights.w1, 'w1');
  requireFiniteArray(weights.b1, 'b1');
  requireFiniteArray(weights.w2, 'w2');
  requireFiniteArray(weights.b2, 'b2');

  return weights;
};

/**
 * On-disk shape of `public/data/trained-readout-v1.json` (and its tiny test
 * fixture, `tests/fixtures/trained-readout-tiny.json`): one shared `D -> H`
 * shape plus one `{ w1, b1, w2, b2 }` entry per arm, each array base64-encoded
 * little-endian float32 (see `.agents/plans/trained-readout/04-authoritative-evaluation-and-artifacts.md`).
 * `arms` is keyed by `GraphMode` in production, but typed as
 * `Record<string, ...>` here so a malformed/missing arm key is a normal,
 * handled `decodeReadoutArtifact` failure rather than a type-level
 * impossibility that can't actually be checked at a JSON boundary.
 */
export interface TrainedReadoutArtifactArm {
  readonly w1: string;
  readonly b1: string;
  readonly w2: string;
  readonly b2: string;
}

export interface TrainedReadoutArtifactJson {
  readonly version: number;
  readonly hiddenSize: number;
  readonly inputSize: number;
  readonly arms: Readonly<Record<string, TrainedReadoutArtifactArm>>;
}

/**
 * Base64 -> little-endian `Float32Array`, matching the tiny-fixture test's
 * own decode (`tests/unit/trained-readout-artifact.test.ts`). `atob` (not
 * Node's `Buffer`) so this runs unmodified in the browser Worker/main
 * thread, the app's real production callers, as well as under Vitest's
 * jsdom environment and any Node `tsx` script — all of which provide it.
 * Almost every real deployment target is little-endian (browsers, Node,
 * this project's CI), so a `Float32Array` view over the decoded bytes
 * already reads them correctly with no explicit byte-swap, matching the
 * artifact's documented little-endian encoding.
 */
const decodeBase64Float32 = (base64: string): Float32Array => {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`Invalid readout artifact: base64-decoded array length ${bytes.byteLength} is not a multiple of 4`);
  }
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
};

/**
 * Decode one arm's `ReadoutWeights` out of a parsed `trained-readout-v1.json`
 * document. Throws (never returns a partial/invalid result) if `arm` is
 * absent from `json.arms` or any of its four arrays fails to base64-decode
 * to a whole number of float32 elements; does **not** call
 * `validateReadoutWeights` itself — callers that have a graph to validate
 * against (the Worker's `init` handler, `ExperimentController`) do that
 * separately, since this function has no graph to check shapes against.
 */
export const decodeReadoutArtifact = (
  json: Readonly<TrainedReadoutArtifactJson>,
  arm: GraphMode | string
): ReadoutWeights => {
  const entry = json.arms[arm];
  if (!entry) {
    throw new Error(`Trained readout artifact has no arm "${arm}" (available: ${Object.keys(json.arms).join(', ')})`);
  }
  return {
    inputSize: json.inputSize,
    hiddenSize: json.hiddenSize,
    w1: decodeBase64Float32(entry.w1),
    b1: decodeBase64Float32(entry.b1),
    w2: decodeBase64Float32(entry.w2),
    b2: decodeBase64Float32(entry.b2)
  };
};

const sigmoid = (value: number): number => 1 / (1 + Math.exp(-value));

/**
 * Run the readout forward pass: `hidden = tanh(W1 . r + b1)`, then
 * `out = [tanh, tanh, sigmoid](W2 . hidden + b2)` in `OUTPUT_POPULATION`
 * order (thrust, yaw clamp to `[-1, 1]` by construction via `tanh`; brake to
 * `[0, 1]` via `sigmoid`). `rate` is a full per-neuron rate array (e.g.
 * `NeuralModelState.rate`); `indices` gathers the `D` output-neuron entries
 * from it (`outputNeuronIndices(graph)`). Arithmetic runs in JS doubles and
 * writes into `Float32Array` outputs, matching `model.ts`'s precision
 * convention. `scratch` must have length `weights.hiddenSize`; `out` must
 * have length 3. Neither is allocated here.
 *
 * `Math.tanh`/`Math.exp` are engine-approximated (the spec permits
 * implementation-defined rounding for both), so a PyTorch port matching
 * this function is a tolerance-based parity check, not a bitwise one; see
 * `.agents/plans/trained-readout/02-gpu-port-and-parity.md`'s tolerance
 * table.
 */
export const readoutForward = (
  weights: Readonly<ReadoutWeights>,
  rate: Float32Array,
  indices: Int32Array,
  scratch: Float32Array,
  out: Float32Array
): void => {
  const { inputSize, hiddenSize, w1, b1, w2, b2 } = weights;

  for (let hidden = 0; hidden < hiddenSize; hidden += 1) {
    let sum = b1[hidden];
    const rowOffset = hidden * inputSize;
    for (let input = 0; input < inputSize; input += 1) {
      sum += w1[rowOffset + input] * rate[indices[input]];
    }
    scratch[hidden] = Math.tanh(sum);
  }

  for (let output = 0; output < 3; output += 1) {
    let sum = b2[output];
    const rowOffset = output * hiddenSize;
    for (let hidden = 0; hidden < hiddenSize; hidden += 1) {
      sum += w2[rowOffset + hidden] * scratch[hidden];
    }
    out[output] = output === OUTPUT_POPULATION.brake ? sigmoid(sum) : Math.tanh(sum);
  }
};
