import { describe, expect, it } from 'vitest';
import { validateGraph } from '../../src/lib/connectome/format';
import {
  createReadoutOutput,
  createReadoutScratch,
  outputNeuronIndices,
  readoutForward,
  readoutParameterCount,
  validateReadoutWeights,
  type ReadoutWeights
} from '../../src/lib/connectome/readout';
import { createTraceGraph, TRACE_GRAPH_OUTPUT_NEURON_INDICES } from '../fixtures/trace-graph';

describe('outputNeuronIndices', () => {
  it('returns ascending output-neuron indices matching the trace graph\'s hand-computed layout', () => {
    const graph = createTraceGraph();
    // Hand-computed from trace-graph.ts's TRACE_GRAPH_OUTPUT_NEURONS layout:
    // two neurons per population (thrust, yaw, brake), in ascending order.
    expect(Array.from(outputNeuronIndices(graph))).toEqual([18, 19, 20, 21, 22, 23]);
    expect(Array.from(outputNeuronIndices(graph))).toEqual(
      Array.from(TRACE_GRAPH_OUTPUT_NEURON_INDICES)
    );
  });
});

describe('readoutParameterCount', () => {
  it('equals the sum of every ReadoutWeights array length for the same D, H', () => {
    const inputSize = 6;
    const hiddenSize = 4;
    const weights: ReadoutWeights = {
      inputSize,
      hiddenSize,
      w1: new Float32Array(hiddenSize * inputSize),
      b1: new Float32Array(hiddenSize),
      w2: new Float32Array(3 * hiddenSize),
      b2: new Float32Array(3)
    };
    const total = weights.w1.length + weights.b1.length + weights.w2.length + weights.b2.length;
    expect(readoutParameterCount(inputSize, hiddenSize)).toBe(total);
    expect(readoutParameterCount(inputSize, hiddenSize)).toBe(43);
  });
});

describe('readoutForward', () => {
  it('matches an analytically computed forward pass, gathering out-of-order indices (H = 3)', () => {
    // D = 3, H = 3. `rate` has decoys at indices 0, 2, 5 set to extreme,
    // unmistakable values; `indices = [4, 1, 3]` gathers the three real
    // (small, ordinary) values out of order and skipping the decoys. w1 is
    // identity-like (hidden[d] = tanh(gathered[d])) and w2 routes each
    // hidden unit to exactly one output (thrust = tanh(hidden0), yaw =
    // tanh(hidden1), brake = sigmoid(hidden2)), so every one of the three
    // gathered inputs drives exactly one output end to end. An
    // implementation that read `rate[0..D-1]` instead of
    // `rate[indices[0..D-1]]` would substitute the huge decoys at rate[0]
    // and rate[2] for the intended rate[4]/rate[3] (rate[1] happens to
    // coincide with `indices[1]`, so yaw alone would not expose that bug):
    // thrust would come out ~0.76 instead of ~0.58 and brake ~0.27 instead
    // of ~0.43 — both well outside the 1e-7 tolerance below.
    const inputSize = 3;
    const hiddenSize = 3;
    const weights: ReadoutWeights = {
      inputSize,
      hiddenSize,
      w1: Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      b1: Float32Array.from([0, 0, 0]),
      w2: Float32Array.from([1, 0, 0, 0, 1, 0, 0, 0, 1]),
      b2: Float32Array.from([0, 0, 0])
    };
    const rate = Float32Array.from([1e6, 0.5, -1e6, -0.3, 0.8, 2e6]);
    const indices = Int32Array.from([4, 1, 3]);
    const scratch = createReadoutScratch(hiddenSize);
    const out = createReadoutOutput();

    readoutForward(weights, rate, indices, scratch, out);

    const hidden0 = Math.tanh(rate[4]);
    const hidden1 = Math.tanh(rate[1]);
    const hidden2 = Math.tanh(rate[3]);
    const expectedThrust = Math.tanh(hidden0);
    const expectedYaw = Math.tanh(hidden1);
    const expectedBrake = 1 / (1 + Math.exp(-hidden2));

    expect(out[0]).toBeCloseTo(expectedThrust, 7);
    expect(out[1]).toBeCloseTo(expectedYaw, 7);
    expect(out[2]).toBeCloseTo(expectedBrake, 7);
  });

  it('saturates to the exact tanh/sigmoid limits when the output bias dominates', () => {
    // w2 = 0 isolates the output layer from the (finite, non-saturating)
    // hidden values entirely, so b2's huge magnitude is what drives
    // saturation here — unlike the previous version of this test, where
    // brake never moved past sigmoid(±2), nowhere close to its [0, 1] ends.
    // Math.tanh(±1e6) and the sigmoid built from Math.exp(∓1e6) are exact
    // ±1/0/1 in double precision (exp over/underflows completely at that
    // magnitude), so these are exact equalities, not tolerances.
    const inputSize = 2;
    const hiddenSize = 2;
    const rate = Float32Array.from([1, 1]);
    const indices = Int32Array.from([0, 1]);
    const scratch = createReadoutScratch(hiddenSize);
    const out = createReadoutOutput();

    const weights = (b2: readonly [number, number, number]): ReadoutWeights => ({
      inputSize,
      hiddenSize,
      w1: Float32Array.from([1, 0, 0, 1]),
      b1: Float32Array.from([0, 0]),
      w2: Float32Array.from([0, 0, 0, 0, 0, 0]),
      b2: Float32Array.from(b2)
    });

    readoutForward(weights([1e6, -1e6, 1e6]), rate, indices, scratch, out);
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(-1);
    expect(out[2]).toBe(1);

    readoutForward(weights([-1e6, 1e6, -1e6]), rate, indices, scratch, out);
    expect(out[0]).toBe(-1);
    expect(out[1]).toBe(1);
    expect(out[2]).toBe(0);
  });
});

describe('validateReadoutWeights', () => {
  const graph = createTraceGraph();
  const inputSize = outputNeuronIndices(graph).length;
  const hiddenSize = 4;

  const validWeights = (): ReadoutWeights => ({
    inputSize,
    hiddenSize,
    w1: new Float32Array(hiddenSize * inputSize),
    b1: new Float32Array(hiddenSize),
    w2: new Float32Array(3 * hiddenSize),
    b2: new Float32Array(3)
  });

  it('accepts weights whose shape matches the graph and returns them', () => {
    const weights = validWeights();
    expect(validateReadoutWeights(weights, graph)).toBe(weights);
  });

  it('rejects a mismatched inputSize', () => {
    const weights = { ...validWeights(), inputSize: inputSize + 1 };
    expect(() => validateReadoutWeights(weights, graph)).toThrow(/inputSize/);
  });

  it('rejects inputSize 0 (a graph with no output-assigned neurons)', () => {
    // A hand-built zero-output-neuron graph, not the trace graph (whose
    // D = 6): validateReadoutWeights must reject even a self-consistent
    // (weights.inputSize === graph's own D) 0-input readout, since it
    // could never read anything from the network.
    const emptyGraph = createTraceGraph();
    // outputPopulationIndex all -1 gives outputNeuronIndices length 0. The
    // `readonly` modifier on the property only blocks reassigning it, not
    // mutating the Int32Array it points to.
    emptyGraph.outputPopulationIndex.fill(-1);
    const weights: ReadoutWeights = {
      inputSize: 0,
      hiddenSize,
      w1: new Float32Array(0),
      b1: new Float32Array(hiddenSize),
      w2: new Float32Array(3 * hiddenSize),
      b2: new Float32Array(3)
    };
    expect(() => validateReadoutWeights(weights, emptyGraph)).toThrow(/inputSize/);
  });

  it('rejects hiddenSize 0 and a non-integer hiddenSize', () => {
    expect(() =>
      validateReadoutWeights({ ...validWeights(), hiddenSize: 0 }, graph)
    ).toThrow(/hiddenSize/);
    expect(() =>
      validateReadoutWeights({ ...validWeights(), hiddenSize: 2.5 }, graph)
    ).toThrow(/hiddenSize/);
  });

  it('rejects a short w1', () => {
    const weights = { ...validWeights(), w1: new Float32Array(hiddenSize * inputSize - 1) };
    expect(() => validateReadoutWeights(weights, graph)).toThrow(/w1/);
  });

  it('rejects a wrong-length b1, w2, or b2', () => {
    expect(() =>
      validateReadoutWeights({ ...validWeights(), b1: new Float32Array(hiddenSize - 1) }, graph)
    ).toThrow(/b1/);
    expect(() =>
      validateReadoutWeights({ ...validWeights(), w2: new Float32Array(3 * hiddenSize - 1) }, graph)
    ).toThrow(/w2/);
    expect(() =>
      validateReadoutWeights({ ...validWeights(), b2: new Float32Array(2) }, graph)
    ).toThrow(/b2/);
  });

  it('rejects a NaN or infinite weight', () => {
    const withNaN = validWeights();
    withNaN.b2[1] = NaN;
    expect(() => validateReadoutWeights(withNaN, graph)).toThrow(/b2/);

    const withInfinity = validWeights();
    withInfinity.w1[0] = Infinity;
    expect(() => validateReadoutWeights(withInfinity, graph)).toThrow(/w1/);
  });
});

describe('createTraceGraph', () => {
  it('produces a graph that passes validateGraph', () => {
    // trace-graph.ts's own construction (CSR row canonicalization, sign/
    // channel/population ranges) is exercised end-to-end here; every other
    // test in this file assumes createTraceGraph() already returns a valid
    // graph without re-checking it.
    expect(() => validateGraph(createTraceGraph())).not.toThrow();
  });
});
