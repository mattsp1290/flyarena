import { beforeEach, describe, expect, it } from 'vitest';
import { encodeGraphBinary, parseGraphBinary, type ConnectomeGraph } from '../../src/lib/connectome/format';
import {
  aggregateOutputs,
  createModelState,
  createOutputBuffer,
  createStepScratch,
  resetModelState,
  runSubsteps,
  type NeuralModelState,
  type StepScratch
} from '../../src/lib/connectome/model';
import {
  createReadoutOutput,
  createReadoutScratch,
  outputNeuronIndices,
  readoutForward,
  type ReadoutWeights
} from '../../src/lib/connectome/readout';
import { computeTelemetry } from '../../src/lib/connectome/telemetry';
import {
  createWorkerRuntime,
  handleWorkerRequest
} from '../../src/lib/worker/neural.worker';
import { MAX_SUBSTEPS_PER_TICK, type WorkerRequest, type WorkerResponse } from '../../src/lib/worker/protocol';
import { createRandomGraph } from '../fixtures/tiny-graph';

/** A small, deterministic (fixed-seed PRNG) `ReadoutWeights` matching `graph`'s own `D`/`H`. */
const createTestReadoutWeights = (graph: ConnectomeGraph, hiddenSize: number, seed: number): ReadoutWeights => {
  const inputSize = outputNeuronIndices(graph).length;
  let state = seed >>> 0;
  const nextValue = (): number => {
    state = (state + 0x9e3779b9) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const unit = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    return unit * 2 - 1;
  };
  const fill = (length: number): Float32Array => Float32Array.from({ length }, () => nextValue() * 0.5);
  return {
    inputSize,
    hiddenSize,
    w1: fill(hiddenSize * inputSize),
    b1: fill(hiddenSize),
    w2: fill(3 * hiddenSize),
    b2: fill(3)
  };
};

/**
 * `handleWorkerRequest` returns `{ response, transfer? }` (its return type
 * grew a transfer list for streamed `rates.buffer` — see
 * `neural.worker.ts`); most call sites in this file only care about the
 * response, so this local wrapper keeps their shape unchanged. The
 * `set-activity`/streaming tests below that specifically care about the
 * transfer list call `handleWorkerRequest` directly instead.
 */
const call = (runtime: ReturnType<typeof createWorkerRuntime>, request: WorkerRequest): WorkerResponse =>
  handleWorkerRequest(runtime, request).response;

/**
 * Parity between the direct in-thread oracle and the Worker message handler.
 * The Worker is never instantiated here: per the work package, jsdom/node
 * cannot host a real browser Worker, so this exercises `handleWorkerRequest`
 * directly and simulates the postMessage transfer boundary with
 * `structuredClone` on every buffer/message that would actually cross it.
 */

const STEPS = 1000;
const SUBSTEPS_PER_TICK = 3;
const FLOAT_TOLERANCE = 1e-5;

/** Deterministic per-step observation generator, independent of the graph's own seed. */
const createChannelSequence = (seed: number): (() => number) => {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    const unit = ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    return unit * 2 - 1; // [-1, 1), matching the graph's inputClamp range
  };
};

const expectClose = (actual: number, expected: number, message: string): void => {
  expect(Math.abs(actual - expected), message).toBeLessThanOrEqual(FLOAT_TOLERANCE);
};

interface OracleHarness {
  graph: ConnectomeGraph;
  state: NeuralModelState;
  scratch: StepScratch;
  outputs: Float32Array;
}

const createOracle = (buffer: ArrayBuffer): OracleHarness => {
  const graph = parseGraphBinary(buffer);
  return {
    graph,
    state: createModelState(graph),
    scratch: createStepScratch(graph),
    outputs: createOutputBuffer(graph)
  };
};

describe('oracle vs Worker parity', () => {
  const sourceGraph = createRandomGraph(0x51a7e5, {
    neuronCount: 40,
    inputChannelCount: 8,
    outputPopulationCount: 3,
    edgeDensity: 3
  });
  const sourceBuffer = encodeGraphBinary(sourceGraph);

  it('matches action features and telemetry within tolerance over 1,000 seeded steps', () => {
    const oracle = createOracle(structuredClone(sourceBuffer));

    const runtime = createWorkerRuntime();
    const initResponse = call(runtime, {
      type: 'init',
      requestId: 'init-1',
      graphBuffer: structuredClone(sourceBuffer),
      mode: 'biological'
    });
    expect(initResponse.ok).toBe(true);

    const nextChannelValue = createChannelSequence(0xc0ffee);
    const maxAbsSeenByPopulation = new Array<number>(oracle.outputs.length).fill(0);

    for (let step = 0; step < STEPS; step += 1) {
      const channelValues = Array.from(
        { length: sourceGraph.metadata.inputChannelCount },
        () => nextChannelValue()
      );

      runSubsteps(
        oracle.graph,
        oracle.state,
        oracle.scratch,
        channelValues,
        SUBSTEPS_PER_TICK,
        oracle.outputs
      );
      const oracleTelemetry = computeTelemetry(oracle.state);

      // Clone both directions of the postMessage boundary: the request (the
      // main thread's outbound message) and the response (the Worker's
      // outbound message), matching what structured clone actually does to
      // plain objects/arrays in a real deployment.
      const response = structuredClone(
        call(runtime, {
          type: 'step',
          requestId: `step-${step}`,
          channelValues: structuredClone(channelValues),
          substeps: SUBSTEPS_PER_TICK
        })
      );
      if (!response.ok || response.type !== 'step') {
        throw new Error(`Worker step ${step} failed: ${JSON.stringify(response)}`);
      }

      for (let population = 0; population < oracle.outputs.length; population += 1) {
        expectClose(
          response.actionFeatures[population],
          oracle.outputs[population],
          `actionFeatures[${population}] at step ${step}`
        );
        maxAbsSeenByPopulation[population] = Math.max(
          maxAbsSeenByPopulation[population],
          Math.abs(oracle.outputs[population])
        );
      }
      expectClose(response.telemetry.meanRate, oracleTelemetry.meanRate, `meanRate at step ${step}`);
      expectClose(response.telemetry.minRate, oracleTelemetry.minRate, `minRate at step ${step}`);
      expectClose(response.telemetry.maxRate, oracleTelemetry.maxRate, `maxRate at step ${step}`);
      expectClose(
        response.telemetry.activeFraction,
        oracleTelemetry.activeFraction,
        `activeFraction at step ${step}`
      );
    }

    // A population that never moves would make its parity comparisons above
    // pass vacuously (0 === 0 on every step); require every output
    // population to actually activate at least once over the run.
    for (let population = 0; population < maxAbsSeenByPopulation.length; population += 1) {
      expect(
        maxAbsSeenByPopulation[population],
        `output population ${population} never activated during the run`
      ).toBeGreaterThan(1e-3);
    }
  });

  it('with activity enabled, rates matches the oracle state.rate every step, and actionFeatures is unchanged versus activity disabled', () => {
    const oracleEnabled = createOracle(structuredClone(sourceBuffer));
    const oracleDisabled = createOracle(structuredClone(sourceBuffer));
    const runtimeEnabled = createWorkerRuntime();
    const runtimeDisabled = createWorkerRuntime();
    call(runtimeEnabled, { type: 'init', requestId: 'init-enabled', graphBuffer: structuredClone(sourceBuffer) });
    call(runtimeDisabled, { type: 'init', requestId: 'init-disabled', graphBuffer: structuredClone(sourceBuffer) });

    const setActivity = handleWorkerRequest(runtimeEnabled, {
      type: 'set-activity',
      requestId: 'activity-on',
      enabled: true
    });
    expect(setActivity.response.ok).toBe(true);
    // No `rates` buffer to transfer for a `set-activity` response itself —
    // only a streaming `step` response ever populates `transfer`.
    expect(setActivity.transfer).toBeUndefined();

    const nextChannelValue = createChannelSequence(0x5ca1e);
    const TICKS = 50;

    for (let step = 0; step < TICKS; step += 1) {
      const channelValues = Array.from(
        { length: sourceGraph.metadata.inputChannelCount },
        () => nextChannelValue()
      );

      runSubsteps(oracleEnabled.graph, oracleEnabled.state, oracleEnabled.scratch, channelValues, SUBSTEPS_PER_TICK, oracleEnabled.outputs);
      runSubsteps(oracleDisabled.graph, oracleDisabled.state, oracleDisabled.scratch, channelValues, SUBSTEPS_PER_TICK, oracleDisabled.outputs);

      const enabledResult = handleWorkerRequest(runtimeEnabled, {
        type: 'step',
        requestId: `step-enabled-${step}`,
        channelValues: structuredClone(channelValues),
        substeps: SUBSTEPS_PER_TICK
      });
      const disabledResponse = call(runtimeDisabled, {
        type: 'step',
        requestId: `step-disabled-${step}`,
        channelValues: structuredClone(channelValues),
        substeps: SUBSTEPS_PER_TICK
      });

      const enabledResponse = enabledResult.response;
      if (!enabledResponse.ok || enabledResponse.type !== 'step') {
        throw new Error(`activity-enabled step ${step} failed: ${JSON.stringify(enabledResponse)}`);
      }
      if (!disabledResponse.ok || disabledResponse.type !== 'step') {
        throw new Error(`activity-disabled step ${step} failed: ${JSON.stringify(disabledResponse)}`);
      }

      expect(enabledResponse.rates).toBeDefined();
      expect(enabledResponse.rates).toHaveLength(sourceGraph.metadata.neuronCount);
      // `handleWorkerRequest` returns `rates` as a fresh copy, so this is a
      // true value-equality check, not an identity/aliasing accident.
      expect(Array.from(enabledResponse.rates!)).toEqual(Array.from(oracleEnabled.state.rate));
      // `rates.buffer` is exactly the one Transferable this response
      // carries — asserted by reference (`toBe`), not `toEqual`: `toEqual`
      // on an array containing an `ArrayBuffer` compares by byte content, so
      // it would also pass if `transfer[0]` were a *different* buffer
      // instance that merely holds identical bytes to `rates.buffer`,
      // rather than proving it is the exact same object a real
      // `postMessage(response, transfer)` call would detach out from under
      // the Worker (thermo-maintainability review S4 — this is the
      // invariant this assertion is actually meant to pin down).
      expect(enabledResult.transfer).toHaveLength(1);
      expect(enabledResult.transfer![0]).toBe(enabledResponse.rates!.buffer);

      // Disabled response carries no `rates` key at all — not merely an
      // `undefined` value — matching the closed-view contract.
      expect('rates' in disabledResponse).toBe(false);

      for (let population = 0; population < oracleEnabled.outputs.length; population += 1) {
        expectClose(
          enabledResponse.actionFeatures[population],
          disabledResponse.actionFeatures[population],
          `actionFeatures[${population}] differs between activity enabled/disabled at step ${step}`
        );
      }
    }
  });

  it('keeps oracle and Worker state in sync across a mid-run reset', () => {
    const oracle = createOracle(structuredClone(sourceBuffer));
    const runtime = createWorkerRuntime();
    call(runtime, {
      type: 'init',
      requestId: 'init-reset',
      graphBuffer: structuredClone(sourceBuffer)
    });

    const nextChannelValue = createChannelSequence(0xfeed);
    const stepOnce = (requestId: string): WorkerResponse => {
      const channelValues = Array.from(
        { length: sourceGraph.metadata.inputChannelCount },
        () => nextChannelValue()
      );
      runSubsteps(
        oracle.graph,
        oracle.state,
        oracle.scratch,
        channelValues,
        SUBSTEPS_PER_TICK,
        oracle.outputs
      );
      return call(runtime, {
        type: 'step',
        requestId,
        channelValues: structuredClone(channelValues),
        substeps: SUBSTEPS_PER_TICK
      });
    };

    for (let step = 0; step < 20; step += 1) stepOnce(`pre-reset-${step}`);
    expect(Array.from(oracle.state.rate).some((value) => value !== 0)).toBe(true);

    resetModelState(oracle.state);
    const resetResponse = call(runtime, { type: 'reset', requestId: 'reset-1' });
    expect(resetResponse.ok).toBe(true);

    const outputs = createOutputBuffer(oracle.graph);
    aggregateOutputs(oracle.graph, oracle.state, outputs);
    expect(Array.from(outputs).every((value) => value === 0)).toBe(true);

    for (let step = 0; step < 20; step += 1) {
      const response = stepOnce(`post-reset-${step}`);
      if (!response.ok || response.type !== 'step') throw new Error('post-reset step failed');
      for (let population = 0; population < oracle.outputs.length; population += 1) {
        expectClose(
          response.actionFeatures[population],
          oracle.outputs[population],
          `post-reset actionFeatures[${population}] at step ${step}`
        );
      }
    }
  });
});

/**
 * WP6's Worker-parity requirement: "with the tiny fixture readout, the
 * Worker's trained action features equal the direct `runSubsteps` +
 * `readoutForward` result for 50 ticks." Uses a freshly generated small
 * graph/readout pair (not literally `tests/fixtures/trained-readout-tiny.json`,
 * which is sized for `tests/fixtures/trace-graph.ts`'s D = 6 rather than this
 * file's own random graphs) — the parity property under test is identical
 * either way: the Worker's `readoutForward` call must exactly match one run
 * in-thread against the same `state.rate`.
 */
describe('trained decoder Worker parity', () => {
  const graph = createRandomGraph(0x7ead0, { neuronCount: 30, inputChannelCount: 6, outputPopulationCount: 3 });
  const buffer = encodeGraphBinary(graph);
  const weights = createTestReadoutWeights(graph, 5, 0xbeef);

  it("matches runSubsteps + readoutForward run directly against the oracle's own state.rate, over 50 ticks", () => {
    const oracle = createOracle(structuredClone(buffer));
    const indices = outputNeuronIndices(oracle.graph);
    const scratch = createReadoutScratch(weights.hiddenSize);
    const out = createReadoutOutput();

    const runtime = createWorkerRuntime();
    const initResponse = call(runtime, {
      type: 'init',
      requestId: 'init-trained',
      graphBuffer: structuredClone(buffer),
      readout: weights
    });
    expect(initResponse.ok).toBe(true);

    const setDecoderResponse = call(runtime, { type: 'set-decoder', requestId: 'sd-1', decoder: 'trained' });
    expect(setDecoderResponse.ok).toBe(true);
    if (setDecoderResponse.ok && setDecoderResponse.type === 'set-decoder') {
      expect(setDecoderResponse.decoder).toBe('trained');
    }

    const nextChannelValue = createChannelSequence(0xd0c0de);
    const TICKS = 50;
    const maxAbsSeen = [0, 0, 0];

    for (let step = 0; step < TICKS; step += 1) {
      const channelValues = Array.from(
        { length: graph.metadata.inputChannelCount },
        () => nextChannelValue()
      );

      runSubsteps(oracle.graph, oracle.state, oracle.scratch, channelValues, SUBSTEPS_PER_TICK, oracle.outputs);
      readoutForward(weights, oracle.state.rate, indices, scratch, out);

      const response = structuredClone(
        call(runtime, {
          type: 'step',
          requestId: `trained-step-${step}`,
          channelValues: structuredClone(channelValues),
          substeps: SUBSTEPS_PER_TICK
        })
      );
      if (!response.ok || response.type !== 'step') {
        throw new Error(`Worker trained step ${step} failed: ${JSON.stringify(response)}`);
      }

      expect(response.actionFeatures).toHaveLength(3);
      for (let output = 0; output < 3; output += 1) {
        expectClose(response.actionFeatures[output], out[output], `trained actionFeatures[${output}] at step ${step}`);
        maxAbsSeen[output] = Math.max(maxAbsSeen[output], Math.abs(out[output]));
      }
    }

    // Non-vacuity: every output must actually have varied, not sat at a
    // fixed activation value (e.g. tanh(0) = 0) for all 50 ticks.
    for (let output = 0; output < 3; output += 1) {
      expect(maxAbsSeen[output], `trained output ${output} never moved during the run`).toBeGreaterThan(1e-3);
    }
  });

  it('the trained decoder differs from the authored decoder on the same graph/inputs (proves it is not silently falling back to aggregateOutputs)', () => {
    const authoredRuntime = createWorkerRuntime();
    const trainedRuntime = createWorkerRuntime();
    call(authoredRuntime, { type: 'init', requestId: 'i-a', graphBuffer: structuredClone(buffer), readout: weights });
    call(trainedRuntime, { type: 'init', requestId: 'i-t', graphBuffer: structuredClone(buffer), readout: weights });
    call(trainedRuntime, { type: 'set-decoder', requestId: 'sd', decoder: 'trained' });

    const channelValues = Array.from({ length: graph.metadata.inputChannelCount }, (_, i) => (i % 2 === 0 ? 0.6 : -0.4));
    let authoredResponse: WorkerResponse | undefined;
    let trainedResponse: WorkerResponse | undefined;
    for (let step = 0; step < 20; step += 1) {
      authoredResponse = call(authoredRuntime, {
        type: 'step',
        requestId: `a-${step}`,
        channelValues: structuredClone(channelValues),
        substeps: SUBSTEPS_PER_TICK
      });
      trainedResponse = call(trainedRuntime, {
        type: 'step',
        requestId: `t-${step}`,
        channelValues: structuredClone(channelValues),
        substeps: SUBSTEPS_PER_TICK
      });
    }
    if (!authoredResponse?.ok || authoredResponse.type !== 'step' || !trainedResponse?.ok || trainedResponse.type !== 'step') {
      throw new Error('authored/trained comparison step failed');
    }
    expect(Array.from(trainedResponse.actionFeatures)).not.toEqual(Array.from(authoredResponse.actionFeatures));
  });
});

describe('Worker request validation', () => {
  let runtime: ReturnType<typeof createWorkerRuntime>;
  const graph = createRandomGraph(0x1, { neuronCount: 6, inputChannelCount: 2, outputPopulationCount: 1 });
  const buffer = encodeGraphBinary(graph);

  beforeEach(() => {
    runtime = createWorkerRuntime();
  });

  it('fails step/reset before init with not-initialized', () => {
    const stepResponse = call(runtime, {
      type: 'step',
      requestId: 's1',
      channelValues: [0, 0],
      substeps: 1
    });
    expect(stepResponse.ok).toBe(false);
    if (!stepResponse.ok) expect(stepResponse.error.code).toBe('not-initialized');

    const resetResponse = call(runtime, { type: 'reset', requestId: 'r1' });
    expect(resetResponse.ok).toBe(false);
    if (!resetResponse.ok) expect(resetResponse.error.code).toBe('not-initialized');
  });

  it('fails a second init with already-initialized', () => {
    call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const second = call(runtime, {
      type: 'init',
      requestId: 'i2',
      graphBuffer: structuredClone(buffer)
    });
    expect(second.ok).toBe(false);
    if (!second.ok) expect(second.error.code).toBe('already-initialized');
  });

  it('rejects a step with the wrong channelValues length', () => {
    call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const response = call(runtime, {
      type: 'step',
      requestId: 's1',
      channelValues: [0],
      substeps: 1
    });
    expect(response.ok).toBe(false);
    if (!response.ok) expect(response.error.code).toBe('invalid-request');
  });

  it('rejects a non-array channelValues', () => {
    call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const response = call(runtime, {
      type: 'step',
      requestId: 's1',
      // Simulates a caller/protocol-version bug: not the declared array type.
      channelValues: { length: 2, 0: 0, 1: 0 } as unknown as number[],
      substeps: 1
    });
    expect(response.ok).toBe(false);
    if (!response.ok) expect(response.error.code).toBe('invalid-request');
  });

  it('rejects a step whose channelValues contains a non-finite entry', () => {
    call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    for (const badValue of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      const response = call(runtime, {
        type: 'step',
        requestId: `s-${badValue}`,
        channelValues: [badValue, 0],
        substeps: 1
      });
      expect(response.ok, `channelValues containing ${badValue}`).toBe(false);
      if (!response.ok) expect(response.error.code).toBe('invalid-request');
    }

    // The rejected step must not have mutated network state: a clean step
    // afterward must produce a finite, non-NaN action feature.
    const clean = call(runtime, {
      type: 'step',
      requestId: 's-clean',
      channelValues: [0.1, 0.1],
      substeps: 1
    });
    expect(clean.ok).toBe(true);
    if (clean.ok && clean.type === 'step') {
      for (const value of clean.actionFeatures) expect(Number.isFinite(value)).toBe(true);
    }
  });

  it('never throws for a malformed message envelope and always returns a response', () => {
    for (const malformed of [null, undefined, 'not-an-object', 42, {}, { type: 'step' }]) {
      const response = call(runtime, malformed as never);
      expect(response.ok).toBe(false);
      if (!response.ok) expect(response.error.code).toBe('invalid-request');
    }
  });

  it('reports an unrecognized request type as unknown/invalid-request, echoing its requestId', () => {
    call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const response = call(runtime, {
      type: 'not-a-real-type',
      requestId: 'x1'
    } as unknown as never);
    expect(response.ok).toBe(false);
    if (!response.ok) {
      expect(response.type).toBe('unknown');
      expect(response.requestId).toBe('x1');
      expect(response.error.code).toBe('invalid-request');
    }
  });

  it('rejects a non-positive-integer substeps value', () => {
    call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const response = call(runtime, {
      type: 'step',
      requestId: 's1',
      channelValues: [0, 0],
      substeps: 0
    });
    expect(response.ok).toBe(false);
    if (!response.ok) expect(response.error.code).toBe('invalid-request');
  });

  it('returns an invalid-graph error instead of throwing for a malformed init buffer', () => {
    const malformed = structuredClone(buffer);
    new DataView(malformed).setUint32(4, 999, true); // bogus formatVersion

    const response = call(runtime, {
      type: 'init',
      requestId: 'i1',
      graphBuffer: malformed
    });

    expect(response.ok).toBe(false);
    if (!response.ok) expect(response.error.code).toBe('invalid-graph');
  });

  it('echoes InitWorkerRequest.mode back on InitWorkerSuccess', () => {
    const withMode = call(runtime, {
      type: 'init',
      requestId: 'i-mode',
      graphBuffer: structuredClone(buffer),
      mode: 'rewired'
    });
    expect(withMode.ok).toBe(true);
    if (withMode.ok && withMode.type === 'init') expect(withMode.mode).toBe('rewired');

    call(runtime, { type: 'dispose', requestId: 'd-mode' });

    const withoutMode = call(runtime, {
      type: 'init',
      requestId: 'i-no-mode',
      graphBuffer: structuredClone(buffer)
    });
    expect(withoutMode.ok).toBe(true);
    if (withoutMode.ok && withoutMode.type === 'init') expect(withoutMode.mode).toBeUndefined();
  });

  it('rejects a substeps value above MAX_SUBSTEPS_PER_TICK', () => {
    call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const response = call(runtime, {
      type: 'step',
      requestId: 's1',
      channelValues: [0, 0],
      substeps: MAX_SUBSTEPS_PER_TICK + 1
    });
    expect(response.ok).toBe(false);
    if (!response.ok) expect(response.error.code).toBe('invalid-request');

    // The bound itself must still be accepted.
    const atBound = call(runtime, {
      type: 'step',
      requestId: 's2',
      channelValues: [0, 0],
      substeps: MAX_SUBSTEPS_PER_TICK
    });
    expect(atBound.ok).toBe(true);
  });

  it('allows re-initializing after dispose', () => {
    call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const disposeResponse = call(runtime, { type: 'dispose', requestId: 'd1' });
    expect(disposeResponse.ok).toBe(true);

    const reInit = call(runtime, {
      type: 'init',
      requestId: 'i2',
      graphBuffer: structuredClone(buffer)
    });
    expect(reInit.ok).toBe(true);
  });

  it('fails set-activity before init with not-initialized', () => {
    const response = call(runtime, { type: 'set-activity', requestId: 'sa1', enabled: true });
    expect(response.ok).toBe(false);
    if (!response.ok) expect(response.error.code).toBe('not-initialized');
  });

  it('rejects a malformed (non-boolean) set-activity enabled value', () => {
    call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const response = call(runtime, {
      type: 'set-activity',
      requestId: 'sa1',
      enabled: 'yes' as unknown as boolean
    });
    expect(response.ok).toBe(false);
    if (!response.ok) expect(response.error.code).toBe('invalid-request');
  });

  it('echoes enabled on success and toggles whether step responses carry rates', () => {
    call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });

    const disabledStep = call(runtime, { type: 'step', requestId: 's1', channelValues: [0, 0], substeps: 1 });
    expect(disabledStep.ok).toBe(true);
    expect('rates' in disabledStep).toBe(false);

    const enableResponse = call(runtime, { type: 'set-activity', requestId: 'sa-on', enabled: true });
    expect(enableResponse.ok).toBe(true);
    if (enableResponse.ok && enableResponse.type === 'set-activity') expect(enableResponse.enabled).toBe(true);

    const enabledStep = call(runtime, { type: 'step', requestId: 's2', channelValues: [0, 0], substeps: 1 });
    expect(enabledStep.ok).toBe(true);
    if (enabledStep.ok && enabledStep.type === 'step') {
      expect(enabledStep.rates).toBeInstanceOf(Float32Array);
      expect(enabledStep.rates).toHaveLength(graph.metadata.neuronCount);
    }

    const disableResponse = call(runtime, { type: 'set-activity', requestId: 'sa-off', enabled: false });
    expect(disableResponse.ok).toBe(true);
    if (disableResponse.ok && disableResponse.type === 'set-activity') expect(disableResponse.enabled).toBe(false);

    const redisabledStep = call(runtime, { type: 'step', requestId: 's3', channelValues: [0, 0], substeps: 1 });
    expect('rates' in redisabledStep).toBe(false);
  });

  it('resets the activity flag to false on a fresh init after dispose, even if it was enabled before dispose', () => {
    call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    call(runtime, { type: 'set-activity', requestId: 'sa-on', enabled: true });
    call(runtime, { type: 'dispose', requestId: 'd1' });
    call(runtime, { type: 'init', requestId: 'i2', graphBuffer: structuredClone(buffer) });

    const response = call(runtime, { type: 'step', requestId: 's1', channelValues: [0, 0], substeps: 1 });
    expect('rates' in response).toBe(false);
  });

  describe('readout / set-decoder', () => {
    const readoutWeights = createTestReadoutWeights(graph, 4, 0xf00d);

    it('defaults to the authored decoder immediately after init, even when readout was supplied', () => {
      call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer), readout: readoutWeights });
      const authoredStep = call(runtime, { type: 'step', requestId: 's1', channelValues: [0.4, -0.2], substeps: 1 });
      expect(authoredStep.ok).toBe(true);

      const outputs = createOutputBuffer(graph);
      const oracleState = createModelState(graph);
      const oracleScratch = createStepScratch(graph);
      runSubsteps(graph, oracleState, oracleScratch, [0.4, -0.2], 1, outputs);
      if (authoredStep.ok && authoredStep.type === 'step') {
        for (let population = 0; population < outputs.length; population += 1) {
          expectClose(authoredStep.actionFeatures[population], outputs[population], `default-decoder actionFeatures[${population}]`);
        }
      }
    });

    it('rejects set-decoder before init with not-initialized', () => {
      const response = call(runtime, { type: 'set-decoder', requestId: 'sd1', decoder: 'authored' });
      expect(response.ok).toBe(false);
      if (!response.ok) expect(response.error.code).toBe('not-initialized');
    });

    it('rejects set-decoder: "trained" with invalid-request when init was given no readout', () => {
      call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
      const response = call(runtime, { type: 'set-decoder', requestId: 'sd1', decoder: 'trained' });
      expect(response.ok).toBe(false);
      if (!response.ok) {
        expect(response.error.code).toBe('invalid-request');
        expect(response.error.message).toMatch(/readout/i);
      }

      // Authored keeps working after the rejected switch — the Worker
      // stays on its default decoder rather than landing in some broken
      // in-between state.
      const stepResponse = call(runtime, { type: 'step', requestId: 's1', channelValues: [0, 0], substeps: 1 });
      expect(stepResponse.ok).toBe(true);
    });

    it('rejects an unrecognized decoder value with invalid-request', () => {
      call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer), readout: readoutWeights });
      const response = call(runtime, {
        type: 'set-decoder',
        requestId: 'sd1',
        decoder: 'not-a-real-decoder' as never
      });
      expect(response.ok).toBe(false);
      if (!response.ok) expect(response.error.code).toBe('invalid-request');
    });

    it('accepts and echoes set-decoder: "trained" when init was given a matching readout, and set-decoder: "authored" switches back', () => {
      call(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer), readout: readoutWeights });

      const toTrained = call(runtime, { type: 'set-decoder', requestId: 'sd1', decoder: 'trained' });
      expect(toTrained.ok).toBe(true);
      if (toTrained.ok && toTrained.type === 'set-decoder') expect(toTrained.decoder).toBe('trained');

      const toAuthored = call(runtime, { type: 'set-decoder', requestId: 'sd2', decoder: 'authored' });
      expect(toAuthored.ok).toBe(true);
      if (toAuthored.ok && toAuthored.type === 'set-decoder') expect(toAuthored.decoder).toBe('authored');
    });

    it('rejects init with invalid-request (not invalid-graph) when readout.inputSize does not match the graph\'s output-neuron count', () => {
      const mismatched: ReadoutWeights = { ...readoutWeights, inputSize: readoutWeights.inputSize + 1 };
      const response = call(runtime, {
        type: 'init',
        requestId: 'i1',
        graphBuffer: structuredClone(buffer),
        readout: mismatched
      });
      expect(response.ok).toBe(false);
      if (!response.ok) {
        expect(response.error.code).toBe('invalid-request');
        expect(response.error.message).toMatch(/inputSize/);
      }

      // The graph buffer itself was fine, so init must still be retryable
      // (idle, not stuck) after the rejection.
      const retry = call(runtime, {
        type: 'init',
        requestId: 'i2',
        graphBuffer: structuredClone(buffer)
      });
      expect(retry.ok).toBe(true);
    });

    it('rejects init with invalid-request when a readout weight is non-finite', () => {
      const withNaN: ReadoutWeights = { ...readoutWeights, b2: Float32Array.from([NaN, 0, 0]) };
      const response = call(runtime, {
        type: 'init',
        requestId: 'i1',
        graphBuffer: structuredClone(buffer),
        readout: withNaN
      });
      expect(response.ok).toBe(false);
      if (!response.ok) expect(response.error.code).toBe('invalid-request');
    });
  });
});
