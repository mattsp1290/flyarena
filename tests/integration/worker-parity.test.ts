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
import { computeTelemetry } from '../../src/lib/connectome/telemetry';
import {
  createWorkerRuntime,
  handleWorkerRequest
} from '../../src/lib/worker/neural.worker';
import { MAX_SUBSTEPS_PER_TICK, type WorkerResponse } from '../../src/lib/worker/protocol';
import { createRandomGraph } from '../fixtures/tiny-graph';

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
    const initResponse = handleWorkerRequest(runtime, {
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
        handleWorkerRequest(runtime, {
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

  it('keeps oracle and Worker state in sync across a mid-run reset', () => {
    const oracle = createOracle(structuredClone(sourceBuffer));
    const runtime = createWorkerRuntime();
    handleWorkerRequest(runtime, {
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
      return handleWorkerRequest(runtime, {
        type: 'step',
        requestId,
        channelValues: structuredClone(channelValues),
        substeps: SUBSTEPS_PER_TICK
      });
    };

    for (let step = 0; step < 20; step += 1) stepOnce(`pre-reset-${step}`);
    expect(Array.from(oracle.state.rate).some((value) => value !== 0)).toBe(true);

    resetModelState(oracle.state);
    const resetResponse = handleWorkerRequest(runtime, { type: 'reset', requestId: 'reset-1' });
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

describe('Worker request validation', () => {
  let runtime: ReturnType<typeof createWorkerRuntime>;
  const graph = createRandomGraph(0x1, { neuronCount: 6, inputChannelCount: 2, outputPopulationCount: 1 });
  const buffer = encodeGraphBinary(graph);

  beforeEach(() => {
    runtime = createWorkerRuntime();
  });

  it('fails step/reset before init with not-initialized', () => {
    const stepResponse = handleWorkerRequest(runtime, {
      type: 'step',
      requestId: 's1',
      channelValues: [0, 0],
      substeps: 1
    });
    expect(stepResponse.ok).toBe(false);
    if (!stepResponse.ok) expect(stepResponse.error.code).toBe('not-initialized');

    const resetResponse = handleWorkerRequest(runtime, { type: 'reset', requestId: 'r1' });
    expect(resetResponse.ok).toBe(false);
    if (!resetResponse.ok) expect(resetResponse.error.code).toBe('not-initialized');
  });

  it('fails a second init with already-initialized', () => {
    handleWorkerRequest(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const second = handleWorkerRequest(runtime, {
      type: 'init',
      requestId: 'i2',
      graphBuffer: structuredClone(buffer)
    });
    expect(second.ok).toBe(false);
    if (!second.ok) expect(second.error.code).toBe('already-initialized');
  });

  it('rejects a step with the wrong channelValues length', () => {
    handleWorkerRequest(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const response = handleWorkerRequest(runtime, {
      type: 'step',
      requestId: 's1',
      channelValues: [0],
      substeps: 1
    });
    expect(response.ok).toBe(false);
    if (!response.ok) expect(response.error.code).toBe('invalid-request');
  });

  it('rejects a non-array channelValues', () => {
    handleWorkerRequest(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const response = handleWorkerRequest(runtime, {
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
    handleWorkerRequest(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    for (const badValue of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
      const response = handleWorkerRequest(runtime, {
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
    const clean = handleWorkerRequest(runtime, {
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
      const response = handleWorkerRequest(runtime, malformed as never);
      expect(response.ok).toBe(false);
      if (!response.ok) expect(response.error.code).toBe('invalid-request');
    }
  });

  it('reports an unrecognized request type as unknown/invalid-request, echoing its requestId', () => {
    handleWorkerRequest(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const response = handleWorkerRequest(runtime, {
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
    handleWorkerRequest(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const response = handleWorkerRequest(runtime, {
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

    const response = handleWorkerRequest(runtime, {
      type: 'init',
      requestId: 'i1',
      graphBuffer: malformed
    });

    expect(response.ok).toBe(false);
    if (!response.ok) expect(response.error.code).toBe('invalid-graph');
  });

  it('echoes InitWorkerRequest.mode back on InitWorkerSuccess', () => {
    const withMode = handleWorkerRequest(runtime, {
      type: 'init',
      requestId: 'i-mode',
      graphBuffer: structuredClone(buffer),
      mode: 'rewired'
    });
    expect(withMode.ok).toBe(true);
    if (withMode.ok && withMode.type === 'init') expect(withMode.mode).toBe('rewired');

    handleWorkerRequest(runtime, { type: 'dispose', requestId: 'd-mode' });

    const withoutMode = handleWorkerRequest(runtime, {
      type: 'init',
      requestId: 'i-no-mode',
      graphBuffer: structuredClone(buffer)
    });
    expect(withoutMode.ok).toBe(true);
    if (withoutMode.ok && withoutMode.type === 'init') expect(withoutMode.mode).toBeUndefined();
  });

  it('rejects a substeps value above MAX_SUBSTEPS_PER_TICK', () => {
    handleWorkerRequest(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const response = handleWorkerRequest(runtime, {
      type: 'step',
      requestId: 's1',
      channelValues: [0, 0],
      substeps: MAX_SUBSTEPS_PER_TICK + 1
    });
    expect(response.ok).toBe(false);
    if (!response.ok) expect(response.error.code).toBe('invalid-request');

    // The bound itself must still be accepted.
    const atBound = handleWorkerRequest(runtime, {
      type: 'step',
      requestId: 's2',
      channelValues: [0, 0],
      substeps: MAX_SUBSTEPS_PER_TICK
    });
    expect(atBound.ok).toBe(true);
  });

  it('allows re-initializing after dispose', () => {
    handleWorkerRequest(runtime, { type: 'init', requestId: 'i1', graphBuffer: structuredClone(buffer) });
    const disposeResponse = handleWorkerRequest(runtime, { type: 'dispose', requestId: 'd1' });
    expect(disposeResponse.ok).toBe(true);

    const reInit = handleWorkerRequest(runtime, {
      type: 'init',
      requestId: 'i2',
      graphBuffer: structuredClone(buffer)
    });
    expect(reInit.ok).toBe(true);
  });
});
