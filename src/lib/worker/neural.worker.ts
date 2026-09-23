import { parseGraphBinary, type ConnectomeGraph } from '../connectome/format';
import {
  createModelState,
  createOutputBuffer,
  createStepScratch,
  resetModelState,
  runSubsteps,
  type NeuralModelState,
  type StepScratch
} from '../connectome/model';
import { computeTelemetry } from '../connectome/telemetry';
import type { WorkerFailure, WorkerRequest, WorkerResponse } from './protocol';

/**
 * The dedicated Web Worker for stepping the sparse connectome oracle off the
 * main thread. All dynamics live in `../connectome/model`; this module only
 * owns request/response plumbing so the Worker and the direct in-thread
 * oracle can never disagree about behavior (see `tests/integration/worker-parity.test.ts`).
 *
 * `handleWorkerRequest` is a pure function of (runtime, request) -> response
 * with no dependency on `self`/`postMessage`, so tests exercise it directly
 * instead of relying on a real browser Worker.
 */

interface WorkerRuntime {
  graph?: ConnectomeGraph;
  state?: NeuralModelState;
  scratch?: StepScratch;
  outputs?: Float32Array;
}

export const createWorkerRuntime = (): WorkerRuntime => ({});

const failure = (
  type: WorkerRequest['type'],
  requestId: string,
  code: WorkerFailure['error']['code'],
  message: string
): WorkerFailure => ({ type, requestId, ok: false, error: { code, message } });

/** Handle one request against `runtime`, mutating it in place as needed. */
export const handleWorkerRequest = (
  runtime: WorkerRuntime,
  request: WorkerRequest
): WorkerResponse => {
  try {
    switch (request.type) {
      case 'init': {
        if (runtime.graph) {
          return failure(
            'init',
            request.requestId,
            'already-initialized',
            'Neural worker is already initialized; dispose it before re-initializing'
          );
        }
        const graph = parseGraphBinary(request.graphBuffer);
        runtime.graph = graph;
        runtime.state = createModelState(graph);
        runtime.scratch = createStepScratch(graph);
        runtime.outputs = createOutputBuffer(graph);
        return {
          type: 'init',
          requestId: request.requestId,
          ok: true,
          neuronCount: graph.metadata.neuronCount,
          edgeCount: graph.metadata.edgeCount,
          inputChannelCount: graph.metadata.inputChannelCount,
          outputPopulationCount: graph.metadata.outputPopulationCount
        };
      }

      case 'reset': {
        if (!runtime.state) {
          return failure(
            'reset',
            request.requestId,
            'not-initialized',
            'Neural worker has not been initialized'
          );
        }
        resetModelState(runtime.state);
        return { type: 'reset', requestId: request.requestId, ok: true };
      }

      case 'step': {
        if (!runtime.graph || !runtime.state || !runtime.scratch || !runtime.outputs) {
          return failure(
            'step',
            request.requestId,
            'not-initialized',
            'Neural worker has not been initialized'
          );
        }
        if (!Number.isInteger(request.substeps) || request.substeps <= 0) {
          return failure(
            'step',
            request.requestId,
            'invalid-request',
            'substeps must be a positive integer'
          );
        }
        if (request.channelValues.length !== runtime.graph.metadata.inputChannelCount) {
          return failure(
            'step',
            request.requestId,
            'invalid-request',
            `channelValues must have length ${runtime.graph.metadata.inputChannelCount}, received ${request.channelValues.length}`
          );
        }
        runSubsteps(
          runtime.graph,
          runtime.state,
          runtime.scratch,
          request.channelValues,
          request.substeps,
          runtime.outputs
        );
        return {
          type: 'step',
          requestId: request.requestId,
          ok: true,
          actionFeatures: Array.from(runtime.outputs),
          telemetry: computeTelemetry(runtime.state)
        };
      }

      case 'dispose': {
        runtime.graph = undefined;
        runtime.state = undefined;
        runtime.scratch = undefined;
        runtime.outputs = undefined;
        return { type: 'dispose', requestId: request.requestId, ok: true };
      }

      default: {
        const unreachable: never = request;
        return failure(
          (unreachable as WorkerRequest).type,
          (unreachable as WorkerRequest).requestId,
          'invalid-request',
          'Unknown worker request type'
        );
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return failure(request.type, request.requestId, 'invalid-graph', message);
  }
};

/** Minimal dedicated-worker-scope surface this module wires itself onto. */
interface NeuralWorkerScope {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage: (message: WorkerResponse) => void;
}

/**
 * True only inside an actual dedicated Worker (no `window`, but `self` and
 * `postMessage` exist). Guards auto-wiring so importing this module under
 * Vitest's jsdom environment (which defines `window`) never attaches a
 * `self.onmessage` handler; tests call `handleWorkerRequest` directly.
 */
const isDedicatedWorkerScope =
  typeof window === 'undefined' &&
  typeof self !== 'undefined' &&
  typeof (self as { postMessage?: unknown }).postMessage === 'function';

if (isDedicatedWorkerScope) {
  const workerScope = self as unknown as NeuralWorkerScope;
  const runtime = createWorkerRuntime();
  workerScope.onmessage = (event: MessageEvent<WorkerRequest>) => {
    workerScope.postMessage(handleWorkerRequest(runtime, event.data));
  };
}
