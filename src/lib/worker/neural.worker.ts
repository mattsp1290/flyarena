import { parseGraphBinary, type ConnectomeGraph, type GraphMode } from '../connectome/format';
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
import { MAX_SUBSTEPS_PER_TICK } from './protocol';
import type { WorkerErrorCode, WorkerFailure, WorkerRequest, WorkerResponse } from './protocol';

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

/**
 * The Worker's own state, as a tagged union: either not yet initialized
 * (`idle`) or holding a fully-initialized graph/state/scratch/outputs quad
 * (`ready`). The four `ready` fields are never independently present —
 * `init` produces all four together, `dispose` clears all four together,
 * and every other handler only ever needs to know "is the runtime ready,"
 * not which of four fields happen to be set. A tagged union makes that
 * atomicity a type-level fact instead of a convention enforced only by a
 * four-way `&&`/`||` check that a future added field could be left out of.
 */
type RuntimeState =
  | { status: 'idle' }
  | {
      status: 'ready';
      graph: ConnectomeGraph;
      state: NeuralModelState;
      scratch: StepScratch;
      outputs: Float32Array;
      /** Echoes the `InitWorkerRequest.mode` the graph was initialized with, if any. */
      mode?: GraphMode;
    };

/**
 * Mutable box holding the current `RuntimeState`. `handleWorkerRequest`
 * reassigns `current` wholesale on every state transition (rather than
 * mutating fields in place), so a transition can never leave a
 * partially-updated mix of old and new fields; the box itself is what
 * callers hold onto so the same identity keeps reflecting the latest state
 * across successive calls.
 */
interface WorkerRuntime {
  current: RuntimeState;
}

export const createWorkerRuntime = (): WorkerRuntime => ({ current: { status: 'idle' } });

const failure = (
  type: WorkerFailure['type'],
  requestId: string,
  code: WorkerFailure['error']['code'],
  message: string
): WorkerFailure => ({ type, requestId, ok: false, error: { code, message } });

/** True for a plausible request envelope: an object with a string `requestId`. */
const isRequestEnvelope = (value: unknown): value is { type: unknown; requestId: string } =>
  typeof value === 'object' &&
  value !== null &&
  typeof (value as { requestId?: unknown }).requestId === 'string';

/** True for a plain array or any typed-array view (but not a `DataView`) of numbers. */
const isNumericArrayLike = (value: unknown): value is ArrayLike<number> =>
  Array.isArray(value) || (ArrayBuffer.isView(value) && !(value instanceof DataView));

/** Handle one request against `runtime`, mutating it in place as needed. */
export const handleWorkerRequest = (
  runtime: WorkerRuntime,
  request: WorkerRequest
): WorkerResponse => {
  // `request` crosses a structured-clone boundary in production, so it is
  // not actually guaranteed to match `WorkerRequest` at runtime. Reject a
  // malformed envelope before touching `request.type` so this function can
  // never throw: a throw here would leave the caller's `requestId` with no
  // matching response at all.
  if (!isRequestEnvelope(request)) {
    return failure('unknown', '', 'invalid-request', 'Worker message must be an object with a string requestId');
  }

  try {
    switch (request.type) {
      case 'init': {
        if (runtime.current.status === 'ready') {
          return failure(
            'init',
            request.requestId,
            'already-initialized',
            'Neural worker is already initialized; dispose it before re-initializing'
          );
        }
        const graph = parseGraphBinary(request.graphBuffer);
        runtime.current = {
          status: 'ready',
          graph,
          state: createModelState(graph),
          scratch: createStepScratch(graph),
          outputs: createOutputBuffer(graph),
          mode: request.mode
        };
        return {
          type: 'init',
          requestId: request.requestId,
          ok: true,
          neuronCount: graph.metadata.neuronCount,
          edgeCount: graph.metadata.edgeCount,
          inputChannelCount: graph.metadata.inputChannelCount,
          outputPopulationCount: graph.metadata.outputPopulationCount,
          mode: request.mode
        };
      }

      case 'reset': {
        if (runtime.current.status !== 'ready') {
          return failure(
            'reset',
            request.requestId,
            'not-initialized',
            'Neural worker has not been initialized'
          );
        }
        resetModelState(runtime.current.state);
        return { type: 'reset', requestId: request.requestId, ok: true };
      }

      case 'step': {
        const stepFailure = (code: WorkerErrorCode, message: string): WorkerFailure =>
          failure('step', request.requestId, code, message);

        if (runtime.current.status !== 'ready') {
          return stepFailure('not-initialized', 'Neural worker has not been initialized');
        }
        const { graph, state, scratch, outputs } = runtime.current;

        if (!Number.isInteger(request.substeps) || request.substeps <= 0) {
          return stepFailure('invalid-request', 'substeps must be a positive integer');
        }
        if (request.substeps > MAX_SUBSTEPS_PER_TICK) {
          return stepFailure(
            'invalid-request',
            `substeps must not exceed MAX_SUBSTEPS_PER_TICK (${MAX_SUBSTEPS_PER_TICK}), received ${request.substeps}`
          );
        }
        if (!isNumericArrayLike(request.channelValues)) {
          return stepFailure('invalid-request', 'channelValues must be an array or typed array');
        }
        if (request.channelValues.length !== graph.metadata.inputChannelCount) {
          return stepFailure(
            'invalid-request',
            `channelValues must have length ${graph.metadata.inputChannelCount}, received ${request.channelValues.length}`
          );
        }
        for (let channel = 0; channel < request.channelValues.length; channel += 1) {
          if (!Number.isFinite(request.channelValues[channel])) {
            return stepFailure(
              'invalid-request',
              `channelValues[${channel}] must be a finite number, received ${String(request.channelValues[channel])}`
            );
          }
        }
        runSubsteps(graph, state, scratch, request.channelValues, request.substeps, outputs);
        return {
          type: 'step',
          requestId: request.requestId,
          ok: true,
          actionFeatures: Array.from(outputs),
          telemetry: computeTelemetry(state)
        };
      }

      case 'dispose': {
        runtime.current = { status: 'idle' };
        return { type: 'dispose', requestId: request.requestId, ok: true };
      }

      default: {
        const unreachable: never = request;
        return failure(
          'unknown',
          (unreachable as WorkerRequest).requestId,
          'invalid-request',
          `Unknown worker request type: ${String((unreachable as WorkerRequest).type)}`
        );
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // `init` failures are almost always a malformed graph buffer; every other
    // request type only reaches `catch` for an unexpected runtime bug, since
    // its own inputs are validated above before touching the model.
    const code = request.type === 'init' ? 'invalid-graph' : 'internal-error';
    return failure(request.type, request.requestId, code, message);
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
