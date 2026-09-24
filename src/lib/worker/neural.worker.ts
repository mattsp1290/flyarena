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
 * (`idle`) or holding a fully-initialized graph/state/scratch/outputs/
 * activity quintet (`ready`). The five `ready` fields are never
 * independently present — `init` produces all five together (`activity`
 * always starting `false`), `dispose` clears all five together, and every
 * other handler only ever needs to know "is the runtime ready," not which
 * of five fields happen to be set. A tagged union makes that atomicity a
 * type-level fact instead of a convention enforced only by a five-way
 * `&&`/`||` check that a future added field could be left out of.
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
      /**
       * Whether `step` responses currently include the full per-neuron
       * `rates` vector (`StepWorkerSuccess.rates`), toggled by
       * `set-activity`. Always `false` immediately after `init` — a Worker
       * never streams full-neuron state unless a caller explicitly opts in
       * for *this* runtime instance (a fresh `init` after `dispose` starts
       * opted out again, even if a caller streamed before the prior
       * `dispose`).
       */
      activity: boolean;
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

/**
 * `handleWorkerRequest`'s return shape: the response to send back, plus an
 * optional transfer list for whatever `Transferable`s that response embeds
 * (currently only `StepWorkerSuccess.rates.buffer`, when activity streaming
 * is on). Kept as a plain data pair — rather than this function calling
 * `postMessage` itself — so it stays a pure `(runtime, request) -> result`
 * function callable directly from tests with no `self`/`postMessage` at all
 * (see `tests/integration/worker-parity.test.ts`); the one production caller
 * (the `isDedicatedWorkerScope` wiring below) is the only place that turns
 * this into an actual `postMessage` call.
 */
export interface HandleWorkerRequestResult {
  response: WorkerResponse;
  transfer?: Transferable[];
}

/** Wrap a bare `WorkerResponse` with no transfer list — the common case for every response but a streaming `step`. */
const respond = (response: WorkerResponse): HandleWorkerRequestResult => ({ response });

/** Handle one request against `runtime`, mutating it in place as needed. */
export const handleWorkerRequest = (
  runtime: WorkerRuntime,
  request: WorkerRequest
): HandleWorkerRequestResult => {
  // `request` crosses a structured-clone boundary in production, so it is
  // not actually guaranteed to match `WorkerRequest` at runtime. Reject a
  // malformed envelope before touching `request.type` so this function can
  // never throw: a throw here would leave the caller's `requestId` with no
  // matching response at all.
  if (!isRequestEnvelope(request)) {
    return respond(
      failure('unknown', '', 'invalid-request', 'Worker message must be an object with a string requestId')
    );
  }

  try {
    switch (request.type) {
      case 'init': {
        if (runtime.current.status === 'ready') {
          return respond(
            failure(
              'init',
              request.requestId,
              'already-initialized',
              'Neural worker is already initialized; dispose it before re-initializing'
            )
          );
        }
        const graph = parseGraphBinary(request.graphBuffer);
        runtime.current = {
          status: 'ready',
          graph,
          state: createModelState(graph),
          scratch: createStepScratch(graph),
          outputs: createOutputBuffer(graph),
          mode: request.mode,
          activity: false
        };
        return respond({
          type: 'init',
          requestId: request.requestId,
          ok: true,
          neuronCount: graph.metadata.neuronCount,
          edgeCount: graph.metadata.edgeCount,
          inputChannelCount: graph.metadata.inputChannelCount,
          outputPopulationCount: graph.metadata.outputPopulationCount,
          mode: request.mode
        });
      }

      case 'reset': {
        if (runtime.current.status !== 'ready') {
          return respond(
            failure('reset', request.requestId, 'not-initialized', 'Neural worker has not been initialized')
          );
        }
        resetModelState(runtime.current.state);
        return respond({ type: 'reset', requestId: request.requestId, ok: true });
      }

      case 'step': {
        const stepFailure = (code: WorkerErrorCode, message: string): WorkerFailure =>
          failure('step', request.requestId, code, message);

        if (runtime.current.status !== 'ready') {
          return respond(stepFailure('not-initialized', 'Neural worker has not been initialized'));
        }
        const { graph, state, scratch, outputs, activity } = runtime.current;

        if (!Number.isInteger(request.substeps) || request.substeps <= 0) {
          return respond(stepFailure('invalid-request', 'substeps must be a positive integer'));
        }
        if (request.substeps > MAX_SUBSTEPS_PER_TICK) {
          return respond(
            stepFailure(
              'invalid-request',
              `substeps must not exceed MAX_SUBSTEPS_PER_TICK (${MAX_SUBSTEPS_PER_TICK}), received ${request.substeps}`
            )
          );
        }
        if (!isNumericArrayLike(request.channelValues)) {
          return respond(stepFailure('invalid-request', 'channelValues must be an array or typed array'));
        }
        if (request.channelValues.length !== graph.metadata.inputChannelCount) {
          return respond(
            stepFailure(
              'invalid-request',
              `channelValues must have length ${graph.metadata.inputChannelCount}, received ${request.channelValues.length}`
            )
          );
        }
        for (let channel = 0; channel < request.channelValues.length; channel += 1) {
          if (!Number.isFinite(request.channelValues[channel])) {
            return respond(
              stepFailure(
                'invalid-request',
                `channelValues[${channel}] must be a finite number, received ${String(request.channelValues[channel])}`
              )
            );
          }
        }
        runSubsteps(graph, state, scratch, request.channelValues, request.substeps, outputs);
        // A fresh copy, not a view onto `state.rate`: `state.rate` is the
        // Worker's own long-lived, reused-every-substep buffer (see
        // `connectome/model.ts`'s allocation-free-substep-loop contract), so
        // handing a view of it across `postMessage` and then transferring
        // its `ArrayBuffer` would detach the Worker's own working buffer out
        // from under it. `.slice()` allocates exactly one `neuronCount`-length
        // `Float32Array` per streaming tick (~4 KB for 1,008 neurons) — the
        // one allocation this path adds, and only while a caller has opted
        // in via `set-activity`; see `StepWorkerSuccess.rates`'s doc comment.
        const rates = activity ? state.rate.slice() : undefined;
        const response: WorkerResponse = {
          type: 'step',
          requestId: request.requestId,
          ok: true,
          actionFeatures: Array.from(outputs),
          telemetry: computeTelemetry(state),
          ...(rates ? { rates } : {})
        };
        // `respond()` (no `transfer` key at all) when not streaming, rather
        // than `{ response, transfer: undefined }` — every other branch of
        // this function goes through `respond()`, so a non-streaming `step`
        // result has the exact same shape (`'transfer' in result === false`)
        // as any other response, not merely an `undefined`-valued key.
        return rates ? { response, transfer: [rates.buffer] } : respond(response);
      }

      case 'set-activity': {
        if (runtime.current.status !== 'ready') {
          return respond(
            failure('set-activity', request.requestId, 'not-initialized', 'Neural worker has not been initialized')
          );
        }
        if (typeof request.enabled !== 'boolean') {
          return respond(
            failure('set-activity', request.requestId, 'invalid-request', 'enabled must be a boolean')
          );
        }
        runtime.current = { ...runtime.current, activity: request.enabled };
        return respond({
          type: 'set-activity',
          requestId: request.requestId,
          ok: true,
          enabled: request.enabled
        });
      }

      case 'dispose': {
        runtime.current = { status: 'idle' };
        return respond({ type: 'dispose', requestId: request.requestId, ok: true });
      }

      default: {
        const unreachable: never = request;
        return respond(
          failure(
            'unknown',
            (unreachable as WorkerRequest).requestId,
            'invalid-request',
            `Unknown worker request type: ${String((unreachable as WorkerRequest).type)}`
          )
        );
      }
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // `init` failures are almost always a malformed graph buffer; every other
    // request type only reaches `catch` for an unexpected runtime bug, since
    // its own inputs are validated above before touching the model.
    const code = request.type === 'init' ? 'invalid-graph' : 'internal-error';
    return respond(failure(request.type, request.requestId, code, message));
  }
};

/** Minimal dedicated-worker-scope surface this module wires itself onto. */
interface NeuralWorkerScope {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage: (message: WorkerResponse, transfer: Transferable[]) => void;
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
    const { response, transfer } = handleWorkerRequest(runtime, event.data);
    workerScope.postMessage(response, transfer ?? []);
  };
}
