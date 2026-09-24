import type { GraphMode } from '../connectome/format';
import type { ReadoutWeights } from '../connectome/readout';
import { WORKER_PROTOCOL_VERSION } from './protocol';
import type {
  DecoderKind,
  DisposeWorkerSuccess,
  InitWorkerSuccess,
  ResetWorkerSuccess,
  SetActivityWorkerSuccess,
  SetDecoderWorkerSuccess,
  StepWorkerSuccess,
  WorkerErrorCode,
  WorkerRequest,
  WorkerResponse
} from './protocol';

/**
 * Promise-based RPC wrapper over the tagged `WorkerRequest`/`WorkerResponse`
 * protocol (`./protocol.ts`), matching responses back to callers by
 * `requestId`. This is the only place that owns request-id generation and
 * the pending-request map; `src/lib/experiment/runner.ts` only ever sees
 * `init`/`reset`/`step`/`dispose`/`setActivity` as plain async methods.
 *
 * Generic over `WorkerLike` (rather than the real DOM `Worker`) so the
 * request/response matching logic is unit-testable without a real Worker
 * thread — see `tests/unit/worker-client.test.ts`, which drives it with a
 * fake in-process `WorkerLike` pair. A real `Worker` instance already
 * satisfies this interface as-is.
 */
export interface WorkerLike {
  // Two overloads, mirroring the real DOM `Worker#postMessage` exactly
  // (required `transfer` array vs. no second argument at all), rather than
  // one signature with an optional param: TypeScript's structural check for
  // assigning a real `Worker` into this interface compares against DOM's
  // own overloaded declaration, and a single optional-`transfer` signature
  // is not assignable from it (the required-`transfer` overload rejects
  // `undefined`). A test double with a single optional-param `postMessage`
  // still satisfies this interface either way.
  postMessage(message: WorkerRequest, transfer: Transferable[]): void;
  postMessage(message: WorkerRequest): void;
  addEventListener: (
    type: 'message' | 'error' | 'messageerror',
    listener: (event: MessageEvent<WorkerResponse> | Event) => void
  ) => void;
  removeEventListener: (
    type: 'message' | 'error' | 'messageerror',
    listener: (event: MessageEvent<WorkerResponse> | Event) => void
  ) => void;
  terminate?: () => void;
}

export class WorkerClientError extends Error {
  /**
   * Machine-checkable failure category: a `WorkerErrorCode` when this wraps
   * a structured `WorkerError` from an actual Worker response (see
   * `handleMessage` below), `'protocol-version-mismatch'` for `init`'s own
   * version check, or `'client-error'` for every other client-side
   * condition (termination, a prior Worker `error`/`messageerror` event, a
   * malformed response) that has no `WorkerErrorCode` of its own. Lets a
   * caller branch on failure kind without parsing `message`.
   */
  readonly code: WorkerErrorCode | 'protocol-version-mismatch' | 'client-error';

  constructor(message: string, code: WorkerErrorCode | 'protocol-version-mismatch' | 'client-error' = 'client-error') {
    super(message);
    this.name = 'WorkerClientError';
    this.code = code;
  }
}

export interface WorkerClient {
  /**
   * Rejects with a `WorkerClientError` (`code: 'protocol-version-mismatch'`)
   * if the Worker's echoed `InitWorkerSuccess.protocolVersion` does not
   * match this bundle's own `WORKER_PROTOCOL_VERSION`. `readout`, when
   * given, is this arm's trained-readout weights (see
   * `InitWorkerRequest.readout`'s doc comment) — rejects with
   * `invalid-request` if they do not match the graph being initialized.
   */
  init: (graphBuffer: ArrayBuffer, mode?: GraphMode, readout?: Readonly<ReadoutWeights>) => Promise<InitWorkerSuccess>;
  reset: () => Promise<ResetWorkerSuccess>;
  /** Resolves with the full `StepWorkerSuccess`, including `rates` when activity streaming is currently enabled. */
  step: (channelValues: ArrayLike<number>, substeps: number) => Promise<StepWorkerSuccess>;
  dispose: () => Promise<DisposeWorkerSuccess>;
  /** Toggles whether subsequent `step` responses include the full per-neuron `rates` vector; rejects with `not-initialized` before `init`. */
  setActivity: (enabled: boolean) => Promise<SetActivityWorkerSuccess>;
  /** Switches this arm's decoding path; rejects with `not-initialized` before `init`, or `invalid-request` for `'trained'` when this Worker was initialized with no `readout`. */
  setDecoder: (decoder: DecoderKind) => Promise<SetDecoderWorkerSuccess>;
  /** Detach listeners and, if the underlying worker supports it, terminate it. Rejects every in-flight request. */
  terminate: () => void;
}

interface PendingRequest {
  resolve: (response: WorkerResponse & { ok: true }) => void;
  reject: (error: Error) => void;
}

let requestCounter = 0;
/** Monotonic, collision-free within one page session; the worker never persists requestIds across reloads. */
const nextRequestId = (): string => {
  requestCounter += 1;
  return `req-${requestCounter}-${Date.now().toString(36)}`;
};

/** Wrap a raw `WorkerLike` (a real dedicated Worker, or a test double) in the typed async API. */
export const createWorkerClient = (worker: WorkerLike): WorkerClient => {
  const pending = new Map<string, PendingRequest>();
  let terminated = false;
  /**
   * Set once the underlying Worker reports an `error`/`messageerror` event
   * (e.g. an uncaught exception during module evaluation). Without this, a
   * Worker that has actually died still looks "usable": every later `send`
   * would create a pending entry that can never be resolved (no more
   * messages are coming), so callers would hang forever instead of seeing
   * a rejection. Distinct from `terminated`, which is this client's own
   * deliberate shutdown rather than something the Worker reported.
   */
  let failedReason: string | undefined;

  const failAllPending = (message: string): void => {
    for (const { reject } of pending.values()) reject(new WorkerClientError(message));
    pending.clear();
  };

  const handleMessage = (event: MessageEvent<WorkerResponse> | Event): void => {
    const response = (event as MessageEvent<WorkerResponse>).data;
    if (!response || typeof response !== 'object' || typeof response.requestId !== 'string') return;
    const entry = pending.get(response.requestId);
    if (!entry) return;
    pending.delete(response.requestId);
    if (response.ok) {
      entry.resolve(response);
    } else {
      entry.reject(new WorkerClientError(`${response.error.code}: ${response.error.message}`, response.error.code));
    }
  };

  const handleError = (event: Event): void => {
    const message = event instanceof ErrorEvent ? event.message : 'worker error event';
    failedReason = message;
    failAllPending(`Neural worker error: ${message}`);
  };

  worker.addEventListener('message', handleMessage);
  worker.addEventListener('error', handleError);
  worker.addEventListener('messageerror', handleError);

  const send = <T extends WorkerResponse & { ok: true }>(
    request: WorkerRequest,
    transfer?: Transferable[]
  ): Promise<T> => {
    if (terminated) return Promise.reject(new WorkerClientError('Worker client has been terminated'));
    if (failedReason !== undefined) {
      return Promise.reject(new WorkerClientError(`Neural worker previously failed: ${failedReason}`));
    }
    return new Promise<T>((resolve, reject) => {
      pending.set(request.requestId, { resolve: resolve as PendingRequest['resolve'], reject });
      try {
        if (transfer) {
          worker.postMessage(request, transfer);
        } else {
          worker.postMessage(request);
        }
      } catch (error) {
        pending.delete(request.requestId);
        reject(error instanceof Error ? error : new WorkerClientError(String(error)));
      }
    });
  };

  return {
    init: async (graphBuffer, mode, readout) => {
      const result = await send<InitWorkerSuccess>(
        { type: 'init', requestId: nextRequestId(), graphBuffer, mode, readout },
        [graphBuffer]
      );
      // The Worker has already transitioned to `ready` by the time this
      // check runs (the check is purely a main-thread-side comparison of
      // the echoed `protocolVersion` against this bundle's own constant),
      // so a mismatch here means the two bundles disagree about the
      // request/response shape itself — worth surfacing as a distinct,
      // structured failure (`code: 'protocol-version-mismatch'`) rather than
      // letting a caller discover it indirectly via a malformed later `step`/
      // `set-activity` response.
      if (result.protocolVersion !== WORKER_PROTOCOL_VERSION) {
        throw new WorkerClientError(
          `Worker protocol version mismatch: main thread expects ${WORKER_PROTOCOL_VERSION}, Worker reported ${String(result.protocolVersion)}`,
          'protocol-version-mismatch'
        );
      }
      return result;
    },
    reset: () => send<ResetWorkerSuccess>({ type: 'reset', requestId: nextRequestId() }),
    step: (channelValues, substeps) =>
      send<StepWorkerSuccess>({ type: 'step', requestId: nextRequestId(), channelValues, substeps }),
    dispose: () => send<DisposeWorkerSuccess>({ type: 'dispose', requestId: nextRequestId() }),
    setActivity: (enabled) =>
      send<SetActivityWorkerSuccess>({ type: 'set-activity', requestId: nextRequestId(), enabled }),
    setDecoder: (decoder) =>
      send<SetDecoderWorkerSuccess>({ type: 'set-decoder', requestId: nextRequestId(), decoder }),
    terminate: () => {
      if (terminated) return;
      terminated = true;
      worker.removeEventListener('message', handleMessage);
      worker.removeEventListener('error', handleError);
      worker.removeEventListener('messageerror', handleError);
      failAllPending('Worker client was terminated');
      worker.terminate?.();
    }
  };
};
