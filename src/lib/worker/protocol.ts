import type { GraphMode } from '../connectome/format';
import type { NeuralTelemetry } from '../connectome/telemetry';

/**
 * Tagged message protocol between the main thread and `neural.worker.ts`.
 * Every request carries a caller-chosen `requestId` so responses can be
 * matched to in-flight calls; every response is a discriminated union on
 * `ok` carrying either typed results or a structured `WorkerError`.
 */

export type WorkerErrorCode =
  | 'invalid-request'
  | 'not-initialized'
  | 'already-initialized'
  | 'invalid-graph'
  | 'internal-error';

/**
 * Upper bound on `StepWorkerRequest.substeps`. A request above this runs
 * synchronously inside the Worker for that many iterations before it can
 * respond to anything else, so this caps how long a single malformed/
 * malicious request can hang the Worker; 64 is generous headroom over the
 * POC's actual per-tick substep count (single digits) while still bounding
 * worst-case latency to a few milliseconds.
 */
export const MAX_SUBSTEPS_PER_TICK = 64;

export interface WorkerError {
  code: WorkerErrorCode;
  message: string;
}

/** Transfers the immutable graph binary once; the worker parses and keeps it. */
export interface InitWorkerRequest {
  type: 'init';
  requestId: string;
  graphBuffer: ArrayBuffer;
  /** Descriptive-only topology label; does not affect parsing or dynamics. */
  mode?: GraphMode;
}

/** Zeroes network state without re-parsing the graph. */
export interface ResetWorkerRequest {
  type: 'reset';
  requestId: string;
}

/**
 * Run one world tick's worth of neural substeps with a held-constant
 * observation and return only action features plus compact telemetry.
 */
export interface StepWorkerRequest {
  type: 'step';
  requestId: string;
  /**
   * Must have length equal to the initialized graph's `inputChannelCount`.
   * `ArrayLike<number>` (rather than `readonly number[]`) so a caller
   * holding sensor readings in a `Float32Array` can send it directly;
   * `stepModel` already accepts anything array-like, and both a plain
   * array and a typed array structured-clone across the Worker boundary
   * without conversion.
   */
  channelValues: ArrayLike<number>;
  /** Number of `stepModel` substeps to run before aggregating outputs. Must not exceed `MAX_SUBSTEPS_PER_TICK`. */
  substeps: number;
}

/** Releases the worker's graph/state buffers; the worker may be reused after `init`. */
export interface DisposeWorkerRequest {
  type: 'dispose';
  requestId: string;
}

export type WorkerRequest =
  | InitWorkerRequest
  | ResetWorkerRequest
  | StepWorkerRequest
  | DisposeWorkerRequest;

export interface InitWorkerSuccess {
  type: 'init';
  requestId: string;
  ok: true;
  neuronCount: number;
  edgeCount: number;
  inputChannelCount: number;
  outputPopulationCount: number;
  /** Echoes `InitWorkerRequest.mode`, so a caller can confirm which arm the Worker actually initialized. */
  mode?: GraphMode;
}

export interface ResetWorkerSuccess {
  type: 'reset';
  requestId: string;
  ok: true;
}

export interface StepWorkerSuccess {
  type: 'step';
  requestId: string;
  ok: true;
  /**
   * Length `outputPopulationCount`; population index conventionally maps to
   * thrust/yaw/brake per `OUTPUT_POPULATION` in `src/lib/arena/actions.ts`.
   */
  actionFeatures: readonly number[];
  telemetry: NeuralTelemetry;
}

export interface DisposeWorkerSuccess {
  type: 'dispose';
  requestId: string;
  ok: true;
}

/**
 * A failed response for any request type, carrying a structured error.
 * `type` is `'unknown'` when the inbound message itself was too malformed
 * (not an object, or missing a string `requestId`) to identify a request
 * type or id at all.
 */
export interface WorkerFailure {
  type: WorkerRequest['type'] | 'unknown';
  requestId: string;
  ok: false;
  error: WorkerError;
}

export type WorkerResponse =
  | InitWorkerSuccess
  | ResetWorkerSuccess
  | StepWorkerSuccess
  | DisposeWorkerSuccess
  | WorkerFailure;
