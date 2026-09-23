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
  /** Must have length equal to the initialized graph's `inputChannelCount`. */
  channelValues: readonly number[];
  /** Number of `stepModel` substeps to run before aggregating outputs. */
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
  /** Length `outputPopulationCount`; population 0/1/2 conventionally map to thrust/yaw/brake. */
  actionFeatures: readonly number[];
  telemetry: NeuralTelemetry;
}

export interface DisposeWorkerSuccess {
  type: 'dispose';
  requestId: string;
  ok: true;
}

/** A failed response for any request type, carrying a structured error. */
export interface WorkerFailure {
  type: WorkerRequest['type'];
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
