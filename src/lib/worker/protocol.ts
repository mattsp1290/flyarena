import type { GraphMode } from '../connectome/format';
import type { ReadoutWeights } from '../connectome/readout';
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

/**
 * Integer version of this request/response shape. Bumped whenever a request
 * or response type gains/loses/reshapes a field in a way a caller might need
 * to branch on. Echoed back in `InitWorkerSuccess.protocolVersion` and
 * checked by `createWorkerClient`'s `init` (`../worker/client.ts`) against
 * this same constant, rejecting with a structured
 * `protocol-version-mismatch` `WorkerClientError` on a mismatch — the main
 * thread and the Worker are always built from the same bundle today, so this
 * should never actually fire in production, but it turns a future
 * build-skew bug (e.g. a stale cached Worker script surviving a deploy) into
 * an immediate, diagnosable rejection instead of the Worker and main thread
 * silently disagreeing about a response shape — see
 * `docs/architecture.md`'s "versioned message protocol" note. `1` marked the
 * shape as of `set-activity`/`StepWorkerSuccess.rates`. `2` marks the shape
 * as of the trained-readout toggle
 * (`.agents/plans/trained-readout/06-browser-integration.md`):
 * `InitWorkerRequest.readout`, `SetDecoderWorkerRequest`/`SetDecoderWorkerSuccess`.
 */
export const WORKER_PROTOCOL_VERSION = 2;

/**
 * Which action-decoding path the Worker runs after each `step`'s neural
 * substeps: `'authored'` (`aggregateOutputs`, the pre-existing 3-population-sum
 * path) or `'trained'` (`readoutForward` over the per-neuron output rates —
 * requires `InitWorkerRequest.readout` to have been supplied at `init`).
 * Every Worker starts `'authored'` immediately after `init`, matching the
 * product's authored-by-default decision
 * (`.agents/plans/trained-readout/00-overview.md`); `set-decoder` is the only
 * way to change it.
 */
export type DecoderKind = 'authored' | 'trained';

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
  /**
   * This arm's trained readout weights, when the trained decoder is
   * available for it. Optional: a Worker initialized without `readout` can
   * still run `'authored'` (the default) but rejects `set-decoder: 'trained'`
   * with `invalid-request`. Validated against the parsed graph
   * (`validateReadoutWeights`) before `init` succeeds; a mismatch (wrong
   * `inputSize`/`hiddenSize`/array lengths, or a non-finite weight) fails the
   * whole `init` call with `invalid-request` — not `invalid-graph`, since the
   * graph buffer itself may be perfectly valid — so a caller can tell a
   * corrupt/mismatched trained-readout artifact apart from a corrupt graph
   * artifact. Never re-validated per step: `handleWorkerRequest`'s `init`
   * case is the one place this runs.
   */
  readout?: Readonly<ReadoutWeights>;
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

/**
 * Toggles whether subsequent `step` responses include the full per-neuron
 * `rates` vector (`StepWorkerSuccess.rates`). Off by default and reset to
 * off on every `init`, so a Worker never streams full-neuron state unless a
 * caller explicitly opts in — see `docs/architecture.md`'s streaming note
 * and `connectome/telemetry.ts`'s contract comment. Allowed in any runtime
 * state except `idle` (returns `not-initialized`, matching `reset`/`step`).
 */
export interface SetActivityWorkerRequest {
  type: 'set-activity';
  requestId: string;
  enabled: boolean;
}

/**
 * Switches which decoding path subsequent `step` responses use, without
 * re-`init`ing the Worker (and therefore without touching neural state —
 * `ExperimentController#setDecoder`, the one production caller, always
 * pairs this with an explicit `ExperimentRunner#reset()` at the orchestration
 * layer, per `.agents/plans/trained-readout/06-browser-integration.md`'s
 * "resets the experiment to tick 0" invariant; the Worker itself does not
 * reset anything here). Allowed in any runtime state except `idle` (returns
 * `not-initialized`, matching `reset`/`step`). Rejected with `invalid-request`
 * when `decoder: 'trained'` is requested but this Worker's `init` was never
 * given `readout`.
 */
export interface SetDecoderWorkerRequest {
  type: 'set-decoder';
  requestId: string;
  decoder: DecoderKind;
}

export type WorkerRequest =
  | InitWorkerRequest
  | ResetWorkerRequest
  | StepWorkerRequest
  | DisposeWorkerRequest
  | SetActivityWorkerRequest
  | SetDecoderWorkerRequest;

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
  /** Echoes `WORKER_PROTOCOL_VERSION` as of this Worker build; `createWorkerClient#init` checks this against its own copy of the constant. */
  protocolVersion: number;
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
  /**
   * Length `neuronCount`; the full per-neuron rate vector as of this step.
   * Present only when `set-activity` most recently enabled streaming for
   * this Worker — absent (not merely empty) otherwise, so
   * `!('rates' in response)` is a caller's cheap on/off check. Each
   * occurrence is a fresh `Float32Array` (`state.rate.slice()`) whose
   * `ArrayBuffer` is transferred (not structured-cloned) in the
   * `postMessage` call that carries this response — see
   * `neural.worker.ts#handleWorkerRequest`'s return type.
   */
  rates?: Float32Array;
}

export interface DisposeWorkerSuccess {
  type: 'dispose';
  requestId: string;
  ok: true;
}

export interface SetActivityWorkerSuccess {
  type: 'set-activity';
  requestId: string;
  ok: true;
  /** Echoes `SetActivityWorkerRequest.enabled`. */
  enabled: boolean;
}

export interface SetDecoderWorkerSuccess {
  type: 'set-decoder';
  requestId: string;
  ok: true;
  /** Echoes `SetDecoderWorkerRequest.decoder`. */
  decoder: DecoderKind;
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
  | SetActivityWorkerSuccess
  | SetDecoderWorkerSuccess
  | WorkerFailure;
