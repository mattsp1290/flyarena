import {
  createDisconnectedGraph,
  encodeGraphBinary,
  parseGraphBinary,
  type ConnectomeGraph,
  type GraphMode
} from '../connectome/format';
import {
  createModelState,
  createOutputBuffer,
  createStepScratch,
  resetModelState,
  runSubsteps
} from '../connectome/model';
import { computeTelemetry } from '../connectome/telemetry';
import type { WorkerClient } from '../worker/client';
import type { AgentBinding, AgentStepInput, AgentStepResult } from './runner';

/**
 * Two ways to satisfy `AgentBinding` (`./runner.ts`): a synchronous CPU
 * oracle running in-thread (used by unit tests, and available as a
 * same-thread fallback) and a `WorkerClient`-backed binding (what
 * `src/App.svelte` actually uses in the browser — one dedicated Worker per
 * arm, per the plan's "one Worker per arm" choice: each arm can carry a
 * different topology and be reset/re-initialized independently without
 * disturbing the other arm's in-flight step).
 */

/**
 * Produce the graph buffer for one topology mode from a biological base
 * buffer: 'biological' is a defensive copy of `baseBuffer` (so the caller's
 * canonical copy survives a Worker `init` transfer); 'rewired' is a
 * defensive copy of the already-compiled rewired artifact
 * (`rewiredBuffer`); 'disconnected' is built at runtime per the plan — the
 * same node set and I/O mapping as the biological arm with `edgeCount`
 * forced to 0 (`connectome/format.ts#createDisconnectedGraph`), re-encoded
 * to the wire format so it can be sent through the same `init` path as a
 * fetched artifact.
 */
export const buildGraphBufferForMode = (
  baseBuffer: ArrayBuffer,
  rewiredBuffer: ArrayBuffer,
  mode: GraphMode
): ArrayBuffer => {
  if (mode === 'biological') return baseBuffer.slice(0);
  if (mode === 'rewired') return rewiredBuffer.slice(0);
  const graph = parseGraphBinary(baseBuffer.slice(0));
  return encodeGraphBinary(createDisconnectedGraph(graph));
};

export interface CreateOracleBindingOptions {
  graphBuffer: ArrayBuffer;
  mode: GraphMode;
  /**
   * Optional per-call artificial latency generator (ms), used only by tests
   * that inject randomized async delays to prove determinism does not
   * depend on response timing (see `tests/unit/experiment-runner.test.ts`).
   * Never set in production.
   */
  simulatedLatencyMs?: () => number;
}

/**
 * Build an `AgentBinding` that steps the synchronous CPU oracle
 * (`connectome/model.ts`) directly, wrapped in a resolved Promise.
 *
 * `AgentBinding#reset` must apply strictly after any previously-issued
 * `step` has settled (FIFO), leaving the arm's neural state exactly as
 * freshly initialized — `ExperimentRunner#reset()` relies on this to zero
 * neural state deterministically regardless of what was in flight when
 * `reset()` was called. A real Worker gets this for free from
 * `postMessage`'s FIFO delivery order; this in-thread binding has no
 * transport to rely on; and `options.simulatedLatencyMs` (test-only) means
 * a `step` call can still be pending its artificial delay when `reset` is
 * requested. Serialize every call — `step` and `reset` alike — through one
 * promise chain so a call is only ever handled after every earlier one has
 * fully finished.
 */
export const createOracleAgentBinding = (options: CreateOracleBindingOptions): AgentBinding => {
  const graph: ConnectomeGraph = parseGraphBinary(options.graphBuffer);
  const state = createModelState(graph);
  const scratch = createStepScratch(graph);
  const outputs = createOutputBuffer(graph);

  let queue: Promise<unknown> = Promise.resolve();
  /** Chain `fn` after every earlier queued call; a rejection never stalls the queue for later callers. */
  const serialize = <T>(fn: () => Promise<T>): Promise<T> => {
    const result = queue.then(fn);
    queue = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  };

  const step = (input: AgentStepInput): Promise<AgentStepResult> =>
    serialize(async () => {
      const latency = options.simulatedLatencyMs?.();
      if (latency && latency > 0) {
        await new Promise<void>((resolve) => setTimeout(resolve, latency));
      }
      runSubsteps(graph, state, scratch, input.channelValues, input.substeps, outputs);
      return { actionFeatures: Array.from(outputs), telemetry: computeTelemetry(state) };
    });

  const reset = (): Promise<void> =>
    serialize(async () => {
      resetModelState(state);
    });

  return {
    step,
    reset,
    info: { topology: options.mode, neuronCount: graph.metadata.neuronCount, edgeCount: graph.metadata.edgeCount }
  };
};

/**
 * Build an `AgentBinding` backed by a `WorkerClient` (`../worker/client.ts`).
 * Initializes the Worker with `graphBuffer` (transferred, not cloned) before
 * returning, so the binding is immediately usable.
 */
export const createWorkerAgentBinding = async (
  client: WorkerClient,
  graphBuffer: ArrayBuffer,
  mode: GraphMode
): Promise<AgentBinding> => {
  const initResult = await client.init(graphBuffer, mode);

  const step: AgentBinding['step'] = async (input) => {
    const result = await client.step(input.channelValues, input.substeps);
    return { actionFeatures: result.actionFeatures, telemetry: result.telemetry };
  };

  const reset: AgentBinding['reset'] = async () => {
    await client.reset();
  };

  return {
    step,
    reset,
    info: { topology: mode, neuronCount: initResult.neuronCount, edgeCount: initResult.edgeCount }
  };
};
