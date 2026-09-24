import { decodeAction, OUTPUT_POPULATION } from '../../src/lib/arena/actions';
import { observeAgent } from '../../src/lib/arena/sensors';
import { createWorld, stepWorld } from '../../src/lib/arena/world';
import type {
  ActionsByAgent,
  AgentId,
  AgentScore,
  DecodedAction,
  ReadonlyWorldState,
  WorldState
} from '../../src/lib/arena/types';
import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import type { ConnectomeGraph } from '../../src/lib/connectome/format';
import {
  createModelState,
  createOutputBuffer,
  createStepScratch,
  runLesionedSubsteps,
  runSubsteps
} from '../../src/lib/connectome/model';
import {
  createReadoutOutput,
  createReadoutScratch,
  outputNeuronIndices,
  readoutForward,
  type ReadoutWeights
} from '../../src/lib/connectome/readout';

/**
 * Headless, single-process episode runner: the authoritative evaluation
 * loop for `.agents/plans/trained-readout/04-authoritative-evaluation-and-artifacts.md`.
 * Built directly on `createWorld`/`observeAgent`/`runSubsteps`/`readoutForward`/
 * `decodeAction`/`stepWorld` (the same closed-loop order documented in
 * `docs/architecture.md`'s "Closed-loop contract": observe -> encode -> K
 * neural substeps -> aggregate -> decode -> world step). Originally written
 * ahead of the closed-loop bean (`flyarena-bb45`) that later added
 * `NEURAL_SUBSTEPS_PER_TICK`, which has since merged to `main` — see the
 * TODO below for what is still outstanding.
 *
 * TODO(flyarena-bb45): the closed-loop bean this evaluator was written ahead
 * of has since merged a reusable substep count
 * (`NEURAL_SUBSTEPS_PER_TICK`, `src/lib/connectome/constants.ts`), which
 * `substeps` now defaults to below, but not yet a reusable closed-loop step
 * function — this evaluator still re-drives `runSubsteps`/`decodeAction`/
 * `stepWorld` by hand instead of calling one shared implementation with
 * `ExperimentRunner` (`src/lib/experiment/runner.ts`). That refactor is
 * still owed; in the interim,
 * `tests/unit/episode-runner-parity.test.ts` guards against this file's
 * per-tick order drifting from `ExperimentRunner`'s by asserting the two
 * produce identical per-tick decoded actions and final scores for the
 * authored decoder against a parked opponent, over the real biological
 * artifact — any change to this file's tick loop must re-run that gate.
 */

/**
 * `parked`: always the zero action (the opponent-parked convention used by
 * training and by the headline evaluation conditions,
 * `.agents/plans/trained-readout/00-overview.md`'s "Opponent slot" row).
 * `authored`: the current shipped path, `aggregateOutputs` (inside
 * `runLesionedSubsteps`, numerically identical to `runSubsteps` when no
 * lesion is set) -> `decodeAction`.
 * `authored-flip-thrust`/`authored-flip-yaw`/`authored-flip-both`: the
 * authored family's decoder-convention-check variants
 * (`.agents/plans/null-explanation/01-decoder-variants.md` WP1) — identical
 * to `authored` except that, after `runLesionedSubsteps` fills the raw
 * `outputs` buffer (the only call site the flip is ever applied at — see
 * `createNeuralRunner`'s `step`, below) and before `decodeAction`, the
 * thrust and/or yaw entries (`OUTPUT_POPULATION.thrust`/`.yaw`,
 * `src/lib/arena/actions.ts`) are negated. Brake is never flipped. These
 * exist to test whether the authored decoder's fixed sign convention (as
 * opposed to the biological connectome's topology) explains the
 * rewiring-null study's below-null biological score; they are not a claim
 * about which convention is "correct" and never change the shipped
 * `decodeAction`/`aggregateOutputs` path itself.
 * `trained`: `readoutForward` on the real per-neuron output rates ->
 * `decodeAction`.
 * `silenced`: `readoutForward` fed an all-zero input vector every tick
 * instead of the real gathered rates — the network itself still steps
 * normally on real sensory input, but the readout never sees it. This is
 * the circuit-silenced control (`04-authoritative-evaluation-and-artifacts.md`'s
 * "silenced" condition, following Fly Dino's practice).
 */
export type EpisodeDecoderKind =
  | 'authored'
  | 'authored-flip-thrust'
  | 'authored-flip-yaw'
  | 'authored-flip-both'
  | 'trained'
  | 'silenced'
  | 'parked';

/**
 * True for the `authored` decoder family: `authored` and its three
 * sign-flip variants (`authored-flip-thrust`/`authored-flip-yaw`/
 * `authored-flip-both`, `.agents/plans/null-explanation/01-decoder-variants.md`).
 * Written as a predicate rather than repeating this set at every call site,
 * so lesion support (and any other authored-family-only behavior) stays in
 * sync across `createAgentRunner`/`createNeuralRunner` with one edit here.
 */
const isAuthoredFamily = (decoder: EpisodeDecoderKind): boolean =>
  decoder === 'authored' ||
  decoder === 'authored-flip-thrust' ||
  decoder === 'authored-flip-yaw' ||
  decoder === 'authored-flip-both';

export interface AgentEpisodeConfig {
  readonly decoder: EpisodeDecoderKind;
  /** Required for every decoder except `parked`. */
  readonly graph?: Readonly<ConnectomeGraph>;
  /** Required for `trained` and `silenced`; ignored otherwise. */
  readonly weights?: Readonly<ReadoutWeights>;
  /**
   * Neuron indices to silence for the whole episode, matching the
   * counterfactual workbench's lesion semantics exactly
   * (`runLesionedSubsteps`, `src/lib/connectome/model.ts`, shared with
   * `stepBranch` in `src/lib/counterfactual/engine.ts`): rates at these
   * indices are zeroed before the substep loop -- clearing whatever this
   * agent's own last tick left there, exactly as `stepBranch` clears a
   * fork's carried-over state before its first scatter -- and again after
   * every substep, before `aggregateOutputs` runs. Valid only for the
   * authored decoder family (`isAuthoredFamily` below); `trained`,
   * `silenced`, and `parked` runners never read per-neuron rate state the
   * same way `authored` does and would silently ignore it, so `runEpisode`
   * throws instead for those. Indices must be sorted ascending, unique, and
   * in `[0, graph.metadata.neuronCount)`; `runEpisode` throws on the first
   * violation. The caller's array is defensively copied (`Int32Array.from`)
   * before use, so mutating or reusing it after `runEpisode` starts has no
   * effect on this episode; the `instanceof Int32Array` check runs on the
   * caller's original value, before that copy, because a WP2 sharded worker
   * deserializing a lesion set across a process boundary (JSON/IPC) can hand
   * back a plain array-like that would otherwise copy zero elements
   * silently instead of throwing. An unset `lesion` and an empty
   * (zero-length) `lesion` are both numerically identical to no lesion at
   * all: both drive the same `runLesionedSubsteps` call, whose zeroing
   * loops are no-ops at length zero.
   */
  readonly lesion?: Int32Array;
}

export interface EpisodeConfig {
  readonly seed: number;
  readonly ticks: number;
  /**
   * Neural substeps per world tick (K). See this file's doc comment.
   * Defaults to `NEURAL_SUBSTEPS_PER_TICK` (`src/lib/connectome/constants.ts`,
   * the production value `ExperimentRunner` itself defaults to) when
   * omitted; pass it explicitly to run a non-production substep count.
   */
  readonly substeps?: number;
  readonly left: AgentEpisodeConfig;
  readonly right: AgentEpisodeConfig;
  /**
   * Diagnostic-only hook, never used by a production caller (`evaluate.ts`
   * never passes it): invoked once per tick, immediately after `stepWorld`,
   * with the exact decoded actions this tick fed into it and the resulting
   * world. Exists so `tests/unit/episode-runner-parity.test.ts` can observe
   * this file's real per-tick closed loop directly -- rather than a
   * hand-driven re-derivation of it from the underlying primitives, which a
   * review pass found could drift from this file's own tick loop without
   * the parity gate noticing (see that test's module doc).
   */
  readonly onTick?: (tick: number, actions: Readonly<Record<AgentId, DecodedAction>>, world: ReadonlyWorldState) => void;
}

export interface AgentScoreResult {
  readonly foodPickups: number;
  readonly hazardContacts: number;
  readonly distanceTravelled: number;
  readonly movementScore: number;
}

export interface EpisodeResult {
  readonly ticks: number;
  readonly left: AgentScoreResult;
  readonly right: AgentScoreResult;
}

const ZERO_ACTION: readonly [number, number, number] = [0, 0, 0];

/**
 * Shared no-op lesion: zero-length, so `runLesionedSubsteps`'s zeroing loops
 * never execute and it is numerically identical to `runSubsteps`. Lets
 * `createNeuralRunner`'s authored branch use one `step` closure regardless
 * of whether `config.lesion` was set.
 */
const EMPTY_LESION = new Int32Array(0);

interface AgentRunner {
  step(world: Readonly<WorldState>): readonly [number, number, number];
}

const createParkedRunner = (): AgentRunner => ({
  step: () => ZERO_ACTION
});

/**
 * Throws unless `lesion` (already confirmed a real `Int32Array` and copied
 * by the caller -- see `createNeuralRunner`'s authored branch) is sorted
 * strictly ascending (which also rules out duplicates) with every index in
 * `[0, neuronCount)`. Order does not change the numeric result -- zeroing a
 * set of indices is order-independent -- but `AgentEpisodeConfig.lesion`'s
 * documented contract is a sorted, unique, in-range `Int32Array`, and
 * enforcing it here catches a caller's indexing bug (an out-of-range or
 * repeated neuron id) instead of silently zeroing the wrong -- or no --
 * neuron.
 */
const validateLesionIndices = (agentId: AgentId, lesion: Int32Array, neuronCount: number): void => {
  for (let i = 0; i < lesion.length; i += 1) {
    const index = lesion[i];
    if (index < 0 || index >= neuronCount) {
      throw new Error(
        `episode: agent "${agentId}" lesion index ${index} is out of range [0, ${neuronCount})`
      );
    }
    if (i > 0 && index <= lesion[i - 1]) {
      throw new Error(
        `episode: agent "${agentId}" lesion indices must be sorted ascending and unique, ` +
          `got ${lesion[i - 1]} then ${index} at position ${i}`
      );
    }
  }
};

/**
 * Builds the per-tick action producer for a non-parked agent. Neural state
 * (`createModelState`) starts at zero, matching WP3's "Neural state reset
 * to zero at episode start" and `createModelState`'s own zero-fill default
 * (`src/lib/connectome/model.ts`) — no explicit reset call is needed.
 */
const createNeuralRunner = (
  agentId: AgentId,
  config: Readonly<AgentEpisodeConfig>,
  substeps: number
): AgentRunner => {
  if (!config.graph) {
    throw new Error(`episode: agent "${agentId}" decoder "${config.decoder}" requires a graph`);
  }
  const graph = config.graph;
  const state = createModelState(graph);
  const scratch = createStepScratch(graph);
  const outputs = createOutputBuffer(graph);

  if (isAuthoredFamily(config.decoder)) {
    let lesion: Int32Array = EMPTY_LESION;
    if (config.lesion) {
      // Checked on the caller's own value, before the defensive copy below,
      // so a non-`Int32Array` array-like (see the `lesion` doc comment)
      // throws instead of the copy silently producing an empty lesion.
      if (!(config.lesion instanceof Int32Array)) {
        throw new Error(`episode: agent "${agentId}" lesion must be an Int32Array`);
      }
      lesion = Int32Array.from(config.lesion);
      validateLesionIndices(agentId, lesion, graph.metadata.neuronCount);
    }
    // Precomputed once per runner (not per tick): which raw output entries
    // this decoder kind flips before `decodeAction`, per the
    // `authored-flip-*` doc comment above. `authored` itself flips neither.
    const flipThrust =
      config.decoder === 'authored-flip-thrust' || config.decoder === 'authored-flip-both';
    const flipYaw = config.decoder === 'authored-flip-yaw' || config.decoder === 'authored-flip-both';
    return {
      step: (world) => {
        const observation = observeAgent(world, agentId);
        runLesionedSubsteps(graph, state, scratch, observation, lesion, substeps, outputs);
        if (flipThrust) outputs[OUTPUT_POPULATION.thrust] *= -1;
        if (flipYaw) outputs[OUTPUT_POPULATION.yaw] *= -1;
        const decoded = decodeAction(Array.from(outputs));
        return [decoded.thrust, decoded.yaw, decoded.brake];
      }
    };
  }

  if (config.decoder === 'trained' || config.decoder === 'silenced') {
    if (!config.weights) {
      throw new Error(`episode: agent "${agentId}" decoder "${config.decoder}" requires weights`);
    }
    const weights = config.weights;
    const indices = outputNeuronIndices(graph);
    const readoutScratch = createReadoutScratch(weights.hiddenSize);
    const readoutOut = createReadoutOutput();
    // Reused, never mutated: the silenced condition's whole point is that
    // this stays all-zero every tick regardless of the network's real
    // state, so `readoutForward`'s gathered input is always zero.
    const zeroRate =
      config.decoder === 'silenced' ? new Float32Array(graph.metadata.neuronCount) : null;

    return {
      step: (world) => {
        const observation = observeAgent(world, agentId);
        runSubsteps(graph, state, scratch, observation, substeps, outputs);
        const readoutInput = zeroRate ?? state.rate;
        readoutForward(weights, readoutInput, indices, readoutScratch, readoutOut);
        const decoded = decodeAction(Array.from(readoutOut));
        return [decoded.thrust, decoded.yaw, decoded.brake];
      }
    };
  }

  throw new Error(`episode: unknown decoder "${String(config.decoder)}"`);
};

const createAgentRunner = (
  agentId: AgentId,
  config: Readonly<AgentEpisodeConfig>,
  substeps: number
): AgentRunner => {
  // Checked before dispatch (not inside createNeuralRunner) so it also
  // covers 'parked', which never reaches createNeuralRunner at all, and so
  // the error fires before any decoder-specific "requires a graph/weights"
  // check -- a lesion on the wrong decoder kind is a caller error regardless
  // of what else the config is missing.
  if (config.lesion && !isAuthoredFamily(config.decoder)) {
    throw new Error(
      `episode: agent "${agentId}" decoder "${config.decoder}" does not support lesion ` +
        '(the authored decoder family only)'
    );
  }
  return config.decoder === 'parked' ? createParkedRunner() : createNeuralRunner(agentId, config, substeps);
};

const toAgentScoreResult = (score: Readonly<AgentScore>): AgentScoreResult => ({
  foodPickups: score.foodPickups,
  hazardContacts: score.hazardContacts,
  distanceTravelled: score.distanceTravelled,
  movementScore: score.movementScore
});

/**
 * Run one deterministic episode: `createWorld(seed)`, then `ticks` fixed
 * 30 Hz steps, each agent decoding through its own configured path. Both
 * agents are always stepped through `stepWorld` together (per-tick, in
 * `OUTPUT_POPULATION` order via `decodeAction`), matching
 * `stepWorld`'s own two-agent design (`src/lib/arena/world.ts`) and the
 * exact action-encoding convention `scripts/training/export-traces.ts`'s
 * `buildSeedTrace` uses (decode once into a `DecodedAction`, then pass the
 * `[thrust, yaw, brake]` array back into `stepWorld`, which decodes it
 * again — idempotent for already-clamped values). `right: 'parked'`
 * reproduces the "opponent receives zero action" training/evaluation
 * convention exactly.
 */
export const runEpisode = (config: Readonly<EpisodeConfig>): EpisodeResult => {
  if (!Number.isInteger(config.ticks) || config.ticks < 0) {
    throw new Error(`episode: ticks must be a non-negative integer, got ${config.ticks}`);
  }
  const substeps = config.substeps ?? NEURAL_SUBSTEPS_PER_TICK;
  if (!Number.isInteger(substeps) || substeps <= 0) {
    throw new Error(`episode: substeps must be a positive integer, got ${substeps}`);
  }

  const leftRunner = createAgentRunner('left', config.left, substeps);
  const rightRunner = createAgentRunner('right', config.right, substeps);

  let world = createWorld(config.seed);
  for (let tick = 0; tick < config.ticks; tick += 1) {
    const leftAction = leftRunner.step(world);
    const rightAction = rightRunner.step(world);
    const actions: ActionsByAgent = { left: leftAction, right: rightAction };
    world = stepWorld(world, actions);
    // decodeAction here is idempotent for these already-clamped tuples (see
    // this function's doc comment); this is the same "actually applied this
    // tick" action `stepWorld` decoded internally, not a re-derivation.
    config.onTick?.(tick, { left: decodeAction(leftAction), right: decodeAction(rightAction) }, world);
  }

  const leftAgent = world.agents.find((agent) => agent.id === 'left');
  const rightAgent = world.agents.find((agent) => agent.id === 'right');
  if (!leftAgent || !rightAgent) {
    throw new Error('episode: world is missing an expected agent');
  }

  return {
    ticks: world.tick,
    left: toAgentScoreResult(leftAgent.score),
    right: toAgentScoreResult(rightAgent.score)
  };
};
