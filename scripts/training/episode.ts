import { decodeAction, OUTPUT_POPULATION } from '../../src/lib/arena/actions';
import { observeAgent } from '../../src/lib/arena/sensors';
import { resolveArenaTask } from '../../src/lib/arena/tasks';
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
  runSubsteps,
  type SubstepObserver
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
  /**
   * Read-only per-substep observer, originally `.agents/plans/
   * null-explanation/02-transfer-and-features.md`'s WP2 regime check
   * (authored-family only), widened by `.agents/plans/readout-attribution/
   * 02-analyses.md`'s WP2 to the `trained`/`silenced` decoders too (so
   * `scripts/attribution/regime.ts` can measure the same clamp-fraction/
   * steady-state-distance statistic on trained-readout trajectories, at the
   * same granularity as the explanation study). Threaded through to
   * `runLesionedSubsteps` (authored family) or `runSubsteps` (`trained`/
   * `silenced`) -- see `SubstepObserver`'s doc comment
   * (`src/lib/connectome/model.ts`) for exactly what it observes and why it
   * takes `channelValues` as well as `rate`. For `trained`, the observer
   * always sees the network's real, unmasked rate (never a `readoutMask`-ed
   * copy -- use `onReadoutInput` below for the readout's own gathered,
   * post-mask view). Only `parked` (no network stepped at all) rejects it;
   * `runEpisode` throws there. Omitting it costs nothing (see
   * `SubstepObserver`'s "optional and additive" doc comment); this is why
   * `tests/unit/episode-runner-parity.test.ts` and the Worker parity test
   * did not need to change for either addition.
   */
  readonly onSubstep?: SubstepObserver;
  /**
   * `.agents/plans/readout-attribution/02-analyses.md`'s WP2 readout-input
   * mask: indices into the readout's own `D`-length input space
   * (`outputNeuronIndices(graph)` order, i.e. `0 <= d < weights.inputSize`
   * -- NOT raw neuron ids the way `lesion` is), zeroed in the readout's
   * input only, leaving the network's real dynamics (`state.rate`)
   * untouched. Valid only for decoder `trained`; `runEpisode` throws for
   * any other decoder, including `silenced` (whose whole input is already
   * always zero -- masking it is meaningless). Must be sorted ascending,
   * unique, and in `[0, weights.inputSize)`; `runEpisode` throws on the
   * first violation, mirroring `lesion`'s own contract. An unset or
   * zero-length mask is bit-identical to no mask; a mask covering every
   * `[0, D)` index is bit-identical to `silenced` -- `readoutForward` only
   * ever reads the `D` gathered positions (`indices[input]`), so it cannot
   * observe whether the rest of the copied buffer still holds the
   * network's real per-neuron rates (unmasked positions) or is all-zero
   * (`silenced`'s buffer): see `createNeuralRunner`'s masked-copy below.
   */
  readonly readoutMask?: Int32Array;
  /**
   * Read-only, allocation-free, off-by-default per-tick observer for the
   * `trained`/`silenced` decoders: called once per tick, immediately before
   * `readoutForward`, with the exact `Float32Array` the readout is about to
   * read (post-mask for `trained`, the shared all-zero buffer for
   * `silenced`) and the 0-based tick index. `runEpisode` throws if this is
   * set for any other decoder. `scripts/attribution/saliency.ts` uses this
   * to record the readout-input trajectories saliency is computed from.
   */
  readonly onReadoutInput?: (rate: Float32Array, tick: number) => void;
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
   * `.agents/plans/task-generality/01-task-plumbing.md`'s WP1: the arena
   * task id (`src/lib/arena/tasks.ts`'s `ARENA_TASKS`) whose config this
   * episode's `createWorld` uses. `undefined` resolves to `'default'`
   * (`ARENA_CONFIG`, unchanged) — every existing caller that never sets this
   * field keeps producing byte-identical output. `resolveArenaTask` throws
   * on an unrecognized id, so a stale/typo'd id fails loudly here rather
   * than silently scoring the wrong task.
   */
  readonly arenaTask?: string;
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

/** Same sorted/unique/in-range contract as `validateLesionIndices`, but against the readout's own `D`-length input space (`readoutMask`'s doc comment), not the graph's raw neuron count. */
const validateReadoutMaskIndices = (agentId: AgentId, mask: Int32Array, inputSize: number): void => {
  for (let i = 0; i < mask.length; i += 1) {
    const index = mask[i];
    if (index < 0 || index >= inputSize) {
      throw new Error(
        `episode: agent "${agentId}" readoutMask index ${index} is out of range [0, ${inputSize})`
      );
    }
    if (i > 0 && index <= mask[i - 1]) {
      throw new Error(
        `episode: agent "${agentId}" readoutMask indices must be sorted ascending and unique, ` +
          `got ${mask[i - 1]} then ${index} at position ${i}`
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
    const onSubstep = config.onSubstep;
    return {
      step: (world) => {
        const observation = observeAgent(world, agentId);
        runLesionedSubsteps(graph, state, scratch, observation, lesion, substeps, outputs, onSubstep);
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

    // `readoutMask` support (`trained` only -- checked in `createAgentRunner`
    // before this branch is ever reached). Preallocated once, shaped like
    // `state.rate` (not `readoutScratch`, the H-length hidden buffer): every
    // tick, the network's real rate is copied in full, then the masked `D`
    // positions (via `indices`, `readoutMask`'s own D-space contract) are
    // zeroed -- dynamics (`state.rate` itself) are never touched. See
    // `AgentEpisodeConfig.readoutMask`'s doc comment for the two identities
    // (empty mask == no mask, full mask == `silenced`) this preserves.
    // `config.decoder === 'trained'` is already guaranteed here --
    // `createAgentRunner` rejects `readoutMask` on any other decoder
    // (including `silenced`) before this function is ever called.
    let mask: Int32Array | null = null;
    let maskedRate: Float32Array | null = null;
    if (config.readoutMask) {
      if (!(config.readoutMask instanceof Int32Array)) {
        throw new Error(`episode: agent "${agentId}" readoutMask must be an Int32Array`);
      }
      mask = Int32Array.from(config.readoutMask);
      validateReadoutMaskIndices(agentId, mask, weights.inputSize);
      maskedRate = new Float32Array(graph.metadata.neuronCount);
    }

    const onReadoutInput = config.onReadoutInput;
    const onSubstep = config.onSubstep;
    let tick = 0;

    return {
      step: (world) => {
        const observation = observeAgent(world, agentId);
        runSubsteps(graph, state, scratch, observation, substeps, outputs, onSubstep);
        let readoutInput = zeroRate ?? state.rate;
        if (mask && maskedRate) {
          maskedRate.set(readoutInput);
          for (let i = 0; i < mask.length; i += 1) maskedRate[indices[mask[i]]] = 0;
          readoutInput = maskedRate;
        }
        onReadoutInput?.(readoutInput, tick);
        tick += 1;
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
  // `onSubstep` is now valid for every decoder that actually steps a
  // network -- the authored family (unchanged) plus `trained`/`silenced`
  // (`.agents/plans/readout-attribution/02-analyses.md`'s WP2, threaded
  // through to `runSubsteps`' own `onSubstep` parameter in
  // `createNeuralRunner`'s trained/silenced branch). Only `parked` (no
  // network stepped at all) rejects it.
  if (config.onSubstep && config.decoder === 'parked') {
    throw new Error(`episode: agent "${agentId}" decoder "parked" does not support onSubstep (no network is stepped)`);
  }
  if (config.readoutMask && config.decoder !== 'trained') {
    throw new Error(
      `episode: agent "${agentId}" decoder "${config.decoder}" does not support readoutMask ` +
        '(the "trained" decoder only)'
    );
  }
  if (config.onReadoutInput && config.decoder !== 'trained' && config.decoder !== 'silenced') {
    throw new Error(
      `episode: agent "${agentId}" decoder "${config.decoder}" does not support onReadoutInput ` +
        '(the "trained"/"silenced" decoders only)'
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

  let world = createWorld(config.seed, resolveArenaTask(config.arenaTask).config);
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
