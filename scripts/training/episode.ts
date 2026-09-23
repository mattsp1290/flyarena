import { decodeAction } from '../../src/lib/arena/actions';
import { observeAgent } from '../../src/lib/arena/sensors';
import { createWorld, stepWorld } from '../../src/lib/arena/world';
import type { ActionsByAgent, AgentId, AgentScore, WorldState } from '../../src/lib/arena/types';
import type { ConnectomeGraph } from '../../src/lib/connectome/format';
import {
  createModelState,
  createOutputBuffer,
  createStepScratch,
  runSubsteps
} from '../../src/lib/connectome/model';
import {
  createReadoutOutput,
  createReadoutScratch,
  outputNeuronIndices,
  readoutForward,
  type ReadoutWeights
} from '../../src/lib/connectome/readout';
import { TRACE_SUBSTEPS } from './export-traces';

/**
 * Headless, single-process episode runner: the authoritative evaluation
 * loop for `.agents/plans/trained-readout/04-authoritative-evaluation-and-artifacts.md`.
 * Built directly on `createWorld`/`observeAgent`/`runSubsteps`/`readoutForward`/
 * `decodeAction`/`stepWorld` (the same closed-loop order documented in
 * `docs/architecture.md`'s "Closed-loop contract": observe -> encode -> K
 * neural substeps -> aggregate -> decode -> world step), because the
 * closed-loop bean (`flyarena-bb45`) and its `NEURAL_SUBSTEPS_PER_TICK` are
 * not on `main` yet.
 *
 * TODO(flyarena-bb45): once the closed-loop bean merges a reusable
 * closed-loop step function and fixes the real substep count `K`, call that
 * function here instead of re-driving `runSubsteps`/`decodeAction`/
 * `stepWorld` by hand, so this evaluator cannot drift from the shipped
 * product's per-tick order. Until then, `substeps` defaults to
 * `TRACE_SUBSTEPS` (`export-traces.ts`) for trace-graph development; a real
 * evaluation run must pass the real `K` explicitly (see `evaluate.ts`).
 */

/**
 * `parked`: always the zero action (the opponent-parked convention used by
 * training and by the headline evaluation conditions,
 * `.agents/plans/trained-readout/00-overview.md`'s "Opponent slot" row).
 * `authored`: the current shipped path, `aggregateOutputs` (inside
 * `runSubsteps`) -> `decodeAction`.
 * `trained`: `readoutForward` on the real per-neuron output rates ->
 * `decodeAction`.
 * `silenced`: `readoutForward` fed an all-zero input vector every tick
 * instead of the real gathered rates — the network itself still steps
 * normally on real sensory input, but the readout never sees it. This is
 * the circuit-silenced control (`04-authoritative-evaluation-and-artifacts.md`'s
 * "silenced" condition, following Fly Dino's practice).
 */
export type EpisodeDecoderKind = 'authored' | 'trained' | 'silenced' | 'parked';

export interface AgentEpisodeConfig {
  readonly decoder: EpisodeDecoderKind;
  /** Required for every decoder except `parked`. */
  readonly graph?: Readonly<ConnectomeGraph>;
  /** Required for `trained` and `silenced`; ignored otherwise. */
  readonly weights?: Readonly<ReadoutWeights>;
}

export interface EpisodeConfig {
  readonly seed: number;
  readonly ticks: number;
  /** Neural substeps per world tick (K). See this file's doc comment. */
  readonly substeps: number;
  readonly left: AgentEpisodeConfig;
  readonly right: AgentEpisodeConfig;
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

interface AgentRunner {
  step(world: Readonly<WorldState>): readonly [number, number, number];
}

const createParkedRunner = (): AgentRunner => ({
  step: () => ZERO_ACTION
});

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

  if (config.decoder === 'authored') {
    return {
      step: (world) => {
        const observation = observeAgent(world, agentId);
        runSubsteps(graph, state, scratch, observation, substeps, outputs);
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
): AgentRunner => (config.decoder === 'parked' ? createParkedRunner() : createNeuralRunner(agentId, config, substeps));

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
  if (!Number.isInteger(config.substeps) || config.substeps <= 0) {
    throw new Error(`episode: substeps must be a positive integer, got ${config.substeps}`);
  }

  const leftRunner = createAgentRunner('left', config.left, config.substeps);
  const rightRunner = createAgentRunner('right', config.right, config.substeps);

  let world = createWorld(config.seed);
  for (let tick = 0; tick < config.ticks; tick += 1) {
    const leftAction = leftRunner.step(world);
    const rightAction = rightRunner.step(world);
    const actions: ActionsByAgent = { left: leftAction, right: rightAction };
    world = stepWorld(world, actions);
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
