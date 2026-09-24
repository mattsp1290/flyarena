// @vitest-environment node
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';

import { decodeAction } from '../../src/lib/arena/actions';
import { observeAgent } from '../../src/lib/arena/sensors';
import { createWorld, stepWorld } from '../../src/lib/arena/world';
import type { AgentId, DecodedAction, WorldState } from '../../src/lib/arena/types';
import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { parseGraphBinary, type ConnectomeGraph } from '../../src/lib/connectome/format';
import type { NeuralTelemetry } from '../../src/lib/connectome/telemetry';
import { createModelState, createOutputBuffer, createStepScratch, runSubsteps } from '../../src/lib/connectome/model';
import { createOracleAgentBinding } from '../../src/lib/experiment/bindings';
import { ExperimentRunner, type AgentBinding, type AgentStepInput, type AgentStepResult } from '../../src/lib/experiment/runner';
import { runEpisode } from '../../scripts/training/episode';

/**
 * WP1 episode re-grounding gate
 * (`.agents/plans/rewiring-null/01-rewired-graph-generation.md`): proves
 * `scripts/training/episode.ts`'s `runEpisode` (the standalone evaluator
 * WP2's authored-null scoring will run in sharded batches) reproduces the
 * exact same per-tick closed loop as the product's own
 * `src/lib/experiment/runner.ts#ExperimentRunner`, before anything is
 * scored against it. Both are driven over the real, committed biological
 * artifact (`public/data/malecns-arena-v1.bin.gz`), the authored decoder on
 * the left, and a parked opponent on the right, for seeds `30001..30003`
 * (the held-out range this plan reserves) at 300 ticks each.
 *
 * **Why the left side is replayed by hand as well as called through
 * `runEpisode`.** `runEpisode` only returns the *final* per-agent score
 * (`EpisodeResult`), not a per-tick trace, so there is no product API to
 * pull per-tick decoded actions from that call directly. The 'authored'
 * decoder path it runs internally
 * (`createNeuralRunner`'s `decoder === 'authored'` branch) is exactly three
 * calls in a fixed order -- `observeAgent` -> `runSubsteps` -> `decodeAction`
 * -- using the same exported primitives this test file also imports; this
 * test drives those same three primitives itself, once per tick, purely to
 * capture what `runEpisode` cannot expose. That replay is not trusted on
 * its own: each seed's block first asserts the replay's own final score
 * (computed the same way `runEpisode` computes it, by feeding its decoded
 * actions into `stepWorld`) equals `runEpisode`'s actual returned
 * `EpisodeResult.left` -- so a divergence between this file's replay and
 * `runEpisode` would fail *that* assertion, not silently pass the per-tick
 * comparison below by comparing two independently-wrong sequences.
 *
 * **The parked opponent.** `ExperimentRunner` has no "parked" concept --
 * `runOneTick` always steps both arms through their `AgentBinding`. A
 * disconnected graph is not equivalent either: `createDisconnectedGraph`
 * (`src/lib/connectome/format.ts`) keeps the input/output maps and still
 * produces a real (zero-driven) neural output, not a guaranteed zero action.
 * `createZeroAgentBinding` below is a test-only `AgentBinding` whose `step`
 * always resolves the zero action -- no product code changes. Since
 * `decodeAction([0, 0, 0])` equals episode.ts's own `ZERO_ACTION`, this is
 * exactly `createParkedRunner`'s behavior. `rightPositions` (tracked via
 * `onTelemetry`) confirms the right agent's position never moves under it,
 * i.e. that this test-only binding really is parked-equivalent.
 *
 * Stop-if-fails: if any seed's left-agent decoded actions or final score
 * diverge between the two paths, the failing `it` throws naming the exact
 * first divergent tick -- see `firstDivergentTickIndex` below -- rather than
 * only reporting an opaque array diff.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

const TICKS = 300;
const SEEDS: readonly number[] = [30001, 30002, 30003];

interface LoadedGraph {
  graph: ConnectomeGraph;
  buffer: ArrayBuffer;
}

const loadBiologicalGraph = (): LoadedGraph => {
  const gzipBytes = readFileSync(resolve(publicDataDir, 'malecns-arena-v1.bin.gz'));
  const binary = gunzipSync(gzipBytes);
  const buffer = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
  return { graph: parseGraphBinary(buffer.slice(0)), buffer };
};

/**
 * Test-only `AgentBinding` standing in for episode.ts's `createParkedRunner`
 * -- see this file's module doc for why `ExperimentRunner` has no built-in
 * equivalent and why no product code changes.
 */
const createZeroAgentBinding = (neuronCount: number): AgentBinding => {
  const zeroTelemetry: NeuralTelemetry = { meanRate: 0, minRate: 0, maxRate: 0, activeFraction: 0 };
  return {
    step: async (): Promise<AgentStepResult> => ({ actionFeatures: [0, 0, 0], telemetry: zeroTelemetry }),
    reset: async (): Promise<void> => {},
    info: { topology: 'biological', neuronCount, edgeCount: 0 }
  };
};

/** Wraps a binding's `step` to also push each call's decoded action into `sink`, in call order, without altering its resolved value. */
const recordingBinding = (binding: AgentBinding, sink: DecodedAction[]): AgentBinding => ({
  ...binding,
  step: async (input: AgentStepInput): Promise<AgentStepResult> => {
    const result = await binding.step(input);
    sink.push(decodeAction(result.actionFeatures));
    return result;
  }
});

const runToFinished = (runner: ExperimentRunner): Promise<void> =>
  new Promise((resolveRun, rejectRun) => {
    const poll = setInterval(() => {
      const status = runner.getStatus();
      if (status === 'finished') {
        clearInterval(poll);
        resolveRun();
      } else if (status === 'error') {
        clearInterval(poll);
        rejectRun(new Error('ExperimentRunner entered the error state during the parity gate'));
      }
    }, 1);
    runner.start();
  });

interface Position {
  x: number;
  z: number;
}

const capturePosition = (world: Readonly<WorldState>, agentId: AgentId): Position => {
  const agent = world.agents.find((candidate) => candidate.id === agentId);
  if (!agent) throw new Error(`parity gate: world is missing agent "${agentId}"`);
  return { x: agent.position.x, z: agent.position.z };
};

/** Index of the first tick at which `a`/`b` disagree on any decoded field, or -1 if they agree over their full shared length and are the same length. */
const firstDivergentTickIndex = (a: readonly DecodedAction[], b: readonly DecodedAction[]): number => {
  const length = Math.min(a.length, b.length);
  for (let tick = 0; tick < length; tick += 1) {
    if (a[tick].thrust !== b[tick].thrust || a[tick].yaw !== b[tick].yaw || a[tick].brake !== b[tick].brake) {
      return tick;
    }
  }
  return a.length === b.length ? -1 : length;
};

describe('episode.ts runEpisode vs ExperimentRunner: per-tick closed-loop parity', () => {
  const { graph, buffer } = loadBiologicalGraph();

  for (const seed of SEEDS) {
    it(`seed ${seed}: left (authored) per-tick decoded actions and final score match, right (parked) never moves`, async () => {
      // --- Path A: the real product `runEpisode` call. ---
      const episodeResult = runEpisode({
        seed,
        ticks: TICKS,
        substeps: NEURAL_SUBSTEPS_PER_TICK,
        left: { decoder: 'authored', graph },
        right: { decoder: 'parked' }
      });

      // --- Path A': per-tick instrumented replay of the exact same primitives
      // episode.ts's authored path composes -- see this file's module doc for
      // why, and note the self-check against `episodeResult.left` below.
      const state = createModelState(graph);
      const scratch = createStepScratch(graph);
      const outputs = createOutputBuffer(graph);
      let replayWorld = createWorld(seed);
      const episodeLeftActions: DecodedAction[] = [];
      const episodeRightPositions: Position[] = [];
      for (let tick = 0; tick < TICKS; tick += 1) {
        const observation = observeAgent(replayWorld, 'left');
        runSubsteps(graph, state, scratch, observation, NEURAL_SUBSTEPS_PER_TICK, outputs);
        const decoded = decodeAction(Array.from(outputs));
        episodeLeftActions.push(decoded);
        replayWorld = stepWorld(replayWorld, {
          left: [decoded.thrust, decoded.yaw, decoded.brake],
          right: [0, 0, 0]
        });
        episodeRightPositions.push(capturePosition(replayWorld, 'right'));
      }
      const replayLeftAgent = replayWorld.agents.find((agent) => agent.id === 'left');
      if (!replayLeftAgent) throw new Error('parity gate: replay world is missing the left agent');
      // Self-check: the hand-driven replay must reach the exact same final
      // score `runEpisode` itself returned, or the per-tick capture above
      // cannot be trusted as representative of `runEpisode`'s real behavior.
      expect(replayLeftAgent.score).toEqual(episodeResult.left);

      // --- Path B: the product closed-loop orchestrator, ExperimentRunner +
      // createOracleAgentBinding, with a test-only zero binding for "parked".
      const leftActions: DecodedAction[] = [];
      const leftBinding = recordingBinding(
        createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' }),
        leftActions
      );
      const rightBinding = createZeroAgentBinding(graph.metadata.neuronCount);
      const rightPositions: Position[] = [];
      const runner = new ExperimentRunner({
        seed,
        totalTicks: TICKS,
        agents: { left: leftBinding, right: rightBinding },
        substepsPerTick: NEURAL_SUBSTEPS_PER_TICK,
        targetTickIntervalMs: 0,
        onTelemetry: () => {
          rightPositions.push(capturePosition(runner.getWorld(), 'right'));
        }
      });
      const initialRightPosition = capturePosition(runner.getWorld(), 'right');
      await runToFinished(runner);

      // --- Comparison 1: per-tick left decoded actions. ---
      const divergentTick = firstDivergentTickIndex(episodeLeftActions, leftActions);
      if (divergentTick !== -1) {
        throw new Error(
          `seed ${seed}: episode.ts and ExperimentRunner left-agent decoded actions first diverge at tick ` +
            `${divergentTick} of ${TICKS}: runEpisode=${JSON.stringify(episodeLeftActions[divergentTick])}, ` +
            `ExperimentRunner=${JSON.stringify(leftActions[divergentTick])}`
        );
      }
      expect(leftActions).toHaveLength(TICKS);

      // --- Comparison 2: final score. ---
      const runnerLeftTelemetry = runner.getTelemetry().agents.left;
      expect({
        foodPickups: runnerLeftTelemetry.foodPickups,
        hazardContacts: runnerLeftTelemetry.hazardContacts,
        distanceTravelled: runnerLeftTelemetry.distanceTravelled,
        movementScore: runnerLeftTelemetry.movementScore
      }).toEqual(episodeResult.left);

      // --- Comparison 3: the right (parked) agent never moves, on either path. ---
      expect(rightPositions).toHaveLength(TICKS);
      for (const position of rightPositions) {
        expect(position).toEqual(initialRightPosition);
      }
      for (const position of episodeRightPositions) {
        expect(position).toEqual(initialRightPosition);
      }
    });
  }
});
