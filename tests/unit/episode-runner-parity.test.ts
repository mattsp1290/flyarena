// @vitest-environment node
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';

import { decodeAction } from '../../src/lib/arena/actions';
import type { AgentId, DecodedAction, ReadonlyWorldState } from '../../src/lib/arena/types';
import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { createDisconnectedGraph, parseGraphBinary, type ConnectomeGraph, type GraphMode } from '../../src/lib/connectome/format';
import { decodeReadoutArtifact, validateReadoutWeights, type ReadoutWeights, type TrainedReadoutArtifactJson } from '../../src/lib/connectome/readout';
import type { NeuralTelemetry } from '../../src/lib/connectome/telemetry';
import { buildGraphBufferForMode, createOracleAgentBinding, createWorkerAgentBinding } from '../../src/lib/experiment/bindings';
import { ExperimentRunner, type AgentBinding, type AgentStepInput, type AgentStepResult } from '../../src/lib/experiment/runner';
import { createWorkerClient } from '../../src/lib/worker/client';
import { runEpisode, type AgentScoreResult } from '../../scripts/training/episode';
import { FakeNeuralWorker } from '../helpers/fake-worker';

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
 * **Per-tick capture comes from `runEpisode` itself, via `onTick`.**
 * `runEpisode` only returns the *final* per-agent score (`EpisodeResult`),
 * with no built-in way to observe its per-tick actions. An earlier version
 * of this test worked around that by hand-replaying episode.ts's authored
 * decoder logic (`observeAgent` -> `runSubsteps` -> `decodeAction`) and
 * comparing *that replica* against `ExperimentRunner`, linking it back to
 * the real `runEpisode` only through a final-score equality check. A
 * review pass (with a mutation test: feeding the left arm a stale
 * observation) confirmed that setup could report a passing per-tick
 * comparison while the actual `runEpisode` tick loop had drifted, only
 * surfacing as an opaque final-score mismatch with no tick index. `onTick`
 * (`EpisodeConfig`, `episode.ts`) removes the need for a replica entirely:
 * it fires once per tick, right after `stepWorld`, with the exact decoded
 * actions that tick fed into it -- so `episodeLeftActions` below is
 * `runEpisode`'s own per-tick record, not a second implementation of it.
 *
 * **The parked opponent.** `ExperimentRunner` has no "parked" concept --
 * `runOneTick` always steps both arms through their `AgentBinding`. A
 * disconnected graph is not equivalent either: `createDisconnectedGraph`
 * (`src/lib/connectome/format.ts`) keeps the input/output maps and still
 * produces a real (zero-driven) neural output, not a guaranteed zero action.
 * `createZeroAgentBinding` below is a test-only `AgentBinding` whose `step`
 * always resolves the zero action -- no product code changes. Since
 * `decodeAction([0, 0, 0])` equals episode.ts's own `ZERO_ACTION`, this is
 * exactly `createParkedRunner`'s behavior. Both paths' right-arm results are
 * checked: `rightPositions` (via `ExperimentRunner`'s `onTelemetry`) and
 * `episodeRightPositions` (via `onTick`) each confirm the right agent's
 * position never moves, and `episodeResult.right` is compared against
 * `ExperimentRunner`'s own right-arm telemetry and asserted to have
 * travelled zero distance -- a review pass found the first version of this
 * test never checked `runEpisode`'s own parked arm at all (a mutated
 * `createParkedRunner` returning a nonzero action still passed).
 *
 * Stop-if-fails: if any seed's left-agent decoded actions or either arm's
 * final score diverge between the two paths, the failing `it` throws
 * naming the exact first divergent tick -- see `firstDivergentTickIndex`
 * below -- rather than only reporting an opaque array diff.
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

/**
 * Rejects with the runner's real error (captured via `errorSink`, populated
 * by an `onError` callback passed at construction) instead of a generic
 * message, if the runner never reaches `finished`.
 */
const runToFinished = (runner: ExperimentRunner, errorSink: { current?: Error }): Promise<void> =>
  new Promise((resolveRun, rejectRun) => {
    let settled = false;
    const poll = setInterval(() => {
      if (settled) return;
      const status = runner.getStatus();
      if (status === 'finished') {
        settled = true;
        clearInterval(poll);
        resolveRun();
      } else if (status === 'error') {
        settled = true;
        clearInterval(poll);
        rejectRun(errorSink.current ?? new Error('ExperimentRunner entered the error state during the parity gate'));
      }
    }, 1);
    runner.start();
  });

interface Position {
  x: number;
  z: number;
}

const capturePosition = (world: ReadonlyWorldState, agentId: AgentId): Position => {
  const agent = world.agents.find((candidate) => candidate.id === agentId);
  if (!agent) throw new Error(`parity gate: world is missing agent "${agentId}"`);
  return { x: agent.position.x, z: agent.position.z };
};

const toScoreResult = (telemetry: {
  foodPickups: number;
  hazardContacts: number;
  distanceTravelled: number;
  movementScore: number;
}): AgentScoreResult => ({
  foodPickups: telemetry.foodPickups,
  hazardContacts: telemetry.hazardContacts,
  distanceTravelled: telemetry.distanceTravelled,
  movementScore: telemetry.movementScore
});

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
  // Loaded once and shared across every seed's `it` below. Safe: `graph` is
  // read-only static structure (`runEpisode`/`createOracleAgentBinding`
  // each allocate their own fresh, per-call mutable neural state from it),
  // and `buffer` is only ever read via a fresh `buffer.slice(0)` defensive
  // copy per `createOracleAgentBinding` call below, never transferred or
  // mutated in place.
  const { graph, buffer } = loadBiologicalGraph();

  for (const seed of SEEDS) {
    it(
      `seed ${seed}: left (authored) per-tick decoded actions and both arms' final scores match, right (parked) never moves`,
      async () => {
        // --- Path A: the real product `runEpisode` call, instrumented via
        // `onTick` (see this file's module doc for why this replaced an
        // earlier hand-replay approach). ---
        const episodeLeftActions: DecodedAction[] = [];
        const episodeRightActions: DecodedAction[] = [];
        const episodeRightPositions: Position[] = [];
        const episodeResult = runEpisode({
          seed,
          ticks: TICKS,
          substeps: NEURAL_SUBSTEPS_PER_TICK,
          left: { decoder: 'authored', graph },
          right: { decoder: 'parked' },
          onTick: (_tick, actions, world) => {
            episodeLeftActions.push(actions.left);
            episodeRightActions.push(actions.right);
            episodeRightPositions.push(capturePosition(world, 'right'));
          }
        });
        expect(episodeLeftActions).toHaveLength(TICKS);
        // Nontriviality guard: the authored decoder over the real graph must
        // actually move the left agent, or a passing comparison below could
        // just mean both paths independently produced an all-zero run.
        expect(episodeResult.left.distanceTravelled).toBeGreaterThan(0);
        expect(episodeResult.right.distanceTravelled).toBe(0);
        // The right arm's *action* must be exactly zero every tick, not just
        // its resulting position: a yaw-only action ([0, 1, 0]) would leave
        // distanceTravelled at 0 (the agent turns in place) while still
        // being a real regression in "parked". `episodeRightPositions`
        // above corroborates this at the world-state level.
        const zeroAction = decodeAction([0, 0, 0]);
        for (const action of episodeRightActions) {
          expect(action).toEqual(zeroAction);
        }

        // --- Path B: the product closed-loop orchestrator, ExperimentRunner
        // + createOracleAgentBinding, with a test-only zero binding for
        // "parked".
        const leftActions: DecodedAction[] = [];
        const leftBinding = recordingBinding(
          createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' }),
          leftActions
        );
        const rightBinding = createZeroAgentBinding(graph.metadata.neuronCount);
        const rightPositions: Position[] = [];
        const errorSink: { current?: Error } = {};
        const runner = new ExperimentRunner({
          seed,
          totalTicks: TICKS,
          agents: { left: leftBinding, right: rightBinding },
          substepsPerTick: NEURAL_SUBSTEPS_PER_TICK,
          targetTickIntervalMs: 0,
          onError: (error) => {
            errorSink.current = error;
          },
          // `runOneTick` (runner.ts) calls this after `stepWorld` has
          // already applied that tick, and only for a tick that was not
          // discarded by a concurrent reset/dispose (neither happens in
          // this test) -- so one call per applied tick, world already
          // advanced, is exactly what `rightPositions.length` below relies
          // on equaling `TICKS`.
          onTelemetry: () => {
            rightPositions.push(capturePosition(runner.getWorld(), 'right'));
          }
        });
        const initialRightPosition = capturePosition(runner.getWorld(), 'right');
        await runToFinished(runner, errorSink);
        expect(runner.getWorld().tick).toBe(TICKS);

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

        // --- Comparison 2: final score, both arms. ---
        const telemetry = runner.getTelemetry();
        expect(toScoreResult(telemetry.agents.left)).toEqual(episodeResult.left);
        expect(toScoreResult(telemetry.agents.right)).toEqual(episodeResult.right);

        // --- Comparison 3: the right (parked) agent never moves, on either path. ---
        expect(rightPositions).toHaveLength(TICKS);
        for (const position of rightPositions) {
          expect(position).toEqual(initialRightPosition);
        }
        for (const position of episodeRightPositions) {
          expect(position).toEqual(initialRightPosition);
        }
      },
      15_000
    );
  }
});

/**
 * WP6/thermo-architecture review addendum: the authored-decoder describe
 * block above is the pre-existing gate; this second block closes the
 * companion coverage gap a thermo review found (`reviews/feat-nom6-trained-toggle-thermo-2026-09-24-766f09c/thermo-architecture/01-critical-and-important.md`) —
 * this branch's own stated purpose is that the browser's Trained decoder
 * reproduces `episode.ts`'s published numbers, but nothing in the suite
 * checked that end to end: `tests/integration/worker-parity.test.ts`'s
 * trained-decoder coverage compares `handleWorkerRequest` against a direct
 * `runSubsteps` + `readoutForward` computation, not against `episode.ts`,
 * and not routed through `ExperimentRunner`.
 *
 * This block drives the same real production stack the authored block
 * above does — `ExperimentRunner` + `createWorkerAgentBinding` — but backed
 * by a real `WorkerClient` (`createWorkerClient`) wrapping a
 * `FakeNeuralWorker` (`tests/helpers/fake-worker.ts`) instead of the
 * in-thread oracle binding: `createOracleAgentBinding` has no
 * `decoder`/`readout` parameter at all (it always runs the authored path),
 * so a trained comparison needs the same Worker-backed binding
 * `App.svelte` actually uses in production, not a same-thread shortcut.
 * `FakeNeuralWorker` forwards every message to `handleWorkerRequest`
 * (`neural.worker.ts`) — the exact function a real dedicated Worker
 * runs — so this is not a reimplementation of the Worker at any layer.
 *
 * Covers all three arms (biological/rewired/disconnected) — matching
 * `ExperimentController#validateTrainedReadout`'s own all-three-arms
 * validation — across two of the plan's held-out seeds (30001, 30002;
 * `loadTrainedReadoutArtifact`'s real, committed
 * `public/data/trained-readout-v1.json` weights, hash-verified in
 * production but loaded directly from disk here the same way the
 * authored block above loads the graph artifacts directly, since this
 * `@vitest-environment node` file has no `fetch` to stub).
 */

const TRAINED_ARMS: readonly GraphMode[] = ['biological', 'rewired', 'disconnected'];
const TRAINED_SEEDS: readonly number[] = [30001, 30002];

const loadRewiredGraph = (): LoadedGraph => {
  const gzipBytes = readFileSync(resolve(publicDataDir, 'malecns-arena-v1-rewired-seed0.bin.gz'));
  const binary = gunzipSync(gzipBytes);
  const buffer = binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
  return { graph: parseGraphBinary(buffer.slice(0)), buffer };
};

const loadTrainedArtifactJson = (): TrainedReadoutArtifactJson =>
  JSON.parse(readFileSync(resolve(publicDataDir, 'trained-readout-v1.json'), 'utf8')) as TrainedReadoutArtifactJson;

describe('episode.ts runEpisode vs ExperimentRunner: per-tick closed-loop parity (decoder: trained)', () => {
  // Loaded once and shared across every arm/seed `it` below — same safety
  // argument as the authored block's own module-scope `graph`/`buffer`
  // (see that describe block's comment just above it): every graph object
  // here is read-only static structure, and every Worker binding built
  // below gets its own fresh buffer (`buildGraphBufferForMode` always
  // returns a new `ArrayBuffer`; see that function's own doc comment).
  const { graph: biologicalGraph, buffer: biologicalBuffer } = loadBiologicalGraph();
  const { graph: rewiredGraph, buffer: rewiredBuffer } = loadRewiredGraph();
  const disconnectedGraph = createDisconnectedGraph(biologicalGraph);
  const trainedArtifactJson = loadTrainedArtifactJson();

  const graphForArm = (mode: GraphMode): ConnectomeGraph =>
    mode === 'biological' ? biologicalGraph : mode === 'rewired' ? rewiredGraph : disconnectedGraph;

  /**
   * Decoded and graph-validated per arm, mirroring
   * `ExperimentController#validateTrainedReadout`'s own per-arm
   * `validateReadoutWeights` check against each arm's actual parsed graph
   * (not just decoded and trusted blind).
   */
  const weightsForArm = (mode: GraphMode): ReadoutWeights =>
    validateReadoutWeights(decodeReadoutArtifact(trainedArtifactJson, mode), graphForArm(mode));

  /**
   * Builds the left `AgentBinding` the same way `ExperimentController#initialize`
   * does in production: a `WorkerClient` (real request/response matching,
   * not stubbed) wrapping a Worker-shaped stand-in, initialized with this
   * arm's readout weights at `init` (matching `createWorkerAgentBinding`'s
   * own "always passed at init regardless of which decoder is initially
   * active" contract), then switched to Trained via
   * `client.setDecoder('trained')` — the same two-step sequence
   * `ExperimentController#setDecoder` performs against the real arms'
   * Workers.
   */
  const createTrainedWorkerBinding = async (mode: GraphMode): Promise<AgentBinding> => {
    const client = createWorkerClient(new FakeNeuralWorker());
    const graphBuffer = buildGraphBufferForMode(biologicalBuffer, rewiredBuffer, mode);
    const binding = await createWorkerAgentBinding(client, graphBuffer, mode, undefined, weightsForArm(mode));
    await client.setDecoder('trained');
    return binding;
  };

  for (const mode of TRAINED_ARMS) {
    for (const seed of TRAINED_SEEDS) {
      it(
        `arm ${mode}, seed ${seed}: left (trained) per-tick decoded actions and both arms' final scores match, right (parked) never moves`,
        async () => {
          const armGraph = graphForArm(mode);
          const armWeights = weightsForArm(mode);

          // --- Path A: the real product `runEpisode` call, decoder: 'trained'. ---
          const episodeLeftActions: DecodedAction[] = [];
          const episodeRightActions: DecodedAction[] = [];
          const episodeRightPositions: Position[] = [];
          const episodeResult = runEpisode({
            seed,
            ticks: TICKS,
            substeps: NEURAL_SUBSTEPS_PER_TICK,
            left: { decoder: 'trained', graph: armGraph, weights: armWeights },
            right: { decoder: 'parked' },
            onTick: (_tick, actions, world) => {
              episodeLeftActions.push(actions.left);
              episodeRightActions.push(actions.right);
              episodeRightPositions.push(capturePosition(world, 'right'));
            }
          });
          expect(episodeLeftActions).toHaveLength(TICKS);
          expect(episodeResult.right.distanceTravelled).toBe(0);
          const zeroAction = decodeAction([0, 0, 0]);
          for (const action of episodeRightActions) {
            expect(action).toEqual(zeroAction);
          }

          // --- Path B: the product closed-loop orchestrator,
          // ExperimentRunner + a real Worker-protocol-backed binding
          // (createWorkerAgentBinding + WorkerClient + handleWorkerRequest
          // via FakeNeuralWorker), decoder switched to 'trained'. ---
          const leftActions: DecodedAction[] = [];
          const leftBinding = recordingBinding(await createTrainedWorkerBinding(mode), leftActions);
          const rightBinding = createZeroAgentBinding(armGraph.metadata.neuronCount);
          const rightPositions: Position[] = [];
          const errorSink: { current?: Error } = {};
          const runner = new ExperimentRunner({
            seed,
            totalTicks: TICKS,
            agents: { left: leftBinding, right: rightBinding },
            substepsPerTick: NEURAL_SUBSTEPS_PER_TICK,
            targetTickIntervalMs: 0,
            onError: (error) => {
              errorSink.current = error;
            },
            onTelemetry: () => {
              rightPositions.push(capturePosition(runner.getWorld(), 'right'));
            }
          });
          const initialRightPosition = capturePosition(runner.getWorld(), 'right');
          await runToFinished(runner, errorSink);
          expect(runner.getWorld().tick).toBe(TICKS);

          // --- Comparison 1: per-tick left decoded actions. ---
          const divergentTick = firstDivergentTickIndex(episodeLeftActions, leftActions);
          if (divergentTick !== -1) {
            throw new Error(
              `arm ${mode} seed ${seed}: episode.ts and ExperimentRunner (trained) left-agent decoded actions first ` +
                `diverge at tick ${divergentTick} of ${TICKS}: runEpisode=${JSON.stringify(episodeLeftActions[divergentTick])}, ` +
                `ExperimentRunner=${JSON.stringify(leftActions[divergentTick])}`
            );
          }
          expect(leftActions).toHaveLength(TICKS);

          // --- Comparison 2: final score, both arms. ---
          const telemetry = runner.getTelemetry();
          expect(toScoreResult(telemetry.agents.left)).toEqual(episodeResult.left);
          expect(toScoreResult(telemetry.agents.right)).toEqual(episodeResult.right);

          // --- Comparison 3: the right (parked) agent never moves, on either path. ---
          expect(rightPositions).toHaveLength(TICKS);
          for (const position of rightPositions) {
            expect(position).toEqual(initialRightPosition);
          }
          for (const position of episodeRightPositions) {
            expect(position).toEqual(initialRightPosition);
          }
        },
        20_000
      );
    }
  }
});
