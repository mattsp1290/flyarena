import { beforeAll, describe, expect, it, vi } from 'vitest';

import { decodeAction } from '../../src/lib/arena/actions';
import { createWorld } from '../../src/lib/arena/world';
import type { AgentScore, DecodedAction } from '../../src/lib/arena/types';
import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import type { ConnectomeGraph } from '../../src/lib/connectome/format';
import { outputNeuronIndices } from '../../src/lib/connectome/readout';
import { createBranch, stepBranch } from '../../src/lib/counterfactual/engine';
import { prepareGraph, type PreparedGraph } from '../../src/lib/counterfactual/targets';
import { loadArenaArtifacts } from '../../src/lib/experiment/assets';
import { runEpisode } from '../../scripts/training/episode';
import { createPublicDataFetch } from '../helpers/fake-worker';
import { createTraceGraph } from '../fixtures/trace-graph';

/**
 * WP1 (`.agents/plans/lesion-atlas/01-lesion-episodes.md`): proves
 * `runEpisode`'s new `left.lesion`/`right.lesion` option reproduces the
 * counterfactual workbench's lesion semantics
 * (`src/lib/counterfactual/engine.ts` `stepBranch`) exactly -- same
 * clamping point (rates zeroed before the substep loop and after every
 * substep, before `aggregateOutputs`), same value (zero) -- on the real
 * biological artifact, from tick 0 with no warmup fork (`stepBranch` driven
 * directly from a fresh `createWorld(seed)`, matching this WP's "drive
 * `stepBranch` directly from a fresh world in the test" fallback).
 *
 * `prepareGraph`'s own `targets` array (not a hand-rolled reimplementation
 * of `TARGET_LABELS`' input/output/bridge filters) supplies the `output`,
 * `input-0`, and `bridge` neuron-index groups, so this test can never drift
 * from what the engine itself would compute for the same graph.
 */

const TICKS = 300;
const SEEDS: readonly number[] = [30001, 30002, 30003];

let prepared: PreparedGraph;

// Stubbed once for the whole file (never unstubbed -- no other test here
// needs the real `fetch`, and vite.config.ts sets no `restoreMocks`/
// `unstubGlobals` option that would tear it down between `it`s), matching
// how `loadArenaArtifacts`/`prepareGraph` are always driven in these tests:
// through the real production loader against the real committed artifact
// under `public/data`, not a hand-rolled fixture graph.
beforeAll(async () => {
  vi.stubGlobal('fetch', createPublicDataFetch());
  const assets = await loadArenaArtifacts('/data');
  prepared = await prepareGraph(assets, 'biological');
});

/** Decoded action + final left score for `runEpisode` with an authored lesion. */
const runEpisodeLesion = (
  graph: Readonly<ConnectomeGraph>,
  seed: number,
  ticks: number,
  lesion: Int32Array
): { actions: DecodedAction[]; score: AgentScore } => {
  const actions: DecodedAction[] = [];
  const result = runEpisode({
    seed,
    ticks,
    substeps: NEURAL_SUBSTEPS_PER_TICK,
    left: { decoder: 'authored', graph, lesion },
    right: { decoder: 'parked' },
    onTick: (_tick, tickActions) => {
      actions.push(tickActions.left);
    }
  });
  return { actions, score: result.left };
};

/**
 * The counterfactual engine's own lesion branch, driven directly (not
 * through `runExperiment`/`Request` validation -- see this file's module
 * doc): `stepBranch`'s internal `aggregateOutputs` -> `decodeAction` writes
 * `branch.outputs` before `stepWorld`, so re-decoding `branch.outputs`
 * immediately after each `stepBranch` call recovers the exact same action
 * that call fed into `stepWorld`, without reimplementing any of
 * `stepBranch`'s internals.
 */
const runEngineLesion = (
  graph: ConnectomeGraph,
  seed: number,
  ticks: number,
  target: readonly number[]
): { actions: DecodedAction[]; score: AgentScore } => {
  const branch = createBranch(graph, createWorld(seed));
  const actions: DecodedAction[] = [];
  for (let tick = 0; tick < ticks; tick += 1) {
    stepBranch(graph, branch, target);
    actions.push(decodeAction(Array.from(branch.outputs)));
  }
  const left = branch.world.agents.find((agent) => agent.id === 'left');
  if (!left) throw new Error('missing left agent');
  return { actions, score: left.score };
};

describe('runEpisode lesion parity vs the counterfactual engine (real biological artifact)', () => {
  for (const targetId of ['output', 'input-0'] as const) {
    for (const seed of SEEDS) {
      it(
        `target "${targetId}", seed ${seed}: per-tick decoded actions and final score match stepBranch over ${TICKS} ticks`,
        () => {
          const target = prepared.targets.find((candidate) => candidate.id === targetId);
          if (!target) throw new Error(`missing target "${targetId}"`);
          expect(target.indices.length).toBeGreaterThan(0);
          const lesion = Int32Array.from(target.indices);

          const episode = runEpisodeLesion(prepared.graph, seed, TICKS, lesion);
          const engine = runEngineLesion(prepared.graph, seed, TICKS, target.indices);

          expect(episode.actions).toHaveLength(TICKS);
          expect(episode.actions).toEqual(engine.actions);
          expect(episode.score).toEqual(engine.score);
        },
        15_000
      );
    }
  }
});

describe('runEpisode lesion parity vs the counterfactual engine: singleton lesions (production shape)', () => {
  const seed = SEEDS[0];

  it('one output neuron lesioned alone matches stepBranch with a one-element target', () => {
    const outputTarget = prepared.targets.find((candidate) => candidate.id === 'output');
    if (!outputTarget) throw new Error('missing output target');
    const index = outputTarget.indices[0];
    const lesion = Int32Array.from([index]);

    const episode = runEpisodeLesion(prepared.graph, seed, TICKS, lesion);
    const engine = runEngineLesion(prepared.graph, seed, TICKS, [index]);

    expect(episode.actions).toEqual(engine.actions);
    expect(episode.score).toEqual(engine.score);
  }, 15_000);

  it('one bridge neuron lesioned alone matches stepBranch with a one-element target', () => {
    const bridgeTarget = prepared.targets.find((candidate) => candidate.id === 'bridge');
    if (!bridgeTarget) throw new Error('missing bridge target');
    const index = bridgeTarget.indices[0];
    const lesion = Int32Array.from([index]);

    const episode = runEpisodeLesion(prepared.graph, seed, TICKS, lesion);
    const engine = runEngineLesion(prepared.graph, seed, TICKS, [index]);

    expect(episode.actions).toEqual(engine.actions);
    expect(episode.score).toEqual(engine.score);
  }, 15_000);
});

describe('runEpisode lesion: decoder gating', () => {
  const graph = createTraceGraph();

  it('throws when combined with trained', () => {
    expect(() =>
      runEpisode({
        seed: 1,
        ticks: 1,
        left: { decoder: 'trained', lesion: Int32Array.from([0]) },
        right: { decoder: 'parked' }
      })
    ).toThrow(/does not support lesion/);
  });

  it('throws when combined with silenced', () => {
    expect(() =>
      runEpisode({
        seed: 1,
        ticks: 1,
        left: { decoder: 'silenced', lesion: Int32Array.from([0]) },
        right: { decoder: 'parked' }
      })
    ).toThrow(/does not support lesion/);
  });

  it('throws when combined with parked', () => {
    expect(() =>
      runEpisode({
        seed: 1,
        ticks: 1,
        left: { decoder: 'parked', lesion: Int32Array.from([0]) },
        right: { decoder: 'parked' }
      })
    ).toThrow(/does not support lesion/);
  });

  it('throws on an out-of-range lesion index', () => {
    expect(() =>
      runEpisode({
        seed: 1,
        ticks: 1,
        left: { decoder: 'authored', graph, lesion: Int32Array.from([graph.metadata.neuronCount]) },
        right: { decoder: 'parked' }
      })
    ).toThrow(/out of range/);
  });

  it('throws on a negative lesion index', () => {
    expect(() =>
      runEpisode({
        seed: 1,
        ticks: 1,
        left: { decoder: 'authored', graph, lesion: Int32Array.from([-1]) },
        right: { decoder: 'parked' }
      })
    ).toThrow(/out of range/);
  });

  it('throws on a duplicate lesion index', () => {
    expect(() =>
      runEpisode({
        seed: 1,
        ticks: 1,
        left: { decoder: 'authored', graph, lesion: Int32Array.from([2, 2]) },
        right: { decoder: 'parked' }
      })
    ).toThrow(/sorted ascending and unique/);
  });

  it('throws on an unsorted lesion (descending pair)', () => {
    expect(() =>
      runEpisode({
        seed: 1,
        ticks: 1,
        left: { decoder: 'authored', graph, lesion: Int32Array.from([3, 1]) },
        right: { decoder: 'parked' }
      })
    ).toThrow(/sorted ascending and unique/);
  });
});

describe('runEpisode lesion: no-op and full-population edge cases (trace graph)', () => {
  const graph = createTraceGraph();

  it('an empty lesion array equals no lesion, bit for bit', () => {
    const withoutLesion = runEpisode({
      seed: 5,
      ticks: 20,
      left: { decoder: 'authored', graph },
      right: { decoder: 'parked' }
    });
    const withEmptyLesion = runEpisode({
      seed: 5,
      ticks: 20,
      left: { decoder: 'authored', graph, lesion: new Int32Array(0) },
      right: { decoder: 'parked' }
    });
    expect(withEmptyLesion).toEqual(withoutLesion);
  });

  it('lesioning every output neuron gives a zero thrust/yaw/brake action every tick', () => {
    const lesion = outputNeuronIndices(graph);
    expect(lesion.length).toBeGreaterThan(0);
    const actions: DecodedAction[] = [];
    const result = runEpisode({
      seed: 9,
      ticks: 30,
      left: { decoder: 'authored', graph, lesion },
      right: { decoder: 'parked' },
      onTick: (_tick, tickActions) => actions.push(tickActions.left)
    });
    expect(actions).toHaveLength(30);
    for (const action of actions) {
      expect(action).toEqual({ thrust: 0, yaw: 0, brake: 0 });
    }
    // Every output-assigned neuron's contribution is zeroed every substep,
    // so the aggregated output -- and therefore the score -- never moves
    // the agent via its own decoded action (any nonzero score here would
    // have to come from spawn defaults, not from this run).
    expect(result.left.distanceTravelled).toBe(0);
  });
});
