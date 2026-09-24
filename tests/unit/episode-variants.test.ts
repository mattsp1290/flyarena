import { describe, expect, it } from 'vitest';

import { decodeAction, OUTPUT_POPULATION } from '../../src/lib/arena/actions';
import { observeAgent } from '../../src/lib/arena/sensors';
import { createWorld, stepWorld } from '../../src/lib/arena/world';
import type { AgentScore } from '../../src/lib/arena/types';
import {
  createModelState,
  createOutputBuffer,
  createStepScratch,
  runSubsteps
} from '../../src/lib/connectome/model';
import { TRACE_SUBSTEPS } from '../../scripts/training/export-traces';
import { runEpisode, type EpisodeDecoderKind } from '../../scripts/training/episode';
import { createTraceGraph } from '../fixtures/trace-graph';

/**
 * Coverage for the `authored-flip-thrust`/`authored-flip-yaw`/
 * `authored-flip-both` decoder variants
 * (`.agents/plans/null-explanation/01-decoder-variants.md` WP1) added to
 * `scripts/training/episode.ts`. `tests/unit/episode.test.ts` already pins
 * the plain `authored` decoder against the committed golden traces --
 * unchanged by this WP, since every `authored-flip-*` branch is additive
 * and `authored` itself flips neither channel.
 */

describe('runEpisode: authored-flip-* decoder variants', () => {
  const TICKS = 12;

  /**
   * Independent, hand-rolled replica of episode.ts's own left-agent tick
   * loop (observeAgent -> runSubsteps -> optional sign flip -> decodeAction
   * -> stepWorld), built directly from the same lower-level primitives
   * `episode.ts` itself calls (`runSubsteps`, `decodeAction`,
   * `OUTPUT_POPULATION`) rather than through `runEpisode`. Comparing this
   * against `runEpisode`'s own `authored-flip-*` output is therefore a real
   * check that episode.ts applies the flip in the right place (after
   * `runSubsteps` fills `outputs`, before `decodeAction`) and to the right
   * indices -- not a tautology against the same code path.
   */
  const runManualFlippedEpisode = (
    seed: number,
    ticks: number,
    flipThrust: boolean,
    flipYaw: boolean
  ): AgentScore => {
    const graph = createTraceGraph();
    const state = createModelState(graph);
    const scratch = createStepScratch(graph);
    const outputs = createOutputBuffer(graph);
    let world = createWorld(seed);
    for (let tick = 0; tick < ticks; tick += 1) {
      const observation = observeAgent(world, 'left');
      runSubsteps(graph, state, scratch, observation, TRACE_SUBSTEPS, outputs);
      if (flipThrust) outputs[OUTPUT_POPULATION.thrust] *= -1;
      if (flipYaw) outputs[OUTPUT_POPULATION.yaw] *= -1;
      const decoded = decodeAction(Array.from(outputs));
      world = stepWorld(world, { left: [decoded.thrust, decoded.yaw, decoded.brake], right: [0, 0, 0] });
    }
    const left = world.agents.find((agent) => agent.id === 'left');
    if (!left) throw new Error('missing left agent');
    return left.score;
  };

  it.each([
    ['authored-flip-both', true, true],
    ['authored-flip-thrust', true, false],
    ['authored-flip-yaw', false, true]
  ] as const satisfies readonly (readonly [EpisodeDecoderKind, boolean, boolean])[])(
    '%s matches an independent flip-then-decode replica over %s ticks',
    (decoder, flipThrust, flipYaw) => {
      const seed = 3;
      const graph = createTraceGraph();
      const result = runEpisode({
        seed,
        ticks: TICKS,
        substeps: TRACE_SUBSTEPS,
        left: { decoder, graph },
        right: { decoder: 'parked' }
      });
      const expected = runManualFlippedEpisode(seed, TICKS, flipThrust, flipYaw);
      expect(result.left).toEqual(expected);
    }
  );

  it('authored-flip-both is not merely a relabeling of authored: their scores diverge over enough ticks', () => {
    const seed = 3;
    const graph = createTraceGraph();
    const authored = runEpisode({
      seed,
      ticks: TICKS,
      substeps: TRACE_SUBSTEPS,
      left: { decoder: 'authored', graph },
      right: { decoder: 'parked' }
    });
    const flipped = runEpisode({
      seed,
      ticks: TICKS,
      substeps: TRACE_SUBSTEPS,
      left: { decoder: 'authored-flip-both', graph },
      right: { decoder: 'parked' }
    });
    expect(flipped.left).not.toEqual(authored.left);
  });

  it('the flip step never touches the brake entry: only OUTPUT_POPULATION.thrust/yaw are negated', () => {
    // Direct check of the transform itself (independent of any particular
    // graph/seed/tick's raw output values), against the same
    // OUTPUT_POPULATION indices episode.ts's flip logic uses.
    const graph = createTraceGraph();
    const world = createWorld(5);
    const state = createModelState(graph);
    const scratch = createStepScratch(graph);
    const rawOutputs = createOutputBuffer(graph);
    const observation = observeAgent(world, 'left');
    runSubsteps(graph, state, scratch, observation, TRACE_SUBSTEPS, rawOutputs);
    const brakeBefore = rawOutputs[OUTPUT_POPULATION.brake];
    expect(brakeBefore).not.toBe(0); // meaningless if brake happens to already be zero here

    for (const [flipThrust, flipYaw] of [
      [true, false],
      [false, true],
      [true, true]
    ] as const) {
      const outputs = Float32Array.from(rawOutputs);
      if (flipThrust) outputs[OUTPUT_POPULATION.thrust] *= -1;
      if (flipYaw) outputs[OUTPUT_POPULATION.yaw] *= -1;
      expect(outputs[OUTPUT_POPULATION.brake]).toBe(brakeBefore);
    }
  });

  it('lesion is still supported for a flip decoder (part of the authored family)', () => {
    const graph = createTraceGraph();
    // Exercises isAuthoredFamily's widened set indirectly: a lesion + a
    // non-authored-family decoder throws (tests/unit/episode-lesion.test.ts
    // covers that for 'trained'/'silenced'/'parked'); this instead confirms
    // a flip variant is accepted, not rejected as if it were one of those.
    expect(() =>
      runEpisode({
        seed: 1,
        ticks: 2,
        substeps: TRACE_SUBSTEPS,
        left: { decoder: 'authored-flip-both', graph, lesion: Int32Array.from([0, 1]) },
        right: { decoder: 'parked' }
      })
    ).not.toThrow();
  });
});
