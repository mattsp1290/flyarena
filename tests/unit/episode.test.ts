import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { decodeAction } from '../../src/lib/arena/actions';
import { createWorld, stepWorld } from '../../src/lib/arena/world';
import type { AgentScore } from '../../src/lib/arena/types';
import {
  createReadoutOutput,
  createReadoutScratch,
  outputNeuronIndices,
  readoutForward,
  type ReadoutWeights
} from '../../src/lib/connectome/readout';
import { DEFAULT_GRAPH_ID, TRACE_SEEDS, TRACE_SUBSTEPS, TRACE_TICKS } from '../../scripts/training/export-traces';
import { runEpisode } from '../../scripts/training/episode';
import { createTraceGraph } from '../fixtures/trace-graph';
import {
  diffCloseEnough,
  FLOAT_ABS_TOLERANCE,
  FLOAT_REL_TOLERANCE,
  GOLDEN_GENERATING_ARCH
} from '../fixtures/cross-arch-tolerance';

const GOLDEN_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../fixtures/golden');

/**
 * The committed golden trace files (`tests/fixtures/golden/`) do not record
 * a per-tick score column (see `export-traces.ts`'s `SeedTraceFile` doc
 * comment): they record `initialWorld` (tick 0) and, per tick, the exact
 * decoded `actions` fed into `stepWorld`. Replaying those actions through
 * `stepWorld`, starting from a fresh `createWorld(seed)` (which is exactly
 * what produced the committed `initialWorld` — see `buildSeedTrace`), is
 * therefore the independent ground truth for "the golden trace's score at
 * the final recorded tick": it uses only `stepWorld` directly on the
 * committed action record, never `episode.ts`'s own observe/model/decode
 * computation.
 *
 * "The final recorded tick" is `TRACE_TICKS` = 60, not the 300 named in
 * `.agents/plans/trained-readout/04-authoritative-evaluation-and-artifacts.md`'s
 * acceptance text ("reproduces the golden trace's score at tick 300
 * exactly"). That plan text is stale, not this test: `export-traces.ts`'s
 * `TRACE_TICKS` doc comment explains the reduction from the plan's stated
 * 300-tick default to 60 (inherited from WP1, predates this branch) —
 * dropping to 60 ticks was the byte-budget-driven, plan-authorized
 * reduction that keeps all four golden seeds within the ≤ 200 KB committed
 * fixture budget. Do not read this file and assume `episode.ts` needs to
 * run 300 ticks against a 60-tick golden fixture; 60 is deliberate and
 * matches every currently committed fixture.
 *
 * Cross-architecture note (see `tests/fixtures/cross-arch-tolerance.ts` and
 * `docs/architecture.md`'s "Determinism scope"): `trace.actions` was
 * recorded on `GOLDEN_GENERATING_ARCH`, but `result.left` below comes from
 * a fresh `runEpisode` call on whatever architecture the test runs on.
 * Those two can differ at the float64 ULP level the same way
 * `golden-traces.test.ts`'s byte-for-byte check can — see that file's doc
 * comment for the measured divergence — so the comparison below is exact
 * only on `GOLDEN_GENERATING_ARCH` and tolerance-based everywhere else.
 *
 * Where the actual cross-arch risk is here, precisely: both
 * `goldenFinalLeftScore` and `runEpisode` call `stepWorld` on *this* run's
 * own architecture — `stepWorld` itself is not being compared across
 * machines, so its `Math.sin`/`Math.cos`/`Math.hypot` calls (`arena/world.ts`)
 * are not a source of divergence *within this comparison* the way they are
 * for `golden-traces.test.ts`'s cross-machine byte comparison. The only
 * cross-arch input is `trace.actions` itself: static committed numbers,
 * recorded on `GOLDEN_GENERATING_ARCH`, fed into `goldenFinalLeftScore`'s
 * local `stepWorld` replay, versus `result.left`'s action values, freshly
 * decoded from this run's own `observeAgent` -> model -> `decodeAction`
 * pipeline. That pipeline's `observeAgent` reads the *local* world's
 * position/heading -- which the local `createWorld`/`stepWorld` produced
 * using this run's own `Math.sin`/`Math.cos`/`Math.hypot` results, not
 * `GOLDEN_GENERATING_ARCH`'s -- so `world.ts` divergence in this run's own
 * trajectory reaches `runEpisode`'s fresh observations exactly as it does
 * for `golden-traces.test.ts`, not just `sensors.ts`'s direct calls; it
 * only stops mattering once decoded into `outputs`, which rounds to
 * `Float32Array` at `state.rate`/`outputs` before `decodeAction` (unlike
 * `stepWorld`'s own float64 state). So a cross-arch flip here requires a
 * `sensors.ts`- or `world.ts`-driven divergence in the fresh run's own
 * trajectory large enough to cross a `Float32Array` rounding boundary in
 * `outputs` -- narrower than `golden-traces.test.ts`'s exposure (which
 * also directly records raw float64 `observations`, with no rounding
 * boundary to cross), and consistent with this comparison matching
 * exactly (0 mismatches, every committed seed) on the real x86_64 CI
 * runner even before this file's tolerance fallback existed (measured
 * during the `fix/golden-cross-arch` PR). Still not a structural
 * guarantee for a future fixture refresh; see `cross-arch-tolerance.ts`'s
 * doc comment for the caveat and the `LEAF_NOISE_FLOOR_REL`/
 * `LEAF_NOISE_FLOOR_ABS` mechanism, which this file does not use (an
 * `AgentScore` has only 2 float leaves, too few for a leaf-count budget to
 * add meaningful protection over the tolerance check alone).
 */
const goldenFinalLeftScore = (seed: number): AgentScore => {
  const trace = JSON.parse(
    readFileSync(resolve(GOLDEN_DIR, `${DEFAULT_GRAPH_ID}-seed-${seed}.json`), 'utf8')
  ) as { actions: number[][] };
  let world = createWorld(seed);
  for (const action of trace.actions) {
    world = stepWorld(world, { left: action, right: [0, 0, 0] });
  }
  const left = world.agents.find((agent) => agent.id === 'left');
  if (!left) throw new Error('missing left agent');
  return left.score;
};

describe('runEpisode: authored decoder vs golden traces', () => {
  // `it.each` with a flat array of primitives passes exactly one value
  // (`seed`) to the callback, so a title with two `%d` placeholders (the
  // pre-existing version of this string) silently renders its second
  // placeholder as "NaN" -- there is no second argument to fill it. Use one
  // placeholder and inline TRACE_TICKS instead.
  it.each(TRACE_SEEDS)(
    `reproduces the golden trace score at tick ${TRACE_TICKS} seed %d ` +
      `(exact on ${GOLDEN_GENERATING_ARCH}, within cross-arch tolerance elsewhere)`,
    (seed) => {
      const graph = createTraceGraph();
      const result = runEpisode({
        seed,
        ticks: TRACE_TICKS,
        substeps: TRACE_SUBSTEPS,
        left: { decoder: 'authored', graph },
        right: { decoder: 'parked' }
      });

      expect(result.ticks).toBe(TRACE_TICKS);

      const expectedScore = goldenFinalLeftScore(seed);
      if (process.arch === GOLDEN_GENERATING_ARCH) {
        expect(result.left).toEqual(expectedScore);
      } else {
        const { mismatches } = diffCloseEnough(expectedScore, result.left, `seed-${seed}.left`);
        expect(
          mismatches,
          `seed ${seed}: runEpisode's left score differs from the golden-actions replay beyond ` +
            `cross-arch float tolerance (process.arch=${process.arch}, fixtures generated on ` +
            `${GOLDEN_GENERATING_ARCH}, abs<=${FLOAT_ABS_TOLERANCE} or rel<=${FLOAT_REL_TOLERANCE}):\n` +
            mismatches.join('\n')
        ).toEqual([]);
      }
    }
  );
});

describe('runEpisode: decoder behavior', () => {
  it('parked never accelerates: the parked agent needs no graph and its score stays at its spawn defaults', () => {
    const graph = createTraceGraph();
    const result = runEpisode({
      seed: 1,
      ticks: 30,
      substeps: TRACE_SUBSTEPS,
      left: { decoder: 'authored', graph },
      right: { decoder: 'parked' }
    });
    // A parked agent that never moves cannot accumulate movement score or
    // pick up food; only a spawn-time hazard overlap (not the case for this
    // seed/config) could change hazardContacts before it ever accelerates.
    expect(result.right.movementScore).toBe(0);
    expect(result.right.foodPickups).toBe(0);
  });

  it('rejects a non-parked agent with no graph', () => {
    expect(() =>
      runEpisode({
        seed: 1,
        ticks: 1,
        substeps: TRACE_SUBSTEPS,
        left: { decoder: 'authored' },
        right: { decoder: 'parked' }
      })
    ).toThrow(/requires a graph/);
  });

  it('rejects a trained/silenced agent with no weights', () => {
    const graph = createTraceGraph();
    expect(() =>
      runEpisode({
        seed: 1,
        ticks: 1,
        substeps: TRACE_SUBSTEPS,
        left: { decoder: 'trained', graph },
        right: { decoder: 'parked' }
      })
    ).toThrow(/requires weights/);
  });

  it('silenced decodes the same constant action every tick (readoutForward on an all-zero input)', () => {
    // episode.ts's 'silenced' decoder feeds readoutForward an all-zero
    // gathered input every tick regardless of the network's real state, so
    // hidden = tanh(b1) and the decoded action are constant across ticks —
    // this is what makes it a circuit-silenced control rather than a
    // relabeling of 'trained'. Verified directly here (not just by
    // determinism): manually compute the one constant decoded action via
    // readoutForward on a zero rate vector, replay it through stepWorld by
    // hand for every tick, and confirm that matches runEpisode's 'silenced'
    // result exactly.
    const graph = createTraceGraph();
    const indices = outputNeuronIndices(graph);
    const D = indices.length;
    const H = 4;
    const weights: ReadoutWeights = {
      inputSize: D,
      hiddenSize: H,
      w1: Float32Array.from({ length: H * D }, (_, i) => 0.05 * ((i % 5) - 2)),
      b1: Float32Array.from({ length: H }, (_, i) => 0.1 * (i - 1)),
      w2: Float32Array.from({ length: 3 * H }, (_, i) => 0.05 * ((i % 3) - 1)),
      b2: Float32Array.from([0.2, -0.1, 0.05])
    };

    const zeroRate = new Float32Array(graph.metadata.neuronCount);
    const scratch = createReadoutScratch(H);
    const out = createReadoutOutput();
    readoutForward(weights, zeroRate, indices, scratch, out);
    const decoded = decodeAction(Array.from(out));
    const constantAction: readonly [number, number, number] = [decoded.thrust, decoded.yaw, decoded.brake];
    // A non-trivial constant action: the test is meaningless if it happens
    // to decode to the zero action.
    expect(constantAction.some((value) => value !== 0)).toBe(true);

    const ticks = 8;
    let world = createWorld(7);
    for (let tick = 0; tick < ticks; tick += 1) {
      world = stepWorld(world, { left: constantAction, right: [0, 0, 0] });
    }
    const expectedLeft = world.agents.find((agent) => agent.id === 'left');
    if (!expectedLeft) throw new Error('missing left agent');

    const result = runEpisode({
      seed: 7,
      ticks,
      substeps: TRACE_SUBSTEPS,
      left: { decoder: 'silenced', graph, weights },
      right: { decoder: 'parked' }
    });

    expect(result.left).toEqual(expectedLeft.score);
  });
});
