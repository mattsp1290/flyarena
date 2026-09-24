import { MAX_SUBSTEPS_PER_TICK } from '../worker/protocol';

/**
 * Neural substeps run per world tick, for every arm. A world tick is
 * `1/30`s of simulated time (`ARENA_CONFIG.fixedDeltaSeconds`); this many
 * `stepModel` calls run against the *same* held-constant observation before
 * the outputs are decoded and the world advances — see
 * `src/lib/connectome/model.ts#runSubsteps`. Chosen as a small constant well
 * under `MAX_SUBSTEPS_PER_TICK` (64): frequent enough re-observation (every
 * 1/30s) matters more for this arena than deep intra-tick integration, and 4
 * keeps worst-case per-tick Worker latency low.
 *
 * Lives here — rather than in `experiment/runner.ts`, which re-exports it
 * for its own existing call sites — so a non-browser consumer (e.g.
 * `scripts/training/export-traces.ts`, which imports this exact value
 * rather than restating it) can resolve one integer constant without also
 * resolving `ExperimentRunner`'s entire module graph (arena world/actions/
 * replay/sensors, the Worker protocol types, Svelte-adjacent orchestration)
 * as a side effect. This module — and its one dependency, the Worker
 * protocol's `MAX_SUBSTEPS_PER_TICK` bound — has no browser-only import at
 * any depth: safe to import from a plain Node/`tsx` script.
 */
export const NEURAL_SUBSTEPS_PER_TICK = 4;

if (NEURAL_SUBSTEPS_PER_TICK > MAX_SUBSTEPS_PER_TICK) {
  throw new Error(
    `NEURAL_SUBSTEPS_PER_TICK (${NEURAL_SUBSTEPS_PER_TICK}) exceeds MAX_SUBSTEPS_PER_TICK (${MAX_SUBSTEPS_PER_TICK})`
  );
}
