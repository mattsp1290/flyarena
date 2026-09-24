// @vitest-environment node
import { describe, expect, it } from 'vitest';

/**
 * `src/lib/connectome/constants.ts`'s own doc comment claims it "has no
 * browser-only import at any depth: safe to import from a plain Node/`tsx`
 * script" (e.g. `scripts/training/export-traces.ts`, and now
 * `scripts/experiments/seed-sweep.ts`). The project's default Vitest
 * environment is `jsdom` (see `vite.config.ts`), which silently provides
 * `window`/`document`/etc. even to a module that should never need them, so
 * running this exact assertion under jsdom would never actually exercise the
 * claim — a real accidental browser-only import (directly, or transitively
 * through `../worker/protocol`) could ship unnoticed. This file overrides
 * the environment for itself only (`@vitest-environment node`, Vitest's
 * documented per-file environment directive) so the import genuinely runs
 * under plain Node with no DOM globals at all.
 *
 * One limit on how far this proves "the same environment `tsx` gives a CLI
 * script": `vite.config.ts`'s `resolve.conditions: ['browser']` still
 * applies to every test file this project runs, including this one — a
 * config-level module-resolution setting, not a DOM global, so it isn't
 * something the `@vitest-environment node` directive above overrides or
 * that the window/document check below can detect. If a dependency ever had
 * separate `"browser"`/`"node"`/`"default"` export-map entries, this test
 * would still resolve the `"browser"` one, while `tsx` (no such condition)
 * would resolve differently. Not a real risk for the two modules this file
 * actually imports (`connectome/constants.ts` and its one dependency,
 * `worker/protocol.ts`, are both plain first-party TS with no conditional
 * exports of their own), but worth knowing if this pattern is reused for a
 * module with a real npm dependency.
 */
describe('connectome/constants.ts (plain Node import, no jsdom)', () => {
  it('has no window/document global in this environment (sanity check that the directive above took effect)', () => {
    expect(typeof window).toBe('undefined');
    expect(typeof document).toBe('undefined');
  });

  it('imports cleanly under plain Node and exposes a valid NEURAL_SUBSTEPS_PER_TICK', async () => {
    const { NEURAL_SUBSTEPS_PER_TICK } = await import('../../src/lib/connectome/constants');
    expect(Number.isInteger(NEURAL_SUBSTEPS_PER_TICK)).toBe(true);
    expect(NEURAL_SUBSTEPS_PER_TICK).toBeGreaterThan(0);
  });

  it("its one dependency (worker/protocol.ts's MAX_SUBSTEPS_PER_TICK) also imports cleanly under plain Node", async () => {
    const { MAX_SUBSTEPS_PER_TICK } = await import('../../src/lib/worker/protocol');
    const { NEURAL_SUBSTEPS_PER_TICK } = await import('../../src/lib/connectome/constants');
    expect(NEURAL_SUBSTEPS_PER_TICK).toBeLessThanOrEqual(MAX_SUBSTEPS_PER_TICK);
  });
});
