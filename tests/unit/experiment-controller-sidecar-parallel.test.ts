import { describe, expect, it, vi } from 'vitest';
import { ExperimentController } from '../../src/lib/experiment/controller';
import { createCallbacks, createWorker, SEED, TOTAL_TICKS, useControllerTestLifecycle } from './experiment-controller-test-helpers';

/**
 * Thermo-maintainability review (Important, I1): `ExperimentController#initialize()`
 * used to chain its four sidecar loads (rewiring-null, null-explanation,
 * pathway-interventions, repertoire-null) in sequence, each waiting on the
 * previous one's settled promise, even though every one of them only needs
 * `artifacts.manifest`/`dataBaseUrl` -- already available the instant the
 * arena artifacts load, not the *result* of any other sidecar load. Fixed
 * by firing all four directly off the manifest in parallel (see
 * `controller.ts`'s `runSidecarLoad` doc comment and each fork's own
 * comment). The per-sidecar test files
 * (`experiment-controller-null-explanation.test.ts`,
 * `experiment-controller-pathway-interventions.test.ts`,
 * `experiment-controller-repertoire-null.test.ts`) each already cover "this
 * one loader still fires when the *specific* loader immediately before it
 * in the old chain fails" -- this file adds the one thing none of those
 * pairwise tests covers: that a single loader failing (throwing) does not
 * block *any* of the other three from firing, all four wired up in the
 * same `initialize()` call at once.
 */

const { trackController } = useControllerTestLifecycle();

describe('ExperimentController: one sidecar loader failing does not block the other three', () => {
  it('still dispatches onRewiringNull/onNullExplanation/onRepertoireNull as "ok" (or their real committed-artifact result) when loadPathwayInterventions throws synchronously', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadPathwayInterventions: () => {
        throw new Error('pathway-interventions loader exploded');
      }
    });
    trackController(controller);

    await controller.initialize();

    // The failing loader is mapped to an honest "unavailable" result, never
    // an unhandled rejection or a hung "loading" state.
    await vi.waitFor(() => expect(callbacks.pathwayInterventionsResults).toHaveLength(1));
    expect(callbacks.pathwayInterventionsResults[0].status).toBe('unavailable');

    // The other three loaders -- fired in parallel, off the same manifest,
    // never chained through the failing one -- all still resolve.
    await vi.waitFor(() => expect(callbacks.rewiringNullResults).toHaveLength(1));
    await vi.waitFor(() => expect(callbacks.nullExplanationResults).toHaveLength(1));
    await vi.waitFor(() => expect(callbacks.repertoireNullResults).toHaveLength(1));
    expect(callbacks.rewiringNullResults[0].status).toBe('ok');
    expect(callbacks.nullExplanationResults[0].status).toBe('ok');
    expect(callbacks.repertoireNullResults[0].status).toBe('ok');
  });

  it('all four sidecar loaders are invoked in the same microtask turn (no fork waits on another to even start)', async () => {
    const started: string[] = [];
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadRewiringNull: async (manifest, dataBaseUrl) => {
        started.push('rewiringNull');
        const { loadRewiringNull } = await import('../../src/lib/experiment/rewiringNull');
        return loadRewiringNull(manifest, dataBaseUrl);
      },
      loadNullExplanation: async (manifest, dataBaseUrl) => {
        started.push('nullExplanation');
        const { loadNullExplanation } = await import('../../src/lib/experiment/nullExplanation');
        return loadNullExplanation(manifest, dataBaseUrl);
      },
      loadPathwayInterventions: async (manifest, dataBaseUrl) => {
        started.push('pathwayInterventions');
        const { loadPathwayInterventions } = await import('../../src/lib/experiment/pathwayInterventions');
        return loadPathwayInterventions(manifest, dataBaseUrl);
      },
      loadRepertoireNull: async (manifest, dataBaseUrl) => {
        started.push('repertoireNull');
        const { loadRepertoireNull } = await import('../../src/lib/experiment/repertoireNull');
        return loadRepertoireNull(manifest, dataBaseUrl);
      }
    });
    trackController(controller);

    await controller.initialize();

    // All four were already invoked by the time `initialize()`'s own
    // returned promise resolves -- if any fork were still chained behind
    // another (the old sequential shape), the later ones would not have
    // started yet at this point (they only would have after that earlier
    // fork's own fetch/verify round trip settled, well after `initialize()`
    // itself returns).
    expect(started.sort()).toEqual(['nullExplanation', 'pathwayInterventions', 'repertoireNull', 'rewiringNull']);
  });
});
