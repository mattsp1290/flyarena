import { describe, expect, it, vi } from 'vitest';
import { ExperimentController } from '../../src/lib/experiment/controller';
import { createCallbacks, createWorker, SEED, TOTAL_TICKS, useControllerTestLifecycle } from './experiment-controller-test-helpers';

/**
 * WP4 of `.agents/plans/pathway-interventions`: controller coverage for
 * `onPathwayInterventions`, mirroring
 * `tests/unit/experiment-controller-null-explanation.test.ts`'s own
 * structure and reasoning (a maintainability-review finding: the new
 * `onPathwayInterventions` seam claimed the same three guarantees the
 * null-explanation seam already has tests for — sequenced after the prior
 * load settles, isolated from a throwing host callback, and a loader throw
 * mapped to `'unavailable'` rather than an unhandled rejection — but shipped
 * with none of its own).
 *
 * This seam sits one fork deeper than the null-explanation one
 * (`nullLoad -> explanationLoad -> interventionsLoad`), so — unlike that
 * file's own precedent, which can assert synchronously right after
 * `await controller.initialize()` — every assertion here that depends on
 * this chain having actually reached `loadPathwayInterventions` uses
 * `vi.waitFor` rather than assuming it settled within `initialize()`'s own
 * (unrelated) await budget.
 */

const { trackController } = useControllerTestLifecycle();

describe('ExperimentController pathway-interventions loading (WP4 of .agents/plans/pathway-interventions)', () => {
  /**
   * `createPublicDataFetch` serves whichever committed `public/data/*` file
   * matches the requested basename, and `pathway-interventions-v1.json`
   * plus its manifest entry are committed there — so this runs against the
   * real shipped artifact by default.
   */
  it('fires onPathwayInterventions with the real artifact as "ok", after onNullExplanation, without blocking reaching "ready"', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks
    });
    trackController(controller);

    await controller.initialize();

    expect(controller.getRunner()).toBeDefined();
    expect(callbacks.statuses).toContain('ready');
    await vi.waitFor(() => expect(callbacks.pathwayInterventionsResults).toHaveLength(1));
    const result = callbacks.pathwayInterventionsResults[0];
    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.data.authored.category).toBe('pathway-supported');
      expect(result.data.trained.trainedRobust).toBe(true);
    }
  });

  it('is "missing" when the injectable loadPathwayInterventions directly returns a stubbed "missing" result', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadPathwayInterventions: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.pathwayInterventionsResults).toHaveLength(1));
    expect(callbacks.pathwayInterventionsResults[0]).toEqual({ status: 'missing', reason: 'stubbed for this test' });
  });

  it('does not fire onPathwayInterventions once disposed before initialize() resolves at all', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks
    });
    trackController(controller);

    const initializing = controller.initialize();
    controller.dispose();
    await initializing;

    expect(callbacks.pathwayInterventionsResults).toHaveLength(0);
  });

  /** A macrotask flush that drains every queued microtask, regardless of how deep this chain is — same reasoning as the null-explanation test file's own `flush()`. */
  const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

  it('drops a pathway-interventions result that settles after dispose()', async () => {
    let resolveInterventions: ((result: { status: 'missing'; reason: string }) => void) | undefined;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadPathwayInterventions: () =>
        new Promise((resolve) => {
          resolveInterventions = resolve;
        })
    });
    trackController(controller);

    await controller.initialize();
    expect(callbacks.statuses).toContain('ready');
    // Waits for the loader to actually have been invoked (this fork sits one
    // hop past the null-explanation load, which needs its own real
    // fetch/verify round trip here since it is not stubbed) before disposing
    // — otherwise `resolveInterventions` may still be unset.
    await vi.waitFor(() => expect(resolveInterventions).toBeDefined());
    expect(callbacks.pathwayInterventionsResults).toHaveLength(0);

    controller.dispose();
    resolveInterventions?.({ status: 'missing', reason: 'settled after dispose' });
    await flush();

    expect(callbacks.pathwayInterventionsResults).toHaveLength(0);
  });

  /**
   * Positive control for the test above: the identical setup, minus
   * `dispose()`. Proves the wait/flush combination reports the result when
   * nothing should be dropping it.
   */
  it('(positive control) the same settle-after-a-delay setup without dispose() reports the result after flush()', async () => {
    let resolveInterventions: ((result: { status: 'missing'; reason: string }) => void) | undefined;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadPathwayInterventions: () =>
        new Promise((resolve) => {
          resolveInterventions = resolve;
        })
    });
    trackController(controller);

    await controller.initialize();
    await vi.waitFor(() => expect(resolveInterventions).toBeDefined());
    expect(callbacks.pathwayInterventionsResults).toHaveLength(0);

    resolveInterventions?.({ status: 'missing', reason: 'settled without dispose' });
    await flush();

    expect(callbacks.pathwayInterventionsResults).toHaveLength(1);
  });

  it('maps a rejecting loadPathwayInterventions to an "unavailable" onPathwayInterventions result instead of an unhandled rejection', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadPathwayInterventions: async () => {
        throw new Error('boom');
      }
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.pathwayInterventionsResults).toHaveLength(1));
    const result = callbacks.pathwayInterventionsResults[0];
    expect(result.status).toBe('unavailable');
    if (result.status === 'unavailable') expect(result.reason).toMatch(/unexpected error.*boom/);
  });

  /**
   * The pathway-interventions load must not be skipped when the host's own
   * `onNullExplanation` callback throws — it is forked off the settled
   * `explanationLoad` promise, never off the `onNullExplanation` dispatch
   * step itself (`controller.ts`'s own comment on this fork).
   */
  it('still fires onPathwayInterventions even when the onNullExplanation host callback throws', async () => {
    const callbacks = createCallbacks();
    callbacks.onNullExplanation = () => {
      throw new Error('onNullExplanation host callback boom');
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadPathwayInterventions: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.pathwayInterventionsResults).toHaveLength(1));
    expect(callbacks.errors.some((message) => message.includes('onNullExplanation host callback boom'))).toBe(true);
    expect(callbacks.pathwayInterventionsResults[0]).toEqual({ status: 'missing', reason: 'stubbed for this test' });
  });

  it('routes a throwing onPathwayInterventions callback to onError instead of an unhandled rejection', async () => {
    const callbacks = createCallbacks();
    callbacks.onPathwayInterventions = () => {
      throw new Error('host callback boom');
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadPathwayInterventions: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.errors.some((message) => message.includes('host callback boom'))).toBe(true));
  });

  it('still fires onPathwayInterventions (attempted unconditionally) even when the null-explanation load itself fails', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadNullExplanation: async () => ({ status: 'unavailable', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.nullExplanationResults).toHaveLength(1));
    expect(callbacks.nullExplanationResults[0]).toEqual({ status: 'unavailable', reason: 'stubbed for this test' });
    // Real `loadPathwayInterventions` still runs against the real committed
    // artifact here (only `loadNullExplanation` is stubbed above) — its own
    // cross-check re-reads `manifest.nullExplanation.sha256` directly, not
    // the resolved `NullExplanationLoadResult`, so a failed explanation load
    // does not prevent it from resolving "ok".
    await vi.waitFor(() => expect(callbacks.pathwayInterventionsResults).toHaveLength(1));
    expect(callbacks.pathwayInterventionsResults[0].status).toBe('ok');
  });

  /**
   * Parallelism (thermo-maintainability review, Important): the four
   * sidecar loads used to chain each onto the previous one's settled
   * promise purely as an authorial habit, even though none needs another's
   * *result* -- only the manifest already in scope at the top of
   * `initialize()`. `loadPathwayInterventions` is invoked immediately, in
   * the same microtask turn as `loadNullExplanation`, not gated behind it
   * (`controller.ts`'s own `runSidecarLoad` doc comment explains why this
   * fix landed). Holds the explanation load open indefinitely to prove the
   * interventions loader does not (and never did need to) wait for it.
   */
  it('invokes loadPathwayInterventions immediately, without waiting for the explanation load to settle', async () => {
    let interventionsStarted = false;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      // Never resolves -- if `loadPathwayInterventions` were still gated
      // behind this settling, `interventionsStarted` would stay false and
      // `pathwayInterventionsResults` would stay empty forever.
      loadNullExplanation: () => new Promise(() => {}),
      loadPathwayInterventions: async () => {
        interventionsStarted = true;
        return { status: 'missing', reason: 'interventions settles' };
      }
    });
    trackController(controller);

    await controller.initialize();
    await vi.waitFor(() => expect(callbacks.pathwayInterventionsResults).toHaveLength(1));
    expect(interventionsStarted).toBe(true);
  });
});
