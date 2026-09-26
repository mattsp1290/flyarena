import { describe, expect, it, vi } from 'vitest';
import { ExperimentController } from '../../src/lib/experiment/controller';
import { createCallbacks, createWorker, SEED, TOTAL_TICKS, useControllerTestLifecycle } from './experiment-controller-test-helpers';

/**
 * WP3 of `.agents/plans/repertoire-null`, wired per `findings-tour`'s own
 * `01-findings-panel.md` ("Add a repertoire load behind the `destroyed`
 * guard, with an injectable seam, following `loadPathwayInterventions`"):
 * controller coverage for `onRepertoireNull`, mirroring
 * `tests/unit/experiment-controller-pathway-interventions.test.ts`'s own
 * structure and reasoning (a maintainability review, Important: the new
 * `onRepertoireNull` seam's doc comments claim the same three guarantees
 * every earlier sidecar seam already has tests for — sequenced after the
 * prior load settles, isolated from a throwing host callback, and a loader
 * throw mapped to `'unavailable'` rather than an unhandled rejection — but
 * shipped with none of its own, and `experiment-controller-test-helpers.ts`'s
 * own `repertoireNullResults`/`onRepertoireNull` additions went unused by
 * any test).
 *
 * This seam sits one fork deeper than the pathway-interventions one
 * (`nullLoad -> explanationLoad -> interventionsLoad -> repertoireLoad`), so
 * every assertion here that depends on this chain having actually reached
 * `loadRepertoireNull` uses `vi.waitFor` rather than assuming it settled
 * within `initialize()`'s own (unrelated) await budget.
 */

const { trackController } = useControllerTestLifecycle();

describe('ExperimentController repertoire-null loading (WP3 of .agents/plans/repertoire-null)', () => {
  /**
   * `createPublicDataFetch` serves whichever committed `public/data/*` file
   * matches the requested basename, and `behavior-repertoire-null-v1.json`
   * plus its manifest entry are committed there — so this runs against the
   * real shipped artifact by default.
   */
  it('fires onRepertoireNull with the real artifact as "ok", after onPathwayInterventions, without blocking reaching "ready"', async () => {
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
    await vi.waitFor(() => expect(callbacks.repertoireNullResults).toHaveLength(1));
    const result = callbacks.repertoireNullResults[0];
    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      // The real shipped result (independently verified, `.agents/plans/repertoire-null/00-overview.md`'s worked example).
      expect(result.data.primary.category).toBe('narrower');
      expect(result.data.robustness.robust).toBe(false);
    }
  });

  it('is "missing" when the injectable loadRepertoireNull directly returns a stubbed "missing" result', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadRepertoireNull: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.repertoireNullResults).toHaveLength(1));
    expect(callbacks.repertoireNullResults[0]).toEqual({ status: 'missing', reason: 'stubbed for this test' });
  });

  it('does not fire onRepertoireNull once disposed before initialize() resolves at all', async () => {
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

    expect(callbacks.repertoireNullResults).toHaveLength(0);
  });

  /** A macrotask flush that drains every queued microtask, regardless of how deep this chain is — same reasoning as the pathway-interventions test file's own `flush()`. */
  const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

  it('drops a repertoire-null result that settles after dispose()', async () => {
    let resolveRepertoire: ((result: { status: 'missing'; reason: string }) => void) | undefined;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadRepertoireNull: () =>
        new Promise((resolve) => {
          resolveRepertoire = resolve;
        })
    });
    trackController(controller);

    await controller.initialize();
    expect(callbacks.statuses).toContain('ready');
    // Waits for the loader to actually have been invoked (this fork sits one
    // hop past the pathway-interventions load, which needs its own real
    // fetch/verify round trip here since it is not stubbed) before disposing
    // — otherwise `resolveRepertoire` may still be unset.
    await vi.waitFor(() => expect(resolveRepertoire).toBeDefined());
    expect(callbacks.repertoireNullResults).toHaveLength(0);

    controller.dispose();
    resolveRepertoire?.({ status: 'missing', reason: 'settled after dispose' });
    await flush();

    expect(callbacks.repertoireNullResults).toHaveLength(0);
  });

  /**
   * Positive control for the test above: the identical setup, minus
   * `dispose()`. Proves the wait/flush combination reports the result when
   * nothing should be dropping it.
   */
  it('(positive control) the same settle-after-a-delay setup without dispose() reports the result after flush()', async () => {
    let resolveRepertoire: ((result: { status: 'missing'; reason: string }) => void) | undefined;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadRepertoireNull: () =>
        new Promise((resolve) => {
          resolveRepertoire = resolve;
        })
    });
    trackController(controller);

    await controller.initialize();
    await vi.waitFor(() => expect(resolveRepertoire).toBeDefined());
    expect(callbacks.repertoireNullResults).toHaveLength(0);

    resolveRepertoire?.({ status: 'missing', reason: 'settled without dispose' });
    await flush();

    expect(callbacks.repertoireNullResults).toHaveLength(1);
  });

  it('maps a rejecting loadRepertoireNull to an "unavailable" onRepertoireNull result instead of an unhandled rejection', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadRepertoireNull: async () => {
        throw new Error('boom');
      }
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.repertoireNullResults).toHaveLength(1));
    const result = callbacks.repertoireNullResults[0];
    expect(result.status).toBe('unavailable');
    if (result.status === 'unavailable') expect(result.reason).toMatch(/unexpected error.*boom/);
  });

  /**
   * The repertoire-null load must not be skipped when the host's own
   * `onPathwayInterventions` callback throws — it is forked off the settled
   * `interventionsLoad` promise, never off the `onPathwayInterventions`
   * dispatch step itself (`controller.ts`'s own comment on this fork).
   */
  it('still fires onRepertoireNull even when the onPathwayInterventions host callback throws', async () => {
    const callbacks = createCallbacks();
    callbacks.onPathwayInterventions = () => {
      throw new Error('onPathwayInterventions host callback boom');
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadRepertoireNull: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.repertoireNullResults).toHaveLength(1));
    expect(callbacks.errors.some((message) => message.includes('onPathwayInterventions host callback boom'))).toBe(true);
    expect(callbacks.repertoireNullResults[0]).toEqual({ status: 'missing', reason: 'stubbed for this test' });
  });

  it('routes a throwing onRepertoireNull callback to onError instead of an unhandled rejection', async () => {
    const callbacks = createCallbacks();
    callbacks.onRepertoireNull = () => {
      throw new Error('host callback boom');
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadRepertoireNull: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.errors.some((message) => message.includes('host callback boom'))).toBe(true));
  });

  it('still fires onRepertoireNull (attempted unconditionally) even when the pathway-interventions load itself fails', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadPathwayInterventions: async () => ({ status: 'unavailable', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.pathwayInterventionsResults).toHaveLength(1));
    expect(callbacks.pathwayInterventionsResults[0]).toEqual({ status: 'unavailable', reason: 'stubbed for this test' });
    // Real `loadRepertoireNull` still runs against the real committed
    // artifact here (only `loadPathwayInterventions` is stubbed above) — its
    // own cross-check re-reads `manifest.binarySha256`/`manifest.rewiringNull.sha256`
    // directly, not the resolved `PathwayInterventionsLoadResult`, so a
    // failed interventions load does not prevent it from resolving "ok".
    await vi.waitFor(() => expect(callbacks.repertoireNullResults).toHaveLength(1));
    expect(callbacks.repertoireNullResults[0].status).toBe('ok');
  });

  /**
   * Ordering: `loadRepertoireNull` must not be invoked until the
   * interventions load's own promise has settled. Waits for the
   * interventions loader to have actually been *called* (a real causal
   * signal, not a guessed number of microtask hops) before asserting the
   * repertoire-null loader has not been called yet, then releases the
   * interventions loader and confirms the repertoire-null loader runs only
   * after that.
   */
  it('does not invoke loadRepertoireNull before the pathway-interventions load settles', async () => {
    let resolveInterventions: (() => void) | undefined;
    let interventionsStarted = false;
    let repertoireStarted = false;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadPathwayInterventions: () => {
        interventionsStarted = true;
        return new Promise((resolve) => {
          resolveInterventions = () => resolve({ status: 'missing', reason: 'interventions settles now' });
        });
      },
      loadRepertoireNull: async () => {
        repertoireStarted = true;
        return { status: 'missing', reason: 'repertoire settles' };
      }
    });
    trackController(controller);

    const initializing = controller.initialize();
    await vi.waitFor(() => expect(interventionsStarted).toBe(true));
    expect(repertoireStarted).toBe(false);
    expect(callbacks.repertoireNullResults).toHaveLength(0);

    resolveInterventions?.();
    await initializing;
    await vi.waitFor(() => expect(callbacks.repertoireNullResults).toHaveLength(1));
    expect(repertoireStarted).toBe(true);
  });
});
