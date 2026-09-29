import { describe, expect, it, vi } from 'vitest';
import { ExperimentController } from '../../src/lib/experiment/controller';
import { createCallbacks, createWorker, SEED, TOTAL_TICKS, useControllerTestLifecycle } from './experiment-controller-test-helpers';

/**
 * WP3 of `.agents/plans/selection-robustness`: controller coverage for
 * `onSelectionRobustness`, mirroring
 * `tests/unit/experiment-controller-task-generality.test.ts`'s own
 * structure and reasoning -- fired in parallel with every other sidecar
 * fork, isolated from a throwing host callback, and a loader throw mapped
 * to `'unavailable'` rather than an unhandled rejection.
 *
 * This is the sixth fork off `artifacts.manifest`/`dataBaseUrl`, fired in
 * parallel with (not sequenced behind) every other sidecar load -- see
 * `runSidecarLoad`'s own doc comment.
 */

const { trackController } = useControllerTestLifecycle();

describe('ExperimentController selection-robustness loading (WP3 of .agents/plans/selection-robustness)', () => {
  /**
   * `createPublicDataFetch` serves whichever committed `public/data/*` file
   * matches the requested basename, and `selection-robustness-v1.json` plus
   * its manifest entry are committed there -- so this runs against the real
   * shipped artifact by default.
   */
  it('fires onSelectionRobustness with the real artifact as "ok", without blocking reaching "ready"', async () => {
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
    await vi.waitFor(() => expect(callbacks.selectionRobustnessResults).toHaveLength(1));
    const result = callbacks.selectionRobustnessResults[0];
    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      // The real shipped result (independently verified against the WP2 chain inputs).
      expect(result.data.overall.robustToSize.verdict).toBe(true);
      expect(result.data.overall.robustToMethod.verdict).toBe(false);
      expect(result.data.overall.mapping.verdict).toBe(false);
    }
  });

  it('is "missing" when the injectable loadSelectionRobustness directly returns a stubbed "missing" result', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadSelectionRobustness: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.selectionRobustnessResults).toHaveLength(1));
    expect(callbacks.selectionRobustnessResults[0]).toEqual({ status: 'missing', reason: 'stubbed for this test' });
  });

  it('does not fire onSelectionRobustness once disposed before initialize() resolves at all', async () => {
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

    expect(callbacks.selectionRobustnessResults).toHaveLength(0);
  });

  /** A macrotask flush that drains every queued microtask, regardless of how deep this chain is -- same reasoning as the sibling sidecar test files' own `flush()`. */
  const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

  it('drops a selection-robustness result that settles after dispose()', async () => {
    let resolveSelectionRobustness: ((result: { status: 'missing'; reason: string }) => void) | undefined;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadSelectionRobustness: () =>
        new Promise((resolve) => {
          resolveSelectionRobustness = resolve;
        })
    });
    trackController(controller);

    await controller.initialize();
    expect(callbacks.statuses).toContain('ready');
    await vi.waitFor(() => expect(resolveSelectionRobustness).toBeDefined());
    expect(callbacks.selectionRobustnessResults).toHaveLength(0);

    controller.dispose();
    resolveSelectionRobustness?.({ status: 'missing', reason: 'settled after dispose' });
    await flush();

    expect(callbacks.selectionRobustnessResults).toHaveLength(0);
  });

  /**
   * Positive control for the test above: the identical setup, minus
   * `dispose()`. Proves the wait/flush combination reports the result when
   * nothing should be dropping it.
   */
  it('(positive control) the same settle-after-a-delay setup without dispose() reports the result after flush()', async () => {
    let resolveSelectionRobustness: ((result: { status: 'missing'; reason: string }) => void) | undefined;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadSelectionRobustness: () =>
        new Promise((resolve) => {
          resolveSelectionRobustness = resolve;
        })
    });
    trackController(controller);

    await controller.initialize();
    await vi.waitFor(() => expect(resolveSelectionRobustness).toBeDefined());
    expect(callbacks.selectionRobustnessResults).toHaveLength(0);

    resolveSelectionRobustness?.({ status: 'missing', reason: 'settled without dispose' });
    await flush();

    expect(callbacks.selectionRobustnessResults).toHaveLength(1);
  });

  it('maps a rejecting loadSelectionRobustness to an "unavailable" onSelectionRobustness result instead of an unhandled rejection', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadSelectionRobustness: async () => {
        throw new Error('boom');
      }
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.selectionRobustnessResults).toHaveLength(1));
    const result = callbacks.selectionRobustnessResults[0];
    expect(result.status).toBe('unavailable');
    if (result.status === 'unavailable') expect(result.reason).toMatch(/unexpected error.*boom/);
  });

  it('still fires onSelectionRobustness even when the onTaskGenerality host callback throws', async () => {
    const callbacks = createCallbacks();
    callbacks.onTaskGenerality = () => {
      throw new Error('onTaskGenerality host callback boom');
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadSelectionRobustness: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.selectionRobustnessResults).toHaveLength(1));
    expect(callbacks.errors.some((message) => message.includes('onTaskGenerality host callback boom'))).toBe(true);
    expect(callbacks.selectionRobustnessResults[0]).toEqual({ status: 'missing', reason: 'stubbed for this test' });
  });

  it('routes a throwing onSelectionRobustness callback to onError instead of an unhandled rejection', async () => {
    const callbacks = createCallbacks();
    callbacks.onSelectionRobustness = () => {
      throw new Error('host callback boom');
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadSelectionRobustness: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.errors.some((message) => message.includes('host callback boom'))).toBe(true));
  });

  it('still fires onSelectionRobustness (attempted unconditionally) even when the task-generality load itself fails', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadTaskGenerality: async () => ({ status: 'unavailable', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.taskGeneralityResults).toHaveLength(1));
    expect(callbacks.taskGeneralityResults[0]).toEqual({ status: 'unavailable', reason: 'stubbed for this test' });
    // Real `loadSelectionRobustness` still runs against the real committed
    // artifact here (only `loadTaskGenerality` is stubbed above) -- its own
    // cross-check re-reads `manifest.binarySha256` directly, not the
    // resolved `TaskGeneralityLoadResult`, so a failed task-generality load
    // does not prevent it from resolving "ok".
    await vi.waitFor(() => expect(callbacks.selectionRobustnessResults).toHaveLength(1));
    expect(callbacks.selectionRobustnessResults[0].status).toBe('ok');
  });

  /**
   * Parallelism: the six sidecar loads are each invoked immediately, off
   * the same manifest, never gated behind another fork's settled promise --
   * see `runSidecarLoad`'s own doc comment.
   */
  it('invokes loadSelectionRobustness immediately, without waiting for the task-generality load to settle', async () => {
    let selectionRobustnessStarted = false;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      // Never resolves -- if `loadSelectionRobustness` were still gated
      // behind this settling, `selectionRobustnessStarted` would stay false
      // and `selectionRobustnessResults` would stay empty forever.
      loadTaskGenerality: () => new Promise(() => {}),
      loadSelectionRobustness: async () => {
        selectionRobustnessStarted = true;
        return { status: 'missing', reason: 'selection-robustness settles' };
      }
    });
    trackController(controller);

    await controller.initialize();
    await vi.waitFor(() => expect(callbacks.selectionRobustnessResults).toHaveLength(1));
    expect(selectionRobustnessStarted).toBe(true);
  });
});
