import { describe, expect, it, vi } from 'vitest';
import { ExperimentController } from '../../src/lib/experiment/controller';
import { createCallbacks, createWorker, SEED, TOTAL_TICKS, useControllerTestLifecycle } from './experiment-controller-test-helpers';

/**
 * WP4 of `.agents/plans/task-generality`: controller coverage for
 * `onTaskGenerality`, mirroring `tests/unit/experiment-controller-repertoire-null.test.ts`'s
 * own structure and reasoning (a dual-review finding: the new
 * `onTaskGenerality` seam's own doc comment claims the same guarantees
 * every earlier sidecar seam already has tests for -- fired in parallel
 * with the prior forks, isolated from a throwing host callback, and a
 * loader throw mapped to `'unavailable'` rather than an unhandled
 * rejection -- but shipped with none of its own beyond the two coarse
 * assertions in `experiment-controller-sidecar-parallel.test.ts`).
 *
 * This is one more fork off `artifacts.manifest`/`dataBaseUrl`, fired in
 * parallel with (not sequenced behind) every other sidecar load -- see
 * `runSidecarLoad`'s own doc comment.
 */

const { trackController } = useControllerTestLifecycle();

describe('ExperimentController task-generality loading (WP4 of .agents/plans/task-generality)', () => {
  /**
   * `createPublicDataFetch` serves whichever committed `public/data/*` file
   * matches the requested basename, and `task-generality-v1.json` plus its
   * manifest entry are committed there -- so this runs against the real
   * shipped artifact by default.
   */
  it('fires onTaskGenerality with the real artifact as "ok", without blocking reaching "ready"', async () => {
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
    await vi.waitFor(() => expect(callbacks.taskGeneralityResults).toHaveLength(1));
    const result = callbacks.taskGeneralityResults[0];
    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      // The real shipped result (independently verified against the WP2/WP3 inputs).
      expect(result.data.overall.authored.verdict).toBe('general');
      expect(result.data.overall.trained.verdict).toBe('task-dependent');
    }
  });

  it('is "missing" when the injectable loadTaskGenerality directly returns a stubbed "missing" result', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadTaskGenerality: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.taskGeneralityResults).toHaveLength(1));
    expect(callbacks.taskGeneralityResults[0]).toEqual({ status: 'missing', reason: 'stubbed for this test' });
  });

  it('does not fire onTaskGenerality once disposed before initialize() resolves at all', async () => {
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

    expect(callbacks.taskGeneralityResults).toHaveLength(0);
  });

  /** A macrotask flush that drains every queued microtask, regardless of how deep this chain is -- same reasoning as the sibling sidecar test files' own `flush()`. */
  const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

  it('drops a task-generality result that settles after dispose()', async () => {
    let resolveTaskGenerality: ((result: { status: 'missing'; reason: string }) => void) | undefined;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadTaskGenerality: () =>
        new Promise((resolve) => {
          resolveTaskGenerality = resolve;
        })
    });
    trackController(controller);

    await controller.initialize();
    expect(callbacks.statuses).toContain('ready');
    await vi.waitFor(() => expect(resolveTaskGenerality).toBeDefined());
    expect(callbacks.taskGeneralityResults).toHaveLength(0);

    controller.dispose();
    resolveTaskGenerality?.({ status: 'missing', reason: 'settled after dispose' });
    await flush();

    expect(callbacks.taskGeneralityResults).toHaveLength(0);
  });

  /**
   * Positive control for the test above: the identical setup, minus
   * `dispose()`. Proves the wait/flush combination reports the result when
   * nothing should be dropping it.
   */
  it('(positive control) the same settle-after-a-delay setup without dispose() reports the result after flush()', async () => {
    let resolveTaskGenerality: ((result: { status: 'missing'; reason: string }) => void) | undefined;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadTaskGenerality: () =>
        new Promise((resolve) => {
          resolveTaskGenerality = resolve;
        })
    });
    trackController(controller);

    await controller.initialize();
    await vi.waitFor(() => expect(resolveTaskGenerality).toBeDefined());
    expect(callbacks.taskGeneralityResults).toHaveLength(0);

    resolveTaskGenerality?.({ status: 'missing', reason: 'settled without dispose' });
    await flush();

    expect(callbacks.taskGeneralityResults).toHaveLength(1);
  });

  it('maps a rejecting loadTaskGenerality to an "unavailable" onTaskGenerality result instead of an unhandled rejection', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadTaskGenerality: async () => {
        throw new Error('boom');
      }
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.taskGeneralityResults).toHaveLength(1));
    const result = callbacks.taskGeneralityResults[0];
    expect(result.status).toBe('unavailable');
    if (result.status === 'unavailable') expect(result.reason).toMatch(/unexpected error.*boom/);
  });

  it('still fires onTaskGenerality even when the onRepertoireNull host callback throws', async () => {
    const callbacks = createCallbacks();
    callbacks.onRepertoireNull = () => {
      throw new Error('onRepertoireNull host callback boom');
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadTaskGenerality: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.taskGeneralityResults).toHaveLength(1));
    expect(callbacks.errors.some((message) => message.includes('onRepertoireNull host callback boom'))).toBe(true);
    expect(callbacks.taskGeneralityResults[0]).toEqual({ status: 'missing', reason: 'stubbed for this test' });
  });

  it('routes a throwing onTaskGenerality callback to onError instead of an unhandled rejection', async () => {
    const callbacks = createCallbacks();
    callbacks.onTaskGenerality = () => {
      throw new Error('host callback boom');
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadTaskGenerality: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.errors.some((message) => message.includes('host callback boom'))).toBe(true));
  });

  it('still fires onTaskGenerality (attempted unconditionally) even when the repertoire-null load itself fails', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadRepertoireNull: async () => ({ status: 'unavailable', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.repertoireNullResults).toHaveLength(1));
    expect(callbacks.repertoireNullResults[0]).toEqual({ status: 'unavailable', reason: 'stubbed for this test' });
    // Real `loadTaskGenerality` still runs against the real committed
    // artifact here (only `loadRepertoireNull` is stubbed above) -- its own
    // cross-check re-reads `manifest.rewiringNull.sha256`/`manifest.pathwayInterventions.sha256`
    // directly, not the resolved `RepertoireNullLoadResult`, so a failed
    // repertoire-null load does not prevent it from resolving "ok".
    await vi.waitFor(() => expect(callbacks.taskGeneralityResults).toHaveLength(1));
    expect(callbacks.taskGeneralityResults[0].status).toBe('ok');
  });

  /**
   * Parallelism: every sidecar load is invoked immediately, off the same
   * manifest, never gated behind another fork's settled promise -- see
   * `runSidecarLoad`'s own doc comment.
   */
  it('invokes loadTaskGenerality immediately, without waiting for the repertoire-null load to settle', async () => {
    let taskGeneralityStarted = false;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      // Never resolves -- if `loadTaskGenerality` were still gated behind
      // this settling, `taskGeneralityStarted` would stay false and
      // `taskGeneralityResults` would stay empty forever.
      loadRepertoireNull: () => new Promise(() => {}),
      loadTaskGenerality: async () => {
        taskGeneralityStarted = true;
        return { status: 'missing', reason: 'task-generality settles' };
      }
    });
    trackController(controller);

    await controller.initialize();
    await vi.waitFor(() => expect(callbacks.taskGeneralityResults).toHaveLength(1));
    expect(taskGeneralityStarted).toBe(true);
  });
});
