import { describe, expect, it, vi } from 'vitest';
import { ExperimentController } from '../../src/lib/experiment/controller';
import { createCallbacks, createWorker, SEED, TOTAL_TICKS, useControllerTestLifecycle } from './experiment-controller-test-helpers';

/**
 * WP3 of `.agents/plans/readout-attribution`: controller coverage for
 * `onReadoutAttribution`, mirroring
 * `tests/unit/experiment-controller-selection-robustness.test.ts`'s own
 * structure and reasoning -- fired in parallel with every other sidecar
 * fork, isolated from a throwing host callback, and a loader throw mapped
 * to `'unavailable'` rather than an unhandled rejection.
 *
 * This is the seventh fork off `artifacts.manifest`/`dataBaseUrl`, fired in
 * parallel with (not sequenced behind) every other sidecar load -- see
 * `runSidecarLoad`'s own doc comment.
 */

const { trackController } = useControllerTestLifecycle();

describe('ExperimentController readout-attribution loading (WP3 of .agents/plans/readout-attribution)', () => {
  /**
   * `createPublicDataFetch` serves whichever committed `public/data/*` file
   * matches the requested basename, and `readout-attribution-v1.json` plus
   * its manifest entry are committed there -- so this runs against the real
   * shipped artifact by default.
   */
  it('fires onReadoutAttribution with the real artifact as "ok", without blocking reaching "ready"', async () => {
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
    await vi.waitFor(() => expect(callbacks.readoutAttributionResults).toHaveLength(1));
    const result = callbacks.readoutAttributionResults[0];
    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      // The real shipped WP2 result: every hypothesis came back inconclusive.
      expect(result.data.hypotheses.H1.outcome).toBe('inconclusive');
      expect(result.data.hypotheses.H2.outcome).toBe('inconclusive');
      expect(result.data.hypotheses.H3.outcome).toBe('inconclusive');
    }
  });

  it('is "missing" when the injectable loadReadoutAttribution directly returns a stubbed "missing" result', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadReadoutAttribution: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.readoutAttributionResults).toHaveLength(1));
    expect(callbacks.readoutAttributionResults[0]).toEqual({ status: 'missing', reason: 'stubbed for this test' });
  });

  it('does not fire onReadoutAttribution once disposed before initialize() resolves at all', async () => {
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

    expect(callbacks.readoutAttributionResults).toHaveLength(0);
  });

  it('maps a rejecting loadReadoutAttribution to an "unavailable" onReadoutAttribution result instead of an unhandled rejection', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadReadoutAttribution: async () => {
        throw new Error('boom');
      }
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.readoutAttributionResults).toHaveLength(1));
    const result = callbacks.readoutAttributionResults[0];
    expect(result.status).toBe('unavailable');
    if (result.status === 'unavailable') expect(result.reason).toMatch(/unexpected error.*boom/);
  });

  it('still fires onReadoutAttribution even when the onSelectionRobustness host callback throws', async () => {
    const callbacks = createCallbacks();
    callbacks.onSelectionRobustness = () => {
      throw new Error('onSelectionRobustness host callback boom');
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadReadoutAttribution: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.readoutAttributionResults).toHaveLength(1));
    expect(callbacks.errors.some((message) => message.includes('onSelectionRobustness host callback boom'))).toBe(true);
    expect(callbacks.readoutAttributionResults[0]).toEqual({ status: 'missing', reason: 'stubbed for this test' });
  });

  it('routes a throwing onReadoutAttribution callback to onError instead of an unhandled rejection', async () => {
    const callbacks = createCallbacks();
    callbacks.onReadoutAttribution = () => {
      throw new Error('host callback boom');
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadReadoutAttribution: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    await vi.waitFor(() => expect(callbacks.errors.some((message) => message.includes('host callback boom'))).toBe(true));
  });

  /**
   * Parallelism: the seven sidecar loads are each invoked immediately, off
   * the same manifest, never gated behind another fork's settled promise --
   * see `runSidecarLoad`'s own doc comment.
   */
  it('invokes loadReadoutAttribution immediately, without waiting for the selection-robustness load to settle', async () => {
    let readoutAttributionStarted = false;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      // Never resolves -- if `loadReadoutAttribution` were still gated
      // behind this settling, `readoutAttributionStarted` would stay false
      // and `readoutAttributionResults` would stay empty forever.
      loadSelectionRobustness: () => new Promise(() => {}),
      loadReadoutAttribution: async () => {
        readoutAttributionStarted = true;
        return { status: 'missing', reason: 'readout-attribution settles' };
      }
    });
    trackController(controller);

    await controller.initialize();
    await vi.waitFor(() => expect(callbacks.readoutAttributionResults).toHaveLength(1));
    expect(readoutAttributionStarted).toBe(true);
  });
});
