import { describe, expect, it, vi } from 'vitest';
import { ExperimentController } from '../../src/lib/experiment/controller';
import type { NullExplanationLoadResult } from '../../src/lib/experiment/nullExplanation';
import { createCallbacks, createWorker, SEED, TOTAL_TICKS, useControllerTestLifecycle } from './experiment-controller-test-helpers';

/**
 * WP4 of `.agents/plans/null-explanation` (`04-ledger-note.md`): controller
 * coverage for `onNullExplanation`, split into its own file (rather than
 * folded into `tests/unit/experiment-controller.test.ts`, which is already
 * near the thermo review's 1000-line threshold — see that file's own doc
 * comment for the precedent). Mirrors `experiment-controller.test.ts`'s
 * "ExperimentController rewiring-null loading (WP4)" describe block
 * structurally: `onNullExplanation` is sequenced after `onRewiringNull`'s
 * own loader settles (`controller.ts`'s `nullLoad`), but fired from an
 * independent promise fork, so the two host callbacks can never take each
 * other down (see the "still fires onNullExplanation even when the
 * onRewiringNull host callback throws" test below).
 */

const { trackController } = useControllerTestLifecycle();

describe('ExperimentController null-explanation loading (WP4 of .agents/plans/null-explanation)', () => {
  /**
   * `createPublicDataFetch` serves whichever committed `public/data/*` file
   * matches the requested basename, and `null-explanation-v1.json` plus its
   * manifest entry are committed there — so this runs against the real
   * shipped artifact by default.
   */
  it('fires onNullExplanation with the real artifact as "ok", after onRewiringNull, without blocking reaching "ready"', async () => {
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
    expect(callbacks.rewiringNullResults).toHaveLength(1);
    // `onNullExplanation` is chained *after* `onRewiringNull` on the same
    // promise (`controller.ts`'s own doc comment) — reaching `ready` proves
    // loading it never blocks Start, but its own extra fetch/verify step can
    // still settle a tick or two after `initialize()`'s returned promise
    // itself resolves (that promise does not await this fire-and-forget
    // chain at all), so this waits for it explicitly rather than assuming
    // the same microtask-interleaving luck `onRewiringNull`'s own assertion
    // (just above) happens to get for free.
    await vi.waitFor(() => expect(callbacks.nullExplanationResults).toHaveLength(1));
    const result = callbacks.nullExplanationResults[0];
    expect(result.status).toBe('ok');
    if (result.status === 'ok') {
      expect(result.data.finding.qualifyingMetrics.length).toBeGreaterThan(0);
    }
  });

  it('is "missing" when the injectable loadNullExplanation directly returns a stubbed "missing" result', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadNullExplanation: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    expect(callbacks.nullExplanationResults).toHaveLength(1);
    expect(callbacks.nullExplanationResults[0]).toEqual({ status: 'missing', reason: 'stubbed for this test' });
  });

  it('does not fire onNullExplanation once disposed before initialize() resolves at all', async () => {
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

    expect(callbacks.nullExplanationResults).toHaveLength(0);
  });

  /**
   * A macrotask flush (`setTimeout(0)`), not a fixed count of
   * `await Promise.resolve()` hops (round-2 dual review, Important —
   * empirically confirmed: two hops settle *before* the chained
   * `loadExplanation(...).catch(...)`'s own inner-promise adoption
   * completes, which made both the negative assertion below and its
   * `destroyed`-guard removal pass vacuously). `flush()` drains every queued
   * microtask regardless of how many hops the real chain has, and is
   * exercised by a genuine positive control just below so a future chain
   * change can never silently make the negative assertion pass for the
   * wrong reason again.
   */
  const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

  /**
   * Same "actual post-ready race" coverage `experiment-controller.test.ts`
   * gives `onRewiringNull`'s own `destroyed` guard, for this load's guard.
   */
  it('drops a null-explanation result that settles after dispose()', async () => {
    let resolveExplanation!: (result: NullExplanationLoadResult) => void;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadNullExplanation: () =>
        new Promise((resolve) => {
          resolveExplanation = resolve;
        })
    });
    trackController(controller);

    await controller.initialize();
    expect(callbacks.statuses).toContain('ready');
    expect(callbacks.nullExplanationResults).toHaveLength(0);

    controller.dispose();
    resolveExplanation({ status: 'missing', reason: 'settled after dispose' });
    await flush();

    expect(callbacks.nullExplanationResults).toHaveLength(0);
  });

  /**
   * Positive control for the test above: the identical setup, minus
   * `dispose()`. Proves `flush()` waits long enough for the result to land
   * when nothing should be dropping it — without this, a `flush()` that
   * resolved too early would make the test above pass for the wrong reason
   * (never actually reaching the callback either way).
   */
  it('(positive control) the same settle-after-a-delay setup without dispose() reports the result after flush()', async () => {
    let resolveExplanation!: (result: NullExplanationLoadResult) => void;
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadNullExplanation: () =>
        new Promise((resolve) => {
          resolveExplanation = resolve;
        })
    });
    trackController(controller);

    await controller.initialize();
    expect(callbacks.nullExplanationResults).toHaveLength(0);

    resolveExplanation({ status: 'missing', reason: 'settled without dispose' });
    await flush();

    expect(callbacks.nullExplanationResults).toHaveLength(1);
  });

  /**
   * Mirrors `experiment-controller.test.ts`'s "maps a rejecting
   * loadRewiringNull to an 'unavailable' onRewiringNull result" test:
   * `NullExplanationLoadResult` now has its own `'unavailable'` variant
   * (round-2 dual review, Important — an earlier version mapped this to
   * `'invalid'`, which `LedgerPanel.svelte` renders as "Explanation failed
   * verification" even though nothing was actually verified).
   */
  it('maps a rejecting loadNullExplanation to an "unavailable" onNullExplanation result instead of an unhandled rejection', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadNullExplanation: async () => {
        throw new Error('boom');
      }
    });
    trackController(controller);

    await controller.initialize();

    expect(callbacks.nullExplanationResults).toHaveLength(1);
    const result = callbacks.nullExplanationResults[0];
    expect(result.status).toBe('unavailable');
    if (result.status === 'unavailable') expect(result.reason).toMatch(/unexpected error.*boom/);
  });

  /**
   * Round-2 dual review, Important: an earlier version chained
   * `loadExplanation`/`onNullExplanation` directly after the
   * `onRewiringNull(result)` callback call, so a throwing `onRewiringNull`
   * host callback silently skipped the null-explanation load too — the two
   * host callbacks must stay independent (`controller.ts`'s `nullLoad` doc
   * comment).
   */
  it('still fires onNullExplanation even when the onRewiringNull host callback throws', async () => {
    const callbacks = createCallbacks();
    callbacks.onRewiringNull = () => {
      throw new Error('onRewiringNull host callback boom');
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadNullExplanation: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    expect(callbacks.errors.some((message) => message.includes('onRewiringNull host callback boom'))).toBe(true);
    expect(callbacks.nullExplanationResults).toHaveLength(1);
    expect(callbacks.nullExplanationResults[0]).toEqual({ status: 'missing', reason: 'stubbed for this test' });
  });

  it('routes a throwing onNullExplanation callback to onError instead of an unhandled rejection', async () => {
    const callbacks = createCallbacks();
    callbacks.onNullExplanation = () => {
      throw new Error('host callback boom');
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadNullExplanation: async () => ({ status: 'missing', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();
    await Promise.resolve();
    await Promise.resolve();

    expect(callbacks.errors.some((message) => message.includes('host callback boom'))).toBe(true);
  });

  it('still fires onNullExplanation (attempted unconditionally) even when the rewiring-null load itself fails', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks,
      loadRewiringNull: async () => ({ status: 'absent', reason: 'stubbed for this test' })
    });
    trackController(controller);

    await controller.initialize();

    expect(callbacks.rewiringNullResults).toHaveLength(1);
    expect(callbacks.rewiringNullResults[0]).toEqual({ status: 'absent', reason: 'stubbed for this test' });
    // Real `loadNullExplanation` still runs against the real committed
    // artifact here (only `loadRewiringNull` is stubbed above) — see the
    // first test's own comment for why this waits explicitly.
    await vi.waitFor(() => expect(callbacks.nullExplanationResults).toHaveLength(1));
    expect(callbacks.nullExplanationResults[0].status).toBe('ok');
  });
});
