import { describe, expect, it, vi } from 'vitest';
import { ExperimentController } from '../../src/lib/experiment/controller';
import type { DecoderKind } from '../../src/lib/worker/protocol';
import { createPublicDataFetch } from '../helpers/fake-worker';
import { createCallbacks, createWorker, SEED, TOTAL_TICKS, useControllerTestLifecycle } from './experiment-controller-test-helpers';

/**
 * Decoder-toggle (`ExperimentController#setDecoder`) and trained-readout
 * artifact loading coverage, split out of `experiment-controller.test.ts`
 * when that file crossed the thermo review's 1000-line threshold (see
 * `reviews/feat-nom6-trained-toggle-thermo-2026-09-24-766f09c/thermo-architecture/01-critical-and-important.md`).
 * Shares its fixture/lifecycle setup with that file via
 * `./experiment-controller-test-helpers.ts` — see that module's doc
 * comment.
 */

const { trackController } = useControllerTestLifecycle();

/**
 * `createPublicDataFetch` (`../helpers/fake-worker.ts`) serves whichever
 * committed `public/data/*` file matches the requested basename, and WP5's
 * production artifacts (`trained-readout-v1.{json,manifest.json}`, D = 48,
 * H = 16, parameterCount = 835) are committed there — so these tests run
 * against the real shipped artifact by default, not a fixture, the same way
 * `ExperimentController#initialize`'s existing tests do for the arena graph.
 */
describe('ExperimentController trained-readout / setDecoder', () => {
  const setUp = async (overrides?: { totalTicks?: number; targetTickIntervalMs?: number }) => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: overrides?.totalTicks ?? TOTAL_TICKS,
      targetTickIntervalMs: overrides?.targetTickIntervalMs,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks
    });
    trackController(controller);
    await controller.initialize();
    return { controller, callbacks };
  };

  it('defaults to the authored decoder and reports the real trained-readout artifact as ok with its manifest D/H/parameterCount', async () => {
    const { controller, callbacks } = await setUp();

    expect(controller.getDecoder()).toBe('authored');
    expect(callbacks.trainedReadoutStatuses).toHaveLength(1);
    const status = callbacks.trainedReadoutStatuses[0];
    expect(status.status).toBe('ok');
    if (status.status === 'ok') {
      expect(status.manifest.D).toBe(48);
      expect(status.manifest.H).toBe(16);
      expect(status.manifest.parameterCount).toBe(835);
    }
    expect(controller.getTrainedReadoutStatus()).toBe(status);
  });

  it('a corrupted trained-readout-v1.json disables Trained with an honest reason; Authored still works and setDecoder("trained") stays a no-op', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'trained-readout-v1.json' }));
    const { controller, callbacks } = await setUp();

    // The required arena graph artifacts are untouched by this corruption,
    // so the experiment itself must still come up fine (WP6's "the app
    // keeps working in Authored mode" invariant).
    expect(controller.getRunner()).toBeDefined();
    expect(callbacks.errors).toHaveLength(0);

    const status = callbacks.trainedReadoutStatuses[0];
    expect(status.status).toBe('unavailable');
    if (status.status === 'unavailable') expect(status.reason).toMatch(/sha256/i);

    await controller.setDecoder('trained');
    expect(controller.getDecoder()).toBe('authored');
    expect(callbacks.decodersApplied).toHaveLength(0);
  });

  it('setDecoder is a no-op while the run is running', async () => {
    const { controller, callbacks } = await setUp({ targetTickIntervalMs: 0 });
    const runner = controller.getRunner();
    runner!.start();
    expect(runner!.getStatus()).toBe('running');

    await controller.setDecoder('trained');

    expect(controller.getDecoder()).toBe('authored');
    expect(callbacks.decodersApplied).toHaveLength(0);
    runner!.pause();
  });

  it('setDecoder applies to both arms and resets the run to tick 0, even from paused', async () => {
    // Real-time pacing left at its default (unlike the "no-op while running"
    // and determinism tests above/below): with it disabled, a 30-tick run
    // can race straight to `finished` between the `waitFor` below and
    // `pause()`, since nothing then bounds how fast ticks resolve.
    const { controller, callbacks } = await setUp();
    const runner = controller.getRunner();
    runner!.start();
    await vi.waitFor(() => expect(runner!.getTelemetry().tick).toBeGreaterThan(0));
    runner!.pause();
    expect(runner!.getStatus()).toBe('paused');

    await controller.setDecoder('trained');

    expect(controller.getDecoder()).toBe('trained');
    expect(callbacks.decodersApplied).toEqual(['trained']);
    expect(runner!.getStatus()).toBe('ready');
    expect(runner!.getTelemetry().tick).toBe(0);
    expect(callbacks.errors).toHaveLength(0);
  });

  it('is a no-op when the requested decoder is already selected', async () => {
    const { controller, callbacks } = await setUp();
    await controller.setDecoder('authored');
    expect(callbacks.decodersApplied).toHaveLength(0);
  });

  /**
   * Round-2 dual review: both independent reviewers found the same gap —
   * `decoderSwitchInFlight` protected `changeTopology` against `setDecoder`,
   * but `setDecoder` never checked it against a second, overlapping call to
   * *itself*. `decoder === this.decoder` alone does not exclude this: the
   * first call has not written `this.decoder` yet when the second call's
   * guards run, so a same-target overlap would previously fire two
   * independent `Promise.all`/`runner.reset()` sequences. Exercised the same
   * way as the round-1 regression test above: the second call is issued
   * synchronously, before the first call's `Promise.all` has any chance to
   * settle.
   */
  it('a second, overlapping setDecoder call for the same target decoder is a no-op (regression: round-2 dual review)', async () => {
    const { controller, callbacks } = await setUp();
    const runner = controller.getRunner()!;

    const first = controller.setDecoder('trained');
    const second = controller.setDecoder('trained');
    await Promise.all([first, second]);

    expect(controller.getDecoder()).toBe('trained');
    // Exactly one apply, not two — the second call's guard must have caught
    // it before it fired its own Worker round trip.
    expect(callbacks.decodersApplied).toEqual(['trained']);
    expect(runner.getTelemetry().tick).toBe(0);
    expect(callbacks.errors).toHaveLength(0);
  });

  /**
   * The WP6 acceptance bar this test targets directly: "two runs with the
   * same seed and Trained give identical score traces over 300 ticks; the
   * Authored and Trained traces differ." `targetTickIntervalMs: 0` disables
   * `ExperimentRunner`'s real-time pacing (see that option's doc comment) so
   * 300 ticks complete in milliseconds instead of ~10 real seconds.
   */
  it('determinism: two Trained runs with the same seed produce identical 300-tick score traces; Authored and Trained differ', async () => {
    const runToCompletion = async (decoder: DecoderKind) => {
      const { controller } = await setUp({ totalTicks: 300, targetTickIntervalMs: 0 });
      if (decoder === 'trained') {
        await controller.setDecoder('trained');
        expect(controller.getDecoder()).toBe('trained');
      }
      const runner = controller.getRunner()!;
      runner.start();
      await vi.waitFor(() => expect(runner.getStatus()).toBe('finished'), { timeout: 10000 });
      return runner.getReplayExport().trace;
    };

    const trainedTraceA = await runToCompletion('trained');
    const trainedTraceB = await runToCompletion('trained');
    const authoredTrace = await runToCompletion('authored');

    expect(trainedTraceA).toEqual(trainedTraceB);
    expect(trainedTraceA).not.toEqual(authoredTrace);
  }, 20000);

  /**
   * Regression test for a race dual review (round 1) found independently in
   * both passes: `changeTopology` had no guard against a `setDecoder` call
   * already in flight, so a topology switch could dispose/re-init an arm's
   * Worker while `setDecoder`'s own `Promise.all` was still awaiting that
   * same arm's `set-decoder` ack — the rebuilt binding's re-apply branch
   * reads `this.decoder`, which `setDecoder` only writes *after* its await
   * resolves, so the rebuilt arm could silently stay on `'authored'` while
   * the controller went on to report `'trained'` for both arms. Fixed by
   * `ExperimentController#decoderSwitchInFlight`, set synchronously (before
   * `setDecoder`'s first `await`) and checked by `changeTopology`'s own
   * guard — this test exercises exactly that window without any artificial
   * timing: `changeTopology` is called synchronously, in the same
   * microtask, right after `setDecoder` is invoked (before its `Promise.all`
   * has any chance to settle).
   */
  it('changeTopology is a no-op while a decoder switch is synchronously in flight (regression: dual review round 1 race)', async () => {
    const { controller, callbacks } = await setUp();
    const runner = controller.getRunner()!;

    const decoderPromise = controller.setDecoder('trained');
    controller.changeTopology('left', 'disconnected');

    expect(callbacks.switchCounts).toHaveLength(0);
    // `topologyApplied` already carries `['left', 'biological']` from
    // `initialize()`'s own initial-topology report — the no-op assertion is
    // that the *rejected* switch's target mode was never applied, not that
    // the array is empty.
    expect(callbacks.topologyApplied.filter(([agentId, mode]) => agentId === 'left' && mode === 'disconnected')).toHaveLength(0);

    await decoderPromise;
    expect(controller.getDecoder()).toBe('trained');
    expect(callbacks.errors).toHaveLength(0);
    // The rejected topology switch must not have silently applied.
    expect(runner.getTelemetry().agents.left.topology).toBe('biological');

    // The controller recovers cleanly once the decoder switch has actually
    // landed: a topology switch issued afterward works normally.
    controller.changeTopology('left', 'disconnected');
    await vi.waitFor(() => expect(callbacks.switchCounts.at(-1)?.left).toBe(0));
    expect(runner.getTelemetry().agents.left.topology).toBe('disconnected');
    expect(callbacks.errors).toHaveLength(0);
  });

  it('setDecoder is a no-op while a topology switch is in flight (the existing, symmetric guard)', async () => {
    const { controller, callbacks } = await setUp();
    controller.changeTopology('left', 'disconnected');
    expect(callbacks.switchCounts.at(-1)?.left).toBe(1);

    await controller.setDecoder('trained');
    expect(controller.getDecoder()).toBe('authored');
    expect(callbacks.decodersApplied).toHaveLength(0);

    await vi.waitFor(() => expect(callbacks.switchCounts.at(-1)?.left).toBe(0));
    expect(callbacks.errors).toHaveLength(0);
  });

  /**
   * Behavioral regression coverage for `changeTopology`'s trained-decoder
   * re-apply branch (`controller.ts`'s `if (this.decoder === 'trained') {
   * await client.setDecoder('trained'); ... }`) — black-box, through real
   * movement rather than peeking at internal Worker state. Authored +
   * disconnected always decodes to the exact zero action (no recurrent path
   * can reach an output-assigned neuron with `edgeCount` 0), so the agent
   * never moves — `distanceTravelled` stays exactly `0` for the whole run.
   * If the re-apply silently failed to land (the exact regression this test
   * guards against), the rebuilt right-arm Worker would still be on
   * `'authored'` and would be indistinguishable from that case. The real
   * shipped trained readout's bias terms (`b1`/`b2`) make a non-zero action
   * — and therefore non-zero movement — overwhelmingly likely even though
   * the disconnected arm's own output-neuron rates decay to zero
   * (`readoutForward` still applies the biases); the real evaluation report
   * (`docs/trained-readout-report.md`) independently confirms this arm's
   * trained mean movement score (~35.5) is far from the authored arm's
   * (~-1.9), so this is not a coincidental/flaky signal.
   */
  it('a topology switch re-applies the trained decoder to the rebuilt binding: disconnected + trained still moves, unlike disconnected + authored', async () => {
    const { controller } = await setUp({ targetTickIntervalMs: 0 });
    const runner = controller.getRunner()!;

    await controller.setDecoder('trained');
    expect(controller.getDecoder()).toBe('trained');

    controller.changeTopology('right', 'disconnected');
    await vi.waitFor(() => expect(runner.getTelemetry().agents.right.topology).toBe('disconnected'));

    runner.start();
    await vi.waitFor(() => expect(runner.getTelemetry().tick).toBeGreaterThan(10));
    runner.pause();

    const trace = runner.getReplayExport().trace;
    expect(trace.at(-1)!.agents.right.distanceTravelled).not.toBe(0);
  });
});

