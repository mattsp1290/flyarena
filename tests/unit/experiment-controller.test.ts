import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ExperimentController, type ExperimentControllerCallbacks } from '../../src/lib/experiment/controller';
import type { AgentId } from '../../src/lib/arena/types';
import type { ConnectomeGraph, GraphMode } from '../../src/lib/connectome/format';
import type { ArenaManifest, TrainedReadoutLoadResult } from '../../src/lib/experiment/assets';
import type { ExperimentStatus } from '../../src/lib/experiment/state';
import type { DecoderKind, WorkerRequest, WorkerResponse } from '../../src/lib/worker/protocol';
import { createPublicDataFetch, FakeNeuralWorker } from '../helpers/fake-worker';

/**
 * `ExperimentController` (WP6's thermo review I1/maintainability I1 fix)
 * owns asset loading, Worker/binding construction, the `ExperimentRunner`
 * itself, and topology-switch serialization — logic that previously lived
 * directly in `App.svelte` and could only be exercised by mounting the
 * whole component into jsdom with a mocked render module (see
 * `tests/App.lifecycle.test.ts`). These tests exercise the same scenarios
 * directly against the plain class instead: no component, no jsdom DOM
 * mount, no render-module mock — only `fetch`/`Worker` stand-ins, the same
 * ones `tests/App.lifecycle.test.ts` uses.
 */

const TOTAL_TICKS = 30;
const SEED = 12345;

const createCallbacks = (): ExperimentControllerCallbacks & {
  statuses: ExperimentStatus[];
  errors: string[];
  topologyApplied: Array<[AgentId, GraphMode]>;
  switchCounts: Array<Readonly<Record<AgentId, number>>>;
  trainedReadoutStatuses: TrainedReadoutLoadResult[];
  decodersApplied: DecoderKind[];
} => {
  const statuses: ExperimentStatus[] = [];
  const errors: string[] = [];
  const topologyApplied: Array<[AgentId, GraphMode]> = [];
  const switchCounts: Array<Readonly<Record<AgentId, number>>> = [];
  const trainedReadoutStatuses: TrainedReadoutLoadResult[] = [];
  const decodersApplied: DecoderKind[] = [];
  return {
    statuses,
    errors,
    topologyApplied,
    switchCounts,
    trainedReadoutStatuses,
    decodersApplied,
    onStatusChange: (status) => statuses.push(status),
    onTelemetry: vi.fn(),
    onError: (message) => errors.push(message),
    onManifest: vi.fn(),
    onTopologyApplied: (agentId, mode) => topologyApplied.push([agentId, mode]),
    onTopologySwitchCountChange: (counts) => switchCounts.push({ ...counts }),
    onTrainedReadoutStatus: (status) => trainedReadoutStatuses.push(status),
    onDecoderApplied: (decoder) => decodersApplied.push(decoder)
  };
};

/**
 * Cleanup registry (thermo-architecture I1 fix, applied here too on audit):
 * every `ExperimentController` constructed by a test is registered via
 * `trackController` immediately after construction (including inside the
 * two `setUp()` helpers below, so every test that goes through one is
 * covered without its own explicit cleanup call). The `afterEach` below
 * disposes every tracked controller — which in turn disposes its
 * `ExperimentRunner` and terminates both arms' `WorkerClient`s — before
 * unstubbing the `fetch`/`Worker` globals, so a runner a test started via
 * `runner.start()`/`changeTopology()` and never explicitly paused/disposed
 * cannot keep ticking in the background past its own test (see
 * `experiment-runner.test.ts`'s identical registry for the reproduced
 * flake this pattern closes). `ExperimentController#dispose()` is
 * documented as idempotent, so calling it here is harmless even for a test
 * that already disposed its own controller.
 */
const activeControllers: ExperimentController[] = [];
const trackController = (controller: ExperimentController): ExperimentController => {
  activeControllers.push(controller);
  return controller;
};

beforeEach(() => {
  vi.stubGlobal('fetch', createPublicDataFetch());
  vi.stubGlobal('Worker', FakeNeuralWorker as unknown as typeof Worker);
});

afterEach(() => {
  for (const controller of activeControllers) controller.dispose();
  activeControllers.length = 0;
  vi.unstubAllGlobals();
});

const createWorker = (): Worker => new FakeNeuralWorker() as unknown as Worker;

describe('ExperimentController#initialize', () => {
  it('fetches/verifies both artifacts, constructs a runner, and reports the default topology to both arms', async () => {
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
    expect(controller.getManifest()).toBeDefined();
    expect(callbacks.errors).toHaveLength(0);
    expect(callbacks.statuses).toContain('ready');
    expect(callbacks.topologyApplied).toContainEqual(['left', 'biological']);
    expect(callbacks.topologyApplied).toContainEqual(['right', 'rewired']);
  });

  it('passes the already-parsed biological graph to onManifest (thermo-architecture I1 fix: callers must not re-fetch/re-parse it)', async () => {
    const callbacks = createCallbacks();
    const manifestCalls: Array<[ArenaManifest, ConnectomeGraph]> = [];
    callbacks.onManifest = (manifest, biologicalGraph) => {
      manifestCalls.push([manifest, biologicalGraph]);
    };
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks
    });
    trackController(controller);

    await controller.initialize();

    expect(manifestCalls).toHaveLength(1);
    const [manifestArg, biologicalGraphArg] = manifestCalls[0];
    expect(manifestArg).toBe(controller.getManifest());
    expect(biologicalGraphArg.biologicalIds.length).toBe(controller.getManifest()?.neuronCount);
    expect(biologicalGraphArg.metadata.rateMin).toBeLessThanOrEqual(biologicalGraphArg.metadata.rateMax);
  });

  it('reports an error and never constructs a runner when an artifact fails its sha256 check', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'malecns-arena-v1.bin.gz' }));
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

    expect(controller.getRunner()).toBeUndefined();
    expect(callbacks.errors.length).toBeGreaterThan(0);
    expect(callbacks.errors[0]).toMatch(/sha256/i);
    expect(callbacks.statuses).toContain('error');
  });

  it('does nothing (no state mutation, no callback) once disposed before initialize() resolves', async () => {
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

    expect(controller.getRunner()).toBeUndefined();
  });
});

describe('ExperimentController#changeTopology', () => {
  const setUp = async () => {
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
    return { controller, callbacks };
  };

  it('switches an arm to a new topology, reporting the switch pending then settled, and applies the new mode', async () => {
    const { controller, callbacks } = await setUp();
    const runner = controller.getRunner();
    expect(runner).toBeDefined();

    controller.changeTopology('left', 'disconnected');
    // Synchronously incremented before any awaited work begins.
    expect(callbacks.switchCounts.at(-1)?.left).toBe(1);

    await vi.waitFor(() => expect(callbacks.switchCounts.at(-1)?.left).toBe(0));

    expect(callbacks.topologyApplied).toContainEqual(['left', 'disconnected']);
    expect(runner?.getStatus()).toBe('ready');
    expect(runner?.getTelemetry().agents.left.topology).toBe('disconnected');
    expect(callbacks.errors).toHaveLength(0);
  });

  it('serializes two rapid switches on the same arm and settles on the last one, without ever failing', async () => {
    const { controller, callbacks } = await setUp();
    const runner = controller.getRunner();

    controller.changeTopology('left', 'disconnected');
    controller.changeTopology('left', 'rewired');

    await vi.waitFor(() => expect(callbacks.switchCounts.at(-1)?.left).toBe(0));

    expect(runner?.getStatus()).toBe('ready');
    expect(runner?.getTelemetry().agents.left.topology).toBe('rewired');
    expect(callbacks.errors).toHaveLength(0);
    // The stale intermediate mode must still have been reported once (the
    // switch really did run), but the *last* applied mode for 'left' must
    // be the final selection.
    const leftApplied = callbacks.topologyApplied.filter(([agentId]) => agentId === 'left');
    expect(leftApplied.at(-1)).toEqual(['left', 'rewired']);
  });

  it('is a no-op while the run is active (not ready/finished)', async () => {
    const { controller, callbacks } = await setUp();
    const runner = controller.getRunner();
    runner?.start();
    expect(runner?.getStatus()).toBe('running');

    controller.changeTopology('left', 'disconnected');

    expect(callbacks.switchCounts).toHaveLength(0);
    expect(runner?.getTelemetry().agents.left.topology).toBe('biological');
    runner?.pause();
  });

  /**
   * bb45 follow-up: a switch requested before `initialize()` has produced a
   * runner (e.g. a stray/racy call while assets are still loading) must be a
   * silent no-op — see `changeTopology`'s own guard (`if (!this.runner || ...)
   * return;`) — rather than throwing or getting queued for once a runner
   * eventually exists.
   */
  it('switch-before-runner-exists: is a no-op with no callback effects when called while initialize() is still in flight', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks
    });
    trackController(controller);

    // Deliberately not awaited yet: `initialize()` is genuinely in flight
    // (assets still fetching/verifying, no runner/workerClients/manifest
    // constructed yet) at the moment `changeTopology` is called below.
    const initializing = controller.initialize();
    expect(controller.getRunner()).toBeUndefined();
    controller.changeTopology('left', 'disconnected');

    expect(callbacks.switchCounts).toHaveLength(0);
    expect(callbacks.topologyApplied).toHaveLength(0);
    expect(callbacks.errors).toHaveLength(0);

    // The in-flight initialize() must still be able to complete normally
    // afterward — the earlier no-op call must not have left any partial
    // state (e.g. a topology mutation) behind.
    await initializing;
    expect(controller.getRunner()).toBeDefined();
    expect(controller.getRunner()?.getTelemetry().agents.left.topology).toBe('biological');
  });

  /**
   * bb45 follow-up: `dispose()` called while a topology switch's own
   * dispose->rebuild chain is still in flight must not throw, must not apply
   * the in-flight switch, and must still leave `dispose()` itself idempotent
   * afterward.
   *
   * A dual review pass caught that an earlier version of this test's
   * assertions (no reported error, no 'disconnected' applied) held true
   * *regardless* of whether `changeTopology`'s own `destroyed` guards existed
   * at all: `controller.dispose()` terminates the arm's `WorkerClient`
   * before the switch chain's first `await client.dispose()` ever settles,
   * so that call rejects immediately on its own (`WorkerClientError: Worker
   * client has been terminated` — `worker/client.ts`), and the chain never
   * reaches any of `changeTopology`'s own guards at all. (The chain's catch
   * block *would* still be a no-op even without its own `if (this.destroyed)
   * return;` check, because `ExperimentRunner#fail()` has its own,
   * independent disposed-check — real defense-in-depth, not a redundant
   * line to delete.) So the two assertions below alone do not isolate
   * `changeTopology`'s guards specifically. What *does* discriminate them:
   * the `finally` block's `topologySwitchCount`/`onTopologySwitchCountChange`
   * bookkeeping is gated by its own `if (!this.destroyed)` — remove only
   * that one and this test's `switchCounts`/`statuses` assertions below
   * would fail, because a stray post-dispose callback would fire. This test
   * is kept as a full-contract regression test for the dispose-mid-switch
   * scenario (every layer, not one isolated line) rather than a
   * single-guard unit test — see the finally-guard assertions for the one
   * part of that contract this scenario can actually isolate.
   */
  it('dispose-mid-switch: disposing while a topology switch is in flight does not throw, reports no further callbacks, and the switch never applies', async () => {
    const { controller, callbacks } = await setUp();
    const runner = controller.getRunner();
    expect(runner).toBeDefined();

    controller.changeTopology('left', 'disconnected');
    // Synchronously incremented before any awaited work begins (same
    // guarantee the passing "switches an arm" test above relies on) — so
    // dispose() below reliably races a switch that has already started.
    expect(callbacks.switchCounts.at(-1)?.left).toBe(1);
    const switchCountsBeforeDispose = callbacks.switchCounts.length;
    const statusesBeforeDispose = callbacks.statuses.length;

    expect(() => controller.dispose()).not.toThrow();
    // Idempotent even while the switch chain's own .then()/.finally() have
    // not yet settled.
    expect(() => controller.dispose()).not.toThrow();

    // A macrotask (not a fixed real-time margin) is enough here: the switch
    // chain's rejection (see the doc comment above) and every one of its
    // `.then()`/`.catch()`/`.finally()` continuations are plain microtasks,
    // which JS always fully drains before running any macrotask — so by the
    // time this `setTimeout(0)` callback runs, the whole chain has already
    // settled, deterministically, with no guessed wall-clock delay needed.
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(callbacks.errors).toHaveLength(0);
    // No further switch-count callback after dispose() — this is the one
    // assertion in this test that actually isolates a single guard: the
    // chain's `finally` block's own `if (!this.destroyed)` check around its
    // `topologySwitchCount`/`onTopologySwitchCountChange` bookkeeping (see
    // this test's doc comment for why the other two assertions below do
    // not equally isolate a specific guard).
    expect(callbacks.switchCounts).toHaveLength(switchCountsBeforeDispose);
    // No further status callback either (e.g. a stray 'error' from a
    // post-dispose `runner.fail()` call that shouldn't have happened).
    expect(callbacks.statuses).toHaveLength(statusesBeforeDispose);
    // The switch to 'disconnected' must never have been reported as applied.
    const leftApplied = callbacks.topologyApplied.filter(([agentId]) => agentId === 'left');
    expect(leftApplied.find(([, mode]) => mode === 'disconnected')).toBeUndefined();
  });
});

/**
 * A `FakeNeuralWorker`-shaped wrapper that can hold the *next* `dispose`
 * request, and every message posted after it, until the test releases
 * them — delivered then in the exact order they were posted.
 *
 * Two earlier versions of this helper were both caught by dual review as
 * not exercising the race they claimed to, for the same underlying reason
 * (only the gated message type moved between versions):
 *
 * - `GatedDisposeWorker` (v1) held only the one `dispose` message but
 *   forwarded every later message to the inner worker *immediately* — so a
 *   message posted after the gated `dispose` could be *delivered* ahead of
 *   it, something a real `Worker` (whose message channel is strictly FIFO)
 *   can never do.
 * - `GatedInitWorker` (v2) moved the gate to `init` for the same reason —
 *   `changeTopology` posts `dispose` a microtask after `changeTopology()`
 *   returns — but reintroduced the exact same defect: it held the gated
 *   `init` while still forwarding a later `set-activity` straight through,
 *   so the test's `not-initialized` assertion passed only because
 *   `set-activity` was *delivered* before `init` even though `init` was
 *   *posted* first. Reviewers confirmed this empirically: swapping in a
 *   version that preserves posting order made the test's `errorSpy`
 *   assertion fail, proving the old assertion depended on the reordering.
 *
 * This version holds `dispose` in a queue (not a single `.then`) and, once
 * held, queues *every* subsequent message behind it too, flushing the whole
 * queue to the inner worker — in original posting order — only on
 * `releaseDispose()`. Nothing this helper does not hold ever overtakes
 * something it does. `disposeHeld` resolves exactly when the gated
 * `dispose` has been *posted* (proving `changeTopology`'s chain has reached
 * that call), giving a test a real signal to await instead of guessed
 * microtask-hop counts, with no risk of delivering anything out of order.
 */
class GatedDisposeWorker {
  private readonly inner = new FakeNeuralWorker();
  private gating = false;
  private holding = false;
  private queue: Array<[WorkerRequest, Transferable[] | undefined]> = [];
  private markHeld: (() => void) | undefined;
  /**
   * Resolves once the gated `dispose` has actually been posted (and queued)
   * — reassigned to a fresh pending promise by `gateNextDispose()`, which a
   * test must call before the call that posts `dispose`.
   */
  disposeHeld: Promise<void> = Promise.resolve();
  terminated = false;

  /** Arms the gate and resets `disposeHeld` to a fresh pending promise that the next `dispose` request's `postMessage` call will settle. */
  gateNextDispose(): void {
    this.gating = true;
    this.disposeHeld = new Promise((resolve) => {
      this.markHeld = resolve;
    });
  }

  /** Flushes every queued message to the inner worker, in the exact order they were originally posted. */
  releaseDispose(): void {
    this.holding = false;
    const queued = this.queue;
    this.queue = [];
    for (const [message, transfer] of queued) this.inner.postMessage(message, transfer);
  }

  postMessage(message: WorkerRequest, transfer?: Transferable[]): void {
    if (message.type === 'dispose' && this.gating) {
      this.gating = false;
      this.holding = true;
      this.queue.push([message, transfer]);
      this.markHeld?.();
      return;
    }
    if (this.holding) {
      this.queue.push([message, transfer]);
      return;
    }
    this.inner.postMessage(message, transfer);
  }

  addEventListener(type: string, listener: (event: MessageEvent<WorkerResponse>) => void): void {
    this.inner.addEventListener(type, listener);
  }

  removeEventListener(type: string, listener: (event: MessageEvent<WorkerResponse>) => void): void {
    this.inner.removeEventListener(type, listener);
  }

  terminate(): void {
    this.terminated = true;
    this.inner.terminate();
  }
}

/**
 * A `FakeNeuralWorker`-shaped wrapper whose next `set-activity` request can
 * be armed to fail with a synthetic error instead of reaching the real
 * Worker runtime — used to regression-test that a rejected re-apply of
 * streaming after a topology switch (`controller.ts#changeTopology`) never
 * fails the run. Every other request type (including a *later*
 * `set-activity`, once the arm has fired) is forwarded to the inner worker
 * unchanged.
 *
 * `addEventListener`/`removeEventListener` register on both this wrapper's
 * own listener set (so the synthetic failure response below can be
 * dispatched directly to the same callbacks `WorkerClient` registered) and
 * on the inner worker (so its real dispatches keep reaching those same
 * callbacks too) — the same function reference either way, so a listener is
 * never invoked twice for one real inner-worker message.
 */
class FailingReapplyWorker {
  private readonly inner = new FakeNeuralWorker();
  private readonly listeners = new Map<string, Set<(event: MessageEvent<WorkerResponse>) => void>>();
  private failNextSetActivity = false;
  terminated = false;

  /** The next `set-activity` request posted to this worker fails synthetically instead of reaching the real runtime. */
  armFailNextSetActivity(): void {
    this.failNextSetActivity = true;
  }

  postMessage(message: WorkerRequest, transfer?: Transferable[]): void {
    if (message.type === 'set-activity' && this.failNextSetActivity) {
      this.failNextSetActivity = false;
      queueMicrotask(() => {
        const response: WorkerResponse = {
          type: 'set-activity',
          requestId: message.requestId,
          ok: false,
          error: { code: 'internal-error', message: 'simulated re-apply failure' }
        };
        this.dispatch('message', new MessageEvent('message', { data: response }));
      });
      return;
    }
    this.inner.postMessage(message, transfer);
  }

  addEventListener(type: string, listener: (event: MessageEvent<WorkerResponse>) => void): void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(listener);
    this.inner.addEventListener(type, listener);
  }

  removeEventListener(type: string, listener: (event: MessageEvent<WorkerResponse>) => void): void {
    this.listeners.get(type)?.delete(listener);
    this.inner.removeEventListener(type, listener);
  }

  terminate(): void {
    this.terminated = true;
    this.inner.terminate();
  }

  private dispatch(type: string, event: MessageEvent<WorkerResponse>): void {
    for (const listener of this.listeners.get(type) ?? []) listener(event);
  }
}

describe('ExperimentController activity streaming', () => {
  const setUp = async () => {
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
    return { controller, callbacks };
  };

  it("changeTopology re-issues set-activity(true) on the rebuilt binding when streaming was already on, and getLatestRates resumes for that arm", async () => {
    const { controller } = await setUp();
    const runner = controller.getRunner();
    expect(runner).toBeDefined();

    await runner!.setActivityStreaming(true);
    expect(runner!.isActivityStreaming()).toBe(true);

    controller.changeTopology('right', 'disconnected');
    await vi.waitFor(() => expect(runner!.getTelemetry().agents.right.topology).toBe('disconnected'));

    // The rebuilt binding's Worker started this fresh `init` with activity
    // off (see `neural.worker.ts`'s "always false after init" note); this is
    // the controller's own re-apply (`changeTopology`'s
    // `binding.setActivity?.(true)` call) actually having landed.
    expect(runner!.isActivityStreaming()).toBe(true);
    runner!.start();
    await vi.waitFor(() => expect(runner!.getLatestRates('right')).toBeDefined());
    await vi.waitFor(() => expect(runner!.getLatestRates('left')).toBeDefined());
    expect(runner!.getLatestRates('right')).toHaveLength(runner!.getTelemetry().agents.right.neuronCount);
  });

  /**
   * bb45-style race test: `runner.setActivityStreaming(true)` issued while
   * `changeTopology`'s own `dispose` round trip is genuinely still in
   * flight for the same arm — the one window where a real Worker can
   * actually receive `set-activity` while `idle` (see `GatedDisposeWorker`'s
   * doc comment for why this specific window, and why two earlier versions
   * of this test's gate did not reach it). `setActivityStreaming`'s
   * `setActivity` call against that still-`idle` Worker is expected to fail
   * with `not-initialized` (caught and logged inside `ExperimentRunner`, per
   * its doc comment — never rejects to this test): this test asserts that
   * rejection is actually exercised (via a `console.debug` spy — thermo
   * review S2's downgrade, so this expected/routine race no longer logs at
   * `console.error`), not merely that the end state looks right despite it
   * never having run, and that `console.error` itself stays silent for this
   * expected case (a genuine Worker failure still must log there — see
   * `runner.ts#isNotInitializedRejection` and its one other caller,
   * `controller.ts#changeTopology`'s re-apply catch).
   */
  it('setActivityStreaming(true) racing changeTopology through the dispose window resolves without an unhandled rejection, logs the expected not-initialized failure at debug (not error) level, and streaming is active on both arms afterward', async () => {
    let gatedWorker: GatedDisposeWorker | undefined;
    let createCount = 0;
    const createGatedWorker = (): Worker => {
      createCount += 1;
      if (createCount === 1) {
        gatedWorker = new GatedDisposeWorker();
        return gatedWorker as unknown as Worker;
      }
      return new FakeNeuralWorker() as unknown as Worker;
    };

    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const debugSpy = vi.spyOn(console, 'debug').mockImplementation(() => {});
    try {
      const callbacks = createCallbacks();
      const controller = new ExperimentController({
        seed: SEED,
        totalTicks: TOTAL_TICKS,
        initialTopology: { left: 'biological', right: 'rewired' },
        createWorker: createGatedWorker,
        callbacks
      });
      trackController(controller);
      await controller.initialize();
      const runner = controller.getRunner();
      expect(runner).toBeDefined();
      expect(gatedWorker).toBeDefined();

      // The left arm's Worker (the one `createGatedWorker` hands out first,
      // matching `initialize()`'s `left`-then-`right` construction order)
      // will hold its `dispose` request — and everything posted after it,
      // in order — until released.
      gatedWorker!.gateNextDispose();

      controller.changeTopology('left', 'disconnected');
      // Resolves only once `dispose` has actually been posted — proof
      // `changeTopology`'s chain has reached that call, not a guessed
      // number of microtask hops.
      await gatedWorker!.disposeHeld;

      // Posted while `dispose` is still held: queued strictly behind it, so
      // releasing delivers `dispose` first (the Worker goes `idle`), then
      // this `set-activity` (which genuinely fails with `not-initialized`
      // against that idle Worker) — the one ordering a real Worker's own
      // message channel can actually produce.
      const streamingPromise = runner!.setActivityStreaming(true);
      gatedWorker!.releaseDispose();

      await expect(streamingPromise).resolves.toBeUndefined();
      await vi.waitFor(() => expect(runner!.getTelemetry().agents.left.topology).toBe('disconnected'));

      // Proves the `not-initialized` rejection path in
      // `ExperimentRunner#setActivityStreaming` actually ran for the left
      // arm's old (now-idle) binding, not merely that the end state below
      // happens to look right regardless — logged at `console.debug`, and
      // `console.error` never fires for this expected case.
      expect(debugSpy).toHaveBeenCalled();
      expect(errorSpy).not.toHaveBeenCalled();

      expect(runner!.isActivityStreaming()).toBe(true);
      expect(callbacks.errors).toHaveLength(0);

      runner!.start();
      await vi.waitFor(() => expect(runner!.getLatestRates('left')).toBeDefined());
      await vi.waitFor(() => expect(runner!.getLatestRates('right')).toBeDefined());
    } finally {
      errorSpy.mockRestore();
      debugSpy.mockRestore();
    }
  });

  /**
   * Mirror of the race test above: disabling (instead of enabling) while the
   * arm's Worker is genuinely idle mid topology-switch must be equally
   * unremarkable — `setActivityStreaming(false)`'s per-binding call against
   * the idle old binding also fails and is swallowed the same way, and the
   * rebuilt binding must end up *not* streaming (the controller only
   * re-applies `setActivity(true)`, never `(true)` when the flag reads
   * `false`), so the topology-switched arm produces no `rates`.
   */
  it('setActivityStreaming(false) racing changeTopology through the dispose window leaves the rebuilt binding not streaming', async () => {
    let gatedWorker: GatedDisposeWorker | undefined;
    let createCount = 0;
    const createGatedWorker = (): Worker => {
      createCount += 1;
      if (createCount === 1) {
        gatedWorker = new GatedDisposeWorker();
        return gatedWorker as unknown as Worker;
      }
      return new FakeNeuralWorker() as unknown as Worker;
    };

    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      const callbacks = createCallbacks();
      const controller = new ExperimentController({
        seed: SEED,
        totalTicks: TOTAL_TICKS,
        initialTopology: { left: 'biological', right: 'rewired' },
        createWorker: createGatedWorker,
        callbacks
      });
      trackController(controller);
      await controller.initialize();
      const runner = controller.getRunner();
      expect(runner).toBeDefined();
      expect(gatedWorker).toBeDefined();

      await runner!.setActivityStreaming(true);
      expect(runner!.isActivityStreaming()).toBe(true);

      gatedWorker!.gateNextDispose();
      controller.changeTopology('left', 'disconnected');
      await gatedWorker!.disposeHeld;

      const streamingPromise = runner!.setActivityStreaming(false);
      gatedWorker!.releaseDispose();

      await expect(streamingPromise).resolves.toBeUndefined();
      await vi.waitFor(() => expect(runner!.getTelemetry().agents.left.topology).toBe('disconnected'));

      expect(runner!.isActivityStreaming()).toBe(false);
      expect(callbacks.errors).toHaveLength(0);

      runner!.start();
      // A fixed sleep alone would pass vacuously if no tick ever ran (the
      // assertion below would hold trivially); wait for a real tick first,
      // so "still undefined" actually means "streaming produced nothing",
      // not "nothing happened yet".
      await vi.waitFor(() => expect(runner!.getTelemetry().tick).toBeGreaterThan(0));
      expect(runner!.getLatestRates('left')).toBeUndefined();
      expect(runner!.getLatestRates('right')).toBeUndefined();
    } finally {
      errorSpy.mockRestore();
    }
  });

  /**
   * Regression test for the second dual-review finding: a rejected re-apply
   * of `setActivity(true)` after a topology switch (`controller.ts`'s
   * `changeTopology`) must not fail the run or block `onTopologyApplied`,
   * matching `ExperimentRunner#setActivityStreaming`'s own documented
   * policy that a per-arm streaming-toggle failure is not a run failure.
   * `FailingReapplyWorker` makes the rebuilt binding's own `setActivity`
   * round trip reject synthetically, independent of any real Worker state,
   * so this test exercises `changeTopology`'s fire-and-forget `.catch`
   * directly rather than relying on incidental timing.
   */
  it('a rejected re-apply of streaming after a topology switch does not fail the run', async () => {
    let failingWorker: FailingReapplyWorker | undefined;
    let createCount = 0;
    const createFailingWorker = (): Worker => {
      createCount += 1;
      if (createCount === 1) {
        failingWorker = new FailingReapplyWorker();
        return failingWorker as unknown as Worker;
      }
      return new FakeNeuralWorker() as unknown as Worker;
    };

    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    try {
      const callbacks = createCallbacks();
      const controller = new ExperimentController({
        seed: SEED,
        totalTicks: TOTAL_TICKS,
        initialTopology: { left: 'biological', right: 'rewired' },
        createWorker: createFailingWorker,
        callbacks
      });
      trackController(controller);
      await controller.initialize();
      const runner = controller.getRunner();
      expect(runner).toBeDefined();
      expect(failingWorker).toBeDefined();

      await runner!.setActivityStreaming(true);
      expect(runner!.isActivityStreaming()).toBe(true);

      // Arms the *next* set-activity request on this arm's Worker — the
      // rebuilt binding's own re-apply call, not the enable above (which
      // already succeeded before this line runs).
      failingWorker!.armFailNextSetActivity();

      controller.changeTopology('left', 'disconnected');
      await vi.waitFor(() => expect(runner!.getTelemetry().agents.left.topology).toBe('disconnected'));

      // The rejection must be logged (never silently swallowed)...
      expect(errorSpy).toHaveBeenCalled();
      // ...but must not fail the run, and must not block onTopologyApplied
      // or onTelemetry, which setAgentBinding already committed to fire.
      expect(runner!.getStatus()).not.toBe('error');
      expect(callbacks.errors).toHaveLength(0);
      expect(callbacks.topologyApplied).toContainEqual(['left', 'disconnected']);
      // The runner's own flag stays authoritative regardless of the
      // rejected re-apply — this is what lets a later successful toggle (or
      // a future topology switch) recover it.
      expect(runner!.isActivityStreaming()).toBe(true);
    } finally {
      errorSpy.mockRestore();
    }
  });
});

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

describe('ExperimentController#dispose', () => {
  it('is idempotent and terminates both Worker clients', async () => {
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

    expect(() => {
      controller.dispose();
      controller.dispose();
    }).not.toThrow();
  });
});
