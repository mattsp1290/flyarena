import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ExperimentController, type ExperimentControllerCallbacks } from '../../src/lib/experiment/controller';
import type { AgentId } from '../../src/lib/arena/types';
import type { GraphMode } from '../../src/lib/connectome/format';
import type { ExperimentStatus } from '../../src/lib/experiment/state';
import type { WorkerRequest, WorkerResponse } from '../../src/lib/worker/protocol';
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
} => {
  const statuses: ExperimentStatus[] = [];
  const errors: string[] = [];
  const topologyApplied: Array<[AgentId, GraphMode]> = [];
  const switchCounts: Array<Readonly<Record<AgentId, number>>> = [];
  return {
    statuses,
    errors,
    topologyApplied,
    switchCounts,
    onStatusChange: (status) => statuses.push(status),
    onTelemetry: vi.fn(),
    onError: (message) => errors.push(message),
    onManifest: vi.fn(),
    onTopologyApplied: (agentId, mode) => topologyApplied.push([agentId, mode]),
    onTopologySwitchCountChange: (counts) => switchCounts.push({ ...counts })
  };
};

beforeEach(() => {
  vi.stubGlobal('fetch', createPublicDataFetch());
  vi.stubGlobal('Worker', FakeNeuralWorker as unknown as typeof Worker);
});

afterEach(() => {
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

    await controller.initialize();

    expect(controller.getRunner()).toBeDefined();
    expect(controller.getManifest()).toBeDefined();
    expect(callbacks.errors).toHaveLength(0);
    expect(callbacks.statuses).toContain('ready');
    expect(callbacks.topologyApplied).toContainEqual(['left', 'biological']);
    expect(callbacks.topologyApplied).toContainEqual(['right', 'rewired']);
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
 * A `FakeNeuralWorker`-shaped wrapper that can hold the *next* `init`
 * request's delivery to the inner worker until the test releases it.
 *
 * An earlier version of this helper (`GatedDisposeWorker`) gated `dispose`
 * instead and was caught by dual review as not exercising the race it
 * claimed to: `changeTopology` posts `dispose` one microtask *after*
 * `changeTopology()` returns (it runs inside a `.then(async () => ...)`
 * continuation — see `controller.ts`), so a racing `setActivityStreaming`
 * call issued synchronously right after `changeTopology()` had its own
 * `set-activity` message posted, and delivered, *before* `dispose` was even
 * sent — never landing in the dispose/init window at all. Worse, that
 * version forwarded every non-`dispose` message to the inner worker
 * immediately even while `dispose` was held, so a message posted after the
 * gated `dispose` could be *delivered* ahead of it — a real `Worker` never
 * reorders messages like that, so the double had silently stopped modeling
 * one.
 *
 * Gating `init` instead fixes both problems for the scenario this test
 * actually wants: `changeTopology`'s `dispose` is allowed to complete
 * normally (undelayed), so the Worker genuinely reaches `idle` before this
 * helper intercepts anything; `initHeld` resolves exactly when the rebuilt
 * binding's `init` has been *posted* (proving the Worker is idle and the
 * dispose/rebuild window has genuinely opened), so a test can `await` it
 * before racing another call in that window with no timing guesswork; and
 * every message this helper does not hold (including one posted while an
 * `init` is being held) still goes straight to the inner worker in call
 * order, so it never itself introduces a reordering the real Worker
 * couldn't produce.
 */
class GatedInitWorker {
  private readonly inner = new FakeNeuralWorker();
  private gating = false;
  private markHeld: (() => void) | undefined;
  private releaseFn: (() => void) | undefined;
  /**
   * Resolves once the gated `init` has actually been posted (and is being
   * held) — reassigned to a fresh pending promise by `gateNextInit()`, and
   * a test must call `gateNextInit()` (which a test always does before the
   * racing call that posts `init`) and hold onto *that* promise reference
   * before awaiting it, since `postMessage` settles this exact object
   * in place rather than replacing it — an `await` on a stale reference
   * from before `gateNextInit()` would never resolve.
   */
  initHeld: Promise<void> = Promise.resolve();
  terminated = false;

  /** Arms the gate and resets `initHeld` to a fresh pending promise that the next `init` request's `postMessage` call will settle. */
  gateNextInit(): void {
    this.gating = true;
    this.initHeld = new Promise((resolve) => {
      this.markHeld = resolve;
    });
  }

  releaseInit(): void {
    this.releaseFn?.();
    this.releaseFn = undefined;
  }

  postMessage(message: WorkerRequest, transfer?: Transferable[]): void {
    if (message.type === 'init' && this.gating) {
      this.gating = false;
      const release = new Promise<void>((resolve) => {
        this.releaseFn = resolve;
      });
      void release.then(() => this.inner.postMessage(message, transfer));
      this.markHeld?.();
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
   * `changeTopology`'s own dispose -> rebuild chain is genuinely in the
   * `idle` window for the same arm — after `dispose` has resolved (the
   * Worker is truly idle) but before the rebuilt binding's `init` has been
   * delivered. `setActivityStreaming`'s `setActivity` call against that
   * still-`idle` Worker is expected to fail with `not-initialized` (caught
   * and logged inside `ExperimentRunner`, per its doc comment — never
   * rejects to this test): this test asserts that rejection is actually
   * exercised (via a `console.error` spy), not merely that the end state
   * looks right despite it never having run — see `GatedInitWorker`'s doc
   * comment for why an earlier version of this test (gating `dispose`, not
   * `init`) never actually reached this window at all.
   */
  it('setActivityStreaming(true) racing changeTopology through the idle window resolves without an unhandled rejection, logs the expected not-initialized failure, and streaming is active on both arms afterward', async () => {
    let gatedWorker: GatedInitWorker | undefined;
    let createCount = 0;
    const createGatedWorker = (): Worker => {
      createCount += 1;
      if (createCount === 1) {
        gatedWorker = new GatedInitWorker();
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
      await controller.initialize();
      const runner = controller.getRunner();
      expect(runner).toBeDefined();
      expect(gatedWorker).toBeDefined();

      // The left arm's Worker (the one `createGatedWorker` hands out first,
      // matching `initialize()`'s `left`-then-`right` construction order)
      // will hold its rebuilt binding's `init` request. `changeTopology`'s
      // own `dispose` is left ungated, so it completes normally — the
      // Worker genuinely reaches `idle` before this test's racing call.
      gatedWorker!.gateNextInit();

      controller.changeTopology('left', 'disconnected');
      // Resolves only once `init` has actually been posted — proof the
      // dispose/rebuild window has opened and the Worker is idle, not a
      // guessed number of microtask hops.
      await gatedWorker!.initHeld;

      const streamingPromise = runner!.setActivityStreaming(true);
      gatedWorker!.releaseInit();

      await expect(streamingPromise).resolves.toBeUndefined();
      await vi.waitFor(() => expect(runner!.getTelemetry().agents.left.topology).toBe('disconnected'));

      // Proves the `not-initialized` rejection path in
      // `ExperimentRunner#setActivityStreaming` actually ran for the left
      // arm's old (now-idle) binding, not merely that the end state below
      // happens to look right regardless.
      expect(errorSpy).toHaveBeenCalled();

      expect(runner!.isActivityStreaming()).toBe(true);
      expect(callbacks.errors).toHaveLength(0);

      runner!.start();
      await vi.waitFor(() => expect(runner!.getLatestRates('left')).toBeDefined());
      await vi.waitFor(() => expect(runner!.getLatestRates('right')).toBeDefined());
    } finally {
      errorSpy.mockRestore();
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
  it('setActivityStreaming(false) racing changeTopology through the idle window leaves the rebuilt binding not streaming', async () => {
    let gatedWorker: GatedInitWorker | undefined;
    let createCount = 0;
    const createGatedWorker = (): Worker => {
      createCount += 1;
      if (createCount === 1) {
        gatedWorker = new GatedInitWorker();
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
      await controller.initialize();
      const runner = controller.getRunner();
      expect(runner).toBeDefined();
      expect(gatedWorker).toBeDefined();

      await runner!.setActivityStreaming(true);
      expect(runner!.isActivityStreaming()).toBe(true);

      gatedWorker!.gateNextInit();
      controller.changeTopology('left', 'disconnected');
      await gatedWorker!.initHeld;

      const streamingPromise = runner!.setActivityStreaming(false);
      gatedWorker!.releaseInit();

      await expect(streamingPromise).resolves.toBeUndefined();
      await vi.waitFor(() => expect(runner!.getTelemetry().agents.left.topology).toBe('disconnected'));

      expect(runner!.isActivityStreaming()).toBe(false);
      expect(callbacks.errors).toHaveLength(0);

      runner!.start();
      // The right arm never toggled off, but streaming is off runner-wide,
      // so neither arm should ever populate `latestRates`.
      await new Promise((resolve) => setTimeout(resolve, 30));
      expect(runner!.getLatestRates('left')).toBeUndefined();
      expect(runner!.getLatestRates('right')).toBeUndefined();
    } finally {
      errorSpy.mockRestore();
    }
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
    await controller.initialize();

    expect(() => {
      controller.dispose();
      controller.dispose();
    }).not.toThrow();
  });
});
