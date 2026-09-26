import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/svelte';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
// `vi.mock` calls are hoisted above imports by Vitest's transform, so this
// static import below receives the mocked module even though it is
// declared after the vi.mock() call in source order.
import App from '../src/App.svelte';
import { createPublicDataFetch, FakeNeuralWorker } from './helpers/fake-worker';

/**
 * jsdom has no WebGL, so a real `ArenaScene` always fails to construct
 * (see tests/App.test.ts's fallback-message test) — which means no test
 * using the real class can ever exercise App.svelte's cleanup paths
 * (dispose-on-unmount, dispose-on-context-loss). Mock the module so those
 * paths can actually be driven and asserted on, same approach as before
 * this bean replaced the placeholder demo loop with the closed-loop runner.
 */
interface MockArenaSceneOptions {
  onContextLost?: (info: { reason: string }) => void;
  onFrame?: (telemetry: { fps: number; frameMs: number }) => void;
}

const instances: Array<{
  options: MockArenaSceneOptions;
  update: ReturnType<typeof vi.fn>;
  dispose: ReturnType<typeof vi.fn>;
  setReducedMotion: ReturnType<typeof vi.fn>;
  setAgentTopology: ReturnType<typeof vi.fn>;
}> = [];

vi.mock('../src/lib/render/ArenaScene', () => {
  class ArenaSceneUnavailableError extends Error {}
  class ArenaScene {
    options: MockArenaSceneOptions;
    update = vi.fn();
    dispose = vi.fn();
    setReducedMotion = vi.fn();
    setAgentTopology = vi.fn();
    constructor(options: MockArenaSceneOptions) {
      this.options = options;
      instances.push(this);
    }
  }
  return { ArenaScene, ArenaSceneUnavailableError };
});

/** Minimal `MediaQueryListEvent` stand-in: jsdom implements neither `matchMedia` nor this event type. */
class FakeMediaQueryListEvent extends Event {
  constructor(public readonly matches: boolean) {
    super('change');
  }
}

/** Minimal `MediaQueryList` stand-in, real enough for `addEventListener('change', ...)`/`removeEventListener` to work via the platform `EventTarget`. */
class FakeMediaQueryList extends EventTarget {
  constructor(
    public readonly media: string,
    public matches: boolean
  ) {
    super();
  }
}

beforeEach(() => {
  vi.stubGlobal('fetch', createPublicDataFetch());
  vi.stubGlobal('Worker', FakeNeuralWorker as unknown as typeof Worker);
});

afterEach(() => {
  cleanup();
  instances.length = 0;
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

/** Wait for the dynamically-imported `ArenaScene` to have been constructed, then return the mock instance. */
const mountAndAwaitScene = async (): Promise<(typeof instances)[number]> => {
  render(App);
  await waitFor(() => expect(instances).toHaveLength(1));
  return instances[0];
};

const waitForReady = async (): Promise<void> => {
  await waitFor(() => expect(screen.getByLabelText(/experiment status: ready/i)).toBeInTheDocument(), {
    timeout: 5000
  });
};

describe('App renderer lifecycle (ArenaScene mocked)', () => {
  it('constructs exactly one ArenaScene against the mounted canvas', async () => {
    await mountAndAwaitScene();
    expect(instances).toHaveLength(1);
  });

  it('disposes the scene when the component unmounts', async () => {
    const instance = await mountAndAwaitScene();
    expect(instance.dispose).not.toHaveBeenCalled();

    cleanup();

    expect(instance.dispose).toHaveBeenCalledTimes(1);
  });

  it('disposes the scene and shows the fallback message when the renderer reports context loss', async () => {
    const instance = await mountAndAwaitScene();
    expect(instance.options.onContextLost).toBeTypeOf('function');

    instance.options.onContextLost?.({ reason: 'webglcontextlost' });
    // ArenaScene.dispose() is deferred via queueMicrotask so it never runs
    // from inside the renderer's own event dispatch — flush microtasks.
    await Promise.resolve();
    await Promise.resolve();

    expect(instance.dispose).toHaveBeenCalledTimes(1);
    const fallback = await screen.findByRole('img', { name: /arena canvas unavailable/i });
    expect(fallback).toHaveTextContent(/context was lost/i);
  });
});

describe('App activity view asset loading (thermo-architecture I1 fix: no duplicate graph fetch)', () => {
  it('fetches the biological graph artifact exactly once, even once the activity panel opens', async () => {
    const fetchedUrls: string[] = [];
    vi.stubGlobal('fetch', async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input.toString();
      fetchedUrls.push(url);
      return createPublicDataFetch()(input);
    });

    render(App);
    await waitForReady();

    // `loadPositions` (fired from `onManifest`, independent of the panel's
    // own expand/collapse state) has to resolve `status: 'ok'` before the
    // "Expand" toggle becomes enabled — waiting for that is what proves the
    // cross-check against the graph completed without a second fetch.
    // Scoped to `section.activity` specifically: WP1 of
    // `.agents/plans/findings-tour` added `FindingsPanel.svelte`'s own
    // "Expand" toggle to the sidebar, so an unscoped `findByRole` now
    // matches two buttons with the same accessible name.
    const activitySection = document.querySelector('section.activity');
    if (!activitySection) throw new Error('section.activity not found');
    const toggle = await within(activitySection as HTMLElement).findByRole('button', { name: 'Expand' });
    await waitFor(() => expect(toggle).toBeEnabled(), { timeout: 5000 });

    // Opening the panel itself renders from data already loaded (no fetch
    // of its own) — click it anyway to confirm the "no second fetch"
    // guarantee holds across expand too, not just at manifest-load time.
    await fireEvent.click(toggle);

    const graphFetches = fetchedUrls.filter((url) => url.endsWith('malecns-arena-v1.bin.gz'));
    expect(graphFetches).toHaveLength(1);
  });
});

describe('App reduced-motion handling', () => {
  it('subscribes to prefers-reduced-motion and forwards changes to the scene, unsubscribing on unmount', async () => {
    const mql = new FakeMediaQueryList('(prefers-reduced-motion: reduce)', false);
    vi.stubGlobal(
      'matchMedia',
      vi.fn((query: string) => (query === mql.media ? mql : new FakeMediaQueryList(query, false)))
    );

    const instance = await mountAndAwaitScene();
    expect(instance.setReducedMotion).not.toHaveBeenCalled();

    mql.dispatchEvent(new FakeMediaQueryListEvent(true));
    expect(instance.setReducedMotion).toHaveBeenCalledTimes(1);
    expect(instance.setReducedMotion).toHaveBeenLastCalledWith(true);

    cleanup();

    // The listener must be removed on unmount: a change event dispatched
    // afterward must not reach the (now-disposed) scene.
    mql.dispatchEvent(new FakeMediaQueryListEvent(false));
    expect(instance.setReducedMotion).toHaveBeenCalledTimes(1);
  });
});

describe('App frame loop (interpolation only, no stepping on the render thread)', () => {
  let rafCallback: FrameRequestCallback | undefined;
  let rafSpy: ReturnType<typeof vi.spyOn>;
  let cafSpy: ReturnType<typeof vi.spyOn>;

  const pump = (nowMs: number): void => {
    const callback = rafCallback;
    rafCallback = undefined;
    callback?.(nowMs);
  };

  const stubRaf = (): void => {
    rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      rafCallback = cb;
      return 1;
    });
    cafSpy = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
  };

  afterEach(() => {
    rafSpy?.mockRestore();
    cafSpy?.mockRestore();
    rafCallback = undefined;
  });

  it('calls scene.update with a valid arena snapshot on every animation frame, both before and after the experiment becomes ready', async () => {
    stubRaf();
    const instance = await mountAndAwaitScene();
    expect(rafCallback).toBeTypeOf('function');

    // The very first frame renders whatever is available yet — either the
    // idle placeholder world (if the asset/Worker pipeline hasn't resolved
    // yet) or the freshly-constructed runner's own world (tick 0 either
    // way): frame() never steps anything itself, it only interpolates.
    pump(1000);
    expect(instance.update).toHaveBeenCalledTimes(1);
    const [firstSnapshot] = instance.update.mock.calls[0];
    expect(firstSnapshot.tick).toBe(0);
    expect(firstSnapshot.agents).toHaveLength(2);

    await waitForReady();
    pump(1016);
    expect(instance.update).toHaveBeenCalledTimes(2);
    const [secondSnapshot] = instance.update.mock.calls[1];
    expect(secondSnapshot.agents).toHaveLength(2);
    expect(secondSnapshot.foods.length).toBeGreaterThan(0);
  });

  it('does not throw or stop the loop across many frames while idle', async () => {
    stubRaf();
    await mountAndAwaitScene();
    for (let frameIndex = 0; frameIndex < 20; frameIndex += 1) {
      pump(1000 + frameIndex * 16);
    }
    expect(rafCallback).toBeTypeOf('function');
  });
});

describe('App experiment controls wiring', () => {
  it('Start moves the status to running, Pause stops it, and Reset returns to ready', async () => {
    await mountAndAwaitScene();
    await waitForReady();

    const startButton = screen.getByRole('button', { name: /^start$/i });
    await fireEvent.click(startButton);

    await waitFor(() => expect(screen.getByLabelText(/experiment status: running/i)).toBeInTheDocument(), {
      timeout: 5000
    });

    const pauseButton = screen.getByRole('button', { name: /^pause$/i });
    const resetButtonWhileRunning = screen.getByRole('button', { name: /^reset$/i });
    // Regression guard for a real shipped bug: Pause/Reset were briefly
    // wired to the same blanket `controlsLocked` flag Start/Seed use, which
    // is unconditionally `true` for the entire duration of every run — so
    // both buttons were disabled the whole time and could never actually be
    // clicked. `fireEvent.click` below does not itself respect `disabled`
    // (unlike a real click or `userEvent.click`), so only this explicit
    // assertion — checked *before* clicking — actually catches it.
    expect(pauseButton).not.toBeDisabled();
    expect(resetButtonWhileRunning).not.toBeDisabled();

    await fireEvent.click(pauseButton);
    await waitFor(() => expect(screen.getByLabelText(/experiment status: paused/i)).toBeInTheDocument(), {
      timeout: 5000
    });

    const resetButton = screen.getByRole('button', { name: /^reset$/i });
    expect(resetButton).not.toBeDisabled();
    await fireEvent.click(resetButton);
    await waitFor(() => expect(screen.getByLabelText(/experiment status: ready/i)).toBeInTheDocument());
  });

  it('changing the seed input resets the world seed shown in telemetry', async () => {
    await mountAndAwaitScene();
    await waitForReady();

    const seedInput = screen.getByLabelText(/^seed$/i) as HTMLInputElement;
    await fireEvent.change(seedInput, { target: { value: '4242' } });

    await waitFor(() => expect(screen.getAllByText('4242').length).toBeGreaterThan(0));
  });
});

describe('App 3D scene topology honesty (regression: the canvas label must track the real topology)', () => {
  it('syncs ArenaScene#setAgentTopology to the default topology on initial load, and to the switched mode after a successful switch — never a stale/no-op label', async () => {
    const instance = await mountAndAwaitScene();
    await waitForReady();

    // Initial load must sync both slots exactly once each, with the
    // default topology — this is what stands between the canvas and the
    // "always says BIO" honesty bug a prior review caught (see
    // `ArenaScene.ts`'s `setAgentTopology` doc comment).
    expect(instance.setAgentTopology).toHaveBeenCalledWith('left', 'biological');
    expect(instance.setAgentTopology).toHaveBeenCalledWith('right', 'rewired');

    // Initial load legitimately syncs 'left' more than once (both the
    // scene-construction-time sync and the controller's own initial-load
    // callback fire, belt-and-suspenders against their independent
    // construction-order races — see `App.svelte`'s `onTopologyApplied`
    // callback comment) — capture that count so the post-switch assertion
    // below only looks at calls made *after* the switch was requested.
    const leftCallsBeforeSwitch = instance.setAgentTopology.mock.calls.filter(
      (call: unknown[]) => call[0] === 'left'
    ).length;

    const leftSelect = screen.getByLabelText(/left arm topology/i) as HTMLSelectElement;
    await fireEvent.change(leftSelect, { target: { value: 'disconnected' } });
    await waitFor(() => expect(leftSelect).toBeEnabled(), { timeout: 5000 });

    // The most recent call for the switched arm must reflect the new mode —
    // never leave the scene claiming the arm is still 'biological' (or any
    // other stale mode) once the switch has actually succeeded.
    const leftCalls = instance.setAgentTopology.mock.calls.filter(
      (call: unknown[]) => call[0] === 'left'
    );
    expect(leftCalls.at(-1)).toEqual(['left', 'disconnected']);
    // No call made *after* the switch was requested may claim 'biological'
    // again for this now-switched arm.
    expect(leftCalls.slice(leftCallsBeforeSwitch).some((call: unknown[]) => call[1] === 'biological')).toBe(
      false
    );
  });
});

describe('App topology-switch race safety (regression for a prior review finding)', () => {
  it('two rapid topology changes on the same arm never reach the error state, and settle on the last selection', async () => {
    await mountAndAwaitScene();
    await waitForReady();

    const leftSelect = screen.getByLabelText(/left arm topology/i) as HTMLSelectElement;
    // Fire both change events back to back, synchronously, before either
    // async dispose()/init() sequence has a chance to complete — this is
    // the same interleaving rapid keyboard <select> navigation produces.
    await fireEvent.change(leftSelect, { target: { value: 'disconnected' } });
    await fireEvent.change(leftSelect, { target: { value: 'rewired' } });

    // While the switch is in flight, the topology selects must be locked.
    expect(leftSelect).toBeDisabled();

    await waitFor(() => expect(leftSelect).toBeEnabled(), { timeout: 5000 });
    expect(screen.queryByLabelText(/experiment status: error/i)).not.toBeInTheDocument();
    expect(screen.getByLabelText(/experiment status: ready/i)).toBeInTheDocument();
    expect(leftSelect.value).toBe('rewired');
  });

  it('Start/Reset are disabled while a topology switch is pending, and re-enabled once it settles, without ever erroring', async () => {
    await mountAndAwaitScene();
    await waitForReady();

    const leftSelect = screen.getByLabelText(/left arm topology/i) as HTMLSelectElement;
    await fireEvent.change(leftSelect, { target: { value: 'disconnected' } });

    const startButton = screen.getByRole('button', { name: /^start$/i });
    const resetButton = screen.getByRole('button', { name: /^reset$/i });
    expect(startButton).toBeDisabled();
    expect(resetButton).toBeDisabled();

    await waitFor(() => expect(startButton).toBeEnabled(), { timeout: 5000 });
    expect(resetButton).toBeEnabled();
    expect(screen.queryByLabelText(/experiment status: error/i)).not.toBeInTheDocument();
  });
});
