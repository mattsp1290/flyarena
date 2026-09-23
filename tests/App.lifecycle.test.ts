import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
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
}> = [];

vi.mock('../src/lib/render/ArenaScene', () => {
  class ArenaSceneUnavailableError extends Error {}
  class ArenaScene {
    options: MockArenaSceneOptions;
    update = vi.fn();
    dispose = vi.fn();
    setReducedMotion = vi.fn();
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
    await fireEvent.click(pauseButton);
    await waitFor(() => expect(screen.getByLabelText(/experiment status: paused/i)).toBeInTheDocument(), {
      timeout: 5000
    });

    const resetButton = screen.getByRole('button', { name: /^reset$/i });
    await fireEvent.click(resetButton);
    await waitFor(() => expect(screen.getByLabelText(/experiment status: ready/i)).toBeInTheDocument());
  });

  it('changing the seed input resets the world seed shown in telemetry', async () => {
    await mountAndAwaitScene();
    await waitForReady();

    const seedInput = screen.getByLabelText(/^seed$/i) as HTMLInputElement;
    await fireEvent.input(seedInput, { target: { value: '4242' } });

    await waitFor(() => expect(screen.getAllByText('4242').length).toBeGreaterThan(0));
  });
});
