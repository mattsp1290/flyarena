import { cleanup, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ARENA_CONFIG } from '../src/lib/arena/config';
// `vi.mock` calls are hoisted above imports by Vitest's transform, so this
// static import below receives the mocked module even though it is
// declared after the vi.mock() call in source order.
import App from '../src/App.svelte';

/**
 * jsdom has no WebGL, so a real `ArenaScene` always fails to construct
 * (see tests/App.test.ts's fallback-message test) — which means no test
 * using the real class can ever exercise App.svelte's cleanup paths
 * (dispose-on-unmount, dispose-on-context-loss). Both of those paths had
 * real bugs during this bean's review (the context-loss handler dropped
 * the scene without disposing it). Mock the module so those paths can
 * actually be driven and asserted on.
 *
 * `App.svelte` loads this module via a dynamic `import()` (so `three`
 * lands in its own bundle chunk, see ArenaScene.ts's WP7 note) rather than
 * a static import; Vitest's `vi.mock` intercepts dynamic imports the same
 * way it intercepts static ones, but every test below still has to wait a
 * tick for that import to resolve before `instances` is populated.
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

/**
 * `App.svelte`'s demo loop calls the real `stepWorld` (deterministic game
 * logic we want exercised, not stubbed) but the frame-loop tests below need
 * to observe *how many times* it ran per animation frame, and occasionally
 * make it throw. Wrap the real implementation in a `vi.fn` rather than
 * replacing it, so ticking behavior stays identical to production.
 */
vi.mock('../src/lib/arena/world', async () => {
  const actual = await vi.importActual<typeof import('../src/lib/arena/world')>('../src/lib/arena/world');
  return { ...actual, stepWorld: vi.fn(actual.stepWorld) };
});

import { stepWorld } from '../src/lib/arena/world';

const stepWorldMock = vi.mocked(stepWorld);

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

afterEach(() => {
  cleanup();
  instances.length = 0;
  vi.clearAllMocks();
});

/** Wait for the dynamically-imported `ArenaScene` to have been constructed, then return the mock instance. */
const mountAndAwaitScene = async (): Promise<(typeof instances)[number]> => {
  render(App);
  await waitFor(() => expect(instances).toHaveLength(1));
  return instances[0];
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
  afterEach(() => {
    vi.unstubAllGlobals();
  });

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

describe('App demo-loop frame() behavior', () => {
  let rafCallback: FrameRequestCallback | undefined;
  let rafSpy: ReturnType<typeof vi.spyOn>;
  let cafSpy: ReturnType<typeof vi.spyOn>;

  const pump = (nowMs: number): void => {
    const callback = rafCallback;
    rafCallback = undefined;
    callback?.(nowMs);
  };

  afterEach(() => {
    rafSpy?.mockRestore();
    cafSpy?.mockRestore();
    rafCallback = undefined;
  });

  const stubRaf = (): void => {
    rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      rafCallback = cb;
      return 1;
    });
    cafSpy = vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
  };

  it('steps once for a normal frame gap, then clamps a huge frame gap to at most MAX_CATCHUP_SECONDS worth of fixed steps', async () => {
    stubRaf();
    await mountAndAwaitScene();
    expect(rafCallback).toBeTypeOf('function');

    // First callback only seeds `lastFrameTimeMs` — App.svelte never steps
    // on the frame that establishes the clock baseline.
    pump(1000);
    expect(stepWorldMock).not.toHaveBeenCalled();
    expect(rafCallback).toBeTypeOf('function');

    // A gap just over one fixed-delta steps the world exactly once — the
    // ordinary, non-clamped path. (+1ms clears the fixedDeltaSeconds
    // threshold with margin: `fixedDeltaSeconds * 1000 / 1000` does not
    // round-trip back to exactly `fixedDeltaSeconds` in IEEE754.)
    const secondFrameMs = 1000 + ARENA_CONFIG.fixedDeltaSeconds * 1000 + 1;
    pump(secondFrameMs);
    expect(stepWorldMock).toHaveBeenCalledTimes(1);

    // A huge (10 real second) gap must not replay ~300 fixed steps: the
    // accumulator clamps to MAX_CATCHUP_SECONDS (5 fixed steps) before the
    // catch-up `while` loop runs, on top of the 1 step already taken above.
    const maxCatchupSteps = 5;
    pump(secondFrameMs + 10_000);
    expect(stepWorldMock).toHaveBeenCalledTimes(1 + maxCatchupSteps);
  });

  it('stops the loop, reports the error, and disposes the scene when stepWorld throws mid-frame', async () => {
    stubRaf();
    const instance = await mountAndAwaitScene();
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    pump(1000); // seed lastFrameTimeMs, no step
    const rafCallCountBeforeThrow = rafSpy.mock.calls.length;

    stepWorldMock.mockImplementationOnce(() => {
      throw new Error('boom');
    });
    // +1ms clears the fixedDeltaSeconds threshold with margin — see the
    // frame-gap test above for why an exact multiple doesn't round-trip.
    pump(1000 + ARENA_CONFIG.fixedDeltaSeconds * 1000 + 1);

    expect(instance.dispose).toHaveBeenCalledTimes(1);
    // The throwing frame must return before reaching its own
    // `requestAnimationFrame(frame)` call — the loop stops, it does not
    // reschedule itself.
    expect(rafSpy.mock.calls.length).toBe(rafCallCountBeforeThrow);
    expect(rafCallback).toBeUndefined();

    const fallback = await screen.findByRole('img', { name: /arena canvas unavailable/i });
    expect(fallback).toHaveTextContent(/arena stopped/i);
    expect(fallback).toHaveTextContent(/boom/i);

    consoleErrorSpy.mockRestore();
  });
});
