import { cleanup, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
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
 */
interface MockArenaSceneOptions {
  onContextLost?: (info: { reason: string }) => void;
  onFrame?: (telemetry: { fps: number; frameMs: number }) => void;
}

const instances: Array<{ options: MockArenaSceneOptions; update: ReturnType<typeof vi.fn>; dispose: ReturnType<typeof vi.fn> }> = [];

vi.mock('../src/lib/render/ArenaScene', () => {
  class ArenaSceneUnavailableError extends Error {}
  class ArenaScene {
    options: MockArenaSceneOptions;
    update = vi.fn();
    dispose = vi.fn();
    constructor(options: MockArenaSceneOptions) {
      this.options = options;
      instances.push(this);
    }
  }
  return { ArenaScene, ArenaSceneUnavailableError };
});

afterEach(() => {
  cleanup();
  instances.length = 0;
  vi.clearAllMocks();
});

describe('App renderer lifecycle (ArenaScene mocked)', () => {
  it('constructs exactly one ArenaScene against the mounted canvas', () => {
    render(App);
    expect(instances).toHaveLength(1);
  });

  it('disposes the scene when the component unmounts', () => {
    const { unmount } = render(App);
    const instance = instances[0];
    expect(instance.dispose).not.toHaveBeenCalled();

    unmount();

    expect(instance.dispose).toHaveBeenCalledTimes(1);
  });

  it('disposes the scene and shows the fallback message when the renderer reports context loss', async () => {
    render(App);
    const instance = instances[0];
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
