import { describe, expect, it, vi } from 'vitest';
import { ArenaScene, ArenaSceneUnavailableError, topologyLabelFor, tracePanelPath } from '../../src/lib/render/ArenaScene';
import type { GraphMode } from '../../src/lib/connectome/format';

/**
 * jsdom implements no WebGL, matching a real browser with no GPU/software
 * rasterizer available. `ArenaScene` must fail fast and predictably in that
 * case (deliverable #6's "graceful fallback message") rather than throwing
 * an unhandled, engine-internal error. We force `getContext` explicitly
 * instead of relying on jsdom's default null-return behavior, so this test
 * does not silently stop covering the fallback path if a future jsdom
 * version starts stubbing a context object.
 */
describe('ArenaScene WebGL availability', () => {
  it('throws ArenaSceneUnavailableError when no WebGL context can be created', () => {
    const canvas = document.createElement('canvas');
    canvas.getContext = (() => null) as typeof canvas.getContext;

    expect(() => new ArenaScene({ canvas })).toThrow(ArenaSceneUnavailableError);
  });

  it('wraps a thrown context-creation failure with a descriptive message', () => {
    const canvas = document.createElement('canvas');
    canvas.getContext = (() => {
      throw new Error('boom');
    }) as typeof canvas.getContext;

    expect(() => new ArenaScene({ canvas })).toThrow(/boom/);
  });
});

/**
 * Regression coverage for a real bug found during review: label sprites are
 * built with `CanvasRenderingContext2D#roundRect`, which is unsupported in
 * Firefox < 112 and Safari < 16 and throws a `TypeError` there. That
 * exception used to escape from inside `ArenaScene`'s constructor *after*
 * the WebGL renderer already existed, leaking the GL context. `ArenaScene`
 * itself can't be constructed under jsdom (no WebGL) to exercise this
 * end-to-end, so this tests the extracted drawing helper directly.
 */
/**
 * Regression coverage for a real shipped honesty bug: `ArenaScene` used to
 * hardcode the left agent's label to "BIO" and the right agent's to
 * "REWIRED" for the scene's entire lifetime, so switching an arm's topology
 * (e.g. to "disconnected") left the canvas still claiming biological
 * provenance for an arm that had none. `ArenaScene` itself cannot be
 * constructed under jsdom (no WebGL — see the describe block above), so the
 * mode -> label mapping it renders from is tested directly here, the same
 * way `tracePanelPath` is.
 */
describe('topologyLabelFor', () => {
  it('maps every GraphMode to its own distinct, honest label', () => {
    expect(topologyLabelFor('biological').text).toBe('BIO');
    expect(topologyLabelFor('rewired').text).toBe('REWIRED');
    expect(topologyLabelFor('disconnected').text).toBe('DISCONNECTED');
  });

  it('never reads BIO for a non-biological mode', () => {
    const nonBiological: GraphMode[] = ['rewired', 'disconnected'];
    for (const mode of nonBiological) {
      expect(topologyLabelFor(mode).text).not.toBe('BIO');
    }
  });
});

describe('tracePanelPath', () => {
  it('uses roundRect when the browser supports it', () => {
    const roundRect = vi.fn();
    const rect = vi.fn();
    const context = { beginPath: vi.fn(), roundRect, rect } as unknown as CanvasRenderingContext2D;

    tracePanelPath(context, 256, 96);

    expect(roundRect).toHaveBeenCalledTimes(1);
    expect(rect).not.toHaveBeenCalled();
  });

  it('falls back to a plain rect on a browser without roundRect support', () => {
    const rect = vi.fn();
    const context = { beginPath: vi.fn(), rect } as unknown as CanvasRenderingContext2D;
    expect('roundRect' in context).toBe(false);

    expect(() => tracePanelPath(context, 256, 96)).not.toThrow();
    expect(rect).toHaveBeenCalledTimes(1);
  });
});
