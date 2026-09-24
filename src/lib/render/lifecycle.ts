import type * as THREE from 'three';
import { disposeObject3D, disposeRenderer } from './dispose';

/**
 * Shared WebGL renderer-lifecycle helpers used by both `ArenaScene.ts` and
 * `ActivityScene.ts` (thermo-maintainability I1 fix). Resize, ResizeObserver
 * wiring, context-loss handling, and the shared tail of disposal/rollback
 * are generic Three.js renderer-lifecycle boilerplate with zero domain
 * content (nothing here knows about agents, food, hazards, neurons, or
 * roles) — duplicating it per scene class is the same category of drift
 * risk `dispose.ts` already exists to prevent one layer down, at the
 * geometry/material/texture level. Both scene classes still own their own
 * domain-specific construction and per-frame `update()`; only this generic
 * plumbing is shared.
 */

interface ResizableRenderer {
  setSize: (width: number, height: number, updateStyle?: boolean) => void;
}

interface ResizableCamera {
  aspect: number;
  updateProjectionMatrix: () => void;
}

/**
 * Clamp to at least 1px (a zero-size renderer/camera is invalid) and apply
 * to both the renderer's drawing buffer and the camera's aspect ratio — the
 * exact resize sequence both scene classes' `resize()` methods used to
 * duplicate.
 */
export const resizeRendererAndCamera = (
  renderer: ResizableRenderer,
  camera: ResizableCamera,
  width: number,
  height: number
): void => {
  const safeWidth = Math.max(1, Math.floor(width));
  const safeHeight = Math.max(1, Math.floor(height));
  renderer.setSize(safeWidth, safeHeight, false);
  camera.aspect = safeWidth / safeHeight;
  camera.updateProjectionMatrix();
};

/**
 * Observe `container`'s content-box size and call `onResize(width, height)`
 * whenever it changes to something usable (both dimensions > 0). Returns
 * `undefined` in an environment with no `ResizeObserver` global — matches
 * both scene classes' prior behavior of silently skipping observation there
 * (resize then only ever happens via an explicit `resize()` call from the
 * host).
 */
export const createCanvasResizeObserver = (
  container: HTMLElement,
  onResize: (width: number, height: number) => void
): ResizeObserver | undefined => {
  if (typeof ResizeObserver === 'undefined') return undefined;
  const observer = new ResizeObserver((entries) => {
    const entry = entries[0];
    if (!entry) return;
    const box = entry.contentBoxSize?.[0];
    const width = box ? box.inlineSize : entry.contentRect.width;
    const height = box ? box.blockSize : entry.contentRect.height;
    if (width > 0 && height > 0) onResize(width, height);
  });
  observer.observe(container);
  return observer;
};

/**
 * Build the shared `webglcontextlost` DOM listener: calls
 * `event.preventDefault()` (required for the browser to later fire
 * `webglcontextrestored` at all) and delegates everything else to `onLost`.
 * Each scene class supplies its own `onLost` closure so it can set its own
 * `contextLost` flag and notify its own host `onContextLost` callback —
 * this only factors out the identical DOM-event wiring around that.
 */
export const createContextLossHandler = (onLost: () => void): ((event: Event) => void) => {
  return (event: Event): void => {
    event.preventDefault();
    onLost();
  };
};

export interface WebglSceneTeardownOptions {
  canvas: HTMLCanvasElement;
  handleContextLost: (event: Event) => void;
  resizeObserver: ResizeObserver | undefined;
  scene: THREE.Scene;
  renderer: { dispose: () => void; forceContextLoss?: () => void };
  controls?: { dispose: () => void };
  /**
   * Pass `true` when the context is already lost (e.g. tearing down from a
   * `webglcontextlost` handler) — `forceContextLoss()` is a harmless no-op
   * there but some WebGL implementations log a spurious warning for it. See
   * `disposeRenderer`'s own `DisposeRendererOptions`.
   */
  skipForceContextLoss?: boolean;
  /**
   * Also empty the scene graph (`scene.clear()`) after disposal. `dispose()`
   * wants this (the instance is done, nothing should keep referencing the
   * scene's children); the constructor's mid-build rollback path does not
   * (the scene is discarded wholesale along with the instance that never
   * finished constructing, so there is no later caller left to observe a
   * non-empty vs. empty scene).
   */
  clearScene?: boolean;
}

/**
 * The teardown sequence shared by both scene classes' `dispose()` and by
 * their constructors' rollback `catch` block on any mid-construction throw:
 * stop observing resize, stop listening for context loss, free every
 * GPU-owned geometry/material/texture under the scene graph, then free the
 * renderer/controls themselves.
 */
export const teardownWebglScene = (options: WebglSceneTeardownOptions): void => {
  options.resizeObserver?.disconnect();
  options.canvas.removeEventListener('webglcontextlost', options.handleContextLost, false);
  disposeObject3D(options.scene);
  disposeRenderer(options.renderer, options.controls, { skipForceContextLoss: options.skipForceContextLoss });
  if (options.clearScene) options.scene.clear();
};
