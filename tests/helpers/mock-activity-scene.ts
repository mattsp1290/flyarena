import { vi } from 'vitest';

/**
 * Shared `ActivityScene` mock, previously hand-duplicated (with the same
 * fields) across `tests/unit/activity-panel.test.ts` and
 * `tests/unit/activity-panel-race.test.ts` (thermo-maintainability review
 * S4, open since round 1) — jsdom has no WebGL, so a real `ActivityScene` can
 * never be constructed under test; both files' `vi.mock('.../render/ActivityScene',
 * ...)` calls swap in this class instead.
 */

export interface MockActivitySceneOptions {
  onContextLost?: (info: { reason: string }) => void;
}

export interface MockActivitySceneInstance {
  options: MockActivitySceneOptions;
  update: ReturnType<typeof vi.fn>;
  clear: ReturnType<typeof vi.fn>;
  dispose: ReturnType<typeof vi.fn>;
  render: ReturnType<typeof vi.fn>;
  setReducedMotion: ReturnType<typeof vi.fn>;
  setMode: ReturnType<typeof vi.fn>;
  setStaticColors: ReturnType<typeof vi.fn>;
  setNoLesionData: ReturnType<typeof vi.fn>;
}

/**
 * Builds the `{ ActivityScene, ActivitySceneUnavailableError }` module shape
 * a `vi.mock('../../src/lib/render/ActivityScene', ...)` factory needs to
 * return. `instances` collects every constructed mock instance, in
 * construction order — pass the same array the calling test file asserts
 * against (and clears in its own `afterEach`).
 *
 * `gate`, if provided, is awaited before the module resolves —
 * `activity-panel-race.test.ts` uses this to hold `ActivityPanel.svelte`'s
 * `await import('../render/ActivityScene')` pending until the test
 * explicitly releases it, reproducing the real "still suspended at the
 * dynamic import" race window (a plain synchronously-resolving mock cannot
 * reach that window — see that test file's own doc comment).
 *
 * Usage (`vi.mock` factory calls are hoisted above imports by Vitest's
 * transform, so `instances`/`gate` must already be declared above the
 * `vi.mock(...)` call in the calling file, same as before this was shared):
 *
 * ```ts
 * const instances: MockActivitySceneInstance[] = [];
 * vi.mock('../../src/lib/render/ActivityScene', () => buildActivitySceneMockModule(instances));
 * ```
 */
export const buildActivitySceneMockModule = async (
  instances: MockActivitySceneInstance[],
  gate?: Promise<void>
): Promise<{ ActivityScene: unknown; ActivitySceneUnavailableError: unknown }> => {
  if (gate) await gate;

  class ActivitySceneUnavailableError extends Error {}
  class ActivityScene {
    options: MockActivitySceneOptions;
    update = vi.fn();
    clear = vi.fn();
    dispose = vi.fn();
    render = vi.fn();
    setReducedMotion = vi.fn();
    setMode = vi.fn();
    setStaticColors = vi.fn();
    setNoLesionData = vi.fn();
    constructor(options: MockActivitySceneOptions) {
      this.options = options;
      instances.push(this);
    }
  }

  return { ActivityScene, ActivitySceneUnavailableError };
};
