import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
// `vi.mock` calls are hoisted above imports by Vitest's transform.
import ActivityPanel from '../../src/lib/ui/ActivityPanel.svelte';
import type { PositionsArtifact, PositionsLoadResult } from '../../src/lib/experiment/assets';
import type { ExperimentRunner } from '../../src/lib/experiment/runner';
import { buildActivitySceneMockModule, type MockActivitySceneInstance } from '../helpers/mock-activity-scene';

/**
 * A dedicated file (not folded into `activity-panel.test.ts`) because the
 * mock module's factory below gates on `importGate` — it runs exactly once
 * per test file (dynamic `import()` caches the resolved module afterward),
 * so this file can hold that one import pending for exactly as long as this
 * one test needs, reproducing the actual round-1 defect scenario: a fast
 * Expand -> Collapse -> Expand *while `ActivityScene`'s dynamic import is
 * still in flight* (the real-world case — the chunk is a genuine network
 * fetch on first open, not something that resolves in the same microtask).
 * `tests/unit/activity-panel.test.ts`'s own "re-entrancy" test only covers
 * the narrower window after the import already resolved (its mock resolves
 * synchronously); dual review (round 2, both reviewers independently) found
 * that narrower test passes even against the unfixed `expand()`, so this
 * file exists specifically to close that gap.
 */

const instances: MockActivitySceneInstance[] = [];

let importGateResolve: (() => void) | undefined;
const importGate = new Promise<void>((resolve) => {
  importGateResolve = resolve;
});

// Thermo-maintainability review S4: the mock `ActivityScene` class itself
// now lives in one shared place (`tests/helpers/mock-activity-scene.ts`),
// used here and by `activity-panel.test.ts` — previously hand-duplicated in
// both files. `importGate` (this file's own reason for existing — see the
// doc comment above) is passed through as the shared helper's `gate`.
vi.mock('../../src/lib/render/ActivityScene', () => buildActivitySceneMockModule(instances, importGate));

const releaseImport = (): void => importGateResolve?.();

const fakePositions = (): PositionsArtifact => ({
  version: 1,
  sourceFile: 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
  sourceSha256: 'x'.repeat(64),
  graphSha256: 'y'.repeat(64),
  units: 'dataset voxel units (unverified)',
  bodyIds: ['1000', '1001', '1002'],
  positionSource: ['soma', 'tosoma', 'none'],
  role: ['sensory', 'bridge', 'descending'],
  xyz: [[1, 2, 3], [4, 5, 6], null],
  coverage: { soma: 1, tosoma: 1, none: 1 },
  roleCounts: { sensory: 1, bridge: 1, descending: 1 }
});

const okPositionsStatus: PositionsLoadResult = {
  status: 'ok',
  positions: fakePositions(),
  rateMin: -1,
  rateMax: 1
};

type MockRunner = ExperimentRunner & { setActivityStreaming: ReturnType<typeof vi.fn>; getLatestRates: ReturnType<typeof vi.fn> };

const makeRunner = (): MockRunner =>
  ({
    setActivityStreaming: vi.fn(async () => undefined),
    getLatestRates: vi.fn(() => undefined)
  }) as unknown as MockRunner;

afterEach(() => {
  cleanup();
  instances.length = 0;
  vi.clearAllMocks();
});

describe('ActivityPanel expand re-entrancy across a still-pending dynamic import (round-1 scenario)', () => {
  it('Expand -> Collapse -> Expand while the ActivityScene chunk is still loading builds exactly one scene, on the newer (current) generation only', async () => {
    const runner = makeRunner();
    let rafCallCount = 0;
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation(() => {
      rafCallCount += 1;
      return 1;
    });
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});

    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false
    });

    // Call A: Expand. Suspends at `await import(...)`, which this file's
    // mock factory holds pending — nothing has been constructed yet.
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    expect(instances).toHaveLength(0);

    // Collapse while A's import is still in flight: the real round-1
    // scenario. `teardown()` finds no scene to dispose (A never built one)
    // but does bump the generation, invalidating A.
    await fireEvent.click(await screen.findByRole('button', { name: /^collapse$/i }));

    // Call B: Expand again. Its own `await import(...)` awaits the SAME
    // underlying module load A's is still waiting on (dynamic import of an
    // already-in-flight specifier never re-triggers the load).
    await fireEvent.click(await screen.findByRole('button', { name: /^expand$/i }));

    // Release the import: both A's and B's suspended calls resume together.
    releaseImport();
    await waitFor(() => expect(instances).toHaveLength(1));

    // The regression this test exists to catch: A (now stale) must not
    // also have built a second scene on the same canvas, and only one rAF
    // loop may end up running.
    expect(instances).toHaveLength(1);
    await waitFor(() => expect(rafCallCount).toBe(1));
    // Give any stray microtask/loop a chance to schedule a second frame
    // before asserting the count stays at exactly one.
    await Promise.resolve();
    await Promise.resolve();
    expect(rafCallCount).toBe(1);

    // Streaming ends up enabled exactly once, by B — A never reaches its
    // own `setActivityStreaming` call at all (it was still suspended at
    // `await import(...)` when it went stale). The one `false` call is
    // `collapse()`'s own `teardown()`, posted before B's `true`.
    expect(runner.setActivityStreaming.mock.calls).toEqual([[false], [true]]);

    rafSpy.mockRestore();
  });
});
