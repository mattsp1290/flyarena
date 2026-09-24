import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
// `vi.mock` calls are hoisted above imports by Vitest's transform (same
// pattern as `tests/App.lifecycle.test.ts`'s `ArenaScene` mock) — jsdom has
// no WebGL, so a real `ActivityScene` can never be constructed under test.
import ActivityPanel from '../../src/lib/ui/ActivityPanel.svelte';
import type { PositionsArtifact, PositionsLoadResult } from '../../src/lib/experiment/assets';
import type { ExperimentRunner } from '../../src/lib/experiment/runner';

interface MockActivitySceneOptions {
  onContextLost?: (info: { reason: string }) => void;
}

interface MockActivitySceneInstance {
  options: MockActivitySceneOptions;
  update: ReturnType<typeof vi.fn>;
  clear: ReturnType<typeof vi.fn>;
  dispose: ReturnType<typeof vi.fn>;
  render: ReturnType<typeof vi.fn>;
  setReducedMotion: ReturnType<typeof vi.fn>;
}

const instances: MockActivitySceneInstance[] = [];

vi.mock('../../src/lib/render/ActivityScene', () => {
  class ActivitySceneUnavailableError extends Error {}
  class ActivityScene {
    options: MockActivitySceneOptions;
    update = vi.fn();
    clear = vi.fn();
    dispose = vi.fn();
    render = vi.fn();
    setReducedMotion = vi.fn();
    constructor(options: MockActivitySceneOptions) {
      this.options = options;
      instances.push(this);
    }
  }
  return { ActivityScene, ActivitySceneUnavailableError };
});

const fakePositions = (neuronCount = 3): PositionsArtifact => ({
  version: 1,
  sourceFile: 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
  sourceSha256: 'x'.repeat(64),
  graphSha256: 'y'.repeat(64),
  units: 'dataset voxel units (unverified)',
  bodyIds: Array.from({ length: neuronCount }, (_, i) => String(1000 + i)),
  positionSource: ['soma', 'tosoma', 'none'].slice(0, neuronCount) as PositionsArtifact['positionSource'],
  role: ['sensory', 'bridge', 'descending'].slice(0, neuronCount) as PositionsArtifact['role'],
  xyz: [[1, 2, 3], [4, 5, 6], null].slice(0, neuronCount) as PositionsArtifact['xyz'],
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

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
}

const deferred = <T>(): Deferred<T> => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
};

/** `setActivityStreaming` returns a controllable, per-call `Promise` instead of an immediately-resolved one, so a test can hold a call "in flight" to reproduce the expand/collapse/expand race. */
const makeControllableRunner = (): MockRunner & { streamingCalls: Array<Deferred<void>> } => {
  const streamingCalls: Array<Deferred<void>> = [];
  const setActivityStreaming = vi.fn(() => {
    const call = deferred<void>();
    streamingCalls.push(call);
    return call.promise;
  });
  return {
    setActivityStreaming,
    getLatestRates: vi.fn(() => undefined),
    streamingCalls
  } as unknown as MockRunner & { streamingCalls: Array<Deferred<void>> };
};

/** Minimal `MediaQueryListEvent`/`MediaQueryList` stand-ins, matching `tests/App.lifecycle.test.ts`'s pattern — jsdom implements neither. */
class FakeMediaQueryListEvent extends Event {
  constructor(public readonly matches: boolean) {
    super('change');
  }
}
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
  vi.unstubAllGlobals();
});

describe('ActivityPanel expand/collapse streaming lifecycle', () => {
  it('expand lazily constructs the scene and enables streaming; collapse disables it and disposes the scene', async () => {
    const runner = makeRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false
    });

    const toggle = screen.getByRole('button', { name: /^expand$/i });
    expect(toggle).toBeEnabled();

    await fireEvent.click(toggle);
    await waitFor(() => expect(instances).toHaveLength(1));
    await waitFor(() => expect(runner.setActivityStreaming).toHaveBeenCalledWith(true));

    const collapseButton = await screen.findByRole('button', { name: /^collapse$/i });
    await fireEvent.click(collapseButton);

    await waitFor(() => expect(runner.setActivityStreaming).toHaveBeenLastCalledWith(false));
    expect(instances[0].dispose).toHaveBeenCalledTimes(1);
  });

  it('disables streaming on unmount if the panel was left expanded', async () => {
    const runner = makeRunner();
    const { unmount } = render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(runner.setActivityStreaming).toHaveBeenCalledWith(true));

    unmount();

    expect(runner.setActivityStreaming).toHaveBeenLastCalledWith(false);
  });
});

describe('ActivityPanel expand re-entrancy (regression: fast Expand -> Collapse -> Expand used to build two scenes on one canvas)', () => {
  it('a stale expand() call resolving late never disposes a newer generation’s scene, and never re-enables streaming for a closed panel', async () => {
    const runner = makeControllableRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false
    });

    // Call A: Expand. Suspends at `await runner.setActivityStreaming(true)`.
    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));
    expect(runner.streamingCalls).toHaveLength(1);

    // Collapse while A is still in flight: must dispose scene A immediately
    // (teardown() does not wait for any Worker round trip).
    await fireEvent.click(await screen.findByRole('button', { name: /^collapse$/i }));
    expect(instances[0].dispose).toHaveBeenCalledTimes(1);
    expect(runner.streamingCalls).toHaveLength(2); // + collapse's own setActivityStreaming(false) call

    // Call B: Expand again. Builds a second scene and issues its own
    // setActivityStreaming(true) call, independent of A's still-pending one.
    await fireEvent.click(await screen.findByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(2));
    expect(runner.streamingCalls).toHaveLength(3);

    // A's original call finally resolves, long after it was superseded.
    runner.streamingCalls[0].resolve(undefined);
    await Promise.resolve();
    await Promise.resolve();

    // The regression: A's stale resume must not dispose B's live scene.
    expect(instances[1].dispose).not.toHaveBeenCalled();

    // Let the rest settle so nothing is left hanging for other tests.
    runner.streamingCalls[1].resolve(undefined);
    runner.streamingCalls[2].resolve(undefined);
    await Promise.resolve();
  });
});

describe('ActivityPanel render-loop error guard', () => {
  it('a throw from the scene stops streaming, disposes the scene, and shows a message instead of freezing silently', async () => {
    const runner = makeRunner();
    let rafCallback: FrameRequestCallback | undefined;
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      rafCallback = cb;
      return 1;
    });
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});

    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));
    await waitFor(() => expect(runner.setActivityStreaming).toHaveBeenCalledWith(true));
    await waitFor(() => expect(rafCallback).toBeTypeOf('function'));

    instances[0].render.mockImplementation(() => {
      throw new Error('boom');
    });
    const callback = rafCallback;
    rafCallback = undefined;
    callback?.(1000);

    await waitFor(() => expect(runner.setActivityStreaming).toHaveBeenLastCalledWith(false));
    expect(instances[0].dispose).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/activity view stopped/i)).toBeInTheDocument();
    // No further frame was scheduled after the throw.
    expect(rafCallback).toBeUndefined();

    rafSpy.mockRestore();
  });
});

describe('ActivityPanel reduced-motion forwarding', () => {
  it('forwards a live prefers-reduced-motion change to the scene, in addition to throttling color updates', async () => {
    const runner = makeRunner();
    const mql = new FakeMediaQueryList('(prefers-reduced-motion: reduce)', false);
    vi.stubGlobal(
      'matchMedia',
      vi.fn((query: string) => (query === mql.media ? mql : new FakeMediaQueryList(query, false)))
    );

    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));
    expect(instances[0].setReducedMotion).not.toHaveBeenCalled();

    mql.dispatchEvent(new FakeMediaQueryListEvent(true));

    expect(instances[0].setReducedMotion).toHaveBeenCalledWith(true);
  });
});

describe('ActivityPanel "no data" repaint', () => {
  it('repaints an arm to the neutral "no data" state once its rates go from present to absent (e.g. after a reset)', async () => {
    const runner = makeRunner();
    let rafCallback: FrameRequestCallback | undefined;
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      rafCallback = cb;
      return 1;
    });
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});

    const rates = new Float32Array([0.1, 0.2, 0.3]);
    runner.getLatestRates.mockReturnValue(rates);

    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));
    await waitFor(() => expect(rafCallback).toBeTypeOf('function'));

    const pump = (nowMs: number): void => {
      const callback = rafCallback;
      rafCallback = undefined;
      callback?.(nowMs);
    };

    pump(1000);
    expect(instances[0].update).toHaveBeenCalledWith('left', rates);
    expect(instances[0].clear).not.toHaveBeenCalled();

    // Simulate a reset: latestRates goes back to undefined for both arms.
    runner.getLatestRates.mockReturnValue(undefined);
    pump(1016);

    expect(instances[0].clear).toHaveBeenCalledWith('left');
    expect(instances[0].clear).toHaveBeenCalledWith('right');

    rafSpy.mockRestore();
  });
});

describe('ActivityPanel positions-status gating', () => {
  it('disables the toggle and shows the reason when positions status is "invalid"', () => {
    const runner = makeRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: { status: 'invalid', reason: 'sha256 mismatch (test)' },
      telemetry: undefined,
      topologySwitchPending: false
    });

    const toggle = screen.getByRole('button', { name: /^expand$/i });
    expect(toggle).toBeDisabled();
    expect(screen.getByText(/sha256 mismatch \(test\)/i)).toBeInTheDocument();
  });

  it('disables the toggle and shows the reason when positions status is "missing"', () => {
    const runner = makeRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: { status: 'missing', reason: 'no positions entry (test)' },
      telemetry: undefined,
      topologySwitchPending: false
    });

    expect(screen.getByRole('button', { name: /^expand$/i })).toBeDisabled();
    expect(screen.getByText(/no positions entry \(test\)/i)).toBeInTheDocument();
  });

  it('disables the toggle while positions status has not resolved yet', () => {
    const runner = makeRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: undefined,
      telemetry: undefined,
      topologySwitchPending: false
    });

    expect(screen.getByRole('button', { name: /^expand$/i })).toBeDisabled();
  });

  it('disables the toggle while a topology switch is pending, even with ok positions and an open panel', async () => {
    const runner = makeRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: true
    });

    expect(screen.getByRole('button', { name: /^expand$/i })).toBeDisabled();
  });
});
