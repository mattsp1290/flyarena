import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
// `vi.mock` calls are hoisted above imports by Vitest's transform (same
// pattern as `tests/App.lifecycle.test.ts`'s `ArenaScene` mock) — jsdom has
// no WebGL, so a real `ActivityScene` can never be constructed under test.
import ActivityPanel from '../../src/lib/ui/ActivityPanel.svelte';
import type { ArenaManifest, PositionsArtifact, PositionsLoadResult } from '../../src/lib/experiment/assets';
import type { ExperimentRunner } from '../../src/lib/experiment/runner';
import type { ConnectomeGraph } from '../../src/lib/connectome/format';
import { loadLesionAtlas, type LesionAtlasLoadResult } from '../../src/lib/experiment/lesionAtlas';
import { buildActivitySceneMockModule, type MockActivitySceneInstance } from '../helpers/mock-activity-scene';

// Only `loadLesionAtlas` (the async fetch/verify path) is mocked — the real
// `lesionAtlasGraphKeyForTopology` pure function is kept, so the panel's own
// topology-mapping logic (disconnected -> 'none', etc.) is exercised for
// real, not re-guessed by a second, independently-authored test double.
vi.mock('../../src/lib/experiment/lesionAtlas', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../src/lib/experiment/lesionAtlas')>();
  return { ...actual, loadLesionAtlas: vi.fn() };
});

const instances: MockActivitySceneInstance[] = [];

// Thermo-maintainability review S4: the mock `ActivityScene` class itself now
// lives in one shared place (`tests/helpers/mock-activity-scene.ts`), used
// here and by `activity-panel-race.test.ts` — previously hand-duplicated in
// both files.
vi.mock('../../src/lib/render/ActivityScene', () => buildActivitySceneMockModule(instances));

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

/** Minimal `ArenaManifest` fixture — only the fields `ActivityPanel`/`loadLesionAtlas`'s call site actually reads; `loadLesionAtlas` itself is mocked, so this never needs to satisfy real sha256/cross-checks. */
const fakeManifest = (withLesionAtlas = true): ArenaManifest =>
  ({
    artifact: 'malecns-arena-v1.bin.gz',
    binaryBytes: 1,
    binarySha256: 'a'.repeat(64),
    edgeCount: 1,
    formatVersion: 1,
    gzipBytes: 1,
    gzipSha256: 'b'.repeat(64),
    inputChannelCount: 8,
    license: 'CC-BY-4.0',
    neuronCount: 3,
    outputPopulationCount: 3,
    rewiredArms: {
      seed0: { artifact: 'x', binaryBytes: 1, binarySha256: 'c'.repeat(64), gzipBytes: 1, gzipSha256: 'd'.repeat(64), swapStats: { edgeCount: 1 } }
    },
    sourceDataset: 'test',
    ...(withLesionAtlas ? { lesionAtlas: { artifact: 'lesion-atlas-v1.json', sha256: 'e'.repeat(64) } } : {})
  }) as ArenaManifest;

const fakeBiologicalGraph = { biologicalIds: BigUint64Array.of(1000n, 1001n, 1002n) } as unknown as ConnectomeGraph;

// Round-2 dual review (Suggestion): `biological` and `rewiredSeed0` must
// carry *different* effect/fdrSignificant data — with identical data, a
// `setStaticColors('right', ...)` assertion would pass even if the right
// (`rewired`) arm were painted from the wrong (`biological`) graph, exactly
// the bug `lesionAtlasGraphKeyForTopology` exists to prevent.
const fakeLesionAtlasOk = (): LesionAtlasLoadResult => ({
  status: 'ok',
  absMax: 1,
  data: {
    version: 1,
    neuronCount: 3,
    bodyIds: ['1000', '1001', '1002'],
    graphs: {
      biological: {
        graphSha256: 'x',
        baseline: 0,
        effect: [0.1, -0.2, 0.3],
        ciLow: [0, 0, 0],
        ciHigh: [0, 0, 0],
        fdrSignificant: [true, false, true]
      },
      rewiredSeed0: {
        graphSha256: 'y',
        baseline: 0,
        effect: [-0.4, 0.5, 0.0],
        ciLow: [0, 0, 0],
        ciHigh: [0, 0, 0],
        fdrSignificant: [false, true, true]
      }
    }
  }
});

type MockRunner = ExperimentRunner & {
  setActivityStreaming: ReturnType<typeof vi.fn>;
  getLatestRates: ReturnType<typeof vi.fn>;
  supportsActivityStreaming: ReturnType<typeof vi.fn>;
};

const makeRunner = (): MockRunner =>
  ({
    setActivityStreaming: vi.fn(async () => undefined),
    getLatestRates: vi.fn(() => undefined),
    // Every browser-constructed binding supports streaming today — default
    // to `true` for both arms so existing tests exercising `frame()` are
    // unaffected; tests exercising the unsupported case override this.
    supportsActivityStreaming: vi.fn(() => true)
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
    supportsActivityStreaming: vi.fn(() => true),
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
    // Thermo-maintainability I2: the dynamic error text must be exposed to
    // assistive tech through a live region (`role="alert"`), not only be
    // visible text — a `role="img"` wrapper with a static `aria-label`
    // would pass the `getByText` assertion above while still hiding the
    // real reason from screen readers.
    expect(screen.getByRole('alert')).toHaveTextContent(/activity view stopped/i);
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

describe('ActivityPanel streaming-support gating (thermo-architecture S1: wire supportsActivityStreaming into the panel)', () => {
  it('shows an explicit "streaming unavailable" message for an arm whose current binding does not support it, instead of a silent "No data yet"', async () => {
    const runner = makeRunner();
    let rafCallback: FrameRequestCallback | undefined;
    const rafSpy = vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      rafCallback = cb;
      return 1;
    });
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});

    // Only the right arm's current binding supports streaming.
    runner.supportsActivityStreaming.mockImplementation((agentId: string) => agentId === 'left');

    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));
    await waitFor(() => expect(rafCallback).toBeTypeOf('function'));

    const callback = rafCallback;
    rafCallback = undefined;
    callback?.(1000);

    await waitFor(() => expect(screen.getByText(/right arm: streaming unavailable for this arm/i)).toBeInTheDocument());
    expect(screen.queryByText(/left arm: streaming unavailable for this arm/i)).not.toBeInTheDocument();

    rafSpy.mockRestore();
  });

  it('shows no "streaming unavailable" message when both arms support streaming (the common case today)', async () => {
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
    await waitFor(() => expect(rafCallback).toBeTypeOf('function'));

    const callback = rafCallback;
    rafCallback = undefined;
    callback?.(1000);

    expect(screen.queryByText(/streaming unavailable for this arm/i)).not.toBeInTheDocument();

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

describe('ActivityPanel lesion-effect color mode (WP3)', () => {
  it('disables the lesion radio, with an honest reason, when the manifest has no lesionAtlas entry', async () => {
    const runner = makeRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(false),
      biologicalGraph: fakeBiologicalGraph
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));

    const lesionRadio = screen.getByRole('radio', { name: /lesion effect \(offline\)/i });
    expect(lesionRadio).toBeDisabled();
    expect(screen.getByText(/no lesion atlas was shipped/i)).toBeInTheDocument();
    expect(loadLesionAtlas).not.toHaveBeenCalled();
  });

  it('does not load the lesion atlas until the mode is actually selected (lazy load)', async () => {
    vi.mocked(loadLesionAtlas).mockResolvedValue(fakeLesionAtlasOk());
    const runner = makeRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));

    // Merely expanding the panel (with the mode still on Live) must not
    // trigger the atlas fetch — only selecting the radio does.
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(loadLesionAtlas).not.toHaveBeenCalled();
  });

  it('a transient "unavailable" (fetch/network) failure shows a non-blocking hint and retries on the next selection, instead of permanently disabling the mode', async () => {
    // Round-2 dual review (Important — both reviewers independently flagged
    // this exact gap): this is the one behavior the whole point of adding
    // `'unavailable'` (as distinct from `'missing'`/`'invalid'`) exists for,
    // and it previously had no test — a regression reverting the "don't
    // memoize an unavailable result" fix in `ensureLesionAtlasLoaded` would
    // have passed the full suite undetected.
    vi.mocked(loadLesionAtlas)
      .mockResolvedValueOnce({ status: 'unavailable', reason: 'network hiccup (test)' })
      .mockResolvedValueOnce(fakeLesionAtlasOk());
    const runner = makeRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph,
      topology: { left: 'biological', right: 'rewired' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));

    const lesionRadio = screen.getByRole('radio', { name: /lesion effect \(offline\)/i });
    await fireEvent.click(lesionRadio);

    // First attempt fails transiently: the radio stays enabled (not the
    // permanently-disabling path `lesionOptionDisabledReason` drives), a
    // non-blocking hint names the reason, and the mode stays on Live.
    await waitFor(() => expect(loadLesionAtlas).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByText(/could not be loaded \(retrying is available\)/i)).toBeInTheDocument());
    expect(screen.getByText(/network hiccup \(test\)/i)).toBeInTheDocument();
    expect(lesionRadio).toBeEnabled();
    expect(lesionRadio).not.toBeChecked();
    expect(instances[0].setMode).not.toHaveBeenCalledWith('lesion');

    // Selecting Lesion again retries the fetch (not a cached failure) and
    // this time succeeds.
    await fireEvent.click(lesionRadio);
    await waitFor(() => expect(loadLesionAtlas).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(instances[0].setMode).toHaveBeenCalledWith('lesion'));
    expect(lesionRadio).toBeChecked();
    // The transient hint from the earlier failed attempt is gone now that
    // the mode is active.
    expect(screen.queryByText(/could not be loaded \(retrying is available\)/i)).not.toBeInTheDocument();
  });

  it('selecting Lesion effect loads the atlas once, disables streaming, and paints static colors for both arms', async () => {
    vi.mocked(loadLesionAtlas).mockResolvedValue(fakeLesionAtlasOk());
    const runner = makeRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph,
      topology: { left: 'biological', right: 'rewired' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));

    const lesionRadio = screen.getByRole('radio', { name: /lesion effect \(offline\)/i });
    await fireEvent.click(lesionRadio);

    await waitFor(() => expect(loadLesionAtlas).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(instances[0].setMode).toHaveBeenCalledWith('lesion'));
    // Each arm is painted from its OWN mapped graph — the fixture's
    // biological/rewiredSeed0 data is deliberately distinct (see
    // `fakeLesionAtlasOk`'s own comment), so this also proves the left arm
    // was never accidentally painted from the rewired graph or vice versa.
    await waitFor(() => expect(instances[0].setStaticColors).toHaveBeenCalledWith('left', [0.1, -0.2, 0.3], [true, false, true], 1));
    await waitFor(() =>
      expect(instances[0].setStaticColors).toHaveBeenCalledWith('right', [-0.4, 0.5, 0.0], [false, true, true], 1)
    );
    expect(runner.setActivityStreaming).toHaveBeenLastCalledWith(false);

    // Selecting Live and back to Lesion must not re-fetch — the load is
    // memoized (WP3's "load the atlas only when the mode is first selected").
    await fireEvent.click(screen.getByRole('radio', { name: /^live rate$/i }));
    await fireEvent.click(lesionRadio);
    expect(loadLesionAtlas).toHaveBeenCalledTimes(1);
  });

  it('the label states the sign convention, and the legend states the shared scale with live per-graph max values (thermo-architecture I2, thermo-suggestion S1)', async () => {
    vi.mocked(loadLesionAtlas).mockResolvedValue(fakeLesionAtlasOk());
    const runner = makeRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph,
      topology: { left: 'biological', right: 'rewired' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));
    await fireEvent.click(screen.getByRole('radio', { name: /lesion effect \(offline\)/i }));
    await waitFor(() => expect(instances[0].setStaticColors).toHaveBeenCalled());

    // Sign convention, stated in words (thermo-suggestion S1) — screen-reader
    // users get this from the label text, not just the legend's spatial
    // blue/vermillion positioning (`aria-hidden` on the color bar itself).
    expect(
      screen.getByText(/negative = this model's score drops when the neuron is silenced, positive = it rises/i)
    ).toBeInTheDocument();

    // Shared-scale honesty caption (thermo-architecture I2), computed live
    // from the loaded atlas data — `fakeLesionAtlasOk`'s biological graph has
    // max |effect| 0.3 (from [0.1, -0.2, 0.3]) and rewiredSeed0 has 0.5 (from
    // [-0.4, 0.5, 0.0]), deliberately different so this test would fail if
    // the wrong graph's data (or the absMax fixture value, 1) were shown
    // instead.
    expect(screen.getByText(/one scale, shared across both graphs/i)).toBeInTheDocument();
    expect(screen.getByText(/biological: 0\.300, rewired seed 0: 0\.500/i)).toBeInTheDocument();
    expect(
      screen.getByText(/a mostly pale arm means its effects are small on this shared scale, not necessarily zero/i)
    ).toBeInTheDocument();
  });

  it("frame() never calls update()/clear() while lesion mode is active, even with fresh rates available", async () => {
    vi.mocked(loadLesionAtlas).mockResolvedValue(fakeLesionAtlasOk());
    const runner = makeRunner();
    const rates = new Float32Array([0.1, 0.2, 0.3]);
    runner.getLatestRates.mockReturnValue(rates);

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
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph,
      topology: { left: 'biological', right: 'rewired' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));
    await waitFor(() => expect(rafCallback).toBeTypeOf('function'));

    await fireEvent.click(screen.getByRole('radio', { name: /lesion effect \(offline\)/i }));
    await waitFor(() => expect(instances[0].setStaticColors).toHaveBeenCalled());

    instances[0].update.mockClear();
    instances[0].clear.mockClear();

    // Pump several frames with fresh rates available every time — in live
    // mode this would call `update()` on every one.
    for (let frameIndex = 0; frameIndex < 5; frameIndex += 1) {
      const callback = rafCallback;
      rafCallback = undefined;
      callback?.(1000 + frameIndex * 16);
      await waitFor(() => expect(rafCallback).toBeTypeOf('function'));
    }

    expect(instances[0].update).not.toHaveBeenCalled();
    expect(instances[0].clear).not.toHaveBeenCalled();
    expect(instances[0].render).toHaveBeenCalled();

    // The debug `data-color-source-*` proof this e2e also checks: still
    // 'lesion' after repeated frames, never flipped back to 'live'.
    const canvas = screen.getByLabelText('Neural activity at soma positions');
    expect(canvas).toHaveAttribute('data-color-source-left', 'lesion');
    expect(canvas).toHaveAttribute('data-color-source-right', 'lesion');

    rafSpy.mockRestore();
  });

  it('a disconnected arm shows "no lesion data" and the scene paints it with setNoLesionData, not a fabricated color', async () => {
    vi.mocked(loadLesionAtlas).mockResolvedValue(fakeLesionAtlasOk());
    const runner = makeRunner();
    const { container } = render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph,
      topology: { left: 'biological', right: 'disconnected' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));

    await fireEvent.click(screen.getByRole('radio', { name: /lesion effect \(offline\)/i }));

    await waitFor(() => expect(instances[0].setNoLesionData).toHaveBeenCalledWith('right'));
    expect(instances[0].setStaticColors).toHaveBeenCalledWith('left', expect.anything(), expect.anything(), expect.anything());
    expect(instances[0].setStaticColors).not.toHaveBeenCalledWith('right', expect.anything(), expect.anything(), expect.anything());
    // Scoped to the dedicated `.streaming-unavailable` paragraph via
    // `querySelector` rather than `getByText` — the same sentence also
    // appears (for a different, accessibility reason) inside the sr-only
    // FDR-significance summary paragraph just below it.
    const noDataParagraph = container.querySelector('p.streaming-unavailable');
    expect(noDataParagraph?.textContent).toMatch(/right arm: no lesion data \(disconnected\)/i);

    const canvas = screen.getByLabelText('Neural activity at soma positions');
    expect(canvas).toHaveAttribute('data-lesion-source-left', 'biological');
    expect(canvas).toHaveAttribute('data-lesion-source-right', 'none');
  });

  it('switching back to Live re-enables streaming and resets lastRatesSeen so the next frame repaints', async () => {
    vi.mocked(loadLesionAtlas).mockResolvedValue(fakeLesionAtlasOk());
    const runner = makeRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph,
      topology: { left: 'biological', right: 'rewired' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));
    await fireEvent.click(screen.getByRole('radio', { name: /lesion effect \(offline\)/i }));
    await waitFor(() => expect(instances[0].setMode).toHaveBeenCalledWith('lesion'));

    await fireEvent.click(screen.getByRole('radio', { name: /^live rate$/i }));

    await waitFor(() => expect(instances[0].setMode).toHaveBeenLastCalledWith('live'));
    expect(runner.setActivityStreaming).toHaveBeenLastCalledWith(true);
  });

  it('switching back to Live repaints both arms to neutral immediately, so lesion colors never linger under the Live label when no fresh tick arrives (round-2 dual review regression)', async () => {
    // Regression coverage: `switchColorMode('live')` used to only reset
    // `lastRatesSeen`/`colorSource` and re-enable streaming, never
    // repainting the scene itself. On an experiment that is ready/paused
    // (no ticks arriving), `frame()`'s own `update()`/`clear()` branches
    // then never fire, and the lesion-effect mode's static diverging colors
    // stayed on screen indefinitely under the "Color: Computed rate" label.
    vi.mocked(loadLesionAtlas).mockResolvedValue(fakeLesionAtlasOk());
    const runner = makeRunner(); // getLatestRates always returns undefined — no run in progress
    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph,
      topology: { left: 'biological', right: 'rewired' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));
    await fireEvent.click(screen.getByRole('radio', { name: /lesion effect \(offline\)/i }));
    await waitFor(() => expect(instances[0].setStaticColors).toHaveBeenCalled());

    await fireEvent.click(screen.getByRole('radio', { name: /^live rate$/i }));

    await waitFor(() => expect(instances[0].clear).toHaveBeenCalledWith('left'));
    expect(instances[0].clear).toHaveBeenCalledWith('right');

    // Round-2, round-2 dual review (Important): the mocked `ActivityScene`
    // is a plain spy that doesn't itself gate `clear()` on `mode` the way
    // the real `ActivityScene.ts` does (`clear()` is a no-op there while
    // `this.mode === 'lesion'`) — so the two assertions above alone would
    // still pass even if `switchColorMode` called `scene.clear()` *before*
    // `scene.setMode('live')`, which against the real scene would silently
    // no-op and reproduce the original bug. Assert the real call order
    // directly against both mocks' own `invocationCallOrder`.
    //
    // `setMode('live')` is called *twice* in this test: once from
    // `expand()`'s own initial sync (`colorMode` starts `'live'`, before the
    // user ever selects Lesion) and once from this test's actual switch
    // back to Live at the end — `lastIndexOf`, not `findIndex`, to get the
    // one this assertion actually cares about (an earlier version of this
    // test used `findIndex` here, found the *first* `'live'` call from
    // `expand()` instead, and passed for the wrong reason regardless of
    // `switchColorMode`'s own call order — caught by mutation-testing this
    // assertion itself, not just the source fix it guards).
    const setModeCallArgs = instances[0].setMode.mock.calls.map((call) => call[0]);
    const setModeLiveCallIndex = setModeCallArgs.lastIndexOf('live');
    expect(setModeLiveCallIndex).toBeGreaterThanOrEqual(0);
    const setModeLiveOrder = instances[0].setMode.mock.invocationCallOrder[setModeLiveCallIndex];
    const clearLeftOrder = instances[0].clear.mock.invocationCallOrder[0];
    expect(clearLeftOrder).toBeGreaterThan(setModeLiveOrder);
  });

  it('a topology change while lesion mode is active re-applies static colors for the changed arm (plan requirement)', async () => {
    vi.mocked(loadLesionAtlas).mockResolvedValue(fakeLesionAtlasOk());
    const runner = makeRunner();
    const { rerender } = render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph,
      topology: { left: 'biological', right: 'rewired' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));
    await fireEvent.click(screen.getByRole('radio', { name: /lesion effect \(offline\)/i }));
    await waitFor(() =>
      expect(instances[0].setStaticColors).toHaveBeenCalledWith('right', [-0.4, 0.5, 0.0], [false, true, true], 1)
    );
    instances[0].setStaticColors.mockClear();
    instances[0].setNoLesionData.mockClear();

    await rerender({
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph,
      topology: { left: 'biological', right: 'disconnected' }
    });

    await waitFor(() => expect(instances[0].setNoLesionData).toHaveBeenCalledWith('right'));
    // Round-2 dual review (comment accuracy): the `$effect` re-applies
    // colors for *both* arms on every rerun (including the left arm, whose
    // topology did not change here) — this assertion is not about whether
    // the left arm was repainted, but that the *right* arm, now
    // disconnected, was correctly routed to `setNoLesionData` and never to
    // `setStaticColors` (which would mean it kept showing stale rewired-arm
    // colors, or worse, picked up the left arm's biological data on a bug
    // that ignored `agentId`).
    expect(instances[0].setStaticColors).not.toHaveBeenCalledWith('right', expect.anything(), expect.anything(), expect.anything());
  });

  it('a slow lazy atlas load never overrides a later "Live" selection (stale-intent race, round-2 dual review regression)', async () => {
    const load = deferred<LesionAtlasLoadResult>();
    vi.mocked(loadLesionAtlas).mockReturnValue(load.promise);
    const runner = makeRunner();
    render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph,
      topology: { left: 'biological', right: 'rewired' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));

    // Click Lesion — the load is now in flight and unresolved.
    await fireEvent.click(screen.getByRole('radio', { name: /lesion effect \(offline\)/i }));
    await waitFor(() => expect(loadLesionAtlas).toHaveBeenCalledTimes(1));

    // Before it resolves, the user selects Live again. A plain second
    // `fireEvent.click` on the Live radio would not actually fire a native
    // `change` event here — `handleColorModeInputChange`'s own DOM
    // "unstick" logic (see its doc comment) already force-reset the Live
    // radio's `checked` back to `true` as soon as the Lesion click was
    // handled, and browsers never fire `change` for a click on an
    // already-checked radio. `fireEvent.change` invokes the same `onchange`
    // handler directly, the same way a re-render race or an assistive-tech
    // "activate" action could, without depending on that native-radio
    // quirk.
    await fireEvent.change(screen.getByRole('radio', { name: /^live rate$/i }));

    // The stale Lesion switch's load now finally resolves successfully.
    load.resolve(fakeLesionAtlasOk());
    // Flush a real macrotask boundary (not just a microtask `Promise.resolve()`
    // hop), so `switchColorMode`'s `await ensureLesionAtlasLoaded()` continuation
    // — and any reactive updates it triggers — has definitely had a chance to
    // run before the assertions below, whichever way the guard resolves.
    await new Promise((resolve) => setTimeout(resolve, 0));

    // The user's later, explicit "Live" choice must win — the stale switch
    // must never flip the mode to lesion behind their back.
    expect(instances[0].setMode).not.toHaveBeenCalledWith('lesion');
    expect(screen.getByRole('radio', { name: /^live rate$/i })).toBeChecked();
  });

  it('the hint paragraph is an always-present aria-live region, not one only mounted once populated (thermo-maintainability I3)', async () => {
    vi.mocked(loadLesionAtlas).mockResolvedValueOnce({ status: 'unavailable', reason: 'network hiccup (test)' });
    const runner = makeRunner();
    const { container } = render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph,
      topology: { left: 'biological', right: 'rewired' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));

    // The live-region node exists from first paint, before any hint text has
    // ever been set — a screen reader must already have this node in the
    // accessibility tree to reliably announce a later *mutation* to it. The
    // earlier version only mounted this element via its own nested `{#if}`,
    // once text existed, which several screen readers (notably VoiceOver/
    // Safari) do not reliably announce on first insertion.
    const hint = container.querySelector('p.hint[aria-live="polite"]');
    expect(hint).not.toBeNull();
    expect(hint?.textContent).toBe('');

    await fireEvent.click(screen.getByRole('radio', { name: /lesion effect \(offline\)/i }));
    await waitFor(() => expect(screen.getByText(/could not be loaded \(retrying is available\)/i)).toBeInTheDocument());

    // Same node (reference equality), now carrying the transient-failure
    // text — a mutation to an already-present live region, not a freshly-
    // inserted one.
    const hintAfter = container.querySelector('p.hint[aria-live="polite"]');
    expect(hintAfter).toBe(hint);
    expect(hintAfter?.textContent).toMatch(/network hiccup \(test\)/i);
  });

  it('the sr-only FDR-significance summary is an always-present aria-live region (thermo-maintainability I3)', async () => {
    vi.mocked(loadLesionAtlas).mockResolvedValue(fakeLesionAtlasOk());
    const runner = makeRunner();
    const { container } = render(ActivityPanel, {
      runner,
      positionsStatus: okPositionsStatus,
      telemetry: undefined,
      topologySwitchPending: false,
      manifest: fakeManifest(),
      biologicalGraph: fakeBiologicalGraph,
      topology: { left: 'biological', right: 'rewired' }
    });

    await fireEvent.click(screen.getByRole('button', { name: /^expand$/i }));
    await waitFor(() => expect(instances).toHaveLength(1));

    const summary = container.querySelector('p.sr-only[aria-live="polite"]');
    expect(summary).not.toBeNull();
    expect(summary?.textContent).toBe('');

    await fireEvent.click(screen.getByRole('radio', { name: /lesion effect \(offline\)/i }));
    await waitFor(() => expect(instances[0].setStaticColors).toHaveBeenCalled());

    const summaryAfter = container.querySelector('p.sr-only[aria-live="polite"]');
    expect(summaryAfter).toBe(summary);
    expect(summaryAfter?.textContent).toMatch(/lesion effect mode/i);
  });
});
