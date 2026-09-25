import { flushSync } from 'svelte';
import { afterEach, describe, expect, it, vi } from 'vitest';
// `vi.mock` calls are hoisted above imports by Vitest's transform.
import { createLesionColorMode, type LesionColorModeHost } from '../../src/lib/ui/activityLesionColorMode.svelte';
import { loadLesionAtlas, type LesionAtlasLoadResult } from '../../src/lib/experiment/lesionAtlas';
import type { ArenaManifest } from '../../src/lib/experiment/assets';
import type { ConnectomeGraph, GraphMode } from '../../src/lib/connectome/format';
import type { ExperimentRunner } from '../../src/lib/experiment/runner';
import type { ActivityScene as ActivitySceneInstance } from '../../src/lib/render/ActivityScene';
import { withEffectRoot } from '../helpers/effect-root.svelte';

/**
 * Unit tests for `activityLesionColorMode.svelte.ts` in isolation — no
 * `ActivityPanel.svelte` mounted, no mocked `ActivityScene` class. This is
 * exactly the "independently unit-testable" property the thermo-architecture
 * review's proposed extraction was for: the state machine (load/retry/
 * stale-guard) is driven directly here, against a hand-written `host` test
 * double and a `vi.fn()`-mocked `loadLesionAtlas`. `tests/unit/activity-panel.test.ts`'s
 * own lesion-mode tests still cover the end-to-end wiring through a mounted
 * `ActivityPanel.svelte`; this file covers the state machine on its own,
 * without a jsdom canvas or a mocked `ActivityScene` class at all.
 *
 * Named `*.svelte.test.ts` (not plain `*.test.ts`), deliberately: it needs
 * real `$state`/`$effect` rune usage (the topology-change test below mutates
 * a reactive prop and expects the controller's own internal `$effect` to
 * re-run), and only a file matching vite-plugin-svelte's own
 * `.svelte.`-infix module pattern gets compiled with rune support. It still
 * matches `tests/**\/*.test.ts` (vitest's own `include` glob), so Vitest
 * discovers it as a normal test file.
 */

vi.mock('../../src/lib/experiment/lesionAtlas', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../src/lib/experiment/lesionAtlas')>();
  return { ...actual, loadLesionAtlas: vi.fn() };
});

interface FakeScene {
  setMode: ReturnType<typeof vi.fn>;
  clear: ReturnType<typeof vi.fn>;
  setStaticColors: ReturnType<typeof vi.fn>;
  setNoLesionData: ReturnType<typeof vi.fn>;
}

const makeFakeScene = (): FakeScene => ({
  setMode: vi.fn(),
  clear: vi.fn(),
  setStaticColors: vi.fn(),
  setNoLesionData: vi.fn()
});

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

// Deliberately distinct biological/rewiredSeed0 data (mirrors
// `activity-panel.test.ts`'s own `fakeLesionAtlasOk` fixture and its doc
// comment on why): a bug that painted the wrong graph for an arm would
// otherwise go undetected.
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

type MockRunner = ExperimentRunner & { setActivityStreaming: ReturnType<typeof vi.fn> };
const makeRunner = (): MockRunner => ({ setActivityStreaming: vi.fn(async () => undefined) }) as unknown as MockRunner;

/** A minimal, mutable test double for `LesionColorModeHost`'s dependencies — `topology` is `$state` so the controller's own internal `$effect` reacts to it being reassigned, the same way it reacts to `ActivityPanel.svelte`'s real reactive `topology` prop. */
class TestHost {
  scene: FakeScene = makeFakeScene();
  runner: MockRunner = makeRunner();
  sceneReady = $state(true);
  manifest = $state<ArenaManifest | undefined>(fakeManifest());
  biologicalGraph = $state<ConnectomeGraph | undefined>(fakeBiologicalGraph);
  topology = $state<Record<'left' | 'right', GraphMode>>({ left: 'biological', right: 'rewired' });
  destroyed = false;
  onEnterLiveCalls = 0;
  paintedArms: Array<'left' | 'right'> = [];

  toHost(): LesionColorModeHost {
    return {
      scene: () => this.scene as unknown as ActivitySceneInstance,
      sceneReady: () => this.sceneReady,
      runner: () => this.runner,
      manifest: () => this.manifest,
      biologicalGraph: () => this.biologicalGraph,
      topology: () => this.topology,
      destroyed: () => this.destroyed,
      onEnterLive: () => {
        this.onEnterLiveCalls += 1;
      },
      onArmPainted: (agentId) => {
        this.paintedArms.push(agentId);
      }
    };
  }
}

const activeRoots: Array<() => void> = [];
afterEach(() => {
  for (const cleanup of activeRoots.splice(0)) cleanup();
  vi.clearAllMocks();
});

describe('createLesionColorMode', () => {
  it('starts on Live, with the lesion option enabled when the manifest has a lesionAtlas entry', () => {
    const host = new TestHost();
    const { value: controller, cleanup } = withEffectRoot(() => createLesionColorMode(host.toHost()));
    activeRoots.push(cleanup);

    expect(controller.colorMode).toBe('live');
    expect(controller.lesionOptionDisabledReason).toBeUndefined();
  });

  it('disables the lesion option with an honest reason when the manifest has no lesionAtlas entry', () => {
    const host = new TestHost();
    host.manifest = fakeManifest(false);
    const { value: controller, cleanup } = withEffectRoot(() => createLesionColorMode(host.toHost()));
    activeRoots.push(cleanup);

    expect(controller.lesionOptionDisabledReason).toMatch(/no lesion atlas was shipped/i);
    expect(loadLesionAtlas).not.toHaveBeenCalled();
  });

  it('switchColorMode("lesion") loads the atlas once, disables streaming, and paints both arms from their own mapped graph', async () => {
    vi.mocked(loadLesionAtlas).mockResolvedValue(fakeLesionAtlasOk());
    const host = new TestHost();
    const { value: controller, cleanup } = withEffectRoot(() => createLesionColorMode(host.toHost()));
    activeRoots.push(cleanup);

    await controller.switchColorMode('lesion');
    // The atlas-derived colors are painted by the controller's own re-apply
    // `$effect` (reacting to `colorMode` becoming `'lesion'`), which flushes
    // asynchronously — force it to run before asserting on it.
    flushSync();

    expect(loadLesionAtlas).toHaveBeenCalledTimes(1);
    expect(controller.colorMode).toBe('lesion');
    expect(host.scene.setMode).toHaveBeenCalledWith('lesion');
    expect(host.scene.setStaticColors).toHaveBeenCalledWith('left', [0.1, -0.2, 0.3], [true, false, true], 1);
    expect(host.scene.setStaticColors).toHaveBeenCalledWith('right', [-0.4, 0.5, 0.0], [false, true, true], 1);
    expect(host.runner.setActivityStreaming).toHaveBeenLastCalledWith(false);
    expect(host.paintedArms.slice().sort()).toEqual(['left', 'right']);

    // Memoized: a second switch away and back must not re-fetch.
    await controller.switchColorMode('live');
    await controller.switchColorMode('lesion');
    expect(loadLesionAtlas).toHaveBeenCalledTimes(1);
  });

  it('a transient "unavailable" failure sets the retry hint without memoizing, and retries on the next attempt', async () => {
    vi.mocked(loadLesionAtlas)
      .mockResolvedValueOnce({ status: 'unavailable', reason: 'network hiccup (test)' })
      .mockResolvedValueOnce(fakeLesionAtlasOk());
    const host = new TestHost();
    const { value: controller, cleanup } = withEffectRoot(() => createLesionColorMode(host.toHost()));
    activeRoots.push(cleanup);

    await controller.switchColorMode('lesion');
    expect(controller.colorMode).toBe('live'); // stayed on Live
    expect(controller.lesionAtlasTransientReason).toBe('network hiccup (test)');
    expect(controller.lesionOptionDisabledReason).toBeUndefined(); // not permanently disabled

    await controller.switchColorMode('lesion');
    expect(loadLesionAtlas).toHaveBeenCalledTimes(2);
    expect(controller.colorMode).toBe('lesion');
    expect(controller.lesionAtlasTransientReason).toBeUndefined();
  });

  it('switching back to Live clears both arms, calls onEnterLive, and re-enables streaming', async () => {
    vi.mocked(loadLesionAtlas).mockResolvedValue(fakeLesionAtlasOk());
    const host = new TestHost();
    const { value: controller, cleanup } = withEffectRoot(() => createLesionColorMode(host.toHost()));
    activeRoots.push(cleanup);

    await controller.switchColorMode('lesion');
    await controller.switchColorMode('live');

    expect(controller.colorMode).toBe('live');
    expect(host.scene.clear).toHaveBeenCalledWith('left');
    expect(host.scene.clear).toHaveBeenCalledWith('right');
    expect(host.onEnterLiveCalls).toBe(1);
    expect(host.runner.setActivityStreaming).toHaveBeenLastCalledWith(true);
  });

  it('a slow lazy atlas load never overrides a later explicit "Live" selection (stale-intent race)', async () => {
    let resolveLoad!: (result: LesionAtlasLoadResult) => void;
    vi.mocked(loadLesionAtlas).mockReturnValue(
      new Promise((resolve) => {
        resolveLoad = resolve;
      })
    );
    const host = new TestHost();
    const { value: controller, cleanup } = withEffectRoot(() => createLesionColorMode(host.toHost()));
    activeRoots.push(cleanup);

    const lesionSwitch = controller.switchColorMode('lesion'); // suspends awaiting the atlas load
    await controller.switchColorMode('live'); // the user's later, explicit choice

    resolveLoad(fakeLesionAtlasOk());
    await lesionSwitch;

    expect(controller.colorMode).toBe('live');
    expect(host.scene.setMode).not.toHaveBeenCalledWith('lesion');
  });

  it('a topology change while lesion mode is active re-applies colors for the changed arm via the reactive effect', async () => {
    vi.mocked(loadLesionAtlas).mockResolvedValue(fakeLesionAtlasOk());
    const host = new TestHost();
    const { value: controller, cleanup } = withEffectRoot(() => createLesionColorMode(host.toHost()));
    activeRoots.push(cleanup);

    await controller.switchColorMode('lesion');
    flushSync();
    host.scene.setStaticColors.mockClear();
    host.scene.setNoLesionData.mockClear();

    host.topology = { left: 'biological', right: 'disconnected' };
    // The controller's own re-apply `$effect` runs on the next flush, same
    // as any other Svelte 5 effect — outside a component's own render cycle
    // (which flushes automatically), this must be requested explicitly.
    flushSync();

    expect(host.scene.setNoLesionData).toHaveBeenCalledWith('right');
    expect(host.scene.setStaticColors).not.toHaveBeenCalledWith('right', expect.anything(), expect.anything(), expect.anything());
  });
});
