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

const instances: Array<{ options: MockActivitySceneOptions; update: ReturnType<typeof vi.fn>; dispose: ReturnType<typeof vi.fn>; render: ReturnType<typeof vi.fn> }> = [];

vi.mock('../../src/lib/render/ActivityScene', () => {
  class ActivitySceneUnavailableError extends Error {}
  class ActivityScene {
    options: MockActivitySceneOptions;
    update = vi.fn();
    dispose = vi.fn();
    render = vi.fn();
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

const makeRunner = (): ExperimentRunner & { setActivityStreaming: ReturnType<typeof vi.fn>; getLatestRates: ReturnType<typeof vi.fn> } =>
  ({
    setActivityStreaming: vi.fn(async () => undefined),
    getLatestRates: vi.fn(() => undefined)
  }) as unknown as ExperimentRunner & { setActivityStreaming: ReturnType<typeof vi.fn>; getLatestRates: ReturnType<typeof vi.fn> };

afterEach(() => {
  cleanup();
  instances.length = 0;
  vi.clearAllMocks();
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
