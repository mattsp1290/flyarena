import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ExperimentController, type ExperimentControllerCallbacks } from '../../src/lib/experiment/controller';
import type { AgentId } from '../../src/lib/arena/types';
import type { GraphMode } from '../../src/lib/connectome/format';
import type { ExperimentStatus } from '../../src/lib/experiment/state';
import { createPublicDataFetch, FakeNeuralWorker } from '../helpers/fake-worker';

/**
 * `ExperimentController` (WP6's thermo review I1/maintainability I1 fix)
 * owns asset loading, Worker/binding construction, the `ExperimentRunner`
 * itself, and topology-switch serialization — logic that previously lived
 * directly in `App.svelte` and could only be exercised by mounting the
 * whole component into jsdom with a mocked render module (see
 * `tests/App.lifecycle.test.ts`). These tests exercise the same scenarios
 * directly against the plain class instead: no component, no jsdom DOM
 * mount, no render-module mock — only `fetch`/`Worker` stand-ins, the same
 * ones `tests/App.lifecycle.test.ts` uses.
 */

const TOTAL_TICKS = 30;
const SEED = 12345;

const createCallbacks = (): ExperimentControllerCallbacks & {
  statuses: ExperimentStatus[];
  errors: string[];
  topologyApplied: Array<[AgentId, GraphMode]>;
  switchCounts: Array<Readonly<Record<AgentId, number>>>;
} => {
  const statuses: ExperimentStatus[] = [];
  const errors: string[] = [];
  const topologyApplied: Array<[AgentId, GraphMode]> = [];
  const switchCounts: Array<Readonly<Record<AgentId, number>>> = [];
  return {
    statuses,
    errors,
    topologyApplied,
    switchCounts,
    onStatusChange: (status) => statuses.push(status),
    onTelemetry: vi.fn(),
    onError: (message) => errors.push(message),
    onManifest: vi.fn(),
    onTopologyApplied: (agentId, mode) => topologyApplied.push([agentId, mode]),
    onTopologySwitchCountChange: (counts) => switchCounts.push({ ...counts })
  };
};

beforeEach(() => {
  vi.stubGlobal('fetch', createPublicDataFetch());
  vi.stubGlobal('Worker', FakeNeuralWorker as unknown as typeof Worker);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

const createWorker = (): Worker => new FakeNeuralWorker() as unknown as Worker;

describe('ExperimentController#initialize', () => {
  it('fetches/verifies both artifacts, constructs a runner, and reports the default topology to both arms', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks
    });

    await controller.initialize();

    expect(controller.getRunner()).toBeDefined();
    expect(controller.getManifest()).toBeDefined();
    expect(callbacks.errors).toHaveLength(0);
    expect(callbacks.statuses).toContain('ready');
    expect(callbacks.topologyApplied).toContainEqual(['left', 'biological']);
    expect(callbacks.topologyApplied).toContainEqual(['right', 'rewired']);
  });

  it('reports an error and never constructs a runner when an artifact fails its sha256 check', async () => {
    vi.stubGlobal('fetch', createPublicDataFetch({ corrupt: 'malecns-arena-v1.bin.gz' }));
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks
    });

    await controller.initialize();

    expect(controller.getRunner()).toBeUndefined();
    expect(callbacks.errors.length).toBeGreaterThan(0);
    expect(callbacks.errors[0]).toMatch(/sha256/i);
    expect(callbacks.statuses).toContain('error');
  });

  it('does nothing (no state mutation, no callback) once disposed before initialize() resolves', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks
    });

    const initializing = controller.initialize();
    controller.dispose();
    await initializing;

    expect(controller.getRunner()).toBeUndefined();
  });
});

describe('ExperimentController#changeTopology', () => {
  const setUp = async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks
    });
    await controller.initialize();
    return { controller, callbacks };
  };

  it('switches an arm to a new topology, reporting the switch pending then settled, and applies the new mode', async () => {
    const { controller, callbacks } = await setUp();
    const runner = controller.getRunner();
    expect(runner).toBeDefined();

    controller.changeTopology('left', 'disconnected');
    // Synchronously incremented before any awaited work begins.
    expect(callbacks.switchCounts.at(-1)?.left).toBe(1);

    await vi.waitFor(() => expect(callbacks.switchCounts.at(-1)?.left).toBe(0));

    expect(callbacks.topologyApplied).toContainEqual(['left', 'disconnected']);
    expect(runner?.getStatus()).toBe('ready');
    expect(runner?.getTelemetry().agents.left.topology).toBe('disconnected');
    expect(callbacks.errors).toHaveLength(0);
  });

  it('serializes two rapid switches on the same arm and settles on the last one, without ever failing', async () => {
    const { controller, callbacks } = await setUp();
    const runner = controller.getRunner();

    controller.changeTopology('left', 'disconnected');
    controller.changeTopology('left', 'rewired');

    await vi.waitFor(() => expect(callbacks.switchCounts.at(-1)?.left).toBe(0));

    expect(runner?.getStatus()).toBe('ready');
    expect(runner?.getTelemetry().agents.left.topology).toBe('rewired');
    expect(callbacks.errors).toHaveLength(0);
    // The stale intermediate mode must still have been reported once (the
    // switch really did run), but the *last* applied mode for 'left' must
    // be the final selection.
    const leftApplied = callbacks.topologyApplied.filter(([agentId]) => agentId === 'left');
    expect(leftApplied.at(-1)).toEqual(['left', 'rewired']);
  });

  it('is a no-op while the run is active (not ready/finished)', async () => {
    const { controller, callbacks } = await setUp();
    const runner = controller.getRunner();
    runner?.start();
    expect(runner?.getStatus()).toBe('running');

    controller.changeTopology('left', 'disconnected');

    expect(callbacks.switchCounts).toHaveLength(0);
    expect(runner?.getTelemetry().agents.left.topology).toBe('biological');
    runner?.pause();
  });
});

describe('ExperimentController#dispose', () => {
  it('is idempotent and terminates both Worker clients', async () => {
    const callbacks = createCallbacks();
    const controller = new ExperimentController({
      seed: SEED,
      totalTicks: TOTAL_TICKS,
      initialTopology: { left: 'biological', right: 'rewired' },
      createWorker,
      callbacks
    });
    await controller.initialize();

    expect(() => {
      controller.dispose();
      controller.dispose();
    }).not.toThrow();
  });
});
