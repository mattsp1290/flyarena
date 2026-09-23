import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { hashReplaySummary } from '../../src/lib/arena/replay';
import {
  createDisconnectedGraph,
  encodeGraphBinary,
  parseGraphBinary,
  type GraphMode
} from '../../src/lib/connectome/format';
import { createOracleAgentBinding } from '../../src/lib/experiment/bindings';
import {
  computeMedian,
  ExperimentRunner,
  NEURAL_SUBSTEPS_PER_TICK,
  type AgentBinding
} from '../../src/lib/experiment/runner';
import { createRandomGraph } from '../fixtures/tiny-graph';

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');

/** Build a fast fixture binding pair for both arms from the same seeded random graph, sharing no mutable state between arms. */
const buildFixtureAgents = (
  seed: number,
  simulatedLatencyMs?: (() => number) | undefined
): Record<'left' | 'right', AgentBinding> => {
  const graph = createRandomGraph(seed, { neuronCount: 24, inputChannelCount: 8, outputPopulationCount: 3 });
  const buffer = encodeGraphBinary(graph);
  return {
    left: createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological', simulatedLatencyMs }),
    right: createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological', simulatedLatencyMs })
  };
};

const runToFinished = (runner: ExperimentRunner): Promise<void> =>
  new Promise((resolveRun, rejectRun) => {
    const poll = setInterval(() => {
      const status = runner.getStatus();
      if (status === 'finished') {
        clearInterval(poll);
        resolveRun();
      } else if (status === 'error') {
        clearInterval(poll);
        rejectRun(new Error('runner entered error state'));
      }
    }, 1);
    runner.start();
  });

describe('NEURAL_SUBSTEPS_PER_TICK', () => {
  it('is a small positive constant within the Worker protocol bound', () => {
    expect(NEURAL_SUBSTEPS_PER_TICK).toBeGreaterThan(0);
    expect(NEURAL_SUBSTEPS_PER_TICK).toBeLessThanOrEqual(64);
  });
});

describe('computeMedian', () => {
  it('handles empty, odd, and even sample sets', () => {
    expect(computeMedian([])).toBe(0);
    expect(computeMedian([5])).toBe(5);
    expect(computeMedian([1, 3, 2])).toBe(2);
    expect(computeMedian([1, 2, 3, 4])).toBe(2.5);
  });
});

describe('ExperimentRunner determinism', () => {
  it('same seed produces an identical final replay hash across two independent runs', async () => {
    const seed = 0xabc123;
    const totalTicks = 120;

    const runOnce = async (): Promise<string> => {
      const runner = new ExperimentRunner({
        seed,
        totalTicks,
        agents: buildFixtureAgents(0x55),
        targetTickIntervalMs: 0
      });
      await runToFinished(runner);
      const world = runner.getWorld();
      return hashReplaySummary({
        schemaVersion: 2,
        configFingerprint: world.configFingerprint,
        seed: world.seed,
        rngState: world.rngState,
        ticks: world.tick,
        timeSeconds: world.timeSeconds,
        agents: world.agents.map((agent) => ({
          id: agent.id,
          position: { ...agent.position },
          velocity: { ...agent.velocity },
          heading: agent.heading,
          activeHazardIds: [...agent.activeHazardIds].sort(),
          score: { ...agent.score }
        })),
        foods: world.foods.map((food) => ({ id: food.id, position: { ...food.position }, radius: food.radius, respawns: food.respawns })),
        hazards: world.hazards.map((hazard) => ({ id: hazard.id, position: { ...hazard.position }, velocity: { ...hazard.velocity }, radius: hazard.radius }))
      });
    };

    const [first, second] = await Promise.all([runOnce(), runOnce()]);
    expect(first).toBe(second);
  });

  it('different randomized async step latency does not change the final tick, score, or replay hash', async () => {
    const seed = 0x77dd;
    const totalTicks = 150;

    const runWithJitter = async (jitterSeed: number): Promise<{ hash: string; tick: number }> => {
      let counter = jitterSeed;
      const nextJitterMs = (): number => {
        counter = (counter * 1103515245 + 12345) >>> 0;
        return (counter % 7) + (counter % 3); // small pseudo-random 0..8ms
      };
      const runner = new ExperimentRunner({
        seed,
        totalTicks,
        agents: buildFixtureAgents(0x66, nextJitterMs),
        targetTickIntervalMs: 0
      });
      await runToFinished(runner);
      const world = runner.getWorld();
      const summary = {
        schemaVersion: 2 as const,
        configFingerprint: world.configFingerprint,
        seed: world.seed,
        rngState: world.rngState,
        ticks: world.tick,
        timeSeconds: world.timeSeconds,
        agents: world.agents.map((agent) => ({
          id: agent.id,
          position: { ...agent.position },
          velocity: { ...agent.velocity },
          heading: agent.heading,
          activeHazardIds: [...agent.activeHazardIds].sort(),
          score: { ...agent.score }
        })),
        foods: world.foods.map((food) => ({ id: food.id, position: { ...food.position }, radius: food.radius, respawns: food.respawns })),
        hazards: world.hazards.map((hazard) => ({ id: hazard.id, position: { ...hazard.position }, velocity: { ...hazard.velocity }, radius: hazard.radius }))
      };
      return { hash: hashReplaySummary(summary), tick: world.tick };
    };

    const [slow, fast] = await Promise.all([runWithJitter(1), runWithJitter(999)]);
    expect(slow.tick).toBe(totalTicks);
    expect(fast.tick).toBe(totalTicks);
    expect(slow.hash).toBe(fast.hash);
  });
});

describe('ExperimentRunner backpressure', () => {
  it('never has more than one in-flight step call per agent at a time (no unbounded catch-up)', async () => {
    let inFlight = 0;
    let maxInFlight = 0;
    const graph = createRandomGraph(0x42, { neuronCount: 16, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);
    const base = createOracleAgentBinding({ graphBuffer: buffer, mode: 'biological' });
    const trackedLeft: AgentBinding = {
      ...base,
      step: async (input) => {
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        try {
          await new Promise((resolve) => setTimeout(resolve, 2));
          return await base.step(input);
        } finally {
          inFlight -= 1;
        }
      }
    };

    const runner = new ExperimentRunner({
      seed: 1,
      totalTicks: 20,
      agents: { left: trackedLeft, right: buildFixtureAgents(0x1).right },
      targetTickIntervalMs: 0
    });
    await runToFinished(runner);
    expect(maxInFlight).toBe(1);
  });

  it('reports behindRealtime when a tick takes longer than the pacing budget, without skipping or reusing actions', async () => {
    const graph = createRandomGraph(0x9, { neuronCount: 16, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);
    const slowLeft = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological', simulatedLatencyMs: () => 20 });
    const right = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' });

    const runner = new ExperimentRunner({
      seed: 1,
      totalTicks: 3,
      agents: { left: slowLeft, right },
      targetTickIntervalMs: 5 // deliberately below the 20ms simulated latency
    });
    await runToFinished(runner);
    expect(runner.getTelemetry().behindRealtime).toBe(true);
    // Every tick must still have actually run to completion (1..totalTicks reached, not skipped).
    expect(runner.getWorld().tick).toBe(3);
  });
});

describe('ExperimentRunner disconnected vs biological (real MaleCNS artifact)', () => {
  const loadRealBiologicalBuffer = (): ArrayBuffer => {
    const gzip = readFileSync(resolve(publicDataDir, 'malecns-arena-v1.bin.gz'));
    const binary = gunzipSync(gzip);
    return binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
  };

  const buildBinding = (buffer: ArrayBuffer, mode: GraphMode): AgentBinding =>
    createOracleAgentBinding({ graphBuffer: buffer, mode });

  it('the disconnected control never produces a non-zero decoded action, while the biological arm does, over the same run', async () => {
    const biologicalBuffer = loadRealBiologicalBuffer();
    // Parse a throwaway copy so this conversion never shares a live typed-array
    // view with the buffer handed to `createOracleAgentBinding` below.
    const disconnectedGraph = createDisconnectedGraph(parseGraphBinary(biologicalBuffer.slice(0)));
    const disconnectedBuffer = encodeGraphBinary(disconnectedGraph);

    const seenNonZero = { biological: false, disconnected: false };

    const observingBinding = (base: AgentBinding, key: 'biological' | 'disconnected'): AgentBinding => ({
      ...base,
      step: async (input) => {
        const result = await base.step(input);
        if (result.actionFeatures.some((value) => Math.abs(value) > 1e-9)) seenNonZero[key] = true;
        return result;
      }
    });

    const runner = new ExperimentRunner({
      seed: 7,
      totalTicks: 90, // 3 simulated seconds — enough for recurrent drive to propagate
      agents: {
        left: observingBinding(buildBinding(biologicalBuffer.slice(0), 'biological'), 'biological'),
        right: observingBinding(buildBinding(disconnectedBuffer, 'disconnected'), 'disconnected')
      },
      targetTickIntervalMs: 0
    });
    await runToFinished(runner);

    expect(seenNonZero.biological).toBe(true);
    expect(seenNonZero.disconnected).toBe(false);

    const telemetry = runner.getTelemetry();
    expect(telemetry.agents.right.edgeCount).toBe(0);
    expect(telemetry.agents.left.edgeCount).toBeGreaterThan(0);
  });
});

describe('ExperimentRunner full-length run', () => {
  it('a 2,700-tick run (90 simulated seconds at 30 Hz) reaches finished', async () => {
    const runner = new ExperimentRunner({
      seed: 42,
      totalTicks: 2700,
      agents: buildFixtureAgents(0x2a),
      targetTickIntervalMs: 0
    });
    await runToFinished(runner);
    expect(runner.getStatus()).toBe('finished');
    expect(runner.getWorld().tick).toBe(2700);
    expect(runner.getWorld().timeSeconds).toBeCloseTo(90, 5);
  });
});

describe('ExperimentRunner pause/resume/reset', () => {
  it('pause stops advancing ticks; resume continues from the same world state', async () => {
    const runner = new ExperimentRunner({
      seed: 3,
      totalTicks: 500,
      // A tiny real (macrotask) per-step delay so ticks actually yield to
      // the event loop between ticks — otherwise an unpaced run over pure
      // microtask-resolving step functions can blow through all 500 ticks
      // before this test's own `setInterval` poll ever gets a turn.
      agents: buildFixtureAgents(0x3, () => 1),
      targetTickIntervalMs: 0
    });

    let pausedAtTick = -1;
    runner.start();
    await new Promise<void>((resolveWait) => {
      const poll = setInterval(() => {
        if (runner.getWorld().tick >= 10) {
          runner.pause();
          clearInterval(poll);
          resolveWait();
        }
      }, 1);
    });
    // Give the in-flight tick (if any) a moment to settle after pause().
    await new Promise((r) => setTimeout(r, 10));
    expect(runner.getStatus()).toBe('paused');
    pausedAtTick = runner.getWorld().tick;
    expect(pausedAtTick).toBeGreaterThanOrEqual(10);

    // Tick count must not advance further while paused.
    await new Promise((r) => setTimeout(r, 20));
    expect(runner.getWorld().tick).toBe(pausedAtTick);

    runner.start();
    await runToFinished(runner);
    expect(runner.getStatus()).toBe('finished');
    expect(runner.getWorld().tick).toBe(500);
  });

  it('reset discards an in-flight tick instead of applying it to the fresh world', async () => {
    const graph = createRandomGraph(0x5, { neuronCount: 16, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);
    const slow = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological', simulatedLatencyMs: () => 30 });
    const fast = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' });

    const runner = new ExperimentRunner({
      seed: 9,
      totalTicks: 100,
      agents: { left: slow, right: fast },
      targetTickIntervalMs: 0
    });
    runner.start();
    // Let a tick get in flight (slow agent takes 30ms), then reset mid-flight.
    await new Promise((r) => setTimeout(r, 5));
    runner.reset(9);
    expect(runner.getWorld().tick).toBe(0);

    // Wait past the in-flight tick's resolution time; it must not have applied.
    await new Promise((r) => setTimeout(r, 60));
    expect(runner.getStatus()).toBe('ready');
    expect(runner.getWorld().tick).toBe(0);
  });
});
