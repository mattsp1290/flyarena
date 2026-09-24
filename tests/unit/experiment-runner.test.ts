import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { createReplaySummary, hashReplaySummary } from '../../src/lib/arena/replay';
import {
  createDisconnectedGraph,
  encodeGraphBinary,
  parseGraphBinary,
  type GraphMode
} from '../../src/lib/connectome/format';
import { createOracleAgentBinding, createWorkerAgentBinding } from '../../src/lib/experiment/bindings';
import {
  computeMedian,
  ExperimentRunner,
  NEURAL_SUBSTEPS_PER_TICK,
  type AgentBinding
} from '../../src/lib/experiment/runner';
import { createWorkerClient } from '../../src/lib/worker/client';
import { createRandomGraph } from '../fixtures/tiny-graph';
import { FakeNeuralWorker } from '../helpers/fake-worker';

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

/**
 * An externally-resolvable/rejectable promise, used below to control
 * exactly when a binding's `step`/`reset` call settles — instead of racing
 * a real `setTimeout` margin against another real timer (see the doc
 * comments on the tests that use this for why: a fixed-ms guess can, in
 * principle, take longer than assumed under a loaded/throttled CI runner,
 * turning a currently-passing test flaky rather than fixing a bug).
 */
const createDeferred = <T = void>(): { promise: Promise<T>; resolve: (value: T) => void } => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
};

/**
 * Poll `predicate` on a short real interval (same pattern as `runToFinished`
 * above) until it's true, instead of guessing a fixed "should be enough"
 * delay and hoping the awaited condition has settled by then. The
 * difference from the anti-pattern this replaces: a guessed sleep duration
 * can, in principle, be too short under a loaded/throttled CI runner,
 * silently turning a currently-passing test flaky; a poll that only
 * resolves once the condition is actually observed true has no such
 * failure mode — it simply takes as long as it takes.
 *
 * `timeoutMs` (bb45 follow-up: this helper previously had no timeout of its
 * own) bounds that "as long as it takes" promise: a genuinely stuck
 * predicate — e.g. a real regression that leaves the runner parked in
 * `running` forever — must fail this specific `waitUntil` call with a clear
 * timeout error inside its own test's timeout window, rather than silently
 * riding on Vitest's own per-test timeout (which would report a generic
 * "test timed out" against whichever assertion happened to be pending, with
 * no indication *which* condition never became true).
 *
 * The default must stay clearly below Vitest's own default per-test timeout
 * (5000ms; `vite.config.ts` does not override `test.testTimeout`) — both
 * dual-review passes independently caught an earlier version of this default
 * (5000ms, matching Vitest's own default exactly) as unable to ever win that
 * race: `waitUntil`'s internal timer and the outer test timer both start
 * close together, so the outer one — reporting a generic "Test timed out",
 * not this function's named error — would always fire first or tie. 2000ms
 * leaves the outer timeout a comfortable multi-second margin to still apply
 * to whatever the test does *after* a `waitUntil` call resolves.
 */
const WAIT_UNTIL_DEFAULT_TIMEOUT_MS = 2000;

const waitUntil = (predicate: () => boolean, timeoutMs = WAIT_UNTIL_DEFAULT_TIMEOUT_MS): Promise<void> =>
  new Promise((resolve, reject) => {
    if (predicate()) {
      resolve();
      return;
    }
    const poll = setInterval(() => {
      if (predicate()) {
        clearInterval(poll);
        clearTimeout(timeout);
        resolve();
      }
    }, 1);
    const timeout = setTimeout(() => {
      clearInterval(poll);
      reject(new Error(`waitUntil: predicate did not become true within ${timeoutMs}ms`));
    }, timeoutMs);
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
      // Reuse the real production summary-construction path rather than a
      // hand-maintained parallel copy of its field mapping: this also means
      // a bug in `createReplaySummary` itself would actually be caught by
      // this determinism test, instead of being invisible to it.
      return hashReplaySummary(createReplaySummary(runner.getWorld()));
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
      return { hash: hashReplaySummary(createReplaySummary(world)), tick: world.tick };
    };

    const [slow, fast] = await Promise.all([runWithJitter(1), runWithJitter(999)]);
    expect(slow.tick).toBe(totalTicks);
    expect(fast.tick).toBe(totalTicks);
    expect(slow.hash).toBe(fast.hash);
  });

  /**
   * WP7 item 2: an irregular render-frame cadence must not change the final
   * simulation state. `getSnapshot(nowMs)` is `ExperimentRunner`'s one
   * concession to the renderer — see `docs/architecture.md`'s closed-loop
   * contract, "Rendering is decoupled from the fixed simulation timestep" —
   * and is documented as read-only/interpolation-only. This is a direct
   * regression test for that contract: calling it at an unpredictable
   * cadence (simulating a browser tab whose `requestAnimationFrame` rate is
   * jittering, throttled, or backgrounded) *genuinely concurrently* with the
   * real tick loop must produce a byte-identical replay hash to a run where
   * it is never called at all.
   *
   * Both dual-review passes independently caught a real bug in an earlier
   * version of this test: with `targetTickIntervalMs: 0` and no per-step
   * latency, `ExperimentRunner#runLoop` never awaits a real timer (`delay()`
   * — see `runner.ts` — is only called when `targetTickIntervalMs > 0`), so
   * all `totalTicks` ticks resolve as one uninterrupted microtask chain.
   * Since JS always drains the entire microtask queue before running any
   * macrotask (`setTimeout`), a frame sampler built on `setTimeout` — as an
   * earlier version of this test was — could not fire even once until
   * *after* the run had already reached `finished`, making the whole test
   * vacuous (independently confirmed by both reviewers with a throwaway
   * probe: exactly one frame, at the final tick). The fix here is the same
   * one both reviews suggested: give both runs a tiny deterministic
   * per-step latency (`simulatedLatencyMs`, already `bindings.ts`'s
   * supported test-only hook — see the randomized-Worker-latency test
   * above), which forces a real macrotask yield every tick so the
   * `setTimeout`-based frame sampler can genuinely interleave. The test then
   * *asserts* that interleaving actually happened (`midRunFrameCount`) —
   * the whole point of the fix — so this test cannot silently regress back
   * to vacuous again without failing on its own guard.
   */
  it('an irregular (deterministic) render-frame sampling cadence — genuinely concurrent getSnapshot() calls — does not change the final tick, score, or replay hash', async () => {
    const seed = 0x9911;
    const totalTicks = 100;
    /**
     * Small, fixed 0-2ms per-step latency: forces one macrotask yield per
     * tick (see the doc comment above) without meaningfully slowing the test
     * down. A *factory* rather than one shared closure: `runWithFrameSampling`
     * calls this once per run, and the two runs execute concurrently
     * (`Promise.all` below) — a single shared mutable counter would let the
     * two runs race for draws from the same sequence, silently making each
     * run's own per-tick latency non-deterministic relative to the other run
     * (still bounded 0-2ms either way, but no longer "identical for both
     * runs" as intended). Two independent generators, each seeded the same,
     * give each run its own deterministic 0-2ms sequence regardless of how
     * the two runs happen to interleave.
     */
    const createStepLatencyMs = (): (() => number) => {
      let stepLatencySeed = 0x5eed;
      return () => {
        stepLatencySeed = (stepLatencySeed * 1103515245 + 12345) >>> 0;
        // `% 3` (not `% 2`): once `stepLatencySeed` grows past ~2^26, this
        // LCG's multiply exceeds `Number.MAX_SAFE_INTEGER`, so JS's float64
        // arithmetic silently loses precision in the *lowest* bits — `% 2`
        // (probing only the single lowest bit) empirically locked to a
        // constant 0 for this seed/multiplier pair, which produced zero
        // real delays and made the whole test vacuous again (caught while
        // implementing this exact fix). `% 3` draws on more of the value's
        // remaining entropy and reliably still varies; the existing
        // randomized-Worker-latency test above uses the same modulus
        // pattern for the same reason.
        return stepLatencySeed % 3;
      };
    };

    const runWithFrameSampling = async (
      sampleFrames: boolean
    ): Promise<{ hash: string; tick: number; midRunFrameCount: number }> => {
      const runner = new ExperimentRunner({
        seed,
        totalTicks,
        agents: buildFixtureAgents(0x51, createStepLatencyMs()),
        targetTickIntervalMs: 0
      });
      let frameCount = 0;
      const ticksAtFrame: number[] = [];
      let frameTimer: ReturnType<typeof setTimeout> | undefined;
      if (sampleFrames) {
        // An irregular, non-periodic cadence — not a fixed rAF-like interval
        // — is the point: a real backgrounded/throttled tab's frame timing
        // is not periodic either. Still fully deterministic (a fixed
        // arithmetic sequence), which is what makes the hash comparison
        // below meaningful across repeated test runs.
        const scheduleNextFrame = (): void => {
          frameTimer = setTimeout(() => {
            frameCount += 1;
            ticksAtFrame.push(runner.getWorld().tick);
            // Read-only per this class's contract; the return value is
            // deliberately discarded — only the *call* (and its potential
            // side effects, if the contract were ever violated) matters here.
            void runner.getSnapshot(performance.now() + ((frameCount * 37) % 23));
            if (runner.getStatus() !== 'finished' && runner.getStatus() !== 'error') scheduleNextFrame();
          }, frameCount % 3);
        };
        scheduleNextFrame();
      }
      try {
        await runToFinished(runner);
      } finally {
        if (frameTimer) clearTimeout(frameTimer);
      }
      const world = runner.getWorld();
      const midRunFrameCount = ticksAtFrame.filter((tick) => tick > 0 && tick < totalTicks).length;
      return { hash: hashReplaySummary(createReplaySummary(world)), tick: world.tick, midRunFrameCount };
    };

    const [withFrames, withoutFrames] = await Promise.all([
      runWithFrameSampling(true),
      runWithFrameSampling(false)
    ]);
    expect(withFrames.tick).toBe(totalTicks);
    expect(withoutFrames.tick).toBe(totalTicks);
    // The guard against this test silently going vacuous again: frames must
    // have genuinely landed *during* the run, not only once at/after the end.
    expect(withFrames.midRunFrameCount).toBeGreaterThan(10);
    expect(withFrames.hash).toBe(withoutFrames.hash);
  });

  /**
   * WP2 stop/go gate: with the activity view closed, the closed-loop
   * simulation must be byte-for-byte unaffected by whether the anatomical
   * activity view happens to be open. `rates` only ever augments
   * `AgentStepResult`/`StepWorkerSuccess` (it plays no role in `decodeAction`
   * or `stepWorld`), so this only needs `WorkerClient`-backed bindings (the
   * oracle binding has no `setActivity` at all, and its dynamics are already
   * covered by the determinism tests above) to prove streaming toggling
   * never perturbs the deterministic replay hash.
   */
  it('replay hash is identical with activity streaming enabled vs disabled', async () => {
    const seed = 0x8899;
    const totalTicks = 80;
    const graph = createRandomGraph(0x88, { neuronCount: 20, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);

    const buildWorkerAgents = async (): Promise<Record<'left' | 'right', AgentBinding>> => {
      const leftClient = createWorkerClient(new FakeNeuralWorker());
      const rightClient = createWorkerClient(new FakeNeuralWorker());
      const [left, right] = await Promise.all([
        createWorkerAgentBinding(leftClient, buffer.slice(0), 'biological'),
        createWorkerAgentBinding(rightClient, buffer.slice(0), 'biological')
      ]);
      return { left, right };
    };

    const runWithStreaming = async (streaming: boolean): Promise<string> => {
      const agents = await buildWorkerAgents();
      const runner = new ExperimentRunner({ seed, totalTicks, agents, targetTickIntervalMs: 0 });
      if (streaming) await runner.setActivityStreaming(true);
      await runToFinished(runner);
      return hashReplaySummary(createReplaySummary(runner.getWorld()));
    };

    const [withStreaming, withoutStreaming] = await Promise.all([
      runWithStreaming(true),
      runWithStreaming(false)
    ]);
    expect(withStreaming).toBe(withoutStreaming);
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

  it('reports each arm’s own step latency independently, not the slower arm’s time for both', async () => {
    const graph = createRandomGraph(0x21, { neuronCount: 16, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);
    const slowLeft = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological', simulatedLatencyMs: () => 25 });
    const fastRight = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological', simulatedLatencyMs: () => 1 });

    const runner = new ExperimentRunner({
      seed: 1,
      totalTicks: 5,
      agents: { left: slowLeft, right: fastRight },
      targetTickIntervalMs: 0
    });
    await runToFinished(runner);

    const telemetry = runner.getTelemetry();
    // Each arm's own median must reflect its own artificial delay, not the
    // other arm's — a bug previously timed both arms from the same start
    // to the same `Promise.all` settlement, so both always reported the
    // slower arm's latency.
    expect(telemetry.agents.left.medianStepLatencyMs).toBeGreaterThan(15);
    expect(telemetry.agents.right.medianStepLatencyMs).toBeLessThan(15);
    expect(telemetry.agents.left.medianStepLatencyMs).toBeGreaterThan(telemetry.agents.right.medianStepLatencyMs);
  });
});

describe('ExperimentRunner setAgentBinding', () => {
  it('throws when the run is active (running/paused), and succeeds from ready/finished', async () => {
    const graph = createRandomGraph(0x33, { neuronCount: 16, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);
    const makeBinding = (): AgentBinding =>
      createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological', simulatedLatencyMs: () => 1 });

    const runner = new ExperimentRunner({
      seed: 1,
      totalTicks: 500,
      agents: { left: makeBinding(), right: makeBinding() },
      targetTickIntervalMs: 0
    });

    // ready: allowed.
    expect(() => runner.setAgentBinding('left', makeBinding())).not.toThrow();
    expect(runner.getStatus()).toBe('ready');

    runner.start();
    await new Promise<void>((resolveWait) => {
      const poll = setInterval(() => {
        if (runner.getStatus() === 'running') {
          clearInterval(poll);
          resolveWait();
        }
      }, 1);
    });
    expect(() => runner.setAgentBinding('left', makeBinding())).toThrow(/ready\/finished/);

    runner.pause();
    expect(() => runner.setAgentBinding('left', makeBinding())).toThrow(/ready\/finished/);

    runner.reset();
    expect(runner.getStatus()).toBe('ready');
    expect(() => runner.setAgentBinding('left', makeBinding())).not.toThrow();
  });

  it('implies a reset: switching a binding mid-way through a finished run resets the world and telemetry', async () => {
    const graph = createRandomGraph(0x44, { neuronCount: 16, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);
    const makeBinding = (): AgentBinding => createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' });

    const runner = new ExperimentRunner({
      seed: 5,
      totalTicks: 10,
      agents: { left: makeBinding(), right: makeBinding() },
      targetTickIntervalMs: 0
    });
    await runToFinished(runner);
    expect(runner.getWorld().tick).toBe(10);

    runner.setAgentBinding('left', makeBinding());
    expect(runner.getStatus()).toBe('ready');
    expect(runner.getWorld().tick).toBe(0);
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
    // Pauses exactly once, at tick 10 — this test resumes the run
    // afterward (below) and must not keep re-pausing every subsequent tick.
    let pausedOnce = false;
    const runner = new ExperimentRunner({
      seed: 3,
      totalTicks: 500,
      agents: buildFixtureAgents(0x3),
      targetTickIntervalMs: 0,
      onTelemetry: (telemetry) => {
        // `pause()` called synchronously from *inside* the tick loop's own
        // `onTelemetry` callback, which fires before `runLoop` decides
        // whether to observe the next tick (see runner.ts's `runLoop`: this
        // callback runs, then `transition({type:'tickCompleted'})`/
        // `setStatus`, then the `while` re-check) — so by construction, no
        // further tick can already be in flight once this fires. This
        // replaces a wall-clock "give any in-flight tick a moment to
        // settle" wait with a genuinely race-free trigger: no real timer,
        // no assumed margin.
        if (!pausedOnce && telemetry.tick >= 10) {
          pausedOnce = true;
          runner.pause();
        }
      }
    });

    runner.start();
    await waitUntil(() => runner.getStatus() === 'paused');
    const pausedAtTick = runner.getWorld().tick;
    expect(pausedAtTick).toBeGreaterThanOrEqual(10);

    // Tick count must not advance further while paused. No real time needs
    // to pass to prove this: the loop already exited synchronously inside
    // the `onTelemetry` callback above (`runLoop` breaks once
    // `status !== 'running'`, right after the tick that triggered this
    // callback), so nothing is left pending that could advance it later
    // regardless of how long this test waits.
    expect(runner.getWorld().tick).toBe(pausedAtTick);

    runner.start();
    await runToFinished(runner);
    expect(runner.getStatus()).toBe('finished');
    expect(runner.getWorld().tick).toBe(500);
  });

  it('reset discards an in-flight tick instead of applying it to the fresh world', async () => {
    const graph = createRandomGraph(0x5, { neuronCount: 16, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);
    const base = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' });
    const fast = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' });

    // A step the test resolves on its own terms, rather than racing a real
    // `setTimeout` against a guessed "long enough" margin for `reset()` to
    // land mid-flight.
    const releaseStep = createDeferred<void>();
    let callCount = 0;
    const slow: AgentBinding = {
      ...base,
      step: async (input) => {
        callCount += 1;
        if (callCount === 1) await releaseStep.promise;
        return base.step(input);
      }
    };

    const runner = new ExperimentRunner({
      seed: 9,
      totalTicks: 100,
      agents: { left: slow, right: fast },
      targetTickIntervalMs: 0
    });
    runner.start();
    // `ExperimentRunner#start()` runs synchronously through `runLoop` ->
    // `runOneTick` -> `AgentBinding#step` up to their first real `await` —
    // see the mid-flight-reset test below for the full trace — so by the
    // time `start()` returns here, `slow.step()`'s first (gated) call has
    // already been invoked. No wall-clock wait is needed before this reset
    // genuinely races an in-flight tick.
    runner.reset(9);
    expect(runner.getWorld().tick).toBe(0);

    releaseStep.resolve();
    await waitUntil(() => runner.getStatus() !== 'running');
    expect(runner.getStatus()).toBe('ready');
    expect(runner.getWorld().tick).toBe(0);
  });

  it('a run started after a mid-flight reset produces the same final hash as a clean, uninterrupted run — the interrupted binding’s neural state is truly zeroed, not left dirty by a late-arriving stale step', async () => {
    // The oracle binding's `step` can still be pending when `reset()`
    // synchronously zeroes neural state; without serializing `step`/`reset`
    // on the binding itself (see bindings.ts#createOracleAgentBinding),
    // that stale step would run `runSubsteps` *after* the reset and leave a
    // nonzero rate behind, even though the runner correctly discards the
    // stale *tick*.
    const graph = createRandomGraph(0x6, { neuronCount: 16, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);
    const seed = 12345;

    const runInterrupted = async (): Promise<string> => {
      const left = createOracleAgentBinding({
        graphBuffer: buffer.slice(0),
        mode: 'biological',
        simulatedLatencyMs: () => 15
      });
      const right = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' });
      const runner = new ExperimentRunner({
        seed,
        totalTicks: 60,
        agents: { left, right },
        targetTickIntervalMs: 0
      });
      runner.start();
      // No wall-clock "interrupt mid-flight" wait: `ExperimentRunner#start()`
      // synchronously drives `runLoop` -> `runOneTick` -> `Promise.all` ->
      // `timedStep` -> `AgentBinding#step` before yielding at its first real
      // `await` (`left`'s `simulatedLatencyMs`-gated `setTimeout`). That
      // whole chain runs synchronously in this call stack, so `left.step()`
      // has already been invoked — and, critically, already enqueued onto
      // the oracle binding's own internal FIFO `serialize` queue — by the
      // time `start()` returns here. Calling `reset()` immediately
      // therefore reliably enqueues `left`'s `reset` *behind* that
      // already-in-flight `step` on the binding's own queue, exactly the
      // FIFO ordering this test exists to verify, with no guessed margin.
      runner.reset(seed);
      // `runToFinished` (below) only resolves once the run genuinely
      // reaches `finished`, so it — not a fixed sleep — is what proves the
      // stale step (and the reset behind it) have both long since settled
      // by the time this assertion runs. `runToFinished` itself calls
      // `runner.start()` again, resuming the run after the reset above.
      await runToFinished(runner);
      return runner.getReplayExport().finalHash;
    };

    const runClean = async (): Promise<string> => {
      const left = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' });
      const right = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' });
      const runner = new ExperimentRunner({ seed, totalTicks: 60, agents: { left, right }, targetTickIntervalMs: 0 });
      await runToFinished(runner);
      return runner.getReplayExport().finalHash;
    };

    const [interrupted, clean] = await Promise.all([runInterrupted(), runClean()]);
    expect(interrupted).toBe(clean);
  });

  it('a step that rejects from a superseded (pre-reset) generation does not fail the new run started after reset', async () => {
    const graph = createRandomGraph(0x7, { neuronCount: 16, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);
    const base = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' });

    // Rejects (simulating a disposed/terminated Worker) on its first call —
    // on the test's own signal, not a real timer — and succeeds normally
    // afterward, modeling a Worker that failed mid topology-switch/reset
    // but is healthy again once reinitialized.
    let callCount = 0;
    const failureGate = createDeferred<void>();
    const flaky: AgentBinding = {
      ...base,
      step: async (input) => {
        callCount += 1;
        if (callCount === 1) {
          await failureGate.promise;
          throw new Error('simulated stale Worker failure');
        }
        return base.step(input);
      }
    };
    const right = createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' });

    const statuses: string[] = [];
    const runner = new ExperimentRunner({
      seed: 3,
      totalTicks: 30,
      agents: { left: flaky, right },
      targetTickIntervalMs: 0,
      onStatusChange: (next) => statuses.push(next)
    });
    runner.start();
    // `flaky.step()`'s first call is already synchronously in flight by
    // this point (see the mid-flight-reset test above for the full trace),
    // so this reset reliably supersedes it before it ever rejects — no
    // wall-clock wait needed.
    runner.reset(3);

    // Let the stale rejection land deterministically, on this test's own
    // terms, rather than guessing a wall-clock margin for it to have
    // settled: releasing the gate now is what actually causes
    // `flaky.step()` to reject, and `waitUntil` below polls the runner's
    // real status — rather than assuming a fixed delay was long enough —
    // for that rejection's effects (`runOneTick`'s generation-mismatch
    // discard) to finish propagating.
    failureGate.resolve();
    await waitUntil(() => runner.getStatus() !== 'running');

    expect(runner.getStatus()).toBe('ready');
    expect(statuses).not.toContain('error');

    // The new run (using the now-healthy `flaky` binding on its 2nd+ calls)
    // must still be able to reach finished.
    runner.start();
    await runToFinished(runner);
    expect(runner.getStatus()).toBe('finished');
  });
});

describe('ExperimentRunner activity streaming', () => {
  const buildWorkerAgents = async (
    buffer: ArrayBuffer
  ): Promise<Record<'left' | 'right', AgentBinding>> => {
    const leftClient = createWorkerClient(new FakeNeuralWorker());
    const rightClient = createWorkerClient(new FakeNeuralWorker());
    const [left, right] = await Promise.all([
      createWorkerAgentBinding(leftClient, buffer.slice(0), 'biological'),
      createWorkerAgentBinding(rightClient, buffer.slice(0), 'biological')
    ]);
    return { left, right };
  };

  /**
   * Wraps `binding.step` with a small real (`setTimeout`-based) per-call
   * delay. With `targetTickIntervalMs: 0` and no delay at all, a
   * `FakeNeuralWorker`-backed run's entire microtask chain (every tick's
   * `postMessage` round trip is `queueMicrotask`-based, not a real timer)
   * drains to completion *before* any macrotask — including a polling
   * `setInterval`, per this file's own `waitUntil`/`runToFinished` — ever
   * gets a chance to run (see the randomized-Worker-latency determinism test
   * above for the same underlying mechanism, there used deliberately; here
   * it would make a "catch the run mid-flight" assertion vacuous instead,
   * since by the time any poll fires the run has already reached
   * `finished`). A tiny fixed delay forces a real macrotask yield every
   * tick, so a test can actually observe an in-progress run.
   */
  const withStepDelay = (binding: AgentBinding, delayMs: number): AgentBinding => ({
    ...binding,
    step: async (input) => {
      await new Promise<void>((resolve) => setTimeout(resolve, delayMs));
      return binding.step(input);
    }
  });

  it('isActivityStreaming/getLatestRates start off, and reflect setActivityStreaming(true) once ticks run', async () => {
    const graph = createRandomGraph(0x9a, { neuronCount: 20, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);
    const built = await buildWorkerAgents(buffer);
    const agents: Record<'left' | 'right', AgentBinding> = {
      left: withStepDelay(built.left, 5),
      right: withStepDelay(built.right, 5)
    };
    const runner = new ExperimentRunner({ seed: 1, totalTicks: 200, agents, targetTickIntervalMs: 0 });

    expect(runner.isActivityStreaming()).toBe(false);
    expect(runner.getLatestRates('left')).toBeUndefined();
    expect(runner.getLatestRates('right')).toBeUndefined();

    await runner.setActivityStreaming(true);
    expect(runner.isActivityStreaming()).toBe(true);
    // Setting the flag does not itself run a tick; no rates exist yet.
    expect(runner.getLatestRates('left')).toBeUndefined();

    runner.start();
    await waitUntil(() => runner.getLatestRates('left') !== undefined && runner.getLatestRates('right') !== undefined);
    const firstLeft = runner.getLatestRates('left');
    expect(firstLeft).toHaveLength(graph.metadata.neuronCount);
    expect(runner.getLatestRates('right')).toHaveLength(graph.metadata.neuronCount);
    // The run must still genuinely be in flight (not already `finished`) at
    // this point — otherwise the next assertion (the reference changing
    // tick to tick) would pass vacuously because no further tick will ever
    // run. `totalTicks: 200` at 5ms/step/arm gives ample headroom over
    // `waitUntil`'s 2s default budget for this to hold reliably.
    expect(runner.getStatus()).toBe('running');

    // Replaced, not accumulated: the array reference (and contents, for a
    // biological arm whose recurrent drive keeps evolving) changes tick to
    // tick rather than being mutated in place or reused stale.
    await waitUntil(() => {
      const next = runner.getLatestRates('left');
      return next !== undefined && next !== firstLeft;
    });

    runner.pause();
    await runner.setActivityStreaming(false);
    expect(runner.isActivityStreaming()).toBe(false);
    // Disabling clears the stored snapshot — a closed view must never render
    // stale pre-disable data if it briefly re-reads state before unmounting.
    expect(runner.getLatestRates('left')).toBeUndefined();
    expect(runner.getLatestRates('right')).toBeUndefined();
  });

  it('reset() keeps the current streaming setting but clears the stale pre-reset rates snapshot', async () => {
    const graph = createRandomGraph(0x9b, { neuronCount: 16, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);
    const agents = await buildWorkerAgents(buffer);
    const runner = new ExperimentRunner({ seed: 2, totalTicks: 500, agents, targetTickIntervalMs: 0 });

    await runner.setActivityStreaming(true);
    runner.start();
    await waitUntil(() => runner.getLatestRates('left') !== undefined);

    runner.reset();
    expect(runner.getStatus()).toBe('ready');
    // The streaming *setting* survives a reset (only the world/neural state
    // resets) — this is what lets a topology switch (which always implies a
    // reset) be recovered by the controller re-issuing set-activity, rather
    // than silently dropping the user's choice.
    expect(runner.isActivityStreaming()).toBe(true);
    // But the stale snapshot from before the reset must not linger.
    expect(runner.getLatestRates('left')).toBeUndefined();

    runner.start();
    await waitUntil(() => runner.getLatestRates('left') !== undefined);
    expect(runner.getLatestRates('left')).toHaveLength(graph.metadata.neuronCount);
  });

  it('setActivityStreaming never rejects even when a binding has no setActivity (the oracle binding)', async () => {
    const graph = createRandomGraph(0x9c, { neuronCount: 12, inputChannelCount: 8, outputPopulationCount: 3 });
    const buffer = encodeGraphBinary(graph);
    const agents: Record<'left' | 'right', AgentBinding> = {
      left: createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' }),
      right: createOracleAgentBinding({ graphBuffer: buffer.slice(0), mode: 'biological' })
    };
    const runner = new ExperimentRunner({ seed: 3, totalTicks: 10, agents, targetTickIntervalMs: 0 });

    await expect(runner.setActivityStreaming(true)).resolves.toBeUndefined();
    expect(runner.isActivityStreaming()).toBe(true);
    // No binding ever produces rates (the oracle binding has no `setActivity`
    // to enable it), so `getLatestRates` stays `undefined` — not an error.
    runner.start();
    await runToFinished(runner);
    expect(runner.getLatestRates('left')).toBeUndefined();
    expect(runner.getLatestRates('right')).toBeUndefined();
  });
});
