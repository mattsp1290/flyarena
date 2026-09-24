// @vitest-environment node
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { gzipSync } from 'node:zlib';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { encodeGraphBinary, SUPPORTED_FORMAT_VERSION, type ConnectomeGraph, type GraphMetadata } from '../../src/lib/connectome/format';
import { runTask, type RegimeWorkerTask } from '../../scripts/null/regime-worker';
import { parseRegimeCheckArgs } from '../../scripts/null/regime-check';
import { sha256Hex } from '../../scripts/training/fsio';

/**
 * `.agents/plans/null-explanation/02-transfer-and-features.md`'s WP2 change
 * surface: "tests/unit/regime-check.test.ts (new): The saturation counter
 * on a fixture that clamps deterministically." Exercises
 * `scripts/null/regime-worker.ts`'s `runTask` directly (in-process, no
 * `fork`/IPC — the same function the forked worker calls per task) against
 * two hand-built single-neuron graphs on disk: one that saturates every
 * substep by construction (`clampFraction` must be exactly `1`), and one
 * far inside the linear regime (`clampFraction` must be exactly `0` and
 * `steadyStateDistance` must be small once the episode has run long enough
 * to approach its linear fixed point).
 */

const CHANNEL_COUNT = 8;
const FOOD_DISTANCE_CHANNEL = 1; // src/lib/arena/sensors.ts's OBSERVATION_CHANNELS[1]

const oneNeuronGraph = (overrides: Partial<GraphMetadata>, selfLoopMagnitude: number, inputWeight: number): ConnectomeGraph => {
  const metadata: GraphMetadata = {
    formatVersion: SUPPORTED_FORMAT_VERSION,
    neuronCount: 1,
    edgeCount: 1,
    inputChannelCount: CHANNEL_COUNT,
    outputPopulationCount: 3,
    timestepSeconds: 1 / 30,
    leakRate: 0.35,
    rateMin: -2,
    rateMax: 2,
    inputClampMin: -1,
    inputClampMax: 1,
    globalGain: 1,
    ...overrides
  };
  return {
    metadata,
    biologicalIds: BigUint64Array.from([1n]),
    presynapticOffsets: Uint32Array.from([0, 1]),
    postsynapticIndices: Uint32Array.from([0]), // self-loop
    contactMagnitudes: Float32Array.from([selfLoopMagnitude]),
    presynapticSigns: Int8Array.from([1]),
    inputChannelIndex: Int32Array.from([FOOD_DISTANCE_CHANNEL]),
    inputWeight: Float32Array.from([inputWeight]),
    outputPopulationIndex: Int32Array.from([0]),
    outputWeight: Float32Array.from([1])
  };
};

/**
 * `analytic 1x1 steady-state map` for `oneNeuronGraph`: `T = O (lambda I - g
 * A)^-1 B` reduces to a single scalar since `neuronCount === 1` and
 * `outputWeight === 1`; `A = [[selfLoopMagnitude]]` (sign `+1`),
 * `B = [0, inputWeight, 0, ..., 0]` (channel `FOOD_DISTANCE_CHANNEL`).
 */
const analyticSteadyStateRow = (
  leakRate: number,
  globalGain: number,
  selfLoopMagnitude: number,
  inputWeight: number
): Float64Array => {
  const denominator = leakRate - globalGain * selfLoopMagnitude;
  const row = new Float64Array(CHANNEL_COUNT);
  row[FOOD_DISTANCE_CHANNEL] = inputWeight / denominator;
  return row;
};

describe('regime-worker runTask: clamp fraction and steady-state distance', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'regime-worker-fixture-'));
  });

  afterEach(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  const writeGraph = (graph: ConnectomeGraph, name: string): { path: string; sha256: string } => {
    const binary = Buffer.from(encodeGraphBinary(graph));
    const sha256 = sha256Hex(binary);
    const path = join(root, `${name}.bin.gz`);
    writeFileSync(path, gzipSync(binary));
    return { path, sha256 };
  };

  const writeSteadyState = (row: Float64Array, name: string): string => {
    const path = join(root, `${name}.steadystate.f64`);
    writeFileSync(path, Buffer.from(row.buffer, row.byteOffset, row.byteLength));
    return path;
  };

  it('reports clampFraction === 1 for a graph that saturates every substep by construction', () => {
    // Unstable by construction: globalGain * selfLoopMagnitude (100) far
    // exceeds leakRate (0.35), so the spectral abscissa exceeds leakRate and
    // the linear system has no stable fixed point -- the recurrent drive
    // only ever grows, so `stepModel`'s clamp is active on every substep
    // from the very first one onward (rate starts at 0, but foodDistance's
    // `nearestEgocentric` fallback of `1` when nothing is in range, or any
    // positive distance otherwise, means the input drive is never exactly
    // zero either -- see this suite's module doc comment).
    const graph = oneNeuronGraph({}, /* selfLoopMagnitude */ 100, /* inputWeight */ 1e6);
    const { path, sha256 } = writeGraph(graph, 'saturating');
    const steadyStateRow = analyticSteadyStateRow(0.35, 1, 100, 1e6);
    const steadyStatePath = writeSteadyState(steadyStateRow, 'saturating');

    const task: RegimeWorkerTask = {
      graphId: 'saturating',
      mode: 'biological',
      path,
      expectedSha256: sha256,
      steadyStatePath,
      heldOutSeeds: [30001, 30002],
      ticks: 30
    };

    const results = runTask(task);
    expect(results).toHaveLength(2);
    for (const result of results) {
      expect(result.seed).toBeGreaterThanOrEqual(30001);
      expect(result.clampFraction).toBe(1);
      expect(Number.isFinite(result.steadyStateDistance)).toBe(true);
    }
  });

  it('reports clampFraction === 0 and a small steadyStateDistance for a well-conditioned, stable graph', () => {
    // Far inside the linear regime: globalGain * selfLoopMagnitude (0.05) is
    // small relative to leakRate (0.35), and inputWeight (0.4) keeps the
    // steady-state rate well inside [-2, 2] for any foodDistance in [0, 1]
    // -- so `stepModel`'s clamp is never active. The time constant is
    // `1 / (0.35 - 0.05)` = 3.33s; 1800 ticks (60s, the production episode
    // length) is ~18 time constants, so the discretized dynamics converge to
    // the linear fixed point to well beyond float32 precision.
    const leakRate = 0.35;
    const globalGain = 1;
    const selfLoopMagnitude = 0.05;
    const inputWeight = 0.4;
    const graph = oneNeuronGraph({ globalGain }, selfLoopMagnitude, inputWeight);
    const { path, sha256 } = writeGraph(graph, 'stable');
    const steadyStateRow = analyticSteadyStateRow(leakRate, globalGain, selfLoopMagnitude, inputWeight);
    const steadyStatePath = writeSteadyState(steadyStateRow, 'stable');

    const task: RegimeWorkerTask = {
      graphId: 'stable',
      mode: 'biological',
      path,
      expectedSha256: sha256,
      steadyStatePath,
      heldOutSeeds: [30001],
      ticks: 1800
    };

    const [result] = runTask(task);
    expect(result.clampFraction).toBe(0);
    // Not exactly 0: the metric averages every tick's distance including the
    // early, not-yet-converged ticks, and the moving foodDistance input
    // keeps re-perturbing the fixed point every tick rather than holding it
    // perfectly constant -- 0.2 is a loose sanity bound, not the tight
    // cross-language numeric gate (that lives in `tests_python/test_transfer.py`,
    // which checks this same algebra against a float64 fixed-point
    // iteration to 1e-9 and a TS finite-difference run to 1e-3 relative).
    expect(result.steadyStateDistance).toBeLessThan(0.2);
  });

  it('throws when the steady-state sidecar length does not match neuronCount x inputChannelCount', () => {
    const graph = oneNeuronGraph({}, 0.05, 0.4);
    const { path, sha256 } = writeGraph(graph, 'bad-sidecar');
    const wrongRow = new Float64Array(CHANNEL_COUNT - 1); // too short
    const steadyStatePath = writeSteadyState(wrongRow, 'bad-sidecar');

    const task: RegimeWorkerTask = {
      graphId: 'bad-sidecar',
      mode: 'biological',
      path,
      expectedSha256: sha256,
      steadyStatePath,
      heldOutSeeds: [30001],
      ticks: 10
    };

    expect(() => runTask(task)).toThrow(/steadystate\.f64.*has \d+ float64 values, expected/);
  });
});

describe('parseRegimeCheckArgs', () => {
  it('applies defaults and requires --rewired-index/--graphs-dir/--steady-state-dir', () => {
    const args = parseRegimeCheckArgs([
      '--rewired-index',
      'i.json',
      '--graphs-dir',
      'g',
      '--steady-state-dir',
      's'
    ]);
    expect(args.biological).toBe(false);
    expect(args.heldOutStart).toBe(30001);
    expect(args.heldOutCount).toBe(10);
    expect(args.ticks).toBe(1800);
    expect(args.shards).toBe(8);
    expect(args.rewiredIndex).toBe(resolve(process.cwd(), 'i.json'));
    expect(args.graphsDir).toBe(resolve(process.cwd(), 'g'));
    expect(args.steadyStateDir).toBe(resolve(process.cwd(), 's'));
  });

  it('throws without --steady-state-dir', () => {
    expect(() => parseRegimeCheckArgs(['--rewired-index', 'i.json', '--graphs-dir', 'g'])).toThrow(
      /--steady-state-dir/
    );
  });

  it('rejects --graph without --biological', () => {
    expect(() =>
      parseRegimeCheckArgs([
        '--graph',
        'x.bin.gz',
        '--rewired-index',
        'i.json',
        '--graphs-dir',
        'g',
        '--steady-state-dir',
        's'
      ])
    ).toThrow(/--graph requires --biological/);
  });

  it('rejects an unknown flag', () => {
    expect(() =>
      parseRegimeCheckArgs(['--rewired-index', 'i.json', '--graphs-dir', 'g', '--steady-state-dir', 's', '--bogus'])
    ).toThrow(/Unknown argument/);
  });
});
