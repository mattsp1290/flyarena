// @vitest-environment node
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { gzipSync } from 'node:zlib';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { encodeGraphBinary, SUPPORTED_FORMAT_VERSION, type ConnectomeGraph, type GraphMetadata } from '../../src/lib/connectome/format';
import { runTask, type RegimeWorkerTask } from '../../scripts/null/regime-task';
import { parseRegimeCheckArgs, verifySteadyStateManifest } from '../../scripts/null/regime-check';
import { sha256Hex } from '../../scripts/training/fsio';

/**
 * `.agents/plans/null-explanation/02-transfer-and-features.md`'s WP2 change
 * surface: "tests/unit/regime-check.test.ts (new): The saturation counter
 * on a fixture that clamps deterministically." Exercises
 * `scripts/null/regime-task.ts`'s `runTask` directly (in-process, no
 * `fork`/IPC — the same function the forked worker calls per task) against
 * hand-built single-neuron graphs on disk: one that saturates every
 * substep by construction (`clampFraction` must be exactly `1`), and one
 * far inside the linear regime (`clampFraction` must be exactly `0` and
 * `steadyStateDistance` must be small once the episode has run long enough
 * to approach its linear fixed point). `runTask` lives in `regime-task.ts`,
 * not `regime-worker.ts`, specifically so importing it here never registers
 * an IPC listener on this test process's own message channel (a dual-review
 * finding -- see `regime-task.ts`'s module doc comment).
 *
 * A separate `describe` block below exercises `regime-check.ts`'s real CLI
 * end to end via `node:child_process.spawnSync` (the actual `fork`/IPC path,
 * the manifest-based steady-state verification, and the `disconnected`/
 * `rewired` branches of `graphFromTaskMode` -- none of which the in-process
 * `runTask` tests above cover, since they only ever use `mode: 'biological'`
 * with a hand-written sidecar; also a dual-review finding).
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

  const writeSteadyState = (row: Float64Array, name: string): { path: string; sha256: string } => {
    const path = join(root, `${name}.steadystate.f64`);
    const bytes = Buffer.from(row.buffer, row.byteOffset, row.byteLength);
    writeFileSync(path, bytes);
    return { path, sha256: sha256Hex(bytes) };
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
    const steadyState = writeSteadyState(steadyStateRow, 'saturating');

    const task: RegimeWorkerTask = {
      graphId: 'saturating',
      mode: 'biological',
      path,
      expectedSha256: sha256,
      steadyStatePath: steadyState.path,
      steadyStateSha256: steadyState.sha256,
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
    const steadyState = writeSteadyState(steadyStateRow, 'stable');

    const task: RegimeWorkerTask = {
      graphId: 'stable',
      mode: 'biological',
      path,
      expectedSha256: sha256,
      steadyStatePath: steadyState.path,
      steadyStateSha256: steadyState.sha256,
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
    const steadyState = writeSteadyState(wrongRow, 'bad-sidecar');

    const task: RegimeWorkerTask = {
      graphId: 'bad-sidecar',
      mode: 'biological',
      path,
      expectedSha256: sha256,
      steadyStatePath: steadyState.path,
      steadyStateSha256: steadyState.sha256,
      heldOutSeeds: [30001],
      ticks: 10
    };

    expect(() => runTask(task)).toThrow(/steadystate\.f64.*has \d+ float64 values, expected/);
  });

  it('throws when the sidecar bytes do not match the expected sha256 (a stale or partial sidecar)', () => {
    const graph = oneNeuronGraph({}, 0.05, 0.4);
    const { path, sha256 } = writeGraph(graph, 'stale-sidecar');
    const steadyStateRow = analyticSteadyStateRow(0.35, 1, 0.05, 0.4);
    const steadyState = writeSteadyState(steadyStateRow, 'stale-sidecar');

    const task: RegimeWorkerTask = {
      graphId: 'stale-sidecar',
      mode: 'biological',
      path,
      expectedSha256: sha256,
      steadyStatePath: steadyState.path,
      steadyStateSha256: 'f'.repeat(64), // deliberately wrong
      heldOutSeeds: [30001],
      ticks: 10
    };

    expect(() => runTask(task)).toThrow(/sha256 .* does not match the manifest's expected/);
  });

  it('discriminates a row-major vs. column-major steady-state sidecar layout on a 2-neuron, multi-channel graph', () => {
    // Two neurons, each mapped to a *different* input channel, no
    // recurrent edges (so the steady-state prediction is exact from the
    // first tick, no convergence needed): neuron 0 <- channel 1
    // (foodDistance), neuron 1 <- channel 3 (hazardDistance). The
    // steady-state map M is 2x8 (row-major: M[i*8+c]); a transposed
    // (column-major, 8x2-as-if-2x8) read would place neuron 1's response
    // to channel 3 at a different offset than a correct row-major read --
    // see this suite's module doc comment (a dual-review finding: the
    // single-neuron fixtures above have neuronCount=1, where row-major and
    // column-major layouts are byte-identical, so they cannot catch a
    // transposed sidecar).
    const metadata: GraphMetadata = {
      formatVersion: SUPPORTED_FORMAT_VERSION,
      neuronCount: 2,
      edgeCount: 0,
      inputChannelCount: CHANNEL_COUNT,
      outputPopulationCount: 3,
      timestepSeconds: 1 / 30,
      leakRate: 0.35,
      rateMin: -2,
      rateMax: 2,
      inputClampMin: -1,
      inputClampMax: 1,
      globalGain: 1
    };
    const graph: ConnectomeGraph = {
      metadata,
      biologicalIds: BigUint64Array.from([1n, 2n]),
      presynapticOffsets: Uint32Array.from([0, 0, 0]),
      postsynapticIndices: Uint32Array.from([]),
      contactMagnitudes: Float32Array.from([]),
      presynapticSigns: Int8Array.from([1, 1]),
      inputChannelIndex: Int32Array.from([1, 3]), // foodDistance, hazardDistance
      inputWeight: Float32Array.from([0.4, 0.6]),
      outputPopulationIndex: Int32Array.from([0, -1]),
      outputWeight: Float32Array.from([1, 0])
    };
    const { path, sha256 } = writeGraph(graph, 'layout-discriminating');

    // With no recurrent edges, A = 0, so M = B / leakRate exactly: row i,
    // channel c is `inputWeight[i] / leakRate` if `inputChannelIndex[i] ==
    // c`, else 0. Row-major (correct): M[0*8+1] = 0.4/0.35, M[1*8+3] =
    // 0.6/0.35, every other entry 0.
    const correctM = new Float64Array(2 * CHANNEL_COUNT);
    correctM[0 * CHANNEL_COUNT + 1] = 0.4 / 0.35;
    correctM[1 * CHANNEL_COUNT + 3] = 0.6 / 0.35;

    // A deliberately *transposed* sidecar: `regime-task.ts` always reads
    // a sidecar as row-major `neuronCount x inputChannelCount`
    // (`M[i * inputChannelCount + c]`), so writing the same values in
    // column-major order (`wrong[c * neuronCount + i] = correctM[i *
    // CHANNEL_COUNT + c]`) and having the worker read it with its usual
    // row-major formula simulates exactly the cross-language layout bug
    // this test exists to catch -- most entries land at the wrong
    // (neuron, channel) slot.
    const transposedM = new Float64Array(2 * CHANNEL_COUNT);
    for (let neuron = 0; neuron < 2; neuron += 1) {
      for (let channel = 0; channel < CHANNEL_COUNT; channel += 1) {
        transposedM[channel * 2 + neuron] = correctM[neuron * CHANNEL_COUNT + channel];
      }
    }

    const runWithSteadyState = (row: Float64Array, name: string) => {
      const steadyState = writeSteadyState(row, name);
      const task: RegimeWorkerTask = {
        graphId: name,
        mode: 'biological',
        path,
        expectedSha256: sha256,
        steadyStatePath: steadyState.path,
        steadyStateSha256: steadyState.sha256,
        heldOutSeeds: [30001],
        ticks: 30
      };
      return runTask(task)[0];
    };

    const correctResult = runWithSteadyState(correctM, 'layout-correct');
    const transposedResult = runWithSteadyState(transposedM, 'layout-transposed');

    // Both runs simulate the exact same graph/episode, so `rate[neuron]`
    // (the actual dynamics) is identical between them -- only the
    // *predicted* `r*` differs, by construction. A correct row-major read
    // must therefore land closer to the true dynamics than a transposed
    // one; asserting the relative ordering (rather than an absolute
    // threshold) avoids needing to reason about how far this
    // no-recurrence-yet-still-per-tick-varying-input system has actually
    // converged after 30 ticks.
    expect(correctResult.steadyStateDistance).toBeLessThan(transposedResult.steadyStateDistance);
  });
});

describe('regime-check.ts CLI: end to end via a real forked worker', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'regime-check-cli-'));
  });

  afterEach(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  it('runs biological/disconnected/rewired through the real fork/IPC path and writes a well-formed regime.json', () => {
    const graphsDir = join(root, 'graphs');
    mkdirSync(graphsDir, { recursive: true });
    const steadyStateDir = join(root, 'steady-state');
    mkdirSync(steadyStateDir, { recursive: true });

    const bioGraph = oneNeuronGraph({}, 0.05, 0.4);
    const bioBinary = Buffer.from(encodeGraphBinary(bioGraph));
    const bioSha256 = sha256Hex(bioBinary);
    const bioPath = join(root, 'biological.bin.gz');
    writeFileSync(bioPath, gzipSync(bioBinary));

    const rewiredGraph = oneNeuronGraph({}, 0.02, 0.3);
    const rewiredBinary = Buffer.from(encodeGraphBinary(rewiredGraph));
    const rewiredSha256 = sha256Hex(rewiredBinary);
    const rewiredGzip = gzipSync(rewiredBinary);
    writeFileSync(join(graphsDir, 'rewired-seed0.bin.gz'), rewiredGzip);

    const index = {
      sourceArtifact: 'biological.bin.gz',
      sourceSha256: bioSha256,
      rewireSourceSha256: sha256Hex(Buffer.from('regime-check-cli-fixture')),
      seeds: [
        {
          seed: 0,
          artifact: 'rewired-seed0.bin.gz',
          binarySha256: rewiredSha256,
          binaryBytes: rewiredBinary.byteLength,
          gzipSha256: sha256Hex(rewiredGzip),
          gzipBytes: rewiredGzip.byteLength,
          stats: { acceptedSwaps: 1, attempts: 1 }
        }
      ]
    };
    const indexPath = join(root, 'index.json');
    writeFileSync(indexPath, JSON.stringify(index));

    // The disconnected control (edgeCount forced to 0 by `graphFromTaskMode`)
    // has A = 0, so M = B / leakRate; biological/rewired's own single
    // recurrent edge means M = B / (leakRate - globalGain*magnitude).
    const analyticRow = (leakRate: number, globalGain: number, selfLoopMagnitude: number, inputWeight: number) => {
      const row = new Float64Array(CHANNEL_COUNT);
      row[FOOD_DISTANCE_CHANNEL] = inputWeight / (leakRate - globalGain * selfLoopMagnitude);
      return row;
    };
    const writeManifestSidecar = (graphId: string, row: Float64Array): { sha256: string } => {
      const bytes = Buffer.from(row.buffer, row.byteOffset, row.byteLength);
      writeFileSync(join(steadyStateDir, `${graphId}.steadystate.f64`), bytes);
      return { sha256: sha256Hex(bytes) };
    };
    const bioEntry = writeManifestSidecar('biological', analyticRow(0.35, 1, 0.05, 0.4));
    const discEntry = writeManifestSidecar('disconnected', analyticRow(0.35, 1, 0, 0.4));
    const rewiredEntry = writeManifestSidecar('rewired-0', analyticRow(0.35, 1, 0.02, 0.3));
    const manifest = {
      version: 1,
      rewireSourceSha256: index.rewireSourceSha256,
      graphs: {
        biological: { graphBinarySha256: bioSha256, sidecarSha256: bioEntry.sha256 },
        disconnected: { graphBinarySha256: bioSha256, sidecarSha256: discEntry.sha256 },
        'rewired-0': { graphBinarySha256: rewiredSha256, sidecarSha256: rewiredEntry.sha256 }
      }
    };
    writeFileSync(join(steadyStateDir, 'manifest.json'), JSON.stringify(manifest, null, 2));

    const out = join(root, 'regime.json');
    const result = spawnSync(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/null/regime-check.ts',
        '--biological',
        '--graph',
        bioPath,
        '--rewired-index',
        indexPath,
        '--graphs-dir',
        graphsDir,
        '--steady-state-dir',
        steadyStateDir,
        '--held-out-start',
        '30001',
        '--held-out-count',
        '2',
        '--ticks',
        '20',
        '--shards',
        '2',
        '--out',
        out
      ],
      { encoding: 'utf8', timeout: 60_000 }
    );
    expect(result.status, result.stderr).toBe(0);

    const parsed = JSON.parse(readFileSync(out, 'utf8'));
    expect(parsed.version).toBe(1);
    expect(parsed.ticks).toBe(20);
    expect(parsed.substeps).toBe(4);
    expect(parsed.seeds).toEqual({ start: 30001, count: 2 });
    expect(parsed.biological.heldOutSeeds).toEqual([30001, 30002]);
    expect(parsed.disconnected.heldOutSeeds).toEqual([30001, 30002]);
    expect(parsed.rewired).toHaveLength(1);
    expect(parsed.rewired[0].seed).toBe(0);
    for (const clampFraction of [...parsed.biological.clampFraction, ...parsed.disconnected.clampFraction, ...parsed.rewired[0].clampFraction]) {
      expect(clampFraction).toBe(0); // far inside the linear regime by construction
    }
  });

  it('rejects a steady-state manifest computed against a different rewire batch', () => {
    const graphsDir = join(root, 'graphs');
    mkdirSync(graphsDir, { recursive: true });
    const steadyStateDir = join(root, 'steady-state');
    mkdirSync(steadyStateDir, { recursive: true });

    const bioGraph = oneNeuronGraph({}, 0.05, 0.4);
    const bioBinary = Buffer.from(encodeGraphBinary(bioGraph));
    const bioSha256 = sha256Hex(bioBinary);
    const bioPath = join(root, 'biological.bin.gz');
    writeFileSync(bioPath, gzipSync(bioBinary));

    const rewiredBinary = bioBinary; // reused verbatim; this test only exercises the manifest-mismatch path
    const rewiredSha256 = bioSha256;
    const rewiredGzip = gzipSync(rewiredBinary);
    writeFileSync(join(graphsDir, 'rewired-seed0.bin.gz'), rewiredGzip);

    const index = {
      sourceArtifact: 'biological.bin.gz',
      sourceSha256: bioSha256,
      rewireSourceSha256: sha256Hex(Buffer.from('this-run')),
      seeds: [
        {
          seed: 0,
          artifact: 'rewired-seed0.bin.gz',
          binarySha256: rewiredSha256,
          binaryBytes: rewiredBinary.byteLength,
          gzipSha256: sha256Hex(rewiredGzip),
          gzipBytes: rewiredGzip.byteLength,
          stats: { acceptedSwaps: 1, attempts: 1 }
        }
      ]
    };
    writeFileSync(join(root, 'index.json'), JSON.stringify(index));
    writeFileSync(
      join(steadyStateDir, 'manifest.json'),
      JSON.stringify({ version: 1, rewireSourceSha256: sha256Hex(Buffer.from('a-different-run')), graphs: {} })
    );

    const out = join(root, 'regime.json');
    const result = spawnSync(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/null/regime-check.ts',
        '--biological',
        '--graph',
        bioPath,
        '--rewired-index',
        join(root, 'index.json'),
        '--graphs-dir',
        graphsDir,
        '--steady-state-dir',
        steadyStateDir,
        '--held-out-count',
        '1',
        '--ticks',
        '10',
        '--shards',
        '1',
        '--out',
        out
      ],
      { encoding: 'utf8', timeout: 60_000 }
    );
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/different rewire batch/);
  });
});

describe('verifySteadyStateManifest', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'verify-steady-state-manifest-'));
  });

  afterEach(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  const writeManifest = (manifest: unknown): void => {
    writeFileSync(join(root, 'manifest.json'), JSON.stringify(manifest));
  };

  it('rejects a manifest missing a required graph entry', () => {
    writeManifest({
      version: 1,
      rewireSourceSha256: 'a'.repeat(64),
      graphs: { biological: { graphBinarySha256: 'b'.repeat(64), sidecarSha256: 'c'.repeat(64) } }
    });

    expect(() =>
      verifySteadyStateManifest(root, 'a'.repeat(64), [
        { graphId: 'biological', expectedSha256: 'b'.repeat(64) },
        { graphId: 'rewired-0', expectedSha256: 'd'.repeat(64) } // no manifest entry
      ])
    ).toThrow(/no steady-state manifest entry/);
  });

  it('rejects a per-graph graphBinarySha256 mismatch (the stale-sidecar scenario the manifest exists for)', () => {
    writeManifest({
      version: 1,
      rewireSourceSha256: 'a'.repeat(64),
      graphs: {
        'rewired-0': { graphBinarySha256: 'stale-sha-from-an-earlier-batch'.padEnd(64, '0'), sidecarSha256: 'c'.repeat(64) }
      }
    });

    expect(() =>
      verifySteadyStateManifest(root, 'a'.repeat(64), [{ graphId: 'rewired-0', expectedSha256: 'd'.repeat(64) }])
    ).toThrow(/computed from a different graph/);
  });

  it('returns each graphId -> sidecarSha256 when every task matches the manifest', () => {
    writeManifest({
      version: 1,
      rewireSourceSha256: 'a'.repeat(64),
      graphs: {
        biological: { graphBinarySha256: 'b'.repeat(64), sidecarSha256: 'sidecar-bio'.padEnd(64, '0') },
        'rewired-0': { graphBinarySha256: 'd'.repeat(64), sidecarSha256: 'sidecar-rewired'.padEnd(64, '0') }
      }
    });

    const result = verifySteadyStateManifest(root, 'a'.repeat(64), [
      { graphId: 'biological', expectedSha256: 'b'.repeat(64) },
      { graphId: 'rewired-0', expectedSha256: 'd'.repeat(64) }
    ]);

    expect(result.get('biological')).toBe('sidecar-bio'.padEnd(64, '0'));
    expect(result.get('rewired-0')).toBe('sidecar-rewired'.padEnd(64, '0'));
  });

  it('rejects a manifest from a different rewire batch before checking any per-graph entry', () => {
    writeManifest({ version: 1, rewireSourceSha256: 'different-batch'.padEnd(64, '0'), graphs: {} });

    expect(() => verifySteadyStateManifest(root, 'this-batch'.padEnd(64, '0'), [])).toThrow(/different rewire batch/);
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
