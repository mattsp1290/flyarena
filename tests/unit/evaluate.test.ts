import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { outputNeuronIndices, validateReadoutWeights } from '../../src/lib/connectome/readout';
import { TRACE_SUBSTEPS } from '../../scripts/training/export-traces';
import { computeGraphIdentity, runExportArms } from '../../scripts/training/export-arms';
import { parseEvaluateArgs, runEvaluate, type EvaluateArgs } from '../../scripts/training/evaluate';
import { createTraceGraph } from '../fixtures/trace-graph';
import { writeTinyRunDir } from '../fixtures/trained-readout-run';

describe('parseEvaluateArgs', () => {
  it('captures a variadic --runs list and applies defaults', () => {
    const args = parseEvaluateArgs(['--runs', 'a', 'b', 'c', '--out', 'out-dir']);
    expect(args.runDirs).toEqual(['a', 'b', 'c']);
    expect(args.outDir).toBe('out-dir');
    expect(args.ticks).toBe(1800);
    expect(args.heldOutStart).toBe(30001);
    expect(args.heldOutCount).toBe(100);
    expect(args.bootstrapResamples).toBe(10000);
  });

  it('requires at least one run directory', () => {
    expect(() => parseEvaluateArgs(['--out', 'x'])).toThrow(/--runs/);
  });

  it('stops --runs at the next flag', () => {
    const args = parseEvaluateArgs(['--runs', 'a', 'b', '--ticks', '10']);
    expect(args.runDirs).toEqual(['a', 'b']);
    expect(args.ticks).toBe(10);
  });

  it('parses --parity-* and --gpu-rerun-* flags', () => {
    const args = parseEvaluateArgs([
      '--runs',
      'a',
      '--parity-graph-sha256',
      'deadbeef',
      '--parity-k',
      '4',
      '--parity-passed-at',
      '2026-09-24T00:00:00Z',
      '--gpu-rerun-max-abs-diff',
      '0.0012',
      '--gpu-rerun-fitness-delta',
      '-3.5'
    ]);
    expect(args.parityGraphSha256).toBe('deadbeef');
    expect(args.parityK).toBe(4);
    expect(args.parityPassedAt).toBe('2026-09-24T00:00:00Z');
    expect(args.gpuRerunMaxAbsDiff).toBeCloseTo(0.0012);
    expect(args.gpuRerunFitnessDelta).toBeCloseTo(-3.5);
  });

  it('rejects a partial set of --parity-* flags', () => {
    expect(() => parseEvaluateArgs(['--runs', 'a', '--parity-k', '4'])).toThrow(/together/);
    expect(() =>
      parseEvaluateArgs(['--runs', 'a', '--parity-graph-sha256', 'x', '--parity-k', '4'])
    ).toThrow(/together/);
  });

  it('rejects a negative --gpu-rerun-max-abs-diff', () => {
    expect(() => parseEvaluateArgs(['--runs', 'a', '--gpu-rerun-max-abs-diff', '-1'])).toThrow(/non-negative/);
  });

  it('accepts a negative --gpu-rerun-fitness-delta (a signed difference), passed alongside --gpu-rerun-max-abs-diff', () => {
    const args = parseEvaluateArgs([
      '--runs',
      'a',
      '--gpu-rerun-max-abs-diff',
      '0.5',
      '--gpu-rerun-fitness-delta',
      '-2.5'
    ]);
    expect(args.gpuRerunFitnessDelta).toBeCloseTo(-2.5);
  });

  it('rejects either --gpu-rerun-* flag passed alone (both or neither: they come from one CUDA rerun)', () => {
    expect(() => parseEvaluateArgs(['--runs', 'a', '--gpu-rerun-max-abs-diff', '0.5'])).toThrow(/together/);
    expect(() => parseEvaluateArgs(['--runs', 'a', '--gpu-rerun-fitness-delta', '-2.5'])).toThrow(/together/);
  });

  it('requireFloat rejects an empty/whitespace --gpu-rerun-fitness-delta rather than silently recording 0', () => {
    expect(() =>
      parseEvaluateArgs(['--runs', 'a', '--gpu-rerun-max-abs-diff', '0', '--gpu-rerun-fitness-delta', ''])
    ).toThrow(/finite number/);
  });
});

interface TinyArtifactArm {
  readonly w1: string;
  readonly b1: string;
  readonly w2: string;
  readonly b2: string;
}
interface TinyArtifact {
  readonly version: number;
  readonly hiddenSize: number;
  readonly inputSize: number;
  readonly arms: Readonly<Record<string, TinyArtifactArm>>;
}
interface TinyReport {
  readonly arms: Readonly<
    Record<string, { readonly replicas: Readonly<Record<string, { readonly trained: { readonly n: number } }>> }>
  >;
  readonly sideBySide: readonly unknown[];
  readonly armPairs: readonly unknown[];
  readonly warnings: readonly string[];
}

const decodeBase64Float32 = (base64: string): Float32Array => {
  const buffer = Buffer.from(base64, 'base64');
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

describe('runEvaluate (tiny fixture, trace graph)', () => {
  const buildFixture = (): { root: string; armsDir: string; runDirs: string[] } => {
    const root = mkdtempSync(join(tmpdir(), 'evaluate-fixture-'));
    const armsRoot = join(root, 'arms');
    const exportResult = runExportArms({ outDir: armsRoot, fixtureRewire: true, fixtureRewireSeed: 3 });

    const graph = createTraceGraph();
    const D = outputNeuronIndices(graph).length;
    const H = 4;

    const runDirs: string[] = [];
    let weightSeed = 1;
    for (const arm of ['biological', 'rewired', 'disconnected'] as const) {
      const dir = join(root, 'runs', `${arm}-101`);
      writeTinyRunDir({ dir, arm, trainerSeed: 101, D, H, substeps: TRACE_SUBSTEPS, weightSeed, includeEnv: true });
      runDirs.push(dir);
      weightSeed += 17;
    }

    return { root, armsDir: exportResult.outDir, runDirs };
  };

  const baseArgs = (armsDir: string, runDirs: readonly string[], outDir: string): EvaluateArgs => ({
    armsDir,
    runDirs,
    outDir,
    outDirExplicit: true,
    ticks: 40,
    substeps: TRACE_SUBSTEPS,
    heldOutStart: 30001,
    heldOutCount: 4,
    bootstrapResamples: 50,
    bootstrapSeed: 12345
  });

  it('writes a shipped artifact + report, and running twice gives byte-identical report.json', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outA = join(root, 'out-a');
      const outB = join(root, 'out-b');

      const resultA = runEvaluate(baseArgs(armsDir, runDirs, outA));
      const resultB = runEvaluate(baseArgs(armsDir, runDirs, outB));

      expect(resultA.warnings).toEqual([]);
      expect(resultA.artifactWritten).toBe(true);
      expect(resultB.artifactWritten).toBe(true);

      const reportA = readFileSync(resolve(outA, 'trained-readout-v1.report.json'));
      const reportB = readFileSync(resolve(outB, 'trained-readout-v1.report.json'));
      expect(reportB.equals(reportA)).toBe(true);

      const artifact = JSON.parse(readFileSync(resolve(outA, 'trained-readout-v1.json'), 'utf8')) as TinyArtifact;
      expect(Object.keys(artifact.arms).sort()).toEqual(['biological', 'disconnected', 'rewired']);

      const graph = createTraceGraph();
      for (const armName of Object.keys(artifact.arms)) {
        const arm = artifact.arms[armName];
        const weights = {
          inputSize: artifact.inputSize,
          hiddenSize: artifact.hiddenSize,
          w1: decodeBase64Float32(arm.w1),
          b1: decodeBase64Float32(arm.b1),
          w2: decodeBase64Float32(arm.w2),
          b2: decodeBase64Float32(arm.b2)
        };
        expect(() => validateReadoutWeights(weights, graph)).not.toThrow();
      }

      const report = JSON.parse(readFileSync(resolve(outA, 'trained-readout-v1.report.json'), 'utf8')) as TinyReport;
      expect(report.arms.biological.replicas['101'].trained.n).toBe(4);
      expect(report.sideBySide.length).toBeGreaterThan(0);
      expect(report.armPairs.length).toBeGreaterThan(0);
      expect(report.warnings).toEqual([]);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('manifest records the CEM config/training-seed RNG note and the parity/gpuRerun fields when passed', () => {
    const root = mkdtempSync(join(tmpdir(), 'evaluate-manifest-fixture-'));
    try {
      const armsRoot = join(root, 'arms');
      const exportResult = runExportArms({ outDir: armsRoot, fixtureRewire: true, fixtureRewireSeed: 3 });
      const graph = createTraceGraph();
      const D = outputNeuronIndices(graph).length;
      const H = 4;
      const cemConfig = {
        population: 128,
        elites: 32,
        generations: 150,
        alpha: 0.7,
        stdFloor: 0.02,
        initStd: 0.5,
        trainingSeedsPerGeneration: 16,
        trainingSeedRange: [1, 10000] as const,
        trainingSeedRng: 'default_rng([trainerSeed, generation])',
        validationSeedRange: [20001, 20064] as const,
        heldOutSeedRange: [30001, 30100] as const
      };

      const runDirs: string[] = [];
      let weightSeed = 1;
      for (const arm of ['biological', 'rewired', 'disconnected'] as const) {
        const dir = join(root, 'runs', `${arm}-101`);
        writeTinyRunDir({ dir, arm, trainerSeed: 101, D, H, substeps: TRACE_SUBSTEPS, weightSeed, includeEnv: true, cemConfig });
        runDirs.push(dir);
        weightSeed += 17;
      }

      // --parity-graph-sha256 must equal the graph actually being evaluated
      // (evaluate.ts now throws otherwise — see the "throws when
      // --parity-graph-sha256 does not match the evaluated graph" test
      // below), so this uses the trace graph's own real identity, not an
      // arbitrary placeholder.
      const realGraphSha256 = computeGraphIdentity().graphArtifactSha256;

      const outDir = join(root, 'out');
      const args: EvaluateArgs = {
        ...baseArgs(exportResult.outDir, runDirs, outDir),
        parityGraphSha256: realGraphSha256,
        parityK: 4,
        parityPassedAt: '2026-09-24T00:00:00Z',
        gpuRerunMaxAbsDiff: 0.0007,
        gpuRerunFitnessDelta: -1.25
      };
      const result = runEvaluate(args);
      expect(result.artifactWritten).toBe(true);
      // These flags are informational, not tied to graphSource==='artifact'
      // (only the "manifest omits the parity block" warning is); passing
      // them should never itself produce a warning.
      expect(result.warnings).toEqual([]);

      const manifest = JSON.parse(readFileSync(resolve(outDir, 'trained-readout-v1.manifest.json'), 'utf8')) as {
        training: Record<string, unknown> | null;
        parity: { graphSha: string; K: number; passedAt: string } | null;
        gpuRerunMaxAbsDiff: number | null;
        gpuRerunFitnessDelta: number | null;
      };
      expect(manifest.training).toMatchObject({
        population: 128,
        elites: 32,
        generations: 150,
        trainingSeedRng: 'default_rng([trainerSeed, generation])'
      });
      expect(manifest.parity).toEqual({ graphSha: realGraphSha256, K: 4, passedAt: '2026-09-24T00:00:00Z' });
      expect(manifest.gpuRerunMaxAbsDiff).toBeCloseTo(0.0007);
      expect(manifest.gpuRerunFitnessDelta).toBeCloseTo(-1.25);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('throws when --parity-graph-sha256 does not match the graph actually being evaluated', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outDir = join(root, 'out-parity-mismatch');
      const args: EvaluateArgs = {
        ...baseArgs(armsDir, runDirs, outDir),
        parityGraphSha256: 'not-the-real-graph-sha',
        parityK: 4,
        parityPassedAt: '2026-09-24T00:00:00Z'
      };
      expect(() => runEvaluate(args)).toThrow(/parity-graph-sha256.*does not match/s);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('CEM config: a later arm’s real config is preserved (not silently dropped) when an earlier arm has none', () => {
    const root = mkdtempSync(join(tmpdir(), 'evaluate-cemconfig-order-'));
    try {
      const armsRoot = join(root, 'arms');
      const exportResult = runExportArms({ outDir: armsRoot, fixtureRewire: true, fixtureRewireSeed: 3 });
      const graph = createTraceGraph();
      const D = outputNeuronIndices(graph).length;
      const H = 4;
      const cemConfig = { population: 128, elites: 32, generations: 150 };

      const runDirs: string[] = [];
      let weightSeed = 1;
      // 'biological' sorts before 'rewired'/'disconnected' in ARM_NAMES, so
      // this exercises the exact order the fix targets: the first-in-order
      // arm ('biological') has NO cemConfig; a later arm ('rewired') does.
      for (const arm of ['biological', 'rewired', 'disconnected'] as const) {
        const dir = join(root, 'runs', `${arm}-101`);
        writeTinyRunDir({
          dir,
          arm,
          trainerSeed: 101,
          D,
          H,
          substeps: TRACE_SUBSTEPS,
          weightSeed,
          includeEnv: true,
          ...(arm === 'biological' ? {} : { cemConfig })
        });
        runDirs.push(dir);
        weightSeed += 17;
      }

      const outDir = join(root, 'out');
      const result = runEvaluate(baseArgs(exportResult.outDir, runDirs, outDir));
      expect(result.artifactWritten).toBe(true);
      // biological (no config) must produce a warning naming the real
      // source arm, not silently pass or blame the wrong arm.
      expect(result.warnings.some((w) => w.includes('biological') && w.includes('no recorded CEM'))).toBe(true);

      const manifest = JSON.parse(readFileSync(resolve(outDir, 'trained-readout-v1.manifest.json'), 'utf8')) as {
        training: Record<string, unknown> | null;
      };
      // The bug this test guards against: rewired's real config would be
      // silently dropped (manifest.training === null) because 'biological'
      // (visited first, no config) was wrongly adopted as the baseline.
      expect(manifest.training).toMatchObject({ population: 128, elites: 32, generations: 150 });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('manifest omits training/parity/gpuRerun fields (nulls) and records no spurious warning in trace-graph mode without them', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outDir = join(root, 'out-no-parity');
      const result = runEvaluate(baseArgs(armsDir, runDirs, outDir));
      expect(result.artifactWritten).toBe(true);
      // Trace-graph fixture runs (graphSource 'trace-graph-fixture') are not
      // production evaluations, so the missing-parity warning must not fire.
      expect(result.warnings).toEqual([]);
      const manifest = JSON.parse(readFileSync(resolve(outDir, 'trained-readout-v1.manifest.json'), 'utf8')) as {
        parity: unknown;
        gpuRerunMaxAbsDiff: unknown;
        gpuRerunFitnessDelta: unknown;
        training: unknown;
      };
      expect(manifest.parity).toBeNull();
      expect(manifest.gpuRerunMaxAbsDiff).toBeNull();
      expect(manifest.gpuRerunFitnessDelta).toBeNull();
      // No cemConfig fields were written by the tiny fixture runs (buildFixture doesn't pass cemConfig).
      expect(manifest.training).toBeNull();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('warns and skips the shipped artifact when a requested arm has no replica-0 (trainerSeed 101) run', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outDir = join(root, 'out-missing');
      // Replace the disconnected arm's replica-0 run with a non-shipped
      // replica (trainerSeed 202): the arm is still requested (it has a
      // run), so it is still scored, but it has no trainerSeed-101 run for
      // the shipped artifact to use.
      const nonShippedDisconnectedDir = join(root, 'runs', 'disconnected-202');
      const graph = createTraceGraph();
      writeTinyRunDir({
        dir: nonShippedDisconnectedDir,
        arm: 'disconnected',
        trainerSeed: 202,
        D: outputNeuronIndices(graph).length,
        H: 4,
        substeps: TRACE_SUBSTEPS,
        weightSeed: 99
      });
      const substitutedRunDirs = runDirs
        .filter((dir) => !dir.includes('disconnected-101'))
        .concat(nonShippedDisconnectedDir);

      const result = runEvaluate(baseArgs(armsDir, substitutedRunDirs, outDir));
      expect(result.artifactWritten).toBe(false);
      expect(result.warnings.some((warning) => warning.includes('disconnected'))).toBe(true);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('report.json is byte-identical regardless of --runs argument order (>= 2 replicas per arm)', () => {
    // A single replica per arm cannot actually exercise the order-fix: with
    // only one trainerSeed per arm, armReplicas' per-arm Map has one entry
    // regardless of --runs order, so sortedReplicas's sort is a no-op and
    // this test would pass even against the pre-fix code. A second replica
    // (trainerSeed 202) per arm makes within-arm ordering — and therefore
    // conditionRng's label-derived seeding, and armPairs'/sideBySide's
    // element order — actually depend on something --runs order could
    // perturb if it weren't fixed.
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const graph = createTraceGraph();
      const D = outputNeuronIndices(graph).length;
      const secondReplicaDirs: string[] = [];
      let weightSeed = 500;
      for (const arm of ['biological', 'rewired', 'disconnected'] as const) {
        const dir = join(root, 'runs', `${arm}-202`);
        writeTinyRunDir({ dir, arm, trainerSeed: 202, D, H: 4, substeps: TRACE_SUBSTEPS, weightSeed, includeEnv: true });
        secondReplicaDirs.push(dir);
        weightSeed += 17;
      }
      const allRunDirs = [...runDirs, ...secondReplicaDirs];
      // A genuine shuffle, not just a reversal: interleave the two
      // replicas across arms differently from `allRunDirs`'s own order.
      const shuffled = [
        secondReplicaDirs[1],
        runDirs[0],
        secondReplicaDirs[2],
        runDirs[1],
        secondReplicaDirs[0],
        runDirs[2]
      ];
      expect([...shuffled].sort()).toEqual([...allRunDirs].sort());

      const outForward = join(root, 'out-forward');
      const outReversed = join(root, 'out-reversed');
      const outShuffled = join(root, 'out-shuffled');

      const forward = runEvaluate(baseArgs(armsDir, allRunDirs, outForward));
      const reversed = runEvaluate(baseArgs(armsDir, [...allRunDirs].reverse(), outReversed));
      const shuffledResult = runEvaluate(baseArgs(armsDir, shuffled, outShuffled));

      expect(forward.artifactWritten).toBe(true);
      expect(reversed.artifactWritten).toBe(true);
      expect(shuffledResult.artifactWritten).toBe(true);
      const reportForward = readFileSync(resolve(outForward, 'trained-readout-v1.report.json'));
      const reportReversed = readFileSync(resolve(outReversed, 'trained-readout-v1.report.json'));
      const reportShuffled = readFileSync(resolve(outShuffled, 'trained-readout-v1.report.json'));
      expect(reportReversed.equals(reportForward)).toBe(true);
      expect(reportShuffled.equals(reportForward)).toBe(true);

      // And: this arm/replica's own CI must not depend on which OTHER
      // arms/replicas were also evaluated (conditionRng's per-label
      // seeding, not just sortedReplicas' ordering).
      const outSubset = join(root, 'out-subset');
      const subsetResult = runEvaluate(
        baseArgs(armsDir, [join(root, 'runs', 'biological-101'), join(root, 'runs', 'biological-202')], outSubset)
      );
      expect(subsetResult.artifactWritten).toBe(false); // rewired/disconnected not evaluated at all
      const fullReport = JSON.parse(readFileSync(resolve(outForward, 'trained-readout-v1.report.json'), 'utf8')) as {
        arms: { biological: { replicas: { '101': { trained: { ci95: [number, number] } } } } };
      };
      const subsetReport = JSON.parse(readFileSync(resolve(outSubset, 'trained-readout-v1.report.json'), 'utf8')) as {
        arms: { biological: { replicas: { '101': { trained: { ci95: [number, number] } } } } };
      };
      expect(subsetReport.arms.biological.replicas['101'].trained.ci95).toEqual(
        fullReport.arms.biological.replicas['101'].trained.ci95
      );
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('throws when a run was trained at a different substep count than --substeps', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const graph = createTraceGraph();
      const mismatchedDir = join(root, 'runs', 'biological-mismatched-substeps');
      writeTinyRunDir({
        dir: mismatchedDir,
        arm: 'biological',
        trainerSeed: 202,
        D: outputNeuronIndices(graph).length,
        H: 4,
        substeps: TRACE_SUBSTEPS + 95, // deliberately different from baseArgs' substeps
        weightSeed: 5
      });
      const outDir = join(root, 'out-substeps-mismatch');
      expect(() => runEvaluate(baseArgs(armsDir, [...runDirs, mismatchedDir], outDir))).toThrow(/substeps/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('rejects config.json fields with the wrong type (e.g. a string trainerSeed)', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const graph = createTraceGraph();
      const badDir = join(root, 'runs', 'biological-bad-config');
      writeTinyRunDir({
        dir: badDir,
        arm: 'biological',
        trainerSeed: 303,
        D: outputNeuronIndices(graph).length,
        H: 4,
        substeps: TRACE_SUBSTEPS,
        weightSeed: 6
      });
      const configPath = resolve(badDir, 'config.json');
      const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
      config.trainerSeed = '303'; // string instead of number
      writeFileSync(configPath, JSON.stringify(config));

      const outDir = join(root, 'out-bad-config');
      expect(() => runEvaluate(baseArgs(armsDir, [...runDirs, badDir], outDir))).toThrow(/trainerSeed/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('rejects an arm bundle whose content does not match its own recorded sha256 (tampered/stale)', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const bundlePath = resolve(armsDir, 'rewired.json');
      const bundle = JSON.parse(readFileSync(bundlePath, 'utf8')) as { contactMagnitudes: number[] };
      bundle.contactMagnitudes = bundle.contactMagnitudes.map((value) => value * 5);
      writeFileSync(bundlePath, JSON.stringify(bundle));

      const outDir = join(root, 'out-tampered-bundle');
      expect(() => runEvaluate(baseArgs(armsDir, runDirs, outDir))).toThrow(/sha256/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('rejects a bundle whose "arm" field does not match its filename (mislabeled bundle)', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const biologicalPath = resolve(armsDir, 'biological.json');
      const rewiredPath = resolve(armsDir, 'rewired.json');
      writeFileSync(rewiredPath, readFileSync(biologicalPath)); // copy biological over rewired verbatim

      const outDir = join(root, 'out-mislabeled-bundle');
      expect(() => runEvaluate(baseArgs(armsDir, runDirs, outDir))).toThrow(/declares arm/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('refuses to write to the default --out (public/data) in trace-graph mode', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const args: EvaluateArgs = {
        ...baseArgs(armsDir, runDirs, 'public/data'),
        outDirExplicit: false
      };
      expect(() => runEvaluate(args)).toThrow(/public\/data/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
