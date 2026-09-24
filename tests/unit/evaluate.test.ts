import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { outputNeuronIndices, validateReadoutWeights } from '../../src/lib/connectome/readout';
import { TRACE_SUBSTEPS } from '../../scripts/training/export-traces';
import { runExportArms } from '../../scripts/training/export-arms';
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

  it('report.json is byte-identical regardless of --runs argument order', () => {
    const { root, armsDir, runDirs } = buildFixture();
    try {
      const outForward = join(root, 'out-forward');
      const outReversed = join(root, 'out-reversed');

      const forward = runEvaluate(baseArgs(armsDir, runDirs, outForward));
      const reversed = runEvaluate(baseArgs(armsDir, [...runDirs].reverse(), outReversed));

      expect(forward.artifactWritten).toBe(true);
      expect(reversed.artifactWritten).toBe(true);
      const reportForward = readFileSync(resolve(outForward, 'trained-readout-v1.report.json'));
      const reportReversed = readFileSync(resolve(outReversed, 'trained-readout-v1.report.json'));
      expect(reportReversed.equals(reportForward)).toBe(true);
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
