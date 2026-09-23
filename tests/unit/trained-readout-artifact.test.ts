import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { readoutParameterCount, validateReadoutWeights, type ReadoutWeights } from '../../src/lib/connectome/readout';
import { loadGraphArtifact } from '../../scripts/training/export-traces';
import { createTraceGraph } from '../fixtures/trace-graph';

/**
 * `tests/fixtures/trained-readout-tiny.json` (+ `.manifest.json`) is a
 * small, hand-generated `trained-readout-v1.json`-shaped fixture (D = 6,
 * matching `tests/fixtures/trace-graph.ts`'s output-neuron count; H = 3),
 * used to exercise the artifact's decode/validate path without a real
 * training run. It is not derived from any real CEM training — its
 * manifest says so explicitly.
 */
const HERE = dirname(fileURLToPath(import.meta.url));
const TINY_ARTIFACT_PATH = resolve(HERE, '../fixtures/trained-readout-tiny.json');
const TINY_MANIFEST_PATH = resolve(HERE, '../fixtures/trained-readout-tiny.manifest.json');

interface ArtifactArm {
  readonly w1: string;
  readonly b1: string;
  readonly w2: string;
  readonly b2: string;
}
interface TrainedReadoutArtifact {
  readonly version: number;
  readonly hiddenSize: number;
  readonly inputSize: number;
  readonly arms: Readonly<Record<string, ArtifactArm>>;
}

const sha256Hex = (data: Buffer | string): string => createHash('sha256').update(data).digest('hex');

const decodeBase64Float32 = (base64: string): Float32Array => {
  const buffer = Buffer.from(base64, 'base64');
  return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
};

const decodeArm = (arm: ArtifactArm, inputSize: number, hiddenSize: number): ReadoutWeights => ({
  inputSize,
  hiddenSize,
  w1: decodeBase64Float32(arm.w1),
  b1: decodeBase64Float32(arm.b1),
  w2: decodeBase64Float32(arm.w2),
  b2: decodeBase64Float32(arm.b2)
});

const readArtifact = (path: string): { artifact: TrainedReadoutArtifact; raw: Buffer } => {
  const raw = readFileSync(path);
  return { artifact: JSON.parse(raw.toString('utf8')) as TrainedReadoutArtifact, raw };
};

describe('trained-readout-v1 artifact fixture (tests/fixtures/trained-readout-tiny.json)', () => {
  it('validateReadoutWeights passes for every arm, and parameter counts agree across arms', () => {
    const { artifact } = readArtifact(TINY_ARTIFACT_PATH);
    const graph = createTraceGraph(); // D = 6, matching this fixture's inputSize.
    expect(artifact.inputSize).toBe(6);

    const armNames = Object.keys(artifact.arms).sort();
    expect(armNames).toEqual(['biological', 'disconnected', 'rewired']);

    const expectedParameterCount = readoutParameterCount(artifact.inputSize, artifact.hiddenSize);
    const paramCounts: number[] = [];
    for (const armName of armNames) {
      const weights = decodeArm(artifact.arms[armName], artifact.inputSize, artifact.hiddenSize);
      expect(() => validateReadoutWeights(weights, graph)).not.toThrow();
      paramCounts.push(weights.w1.length + weights.b1.length + weights.w2.length + weights.b2.length);
    }
    expect(new Set(paramCounts).size).toBe(1);
    expect(paramCounts[0]).toBe(expectedParameterCount);
  });

  it('manifest sha256 equals the sha256 of the artifact fixture bytes', () => {
    const { raw } = readArtifact(TINY_ARTIFACT_PATH);
    const manifest = JSON.parse(readFileSync(TINY_MANIFEST_PATH, 'utf8')) as { artifactSha256: string };
    expect(manifest.artifactSha256).toBe(sha256Hex(raw));
  });
});

describe('trained-readout-v1 real artifact (public/data/), when present', () => {
  // WP4 (this bean) never writes these files (that is WP5, the production
  // run on the real MaleCNS graph). This block is here so it activates
  // automatically once WP5 lands, without another WP having to remember to
  // add it.
  const realArtifactPath = resolve(process.cwd(), 'public/data/trained-readout-v1.json');
  const realManifestPath = resolve(process.cwd(), 'public/data/trained-readout-v1.manifest.json');
  const realGraphPath = resolve(process.cwd(), 'public/data/malecns-arena-v1.bin.gz');
  const filesExist = existsSync(realArtifactPath) && existsSync(realManifestPath) && existsSync(realGraphPath);

  it.skipIf(!filesExist)(
    'validateReadoutWeights passes for every arm, parameter counts agree, and the manifest sha256 matches',
    () => {
      const { artifact, raw } = readArtifact(realArtifactPath);
      const manifest = JSON.parse(readFileSync(realManifestPath, 'utf8')) as { artifactSha256: string };
      expect(manifest.artifactSha256).toBe(sha256Hex(raw));

      const graph = loadGraphArtifact(realGraphPath);
      const expectedParameterCount = readoutParameterCount(artifact.inputSize, artifact.hiddenSize);
      const paramCounts: number[] = [];
      for (const armName of Object.keys(artifact.arms)) {
        const weights = decodeArm(artifact.arms[armName], artifact.inputSize, artifact.hiddenSize);
        expect(() => validateReadoutWeights(weights, graph)).not.toThrow();
        paramCounts.push(weights.w1.length + weights.b1.length + weights.w2.length + weights.b2.length);
      }
      expect(new Set(paramCounts).size).toBe(1);
      expect(paramCounts[0]).toBe(expectedParameterCount);
    }
  );
});
