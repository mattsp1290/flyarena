import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { readoutParameterCount, type ReadoutWeights } from '../../src/lib/connectome/readout';
import { ARM_NAMES, type ArmName } from './arms';
import { readNpyFloat32Array } from './npy';

/**
 * The run-directory contract `scripts/training/evaluate.ts` defines and
 * consumes (WP3, `training/src/flyarena_training/cem.py`, is not built yet
 * — see `RunConfig`'s doc comment below). Extracted from `evaluate.ts`
 * (round-1's `graph-provenance` S9, unaddressed across two fix rounds) so a
 * future WP3 driver can depend on the contract (`RunConfig`, `readRunDir`)
 * without importing all of `evaluate.ts`'s CLI/report-writing code.
 */

/**
 * `<run-dir>/`:
 *   - `config.json` (required): a `RunConfig` (below), one arm/replica's
 *     training configuration.
 *   - `theta_final.npy` (required): a 1-D little-endian float32 (`<f4`) or
 *     float64 (`<f8`) NumPy array (`scripts/training/npy.ts`) of length
 *     `config.parameterCount`, the flat concatenation, in this exact
 *     order, `[w1 (H×D, row-major), b1 (H), w2 (3×H, row-major), b2 (3)]`
 *     — the same order `readoutParameterCount` sums and `ReadoutWeights`
 *     (`src/lib/connectome/readout.ts`) declares its fields in. This is
 *     "the published candidate: the final CEM mean"
 *     (`03-cem-training.md`), not `theta_best.npy` (best-ever candidate;
 *     not read by this evaluator).
 *   - `env.json` (optional): torch/CUDA/device provenance. When present for
 *     the shipped replica (trainerSeed 101), it is copied into
 *     `trained-readout-v1.manifest.json`'s `env` map, keyed by arm; a
 *     missing `env.json` for a shipped replica is recorded as a report
 *     warning, not an error (see `runEvaluate`'s shipped-artifact section).
 *   - `generations.csv` (optional): not read by this evaluator.
 */
export interface RunConfig {
  readonly arm: ArmName;
  /** Replica identity: one of 101, 202, 303 per WP3's default trainer-seed set. */
  readonly trainerSeed: number;
  readonly D: number;
  readonly H: number;
  readonly parameterCount: number;
  readonly substeps: number;
  /** Optional: the `export-arms.ts` bundle sha256 this run was trained against. */
  readonly armBundleSha256?: string;
}

export interface LoadedRun {
  readonly dir: string;
  readonly config: RunConfig;
  readonly weights: ReadoutWeights;
  readonly weightsSha256: string;
  readonly env: unknown | null;
}

const sha256Hex = (data: Uint8Array | string): string => createHash('sha256').update(data).digest('hex');

const POSITIVE_INT_RUN_CONFIG_FIELDS = ['trainerSeed', 'D', 'H', 'parameterCount', 'substeps'] as const;

/** Read, validate, and decode one run directory per the contract above. */
export const readRunDir = (dir: string): LoadedRun => {
  const configPath = resolve(dir, 'config.json');
  const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;

  if (typeof config.arm !== 'string' || !ARM_NAMES.includes(config.arm as ArmName)) {
    throw new Error(`evaluate: ${configPath}'s "arm" must be one of ${ARM_NAMES.join(', ')}, got ${JSON.stringify(config.arm)}`);
  }
  for (const field of POSITIVE_INT_RUN_CONFIG_FIELDS) {
    const value = config[field];
    if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
      throw new Error(`evaluate: ${configPath}'s "${field}" must be a positive integer, got ${JSON.stringify(value)}`);
    }
  }
  if (config.armBundleSha256 !== undefined && typeof config.armBundleSha256 !== 'string') {
    throw new Error(`evaluate: ${configPath}'s "armBundleSha256" must be a string when present`);
  }
  const runConfig = config as unknown as RunConfig;
  const { D, H, parameterCount } = runConfig;
  const expectedLength = readoutParameterCount(D, H);
  if (parameterCount !== expectedLength) {
    throw new Error(
      `evaluate: ${configPath}'s parameterCount ${parameterCount} does not equal ` +
        `readoutParameterCount(D=${D}, H=${H}) = ${expectedLength}`
    );
  }

  const thetaPath = resolve(dir, 'theta_final.npy');
  const theta = readNpyFloat32Array(thetaPath);
  if (theta.length !== parameterCount) {
    throw new Error(
      `evaluate: ${thetaPath} has ${theta.length} values, expected ${parameterCount} ` +
        `(config.json's parameterCount) for run "${dir}"`
    );
  }
  const weightsSha256 = sha256Hex(Buffer.from(theta.buffer, theta.byteOffset, theta.byteLength));

  let cursor = 0;
  const take = (count: number): Float32Array => {
    const slice = Float32Array.from(theta.subarray(cursor, cursor + count));
    cursor += count;
    return slice;
  };
  const weights: ReadoutWeights = {
    inputSize: D,
    hiddenSize: H,
    w1: take(H * D),
    b1: take(H),
    w2: take(3 * H),
    b2: take(3)
  };

  const envPath = resolve(dir, 'env.json');
  const env = existsSync(envPath) ? (JSON.parse(readFileSync(envPath, 'utf8')) as unknown) : null;

  return { dir, config: runConfig, weights, weightsSha256, env };
};
