import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { readoutParameterCount, type ReadoutWeights } from '../../src/lib/connectome/readout';
import { ARM_NAMES, type ArmName } from './arms';
import { readNpyFloat32Array } from './npy';
import { readoutFromFlat } from '../../src/lib/connectome/readout-serialization';

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
  /**
   * CEM hyperparameters and seed policy, as written by `flyarena-train`'s
   * `_build_run_config` (`training/src/flyarena_training/cli.py`). All
   * optional here because older/tiny test-fixture run dirs
   * (`tests/fixtures/trained-readout-run.ts`) predate these fields;
   * `evaluate.ts`'s manifest `training` block copies them from a real run's
   * `config.json` verbatim when present.
   */
  readonly population?: number;
  readonly elites?: number;
  readonly generations?: number;
  readonly alpha?: number;
  readonly stdFloor?: number;
  readonly initStd?: number;
  readonly trainingSeedsPerGeneration?: number;
  readonly trainingSeedRange?: readonly [number, number];
  /**
   * Records the CLI's deliberate deviation from
   * `.agents/plans/trained-readout/03-cem-training.md`'s literal
   * `trainer_seed + generation` training-seed formula (aliases replicas on
   * shared generation values) in favor of
   * `numpy.random.default_rng([trainerSeed, generation])`
   * (`training/src/flyarena_training/seeds.py`), so the manifest can state
   * it rather than silently assume the plan's formula.
   */
  readonly trainingSeedRng?: string;
  readonly validationSeedRange?: readonly [number, number];
  readonly heldOutSeedRange?: readonly [number, number];
  readonly bestValidationFitness?: number;
  /**
   * `.agents/plans/task-generality/01-task-plumbing.md`'s WP1: the arena
   * task id and fingerprint (`src/lib/arena/tasks.ts`) `flyarena-train`'s
   * `--arena-task` flag resolved this run against, always recorded together
   * by `cli.py`'s `_build_run_config`. Optional here only for older/tiny
   * fixture run directories that predate arena tasks
   * (`tests/fixtures/trained-readout-run.ts`); `null-trained-worker.ts`'s
   * `assertRunMatchesExpectedIdentity` requires `arenaTaskFingerprint`
   * present and matching for any run it scores.
   */
  readonly arenaTask?: string;
  readonly arenaTaskFingerprint?: string;
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

  const weights = readoutFromFlat(theta, D, H);

  const envPath = resolve(dir, 'env.json');
  const env = existsSync(envPath) ? (JSON.parse(readFileSync(envPath, 'utf8')) as unknown) : null;

  return { dir, config: runConfig, weights, weightsSha256, env };
};

// ---------------------------------------------------------------------------
// CEM-config reconciliation (manifest `training` block)
// ---------------------------------------------------------------------------

/**
 * CEM hyperparameters + training-seed RNG policy fields carried from a
 * `RunConfig` into the manifest's `training` block
 * (`05-production-run.md` step 4: every arm's replica-0 run must share an
 * identical config except `--arm`/`--replica-seed`). Module-level (not
 * declared inside `deriveTrainingBlock`) since the field list is fixed and
 * doesn't depend on any call's arguments — no reason to re-create it per call.
 */
export const CEM_CONFIG_FIELDS = [
  'population',
  'elites',
  'generations',
  'alpha',
  'stdFloor',
  'initStd',
  'trainingSeedsPerGeneration',
  'trainingSeedRange',
  'trainingSeedRng',
  'validationSeedRange',
  'heldOutSeedRange'
] as const;

/**
 * The subset of `CEM_CONFIG_FIELDS` that `public/data/trained-readout-v1.manifest.json`'s
 * `training` block actually carries (it has no `trainingSeedRange`/
 * `trainingSeedRng`/`validationSeedRange`/`heldOutSeedRange` -- those are
 * per-run-directory provenance, not per-study manifest fields). Derived
 * (not a second hand-typed literal array) so a future rename of any
 * `CEM_CONFIG_FIELDS` entry can never silently leave this list stale (a
 * thermo-maintainability review suggestion). Consumed by
 * `scripts/null/train-sample.sh`'s own preflight (via its own Python
 * reimplementation, which independently reads the same manifest keys) and
 * `scripts/null/null-trained-evaluate-graph-list.ts`'s
 * `assertConfigsMatchManifest`.
 */
export const MANIFEST_TRACKED_FIELDS = CEM_CONFIG_FIELDS.filter(
  (field) => !field.endsWith('Range') && field !== 'trainingSeedRng'
);

/** A candidate whose every `CEM_CONFIG_FIELDS` value is `undefined` — an arm with no recorded CEM hyperparameters at all. */
export const isEmptyCemConfig = (candidate: Readonly<Record<string, unknown>>): boolean =>
  Object.values(candidate).every((value) => value === undefined);

export interface TrainingBlockResult {
  readonly trainingBlock: Record<string, unknown> | null;
  readonly warnings: readonly string[];
}

/**
 * Reconciles each arm's shipped-replica (trainerSeed 101) CEM config into
 * one manifest `training` block. Picks the first `ARM_NAMES`-order arm that
 * actually HAS a recorded (non-empty) config as the baseline — not merely
 * the first arm seen — so an arm with an older/tiny run dir lacking these
 * fields (all `undefined`) is skipped when picking the baseline rather than
 * adopted as one, and never silently discards a later arm's real, present
 * config (a single forward pass over `ARM_NAMES` could only compare each arm
 * against a baseline established by an *earlier* arm, so an early arm with
 * no recorded config would never get flagged even when a later arm does have
 * one, and would wrongly become the — empty — baseline). Any other arm whose
 * candidate disagrees with the baseline produces a warning rather than
 * silently overwriting or being dropped.
 *
 * Extracted from `evaluate.ts`'s `runEvaluate` (thermo-maintainability
 * review: this was previously ~62 lines inlined in an already ~540-line
 * function) as a pure function of `shippedConfig` so it can be unit-tested
 * directly against constructed maps, without spinning up full run
 * directories and calling the entire evaluation pipeline.
 */
export const deriveTrainingBlock = (
  shippedConfig: Readonly<Partial<Record<ArmName, RunConfig>>>
): TrainingBlockResult => {
  const warnings: string[] = [];

  const cemCandidates: Partial<Record<ArmName, Record<string, unknown>>> = {};
  for (const arm of ARM_NAMES) {
    const config = shippedConfig[arm];
    if (!config) continue;
    const candidate: Record<string, unknown> = {};
    for (const field of CEM_CONFIG_FIELDS) candidate[field] = config[field];
    cemCandidates[arm] = candidate;
  }
  const armsWithShippedConfig = ARM_NAMES.filter((arm) => cemCandidates[arm] !== undefined);
  const trainingBlockSourceArm = armsWithShippedConfig.find((arm) => !isEmptyCemConfig(cemCandidates[arm]!)) ?? null;
  const trainingBlock: Record<string, unknown> | null =
    trainingBlockSourceArm !== null ? cemCandidates[trainingBlockSourceArm]! : null;

  for (const arm of armsWithShippedConfig) {
    const candidate = cemCandidates[arm]!;
    if (isEmptyCemConfig(candidate)) {
      if (trainingBlockSourceArm !== null) {
        warnings.push(
          `arm "${arm}" replica 0's config.json has no recorded CEM hyperparameters, while arm ` +
            `"${trainingBlockSourceArm}" does; manifest's training block cannot include arm "${arm}"'s values.`
        );
      }
      continue;
    }
    if (arm === trainingBlockSourceArm) continue; // it IS the baseline; nothing to compare it against
    if (JSON.stringify(candidate) !== JSON.stringify(trainingBlock)) {
      warnings.push(
        `arm "${arm}" replica 0's CEM config/training-seed policy differs from arm ` +
          `"${trainingBlockSourceArm}"'s (run must use identical config except --arm/--replica-seed); ` +
          `manifest's training block reports arm "${trainingBlockSourceArm}"'s, not this one.`
      );
    }
  }

  return { trainingBlock, warnings };
};
