import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { resolveArenaTask } from '../../src/lib/arena/tasks';
import { readoutParameterCount } from '../../src/lib/connectome/readout';
import { mulberry32 } from '../../src/lib/random/mulberry32';
import type { ArmName } from '../../scripts/training/arms';
import type { RunConfig } from '../../scripts/training/run-dir';
import { writeNpyFloat32Array } from '../../scripts/training/npy';

/**
 * Builds a tiny synthetic WP3 run directory satisfying the run-dir contract
 * `scripts/training/evaluate.ts` documents (WP3, `training/src/flyarena_training/cem.py`,
 * is not built yet — see that file's header comment). Used only by this
 * WP's own tests (`tests/unit/*.test.ts`): writes `config.json` and
 * `theta_final.npy` (and, optionally, a placeholder `env.json`) to `dir`,
 * deterministically for a given `weightSeed`.
 */
export interface TinyRunOptions {
  readonly dir: string;
  readonly arm: ArmName;
  readonly trainerSeed: number;
  readonly D: number;
  readonly H: number;
  readonly substeps: number;
  readonly weightSeed: number;
  readonly includeEnv?: boolean;
  /** Optional `RunConfig.armBundleSha256`, for tests exercising the arm-bundle-sha256 compatibility check. */
  readonly armBundleSha256?: string;
  /** Arena task id this run is recorded as trained under (default `'default'`, matching every existing test's implicit assumption). */
  readonly arenaTask?: string;
  /** Omits `arenaTask`/`arenaTaskFingerprint` entirely, for a test of the "run directory predates arena tasks" fail-closed path. */
  readonly omitArenaTaskFingerprint?: boolean;
  /**
   * Optional CEM-config fields (`RunConfig`'s optional properties), for
   * tests that exercise `evaluate.ts`'s manifest `training` block (the
   * real `flyarena-train` CLI always writes these; older/tiny fixture runs
   * omit them by default).
   */
  readonly cemConfig?: Pick<
    RunConfig,
    | 'population'
    | 'elites'
    | 'generations'
    | 'alpha'
    | 'stdFloor'
    | 'initStd'
    | 'trainingSeedsPerGeneration'
    | 'trainingSeedRange'
    | 'trainingSeedRng'
    | 'validationSeedRange'
    | 'heldOutSeedRange'
  >;
}

export const writeTinyRunDir = (options: Readonly<TinyRunOptions>): void => {
  const { dir, arm, trainerSeed, D, H, substeps, weightSeed, includeEnv, armBundleSha256, cemConfig } = options;
  const omitArenaTaskFingerprint = options.omitArenaTaskFingerprint ?? false;
  const resolvedArenaTask = resolveArenaTask(options.arenaTask);
  mkdirSync(dir, { recursive: true });

  const parameterCount = readoutParameterCount(D, H);
  const random = mulberry32(weightSeed);
  // Small magnitude weights: readoutForward's tanh/sigmoid stay well away
  // from saturation, so silenced-vs-trained/authored comparisons in tests
  // are numerically meaningful rather than all pinned to +-1.
  const theta = Float32Array.from({ length: parameterCount }, () => (random() * 2 - 1) * 0.3);
  writeNpyFloat32Array(resolve(dir, 'theta_final.npy'), theta);

  const config: RunConfig = {
    arm,
    trainerSeed,
    D,
    H,
    parameterCount,
    substeps,
    ...(armBundleSha256 !== undefined ? { armBundleSha256 } : {}),
    ...(omitArenaTaskFingerprint
      ? {}
      : { arenaTask: resolvedArenaTask.id, arenaTaskFingerprint: resolvedArenaTask.fingerprint }),
    ...cemConfig
  };
  writeFileSync(resolve(dir, 'config.json'), JSON.stringify(config));

  if (includeEnv) {
    writeFileSync(
      resolve(dir, 'env.json'),
      JSON.stringify({ device: 'cpu', torch: 'n/a (test fixture)', note: 'synthetic run dir, not a real training run' })
    );
  }
};
