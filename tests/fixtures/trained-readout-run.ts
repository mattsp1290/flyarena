import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { readoutParameterCount } from '../../src/lib/connectome/readout';
import { mulberry32 } from '../../src/lib/random/mulberry32';
import type { ArmName } from '../../scripts/training/export-arms';
import type { RunConfig } from '../../scripts/training/evaluate';
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
}

export const writeTinyRunDir = (options: Readonly<TinyRunOptions>): void => {
  const { dir, arm, trainerSeed, D, H, substeps, weightSeed, includeEnv } = options;
  mkdirSync(dir, { recursive: true });

  const parameterCount = readoutParameterCount(D, H);
  const random = mulberry32(weightSeed);
  // Small magnitude weights: readoutForward's tanh/sigmoid stay well away
  // from saturation, so silenced-vs-trained/authored comparisons in tests
  // are numerically meaningful rather than all pinned to +-1.
  const theta = Float32Array.from({ length: parameterCount }, () => (random() * 2 - 1) * 0.3);
  writeNpyFloat32Array(resolve(dir, 'theta_final.npy'), theta);

  const config: RunConfig = { arm, trainerSeed, D, H, parameterCount, substeps };
  writeFileSync(resolve(dir, 'config.json'), JSON.stringify(config));

  if (includeEnv) {
    writeFileSync(
      resolve(dir, 'env.json'),
      JSON.stringify({ device: 'cpu', torch: 'n/a (test fixture)', note: 'synthetic run dir, not a real training run' })
    );
  }
};
