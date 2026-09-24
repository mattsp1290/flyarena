import { describe, expect, it } from 'vitest';

import type { ArmName } from '../../scripts/training/arms';
import { deriveTrainingBlock, isEmptyCemConfig, type RunConfig } from '../../scripts/training/run-dir';

/**
 * `deriveTrainingBlock` (extracted from `evaluate.ts`'s `runEvaluate` —
 * thermo-maintainability review finding 1: this CEM-config reconciliation
 * logic was previously ~62 lines inlined in an already ~540-line function).
 * Exercised directly against constructed `shippedConfig` maps, mirroring how
 * `report.ts`'s `nearInputIndependentPolicyArms`/`structurallyZeroReadoutInputArms`
 * are tested, without spinning up full run directories or calling the full
 * `runEvaluate` pipeline.
 */

const baseConfig = (overrides: Partial<RunConfig> = {}): RunConfig => ({
  arm: 'biological',
  trainerSeed: 101,
  D: 6,
  H: 4,
  parameterCount: 100,
  substeps: 4,
  ...overrides
});

describe('isEmptyCemConfig', () => {
  it('is true when every field is undefined', () => {
    expect(isEmptyCemConfig({ population: undefined, elites: undefined })).toBe(true);
  });

  it('is false when at least one field is set', () => {
    expect(isEmptyCemConfig({ population: 128, elites: undefined })).toBe(false);
  });
});

describe('deriveTrainingBlock', () => {
  const cemFields = { population: 128, elites: 32, generations: 150 };

  it('returns null/no warnings when no arm has a shipped config', () => {
    const { trainingBlock, warnings } = deriveTrainingBlock({});
    expect(trainingBlock).toBeNull();
    expect(warnings).toEqual([]);
  });

  it('returns the one shipped config unchanged when only one arm has it', () => {
    const shippedConfig: Partial<Record<ArmName, RunConfig>> = {
      biological: baseConfig(cemFields)
    };
    const { trainingBlock, warnings } = deriveTrainingBlock(shippedConfig);
    expect(trainingBlock).toMatchObject(cemFields);
    expect(warnings).toEqual([]);
  });

  it("preserves a later arm's real config when an earlier arm (ARM_NAMES order) has none recorded", () => {
    // 'biological' sorts before 'rewired' in ARM_NAMES: this is exactly the
    // order the extracted fix targets. A naive single forward pass would
    // adopt 'biological' (all fields undefined) as the baseline and then
    // treat 'rewired's real config as "differing", silently dropping it.
    const shippedConfig: Partial<Record<ArmName, RunConfig>> = {
      biological: baseConfig({ arm: 'biological' }), // no cemFields at all
      rewired: baseConfig({ arm: 'rewired', ...cemFields })
    };
    const { trainingBlock, warnings } = deriveTrainingBlock(shippedConfig);
    expect(trainingBlock).toMatchObject(cemFields);
    expect(warnings.some((w) => w.includes('biological') && w.includes('no recorded CEM'))).toBe(true);
  });

  it('warns when an arm’s config disagrees with the baseline, but still reports the baseline', () => {
    const shippedConfig: Partial<Record<ArmName, RunConfig>> = {
      biological: baseConfig({ arm: 'biological', ...cemFields }),
      rewired: baseConfig({ arm: 'rewired', population: 64, elites: 32, generations: 150 })
    };
    const { trainingBlock, warnings } = deriveTrainingBlock(shippedConfig);
    expect(trainingBlock).toMatchObject(cemFields);
    expect(warnings.some((w) => w.includes('rewired') && w.includes('differs from arm "biological"'))).toBe(true);
  });

  it('does not warn when every arm with a config agrees with the baseline', () => {
    const shippedConfig: Partial<Record<ArmName, RunConfig>> = {
      biological: baseConfig({ arm: 'biological', ...cemFields }),
      rewired: baseConfig({ arm: 'rewired', ...cemFields }),
      disconnected: baseConfig({ arm: 'disconnected', ...cemFields })
    };
    const { warnings } = deriveTrainingBlock(shippedConfig);
    expect(warnings).toEqual([]);
  });
});
