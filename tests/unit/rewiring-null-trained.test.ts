import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { validateRewiringNullTrained } from '../../src/lib/experiment/rewiringNull';

/**
 * WP1 of `.agents/plans/findings-tour`: unit coverage for
 * `validateRewiringNullTrained`, the new shape check for
 * `RewiringNullArtifact.trained` (deliberately typed `unknown` at load
 * time — see that field's own doc comment) against the real shape
 * `scripts/null/null-report-trained.ts#buildTrainedSection` produces,
 * confirmed against the shipped `public/data/rewiring-null-v1.json`'s own
 * `trained` object.
 */

const here = dirname(fileURLToPath(import.meta.url));
const publicDataDir = resolve(here, '../../public/data');
const realArtifact = JSON.parse(readFileSync(resolve(publicDataDir, 'rewiring-null-v1.json'), 'utf-8')) as {
  trained: unknown;
};

describe('validateRewiringNullTrained', () => {
  it('accepts the real shipped trained section (rewiring-null-v1.json)', () => {
    const result = validateRewiringNullTrained(realArtifact.trained);
    expect(result).toBeDefined();
    expect(result?.replicaSeed).toBe(101);
    expect(result?.bioPercentile).toBe(0);
    expect(result?.rewiredCount).toBe(20);
    expect(result?.bioReplicaPercentiles).toEqual([
      { trainerSeed: 101, percentile: 0 },
      { trainerSeed: 202, percentile: 0.4 },
      { trainerSeed: 303, percentile: 0 }
    ]);
  });

  it('rejects undefined/null/non-object values', () => {
    expect(validateRewiringNullTrained(undefined)).toBeUndefined();
    expect(validateRewiringNullTrained(null)).toBeUndefined();
    expect(validateRewiringNullTrained('nope')).toBeUndefined();
    expect(validateRewiringNullTrained(42)).toBeUndefined();
  });

  it('rejects a section missing replicaSeed', () => {
    const { replicaSeed: _replicaSeed, ...rest } = realArtifact.trained as Record<string, unknown>;
    expect(validateRewiringNullTrained(rest)).toBeUndefined();
  });

  it('rejects a bioPercentile outside [0, 1]', () => {
    const malformed = { ...(realArtifact.trained as Record<string, unknown>), bioPercentile: 1.5 };
    expect(validateRewiringNullTrained(malformed)).toBeUndefined();
  });

  it('rejects an empty rewired array', () => {
    const malformed = { ...(realArtifact.trained as Record<string, unknown>), rewired: [] };
    expect(validateRewiringNullTrained(malformed)).toBeUndefined();
  });

  it('rejects an empty bioReplicaPercentiles array', () => {
    const malformed = { ...(realArtifact.trained as Record<string, unknown>), bioReplicaPercentiles: [] };
    expect(validateRewiringNullTrained(malformed)).toBeUndefined();
  });

  it('rejects a bioReplicaPercentiles entry with a percentile outside [0, 1]', () => {
    const malformed = {
      ...(realArtifact.trained as Record<string, unknown>),
      bioReplicaPercentiles: [{ trainerSeed: 101, percentile: -0.1 }]
    };
    expect(validateRewiringNullTrained(malformed)).toBeUndefined();
  });

  it('rejects a section whose headline bioPercentile disagrees with the matching per-seed entry', () => {
    const malformed = {
      ...(realArtifact.trained as Record<string, unknown>),
      bioPercentile: 0.9
    };
    expect(validateRewiringNullTrained(malformed)).toBeUndefined();
  });

  it('rejects a section with no per-seed entry at all for the headline replicaSeed', () => {
    const trained = realArtifact.trained as Record<string, unknown>;
    const malformed = {
      ...trained,
      bioReplicaPercentiles: (trained.bioReplicaPercentiles as Array<{ trainerSeed: number }>).filter(
        (entry) => entry.trainerSeed !== 101
      )
    };
    expect(validateRewiringNullTrained(malformed)).toBeUndefined();
  });

  it('is permissive of extra/unrecognized fields (does not require the full TrainedSection shape)', () => {
    const trained = realArtifact.trained as Record<string, unknown>;
    const minimal = {
      replicaSeed: trained.replicaSeed,
      bioPercentile: trained.bioPercentile,
      rewired: trained.rewired,
      bioReplicaPercentiles: trained.bioReplicaPercentiles
    };
    expect(validateRewiringNullTrained(minimal)).toBeDefined();
  });
});
