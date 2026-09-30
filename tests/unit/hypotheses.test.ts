import { describe, expect, it } from 'vitest';

import { evaluateH1, evaluateH2, evaluateH3, newlyConnectedThrustDIndices } from '../../scripts/attribution/hypotheses';
import { OUTPUT_POPULATION } from '../../src/lib/arena/actions';

/**
 * `.agents/plans/readout-attribution/02-analyses.md`'s WP2 test
 * requirement: "The H2 and H3 equivalence rules are tested for supported,
 * not supported, and inconclusive." Plus H1's regime gate / cluster-vs-
 * neuron bootstrap disagreement rule, and `newlyConnectedThrustDIndices`'s
 * thrust-population filter.
 */

const linkageEntry = (id: string, overrides: Partial<{ rhoThrust: number; ciCluster: [number, number]; ciNeuron: [number, number] }> = {}) => ({
  id,
  rhoThrust: overrides.rhoThrust ?? 0.5,
  rhoYaw: 0,
  ciCluster: overrides.ciCluster ?? ([0.1, 0.8] as [number, number]),
  ciNeuron: overrides.ciNeuron ?? ([0.1, 0.8] as [number, number]),
  clusterCount: 28
});
const regimeEntry = (id: string, valid: boolean) => ({ id, valid });

describe('evaluateH1', () => {
  const seeds = ['biological-seed101', 'biological-seed202', 'biological-seed303'];

  it('is supported when all 3 biological seeds pass the regime gate, meet the rho threshold, and the two bootstraps agree', () => {
    const linkageById = new Map(seeds.map((id) => [id, linkageEntry(id)]));
    const regimeById = new Map(seeds.map((id) => [id, regimeEntry(id, true)]));
    const result = evaluateH1(linkageById, regimeById);
    expect(result.outcome).toBe('supported');
  });

  it('is inconclusive (regime-invalid) when any seed fails the regime gate', () => {
    const linkageById = new Map(seeds.map((id) => [id, linkageEntry(id)]));
    const regimeById = new Map(seeds.map((id, i) => [id, regimeEntry(id, i !== 1)]));
    const result = evaluateH1(linkageById, regimeById);
    expect(result.outcome).toBe('inconclusive');
    expect(result.reason).toBe('regime-invalid');
  });

  it('is inconclusive when the cluster and neuron bootstraps disagree on excluding 0', () => {
    const linkageById = new Map(
      seeds.map((id, i) => [
        id,
        linkageEntry(id, i === 0 ? { ciCluster: [0.1, 0.8], ciNeuron: [-0.1, 0.8] } : {})
      ])
    );
    const regimeById = new Map(seeds.map((id) => [id, regimeEntry(id, true)]));
    const result = evaluateH1(linkageById, regimeById);
    expect(result.outcome).toBe('inconclusive');
    expect(result.reason).toBe('cluster-neuron-bootstrap-disagreement');
  });

  it('is inconclusive when rho or the cluster CI fails the threshold in at least one seed', () => {
    const linkageById = new Map(
      seeds.map((id, i) => [id, linkageEntry(id, i === 2 ? { rhoThrust: 0.1 } : {})])
    );
    const regimeById = new Map(seeds.map((id) => [id, regimeEntry(id, true)]));
    const result = evaluateH1(linkageById, regimeById);
    expect(result.outcome).toBe('inconclusive');
    expect(result.reason).toBe('threshold-not-met-in-all-seeds');
  });
});

describe('evaluateH2', () => {
  const build = (ratios: { bio: number; rewired: number }[]) => {
    const pairs: readonly (readonly [string, number])[] = [
      ['biological-seed101', ratios[0].bio],
      ['rewired-seed0-seed101', ratios[0].rewired],
      ['biological-seed202', ratios[1].bio],
      ['rewired-seed0-seed202', ratios[1].rewired],
      ['biological-seed303', ratios[2].bio],
      ['rewired-seed0-seed303', ratios[2].rewired]
    ];
    return new Map(pairs.map(([id, ratio]) => [id, { id, defined: true, ratio }]));
  };

  it('uses a t-based CI, not a percentile bootstrap: [0.05, -0.02, 0.08] is inconclusive, not "supported"', () => {
    // A dual-review finding: a percentile bootstrap of the mean at n = 3
    // can never extend past [min, max] of the sample and made this exact
    // case "supported" (bootstrap 90% CI ~ [0.0033, 0.0700]); the honest
    // t-based 90% CI for these 3 numbers is ~[-0.0498, 0.1232], which
    // straddles +0.10.
    const independenceById = build([
      { bio: 0.55, rewired: 0.5 }, // difference 0.05
      { bio: 0.48, rewired: 0.5 }, // difference -0.02
      { bio: 0.58, rewired: 0.5 } // difference 0.08
    ]);
    const result = evaluateH2(independenceById);
    expect(result.outcome).toBe('inconclusive');
  });

  it('is supported when the paired-difference 90% CI lies entirely inside +-0.10', () => {
    const independenceById = build([
      { bio: 0.5, rewired: 0.52 },
      { bio: 0.48, rewired: 0.5 },
      { bio: 0.51, rewired: 0.49 }
    ]);
    const result = evaluateH2(independenceById);
    expect(result.outcome).toBe('supported');
  });

  it('is not supported when the CI excludes 0 and lies entirely outside +-0.10', () => {
    const independenceById = build([
      { bio: 0.9, rewired: 0.2 },
      { bio: 0.85, rewired: 0.18 },
      { bio: 0.95, rewired: 0.22 }
    ]);
    const result = evaluateH2(independenceById);
    expect(result.outcome).toBe('not-supported');
  });

  it('is inconclusive when the CI straddles the +-0.10 boundary', () => {
    const independenceById = build([
      { bio: 0.6, rewired: 0.45 },
      { bio: 0.5, rewired: 0.5 },
      { bio: 0.55, rewired: 0.5 }
    ]);
    const result = evaluateH2(independenceById);
    expect(result.outcome).toBe('inconclusive');
  });

  it('is inconclusive when any pair has an undefined independence share', () => {
    const independenceById = new Map([
      ['biological-seed101', { id: 'biological-seed101', defined: false, ratio: null }],
      ['rewired-seed0-seed101', { id: 'rewired-seed0-seed101', defined: true, ratio: 0.5 }],
      ['biological-seed202', { id: 'biological-seed202', defined: true, ratio: 0.5 }],
      ['rewired-seed0-seed202', { id: 'rewired-seed0-seed202', defined: true, ratio: 0.5 }],
      ['biological-seed303', { id: 'biological-seed303', defined: true, ratio: 0.5 }],
      ['rewired-seed0-seed303', { id: 'rewired-seed0-seed303', defined: true, ratio: 0.5 }]
    ]);
    const result = evaluateH2(independenceById);
    expect(result.outcome).toBe('inconclusive');
    expect(result.reason).toBe('undefined-independence-share');
  });
});

describe('evaluateH3', () => {
  const saliencyEntry = (id: string, thrust: readonly number[]) => [id, { id, thrust, yaw: thrust }] as const;

  it('is supported when the mean paired ratio\'s 90% upper bound is <= 1.25', () => {
    const saliencyById = new Map([
      saliencyEntry('P-seed101', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('biological-seed101', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('P-seed202', [0.11, 0.11, 0.11, 0.11]),
      saliencyEntry('biological-seed202', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('P-seed303', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('biological-seed303', [0.1, 0.1, 0.1, 0.1])
    ]);
    const result = evaluateH3(saliencyById, [0, 1]);
    expect(result.outcome).toBe('supported');
  });

  it('is not supported when the ratio\'s 90% lower bound is > 1.25', () => {
    const saliencyById = new Map([
      saliencyEntry('P-seed101', [0.3, 0.3, 0.3, 0.3]),
      saliencyEntry('biological-seed101', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('P-seed202', [0.32, 0.32, 0.32, 0.32]),
      saliencyEntry('biological-seed202', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('P-seed303', [0.29, 0.29, 0.29, 0.29]),
      saliencyEntry('biological-seed303', [0.1, 0.1, 0.1, 0.1])
    ]);
    const result = evaluateH3(saliencyById, [0, 1]);
    expect(result.outcome).toBe('not-supported');
  });

  it('is inconclusive when the CI straddles 1.25', () => {
    const saliencyById = new Map([
      saliencyEntry('P-seed101', [0.15, 0.15, 0.15, 0.15]),
      saliencyEntry('biological-seed101', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('P-seed202', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('biological-seed202', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('P-seed303', [0.2, 0.2, 0.2, 0.2]),
      saliencyEntry('biological-seed303', [0.1, 0.1, 0.1, 0.1])
    ]);
    const result = evaluateH3(saliencyById, [0, 1]);
    expect(result.outcome).toBe('inconclusive');
  });

  it('is inconclusive with no newly connected thrust neurons', () => {
    const saliencyById = new Map([saliencyEntry('P-seed101', [0.1]), saliencyEntry('biological-seed101', [0.1])]);
    const result = evaluateH3(saliencyById, []);
    expect(result.outcome).toBe('inconclusive');
    expect(result.reason).toBe('no-newly-connected-thrust-neurons');
  });

  it('is inconclusive (not "not-supported") when a biological mean saliency is exactly 0 (an undefined ratio)', () => {
    // A dual-review finding: the previous implementation turned a 0/0 ratio
    // into `+Infinity`, which made `ci[0] > 1.25` true and reported
    // "not-supported" from an undefined ratio.
    const saliencyById = new Map([
      saliencyEntry('P-seed101', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('biological-seed101', [0, 0, 0, 0]),
      saliencyEntry('P-seed202', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('biological-seed202', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('P-seed303', [0.1, 0.1, 0.1, 0.1]),
      saliencyEntry('biological-seed303', [0.1, 0.1, 0.1, 0.1])
    ]);
    const result = evaluateH3(saliencyById, [0, 1]);
    expect(result.outcome).toBe('inconclusive');
    expect(result.reason).toBe('undefined-saliency-ratio');
  });
});

describe('newlyConnectedThrustDIndices', () => {
  it('keeps only added-edge targets assigned to the thrust population, mapped to D-space', () => {
    // Raw neurons 5 (thrust), 6 (yaw), 7 (thrust, but deliberately left out of
    // `indices` below -> excluded, exercising the `dByNeuron.get(edge.post)
    // === undefined` branch even though `outputPopulationIndex[7]` itself is
    // thrust).
    const outputPopulationIndex = Int32Array.from([-1, -1, -1, -1, -1, OUTPUT_POPULATION.thrust, OUTPUT_POPULATION.yaw, OUTPUT_POPULATION.thrust]);
    const indices = Int32Array.from([5, 6]); // D-space order for this hand-built fixture -- 7 is not in it
    const addedEdges = [
      { pre: 1, post: 5 }, // thrust, output-assigned -> d=0
      { pre: 2, post: 6 }, // yaw -> excluded
      { pre: 3, post: 9 } // not in indices at all -> excluded
    ];
    const result = newlyConnectedThrustDIndices(addedEdges, outputPopulationIndex, indices);
    expect(result).toEqual([0]);
  });

  it('deduplicates repeated targets', () => {
    const outputPopulationIndex = Int32Array.from([OUTPUT_POPULATION.thrust, OUTPUT_POPULATION.thrust]);
    const indices = Int32Array.from([0, 1]);
    const addedEdges = [
      { pre: 9, post: 0 },
      { pre: 10, post: 0 },
      { pre: 11, post: 1 }
    ];
    const result = newlyConnectedThrustDIndices(addedEdges, outputPopulationIndex, indices);
    expect(result).toEqual([0, 1]);
  });
});
