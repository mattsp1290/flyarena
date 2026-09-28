import { mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { OUTPUT_POPULATION } from '../../src/lib/arena/actions';
import { outputNeuronIndices } from '../../src/lib/connectome/readout';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { conditionRng } from '../training/stats';
import {
  DEFAULT_ARCHIVE_PATH,
  DEFAULT_ARCHIVED_INTERVENTION_INDEX_PATH,
  DEFAULT_MANIFEST_PATH,
  defaultResolveGraphConfig,
  graphForEntry,
  loadArchive,
  parsePathFlags,
  resolvePathFlag
} from './shared';

/**
 * `.agents/plans/readout-attribution/02-analyses.md`'s WP2: evaluates
 * H1-H3 mechanically against the predeclared rules in `00-overview.md`.
 * Reads every other WP2 analysis's output (`saliency.json`,
 * `independence.json`, `linkage.json`, `regime.json`) plus the archive and
 * `intervention-swaps-v1.json`; produces no new simulation.
 *
 * H1's rho/CI threshold (>= 0.3, cluster CI excluding 0, unanimous across
 * the 3 biological trainer seeds), H2's TOST-style equivalence bounds
 * (+/-0.10 on the paired independence-share difference), and H3's
 * redundancy bound (<= 1.25 on the mean paired saliency ratio) are exactly
 * `00-overview.md`'s "Predeclared hypotheses and outcome rules" text.
 *
 * No multiple-comparison correction is applied across H1-H3 (3
 * hypotheses); this is disclosed in the output (`hypothesisCount`), per the
 * plan's honesty requirement.
 */

export type HypothesisOutcome = 'supported' | 'not-supported' | 'inconclusive';

export interface HypothesisResult {
  readonly outcome: HypothesisOutcome;
  readonly reason?: string;
  readonly evidence: unknown;
}

// ---------------------------------------------------------------------------
// Shared bootstrap CI (90%) -- `training/stats.ts`'s `bootstrapCI` is fixed
// at 95%; H2/H3 both need a 90% CI (TOST-style), so this module has its own
// small helper rather than parameterizing that shared 95%-CI-everywhere
// convention used throughout the rest of this repo.
// ---------------------------------------------------------------------------

const bootstrapMeanCI90 = (values: readonly number[], resamples: number, rng: () => number): readonly [number, number] => {
  const n = values.length;
  const draws = new Array<number>(resamples);
  for (let r = 0; r < resamples; r += 1) {
    let sum = 0;
    for (let i = 0; i < n; i += 1) sum += values[Math.floor(rng() * n)];
    draws[r] = sum / n;
  }
  draws.sort((a, b) => a - b);
  const lowIndex = Math.floor(0.05 * resamples);
  const highIndex = Math.min(resamples - 1, Math.ceil(0.95 * resamples) - 1);
  return [draws[lowIndex], draws[highIndex]];
};

// ---------------------------------------------------------------------------
// Input shapes (as written by this WP's other CLIs)
// ---------------------------------------------------------------------------

interface SaliencyEntry {
  readonly id: string;
  readonly thrust: readonly number[];
  readonly yaw: readonly number[];
}
interface IndependenceEntry {
  readonly id: string;
  readonly defined: boolean;
  readonly ratio: number | null;
}
interface LinkageEntry {
  readonly id: string;
  readonly rhoThrust: number;
  readonly ciCluster: readonly [number, number];
  readonly ciNeuron: readonly [number, number];
  readonly clusterCount: number;
}
interface RegimeEntry {
  readonly id: string;
  readonly valid: boolean;
}
interface SwapEdge {
  readonly pre: number;
  readonly post: number;
}
interface InterventionSwapsArchive {
  readonly swaps: readonly { readonly id: string; readonly addedEdges: readonly SwapEdge[] }[];
}

const byId = <T extends { readonly id: string }>(entries: readonly T[]): Map<string, T> =>
  new Map(entries.map((e) => [e.id, e]));

const excludesZero = (ci: readonly [number, number]): boolean => ci[0] > 0 || ci[1] < 0;

// ---------------------------------------------------------------------------
// H1 -- routing-around (correlational)
// ---------------------------------------------------------------------------

const BIOLOGICAL_SEED_IDS = ['biological-seed101', 'biological-seed202', 'biological-seed303'];
const RHO_THRESHOLD = 0.3;

interface H1SeedEvidence {
  readonly id: string;
  readonly regimeValid: boolean;
  readonly rho: number;
  readonly ciCluster: readonly [number, number];
  readonly ciNeuron: readonly [number, number];
  readonly clusterExcludesZero: boolean;
  readonly neuronExcludesZero: boolean;
  readonly agree: boolean;
  readonly meetsThreshold: boolean;
}

export const evaluateH1 = (
  linkageById: Map<string, LinkageEntry>,
  regimeById: Map<string, RegimeEntry>
): HypothesisResult => {
  const perSeed: H1SeedEvidence[] = BIOLOGICAL_SEED_IDS.map((id) => {
    const linkage = linkageById.get(id);
    const regime = regimeById.get(id);
    if (!linkage || !regime) throw new Error(`hypotheses: H1 missing linkage/regime data for "${id}"`);
    const clusterExcludesZero = excludesZero(linkage.ciCluster);
    const neuronExcludesZero = excludesZero(linkage.ciNeuron);
    return {
      id,
      regimeValid: regime.valid,
      rho: linkage.rhoThrust,
      ciCluster: linkage.ciCluster,
      ciNeuron: linkage.ciNeuron,
      clusterExcludesZero,
      neuronExcludesZero,
      agree: clusterExcludesZero === neuronExcludesZero,
      meetsThreshold: linkage.rhoThrust >= RHO_THRESHOLD && clusterExcludesZero
    };
  });

  const anyRegimeInvalid = perSeed.some((s) => !s.regimeValid);
  const anyDisagreement = perSeed.some((s) => !s.agree);
  const allMeetThreshold = perSeed.every((s) => s.meetsThreshold);

  let outcome: HypothesisOutcome = 'inconclusive';
  let reason: string | undefined;
  if (anyRegimeInvalid) {
    reason = 'regime-invalid';
  } else if (anyDisagreement) {
    reason = 'cluster-neuron-bootstrap-disagreement';
  } else if (allMeetThreshold) {
    outcome = 'supported';
  } else {
    reason = 'threshold-not-met-in-all-seeds';
  }

  return { outcome, reason, evidence: { rhoThreshold: RHO_THRESHOLD, seeds: perSeed } };
};

// ---------------------------------------------------------------------------
// H2 -- constant-policy equivalence
// ---------------------------------------------------------------------------

const H2_SEED_PAIRS: readonly { readonly seed: number; readonly biological: string; readonly rewired: string }[] = [
  { seed: 101, biological: 'biological-seed101', rewired: 'rewired-seed0-seed101' },
  { seed: 202, biological: 'biological-seed202', rewired: 'rewired-seed0-seed202' },
  { seed: 303, biological: 'biological-seed303', rewired: 'rewired-seed0-seed303' }
];
const H2_EQUIVALENCE_BOUND = 0.1;

export const evaluateH2 = (
  independenceById: Map<string, IndependenceEntry>,
  resamples: number,
  bootstrapSeed: number
): HypothesisResult => {
  const pairs = H2_SEED_PAIRS.map(({ seed, biological, rewired }) => {
    const bio = independenceById.get(biological);
    const rew = independenceById.get(rewired);
    if (!bio || !rew) throw new Error(`hypotheses: H2 missing independence data for seed ${seed}`);
    return { seed, biological: bio, rewired: rew };
  });

  const undefinedPairs = pairs.filter((p) => !p.biological.defined || !p.rewired.defined);
  if (undefinedPairs.length > 0) {
    return {
      outcome: 'inconclusive',
      reason: 'undefined-independence-share',
      evidence: { pairs, undefinedSeeds: undefinedPairs.map((p) => p.seed) }
    };
  }

  const differences = pairs.map((p) => (p.biological.ratio as number) - (p.rewired.ratio as number));
  const rng = conditionRng(bootstrapSeed, 'H2');
  const ci = bootstrapMeanCI90(differences, resamples, rng);

  // "Supported": the whole CI lies inside (-0.10, 0.10). "Not supported":
  // the whole CI lies OUTSIDE [-0.10, 0.10] on one side (`ci[0] >
  // H2_EQUIVALENCE_BOUND` or `ci[1] < -H2_EQUIVALENCE_BOUND`) -- which
  // necessarily also excludes 0, so the plan's "excludes 0 and lies outside
  // +-0.10" is checked as one entirely-outside condition, not two
  // independent ones (a CI that merely POKES past +-0.10 on one side while
  // still straddling it is neither "inside" nor "entirely outside", and is
  // therefore `inconclusive` below, by falling through both branches).
  let outcome: HypothesisOutcome = 'inconclusive';
  if (ci[0] > -H2_EQUIVALENCE_BOUND && ci[1] < H2_EQUIVALENCE_BOUND) {
    outcome = 'supported';
  } else if (ci[0] > H2_EQUIVALENCE_BOUND || ci[1] < -H2_EQUIVALENCE_BOUND) {
    outcome = 'not-supported';
  }

  return {
    outcome,
    evidence: { equivalenceBound: H2_EQUIVALENCE_BOUND, differences, ci, n: pairs.length }
  };
};

// ---------------------------------------------------------------------------
// H3 -- P redundancy
// ---------------------------------------------------------------------------

const H3_SEEDS = [101, 202, 303];
const H3_RATIO_BOUND = 1.25;

export const evaluateH3 = (
  saliencyById: Map<string, SaliencyEntry>,
  newlyConnectedThrustDIndices: readonly number[],
  resamples: number,
  bootstrapSeed: number
): HypothesisResult => {
  if (newlyConnectedThrustDIndices.length === 0) {
    return {
      outcome: 'inconclusive',
      reason: 'no-newly-connected-thrust-neurons',
      evidence: { newlyConnectedThrustDIndices }
    };
  }
  const meanAt = (entry: SaliencyEntry): number =>
    newlyConnectedThrustDIndices.reduce((sum, d) => sum + entry.thrust[d], 0) / newlyConnectedThrustDIndices.length;

  const ratios = H3_SEEDS.map((seed) => {
    const p = saliencyById.get(`P-seed${seed}`);
    const bio = saliencyById.get(`biological-seed${seed}`);
    if (!p || !bio) throw new Error(`hypotheses: H3 missing saliency data for seed ${seed}`);
    const pMean = meanAt(p);
    const bioMean = meanAt(bio);
    return { seed, pMean, bioMean, ratio: bioMean !== 0 ? pMean / bioMean : Number.POSITIVE_INFINITY };
  });

  const rng = conditionRng(bootstrapSeed, 'H3');
  const ci = bootstrapMeanCI90(
    ratios.map((r) => r.ratio),
    resamples,
    rng
  );

  let outcome: HypothesisOutcome = 'inconclusive';
  if (ci[1] <= H3_RATIO_BOUND) outcome = 'supported';
  else if (ci[0] > H3_RATIO_BOUND) outcome = 'not-supported';

  return {
    outcome,
    evidence: { ratioBound: H3_RATIO_BOUND, ratios, ci, newlyConnectedThrustDIndices }
  };
};

/** P's added-edge thrust endpoints, mapped to D-space indices (`outputNeuronIndices` order) -- `post` is the newly-connected target (`interventions.py`: "collaterals ... onto thrust neurons"), restricted to targets whose `outputPopulationIndex === OUTPUT_POPULATION.thrust`. */
export const newlyConnectedThrustDIndices = (
  addedEdges: readonly SwapEdge[],
  outputPopulationIndex: Int32Array,
  indices: Int32Array
): number[] => {
  const dByNeuron = new Map<number, number>();
  for (let d = 0; d < indices.length; d += 1) dByNeuron.set(indices[d], d);
  const result = new Set<number>();
  for (const edge of addedEdges) {
    if (outputPopulationIndex[edge.post] !== OUTPUT_POPULATION.thrust) continue;
    const d = dByNeuron.get(edge.post);
    if (d !== undefined) result.add(d);
  }
  return [...result].sort((a, b) => a - b);
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

interface HypothesesArgs {
  readonly archivePath: string;
  readonly saliencyPath: string;
  readonly independencePath: string;
  readonly linkagePath: string;
  readonly regimePath: string;
  readonly swapsPath: string;
  readonly manifestPath: string;
  readonly interventionIndexPath?: string;
  readonly archivedInterventionIndexPath: string;
  readonly out: string;
  readonly resamples: number;
  readonly bootstrapSeed: number;
}

const parseArgs = (argv: readonly string[]): HypothesesArgs => {
  let archivePath = DEFAULT_ARCHIVE_PATH;
  let saliencyPath = resolve(process.cwd(), 'training/runs/attribution/saliency.json');
  let independencePath = resolve(process.cwd(), 'training/runs/attribution/independence.json');
  let linkagePath = resolve(process.cwd(), 'training/runs/attribution/linkage.json');
  let regimePath = resolve(process.cwd(), 'training/runs/attribution/regime.json');
  let swapsPath = resolve(process.cwd(), 'training/archive/intervention-swaps-v1.json');
  let manifestPath = DEFAULT_MANIFEST_PATH;
  let interventionIndexPath: string | undefined;
  let archivedInterventionIndexPath = DEFAULT_ARCHIVED_INTERVENTION_INDEX_PATH;
  let out = resolve(process.cwd(), 'training/runs/attribution/hypotheses.json');
  let resamples = 5000;
  let bootstrapSeed = 1;
  parsePathFlags('hypotheses', argv, {
    '--archive': (v) => (archivePath = resolvePathFlag(v)),
    '--saliency': (v) => (saliencyPath = resolvePathFlag(v)),
    '--independence': (v) => (independencePath = resolvePathFlag(v)),
    '--linkage': (v) => (linkagePath = resolvePathFlag(v)),
    '--regime': (v) => (regimePath = resolvePathFlag(v)),
    '--swaps': (v) => (swapsPath = resolvePathFlag(v)),
    '--manifest': (v) => (manifestPath = resolvePathFlag(v)),
    '--intervention-index': (v) => (interventionIndexPath = resolvePathFlag(v)),
    '--archived-intervention-index': (v) => (archivedInterventionIndexPath = resolvePathFlag(v)),
    '--out': (v) => (out = resolvePathFlag(v)),
    '--resamples': (v) => (resamples = Number(v)),
    '--bootstrap-seed': (v) => (bootstrapSeed = Number(v))
  });
  return {
    archivePath,
    saliencyPath,
    independencePath,
    linkagePath,
    regimePath,
    swapsPath,
    manifestPath,
    interventionIndexPath,
    archivedInterventionIndexPath,
    out,
    resamples,
    bootstrapSeed
  };
};

export const runHypotheses = (
  args: Readonly<HypothesesArgs>
): { readonly out: string; readonly sha256: string } => {
  const readouts = loadArchive(args.archivePath);
  const saliencyById = byId(
    (JSON.parse(readFileSync(args.saliencyPath, 'utf8')) as { entries: SaliencyEntry[] }).entries
  );
  const independenceById = byId(
    (JSON.parse(readFileSync(args.independencePath, 'utf8')) as { entries: IndependenceEntry[] }).entries
  );
  const linkageById = byId(
    (JSON.parse(readFileSync(args.linkagePath, 'utf8')) as { readouts: LinkageEntry[] }).readouts
  );
  const regimeById = byId((JSON.parse(readFileSync(args.regimePath, 'utf8')) as { entries: RegimeEntry[] }).entries);
  const swaps = JSON.parse(readFileSync(args.swapsPath, 'utf8')) as InterventionSwapsArchive;
  const pSwap = swaps.swaps.find((s) => s.id === 'P');
  if (!pSwap) throw new Error('hypotheses: intervention-swaps-v1.json has no "P" entry');

  const biologicalEntry = readouts.find((r) => r.id === 'biological-seed101');
  if (!biologicalEntry) throw new Error('hypotheses: archive has no "biological-seed101" entry to resolve the graph from');
  const resolveConfig = defaultResolveGraphConfig(args);
  const biologicalGraph = graphForEntry(biologicalEntry, resolveConfig);
  const indices = outputNeuronIndices(biologicalGraph);
  const thrustDIndices = newlyConnectedThrustDIndices(pSwap.addedEdges, biologicalGraph.outputPopulationIndex, indices);

  const h1 = evaluateH1(linkageById, regimeById);
  const h2 = evaluateH2(independenceById, args.resamples, args.bootstrapSeed);
  const h3 = evaluateH3(saliencyById, thrustDIndices, args.resamples, args.bootstrapSeed);

  const body = JSON.stringify({
    version: 1,
    hypothesisCount: 3,
    multipleComparisonCorrection: 'none',
    H1: h1,
    H2: h2,
    H3: h3
  });
  mkdirSync(resolve(args.out, '..'), { recursive: true });
  atomicWriteFileSync(args.out, body);
  return { out: args.out, sha256: sha256Hex(body) };
};

const main = (): void => {
  const args = parseArgs(process.argv.slice(2));
  const result = runHypotheses(args);
  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing summary output.
  console.log(`hypotheses: wrote ${result.out} (sha256 ${result.sha256})`);
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
