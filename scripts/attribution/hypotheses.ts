import { mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { OUTPUT_POPULATION } from '../../src/lib/arena/actions';
import { outputNeuronIndices } from '../../src/lib/connectome/readout';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import {
  assertSameArchive,
  computeArchiveSha256,
  DEFAULT_ARCHIVE_PATH,
  DEFAULT_ARCHIVED_INTERVENTION_INDEX_PATH,
  DEFAULT_MANIFEST_PATH,
  defaultResolveGraphConfig,
  graphForEntry,
  loadArchive,
  parseFlags,
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
// Mean 90% CI for H2/H3, both always paired on exactly the 3 predeclared
// trainer seeds (`H2_SEED_PAIRS`/`H3_SEEDS`).
//
// A dual-review finding: a percentile bootstrap of the mean at n = 3 has
// only 10 distinct resample multisets and can never extend beyond
// `[min(values), max(values)]` -- P(every one of 3 draws lands on the same
// extreme value) = 1/27 ~= 3.7% < 5%, so its 90% interval is narrower than
// the data's real sampling uncertainty and makes a TOST-style "supported"
// verdict too easy to reach (worked example, an earlier version of this
// function: paired differences [0.05, -0.02, 0.08] bootstrap to a 90% CI of
// [0.0033, 0.0700], entirely inside +-0.10 -- "supported" -- while the
// textbook t-based 90% CI for the same 3 numbers is [-0.0498, 0.1232],
// which straddles +0.10 and is honestly `inconclusive`). This module uses
// the standard one-sample t interval instead (df = n - 1, always 2 here
// since n is always 3), which is the conventional TOST construction for a
// small paired sample and does not silently understate uncertainty at
// n = 3. `00-overview.md`'s own text ("With n = 3 pairs the resolution is
// coarse") is a plan requirement that this interval actually reflects,
// rather than one the previous bootstrap-based interval quietly hid.
// ---------------------------------------------------------------------------

/** Two-sided 90% (one-sided 95%) t critical value, keyed by degrees of freedom (n - 1). Only n = 3 (df = 2) is ever used by this module today; a couple of neighbors are kept for headroom, and an unlisted df throws rather than silently extrapolating. */
const T_95_ONE_SIDED: Readonly<Record<number, number>> = { 1: 6.313752, 2: 2.919986, 3: 2.353363, 4: 2.131847 };

const tMeanCI90 = (values: readonly number[]): readonly [number, number] => {
  const n = values.length;
  const t = T_95_ONE_SIDED[n - 1];
  if (n < 2 || t === undefined) {
    throw new Error(`hypotheses: tMeanCI90 needs 2 <= n <= ${Object.keys(T_95_ONE_SIDED).length + 1}, got n=${n}`);
  }
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (n - 1);
  const halfWidth = t * Math.sqrt(variance / n);
  return [mean - halfWidth, mean + halfWidth];
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
  /** `null` when `linkage.py` flagged this readout `degenerate` (a constant `|T_clear|` or saliency vector -- Spearman's rho is mathematically undefined, not "measured zero"). */
  readonly degenerate?: boolean;
  readonly rhoThrust: number | null;
  readonly ciCluster: readonly [number, number] | null;
  readonly ciNeuron: readonly [number, number] | null;
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
  readonly degenerate: boolean;
  readonly rho: number | null;
  readonly ciCluster: readonly [number, number] | null;
  readonly ciNeuron: readonly [number, number] | null;
  readonly clusterExcludesZero: boolean;
  readonly neuronExcludesZero: boolean;
  readonly agree: boolean;
  readonly meetsThreshold: boolean;
}

/**
 * H1 has only two outcomes -- `supported` or `inconclusive` -- never
 * `not-supported`, unlike H2/H3. This is deliberate, grounded in
 * `00-overview.md`'s own text: H1's rule only ever defines what "consistent
 * with routing around" (`supported`) means, plus two named routes to
 * `inconclusive` (regime-invalid, cluster/neuron-bootstrap disagreement);
 * it never defines a "refuted"/`not-supported` condition, unlike H2
 * ("'Not supported' means...") and H3 (same). H1 is explicitly correlational
 * ("H1 is always described in correlational language ('consistent with')")
 * -- a `not-supported` verdict on a correlational hypothesis at n = 3
 * biological seeds would itself overclaim (absence of a threshold-meeting
 * correlation is not evidence the correlation is truly absent), so a
 * regime-valid, bootstrap-agreeing, threshold-missing result is reported as
 * `inconclusive` with `reason: 'threshold-not-met-in-all-seeds'`, not
 * `not-supported` (a dual-review question, resolved against the plan text
 * above rather than by adding a branch the plan never asked for).
 */
export const evaluateH1 = (
  linkageById: Map<string, LinkageEntry>,
  regimeById: Map<string, RegimeEntry>
): HypothesisResult => {
  const perSeed: H1SeedEvidence[] = BIOLOGICAL_SEED_IDS.map((id) => {
    const linkage = linkageById.get(id);
    const regime = regimeById.get(id);
    if (!linkage || !regime) throw new Error(`hypotheses: H1 missing linkage/regime data for "${id}"`);
    // `linkage.py` flags a biological seed `degenerate` when its |T_clear|
    // or thrust-saliency vector is constant, making Spearman's rho
    // mathematically undefined (not a real "no correlation" measurement).
    // Treated as failing threshold/agreement here, never crashing on a
    // `null` CI -- `00-overview.md` has no real archived biological
    // readout that hits this (verified: none of the three do), but this
    // path is exercised defensively rather than left to throw a TypeError.
    const degenerate = linkage.degenerate === true || linkage.rhoThrust === null || linkage.ciCluster === null || linkage.ciNeuron === null;
    const clusterExcludesZero = !degenerate && excludesZero(linkage.ciCluster as readonly [number, number]);
    const neuronExcludesZero = !degenerate && excludesZero(linkage.ciNeuron as readonly [number, number]);
    return {
      id,
      regimeValid: regime.valid,
      degenerate,
      rho: linkage.rhoThrust,
      ciCluster: linkage.ciCluster,
      ciNeuron: linkage.ciNeuron,
      clusterExcludesZero,
      neuronExcludesZero,
      agree: degenerate || clusterExcludesZero === neuronExcludesZero,
      meetsThreshold: !degenerate && (linkage.rhoThrust as number) >= RHO_THRESHOLD && clusterExcludesZero
    };
  });

  const anyRegimeInvalid = perSeed.some((s) => !s.regimeValid);
  const anyDegenerate = perSeed.some((s) => s.degenerate);
  const anyDisagreement = perSeed.some((s) => !s.agree);
  const allMeetThreshold = perSeed.every((s) => s.meetsThreshold);

  let outcome: HypothesisOutcome = 'inconclusive';
  let reason: string | undefined;
  if (anyRegimeInvalid) {
    reason = 'regime-invalid';
  } else if (anyDegenerate) {
    reason = 'degenerate-correlation';
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

export const evaluateH2 = (independenceById: Map<string, IndependenceEntry>): HypothesisResult => {
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
  const ci = tMeanCI90(differences);

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
  newlyConnectedThrustDIndices: readonly number[]
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

  // `ratio: null` when the biological mean is not strictly positive (mean
  // absolute saliency is non-negative by construction, so this is only the
  // degenerate all-zero case) -- previously `bioMean !== 0 ? ... :
  // Number.POSITIVE_INFINITY`, which silently turned a genuinely undefined
  // 0/0 ratio into `+Infinity` (itself then serialized as `null` by
  // `JSON.stringify`, indistinguishable from missing data) and, worse, made
  // `ci[0] > H3_RATIO_BOUND` true -- reporting a real `not-supported`
  // verdict from an undefined ratio (a dual-review finding). A seed with an
  // undefined ratio now makes the whole hypothesis `inconclusive`, the same
  // failure-closed convention `evaluateH2` already uses for
  // `defined: false` independence shares.
  const ratios = H3_SEEDS.map((seed) => {
    const p = saliencyById.get(`P-seed${seed}`);
    const bio = saliencyById.get(`biological-seed${seed}`);
    if (!p || !bio) throw new Error(`hypotheses: H3 missing saliency data for seed ${seed}`);
    const pMean = meanAt(p);
    const bioMean = meanAt(bio);
    return { seed, pMean, bioMean, ratio: bioMean > 0 ? pMean / bioMean : null };
  });

  const undefinedSeeds = ratios.filter((r) => r.ratio === null).map((r) => r.seed);
  if (undefinedSeeds.length > 0) {
    return {
      outcome: 'inconclusive',
      reason: 'undefined-saliency-ratio',
      evidence: { ratioBound: H3_RATIO_BOUND, ratios, undefinedSeeds, newlyConnectedThrustDIndices }
    };
  }

  const ci = tMeanCI90(ratios.map((r) => r.ratio as number));

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
  parseFlags('hypotheses', argv, {
    '--archive': (v) => (archivePath = resolvePathFlag(v)),
    '--saliency': (v) => (saliencyPath = resolvePathFlag(v)),
    '--independence': (v) => (independencePath = resolvePathFlag(v)),
    '--linkage': (v) => (linkagePath = resolvePathFlag(v)),
    '--regime': (v) => (regimePath = resolvePathFlag(v)),
    '--swaps': (v) => (swapsPath = resolvePathFlag(v)),
    '--manifest': (v) => (manifestPath = resolvePathFlag(v)),
    '--intervention-index': (v) => (interventionIndexPath = resolvePathFlag(v)),
    '--archived-intervention-index': (v) => (archivedInterventionIndexPath = resolvePathFlag(v)),
    '--out': (v) => (out = resolvePathFlag(v))
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
    out
  };
};

export const runHypotheses = (
  args: Readonly<HypothesesArgs>
): { readonly out: string; readonly sha256: string } => {
  const readouts = loadArchive(args.archivePath);
  const archiveSha256 = computeArchiveSha256(args.archivePath);

  const saliencyOutput = JSON.parse(readFileSync(args.saliencyPath, 'utf8')) as {
    readonly archiveSha256?: string;
    readonly entries: SaliencyEntry[];
  };
  assertSameArchive('hypotheses', args.saliencyPath, saliencyOutput.archiveSha256, archiveSha256);
  const saliencyById = byId(saliencyOutput.entries);

  const independenceOutput = JSON.parse(readFileSync(args.independencePath, 'utf8')) as {
    readonly archiveSha256?: string;
    readonly entries: IndependenceEntry[];
  };
  assertSameArchive('hypotheses', args.independencePath, independenceOutput.archiveSha256, archiveSha256);
  const independenceById = byId(independenceOutput.entries);

  const linkageOutput = JSON.parse(readFileSync(args.linkagePath, 'utf8')) as {
    readonly archiveSha256?: string;
    readonly readouts: LinkageEntry[];
  };
  assertSameArchive('hypotheses', args.linkagePath, linkageOutput.archiveSha256, archiveSha256);
  const linkageById = byId(linkageOutput.readouts);

  const regimeOutput = JSON.parse(readFileSync(args.regimePath, 'utf8')) as {
    readonly archiveSha256?: string;
    readonly entries: RegimeEntry[];
  };
  assertSameArchive('hypotheses', args.regimePath, regimeOutput.archiveSha256, archiveSha256);
  const regimeById = byId(regimeOutput.entries);

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
  const h2 = evaluateH2(independenceById);
  const h3 = evaluateH3(saliencyById, thrustDIndices);

  const body = JSON.stringify({
    version: 1,
    archiveSha256,
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
