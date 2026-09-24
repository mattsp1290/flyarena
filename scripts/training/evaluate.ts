import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  outputNeuronIndices,
  readoutParameterCount,
  validateReadoutWeights,
  type ReadoutWeights
} from '../../src/lib/connectome/readout';
import type { ConnectomeGraph } from '../../src/lib/connectome/format';
import { mulberry32 } from '../../src/lib/random/mulberry32';
import {
  computeArmBundleSha256,
  computeGraphIdentity,
  deserializeArmBundle,
  expectedProvenanceKind,
  DEFAULT_ARMS_OUT_DIR,
  type ArmName,
  type ArmProvenance,
  type SerializedArmBundle
} from './export-arms';
import { runEpisode } from './episode';
import { readNpyFloat32Array } from './npy';
import { TRACE_SUBSTEPS } from './export-traces';

const ARM_NAMES: readonly ArmName[] = ['biological', 'rewired', 'disconnected'];

/**
 * The authoritative TypeScript rescorer: the Node code the browser's own
 * readout forward pass shares (`src/lib/connectome/readout.ts`), run here
 * headlessly (`episode.ts`) over held-out seeds, for every arm/replica a
 * caller hands it. This is the one place `.agents/plans/trained-readout/00-overview.md`'s
 * "TypeScript is the authoritative evaluator; PyTorch is training-only" key
 * decision is enforced: whatever `training/`'s (WP2/WP3, GPU-side) fitness
 * says, only this script's numbers are ever published.
 *
 * ## Run-directory contract (defined here; WP3 is not built yet)
 *
 * `training/src/flyarena_training/cem.py` (WP3,
 * `.agents/plans/trained-readout/03-cem-training.md`) does not exist on
 * `main` yet. This is the contract this evaluator defines and consumes;
 * WP3 must honor it. `tests/fixtures/trained-readout-run.ts` builds a tiny
 * synthetic run directory satisfying it, for this file's own tests.
 *
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
 *
 * `substeps` must equal the evaluation's own `--substeps` (`args.substeps`,
 * default `TRACE_SUBSTEPS`): `runEvaluate` throws if any run was trained at
 * a different `K`, rather than silently rescoring under different dynamics
 * than it was trained on. `armBundleSha256`, when present, is cross-checked
 * against the (verified) arm bundle actually loaded, so a WP3 driver can
 * pin exactly which exported bundle it trained against.
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

/** Replica 0 (`03-cem-training.md`: "Replica 0 is the one shipped to the browser"). */
const SHIPPED_TRAINER_SEED = 101;

const sha256Hex = (data: Uint8Array | string): string => createHash('sha256').update(data).digest('hex');

/**
 * `git rev-parse HEAD` for the manifest's `evaluatorGitRev` field (the
 * plan's "evaluator git rev" manifest field). `null` on any failure (not a
 * git checkout, `git` not on `PATH`, etc.) — informational provenance, not
 * a gate. Never called from `report.json`'s construction, which must stay
 * byte-identical across repeated runs in the same checkout; a manifest is
 * expected to change when the evaluator's own code does.
 */
const evaluatorGitRev = (): string | null => {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], { cwd: process.cwd(), stdio: ['ignore', 'pipe', 'ignore'] })
      .toString('utf8')
      .trim();
  } catch {
    return null;
  }
};

interface LoadedRun {
  readonly dir: string;
  readonly config: RunConfig;
  readonly weights: ReadoutWeights;
  readonly weightsSha256: string;
  readonly env: unknown | null;
}

const POSITIVE_INT_RUN_CONFIG_FIELDS = ['trainerSeed', 'D', 'H', 'parameterCount', 'substeps'] as const;

const readRunDir = (dir: string): LoadedRun => {
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

interface ArmGraphs {
  readonly biological?: ConnectomeGraph;
  readonly rewired?: ConnectomeGraph;
  readonly disconnected?: ConnectomeGraph;
  readonly graphArtifactSha256: string;
  readonly bundleSha256: Partial<Record<ArmName, string>>;
  readonly bundleProvenance: Partial<Record<ArmName, ArmProvenance>>;
}

/**
 * Loads and *verifies* each requested arm's exported bundle
 * (`export-arms.ts`'s `SerializedArmBundle`): a bundle's own `sha256`,
 * `arm` field, and `provenance.kind` are otherwise self-declared claims a
 * hand-edited or mislabeled file could lie about (e.g. copying
 * `biological.json` over `rewired.json`, or hand-tweaking magnitudes after
 * export). Recomputing the hash and cross-checking `arm`/`provenance`
 * against what this evaluator actually expects for that arm + graph source
 * is what makes `export-arms.ts`'s "a fixture swap can never be mistaken
 * for the product's rewired arm" claim true at the one place it matters:
 * here, right before the numbers it produces get published.
 */
const loadArmGraphs = (
  armsDir: string,
  needed: ReadonlySet<ArmName>,
  expectedGraphSource: 'artifact' | 'trace-graph-fixture'
): ArmGraphs => {
  const graphs: Partial<Record<ArmName, ConnectomeGraph>> = {};
  const bundleSha256: Partial<Record<ArmName, string>> = {};
  const bundleProvenance: Partial<Record<ArmName, ArmProvenance>> = {};
  let graphArtifactSha256: string | undefined;
  for (const arm of needed) {
    const bundlePath = resolve(armsDir, `${arm}.json`);
    if (!existsSync(bundlePath)) {
      throw new Error(
        `evaluate: missing arm bundle ${bundlePath}; run "npm run training:export-arms" for ` +
          'this graph first'
      );
    }
    const bundle = JSON.parse(readFileSync(bundlePath, 'utf8')) as SerializedArmBundle;
    if (bundle.formatVersion !== 1) {
      throw new Error(`evaluate: ${bundlePath} has unsupported formatVersion ${bundle.formatVersion}`);
    }
    if (bundle.arm !== arm) {
      throw new Error(`evaluate: ${bundlePath} declares arm "${bundle.arm}", expected "${arm}"`);
    }
    const { sha256, ...withoutHash } = bundle;
    if (computeArmBundleSha256(withoutHash) !== sha256) {
      throw new Error(`evaluate: ${bundlePath}'s sha256 does not match its content (tampered or stale file?)`);
    }
    if (bundle.graphSource !== expectedGraphSource) {
      throw new Error(
        `evaluate: ${bundlePath}'s graphSource "${bundle.graphSource}" does not match the ` +
          `evaluation graph's source "${expectedGraphSource}"`
      );
    }
    const expectedKind = expectedProvenanceKind(arm, bundle.graphSource);
    if (bundle.provenance.kind !== expectedKind) {
      throw new Error(
        `evaluate: ${bundlePath}'s provenance.kind "${bundle.provenance.kind}" is not valid for ` +
          `arm "${arm}" with graphSource "${bundle.graphSource}" (expected "${expectedKind}")`
      );
    }

    graphs[arm] = deserializeArmBundle(bundle);
    bundleSha256[arm] = bundle.sha256;
    bundleProvenance[arm] = bundle.provenance;
    if (graphArtifactSha256 === undefined) {
      graphArtifactSha256 = bundle.graphArtifactSha256;
    } else if (graphArtifactSha256 !== bundle.graphArtifactSha256) {
      throw new Error(`evaluate: arm bundles under ${armsDir} disagree on graphArtifactSha256`);
    }
  }
  if (graphArtifactSha256 === undefined) {
    throw new Error('evaluate: no arms requested (no run directories given)');
  }
  return { ...graphs, graphArtifactSha256, bundleSha256, bundleProvenance };
};

// ---------------------------------------------------------------------------
// Statistics: mean/median/std, seeded bootstrap CIs, paired differences.
// ---------------------------------------------------------------------------

interface ConditionStats {
  readonly n: number;
  readonly mean: number;
  readonly median: number;
  /** Population standard deviation (divide-by-n): descriptive, not inferential (the CI is). */
  readonly std: number;
  readonly ci95: readonly [number, number];
}

interface PairedStats {
  readonly n: number;
  readonly meanDifference: number;
  readonly ci95: readonly [number, number];
}

const mean = (values: readonly number[]): number => values.reduce((sum, value) => sum + value, 0) / values.length;

const median = (values: readonly number[]): number => {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  return n % 2 === 1 ? sorted[(n - 1) / 2] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
};

const std = (values: readonly number[]): number => {
  const m = mean(values);
  const variance = values.reduce((sum, value) => sum + (value - m) ** 2, 0) / values.length;
  return Math.sqrt(variance);
};

/** Bootstrap the 95% CI of `statistic(resample)` over `resamples` seeded, with-replacement resamples. */
const bootstrapCI = (
  n: number,
  resamples: number,
  rng: () => number,
  statistic: (pickIndex: () => number) => number
): readonly [number, number] => {
  const draws = new Array<number>(resamples);
  for (let r = 0; r < resamples; r += 1) {
    draws[r] = statistic(() => Math.floor(rng() * n));
  }
  draws.sort((a, b) => a - b);
  const lowIndex = Math.floor(0.025 * resamples);
  const highIndex = Math.min(resamples - 1, Math.ceil(0.975 * resamples) - 1);
  return [draws[lowIndex], draws[highIndex]];
};

const conditionStats = (
  values: readonly number[],
  resamples: number,
  rng: () => number
): ConditionStats => ({
  n: values.length,
  mean: mean(values),
  median: median(values),
  std: std(values),
  ci95: bootstrapCI(values.length, resamples, rng, (pickIndex) => {
    let sum = 0;
    for (let i = 0; i < values.length; i += 1) sum += values[pickIndex()];
    return sum / values.length;
  })
});

/** Paired difference `a[i] - b[i]` for same-seed pairs, with a seeded bootstrap CI on the mean difference. */
const pairedStats = (
  a: readonly number[],
  b: readonly number[],
  resamples: number,
  rng: () => number
): PairedStats => {
  if (a.length !== b.length) throw new Error('evaluate: paired series must have equal length');
  const diffs = a.map((value, index) => value - b[index]);
  return {
    n: diffs.length,
    meanDifference: mean(diffs),
    ci95: bootstrapCI(diffs.length, resamples, rng, (pickIndex) => {
      let sum = 0;
      for (let i = 0; i < diffs.length; i += 1) sum += diffs[pickIndex()];
      return sum / diffs.length;
    })
  };
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface EvaluateArgs {
  readonly graphPath?: string;
  readonly armsDir?: string;
  readonly runDirs: readonly string[];
  readonly outDir: string;
  /** True only when `--out` was actually passed, not merely defaulted (see `runEvaluate`'s public/data guard). */
  readonly outDirExplicit: boolean;
  readonly ticks: number;
  readonly substeps: number;
  readonly heldOutStart: number;
  readonly heldOutCount: number;
  readonly bootstrapResamples: number;
  readonly bootstrapSeed: number;
}

const DEFAULT_OUT_DIR = 'public/data';

/** 'E','V','A','L' as a fixed default seed; arbitrary but stable across runs. */
const DEFAULT_BOOTSTRAP_SEED = 0x4556_414c;
const DEFAULT_TICKS = 1800;
const DEFAULT_HELD_OUT_START = 30001;
const DEFAULT_HELD_OUT_COUNT = 100;
const DEFAULT_BOOTSTRAP_RESAMPLES = 10000;

const requireValue = (flag: string, value: string | undefined): string => {
  if (value === undefined || value.startsWith('--')) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
};

const requirePositiveInt = (flag: string, value: string | undefined): number => {
  const raw = requireValue(flag, value);
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} must be a positive integer, got "${raw}"`);
  }
  return parsed;
};

/** Like `requirePositiveInt`, but accepts 0 — for seed-like flags, where 0 is a meaningful seed. */
const requireNonNegativeInt = (flag: string, value: string | undefined): number => {
  const raw = requireValue(flag, value);
  const parsed = Number(raw);
  if (!Number.isInteger(parsed) || parsed < 0) {
    throw new Error(`${flag} must be a non-negative integer, got "${raw}"`);
  }
  return parsed;
};

export const parseEvaluateArgs = (argv: readonly string[]): EvaluateArgs => {
  let graphPath: string | undefined;
  let armsDir: string | undefined;
  const runDirs: string[] = [];
  let outDir = DEFAULT_OUT_DIR;
  let outDirExplicit = false;
  let ticks = DEFAULT_TICKS;
  let substeps = TRACE_SUBSTEPS;
  let heldOutStart = DEFAULT_HELD_OUT_START;
  let heldOutCount = DEFAULT_HELD_OUT_COUNT;
  let bootstrapResamples = DEFAULT_BOOTSTRAP_RESAMPLES;
  let bootstrapSeed = DEFAULT_BOOTSTRAP_SEED;

  let index = 0;
  while (index < argv.length) {
    const flag = argv[index];
    if (flag === '--graph') {
      graphPath = requireValue(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--arms-dir') {
      armsDir = requireValue(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--runs') {
      index += 1;
      let consumed = 0;
      while (index < argv.length && !argv[index].startsWith('--')) {
        runDirs.push(argv[index]);
        index += 1;
        consumed += 1;
      }
      if (consumed === 0) throw new Error('--runs requires at least one run directory');
    } else if (flag === '--out') {
      outDir = requireValue(flag, argv[index + 1]);
      outDirExplicit = true;
      index += 2;
    } else if (flag === '--ticks') {
      ticks = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--substeps') {
      substeps = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--held-out-start') {
      heldOutStart = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--held-out-count') {
      heldOutCount = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--bootstrap-resamples') {
      bootstrapResamples = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--bootstrap-seed') {
      bootstrapSeed = requireNonNegativeInt(flag, argv[index + 1]);
      index += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (runDirs.length === 0) throw new Error('--runs requires at least one run directory');

  return {
    graphPath,
    armsDir,
    runDirs,
    outDir,
    outDirExplicit,
    ticks,
    substeps,
    heldOutStart,
    heldOutCount,
    bootstrapResamples,
    bootstrapSeed
  };
};

// ---------------------------------------------------------------------------
// Core evaluation
// ---------------------------------------------------------------------------

export interface RunEvaluateResult {
  readonly warnings: readonly string[];
  readonly reportPath: string;
  readonly artifactWritten: boolean;
}

/** Core logic, separated from CLI parsing/`main` so tests can call it in-process without a subprocess. */
export const runEvaluate = (args: Readonly<EvaluateArgs>): RunEvaluateResult => {
  const graphIdentity = computeGraphIdentity(args.graphPath);

  // Trace-graph dev mode (no --graph) must never silently overwrite the
  // real shipped artifact: `npm run training:evaluate -- --runs <dirs>`
  // with no --graph defaults to the trace graph AND to --out public/data,
  // which would otherwise clobber the product's public/data/trained-readout-v1.*
  // with fixture output. Mirrors export-traces.ts's --out overwrite guard.
  if (graphIdentity.graphSource !== 'artifact') {
    const resolvedOutDir = resolve(process.cwd(), args.outDir);
    const resolvedDefaultOutDir = resolve(process.cwd(), DEFAULT_OUT_DIR);
    if (!args.outDirExplicit || resolvedOutDir === resolvedDefaultOutDir) {
      throw new Error(
        'evaluate: trace-graph mode (no --graph) refuses to write to the default --out ' +
          `(${DEFAULT_OUT_DIR}); pass --out <scratch dir> explicitly, or pass --graph <artifact> ` +
          'for a real evaluation run.'
      );
    }
  }

  const runs = args.runDirs.map((dir) => readRunDir(dir));
  const seenKeys = new Set<string>();
  for (const run of runs) {
    const key = `${run.config.arm}|${run.config.trainerSeed}`;
    if (seenKeys.has(key)) {
      throw new Error(`evaluate: duplicate run for arm "${run.config.arm}" trainerSeed ${run.config.trainerSeed}`);
    }
    seenKeys.add(key);
    if (run.config.substeps !== args.substeps) {
      throw new Error(
        `evaluate: ${run.dir}/config.json was trained at substeps=${run.config.substeps} but ` +
          `evaluation is running at --substeps ${args.substeps}; pass --substeps ${run.config.substeps} ` +
          'or re-check which run this is.'
      );
    }
  }

  const neededArms = new Set(runs.map((run) => run.config.arm));
  const armsDir = args.armsDir ?? resolve(DEFAULT_ARMS_OUT_DIR, graphIdentity.graphArtifactSha256);
  const armGraphs = loadArmGraphs(armsDir, neededArms, graphIdentity.graphSource);
  if (armGraphs.graphArtifactSha256 !== graphIdentity.graphArtifactSha256) {
    throw new Error(
      `evaluate: arm bundles under ${armsDir} were exported from a different graph ` +
        `(bundle graphArtifactSha256=${armGraphs.graphArtifactSha256}) than --graph resolves to ` +
        `(${graphIdentity.graphArtifactSha256})`
    );
  }
  for (const run of runs) {
    const loadedSha256 = armGraphs.bundleSha256[run.config.arm];
    if (run.config.armBundleSha256 && loadedSha256 && run.config.armBundleSha256 !== loadedSha256) {
      throw new Error(
        `evaluate: ${run.dir}/config.json was trained against arm bundle sha256 ` +
          `${run.config.armBundleSha256}, but the loaded "${run.config.arm}" bundle is ${loadedSha256}`
      );
    }
  }

  const armNames = (['biological', 'rewired', 'disconnected'] as const).filter((arm) => armGraphs[arm]);
  const armD = new Map<ArmName, number>(armNames.map((arm) => [arm, outputNeuronIndices(armGraphs[arm]!).length]));
  const distinctD = new Set(armD.values());
  if (distinctD.size > 1) {
    throw new Error(`evaluate: D (output-neuron count) differs across loaded arm graphs under ${armsDir}`);
  }

  const heldOutSeeds = Array.from({ length: args.heldOutCount }, (_, i) => args.heldOutStart + i);
  const rng = mulberry32(args.bootstrapSeed);

  const scoreCache = new Map<string, number>();
  const computeLeftScore = (
    arm: ArmName,
    decoder: 'trained' | 'authored' | 'silenced',
    trainerSeed: number | null,
    weights: ReadoutWeights | null,
    seed: number
  ): number => {
    const key = `${arm}|${decoder}|${trainerSeed ?? 'na'}|${seed}`;
    const cached = scoreCache.get(key);
    if (cached !== undefined) return cached;
    const graph = armGraphs[arm];
    if (!graph) throw new Error(`evaluate: arm "${arm}" graph is not loaded`);
    const result = runEpisode({
      seed,
      ticks: args.ticks,
      substeps: args.substeps,
      left: { decoder, graph, weights: weights ?? undefined },
      right: { decoder: 'parked' }
    });
    scoreCache.set(key, result.left.movementScore);
    return result.left.movementScore;
  };

  const armReplicas = new Map<ArmName, Map<number, LoadedRun>>();
  for (const run of runs) {
    if (!armReplicas.has(run.config.arm)) armReplicas.set(run.config.arm, new Map());
    armReplicas.get(run.config.arm)!.set(run.config.trainerSeed, run);
  }
  /**
   * Ascending-by-trainerSeed entries for one arm's replicas. `armReplicas`'s
   * `Map` iterates in insertion order, which is the order run dirs appeared
   * on `--runs` — sorting here makes every downstream loop order (and
   * therefore the single shared bootstrap `rng`'s draw sequence, and
   * `armPairs`'s element order) independent of `--runs` argument order, so
   * the same *set* of runs always produces the same `report.json` bytes,
   * not just the same argv.
   */
  const sortedReplicas = (arm: ArmName): Array<[number, LoadedRun]> =>
    [...(armReplicas.get(arm) ?? new Map<number, LoadedRun>())].sort(([a], [b]) => a - b);

  const warnings: string[] = [];

  const armsReport: Record<string, unknown> = {};
  for (const arm of armNames) {
    const graph = armGraphs[arm]!;
    const D = armD.get(arm)!;

    const authoredScores = heldOutSeeds.map((seed) => computeLeftScore(arm, 'authored', null, null, seed));
    const authoredStats = conditionStats(authoredScores, args.bootstrapResamples, rng);

    const replicasReport: Record<string, unknown> = {};
    for (const [trainerSeed, run] of sortedReplicas(arm)) {
      validateReadoutWeights(run.weights, graph);
      const trainedScores = heldOutSeeds.map((seed) => computeLeftScore(arm, 'trained', trainerSeed, run.weights, seed));
      const silencedScores = heldOutSeeds.map((seed) => computeLeftScore(arm, 'silenced', trainerSeed, run.weights, seed));
      replicasReport[String(trainerSeed)] = {
        H: run.config.H,
        parameterCount: run.config.parameterCount,
        weightsSha256: run.weightsSha256,
        env: run.env,
        trained: conditionStats(trainedScores, args.bootstrapResamples, rng),
        silenced: conditionStats(silencedScores, args.bootstrapResamples, rng),
        pairedTrainedVsAuthored: pairedStats(trainedScores, authoredScores, args.bootstrapResamples, rng),
        pairedTrainedVsSilenced: pairedStats(trainedScores, silencedScores, args.bootstrapResamples, rng)
      };
    }

    armsReport[arm] = {
      D,
      provenance: armGraphs.bundleProvenance[arm],
      armBundleSha256: armGraphs.bundleSha256[arm],
      authored: authoredStats,
      replicas: replicasReport
    };
  }

  const armPairs: unknown[] = [];
  for (let i = 0; i < armNames.length; i += 1) {
    for (let j = i + 1; j < armNames.length; j += 1) {
      const armA = armNames[i];
      const armB = armNames[j];

      const authoredA = heldOutSeeds.map((seed) => computeLeftScore(armA, 'authored', null, null, seed));
      const authoredB = heldOutSeeds.map((seed) => computeLeftScore(armB, 'authored', null, null, seed));
      armPairs.push({
        condition: 'authored',
        trainerSeed: null,
        armA,
        armB,
        pairedDifference: pairedStats(authoredA, authoredB, args.bootstrapResamples, rng)
      });

      const replicasB = armReplicas.get(armB) ?? new Map<number, LoadedRun>();
      for (const [trainerSeed, runA] of sortedReplicas(armA)) {
        const runB = replicasB.get(trainerSeed);
        if (!runB) continue;
        for (const condition of ['trained', 'silenced'] as const) {
          const scoresA = heldOutSeeds.map((seed) => computeLeftScore(armA, condition, trainerSeed, runA.weights, seed));
          const scoresB = heldOutSeeds.map((seed) => computeLeftScore(armB, condition, trainerSeed, runB.weights, seed));
          armPairs.push({
            condition,
            trainerSeed,
            armA,
            armB,
            pairedDifference: pairedStats(scoresA, scoresB, args.bootstrapResamples, rng)
          });
        }
      }
    }
  }

  const sideBySide: unknown[] = [];
  if (armGraphs.biological && armGraphs.rewired) {
    const leftAuthored: number[] = [];
    const rightAuthored: number[] = [];
    for (const seed of heldOutSeeds) {
      const result = runEpisode({
        seed,
        ticks: args.ticks,
        substeps: args.substeps,
        left: { decoder: 'authored', graph: armGraphs.biological },
        right: { decoder: 'authored', graph: armGraphs.rewired }
      });
      leftAuthored.push(result.left.movementScore);
      rightAuthored.push(result.right.movementScore);
    }
    sideBySide.push({
      label: 'authored-side-by-side',
      leftArm: 'biological',
      rightArm: 'rewired',
      replica: null,
      left: conditionStats(leftAuthored, args.bootstrapResamples, rng),
      right: conditionStats(rightAuthored, args.bootstrapResamples, rng),
      pairedLeftMinusRight: pairedStats(leftAuthored, rightAuthored, args.bootstrapResamples, rng)
    });

    const rewiredReplicas = armReplicas.get('rewired') ?? new Map<number, LoadedRun>();
    for (const [trainerSeed, bioRun] of sortedReplicas('biological')) {
      const rewiredRun = rewiredReplicas.get(trainerSeed);
      if (!rewiredRun) continue;
      const leftTrained: number[] = [];
      const rightTrained: number[] = [];
      for (const seed of heldOutSeeds) {
        const result = runEpisode({
          seed,
          ticks: args.ticks,
          substeps: args.substeps,
          left: { decoder: 'trained', graph: armGraphs.biological, weights: bioRun.weights },
          right: { decoder: 'trained', graph: armGraphs.rewired, weights: rewiredRun.weights }
        });
        leftTrained.push(result.left.movementScore);
        rightTrained.push(result.right.movementScore);
      }
      sideBySide.push({
        label: 'trained-side-by-side',
        leftArm: 'biological',
        rightArm: 'rewired',
        replica: trainerSeed,
        left: conditionStats(leftTrained, args.bootstrapResamples, rng),
        right: conditionStats(rightTrained, args.bootstrapResamples, rng),
        pairedLeftMinusRight: pairedStats(leftTrained, rightTrained, args.bootstrapResamples, rng)
      });
    }
  } else {
    warnings.push(
      'trained-side-by-side / authored-side-by-side skipped: both a "biological" and a ' +
        '"rewired" arm are required (via --runs) and at least one was not provided.'
    );
  }

  // Shipped artifact (replica 0 = trainerSeed 101): the plan's format is
  // `arms: { biological, rewired, disconnected }` — all three, always — so
  // completeness requires all three `ARM_NAMES`, not merely the arms this
  // invocation happened to evaluate. Every skip reason is pushed as a
  // warning: a silently-missing shipped artifact is exactly the kind of
  // mistake a WP5 production run must not be able to make quietly.
  const shippedWeights: Partial<Record<ArmName, ReadoutWeights>> = {};
  const shippedEnv: Partial<Record<ArmName, unknown>> = {};
  let shippedD: number | null = null;
  let shippedH: number | null = null;
  let complete = true;
  for (const arm of ARM_NAMES) {
    const run = armReplicas.get(arm)?.get(SHIPPED_TRAINER_SEED);
    if (!run) {
      complete = false;
      warnings.push(
        `trained-readout-v1.json not written: no replica 0 (trainerSeed ${SHIPPED_TRAINER_SEED}) run for arm "${arm}".`
      );
      continue;
    }
    if (shippedD !== null && shippedD !== run.config.D) {
      complete = false;
      warnings.push(
        `trained-readout-v1.json not written: arm "${arm}"'s D (${run.config.D}) differs from ${shippedD}.`
      );
    }
    if (shippedH !== null && shippedH !== run.config.H) {
      complete = false;
      warnings.push(
        `trained-readout-v1.json not written: arm "${arm}"'s H (${run.config.H}) differs from ${shippedH}.`
      );
    }
    shippedWeights[arm] = run.weights;
    shippedEnv[arm] = run.env;
    if (run.env === null) {
      warnings.push(`arm "${arm}" replica 0 run "${run.dir}" has no env.json; manifest omits its device/torch provenance.`);
    }
    shippedD ??= run.config.D;
    shippedH ??= run.config.H;
  }

  const report = {
    formatVersion: 1,
    graph: {
      source: graphIdentity.graphSource,
      path: graphIdentity.graphArtifactPath,
      sha256: graphIdentity.graphArtifactSha256
    },
    evaluation: {
      ticks: args.ticks,
      substeps: args.substeps,
      heldOutSeeds: { start: args.heldOutStart, count: args.heldOutCount },
      bootstrap: { resamples: args.bootstrapResamples, seed: args.bootstrapSeed },
      opponentParked: true
    },
    arms: armsReport,
    armPairs,
    sideBySide,
    warnings
  };

  const outDir = resolve(process.cwd(), args.outDir);
  mkdirSync(outDir, { recursive: true });
  const reportPath = resolve(outDir, 'trained-readout-v1.report.json');
  writeFileSync(reportPath, JSON.stringify(report));

  let artifactWritten = false;
  if (complete && shippedD !== null && shippedH !== null) {
    const encodeBase64 = (values: Float32Array): string =>
      Buffer.from(values.buffer, values.byteOffset, values.byteLength).toString('base64');
    const arms: Record<string, unknown> = {};
    for (const arm of ARM_NAMES) {
      const weights = shippedWeights[arm]!;
      arms[arm] = {
        w1: encodeBase64(weights.w1),
        b1: encodeBase64(weights.b1),
        w2: encodeBase64(weights.w2),
        b2: encodeBase64(weights.b2)
      };
    }
    const artifact = { version: 1, hiddenSize: shippedH, inputSize: shippedD, arms };
    const artifactContents = JSON.stringify(artifact);
    writeFileSync(resolve(outDir, 'trained-readout-v1.json'), artifactContents);
    const artifactSha256 = sha256Hex(artifactContents);

    // Already-verified (loadArmGraphs recomputed and checked each of
    // these against its bundle's own content) — no need to re-read and
    // re-trust the files from disk a second time here.
    const armBundleSha256: Record<string, string> = {};
    const armProvenance: Record<string, ArmProvenance> = {};
    for (const arm of ARM_NAMES) {
      armBundleSha256[arm] = armGraphs.bundleSha256[arm]!;
      armProvenance[arm] = armGraphs.bundleProvenance[arm]!;
    }

    const manifest = {
      version: 1,
      artifactSha256,
      graphArtifactSha256: graphIdentity.graphArtifactSha256,
      graphSource: graphIdentity.graphSource,
      armBundleSha256,
      armProvenance,
      D: shippedD,
      H: shippedH,
      parameterCount: readoutParameterCount(shippedD, shippedH),
      shippedReplicaTrainerSeed: SHIPPED_TRAINER_SEED,
      heldOutSeeds: { start: args.heldOutStart, count: args.heldOutCount },
      /** Per-arm shipped-replica torch/CUDA/device provenance, when its run had an `env.json`; `null` otherwise. */
      env: shippedEnv,
      evaluatorGitRev: evaluatorGitRev()
    };
    writeFileSync(resolve(outDir, 'trained-readout-v1.manifest.json'), JSON.stringify(manifest));
    artifactWritten = true;
  }

  return { warnings, reportPath, artifactWritten };
};

const main = (): void => {
  try {
    const args = parseEvaluateArgs(process.argv.slice(2));
    const result = runEvaluate(args);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `evaluate: wrote ${result.reportPath}${result.artifactWritten ? ' and trained-readout-v1.{json,manifest.json}' : ''}` +
        (result.warnings.length > 0 ? `\nwarnings:\n  ${result.warnings.join('\n  ')}` : '')
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`evaluate failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
