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
  computeGraphIdentity,
  deserializeArmBundle,
  DEFAULT_ARMS_OUT_DIR,
  type ArmName,
  type SerializedArmBundle
} from './export-arms';
import { runEpisode } from './episode';
import { readNpyFloat32Array } from './npy';
import { TRACE_SUBSTEPS } from './export-traces';

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
 *   - `env.json` (optional): torch/CUDA/device provenance, copied into the
 *     shipped manifest when present; a missing file is recorded as a
 *     report warning, not an error.
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
}

/** Replica 0 (`03-cem-training.md`: "Replica 0 is the one shipped to the browser"). */
const SHIPPED_TRAINER_SEED = 101;

const sha256Hex = (data: Uint8Array | string): string => createHash('sha256').update(data).digest('hex');

interface LoadedRun {
  readonly dir: string;
  readonly config: RunConfig;
  readonly weights: ReadoutWeights;
  readonly weightsSha256: string;
  readonly env: unknown | null;
}

const REQUIRED_RUN_CONFIG_FIELDS = ['arm', 'trainerSeed', 'D', 'H', 'parameterCount', 'substeps'] as const;

const readRunDir = (dir: string): LoadedRun => {
  const configPath = resolve(dir, 'config.json');
  const config = JSON.parse(readFileSync(configPath, 'utf8')) as Partial<RunConfig>;
  for (const field of REQUIRED_RUN_CONFIG_FIELDS) {
    if (config[field] === undefined) {
      throw new Error(`evaluate: ${configPath} is missing required field "${field}"`);
    }
  }
  const runConfig = config as RunConfig;
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
}

const loadArmGraphs = (armsDir: string, needed: ReadonlySet<ArmName>): ArmGraphs => {
  const graphs: Partial<Record<ArmName, ConnectomeGraph>> = {};
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
    graphs[arm] = deserializeArmBundle(bundle);
    if (graphArtifactSha256 === undefined) {
      graphArtifactSha256 = bundle.graphArtifactSha256;
    } else if (graphArtifactSha256 !== bundle.graphArtifactSha256) {
      throw new Error(`evaluate: arm bundles under ${armsDir} disagree on graphArtifactSha256`);
    }
  }
  if (graphArtifactSha256 === undefined) {
    throw new Error('evaluate: no arms requested (no run directories given)');
  }
  return { ...graphs, graphArtifactSha256 };
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
  readonly ticks: number;
  readonly substeps: number;
  readonly heldOutStart: number;
  readonly heldOutCount: number;
  readonly bootstrapResamples: number;
  readonly bootstrapSeed: number;
}

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

export const parseEvaluateArgs = (argv: readonly string[]): EvaluateArgs => {
  let graphPath: string | undefined;
  let armsDir: string | undefined;
  const runDirs: string[] = [];
  let outDir = 'public/data';
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
      bootstrapSeed = requirePositiveInt(flag, argv[index + 1]);
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
  const runs = args.runDirs.map((dir) => readRunDir(dir));
  const seenKeys = new Set<string>();
  for (const run of runs) {
    const key = `${run.config.arm}|${run.config.trainerSeed}`;
    if (seenKeys.has(key)) {
      throw new Error(`evaluate: duplicate run for arm "${run.config.arm}" trainerSeed ${run.config.trainerSeed}`);
    }
    seenKeys.add(key);
  }

  const neededArms = new Set(runs.map((run) => run.config.arm));
  const graphIdentity = computeGraphIdentity(args.graphPath);
  const armsDir = args.armsDir ?? resolve(DEFAULT_ARMS_OUT_DIR, graphIdentity.graphArtifactSha256);
  const armGraphs = loadArmGraphs(armsDir, neededArms);
  if (armGraphs.graphArtifactSha256 !== graphIdentity.graphArtifactSha256) {
    throw new Error(
      `evaluate: arm bundles under ${armsDir} were exported from a different graph ` +
        `(bundle graphArtifactSha256=${armGraphs.graphArtifactSha256}) than --graph resolves to ` +
        `(${graphIdentity.graphArtifactSha256})`
    );
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

  const warnings: string[] = [];

  const armsReport: Record<string, unknown> = {};
  for (const arm of armNames) {
    const graph = armGraphs[arm]!;
    const D = armD.get(arm)!;

    const authoredScores = heldOutSeeds.map((seed) => computeLeftScore(arm, 'authored', null, null, seed));
    const authoredStats = conditionStats(authoredScores, args.bootstrapResamples, rng);

    const replicaMap = armReplicas.get(arm) ?? new Map<number, LoadedRun>();
    const replicasReport: Record<string, unknown> = {};
    for (const [trainerSeed, run] of replicaMap) {
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

    armsReport[arm] = { D, authored: authoredStats, replicas: replicasReport };
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

      const replicasA = armReplicas.get(armA) ?? new Map<number, LoadedRun>();
      const replicasB = armReplicas.get(armB) ?? new Map<number, LoadedRun>();
      for (const [trainerSeed, runA] of replicasA) {
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

    const bioReplicas = armReplicas.get('biological') ?? new Map<number, LoadedRun>();
    const rewiredReplicas = armReplicas.get('rewired') ?? new Map<number, LoadedRun>();
    for (const [trainerSeed, bioRun] of bioReplicas) {
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

  // Shipped artifact (replica 0 = trainerSeed 101) for every requested arm.
  const shippedWeights: Partial<Record<ArmName, ReadoutWeights>> = {};
  let shippedD: number | null = null;
  let shippedH: number | null = null;
  let complete = armNames.length > 0;
  for (const arm of armNames) {
    const run = (armReplicas.get(arm) ?? new Map<number, LoadedRun>()).get(SHIPPED_TRAINER_SEED);
    if (!run) {
      complete = false;
      warnings.push(`trained-readout-v1.json not written: no replica 0 (trainerSeed ${SHIPPED_TRAINER_SEED}) run for arm "${arm}".`);
      continue;
    }
    shippedWeights[arm] = run.weights;
    if (shippedD === null) shippedD = run.config.D;
    else if (shippedD !== run.config.D) complete = false;
    if (shippedH === null) shippedH = run.config.H;
    else if (shippedH !== run.config.H) complete = false;
  }
  if (armNames.length === 0) complete = false;

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
    for (const arm of armNames) {
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

    const armBundleSha256: Record<string, string> = {};
    for (const arm of armNames) {
      const bundlePath = resolve(armsDir, `${arm}.json`);
      const bundle = JSON.parse(readFileSync(bundlePath, 'utf8')) as SerializedArmBundle;
      armBundleSha256[arm] = bundle.sha256;
    }

    const manifest = {
      version: 1,
      artifactSha256,
      graphArtifactSha256: graphIdentity.graphArtifactSha256,
      armBundleSha256,
      D: shippedD,
      H: shippedH,
      parameterCount: readoutParameterCount(shippedD, shippedH),
      shippedReplicaTrainerSeed: SHIPPED_TRAINER_SEED,
      heldOutSeeds: { start: args.heldOutStart, count: args.heldOutCount }
    };
    writeFileSync(resolve(outDir, 'trained-readout-v1.manifest.json'), JSON.stringify(manifest));
    artifactWritten = true;
  }

  return { warnings, reportPath, artifactWritten };
};

const main = (): void => {
  const args = parseEvaluateArgs(process.argv.slice(2));
  try {
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
