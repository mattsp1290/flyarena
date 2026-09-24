import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  outputNeuronIndices,
  readoutParameterCount,
  validateReadoutWeights,
  type ReadoutWeights
} from '../../src/lib/connectome/readout';
import type { ConnectomeGraph } from '../../src/lib/connectome/format';
import { ARM_NAMES, type ArmName } from './arms';
import {
  computeArmBundleSha256,
  computeGraphIdentity,
  deserializeArmBundle,
  expectedProvenanceKind,
  DEFAULT_ARMS_OUT_DIR,
  type ArmProvenance,
  type SerializedArmBundle
} from './export-arms';
import { runEpisode } from './episode';
import { TRACE_SUBSTEPS } from './export-traces';
import { requireFloat, requireNonNegativeInt, requirePositiveInt, requireValue } from './cli';
import { readRunDir, type LoadedRun } from './run-dir';
import { conditionRng, conditionStats, pairedStats } from './stats';
import {
  renderReportMarkdown,
  type ArmPairReport,
  type ArmReplicaReport,
  type ArmReport,
  type EvaluationReport,
  type SideBySideReport
} from './report';

/**
 * The authoritative TypeScript rescorer: the Node code the browser's own
 * readout forward pass shares (`src/lib/connectome/readout.ts`), run here
 * headlessly (`episode.ts`) over held-out seeds, for every arm/replica a
 * caller hands it. This is the one place `.agents/plans/trained-readout/00-overview.md`'s
 * "TypeScript is the authoritative evaluator; PyTorch is training-only" key
 * decision is enforced: whatever `training/`'s (WP2/WP3, GPU-side) fitness
 * says, only this script's numbers are ever published.
 *
 * The run-directory contract this evaluator consumes (`config.json` +
 * `theta_final.npy`) is defined and documented in `./run-dir.ts`; the
 * statistics (mean/median/std, seeded bootstrap CIs, paired differences,
 * `conditionRng`) live in `./stats.ts`; `docs/trained-readout-report.md`
 * generation lives in `./report.ts`. This file is the orchestrator: CLI
 * parsing, arm-bundle loading/verification, the per-arm/per-pair/side-by-side
 * evaluation loops, and writing the four artifacts.
 */

/** Replica 0 (`03-cem-training.md`: "Replica 0 is the one shipped to the browser"). */
const SHIPPED_TRAINER_SEED = 101;

const sha256Hex = (data: Uint8Array | string): string => createHash('sha256').update(data).digest('hex');

/**
 * A topological (not statistical) guarantee that this arm's readout input —
 * the gathered rates of its output-assigned neurons — is exactly zero on
 * every tick of every episode, for every possible weights/seed. True only
 * when BOTH (a) the graph has no edges at all (`metadata.edgeCount === 0`,
 * so no recurrent synaptic drive can ever reach any neuron — see
 * `runSubsteps`/`stepModel` in `src/lib/connectome/model.ts`), AND (b) none
 * of the output-assigned neurons is itself directly wired to an input
 * channel (`inputChannelIndex[neuron] < 0` for all of them — `stepModel`
 * injects external sensory drive into a neuron independently of edges when
 * it *is* channel-mapped, so a zero-edge graph alone does not guarantee a
 * zero rate for such a neuron). Under both conditions, every output-assigned
 * neuron's rate starts at 0 (`createModelState`'s zero-fill) and never
 * receives any nonzero drive from either source, so leaky integration keeps
 * it at exactly 0 for the whole episode, regardless of weight saturation,
 * fitness scale, or anything statistical — this is what makes the
 * "structurally zero" report finding (`report.ts`) a fact about the graph
 * rather than an inference from identical trained/silenced scores (review
 * finding: the prior version inferred this from score identity alone, which
 * saturated tanh/sigmoid units or coincidentally-matching actions could also
 * produce for a genuinely nonzero input).
 */
const graphGuaranteesZeroReadoutInput = (graph: Readonly<ConnectomeGraph>): boolean => {
  if (graph.metadata.edgeCount !== 0) return false;
  const indices = outputNeuronIndices(graph);
  for (let i = 0; i < indices.length; i += 1) {
    if (graph.inputChannelIndex[indices[i]] >= 0) return false;
  }
  return true;
};

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
// CLI
// ---------------------------------------------------------------------------

export interface EvaluateArgs {
  readonly graphPath?: string;
  readonly armsDir?: string;
  readonly runDirs: readonly string[];
  readonly outDir: string;
  /** True only when `--out` was actually passed, not merely defaulted (see `runEvaluate`'s public/data guard). */
  readonly outDirExplicit: boolean;
  /** Explicit override for `docs/trained-readout-report.md`'s path; see `resolveReportMdPath`. */
  readonly reportMdPath?: string;
  readonly ticks: number;
  readonly substeps: number;
  readonly heldOutStart: number;
  readonly heldOutCount: number;
  readonly bootstrapResamples: number;
  readonly bootstrapSeed: number;
  /**
   * The WP5 production-run parity gate this evaluation was run under
   * (`.agents/plans/trained-readout/05-production-run.md` step 2a /
   * acceptance: "Real-graph parity ... passed at the production K, recorded
   * in the manifest as `parity: { graphSha, K, passedAt }`"). All three or
   * none: a partial set is refused (see `parseEvaluateArgs`) rather than
   * publishing a manifest that looks like it recorded a real gate result
   * from an incomplete one.
   */
  readonly parityGraphSha256?: string;
  readonly parityK?: number;
  /** ISO-8601 timestamp the parity suite passed at (recorded by hand from the run, not computed here). */
  readonly parityPassedAt?: string;
  /**
   * One CUDA rerun of one replica's training, measured and recorded as
   * informational (never a gate) per 05's acceptance criterion: "its max-abs
   * `theta_final` diff and its TS held-out fitness difference are recorded
   * in the manifest as `gpuRerunMaxAbsDiff` / `gpuRerunFitnessDelta`".
   */
  readonly gpuRerunMaxAbsDiff?: number;
  readonly gpuRerunFitnessDelta?: number;
}

const DEFAULT_OUT_DIR = 'public/data';
const DEFAULT_REPORT_MD_PATH = 'docs/trained-readout-report.md';

/** 'E','V','A','L' as a fixed default seed; arbitrary but stable across runs. */
const DEFAULT_BOOTSTRAP_SEED = 0x4556_414c;
const DEFAULT_TICKS = 1800;
const DEFAULT_HELD_OUT_START = 30001;
const DEFAULT_HELD_OUT_COUNT = 100;
const DEFAULT_BOOTSTRAP_RESAMPLES = 10000;

export const parseEvaluateArgs = (argv: readonly string[]): EvaluateArgs => {
  let graphPath: string | undefined;
  let armsDir: string | undefined;
  const runDirs: string[] = [];
  let outDir = DEFAULT_OUT_DIR;
  let outDirExplicit = false;
  let reportMdPath: string | undefined;
  let ticks = DEFAULT_TICKS;
  let substeps = TRACE_SUBSTEPS;
  let heldOutStart = DEFAULT_HELD_OUT_START;
  let heldOutCount = DEFAULT_HELD_OUT_COUNT;
  let bootstrapResamples = DEFAULT_BOOTSTRAP_RESAMPLES;
  let bootstrapSeed = DEFAULT_BOOTSTRAP_SEED;
  let parityGraphSha256: string | undefined;
  let parityK: number | undefined;
  let parityPassedAt: string | undefined;
  let gpuRerunMaxAbsDiff: number | undefined;
  let gpuRerunFitnessDelta: number | undefined;

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
    } else if (flag === '--report-md') {
      reportMdPath = requireValue(flag, argv[index + 1]);
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
    } else if (flag === '--parity-graph-sha256') {
      parityGraphSha256 = requireValue(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--parity-k') {
      parityK = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--parity-passed-at') {
      parityPassedAt = requireValue(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--gpu-rerun-max-abs-diff') {
      gpuRerunMaxAbsDiff = requireFloat(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--gpu-rerun-fitness-delta') {
      gpuRerunFitnessDelta = requireFloat(flag, argv[index + 1]);
      index += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (runDirs.length === 0) throw new Error('--runs requires at least one run directory');

  const parityFlagsGiven = [parityGraphSha256, parityK, parityPassedAt].filter((v) => v !== undefined).length;
  if (parityFlagsGiven > 0 && parityFlagsGiven < 3) {
    throw new Error(
      '--parity-graph-sha256/--parity-k/--parity-passed-at must be passed together (all three or none), ' +
        'so the manifest never records a partial parity gate result'
    );
  }
  if (gpuRerunMaxAbsDiff !== undefined && gpuRerunMaxAbsDiff < 0) {
    throw new Error(`--gpu-rerun-max-abs-diff must be non-negative, got ${gpuRerunMaxAbsDiff}`);
  }
  const gpuRerunFlagsGiven = [gpuRerunMaxAbsDiff, gpuRerunFitnessDelta].filter((v) => v !== undefined).length;
  if (gpuRerunFlagsGiven === 1) {
    throw new Error(
      '--gpu-rerun-max-abs-diff/--gpu-rerun-fitness-delta must be passed together (both or neither): they ' +
        'come from one CUDA rerun measurement, so recording only one would publish a partial result'
    );
  }

  return {
    graphPath,
    armsDir,
    runDirs,
    outDir,
    outDirExplicit,
    reportMdPath,
    ticks,
    substeps,
    heldOutStart,
    heldOutCount,
    bootstrapResamples,
    bootstrapSeed,
    parityGraphSha256,
    parityK,
    parityPassedAt,
    gpuRerunMaxAbsDiff,
    gpuRerunFitnessDelta
  };
};

/** Whether `outDir` resolves to the real shipped `public/data` — the one definition both guards below share. */
const isDefaultOutDir = (outDir: string): boolean =>
  resolve(process.cwd(), outDir) === resolve(process.cwd(), DEFAULT_OUT_DIR);

const isDefaultReportMdPath = (reportMdPath: string): boolean =>
  resolve(process.cwd(), reportMdPath) === resolve(process.cwd(), DEFAULT_REPORT_MD_PATH);

/**
 * `docs/trained-readout-report.md`'s path, per WP4's plan text (it names
 * that exact path for the real, shipped evaluation). `--report-md`
 * overrides unconditionally. Otherwise: default to `docs/` only when `--out`
 * itself resolves to the real shipped `public/data` — i.e. only for a real
 * evaluation run, mirroring the trace-graph-mode `--out` guard in
 * `runEvaluate` below. Every other invocation (trace-graph dev mode, and
 * every test) writes the report markdown alongside `--out`'s own
 * report.json/artifact, so tests never touch `docs/`.
 */
export const resolveReportMdPath = (args: Readonly<EvaluateArgs>): string => {
  if (args.reportMdPath) return resolve(process.cwd(), args.reportMdPath);
  if (isDefaultOutDir(args.outDir)) return resolve(process.cwd(), DEFAULT_REPORT_MD_PATH);
  return resolve(process.cwd(), args.outDir, 'trained-readout-report.md');
};

// ---------------------------------------------------------------------------
// Core evaluation
// ---------------------------------------------------------------------------

export interface RunEvaluateResult {
  readonly warnings: readonly string[];
  readonly reportPath: string;
  readonly reportMdPath: string;
  readonly artifactWritten: boolean;
}

/** Core logic, separated from CLI parsing/`main` so tests can call it in-process without a subprocess. */
export const runEvaluate = (args: Readonly<EvaluateArgs>): RunEvaluateResult => {
  const graphIdentity = computeGraphIdentity(args.graphPath);

  // --parity-graph-sha256 is a caller-supplied claim ("the parity suite
  // passed against this graph"); nothing upstream cross-checks it against
  // the graph actually being evaluated. Without this, a stale or mistyped
  // sha would publish what looks like a passed real-graph parity gate for a
  // *different* graph than the one this run actually scored, with no
  // warning (review finding: both independent reviewers flagged this).
  if (args.parityGraphSha256 !== undefined && args.parityGraphSha256 !== graphIdentity.graphArtifactSha256) {
    throw new Error(
      `evaluate: --parity-graph-sha256 ${args.parityGraphSha256} does not match the graph actually being ` +
        `evaluated (graphArtifactSha256 ${graphIdentity.graphArtifactSha256}); the parity suite must be ` +
        're-run against this exact graph before its result can be recorded'
    );
  }

  // Trace-graph dev mode (no --graph) must never silently overwrite the
  // real shipped artifact: `npm run training:evaluate -- --runs <dirs>`
  // with no --graph defaults to the trace graph AND to --out public/data,
  // which would otherwise clobber the product's public/data/trained-readout-v1.*
  // with fixture output. Mirrors export-traces.ts's --out overwrite guard.
  // The same applies to the report markdown: an explicit `--report-md
  // docs/trained-readout-report.md` (or a resolved default `--report-md`
  // that happens to land there) must not let a trace-graph run clobber the
  // real shipped docs/ page with fixture numbers either.
  if (graphIdentity.graphSource !== 'artifact') {
    if (!args.outDirExplicit || isDefaultOutDir(args.outDir)) {
      throw new Error(
        'evaluate: trace-graph mode (no --graph) refuses to write to the default --out ' +
          `(${DEFAULT_OUT_DIR}); pass --out <scratch dir> explicitly, or pass --graph <artifact> ` +
          'for a real evaluation run.'
      );
    }
    if (isDefaultReportMdPath(resolveReportMdPath(args))) {
      throw new Error(
        'evaluate: trace-graph mode (no --graph) refuses to write to the default report path ' +
          `(${DEFAULT_REPORT_MD_PATH}); pass --report-md <scratch path> explicitly, or pass ` +
          '--graph <artifact> for a real evaluation run.'
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

  const armNames = ARM_NAMES.filter((arm) => armGraphs[arm]);
  const armD = new Map<ArmName, number>(armNames.map((arm) => [arm, outputNeuronIndices(armGraphs[arm]!).length]));
  const distinctD = new Set(armD.values());
  if (distinctD.size > 1) {
    throw new Error(`evaluate: D (output-neuron count) differs across loaded arm graphs under ${armsDir}`);
  }

  const heldOutSeeds = Array.from({ length: args.heldOutCount }, (_, i) => args.heldOutStart + i);
  /** Statistic-scoped bootstrap RNG; see `stats.ts`'s `conditionRng` doc comment. */
  const rngFor = (label: string): (() => number) => conditionRng(args.bootstrapSeed, label);

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

  /** `computeLeftScore` over every held-out seed, for one (arm, decoder, trainerSeed, weights) condition. */
  const scoresFor = (
    arm: ArmName,
    decoder: 'trained' | 'authored' | 'silenced',
    trainerSeed: number | null,
    weights: ReadoutWeights | null
  ): number[] => heldOutSeeds.map((seed) => computeLeftScore(arm, decoder, trainerSeed, weights, seed));

  const armReplicas = new Map<ArmName, Map<number, LoadedRun>>();
  for (const run of runs) {
    if (!armReplicas.has(run.config.arm)) armReplicas.set(run.config.arm, new Map());
    armReplicas.get(run.config.arm)!.set(run.config.trainerSeed, run);
  }
  /**
   * Ascending-by-trainerSeed entries for one arm's replicas. `armReplicas`'s
   * `Map` iterates in insertion order, which is the order run dirs appeared
   * on `--runs` — sorting here makes `armPairs`'s and `replicas`'s element
   * order independent of `--runs` argument order, so the same *set* of runs
   * always produces the same `report.json` bytes, not just the same argv.
   * (Each statistic's bootstrap CI is independently seeded via
   * `conditionRng`, so it no longer depends on iteration order or on what
   * else this invocation evaluated — see `stats.ts`'s `conditionRng` doc
   * comment.)
   */
  const sortedReplicas = (arm: ArmName): Array<[number, LoadedRun]> =>
    [...(armReplicas.get(arm) ?? new Map<number, LoadedRun>())].sort(([a], [b]) => a - b);

  const warnings: string[] = [];

  const armsReport: Record<string, ArmReport> = {};
  for (const arm of armNames) {
    const graph = armGraphs[arm]!;
    const D = armD.get(arm)!;

    const authoredScores = scoresFor(arm, 'authored', null, null);
    const authoredStats = conditionStats(authoredScores, args.bootstrapResamples, rngFor(`${arm}|authored`));

    const replicasReport: Record<string, ArmReplicaReport> = {};
    for (const [trainerSeed, run] of sortedReplicas(arm)) {
      validateReadoutWeights(run.weights, graph);
      const trainedScores = scoresFor(arm, 'trained', trainerSeed, run.weights);
      const silencedScores = scoresFor(arm, 'silenced', trainerSeed, run.weights);
      replicasReport[String(trainerSeed)] = {
        H: run.config.H,
        parameterCount: run.config.parameterCount,
        weightsSha256: run.weightsSha256,
        env: run.env,
        trained: conditionStats(trainedScores, args.bootstrapResamples, rngFor(`${arm}|trained|${trainerSeed}`)),
        silenced: conditionStats(silencedScores, args.bootstrapResamples, rngFor(`${arm}|silenced|${trainerSeed}`)),
        pairedTrainedVsAuthored: pairedStats(
          trainedScores,
          authoredScores,
          args.bootstrapResamples,
          rngFor(`${arm}|paired-trained-vs-authored|${trainerSeed}`)
        ),
        pairedTrainedVsSilenced: pairedStats(
          trainedScores,
          silencedScores,
          args.bootstrapResamples,
          rngFor(`${arm}|paired-trained-vs-silenced|${trainerSeed}`)
        )
      };
    }

    armsReport[arm] = {
      D,
      provenance: armGraphs.bundleProvenance[arm],
      armBundleSha256: armGraphs.bundleSha256[arm],
      authored: authoredStats,
      replicas: replicasReport,
      structurallyZeroInput: graphGuaranteesZeroReadoutInput(graph)
    };
  }

  const armPairs: ArmPairReport[] = [];
  for (let i = 0; i < armNames.length; i += 1) {
    for (let j = i + 1; j < armNames.length; j += 1) {
      const armA = armNames[i];
      const armB = armNames[j];

      const authoredA = scoresFor(armA, 'authored', null, null);
      const authoredB = scoresFor(armB, 'authored', null, null);
      armPairs.push({
        condition: 'authored',
        trainerSeed: null,
        armA,
        armB,
        pairedDifference: pairedStats(
          authoredA,
          authoredB,
          args.bootstrapResamples,
          rngFor(`armpair|authored|${armA}|${armB}`)
        )
      });

      const replicasB = armReplicas.get(armB) ?? new Map<number, LoadedRun>();
      for (const [trainerSeed, runA] of sortedReplicas(armA)) {
        const runB = replicasB.get(trainerSeed);
        if (!runB) continue;
        for (const condition of ['trained', 'silenced'] as const) {
          const scoresA = scoresFor(armA, condition, trainerSeed, runA.weights);
          const scoresB = scoresFor(armB, condition, trainerSeed, runB.weights);
          armPairs.push({
            condition,
            trainerSeed,
            armA,
            armB,
            pairedDifference: pairedStats(
              scoresA,
              scoresB,
              args.bootstrapResamples,
              rngFor(`armpair|${condition}|${armA}|${armB}|${trainerSeed}`)
            )
          });
        }
      }
    }
  }

  const sideBySide: SideBySideReport[] = [];
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
      left: conditionStats(leftAuthored, args.bootstrapResamples, rngFor('sidebyside|authored|left')),
      right: conditionStats(rightAuthored, args.bootstrapResamples, rngFor('sidebyside|authored|right')),
      pairedLeftMinusRight: pairedStats(
        leftAuthored,
        rightAuthored,
        args.bootstrapResamples,
        rngFor('sidebyside|authored|paired')
      )
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
        left: conditionStats(leftTrained, args.bootstrapResamples, rngFor(`sidebyside|trained|${trainerSeed}|left`)),
        right: conditionStats(
          rightTrained,
          args.bootstrapResamples,
          rngFor(`sidebyside|trained|${trainerSeed}|right`)
        ),
        pairedLeftMinusRight: pairedStats(
          leftTrained,
          rightTrained,
          args.bootstrapResamples,
          rngFor(`sidebyside|trained|${trainerSeed}|paired`)
        )
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
  const shippedConfig: Partial<Record<ArmName, LoadedRun['config']>> = {};
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
    shippedConfig[arm] = run.config;
    if (run.env === null) {
      warnings.push(`arm "${arm}" replica 0 run "${run.dir}" has no env.json; manifest omits its device/torch provenance.`);
    }
    shippedD ??= run.config.D;
    shippedH ??= run.config.H;
  }

  // CEM hyperparameters + training-seed RNG policy: every arm's replica-0
  // run is required to share an identical config except --arm/--replica-seed
  // (`05-production-run.md` step 4), so this reports one representative and
  // flags any arm whose recorded hyperparameters actually disagree with it
  // (or has none recorded at all), rather than silently publishing whichever
  // arm happened to load last. The representative is the first ARM_NAMES
  // entry that actually HAS a recorded config — not merely the first arm
  // seen — so an arm with an older/tiny run dir lacking these fields (all
  // `undefined`) is skipped when picking the baseline rather than adopted as
  // one, and never silently discards a later arm's real, present config
  // (review finding: adopting the first-seen arm unconditionally could pick
  // an empty baseline, "differ" against every real config that follows, and
  // then null the whole block out because the adopted baseline itself was
  // empty — dropping every arm's real data with a misleading warning).
  const CEM_CONFIG_FIELDS = [
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
  const isEmptyCemConfig = (candidate: Readonly<Record<string, unknown>>): boolean =>
    Object.values(candidate).every((value) => value === undefined);

  // Two passes, deliberately: a single forward pass over ARM_NAMES can only
  // compare each arm against a baseline established by an *earlier* arm, so
  // an early arm with no recorded config (e.g. "biological", first in
  // ARM_NAMES order) would never get flagged even when a *later* arm (e.g.
  // "rewired") does have one — the loop reaches the empty arm before any
  // baseline exists to contrast it against. Computing every arm's candidate
  // first, then picking the baseline as the first *non-empty* one regardless
  // of position, makes the warning (and the published `training` block)
  // independent of which arm happens to come first in ARM_NAMES.
  const cemCandidates: Partial<Record<ArmName, Record<string, unknown>>> = {};
  for (const arm of ARM_NAMES) {
    const config = shippedConfig[arm];
    if (!config) continue;
    const candidate: Record<string, unknown> = {};
    for (const field of CEM_CONFIG_FIELDS) candidate[field] = config[field];
    cemCandidates[arm] = candidate;
  }
  const armsWithShippedConfig = ARM_NAMES.filter((arm) => cemCandidates[arm] !== undefined);
  const trainingBlockSourceArm =
    armsWithShippedConfig.find((arm) => !isEmptyCemConfig(cemCandidates[arm]!)) ?? null;
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

  // Real-graph (production) evaluation must record the WP5 parity gate that
  // is supposed to have passed before any production CEM run
  // (`05-production-run.md` step 2a / acceptance) — flagged here, not
  // silently omitted, when the shipped artifact is actually being written.
  if (graphIdentity.graphSource === 'artifact' && complete) {
    const parityGiven =
      args.parityGraphSha256 !== undefined && args.parityK !== undefined && args.parityPassedAt !== undefined;
    if (!parityGiven) {
      warnings.push(
        'manifest omits the parity block: pass --parity-graph-sha256/--parity-k/--parity-passed-at to ' +
          'record the WP5 real-graph parity gate result.'
      );
    }
  }

  const report: EvaluationReport = {
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

  const reportMdPath = resolveReportMdPath(args);
  mkdirSync(dirname(reportMdPath), { recursive: true });
  writeFileSync(reportMdPath, renderReportMarkdown(report));

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
      evaluatorGitRev: evaluatorGitRev(),
      /**
       * CEM hyperparameters (population/elites/generations/etc.) and the
       * training-seed RNG policy, from the shipped replicas' `config.json`
       * (`00-overview.md`/`03-cem-training.md`); `null` when no shipped run
       * recorded these fields (e.g. an older run dir).
       */
      training: trainingBlock,
      /**
       * The WP5 real-graph parity gate result (`05-production-run.md` step
       * 2a), from `--parity-graph-sha256`/`--parity-k`/`--parity-passed-at`;
       * `null` when not passed (see the "manifest omits the parity block"
       * warning above).
       */
      parity:
        args.parityGraphSha256 !== undefined && args.parityK !== undefined && args.parityPassedAt !== undefined
          ? { graphSha: args.parityGraphSha256, K: args.parityK, passedAt: args.parityPassedAt }
          : null,
      /** One CUDA rerun of one replica's training, informational only (never a gate); `null` when not measured/passed. */
      gpuRerunMaxAbsDiff: args.gpuRerunMaxAbsDiff ?? null,
      gpuRerunFitnessDelta: args.gpuRerunFitnessDelta ?? null
    };
    writeFileSync(resolve(outDir, 'trained-readout-v1.manifest.json'), JSON.stringify(manifest));
    artifactWritten = true;
  }

  return { warnings, reportPath, reportMdPath, artifactWritten };
};

const main = (): void => {
  try {
    const args = parseEvaluateArgs(process.argv.slice(2));
    const result = runEvaluate(args);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `evaluate: wrote ${result.reportPath} and ${result.reportMdPath}` +
        `${result.artifactWritten ? ', and trained-readout-v1.{json,manifest.json}' : ''}` +
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
