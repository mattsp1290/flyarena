import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import type { ConnectomeGraph } from '../../src/lib/connectome/format';
import { OBSERVATION_CHANNELS } from '../../src/lib/arena/sensors';
import { resolveArenaTask } from '../../src/lib/arena/tasks';
import type { ArenaConfig } from '../../src/lib/arena/config';
import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { buildSeedTrace, parseGraphArtifactBytes, TRACE_SUBSTEPS } from '../training/export-traces';
import { collectRepoRelativeDependencies, computeSourceIdentitySha256 } from '../lib/import-graph';
import { quantileIndex } from './null-stats';
import { parseArenaTaskArg } from './arena-task-fields';

/**
 * `.agents/plans/task-generality/02-authored-runs.md`'s WP2 step 3
 * ("clearance measurement"): per non-default arena task, measured
 * distributions of the wall-clearance and food-distance observation
 * channels over biological authored traces, for `docs/task-generality-report.md`'s
 * limitations section to quote directly (`00-overview.md`: "WP2 measures
 * each variant's actual clearance distribution from traces, and the report
 * quotes the measured values, not a formula").
 *
 * **Plan deviation, recorded here because it drove this file's existence**:
 * `02-authored-runs.md` names `export-traces.ts --arena-task t` as the tool
 * for this step, at "1800 ticks" over "10 seeds". That flag combination
 * does not exist and cannot be made to produce this measurement:
 * `export-traces.ts`'s `--arena-task` mode is hardcoded to
 * `TASK_TRACE_SEED = 1` (one seed, not ten), `TASK_TRACE_TICKS = 200` (not
 * 1800, to stay inside its own committed-fixture byte budget), and always
 * traces the synthetic `createTraceGraph()` fixture rather than the real
 * biological connectome — its CLI explicitly throws if `--arena-task` is
 * combined with `--graph`/`--substeps` (see `parseArgs`'s own guard),
 * because that mode's whole job is producing small, committed
 * `tests/fixtures/golden/tasks/<id>/` parity fixtures, not an ad hoc
 * statistical sample. This script is the real biological-graph, N-seed,
 * arbitrary-ticks tool the measurement actually needs. It does not
 * duplicate `export-traces.ts`'s tracing logic: `buildSeedTrace`/
 * `parseGraphArtifactBytes` are already-exported, pure, generic-over-config
 * functions there, and this script imports and reuses them directly rather
 * than re-implementing the world/model step loop (`parseGraphArtifactBytes`
 * was split out of that file's `loadGraphArtifact` for this script's own
 * single-read-hash-and-parse need — see its doc comment there).
 */

export const CLEARANCE_CHANNELS = ['foodDistance', 'forwardClearance', 'leftClearance', 'rightClearance'] as const;
export type ClearanceChannel = (typeof CLEARANCE_CHANNELS)[number];

const CHANNEL_INDEX: Record<ClearanceChannel, number> = Object.fromEntries(
  CLEARANCE_CHANNELS.map((channel) => {
    const index = OBSERVATION_CHANNELS.indexOf(channel);
    if (index === -1) throw new Error(`task-clearance: "${channel}" is not an OBSERVATION_CHANNELS entry`);
    return [channel, index];
  })
) as Record<ClearanceChannel, number>;

export interface ChannelPercentiles {
  readonly p5: number;
  readonly p50: number;
  readonly p95: number;
  readonly max: number;
  readonly n: number;
  /**
   * Fraction of samples exactly at `1` (`sensors.ts`'s `clamp01` clamps every
   * one of `CLEARANCE_CHANNELS` to `[0, 1]`, so `1` unambiguously means "at
   * or beyond `sensorRange`", not merely "a large but finite distance").
   * Maintainability-review finding (task-generality WP2 dual review): a
   * capped p95/max of exactly `1` reads as a real measured distance unless
   * the report also states how much of the mass is saturated -- most acute
   * in `sparse-food` (an enlarged arena over the same `sensorRange`), where
   * `00-overview.md` itself predicts `foodDistance` saturates "far more
   * often". Reported so `docs/task-generality-report.md` can quote it
   * directly next to a saturated percentile instead of only the capped
   * value.
   */
  readonly fractionSaturated: number;
}

/**
 * 5th/50th/95th percentile plus maximum of `values`, using `null-stats.ts`'s
 * `quantileIndex` (the same low-tail-floor / high-tail-ceil-minus-one
 * convention every other predeclared-statistics module in this study uses)
 * rather than a separately-invented percentile rule. Throws on a non-finite
 * sample (`NaN`/`Infinity`) rather than letting it silently corrupt the sort
 * order and every percentile downstream of it (the same class of guard
 * `intervention-report-validation.ts`'s `assertFiniteScores` applies to its
 * own inputs).
 */
export const computeChannelPercentiles = (values: readonly number[]): ChannelPercentiles => {
  if (values.length === 0) throw new Error('task-clearance: computeChannelPercentiles requires at least one value');
  const badIndex = values.findIndex((value) => !Number.isFinite(value));
  if (badIndex !== -1) {
    throw new Error(`task-clearance: computeChannelPercentiles: values[${badIndex}] is not a finite number`);
  }
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const saturatedCount = sorted.reduce((count, value) => (value === 1 ? count + 1 : count), 0);
  return {
    p5: sorted[quantileIndex(n, 0.05)],
    p50: sorted[quantileIndex(n, 0.5)],
    p95: sorted[quantileIndex(n, 0.95)],
    max: sorted[n - 1],
    n,
    fractionSaturated: saturatedCount / n
  };
};

/**
 * Runs `buildSeedTrace` once per seed in `[seedStart, seedStart + seedCount)`
 * against `graph` under `resolved.config`, and returns every recorded tick's
 * value for each of `CLEARANCE_CHANNELS`, pooled across all seeds (seed
 * order, then tick order within a seed — deterministic, matching
 * `buildSeedTrace`'s own per-tick loop). Pure and side-effect free: no file
 * I/O, so a caller (or a test) can exercise this against any in-memory
 * graph without touching disk.
 *
 * Cross-checks every trace's own recorded `configFingerprint`
 * (`createWorld`'s, threaded through unchanged by `buildSeedTrace`) against
 * `resolved.fingerprint` — a maintainability-review finding (task-generality
 * WP2 dual review): without this, a future edit that accidentally dropped
 * `{ arenaConfig }` from the `buildSeedTrace` call below would silently
 * measure the *default* task's arena for every task id, and no existing
 * check would notice (`TaskClearanceReport.arenaTask` comes from `resolved`
 * directly, not from anything the trace itself measured).
 */
export const collectClearanceSamples = (
  graph: Readonly<ConnectomeGraph>,
  resolved: { readonly config: Readonly<ArenaConfig>; readonly fingerprint: string },
  seedStart: number,
  seedCount: number,
  ticks: number,
  substeps: number
): Record<ClearanceChannel, number[]> => {
  const samples: Record<ClearanceChannel, number[]> = {
    foodDistance: [],
    forwardClearance: [],
    leftClearance: [],
    rightClearance: []
  };
  for (let seed = seedStart; seed < seedStart + seedCount; seed += 1) {
    const trace = buildSeedTrace(graph, 'task-clearance', seed, ticks, substeps, { arenaConfig: resolved.config });
    if (trace.configFingerprint !== resolved.fingerprint) {
      throw new Error(
        `task-clearance: seed ${seed}'s trace was recorded under a different arena config than requested ` +
          `(trace ${trace.configFingerprint}, requested ${resolved.fingerprint})`
      );
    }
    for (const observation of trace.observations) {
      for (const channel of CLEARANCE_CHANNELS) {
        samples[channel].push(observation[CHANNEL_INDEX[channel]]);
      }
    }
  }
  return samples;
};

/**
 * `scripts/lib/import-graph.ts`'s code-identity scheme (`regime-check.ts`'s
 * `regimeProducer`, `repertoire-report.ts`'s `repertoireNullProducer`,
 * `intervention-artifact.ts`'s own `producer` all follow this exact shape)
 * -- a maintainability-review finding: this file was the one report script
 * in this family that omitted it, so no downstream consumer (WP4) could
 * verify a `clearance.json` was produced by the code at the commit being
 * published.
 */
export interface TaskClearanceProducer {
  readonly script: string;
  readonly sourceSha256: string;
  readonly dependencies: readonly string[];
}

export interface TaskClearanceReport {
  readonly version: 1;
  readonly arenaTask: { readonly id: string; readonly fingerprint: string };
  readonly sensorRange: number;
  readonly graph: { readonly path: string; readonly sha256: string };
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
  readonly channels: Record<ClearanceChannel, ChannelPercentiles>;
  readonly producer: TaskClearanceProducer;
  readonly host: { readonly arch: string; readonly node: string };
}

/**
 * `taskClearanceProducer()`'s own doc comment for why this exists. Computed
 * once and passed in by the caller (`runTaskClearance`, or a test) rather
 * than read from `import.meta.url`/the filesystem inside this function --
 * this keeps `buildTaskClearanceReport` itself pure (deterministic given
 * its arguments, no I/O of its own), matching every other pure function in
 * this file.
 */
export const taskClearanceProducer = (): TaskClearanceProducer => {
  const entryFile = fileURLToPath(import.meta.url);
  const dependencies = collectRepoRelativeDependencies(entryFile, repoRoot);
  return {
    script: 'scripts/null/task-clearance.ts',
    sourceSha256: computeSourceIdentitySha256(repoRoot, dependencies),
    dependencies
  };
};

/** Pure: takes an already-loaded/validated graph, its already-hashed bytes, and its already-computed producer/host info, so this has no filesystem or `process.*` dependency of its own (see `runTaskClearance` for the I/O-performing caller). */
export const buildTaskClearanceReport = (
  graph: Readonly<ConnectomeGraph>,
  graphPath: string,
  graphSha256: string,
  arenaTaskId: string | undefined,
  seedStart: number,
  seedCount: number,
  ticks: number,
  substeps: number,
  producer: Readonly<TaskClearanceProducer>,
  host: { readonly arch: string; readonly node: string }
): TaskClearanceReport => {
  const resolved = resolveArenaTask(arenaTaskId);
  const samples = collectClearanceSamples(graph, resolved, seedStart, seedCount, ticks, substeps);
  const channels = Object.fromEntries(
    CLEARANCE_CHANNELS.map((channel) => [channel, computeChannelPercentiles(samples[channel])])
  ) as Record<ClearanceChannel, ChannelPercentiles>;
  return {
    version: 1,
    arenaTask: { id: resolved.id, fingerprint: resolved.fingerprint },
    sensorRange: resolved.config.sensorRange,
    graph: { path: graphPath, sha256: graphSha256 },
    seeds: { start: seedStart, count: seedCount },
    ticks,
    substeps,
    channels,
    producer,
    host
  };
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

const DEFAULT_GRAPH = resolve(repoRoot, 'public/data/malecns-arena-v1.bin.gz');
const DEFAULT_SEED_START = 1;
const DEFAULT_SEED_COUNT = 10;
const DEFAULT_TICKS = 1800;

export interface TaskClearanceArgs {
  readonly graph: string;
  readonly arenaTask?: string;
  readonly seedStart: number;
  readonly seedCount: number;
  readonly ticks: number;
  readonly substeps: number;
  readonly out: string;
}

export const parseTaskClearanceArgs = (argv: readonly string[]): TaskClearanceArgs => {
  let graph = DEFAULT_GRAPH;
  let arenaTask: string | undefined;
  let seedStart = DEFAULT_SEED_START;
  let seedCount = DEFAULT_SEED_COUNT;
  let ticks = DEFAULT_TICKS;
  let substeps = TRACE_SUBSTEPS;
  let out: string | undefined;

  let i = 0;
  while (i < argv.length) {
    const flag = argv[i];
    if (flag === '--graph') {
      graph = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--arena-task') {
      arenaTask = parseArenaTaskArg(requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--seed-start') {
      seedStart = requireNonNegativeInt(flag, argv[i + 1]);
      i += 2;
    } else if (flag === '--seed-count') {
      seedCount = requirePositiveInt(flag, argv[i + 1]);
      i += 2;
    } else if (flag === '--ticks') {
      ticks = requirePositiveInt(flag, argv[i + 1]);
      i += 2;
    } else if (flag === '--substeps') {
      substeps = requirePositiveInt(flag, argv[i + 1]);
      i += 2;
    } else if (flag === '--out') {
      out = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  // Derived from the resolved task id (so an omitted --arena-task derives
  // ".../default/clearance.json", consistent with every other flag here
  // rather than a special-cased "required" rule) -- never overridden when
  // --out was passed explicitly.
  const resolvedId = resolveArenaTask(arenaTask).id;
  const resolvedOut = out ?? resolve(repoRoot, 'training', 'runs', 'tasks', resolvedId, 'clearance.json');

  return { graph, arenaTask, seedStart, seedCount, ticks, substeps, out: resolvedOut };
};

/**
 * A path under `repoRoot` is recorded relative to it (e.g.
 * `public/data/malecns-arena-v1.bin.gz`), so `clearance.json` stays
 * reproducible across checkouts/machines rather than embedding one
 * machine's absolute filesystem prefix (a maintainability-review finding).
 * A `--graph` path outside `repoRoot` (a genuinely out-of-tree fixture) is
 * left absolute, since there is no shorter reproducible form for it.
 */
const displayGraphPath = (graphPath: string): string => {
  const rel = relative(repoRoot, graphPath);
  return rel.startsWith('..') ? graphPath : rel;
};

export const runTaskClearance = (
  args: Readonly<TaskClearanceArgs>
): { readonly out: string; readonly report: TaskClearanceReport } => {
  // Read once: both the sha256 and the parsed graph come from this exact
  // buffer, so the recorded hash always describes the bytes actually
  // traced (see `parseGraphArtifactBytes`'s own doc comment for the TOCTOU
  // this closes — a dual-review finding).
  const graphBytes = readFileSync(args.graph);
  const graph = parseGraphArtifactBytes(graphBytes);

  const report = buildTaskClearanceReport(
    graph,
    displayGraphPath(args.graph),
    sha256Hex(graphBytes),
    args.arenaTask,
    args.seedStart,
    args.seedCount,
    args.ticks,
    args.substeps,
    taskClearanceProducer(),
    { arch: process.arch, node: process.version }
  );

  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, JSON.stringify(report));
  return { out: args.out, report };
};

const main = (): void => {
  try {
    const args = parseTaskClearanceArgs(process.argv.slice(2));
    const { out, report } = runTaskClearance(args);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `task-clearance: wrote ${out} (arenaTask=${report.arenaTask.id} seeds=${report.seeds.start}..` +
        `${report.seeds.start + report.seeds.count - 1} ticks=${report.ticks})`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`task-clearance failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
