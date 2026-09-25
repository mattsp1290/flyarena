import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { parseGraphBinary } from '../../src/lib/connectome/format';
import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { runShardedEvaluation } from '../null/null-evaluate';

/**
 * `.agents/plans/lesion-atlas/02-atlas-computation.md`'s WP2 driver:
 * scores every single-neuron lesion (plus one unlesioned baseline) for the
 * biological graph and the shipped rewired-seed-0 control, on the same
 * held-out condition the null studies use (authored decoder, opponent
 * parked, seeds `30001..30100`, `T=1800`, `K=NEURAL_SUBSTEPS_PER_TICK`),
 * sharded across `node:child_process.fork`ed copies of `atlas-worker.ts`.
 *
 * Reuse, not reimplementation, of the fork/IPC sharding mechanism: this
 * calls `scripts/null/null-evaluate.ts`'s generic
 * `runShardedEvaluation<Task, Result, Message>` directly, the same way
 * `null-trained-evaluate.ts` does, pinning `AtlasWorkerTask`/
 * `AtlasSeedResult`/`AtlasWorkerMessage` (this file's own task/result/
 * message shapes -- redeclared rather than imported from
 * `scripts/null/null-worker.ts`, since an atlas task has no decoder-variant
 * or graph-mode concept, only a `lesionIndex`) as its three type
 * parameters. `runShardedEvaluation`'s own thrown errors are still
 * hardcoded with a `null-evaluate:` prefix (it is, after all, that
 * module's function), re-prefixed below.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

const DEFAULT_MANIFEST = resolve(repoRoot, 'public/data/malecns-arena-v1.manifest.json');
// Exported so atlas-report.ts's shipped-artifact guard can require a run to
// actually match this condition (not merely cover every neuron) before
// letting it overwrite the shipped public/data/lesion-atlas-v1.json -- the
// two files would otherwise be free to drift apart on what "shipped-grade"
// means.
export const DEFAULT_HELD_OUT_START = 30001;
export const DEFAULT_HELD_OUT_COUNT = 100;
export const DEFAULT_TICKS = 1800;
const DEFAULT_SHARDS = 18;
const DEFAULT_OUT = resolve(repoRoot, 'training/runs/lesion/atlas-raw.json');

/** The atlas's two graphs. */
export type AtlasGraphKey = 'biological' | 'rewiredSeed0';

/** CLI spelling (`--graphs biological,rewired-seed0`, the plan's own wording) for each `AtlasGraphKey`. */
const CLI_GRAPH_NAME: Readonly<Record<AtlasGraphKey, string>> = {
  biological: 'biological',
  rewiredSeed0: 'rewired-seed0'
};

const ALL_GRAPH_KEYS: readonly AtlasGraphKey[] = ['biological', 'rewiredSeed0'];

const parseGraphKeys = (raw: string): readonly AtlasGraphKey[] => {
  const requested = raw.split(',').map((s) => s.trim());
  const keys: AtlasGraphKey[] = [];
  for (const name of requested) {
    const found = ALL_GRAPH_KEYS.find((key) => CLI_GRAPH_NAME[key] === name);
    if (!found) {
      throw new Error(`--graphs: unknown graph "${name}" (expected "biological" and/or "rewired-seed0")`);
    }
    if (keys.includes(found)) throw new Error(`--graphs: "${name}" listed more than once`);
    keys.push(found);
  }
  if (keys.length === 0) throw new Error('--graphs must name at least one graph');
  return keys;
};

// ---------------------------------------------------------------------------
// Task/result/message shapes -- this module's own, independent of
// scripts/null/null-worker.ts's Null* types (see this module's doc comment).
// ---------------------------------------------------------------------------

export interface AtlasWorkerTask {
  readonly graphId: string;
  readonly path: string;
  readonly expectedSha256: string;
  readonly heldOutSeeds: readonly number[];
  readonly ticks: number;
  /** `null` means the unlesioned baseline for this graph; otherwise the single neuron index to silence for the whole episode. */
  readonly lesionIndex: number | null;
}

export interface AtlasSeedResult {
  readonly seed: number;
  readonly movementScore: number;
  readonly foodPickups: number;
  readonly hazardContacts: number;
}

export interface AtlasWorkerResultMessage {
  readonly type: 'result';
  readonly graphId: string;
  readonly results: readonly AtlasSeedResult[];
}

export interface AtlasWorkerErrorMessage {
  readonly type: 'error';
  readonly graphId: string;
  readonly message: string;
}

export type AtlasWorkerMessage = AtlasWorkerResultMessage | AtlasWorkerErrorMessage;

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

interface ArenaManifestShape {
  readonly artifact: string;
  readonly binarySha256: string;
  /** The *gzip* sha256 of the biological artifact -- what `malecns-arena-v1.positions.json`'s own `graphSha256` field is built against (`src/lib/experiment/assets.ts`'s `loadPositions`: `positions.graphSha256 !== manifest.gzipSha256`), not the decompressed `binarySha256`. Recorded per graph below so `atlas-report.ts` can reuse the same cross-check without re-deriving it. */
  readonly gzipSha256: string;
  readonly neuronCount: number;
  readonly rewiredArms?: {
    readonly seed0?: { readonly artifact: string; readonly binarySha256: string; readonly gzipSha256: string };
  };
}

const readManifest = (path: string): ArenaManifestShape => {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as Partial<ArenaManifestShape>;
  if (typeof parsed.artifact !== 'string' || typeof parsed.binarySha256 !== 'string' || typeof parsed.gzipSha256 !== 'string') {
    throw new Error(`atlas-evaluate: ${path} is missing artifact/binarySha256/gzipSha256`);
  }
  if (typeof parsed.neuronCount !== 'number' || !Number.isInteger(parsed.neuronCount) || parsed.neuronCount <= 0) {
    throw new Error(`atlas-evaluate: ${path} is missing a positive integer neuronCount`);
  }
  return parsed as ArenaManifestShape;
};

export interface GraphSpec {
  readonly key: AtlasGraphKey;
  readonly path: string;
  readonly expectedSha256: string;
  /** The artifact's gzip sha256 -- see `ArenaManifestShape.gzipSha256`'s doc comment for why this is recorded separately from `expectedSha256` (the decompressed sha). */
  readonly gzipSha256: string;
}

const graphSpecFor = (key: AtlasGraphKey, manifest: ArenaManifestShape, graphsDir: string): GraphSpec => {
  if (key === 'biological') {
    return {
      key,
      path: resolve(graphsDir, manifest.artifact),
      expectedSha256: manifest.binarySha256,
      gzipSha256: manifest.gzipSha256
    };
  }
  const seed0 = manifest.rewiredArms?.seed0;
  if (!seed0) {
    throw new Error('atlas-evaluate: manifest has no rewiredArms.seed0 entry -- cannot score "rewired-seed0"');
  }
  return { key, path: resolve(graphsDir, seed0.artifact), expectedSha256: seed0.binarySha256, gzipSha256: seed0.gzipSha256 };
};

/**
 * Verify every requested graph's decompressed sha256 against the manifest
 * before any shard is forked -- mirrors `null-evaluate.ts`'s
 * `verifyRewiredFiles`/`verifyBiologicalSource` up-front check -- and also
 * that each graph's own `neuronCount` agrees with the manifest's -- both
 * graphs feed the same lesion-index range (`buildTasks` below, sized off
 * `manifest.neuronCount` alone) and the same `positions.json` body-ID/role
 * lookup at report time, so a rewired-seed-0 artifact with a different
 * neuron count would silently mis-lesion or mis-label neurons rather than
 * fail loudly. Parses each graph once here specifically to check this; the
 * bytes are re-read and re-parsed independently by whichever worker later
 * scores that graph (`atlas-worker.ts`'s own `loadVerifiedGraphBinary`),
 * matching `null-evaluate.ts`'s existing "verify up front, workers verify
 * again independently" pattern.
 */
export const verifyGraphFiles = (specs: readonly GraphSpec[], expectedNeuronCount: number): void => {
  const mismatches: string[] = [];
  for (const spec of specs) {
    let gzipBytes: Buffer;
    try {
      gzipBytes = readFileSync(spec.path);
    } catch (error) {
      mismatches.push(`${spec.key}: cannot read ${spec.path} (${error instanceof Error ? error.message : String(error)})`);
      continue;
    }
    const binary = gunzipSync(gzipBytes);
    const actual = sha256Hex(binary);
    if (actual !== spec.expectedSha256) {
      mismatches.push(`${spec.key}: ${spec.path} decompressed sha256 ${actual} does not match manifest (${spec.expectedSha256})`);
      continue;
    }
    const parsedNeuronCount = parseGraphBinary(binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength)).metadata
      .neuronCount;
    if (parsedNeuronCount !== expectedNeuronCount) {
      mismatches.push(
        `${spec.key}: ${spec.path} has neuronCount ${parsedNeuronCount}, but the manifest's neuronCount is ${expectedNeuronCount}`
      );
    }
  }
  if (mismatches.length > 0) {
    throw new Error(`atlas-evaluate: ${mismatches.length} graph file(s) failed verification:\n${mismatches.join('\n')}`);
  }
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface AtlasEvaluateArgs {
  readonly graphs: readonly AtlasGraphKey[];
  readonly manifest: string;
  readonly heldOutStart: number;
  readonly heldOutCount: number;
  readonly ticks: number;
  readonly shards: number;
  readonly out: string;
  /**
   * Calibration-only escape hatch, not part of the plan's production CLI
   * surface: restricts each requested graph's lesion sweep to indices
   * `0..maxLesions-1` instead of every neuron, so the WP2 acceptance gate
   * ("a calibration of 10 lesions projects the full run's wall time") can
   * be measured with the real CLI, the real artifacts, and the real
   * per-episode cost -- including fork/IPC overhead -- rather than a
   * separate ad-hoc script. Left unset, behavior is exactly the plan's
   * spec (every neuron, both graphs). The baseline task is never skipped.
   */
  readonly maxLesions?: number;
}

export const parseAtlasEvaluateArgs = (argv: readonly string[]): AtlasEvaluateArgs => {
  let graphs: readonly AtlasGraphKey[] = ALL_GRAPH_KEYS;
  let manifest = DEFAULT_MANIFEST;
  let heldOutStart = DEFAULT_HELD_OUT_START;
  let heldOutCount = DEFAULT_HELD_OUT_COUNT;
  let ticks = DEFAULT_TICKS;
  let shards = DEFAULT_SHARDS;
  let out = DEFAULT_OUT;
  let maxLesions: number | undefined;

  let index = 0;
  while (index < argv.length) {
    const flag = argv[index];
    if (flag === '--graphs') {
      graphs = parseGraphKeys(requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--manifest') {
      manifest = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--held-out-start') {
      heldOutStart = requireNonNegativeInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--held-out-count') {
      heldOutCount = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--ticks') {
      ticks = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--shards') {
      shards = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else if (flag === '--out') {
      out = resolve(process.cwd(), requireValue(flag, argv[index + 1]));
      index += 2;
    } else if (flag === '--max-lesions') {
      maxLesions = requirePositiveInt(flag, argv[index + 1]);
      index += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (!out.endsWith('.json')) throw new Error(`--out must end with ".json" (got "${out}")`);

  return { graphs, manifest, heldOutStart, heldOutCount, ticks, shards, out, maxLesions };
};

// ---------------------------------------------------------------------------
// Task list
// ---------------------------------------------------------------------------

/** Canonical `(graph, index)` order: `ALL_GRAPH_KEYS` order, baseline before every lesion, lesion indices ascending -- never collection/completion order (see `runShardedEvaluation`'s own determinism contract). */
export const buildTasks = (
  args: Readonly<AtlasEvaluateArgs>,
  specs: ReadonlyMap<AtlasGraphKey, GraphSpec>,
  neuronCount: number
): AtlasWorkerTask[] => {
  const heldOutSeeds = Array.from({ length: args.heldOutCount }, (_, i) => args.heldOutStart + i);
  const lesionCount = args.maxLesions !== undefined ? Math.min(args.maxLesions, neuronCount) : neuronCount;
  const tasks: AtlasWorkerTask[] = [];
  for (const key of ALL_GRAPH_KEYS) {
    if (!args.graphs.includes(key)) continue;
    const spec = specs.get(key);
    if (!spec) throw new Error(`atlas-evaluate: no graph spec for "${key}"`);
    const common = {
      path: spec.path,
      expectedSha256: spec.expectedSha256,
      heldOutSeeds,
      ticks: args.ticks
    };
    tasks.push({ graphId: `${key}|baseline`, lesionIndex: null, ...common });
    for (let i = 0; i < lesionCount; i += 1) {
      tasks.push({ graphId: `${key}|${i}`, lesionIndex: i, ...common });
    }
  }
  return tasks;
};

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

export interface AtlasLesionRaw {
  readonly index: number;
  readonly movementScore: readonly number[];
}

export interface AtlasGraphRaw {
  readonly graphSha256: string;
  /** See `ArenaManifestShape.gzipSha256`'s doc comment: this is what `atlas-report.ts` cross-checks the positions sidecar's `graphSha256` field against, not `graphSha256` above (which is the decompressed sha). */
  readonly graphGzipSha256: string;
  readonly heldOutSeeds: readonly number[];
  readonly baselineMovementScore: readonly number[];
  /** Sorted by `index` ascending. */
  readonly lesion: readonly AtlasLesionRaw[];
}

export interface AtlasEvaluationRaw {
  readonly version: 1;
  readonly neuronCount: number;
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
  readonly graphs: { readonly biological?: AtlasGraphRaw; readonly rewiredSeed0?: AtlasGraphRaw };
  readonly host: { readonly arch: string; readonly node: string };
  /**
   * Present only when `--max-lesions` restricted this run (the
   * calibration-only escape hatch -- see `AtlasEvaluateArgs.maxLesions`'s
   * doc comment). Absent on every production/shipped run. This makes a
   * calibration `atlas-raw.json` self-describing as one, on top of (not
   * instead of) `atlas-report.ts`'s own `rawShippedGradeProblems` check,
   * which derives the same fact independently from the actual per-graph
   * lesion-array length rather than trusting this field.
   */
  readonly maxLesions?: number;
}

export const assembleRaw = (
  args: Readonly<AtlasEvaluateArgs>,
  specs: ReadonlyMap<AtlasGraphKey, GraphSpec>,
  neuronCount: number,
  tasks: readonly AtlasWorkerTask[],
  results: ReadonlyMap<string, readonly AtlasSeedResult[]>
): AtlasEvaluationRaw => {
  const require = (graphId: string): readonly AtlasSeedResult[] => {
    const found = results.get(graphId);
    if (!found) throw new Error(`atlas-evaluate: missing results for "${graphId}"`);
    return found;
  };

  const graphs: { biological?: AtlasGraphRaw; rewiredSeed0?: AtlasGraphRaw } = {};
  for (const key of ALL_GRAPH_KEYS) {
    if (!args.graphs.includes(key)) continue;
    const spec = specs.get(key);
    if (!spec) throw new Error(`atlas-evaluate: no graph spec for "${key}"`);
    const graphTasks = tasks.filter((task) => task.graphId.startsWith(`${key}|`));
    const baselineTask = graphTasks.find((task) => task.lesionIndex === null);
    if (!baselineTask) throw new Error(`atlas-evaluate: no baseline task for "${key}"`);
    const baseline = require(baselineTask.graphId);
    const lesion: AtlasLesionRaw[] = graphTasks
      .filter((task) => task.lesionIndex !== null)
      .map((task) => ({
        index: task.lesionIndex as number,
        movementScore: require(task.graphId).map((r) => r.movementScore)
      }));
    graphs[key] = {
      graphSha256: spec.expectedSha256,
      graphGzipSha256: spec.gzipSha256,
      heldOutSeeds: baseline.map((r) => r.seed),
      baselineMovementScore: baseline.map((r) => r.movementScore),
      lesion
    };
  }

  return {
    version: 1,
    neuronCount,
    seeds: { start: args.heldOutStart, count: args.heldOutCount },
    ticks: args.ticks,
    substeps: NEURAL_SUBSTEPS_PER_TICK,
    graphs,
    host: { arch: process.arch, node: process.version },
    ...(args.maxLesions !== undefined ? { maxLesions: args.maxLesions } : {})
  };
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

const runMetaPathFor = (outPath: string): string => {
  if (!outPath.endsWith('.json')) {
    throw new Error(`atlas-evaluate: expected a ".json" output path, got "${outPath}"`);
  }
  return `${outPath.slice(0, -'.json'.length)}.run.json`;
};

export const runAtlasEvaluate = async (
  args: Readonly<AtlasEvaluateArgs>
): Promise<{ out: string; runMetaOut: string; taskCount: number; episodeCount: number; elapsedMs: number }> => {
  const manifest = readManifest(args.manifest);
  const graphsDir = dirname(args.manifest);
  const specs = new Map<AtlasGraphKey, GraphSpec>(
    args.graphs.map((key) => [key, graphSpecFor(key, manifest, graphsDir)])
  );
  verifyGraphFiles([...specs.values()], manifest.neuronCount);

  const tasks = buildTasks(args, specs, manifest.neuronCount);
  const workerPath = fileURLToPath(new URL('./atlas-worker.ts', import.meta.url));

  const started = performance.now();
  let results: Map<string, readonly AtlasSeedResult[]>;
  try {
    results = await runShardedEvaluation<AtlasWorkerTask, AtlasSeedResult, AtlasWorkerMessage>(
      tasks,
      args.shards,
      workerPath
    );
  } catch (error) {
    // `runShardedEvaluation`'s own thrown messages are prefixed
    // "null-evaluate: ..." (it is, after all, that module's function) --
    // re-prefixed here so an atlas-run failure points an operator at this
    // script, not at the unrelated null study. `cause` preserves the
    // original error (and its stack) for anyone inspecting it
    // programmatically, even though the top-level message is rewritten.
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(message.replace(/^null-evaluate:/, 'atlas-evaluate:'), { cause: error });
  }
  const elapsedMs = performance.now() - started;
  const episodeCount = tasks.length * args.heldOutCount;
  const perEpisodeMs = elapsedMs / episodeCount;

  const raw = assembleRaw(args, specs, manifest.neuronCount, tasks, results);
  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, JSON.stringify(raw));

  const runMetaOut = runMetaPathFor(args.out);
  atomicWriteFileSync(runMetaOut, `${JSON.stringify({ shards: args.shards, elapsedMs, perEpisodeMs }, null, 2)}\n`);

  return { out: args.out, runMetaOut, taskCount: tasks.length, episodeCount, elapsedMs };
};

const main = async (): Promise<void> => {
  try {
    const args = parseAtlasEvaluateArgs(process.argv.slice(2));
    const { out, runMetaOut, taskCount, episodeCount, elapsedMs } = await runAtlasEvaluate(args);
    const perEpisodeMs = elapsedMs / episodeCount;
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `atlas-evaluate: wrote ${out} and ${runMetaOut} (${taskCount} tasks x ${args.heldOutCount} seeds = ` +
        `${episodeCount} episodes) in ${(elapsedMs / 1000).toFixed(1)}s (${perEpisodeMs.toFixed(1)} ms/episode, ` +
        `${args.shards} shards)`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`atlas-evaluate failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) void main();
