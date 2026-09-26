import { fileURLToPath } from 'node:url';
import { mkdirSync, readFileSync, realpathSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

import { decodeAction } from '../../src/lib/arena/actions';
import type { ArenaConfig } from '../../src/lib/arena/config';
import { observeAgent } from '../../src/lib/arena/sensors';
import { ARENA_TASK_IDS, resolveArenaTask, type ArenaTaskId } from '../../src/lib/arena/tasks';
import { createWorld, stepWorld } from '../../src/lib/arena/world';
import type { AgentId, WorldState } from '../../src/lib/arena/types';
import {
  parseGraphBinary,
  validateGraph,
  type ConnectomeGraph
} from '../../src/lib/connectome/format';
import {
  createModelState,
  createOutputBuffer,
  createStepScratch,
  runSubsteps
} from '../../src/lib/connectome/model';
import {
  createReadoutOutput,
  createReadoutScratch,
  outputNeuronIndices,
  readoutForward,
  validateReadoutWeights,
  type ReadoutWeights
} from '../../src/lib/connectome/readout';
import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { mulberry32 } from '../../src/lib/random/mulberry32';
import { createTraceGraph } from '../../tests/fixtures/trace-graph';
import { requireValue } from './cli';

/**
 * Exports deterministic golden traces from the existing TypeScript arena +
 * rate model, so a later PyTorch port (WP2,
 * `.agents/plans/trained-readout/02-gpu-port-and-parity.md`) can be checked
 * against them. Run with no flags to regenerate the committed fixtures under
 * `tests/fixtures/golden/` (`npm run training:traces`); `tests/unit/golden-traces.test.ts`
 * regenerates every committed file in-process (importing `buildGoldenFiles`/
 * `buildSeedTrace` below, not shelling out to this file) and compares it
 * against the committed bytes, so a TS behavior change forces a deliberate
 * trace refresh.
 *
 * Cross-architecture determinism: `stepModel`/`aggregateOutputs`
 * (`connectome/model.ts`) use only `+`/`-`/`*`, which IEEE-754 guarantees
 * bit-identical across architectures, but the observation/physics path
 * (`arena/sensors.ts`'s `Math.atan2`/`Math.sin`/`Math.cos`/`Math.hypot`, and
 * `arena/world.ts`'s `Math.sin`/`Math.cos`/`Math.hypot` in `createWorld`/
 * `stepWorld`) calls transcendental `Math.*` functions, whose rounding is
 * implementation-defined per the ECMAScript spec and not guaranteed
 * bit-identical across architectures (V8's libm differs between x86_64 and
 * arm64). The committed fixtures here were generated on `linux-arm64`;
 * `golden-traces.test.ts` therefore does a strict byte-for-byte comparison
 * only when `process.arch` matches that, and a tight float-tolerance
 * structural comparison otherwise (measured divergence and the tolerance it
 * justifies: `tests/fixtures/cross-arch-tolerance.ts`; see also
 * `docs/architecture.md`'s "Determinism scope"). This exporter itself is
 * unaffected — it always writes whatever this process's `Math.*`
 * implementation actually produces; only the *test*'s comparison strategy
 * is architecture-aware. Regenerate committed fixtures (`npm run
 * training:traces`) on `linux-arm64` to keep the byte-exact path
 * meaningful; regenerating on another architecture would silently change
 * `GOLDEN_GENERATING_ARCH`'s assumption in `cross-arch-tolerance.ts` and
 * must update that constant too.
 *
 * `--graph`, `--substeps`, and `--out` let a later work package (WP5) export
 * traces from a real compiled graph artifact at the real closed-loop
 * substep count, written outside this directory (e.g.
 * `training/runs/traces/`) rather than overwriting the committed fixtures.
 * `--graph` accepts either a raw `.bin` graph or a gzip-compressed one
 * (detected by the gzip magic bytes, not the filename) — the one real
 * artifact currently in this repo, `public/data/malecns-arena-v1.bin.gz`, is
 * gzip-compressed.
 *
 * `--include-world` adds a per-tick `worldAfter` column (full post-step
 * world state: agent pose/velocity, foods, hazards, hazard-contact state,
 * and score) to every seed file, at full round-trip double precision. This
 * is what WP2's PyTorch parity port needs to teacher-force `step_world` and
 * check food/hazard contact events exactly
 * (`.agents/plans/trained-readout/02-gpu-port-and-parity.md`'s "Parity
 * tolerances" table) — `initialWorld` alone (tick 0 only) cannot satisfy
 * that table for ticks 1..N. It is off by default: the committed fixtures
 * under `tests/fixtures/golden/` stay within the ≤ 200 KB budget precisely
 * by *not* recording a per-tick world column (see `TRACE_TICKS`'s comment).
 * WP2 must generate its own full-coverage trace on demand, uncommitted:
 * `npm run training:traces -- --include-world --out training/runs/<dir>`
 * (or any other gitignored `--out` path) — never into
 * `tests/fixtures/golden/`, which the overwrite guard below refuses.
 *
 * Golden files are written as compact (non-indented) JSON: the per-tick
 * arrays dominate file size, and the ≤ 200 KB committed budget
 * (`.agents/plans/trained-readout/01-golden-traces-and-readout-contract.md`)
 * leaves no room for pretty-printing whitespace.
 */

/**
 * Neural substeps per world tick, recorded in every exported file. Aliases
 * the real closed-loop's own substep count (`src/lib/connectome/constants.ts`)
 * rather than restating the number here, so the two can never silently
 * drift apart — the golden traces this exporter produces are only a
 * faithful record of the real closed loop's dynamics if this matches it
 * exactly.
 */
export const TRACE_SUBSTEPS = NEURAL_SUBSTEPS_PER_TICK;

/**
 * Ticks per exported trace.
 *
 * Deviation from the work package's stated default (300): even after
 * dropping per-tick world state (see `SeedTraceFile`'s comment) and scoping
 * the readout case to one seed, the remaining per-tick columns
 * (observation x8, rate x24, aggregated outputs x3, decoded action x3, full
 * round-trip-precision doubles per this file's own requirement) measure
 * roughly 766 bytes/tick/seed once written; at 300 ticks x 4 seeds that is
 * roughly 900 KB, still over 4x the ≤ 200 KB committed budget in
 * `.agents/plans/trained-readout/01-golden-traces-and-readout-contract.md`.
 * That budget note explicitly anticipates this and authorizes "a documented
 * reduction consistent with the plan's intent, e.g. fewer committed seeds,
 * rather than silently truncating." 60 ticks keeps all four seeds (needed
 * for WP2's "all four golden seeds" parity coverage) and exactly covers the
 * 60-tick window WP2's free-running check needs, while fitting the byte
 * budget (see the size report `npm run training:traces` prints; the readout
 * case below is also scoped to one seed for the same reason).
 */
export const TRACE_TICKS = 60;

/** Seeds exported by default (committed fixtures). */
export const TRACE_SEEDS: readonly number[] = [1, 2, 3, 12345];

/** Fixed hidden width for the committed golden readout case. */
const READOUT_CASE_HIDDEN_SIZE = 4;

const DEFAULT_OUT_DIR = 'tests/fixtures/golden';
/** `graphId` used for the committed fixtures; exported so tests reuse it instead of a copied literal. */
export const DEFAULT_GRAPH_ID = 'trace-graph';

/**
 * `.agents/plans/task-generality/01-task-plumbing.md`'s WP1: `--arena-task`'s
 * own committed output directory, one per non-`default` task id, alongside
 * (not inside) the default fixtures — never `DEFAULT_OUT_DIR` itself, so an
 * arena-task export can never collide with or overwrite the default traces.
 */
const TASK_TRACE_OUT_DIR = (id: string): string => `${DEFAULT_OUT_DIR}/tasks/${id}`;
/**
 * Fewer ticks and a single seed than the default export's `TRACE_TICKS`/
 * `TRACE_SEEDS` — `training/tests/test_tasks.py`'s per-task parity check
 * needs only one seed's worth of rate/output/action parity per task, not the
 * default's four-seed coverage.
 *
 * Deviation from `.agents/plans/task-generality/01-task-plumbing.md`'s
 * stated 300 ticks, for the same reason `TRACE_TICKS` itself already
 * deviates from its own plan's stated default (see that constant's doc
 * comment): measured at 300 ticks, one seed's trace plus the shared
 * `<graphId>.json` totalled ~214 KB, over the ≤ 200 KB per-task budget
 * (`01-task-plumbing.md`'s own change-surface row). 200 ticks measures
 * ~131–146 KB across all four task variants (comfortable margin), and this
 * suite's parity checks (rate/output/action agreement per tick) don't
 * depend on any particular tick count.
 */
export const TASK_TRACE_TICKS = 200;
export const TASK_TRACE_SEED = 1;

export interface CliArgs {
  graphPath?: string;
  substeps: number;
  outDir: string;
  /** True only when `--out` was actually passed, not merely defaulted. */
  outDirExplicit: boolean;
  includeWorld: boolean;
  /** `--arena-task <id>`. `undefined` or `'default'` behaves exactly as if this flag were absent. */
  arenaTask?: ArenaTaskId;
}

/**
 * Canonicalize a path for the overwrite-guard comparison below: `resolve()`
 * always, plus `realpathSync` when the path already exists (so a symlink or
 * a `..`-laden alias that resolves to the same directory as
 * `DEFAULT_OUT_DIR` is still caught, not just a textually-identical path). A
 * not-yet-existing `--out` directory can't be realpath'd, so `resolve()`'s
 * normalization is what's compared for it.
 */
const canonicalizePath = (path: string): string => {
  const resolved = resolve(process.cwd(), path);
  try {
    return realpathSync(resolved);
  } catch {
    return resolved;
  }
};

export const parseArgs = (argv: readonly string[]): CliArgs => {
  let graphPath: string | undefined;
  let substeps = TRACE_SUBSTEPS;
  let outDir = DEFAULT_OUT_DIR;
  let outDirExplicit = false;
  let substepsExplicit = false;
  let includeWorld = false;
  let arenaTask: ArenaTaskId | undefined;

  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    if (flag === '--graph') {
      graphPath = requireValue(flag, argv[index + 1]);
      index += 1;
    } else if (flag === '--substeps') {
      const value = requireValue(flag, argv[index + 1]);
      const parsed = Number(value);
      if (!Number.isInteger(parsed) || parsed <= 0) {
        throw new Error(`--substeps must be a positive integer, got "${value}"`);
      }
      substeps = parsed;
      substepsExplicit = true;
      index += 1;
    } else if (flag === '--out') {
      outDir = requireValue(flag, argv[index + 1]);
      outDirExplicit = true;
      index += 1;
    } else if (flag === '--include-world') {
      includeWorld = true;
    } else if (flag === '--arena-task') {
      const value = requireValue(flag, argv[index + 1]);
      if (!(ARENA_TASK_IDS as readonly string[]).includes(value)) {
        throw new Error(`--arena-task must be one of ${ARENA_TASK_IDS.join(', ')} (got "${value}")`);
      }
      arenaTask = value as ArenaTaskId;
      index += 1;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (arenaTask !== undefined && arenaTask !== 'default' && (graphPath !== undefined || substepsExplicit || includeWorld)) {
    throw new Error('--arena-task cannot be combined with --graph/--substeps/--include-world');
  }
  // A non-default task's own canonical, committed output directory
  // (`tests/fixtures/golden/tasks/<id>/`) -- unlike --graph/--substeps/
  // --include-world below, an explicit --out is not required for this case,
  // since this location is exactly as canonical (and as safe from
  // overwriting the default fixtures) as DEFAULT_OUT_DIR is for the default
  // task. `arenaTask` is narrowed directly here (not through a separately
  // reused boolean) so this line never needs a non-null assertion.
  if (arenaTask !== undefined && arenaTask !== 'default' && !outDirExplicit) {
    outDir = TASK_TRACE_OUT_DIR(arenaTask);
  }

  // A non-default graph, substep count, world-state column, or arena task
  // producing output that lands in the committed fixture directory (by
  // accident, with no --out, or on purpose, with an --out that still
  // resolves to it) would silently corrupt tests/fixtures/golden/ with
  // non-canonical data. `--arena-task` alone never needs an explicit --out
  // (it already redirects to its own safe directory above), so only the
  // "must pass --out explicitly" branch is scoped to the other three flags —
  // but the "does --out resolve to the default directory" branch below must
  // cover every non-default export, --arena-task included, or an explicit
  // `--out tests/fixtures/golden` would silently overwrite the real default
  // fixtures with task-variant data (a dual-review finding).
  const graphSubstepsWorldNonDefault = graphPath !== undefined || substepsExplicit || includeWorld;
  const nonDefaultExport = graphSubstepsWorldNonDefault || (arenaTask !== undefined && arenaTask !== 'default');
  if (graphSubstepsWorldNonDefault && !outDirExplicit) {
    throw new Error(
      '--graph/--substeps/--include-world change what gets exported; pass --out explicitly so ' +
        `the result cannot land in the committed ${DEFAULT_OUT_DIR}/ directory by accident`
    );
  }
  if (nonDefaultExport && outDirExplicit && canonicalizePath(outDir) === canonicalizePath(DEFAULT_OUT_DIR)) {
    throw new Error(
      '--graph/--substeps/--include-world/--arena-task change what gets exported; --out ' +
        `("${outDir}") resolves to the committed ${DEFAULT_OUT_DIR}/ directory, which would ` +
        'overwrite the committed golden fixtures with non-canonical data. Pass a different ' +
        '--out path (e.g. a gitignored directory such as training/runs/, or omit --out to use ' +
        "--arena-task's own tests/fixtures/golden/tasks/<id>/ directory)."
    );
  }

  return { graphPath, substeps, outDir, outDirExplicit, includeWorld, arenaTask };
};

const GZIP_MAGIC = [0x1f, 0x8b];
const isGzip = (buffer: Readonly<Buffer>): boolean =>
  buffer.length >= 2 && buffer[0] === GZIP_MAGIC[0] && buffer[1] === GZIP_MAGIC[1];

/**
 * Read a binary graph artifact from disk, matching format.ts's expected
 * ArrayBuffer input. Detects a gzip-compressed artifact by its magic bytes
 * (not the filename) and decompresses it first: `public/data/*.bin.gz`
 * (produced by `scripts/data/compile.py`'s `binfmt.write_gzip_deterministic`)
 * is gzip, not a raw `.bin`.
 */
export const loadGraphArtifact = (path: string): ConnectomeGraph => {
  const raw = readFileSync(path);
  const buffer = isGzip(raw) ? gunzipSync(raw) : raw;
  const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  return parseGraphBinary(arrayBuffer);
};

/**
 * Derive a `graphId` from a graph artifact path. `basename(path,
 * extname(path))` alone strips only the last extension, so
 * `malecns-arena-v1.bin.gz` would mangle to `graphId = "malecns-arena-v1.bin"`
 * — the stray `.bin` then propagates into every `SerializedArmBundle.graphId`
 * (`export-arms.ts`) and every trace file's `graphId` field. Strip a
 * trailing `.gz` first, then the remaining extension (`.bin`, or whatever
 * else a non-gzip artifact uses), so `foo.bin.gz` and `foo.bin` both yield
 * `graphId = "foo"`. Not a correctness gate — nothing compares `graphId` for
 * equality; `graphArtifactSha256` is the real identity — but it is recorded
 * provenance and should not be silently wrong.
 */
export const graphIdFromPath = (path: string): string => {
  const withoutGz = path.endsWith('.gz') ? path.slice(0, -'.gz'.length) : path;
  return basename(withoutGz, extname(withoutGz));
};

interface SerializedGraph {
  metadata: ConnectomeGraph['metadata'];
  biologicalIds: string[];
  presynapticOffsets: number[];
  postsynapticIndices: number[];
  contactMagnitudes: number[];
  presynapticSigns: number[];
  inputChannelIndex: number[];
  inputWeight: number[];
  outputPopulationIndex: number[];
  outputWeight: number[];
}

/** Serialize a graph's arrays once (JSON has no typed-array/BigInt support). */
const serializeGraph = (graph: Readonly<ConnectomeGraph>): SerializedGraph => ({
  metadata: graph.metadata,
  biologicalIds: Array.from(graph.biologicalIds, (id) => id.toString()),
  presynapticOffsets: Array.from(graph.presynapticOffsets),
  postsynapticIndices: Array.from(graph.postsynapticIndices),
  contactMagnitudes: Array.from(graph.contactMagnitudes),
  presynapticSigns: Array.from(graph.presynapticSigns),
  inputChannelIndex: Array.from(graph.inputChannelIndex),
  inputWeight: Array.from(graph.inputWeight),
  outputPopulationIndex: Array.from(graph.outputPopulationIndex),
  outputWeight: Array.from(graph.outputWeight)
});

/**
 * `initialWorld` (below) records only one snapshot per seed file (the
 * `WorldState` `createWorld` produces, before any tick), but is still
 * written columnar-without-ids for the same reason as the per-tick trace
 * columns: no repeated per-entity key/id text. Entity order is
 * `createWorld`'s own fixed construction order and is stable for the
 * lifetime of a run: `agentIds` gives the order `agent*` rows are in
 * (`['left', 'right']`); foods are `food-0..food-{foodCount-1}`; hazards are
 * `hazard-0..hazard-{hazardCount-1}` (`world.ts`'s `createWorld`).
 * `previousPosition`/`previousHeading` are safe to omit at any tick:
 * `stepWorld` overwrites both unconditionally at the start of every tick
 * regardless of their entering value, so they never carry information
 * forward. `activeHazardIds` is different — `world.ts`'s `processContacts`
 * reads the *previous* tick's `activeHazardIds` to detect a new contact
 * (transition into overlap), so it does carry information forward in
 * general; it is safe to omit here only because `createWorld` always starts
 * every agent with `activeHazardIds: []` (`world.ts`'s `createAgent`), which
 * is exactly the value at the one tick (0) this interface is ever used for.
 * `radius` fields are omitted too: they equal the fixed arena config's
 * per-entity radius for the lifetime of a run.
 */
interface SerializedWorld {
  tick: number;
  timeSeconds: number;
  rngState: number;
  agentIds: AgentId[];
  agentPositions: number[][];
  agentVelocities: number[][];
  agentHeadings: number[];
  /** Per agent, `[foodPickups, hazardContacts, distanceTravelled, movementScore]`. */
  agentScores: number[][];
  foodPositions: number[][];
  foodRespawns: number[];
  hazardPositions: number[][];
  hazardVelocities: number[][];
}

const serializeWorld = (world: Readonly<WorldState>): SerializedWorld => ({
  tick: world.tick,
  timeSeconds: world.timeSeconds,
  rngState: world.rngState,
  agentIds: world.agents.map((agent) => agent.id),
  agentPositions: world.agents.map((agent) => [agent.position.x, agent.position.z]),
  agentVelocities: world.agents.map((agent) => [agent.velocity.x, agent.velocity.z]),
  agentHeadings: world.agents.map((agent) => agent.heading),
  agentScores: world.agents.map((agent) => [
    agent.score.foodPickups,
    agent.score.hazardContacts,
    agent.score.distanceTravelled,
    agent.score.movementScore
  ]),
  foodPositions: world.foods.map((food) => [food.position.x, food.position.z]),
  foodRespawns: world.foods.map((food) => food.respawns),
  hazardPositions: world.hazards.map((hazard) => [hazard.position.x, hazard.position.z]),
  hazardVelocities: world.hazards.map((hazard) => [hazard.velocity.x, hazard.velocity.z])
});

/**
 * Per-tick (post-`stepWorld`) world state, used only by the opt-in
 * `worldAfter` column below (`buildSeedTrace`'s `includeWorld` option) — not
 * by `initialWorld`, which stays exactly `SerializedWorld` so the committed
 * fixtures' bytes are unaffected by this type existing. Extends
 * `SerializedWorld` with `agentActiveHazardIds`: unlike `initialWorld`
 * (always tick 0, where `activeHazardIds` is always `[]` by construction —
 * see `SerializedWorld`'s comment), a post-step tick can have non-empty
 * `activeHazardIds`, and `world.ts`'s `processContacts` reads the
 * *previous* tick's value to detect a new contact. Omitting it here would
 * leave WP2 unable to reproduce hazard-contact events exactly when
 * teacher-forcing from a recorded prior tick rather than tick 0.
 */
interface SerializedWorldTick extends SerializedWorld {
  agentActiveHazardIds: string[][];
}

const serializeWorldTick = (world: Readonly<WorldState>): SerializedWorldTick => ({
  ...serializeWorld(world),
  agentActiveHazardIds: world.agents.map((agent) => [...agent.activeHazardIds])
});

interface SeedTraceFile {
  formatVersion: 1;
  /** `${graphId}.json` (next to this file) is the one serialized graph every seed file shares. */
  graphId: string;
  seed: number;
  ticks: number;
  substeps: number;
  /**
   * `world.configFingerprint` (`src/lib/arena/config.ts`'s
   * `createArenaConfigFingerprint`): identifies exactly which
   * `ARENA_CONFIG` values this trace was produced under, so a consumer
   * replaying `initialWorld` + `actions` through its own `stepWorld` (or a
   * port of it) can check it is using a matching config before trusting the
   * replay, rather than silently assuming `ARENA_CONFIG` never changed.
   */
  configFingerprint: string;
  /**
   * World state before tick 0's step: the "prior state" for teacher-forcing
   * tick 0, and (with `actions`, replayed through `stepWorld`) sufficient to
   * exactly reconstruct every subsequent tick's world state. A per-tick
   * `world` column is deliberately not recorded — it would roughly double
   * this file's size for state that `initialWorld` + `actions` already
   * determines exactly (`stepWorld` is a pure, deterministic function).
   */
  initialWorld: SerializedWorld;
  /**
   * Per-tick columns, each of length `ticks` (row `i` is tick `i`), rather
   * than an array of per-tick objects: this is the dominant cost in the
   * committed byte budget, and columnar arrays carry no repeated per-tick
   * key text the way `{ observation: [...], rateAfter: [...], ... }[]`
   * would. `left`'s 8-channel observation of the world state entering each
   * tick; `left`'s per-neuron rates after `substeps` substeps;
   * `aggregateOutputs`'s per-population sums; `decodeAction(outputs)` in
   * `OUTPUT_POPULATION` order (thrust, yaw, brake) — the action actually
   * fed into `stepWorld` for `left`, `right` always receiving the zero
   * action (parked opponent, per
   * `.agents/plans/trained-readout/00-overview.md`'s "opponent slot"
   * decision).
   */
  observations: number[][];
  ratesAfter: number[][];
  outputs: number[][];
  actions: number[][];
  /**
   * Post-step world state for every tick (row `i` is the state *after*
   * tick `i`'s `stepWorld` call), present only when `buildSeedTrace` is
   * called with `{ includeWorld: true }`. Absent (not merely `undefined`)
   * in the committed default export, so the committed fixtures' bytes are
   * unaffected by this field's existence. See this file's doc comment for
   * why and how WP2 should generate this.
   */
  worldAfter?: SerializedWorldTick[];
}

export interface BuildSeedTraceOptions {
  /** Record a per-tick `worldAfter` column. Default `false` (budget). */
  includeWorld?: boolean;
  /**
   * `.agents/plans/task-generality/01-task-plumbing.md`'s WP1: the arena
   * config this trace's `createWorld` uses. Defaults to `ARENA_CONFIG`
   * (`createWorld`'s own default) — omitting it is unchanged from before
   * this field existed, so the committed default fixtures stay byte-identical.
   */
  arenaConfig?: Readonly<ArenaConfig>;
}

/**
 * Run one deterministic seed: the authored decode path
 * (`aggregateOutputs` -> `decodeAction`) drives `left`; `right` always
 * receives the zero action.
 */
export const buildSeedTrace = (
  graph: Readonly<ConnectomeGraph>,
  graphId: string,
  seed: number,
  ticks: number,
  substeps: number,
  options: BuildSeedTraceOptions = {}
): SeedTraceFile => {
  const includeWorld = options.includeWorld ?? false;
  let world = createWorld(seed, options.arenaConfig);
  const configFingerprint = world.configFingerprint;
  const initialWorld = serializeWorld(world);

  const state = createModelState(graph);
  const scratch = createStepScratch(graph);
  const outputs = createOutputBuffer(graph);

  const observations: number[][] = [];
  const ratesAfter: number[][] = [];
  const outputColumns: number[][] = [];
  const actions: number[][] = [];
  const worldAfter: SerializedWorldTick[] = [];

  for (let tick = 0; tick < ticks; tick += 1) {
    const observation = observeAgent(world, 'left');
    runSubsteps(graph, state, scratch, observation, substeps, outputs);
    const decodedAction = decodeAction(Array.from(outputs));

    observations.push(Array.from(observation));
    ratesAfter.push(Array.from(state.rate));
    outputColumns.push(Array.from(outputs));
    actions.push([decodedAction.thrust, decodedAction.yaw, decodedAction.brake]);

    world = stepWorld(world, {
      left: [decodedAction.thrust, decodedAction.yaw, decodedAction.brake],
      right: [0, 0, 0]
    });

    if (includeWorld) worldAfter.push(serializeWorldTick(world));
  }

  return {
    formatVersion: 1,
    graphId,
    seed,
    ticks,
    substeps,
    configFingerprint,
    initialWorld,
    observations,
    ratesAfter,
    outputs: outputColumns,
    actions,
    ...(includeWorld ? { worldAfter } : {})
  };
};

/**
 * `ReadoutWeights` as committed JSON. `w1`/`b1`/`w2`/`b2` are plain
 * `number[]`, not `Float32Array` — `JSON.stringify(Float32Array)` does not
 * produce a JSON array (typed arrays are not `Array.isArray`, so
 * `JSON.stringify` falls back to serializing them as an index-keyed plain
 * object, e.g. `{"0":1.5,"1":2}`), which would silently hand a PyTorch
 * loader dictionaries instead of vectors and would fail this repo's own
 * `validateReadoutWeights` on reload (its `.length` checks see `undefined`).
 */
interface SerializedReadoutWeights {
  inputSize: number;
  hiddenSize: number;
  w1: number[];
  b1: number[];
  w2: number[];
  b2: number[];
}

const serializeReadoutWeights = (weights: Readonly<ReadoutWeights>): SerializedReadoutWeights => ({
  inputSize: weights.inputSize,
  hiddenSize: weights.hiddenSize,
  w1: Array.from(weights.w1),
  b1: Array.from(weights.b1),
  w2: Array.from(weights.w2),
  b2: Array.from(weights.b2)
});

interface ReadoutCaseFile {
  formatVersion: 1;
  graphId: string;
  weights: SerializedReadoutWeights;
  /**
   * `readoutForward` outputs for every recorded tick's `rateAfter`, for one
   * designated seed (the same seed `tests/unit/golden-traces.test.ts`
   * regenerates). Scoped to one seed rather than all committed seeds to
   * stay inside the ≤ 200 KB committed budget; see `TRACE_TICKS`'s comment.
   */
  seed: number;
  outputs: number[][];
}

/** Fixed seeded random `ReadoutWeights` (H = 4), shared by every committed seed trace. */
const buildReadoutWeights = (graph: Readonly<ConnectomeGraph>): Readonly<ReadoutWeights> => {
  const inputSize = outputNeuronIndices(graph).length;
  const hiddenSize = READOUT_CASE_HIDDEN_SIZE;
  const random = mulberry32(0x5245_4144); // 'READ', arbitrary fixed constant
  const uniform = (): number => random() * 2 - 1;

  const w1 = Float32Array.from({ length: hiddenSize * inputSize }, uniform);
  const b1 = Float32Array.from({ length: hiddenSize }, uniform);
  const w2 = Float32Array.from({ length: 3 * hiddenSize }, uniform);
  const b2 = Float32Array.from({ length: 3 }, uniform);

  const weights: ReadoutWeights = { inputSize, hiddenSize, w1, b1, w2, b2 };
  // buildReadoutCase's own reference producer exercises the same validator
  // a consumer would run when loading this file back.
  return validateReadoutWeights(weights, graph);
};

const buildReadoutCase = (
  graph: Readonly<ConnectomeGraph>,
  graphId: string,
  trace: Readonly<SeedTraceFile>
): ReadoutCaseFile => {
  const weights = buildReadoutWeights(graph);
  const indices = outputNeuronIndices(graph);
  const scratch = createReadoutScratch(weights.hiddenSize);
  const out = createReadoutOutput();

  const outputs = trace.ratesAfter.map((rateAfter) => {
    const rate = Float32Array.from(rateAfter);
    readoutForward(weights, rate, indices, scratch, out);
    return Array.from(out);
  });

  return {
    formatVersion: 1,
    graphId,
    weights: serializeReadoutWeights(weights),
    seed: trace.seed,
    outputs
  };
};

/**
 * One golden file: its committed-relative filename and the value that
 * becomes its JSON contents. Not exported: nothing imports it by name
 * (`buildGoldenFiles`'s return type is inferred structurally at its one
 * call site, `tests/unit/golden-traces.test.ts`).
 */
interface GoldenFile {
  fileName: string;
  value: unknown;
}

/**
 * Build every golden file this exporter produces for one `(graph, graphId)`
 * pair, in the same order `main()` writes them. Exported so
 * `tests/unit/golden-traces.test.ts` can regression-check every committed
 * file (not just one seed) against the same code path `main()` uses,
 * instead of re-deriving a parallel notion of "the golden files."
 */
export const buildGoldenFiles = (
  graph: Readonly<ConnectomeGraph>,
  graphId: string,
  substeps: number,
  options: BuildSeedTraceOptions = {}
): GoldenFile[] => {
  const traces = TRACE_SEEDS.map((seed) =>
    buildSeedTrace(graph, graphId, seed, TRACE_TICKS, substeps, options)
  );
  // Scoped to the first committed seed (1), the same one golden-traces.test.ts regenerates.
  const readoutCase = buildReadoutCase(graph, graphId, traces[0]);

  return [
    { fileName: `${graphId}.json`, value: serializeGraph(graph) },
    ...traces.map((trace) => ({ fileName: `${graphId}-seed-${trace.seed}.json`, value: trace })),
    { fileName: `${graphId}-readout.json`, value: readoutCase }
  ];
};

/**
 * `--arena-task <id>` golden files (task-generality WP1): one seed
 * (`TASK_TRACE_SEED`), `TASK_TRACE_TICKS` ticks, no readout case (the
 * per-task parity suite checks world/rate/output/action parity only — see
 * `TASK_TRACE_TICKS`'s own doc comment for the byte-budget reasoning).
 * Exported so `tests/unit/export-traces.test.ts` can regression-check the
 * committed per-task files the same way `golden-traces.test.ts` does for
 * `buildGoldenFiles`.
 */
export const buildTaskGoldenFiles = (
  graph: Readonly<ConnectomeGraph>,
  graphId: string,
  substeps: number,
  arenaConfig: Readonly<ArenaConfig>
): GoldenFile[] => {
  const trace = buildSeedTrace(graph, graphId, TASK_TRACE_SEED, TASK_TRACE_TICKS, substeps, { arenaConfig });
  return [
    { fileName: `${graphId}.json`, value: serializeGraph(graph) },
    { fileName: `${graphId}-seed-${trace.seed}.json`, value: trace }
  ];
};

const writeJson = (path: string, value: unknown): number => {
  mkdirSync(dirname(path), { recursive: true });
  const contents = JSON.stringify(value);
  writeFileSync(path, contents);
  return Buffer.byteLength(contents);
};

const main = (): void => {
  const args = parseArgs(process.argv.slice(2));

  const graph = args.graphPath ? loadGraphArtifact(args.graphPath) : createTraceGraph();
  validateGraph(graph);
  const graphId = args.graphPath ? graphIdFromPath(args.graphPath) : DEFAULT_GRAPH_ID;

  const outDir = resolve(process.cwd(), args.outDir);
  const isNonDefaultTask = args.arenaTask !== undefined && args.arenaTask !== 'default';
  const files = isNonDefaultTask
    ? buildTaskGoldenFiles(graph, graphId, args.substeps, resolveArenaTask(args.arenaTask).config)
    : buildGoldenFiles(graph, graphId, args.substeps, { includeWorld: args.includeWorld });

  let totalBytes = 0;
  for (const { fileName, value } of files) {
    totalBytes += writeJson(resolve(outDir, fileName), value);
  }

  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
  console.log(
    `Wrote ${files.length} file(s) to ${outDir} (${totalBytes} bytes total): ` +
      files.map((f) => f.fileName).join(', ')
  );
};

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    main();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`export-traces failed: ${message}`);
    process.exit(1);
  }
}
