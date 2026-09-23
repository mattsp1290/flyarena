import { fileURLToPath } from 'node:url';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, resolve } from 'node:path';

import { decodeAction } from '../../src/lib/arena/actions';
import { observeAgent } from '../../src/lib/arena/sensors';
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
import { createTraceGraph, mulberry32 } from '../../tests/fixtures/trace-graph';

/**
 * Exports deterministic golden traces from the existing TypeScript arena +
 * rate model, so a later PyTorch port (WP2,
 * `.agents/plans/trained-readout/02-gpu-port-and-parity.md`) can be checked
 * against them. Run with no flags to regenerate the committed fixtures under
 * `tests/fixtures/golden/` (`npm run training:traces`); `tests/unit/golden-traces.test.ts`
 * regenerates seed 1 in-process (importing `buildSeedTrace` below, not
 * shelling out to this file) and deep-equals it against the committed file,
 * so a TS behavior change forces a deliberate trace refresh.
 *
 * `--graph`, `--substeps`, and `--out` let a later work package (WP5) export
 * traces from a real compiled graph artifact at the real closed-loop
 * substep count, written outside this directory (e.g.
 * `training/runs/traces/`) rather than overwriting the committed fixtures.
 *
 * Golden files are written as compact (non-indented) JSON: the per-tick
 * arrays dominate file size, and the ≤ 200 KB committed budget
 * (`.agents/plans/trained-readout/01-golden-traces-and-readout-contract.md`)
 * leaves no room for pretty-printing whitespace.
 */

/** Neural substeps per world tick. Until `flyarena-bb45` fixes the closed
 * loop's real substep count, this is a documented constant, recorded in
 * every exported file. */
export const TRACE_SUBSTEPS = 4;

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

interface CliArgs {
  graphPath?: string;
  substeps: number;
  outDir: string;
  /** True only when `--out` was actually passed, not merely defaulted. */
  outDirExplicit: boolean;
}

/** A missing option value must not silently consume the next flag instead. */
const requireValue = (flag: string, value: string | undefined): string => {
  if (value === undefined || value.startsWith('--')) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
};

const parseArgs = (argv: readonly string[]): CliArgs => {
  let graphPath: string | undefined;
  let substeps = TRACE_SUBSTEPS;
  let outDir = DEFAULT_OUT_DIR;
  let outDirExplicit = false;
  let substepsExplicit = false;

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
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  // A non-default graph or substep count producing output that lands in the
  // committed fixture directory by accident (no --out given) would silently
  // corrupt tests/fixtures/golden/ with non-canonical data.
  if ((graphPath !== undefined || substepsExplicit) && !outDirExplicit) {
    throw new Error(
      '--graph/--substeps change what gets exported; pass --out explicitly so the result ' +
        `cannot land in the committed ${DEFAULT_OUT_DIR}/ directory by accident`
    );
  }

  return { graphPath, substeps, outDir, outDirExplicit };
};

/** Read a binary graph artifact from disk, matching format.ts's expected ArrayBuffer input. */
const loadGraphArtifact = (path: string): ConnectomeGraph => {
  const buffer = readFileSync(path);
  const arrayBuffer = buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
  return parseGraphBinary(arrayBuffer);
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
  substeps: number
): SeedTraceFile => {
  let world = createWorld(seed);
  const configFingerprint = world.configFingerprint;
  const initialWorld = serializeWorld(world);

  const state = createModelState(graph);
  const scratch = createStepScratch(graph);
  const outputs = createOutputBuffer(graph);

  const observations: number[][] = [];
  const ratesAfter: number[][] = [];
  const outputColumns: number[][] = [];
  const actions: number[][] = [];

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
    actions
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

/** One golden file: its committed-relative filename and the value that becomes its JSON contents. */
export interface GoldenFile {
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
  substeps: number
): GoldenFile[] => {
  const traces = TRACE_SEEDS.map((seed) => buildSeedTrace(graph, graphId, seed, TRACE_TICKS, substeps));
  // Scoped to the first committed seed (1), the same one golden-traces.test.ts regenerates.
  const readoutCase = buildReadoutCase(graph, graphId, traces[0]);

  return [
    { fileName: `${graphId}.json`, value: serializeGraph(graph) },
    ...traces.map((trace) => ({ fileName: `${graphId}-seed-${trace.seed}.json`, value: trace })),
    { fileName: `${graphId}-readout.json`, value: readoutCase }
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
  const graphId = args.graphPath
    ? basename(args.graphPath, extname(args.graphPath))
    : DEFAULT_GRAPH_ID;

  const outDir = resolve(process.cwd(), args.outDir);
  const files = buildGoldenFiles(graph, graphId, args.substeps);

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
