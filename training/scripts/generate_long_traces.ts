import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';

import { createWorld } from '../../src/lib/arena/world';
import { validateGraph, type ConnectomeGraph } from '../../src/lib/connectome/format';
import { createTraceGraph } from '../../tests/fixtures/trace-graph';
import { buildSeedTrace, type BuildSeedTraceOptions } from '../../scripts/training/export-traces';

/**
 * Generates longer, wider-seed-coverage `--include-world` traces than
 * `scripts/training/export-traces.ts`'s CLI exposes, for
 * `training/tests/test_world_event_coverage.py` and the batched-world
 * parity tests.
 *
 * Why a separate script instead of extending `export-traces.ts`'s CLI: that
 * exporter's `parseArgs` deliberately only accepts `--graph`/`--substeps`/
 * `--out`/`--include-world` (see its module doc's overwrite-guard
 * reasoning) and always uses its own fixed `TRACE_SEEDS`/`TRACE_TICKS`
 * constants; this script reuses its `buildSeedTrace` (the actual
 * `createWorld`+`stepWorld`+rate-model rollout, unchanged) with a
 * configurable tick count and seed list instead of editing that exporter's
 * committed-fixture-focused CLI contract.
 *
 * The committed golden fixtures (60 ticks x 4 seeds) never trigger a food
 * pickup, wall clamp, or hazard contact (thermo-architecture review finding
 * #2) — this script's default of many more ticks across many more seeds
 * does, purely by the authored closed-loop policy (`aggregateOutputs` ->
 * `decodeAction`) wandering the trace-graph's small arena for long enough;
 * no action scripting/injection was needed to reach useful coverage (see
 * this script's own printed event-count summary, and
 * `training/README.md`'s "Longer traces for event coverage" section for the
 * measured counts).
 *
 * Output is never committed (`training/runs/` is gitignored); run via:
 *   npx tsx training/scripts/generate_long_traces.ts --out training/runs/traces/<name> [--ticks N] [--seeds 1,2,3]
 */

const DEFAULT_TICKS = 2000;
// Hand-picked by `training/scripts/probe_events.ts` (a throwaway sweep over
// seeds 1..800, not committed) for combined coverage of all three rare
// events: seeds 120/349/766/152/598/750/501/595/37/216 lead in wall-clamp
// ticks, 377/132/163 add extra food-respawn/hazard-contact volume. Measured
// totals at `DEFAULT_TICKS`, printed by this script and recorded in
// `training/README.md`: 63 wall-clamp ticks, several dozen distinct food
// respawns, several dozen distinct hazard-contact transitions.
const DEFAULT_SEEDS = [120, 349, 766, 152, 598, 750, 501, 595, 37, 216, 377, 132, 163];
const GRAPH_ID = 'trace-graph';
const SUBSTEPS = 4;

interface Args {
  outDir: string;
  ticks: number;
  seeds: number[];
}

const requireValue = (flag: string, value: string | undefined): string => {
  if (value === undefined || value.startsWith('--')) {
    throw new Error(`${flag} requires a value`);
  }
  return value;
};

const parseArgs = (argv: readonly string[]): Args => {
  let outDir: string | undefined;
  let ticks = DEFAULT_TICKS;
  let seeds = DEFAULT_SEEDS;

  for (let index = 0; index < argv.length; index += 1) {
    const flag = argv[index];
    if (flag === '--out') {
      outDir = requireValue(flag, argv[index + 1]);
      index += 1;
    } else if (flag === '--ticks') {
      const value = requireValue(flag, argv[index + 1]);
      const parsed = Number(value);
      if (!Number.isInteger(parsed) || parsed <= 0) throw new Error(`--ticks must be a positive integer, got "${value}"`);
      ticks = parsed;
      index += 1;
    } else if (flag === '--seeds') {
      const value = requireValue(flag, argv[index + 1]);
      seeds = value.split(',').map((raw) => {
        const parsed = Number(raw);
        if (!Number.isInteger(parsed)) throw new Error(`--seeds must be a comma-separated list of integers, got "${raw}"`);
        return parsed;
      });
      index += 1;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  if (!outDir) throw new Error('--out is required (a gitignored directory, e.g. training/runs/traces/<name>)');
  return { outDir, ticks, seeds };
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

const writeJson = (path: string, value: unknown): number => {
  mkdirSync(resolve(path, '..'), { recursive: true });
  const contents = JSON.stringify(value);
  writeFileSync(path, contents);
  return Buffer.byteLength(contents);
};

const options: BuildSeedTraceOptions = { includeWorld: true };

const main = (): void => {
  const args = parseArgs(process.argv.slice(2));
  const graph = createTraceGraph();
  validateGraph(graph);

  const outDir = resolve(process.cwd(), args.outDir);
  let totalBytes = writeJson(resolve(outDir, `${GRAPH_ID}.json`), serializeGraph(graph));

  let totalFoodRespawns = 0;
  let totalHazardContacts = 0;
  let totalWallClamps = 0;

  for (const seed of args.seeds) {
    const trace = buildSeedTrace(graph, GRAPH_ID, seed, args.ticks, SUBSTEPS, options);
    totalBytes += writeJson(resolve(outDir, `${GRAPH_ID}-seed-${seed}.json`), trace);

    // Informational event-count summary, printed below, so a human (or the
    // conftest fixture) can see at a glance whether this run produced
    // useful coverage without re-parsing the written JSON. `foodRespawns`/
    // `agentScores[][1]` are cumulative-to-date counters (not per-tick
    // deltas), so only the *last* tick's values give the true total count —
    // summing every tick would massively over-count.
    const worldAfter = trace.worldAfter ?? [];
    const lastTick = worldAfter[worldAfter.length - 1];
    if (lastTick) {
      totalFoodRespawns += lastTick.foodRespawns.reduce((sum, value) => sum + value, 0);
      totalHazardContacts += lastTick.agentScores.reduce((sum, score) => sum + score[1], 0);
    }
    // Wall clamps aren't a recorded counter; detect them the same way the
    // Python coverage test does, by checking whether a position landed
    // exactly on an arena boundary (a clamp always leaves it there exactly;
    // free motion essentially never does, in a continuous space).
    const config = createWorld(seed).config;
    const maxX = config.halfWidth - config.agentRadius;
    const maxZ = config.halfDepth - config.agentRadius;
    for (const tick of worldAfter) {
      for (const position of tick.agentPositions) {
        if (Math.abs(position[0]) === maxX || Math.abs(position[1]) === maxZ) totalWallClamps += 1;
      }
    }
  }

  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
  console.log(
    `Wrote ${args.seeds.length + 1} file(s) to ${outDir} (${totalBytes} bytes total, ` +
      `${args.ticks} ticks x ${args.seeds.length} seeds): food respawns=${totalFoodRespawns}, ` +
      `hazard contacts=${totalHazardContacts}, wall-clamp ticks=${totalWallClamps}`
  );
};

try {
  main();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
  console.error(`generate_long_traces failed: ${message}`);
  process.exit(1);
}
