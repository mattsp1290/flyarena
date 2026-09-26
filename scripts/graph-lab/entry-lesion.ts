import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { graphFromTaskMode, loadVerifiedGraphBinary } from '../null/null-worker-shared';
import { conditionRng, pairedStats } from '../training/stats';
import { runOnWorkers } from './shard';
import type { LesionSeedResult, LesionWorkerMessage, LesionWorkerTask } from './worker-lesion';

/**
 * `jobs.py`'s `default_runner` invokes this as
 * `node entry-lesion.mjs <args.json path>` (esbuild-bundled to
 * `backend/graph_lab/js/entry-lesion.mjs`; `worker-lesion.mjs` sits
 * alongside it in the same output directory).
 *
 * WP1 scope note: this wires the graph load
 * (`biological`/`disconnected` only -- `rewired:<seed>` regeneration is
 * `jobs.py`'s `default_runner`'s job in WP2, per
 * `.agents/plans/graph-lab/02-job-engines.md`) and the sharded
 * scoring/statistics for real, against real fixture data
 * (`tests/unit/graph-lab-entries.test.ts` runs this built `.mjs` from an
 * unrelated cwd against a temp-directory fixture graph). It intentionally
 * does not implement every WP2 bound (e.g. the GPU-busy check has no
 * analog here) -- those live with the job kinds WP2 actually wires.
 */

interface LesionArgs {
  readonly dataDir: string;
  readonly mode: 'biological' | 'disconnected';
  readonly graphPath: string;
  readonly expectedSha256: string;
  readonly sets: readonly (readonly number[])[];
  readonly seedStart: number;
  readonly seedCount: number;
  readonly ticks: number;
  readonly shards?: number;
  readonly bootstrapResamples?: number;
}

interface SetResult {
  readonly indices: readonly number[];
  readonly bodyIds: readonly string[];
  readonly effect: { readonly n: number; readonly meanDifference: number; readonly ci95: readonly [number, number] };
  readonly n: number;
}

const printProgress = (progress: Record<string, unknown>): void => {
  process.stdout.write(`${JSON.stringify({ type: 'progress', progress })}\n`);
};

const printResult = (result: unknown): void => {
  process.stdout.write(`${JSON.stringify({ type: 'result', result })}\n`);
};

const printError = (message: string): void => {
  process.stdout.write(`${JSON.stringify({ type: 'error', message })}\n`);
};

const buildTasks = (args: LesionArgs): LesionWorkerTask[] => {
  const heldOutSeeds = Array.from({ length: args.seedCount }, (_, index) => args.seedStart + index);
  const baseline: LesionWorkerTask = {
    graphId: 'baseline',
    mode: args.mode,
    path: args.graphPath,
    expectedSha256: args.expectedSha256,
    heldOutSeeds,
    ticks: args.ticks,
    lesion: []
  };
  const setTasks = args.sets.map(
    (indices, setIndex): LesionWorkerTask => ({
      graphId: `set-${setIndex}`,
      mode: args.mode,
      path: args.graphPath,
      expectedSha256: args.expectedSha256,
      heldOutSeeds,
      ticks: args.ticks,
      lesion: [...indices].sort((a, b) => a - b)
    })
  );
  return [baseline, ...setTasks];
};

const main = async (): Promise<void> => {
  const argsPath = process.argv[2];
  if (!argsPath) throw new Error('entry-lesion: missing required args-file argument');
  const args: LesionArgs = JSON.parse(readFileSync(argsPath, 'utf8'));

  const tasks = buildTasks(args);
  const workerPath = fileURLToPath(new URL('./worker-lesion.mjs', import.meta.url));
  const shardCount = args.shards ?? Math.min(4, tasks.length);
  const resamples = args.bootstrapResamples ?? 2000;

  printProgress({ completed: 0, total: tasks.length });
  const results = await runOnWorkers<LesionWorkerTask, LesionSeedResult, LesionWorkerMessage>(
    tasks,
    shardCount,
    workerPath,
    (completed, total) => printProgress({ completed, total })
  );

  const baselineScores = results.get('baseline');
  if (!baselineScores) throw new Error('entry-lesion: baseline task produced no results');
  const baselineMovement = baselineScores.map((r) => r.movementScore);

  // Loaded once here (not per set, and not duplicating the workers' own
  // loads across process boundaries) purely to map each set's neuron
  // indices to the real biological ids the plan's result shape asks for
  // (`bodyIds`) -- a read-only, already-sha-verified load, not a second
  // scoring pass.
  const graphBinary = loadVerifiedGraphBinary('entry-lesion', args.graphPath, args.expectedSha256);
  const graph = graphFromTaskMode(args.mode, graphBinary);

  const sets: SetResult[] = args.sets.map((indices, setIndex) => {
    const graphId = `set-${setIndex}`;
    const setScores = results.get(graphId);
    if (!setScores) throw new Error(`entry-lesion: ${graphId} produced no results`);
    const setMovement = setScores.map((r) => r.movementScore);
    const rng = conditionRng(args.seedStart, `graph-lab-lesion|${graphId}`);
    const effect = pairedStats(setMovement, baselineMovement, resamples, rng);
    const sortedIndices = [...indices].sort((a, b) => a - b);
    return {
      indices: sortedIndices,
      bodyIds: sortedIndices.map((index) => graph.biologicalIds[index].toString()),
      effect,
      n: effect.n
    };
  });

  printResult({
    graph: args.mode,
    graphSha256: args.expectedSha256,
    dataDir: args.dataDir,
    host: { arch: process.arch, node: process.version },
    label: 'Computed on DGX (private, not published)',
    baseline: {
      n: baselineMovement.length,
      mean: baselineMovement.reduce((sum, value) => sum + value, 0) / baselineMovement.length
    },
    sets
  });
};

main().catch((error: unknown) => {
  printError(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
