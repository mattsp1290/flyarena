import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { conditionRng, pairedStats } from '../training/stats';
import { runOnWorkers } from './shard';
import type { ScoreSeedResult, ScoreWorkerMessage, ScoreWorkerTask } from './worker-score';

/**
 * `entry-swapset.mjs`: scores a list of already-built graph binaries
 * (the intervened graph plus its class-matched controls) against a parked
 * opponent with `worker-score.mjs` -- "score with the bundled null-worker
 * logic", per `.agents/plans/graph-lab/02-job-engines.md`'s swap-set
 * engine description.
 *
 * WP1 scope note: building those graph binaries from a swap set
 * (`swap_ops.py`, class-matched random controls, invariant checks) is
 * WP2's engine work, done in Python before this entry ever runs --
 * `jobs.py`'s `default_runner` 501s `kind: "swapset"` for now (see
 * `entry-atlas-reeval.ts`'s identical scope note). This entry only does
 * the scoring half, for real, against however many pre-built graphs its
 * args file lists -- WP2 wires it up by having its own engine module write
 * that args file. If `baselineGraphId` names one of `graphs`, every other
 * graph's paired difference against it is also computed, matching the
 * lesion entry's own baseline-diff shape.
 */
interface SwapsetGraphSpec {
  readonly graphId: string;
  readonly path: string;
  readonly expectedSha256: string;
}

interface SwapsetArgs {
  readonly dataDir: string;
  readonly graphs: readonly SwapsetGraphSpec[];
  readonly baselineGraphId?: string;
  readonly seedStart: number;
  readonly seedCount: number;
  readonly ticks: number;
  readonly shards?: number;
  readonly bootstrapResamples?: number;
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

const main = async (): Promise<void> => {
  const argsPath = process.argv[2];
  if (!argsPath) throw new Error('entry-swapset: missing required args-file argument');
  const args: SwapsetArgs = JSON.parse(readFileSync(argsPath, 'utf8'));
  if (args.graphs.length === 0) throw new Error('entry-swapset: at least one graph is required');

  const heldOutSeeds = Array.from({ length: args.seedCount }, (_, index) => args.seedStart + index);
  const tasks: ScoreWorkerTask[] = args.graphs.map((graph) => ({
    graphId: graph.graphId,
    path: graph.path,
    expectedSha256: graph.expectedSha256,
    heldOutSeeds,
    ticks: args.ticks
  }));

  const workerPath = fileURLToPath(new URL('./worker-score.mjs', import.meta.url));
  const shardCount = args.shards ?? Math.min(4, tasks.length);
  const resamples = args.bootstrapResamples ?? 2000;

  printProgress({ completed: 0, total: tasks.length });
  const results = await runOnWorkers<ScoreWorkerTask, ScoreSeedResult, ScoreWorkerMessage>(
    tasks,
    shardCount,
    workerPath,
    (completed, total) => printProgress({ completed, total })
  );

  const scores = args.graphs.map((graph) => {
    const seedResults = results.get(graph.graphId);
    if (!seedResults) throw new Error(`entry-swapset: ${graph.graphId} produced no results`);
    return {
      graphId: graph.graphId,
      n: seedResults.length,
      mean: seedResults.reduce((sum, r) => sum + r.movementScore, 0) / seedResults.length,
      movementScores: seedResults.map((r) => r.movementScore)
    };
  });

  const baseline = args.baselineGraphId ? scores.find((s) => s.graphId === args.baselineGraphId) : undefined;
  const paired = baseline
    ? scores
        .filter((s) => s.graphId !== baseline.graphId)
        .map((s) => ({
          graphId: s.graphId,
          effect: pairedStats(
            s.movementScores,
            baseline.movementScores,
            resamples,
            conditionRng(args.seedStart, `graph-lab-swapset|${s.graphId}`)
          )
        }))
    : [];

  printResult({
    dataDir: args.dataDir,
    host: { arch: process.arch, node: process.version },
    label: 'Computed on DGX (private, not published)',
    scores: scores.map(({ graphId, n, mean }) => ({ graphId, n, mean })),
    baselineGraphId: args.baselineGraphId,
    paired
  });
};

main().catch((error: unknown) => {
  printError(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
