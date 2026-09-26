import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { graphFromTaskMode, loadVerifiedGraphBinary } from '../null/null-worker-shared';
import { runShardedEvaluation } from '../null/sharded-evaluation';
import { conditionRng, pairedStats } from '../training/stats';
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
 *
 * Sharding: uses `scripts/null/sharded-evaluation.ts`'s `runShardedEvaluation`
 * directly (not a graph-lab-local reimplementation). That module was
 * extracted from `null-evaluate.ts` on `origin/main` specifically to be
 * self-contained (its only import is `node:child_process`) and free of
 * the top-level `if (process.argv[1] === fileURLToPath(import.meta.url))
 * main();` CLI-invocation guard that made `null-evaluate.ts` itself unsafe
 * to bundle (see this WP's earlier commits/report for the empirical
 * reproduction of that hazard). `LesionWorkerTask`/`LesionSeedResult`/
 * `LesionWorkerMessage` already satisfy its generic constraints
 * (`graphId` + `heldOutSeeds`; `seed`; the `result`/`error` message
 * shape), so no task-shape changes were needed. This also closes two
 * safety gaps a graph-lab-local `shard.ts` had dropped versus the
 * canonical version (a review finding): a `child.on('error', ...)`
 * handler (a spawn-level failure, e.g. `EMFILE`/`ENOMEM` under GPU/CPU
 * contention, now fails fast instead of hanging until the job's
 * wall-clock ceiling) and the "clean exit while a task is still in
 * flight" check. It also adds a self-checking protocol (a worker's reply
 * must match the task actually in flight, by `graphId` and per-seed
 * `seed`) and a completeness check (every task must have a result) that
 * this file's own error handling didn't have before either.
 *
 * Trade-off accepted: `runShardedEvaluation` has no per-task-completion
 * progress hook (unlike the graph-lab-local scheduler it replaces), so
 * only one `progress` line is emitted (before sharding starts) rather
 * than one per completed task. Correctness/safety over progress
 * granularity -- neither reviewer asked for finer-grained progress, and
 * `jobs.py`'s job store never required it (the job's terminal status
 * always comes from the final `result`/`error` line, not from progress).
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
  /**
   * `.agents/plans/task-generality/01-task-plumbing.md`'s WP1: forwarded
   * verbatim to each task's `arenaTask` field (see `worker-lesion.ts`),
   * which `runEpisode` resolves the same way `null-worker.ts` does.
   * Absent means `'default'` (`ARENA_CONFIG`, unchanged) -- `jobs.py`'s
   * `default_runner` does not set this yet (WP1's own job models have no
   * arena-task field), so every request today runs the default task; this
   * field exists so a future job-model addition needs no further entry
   * changes.
   */
  readonly arenaTask?: string;
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
    lesion: [],
    arenaTask: args.arenaTask
  };
  const setTasks = args.sets.map(
    (indices, setIndex): LesionWorkerTask => ({
      graphId: `set-${setIndex}`,
      mode: args.mode,
      path: args.graphPath,
      expectedSha256: args.expectedSha256,
      heldOutSeeds,
      ticks: args.ticks,
      lesion: [...indices].sort((a, b) => a - b),
      arenaTask: args.arenaTask
    })
  );
  return [baseline, ...setTasks];
};

const main = async (): Promise<void> => {
  const argsPath = process.argv[2];
  if (!argsPath) throw new Error('entry-lesion: missing required args-file argument');
  const args: LesionArgs = JSON.parse(readFileSync(argsPath, 'utf8'));
  // `LesionArgs.mode`'s TS type only claims `'biological' | 'disconnected'`
  // at compile time -- this crosses a JSON file boundary (Python's
  // `default_runner` writes it, this process just reads and trusts
  // whatever's on disk) with no runtime guarantee behind that type. A
  // stray third value would otherwise reach `graphFromTaskMode`
  // (`null-worker-shared.ts`, unmodified), whose own `mode === 'rewired'`/
  // `mode === 'biological'` checks fall through to the `disconnected`
  // derivation for *anything else* -- silently mis-scoring a request
  // instead of erroring. Defense in depth: `jobs.py`'s own
  // `default_runner` only ever writes one of these two values today, but
  // this check does not rely on that staying true.
  if (args.mode !== 'biological' && args.mode !== 'disconnected') {
    throw new Error(`entry-lesion: unrecognized mode ${JSON.stringify(args.mode)}`);
  }

  const tasks = buildTasks(args);
  const workerPath = fileURLToPath(new URL('./worker-lesion.mjs', import.meta.url));
  const shardCount = args.shards ?? Math.min(4, tasks.length);
  const resamples = args.bootstrapResamples ?? 2000;

  printProgress({ completed: 0, total: tasks.length });
  const results = await runShardedEvaluation<LesionWorkerTask, LesionSeedResult, LesionWorkerMessage>(
    tasks,
    shardCount,
    workerPath
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
