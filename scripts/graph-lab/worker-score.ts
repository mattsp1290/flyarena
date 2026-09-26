import { parseGraphBinary } from '../../src/lib/connectome/format';
import { resolveArenaTask } from '../../src/lib/arena/tasks';
import { runEpisode } from '../training/episode';
import { assertFiniteScores, loadVerifiedGraphBinary, runWorkerMain } from '../null/null-worker-shared';

/**
 * `entry-swapset.ts`'s child process: scores one already-built graph binary
 * (the intervened graph, or one class-matched control) against a parked
 * opponent with the authored decoder over a set of seeds -- "the existing
 * null-worker logic", per `.agents/plans/graph-lab/02-job-engines.md`'s
 * swap-set engine description.
 *
 * Deliberately its own file rather than reusing `null-worker.ts` (WP1's
 * change surface names `worker-score.ts` as a new file, and `null-worker.ts`
 * is out of scope to modify): a swap-set graph is neither `biological`,
 * `rewired`, nor `disconnected` -- it has no `GraphMode` to derive from, it
 * *is* the concrete graph, already fully built (in Python, from
 * `swap_ops.py`, in WP2) before this worker ever sees it. So this task
 * carries a verified binary straight into `parseGraphBinary`, skipping
 * `null-worker-shared.ts`'s `graphFromTaskMode` (which exists specifically
 * to derive `disconnected`/select `rewired` from a `GraphMode` label) --
 * every other piece (`loadVerifiedGraphBinary`, `assertFiniteScores`,
 * `runWorkerMain`) is shared unchanged.
 */
export interface ScoreWorkerTask {
  readonly graphId: string;
  readonly path: string;
  readonly expectedSha256: string;
  readonly heldOutSeeds: readonly number[];
  readonly ticks: number;
  /** `.agents/plans/task-generality/01-task-plumbing.md`'s WP1: the arena task id this task's `runEpisode` call resolves and scores against. Absent means `'default'` (`ARENA_CONFIG`, unchanged). */
  readonly arenaTask?: string;
}

export interface ScoreSeedResult {
  readonly seed: number;
  readonly movementScore: number;
  readonly foodPickups: number;
  readonly hazardContacts: number;
}

export interface ScoreWorkerResultMessage {
  readonly type: 'result';
  readonly graphId: string;
  readonly results: readonly ScoreSeedResult[];
}

export interface ScoreWorkerErrorMessage {
  readonly type: 'error';
  readonly graphId: string;
  readonly message: string;
}

export type ScoreWorkerMessage = ScoreWorkerResultMessage | ScoreWorkerErrorMessage;

/** Mirrors `worker-lesion.ts`'s own `validateTaskArenaTask` (see there for the full rationale). */
const validateTaskArenaTask = (arenaTask: unknown): string | undefined => {
  if (arenaTask === undefined) return undefined;
  if (typeof arenaTask !== 'string') {
    throw new Error(`worker-score: task.arenaTask is not a string: ${JSON.stringify(arenaTask)}`);
  }
  resolveArenaTask(arenaTask); // throws with a specific message on an unrecognized id
  return arenaTask;
};

const runTask = (task: ScoreWorkerTask): readonly ScoreSeedResult[] => {
  const graphBinary = loadVerifiedGraphBinary('worker-score', task.path, task.expectedSha256);
  const graph = parseGraphBinary(graphBinary.slice(0));
  const arenaTask = validateTaskArenaTask(task.arenaTask);

  return task.heldOutSeeds.map((seed) => {
    const result = runEpisode({
      seed,
      ticks: task.ticks,
      arenaTask,
      left: { decoder: 'authored', graph },
      right: { decoder: 'parked' }
    });
    const { movementScore, foodPickups, hazardContacts } = result.left;
    assertFiniteScores('worker-score', task.graphId, seed, { movementScore, foodPickups, hazardContacts });
    return { seed, movementScore, foodPickups, hazardContacts };
  });
};

runWorkerMain(runTask);
