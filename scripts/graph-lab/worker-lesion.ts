import type { GraphMode } from '../../src/lib/connectome/format';
import { resolveArenaTask } from '../../src/lib/arena/tasks';
import { runEpisode } from '../training/episode';
import {
  assertFiniteScores,
  graphFromTaskMode,
  loadVerifiedGraphBinary,
  runWorkerMain
} from '../null/null-worker-shared';

/**
 * `entry-lesion.ts`'s child process (forked by `scripts/null/sharded-evaluation.ts`'s
 * `runShardedEvaluation`, one task per lesion set plus one baseline task
 * with an empty `lesion`). Structurally a thin variant of
 * `scripts/null/null-worker.ts`'s own `runTask` -- reusing the exact same
 * graph-loading and finite-score helpers from `null-worker-shared.ts`
 * (never modified; see this WP's scope note in `entry-lesion.ts`) -- with
 * one addition: each task carries its own `lesion` set, passed straight
 * through to `runEpisode`'s `left.lesion` (`scripts/training/episode.ts`),
 * which already validates it (range, sorted-unique) before running. An
 * empty `lesion` array is numerically identical to no lesion at all
 * (`episode.ts`'s own documented convention), which is exactly what the
 * baseline task uses.
 */
export interface LesionWorkerTask {
  readonly graphId: string;
  readonly mode: GraphMode;
  /** Path to the gzip-compressed biological graph binary (also the source for a `disconnected` derivation). */
  readonly path: string;
  readonly expectedSha256: string;
  readonly heldOutSeeds: readonly number[];
  readonly ticks: number;
  /** Sorted, unique neuron indices to lesion for this task; empty means the unlesioned baseline. */
  readonly lesion: readonly number[];
  /** `.agents/plans/task-generality/01-task-plumbing.md`'s WP1: the arena task id this task's `runEpisode` call resolves and scores against. Absent means `'default'` (`ARENA_CONFIG`, unchanged) -- see `validateTaskArenaTask` below for its IPC-boundary validation, mirroring `null-worker.ts`'s own. */
  readonly arenaTask?: string;
}

export interface LesionSeedResult {
  readonly seed: number;
  readonly movementScore: number;
  readonly foodPickups: number;
  readonly hazardContacts: number;
}

export interface LesionWorkerResultMessage {
  readonly type: 'result';
  readonly graphId: string;
  readonly results: readonly LesionSeedResult[];
}

export interface LesionWorkerErrorMessage {
  readonly type: 'error';
  readonly graphId: string;
  readonly message: string;
}

export type LesionWorkerMessage = LesionWorkerResultMessage | LesionWorkerErrorMessage;

/**
 * Mirrors `null-worker.ts`'s own `validateTaskArenaTask`: `task.arenaTask`
 * crosses the `fork`/IPC boundary as plain JSON, so it must be re-validated
 * here rather than trusted from `LesionWorkerTask`'s compile-time type.
 * Delegates to `resolveArenaTask` (the single source of truth for valid
 * ids) instead of re-declaring the valid-id set.
 */
const validateTaskArenaTask = (arenaTask: unknown): string | undefined => {
  if (arenaTask === undefined) return undefined;
  if (typeof arenaTask !== 'string') {
    throw new Error(`worker-lesion: task.arenaTask is not a string: ${JSON.stringify(arenaTask)}`);
  }
  resolveArenaTask(arenaTask); // throws with a specific message on an unrecognized id
  return arenaTask;
};

const runTask = (task: LesionWorkerTask): readonly LesionSeedResult[] => {
  const graphBinary = loadVerifiedGraphBinary('worker-lesion', task.path, task.expectedSha256);
  const graph = graphFromTaskMode(task.mode, graphBinary);
  const lesion = task.lesion.length > 0 ? Int32Array.from(task.lesion) : undefined;
  const arenaTask = validateTaskArenaTask(task.arenaTask);

  return task.heldOutSeeds.map((seed) => {
    const result = runEpisode({
      seed,
      ticks: task.ticks,
      arenaTask,
      left: { decoder: 'authored', graph, lesion },
      right: { decoder: 'parked' }
    });
    const { movementScore, foodPickups, hazardContacts } = result.left;
    assertFiniteScores('worker-lesion', task.graphId, seed, { movementScore, foodPickups, hazardContacts });
    return { seed, movementScore, foodPickups, hazardContacts };
  });
};

runWorkerMain(runTask);
