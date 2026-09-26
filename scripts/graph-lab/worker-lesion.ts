import type { GraphMode } from '../../src/lib/connectome/format';
import { runEpisode } from '../training/episode';
import {
  assertFiniteScores,
  graphFromTaskMode,
  loadVerifiedGraphBinary,
  runWorkerMain
} from '../null/null-worker-shared';

/**
 * `entry-lesion.ts`'s child process (forked by `shard.ts`'s `runOnWorkers`,
 * one task per lesion set plus one baseline task with an empty `lesion`).
 * Structurally a thin variant of `scripts/null/null-worker.ts`'s own
 * `runTask` -- reusing the exact same graph-loading and finite-score
 * helpers from `null-worker-shared.ts` (never modified; see this WP's
 * scope note in `entry-lesion.ts`) -- with one addition: each task carries
 * its own `lesion` set, passed straight through to `runEpisode`'s
 * `left.lesion` (`scripts/training/episode.ts`), which already validates
 * it (range, sorted-unique) before running. An empty `lesion` array is
 * numerically identical to no lesion at all (`episode.ts`'s own
 * documented convention), which is exactly what the baseline task uses.
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

const runTask = (task: LesionWorkerTask): readonly LesionSeedResult[] => {
  const graphBinary = loadVerifiedGraphBinary('worker-lesion', task.path, task.expectedSha256);
  const graph = graphFromTaskMode(task.mode, graphBinary);
  const lesion = task.lesion.length > 0 ? Int32Array.from(task.lesion) : undefined;

  return task.heldOutSeeds.map((seed) => {
    const result = runEpisode({
      seed,
      ticks: task.ticks,
      left: { decoder: 'authored', graph, lesion },
      right: { decoder: 'parked' }
    });
    const { movementScore, foodPickups, hazardContacts } = result.left;
    assertFiniteScores('worker-lesion', task.graphId, seed, { movementScore, foodPickups, hazardContacts });
    return { seed, movementScore, foodPickups, hazardContacts };
  });
};

runWorkerMain(runTask);
