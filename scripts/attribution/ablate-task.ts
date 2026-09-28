import { readoutFromFlat } from '../../src/lib/connectome/readout-serialization';
import { runEpisode } from '../training/episode';
import { assertFiniteScores } from '../null/null-worker-shared';
import { resolveReadoutGraph } from './resolve-graph';

/**
 * The pure, side-effect-free task `ablate-worker.ts`'s
 * `node:child_process.fork`ed leaf runs (`.agents/plans/readout-attribution/
 * 02-analyses.md`'s WP2 analysis 2: readout-input ablation). Split out
 * (matching `regime-task.ts`'s own split from `regime-worker.ts`) so this
 * module stays importable -- e.g. by a unit test -- without also
 * registering a `process.on('message', ...)` listener on the importing
 * process's own IPC channel (`ablate-worker.ts`'s doc comment).
 *
 * One task is one `(readout, ablated-input-or-baseline)` pair, scored over
 * its own held-out seeds with decoder `trained` and (for a non-baseline
 * task) a single-index `readoutMask` -- mirrors `null-trained-worker.ts`'s
 * shape (a readout + weights, scored with `trained`) but resolves its graph
 * through `resolve-graph.ts` (sha-verified against the archive's
 * `graphGzipSha256`/`graphBinarySha256`) rather than an `export-arms.ts`
 * bundle, since ablation must run against the SAME graph resolution WP2's
 * other analyses use, including the intervention ids `null-trained-worker.ts`
 * never had to resolve.
 *
 * `task.graphId` here is `ablate.ts`'s own synthetic per-task key (e.g.
 * `"biological-seed101|baseline"`/`"biological-seed101|d12"`), NOT
 * necessarily the archive's own `graphId` -- `runShardedEvaluation`'s
 * generic `Task`/`Result`/`Message` protocol only requires this field to be
 * a stable, per-task-unique string for its own result bookkeeping and
 * self-checking wire protocol (`sharded-evaluation.ts`'s doc comment); it
 * is never interpreted as a real connectome-graph id by this worker or by
 * `resolveReadoutGraph`, which reads `task.resolvedGraphId` instead.
 */

export interface AblateWorkerTask {
  readonly graphId: string;
  readonly resolvedGraphId: string;
  readonly graphGzipSha256: string | null;
  readonly graphBinarySha256: string | null;
  readonly manifestPath: string;
  readonly interventionIndexPath?: string;
  readonly archivedInterventionIndexPath?: string;
  /** Flat `[w1, b1, w2, b2]` theta, matching `run-dir.ts`'s documented layout. */
  readonly theta: readonly number[];
  readonly D: number;
  readonly H: number;
  /** `null` for the baseline (unmasked) task; a single-element array for a one-input ablation. */
  readonly mask: readonly number[] | null;
  readonly heldOutSeeds: readonly number[];
  readonly ticks: number;
  readonly arenaTask?: string;
}

export interface AblateSeedResult {
  readonly seed: number;
  readonly movementScore: number;
}

export interface AblateWorkerResultMessage {
  readonly type: 'result';
  readonly graphId: string;
  readonly results: readonly AblateSeedResult[];
}

export interface AblateWorkerErrorMessage {
  readonly type: 'error';
  readonly graphId: string;
  readonly message: string;
}

export type AblateWorkerMessage = AblateWorkerResultMessage | AblateWorkerErrorMessage;

export const runTask = (task: AblateWorkerTask): readonly AblateSeedResult[] => {
  const graph = resolveReadoutGraph(
    { graphId: task.resolvedGraphId, graphGzipSha256: task.graphGzipSha256, graphBinarySha256: task.graphBinarySha256 },
    {
      manifestPath: task.manifestPath,
      interventionIndexPath: task.interventionIndexPath,
      archivedInterventionIndexPath: task.archivedInterventionIndexPath
    }
  );
  const weights = readoutFromFlat(Float32Array.from(task.theta), task.D, task.H);
  const readoutMask = task.mask ? Int32Array.from(task.mask) : undefined;

  return task.heldOutSeeds.map((seed) => {
    const result = runEpisode({
      seed,
      ticks: task.ticks,
      arenaTask: task.arenaTask,
      left: { decoder: 'trained', graph, weights, readoutMask },
      right: { decoder: 'parked' }
    });
    assertFiniteScores('ablate-worker', task.graphId, seed, result.left);
    return { seed, movementScore: result.left.movementScore };
  });
};
