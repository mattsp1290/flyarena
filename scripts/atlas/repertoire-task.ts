import type { Metrics } from '../../src/lib/atlas/types';
import type { ArmName } from '../training/arms';
import { verifyAndEvaluateSearchGraph, type ExpectedGraphIdentity } from './verify-search-graph';

/**
 * The pure, side-effect-free core of `repertoire-worker.ts`'s forked leaf:
 * verify + re-evaluate one search file, then trim/rename its result into
 * `RepertoireEvaluatedEntry`. Deliberately split out of `repertoire-worker.ts`
 * itself (which registers a real `process.on('message', ...)` listener at
 * module load, a side effect of being imported at all) so
 * `tests/unit/repertoire.test.ts` can import `runRepertoireTask` directly,
 * in-process, without also registering a listener on the *test process's
 * own* IPC channel -- the exact hazard `scripts/null/null-worker-shared.ts`'s
 * own doc comment documents (`regime-task.ts` was split out of
 * `regime-worker.ts` for the identical reason).
 */

export interface RepertoireWorkerTask {
  /** `<graphId>@<searchSeed>`, e.g. `"rewired-3@1730"` -- unique per task, matching `repertoire-plan.ts`'s per-entry key. */
  readonly graphId: string;
  readonly searchPath: string;
  readonly expected: ExpectedGraphIdentity;
  readonly arm: ArmName;
  readonly rewiringSeed: number | null;
  readonly searchSeed: number;
}

/** One occupied cell's audit-relevant fields, trimmed of the raw per-candidate `discovery`/`evaluations`/`replay` payloads `evaluateSearchOnGraph` also returns -- `repertoire-metrics.ts`'s `occupied`/`qd`/`span`/`heldoutMedian` need only these. `heldoutOwn` renames `GraphEvaluationResultOwn`'s internal `heldout.biological` key (`.agents/plans/repertoire-null/02-runs-and-reevaluation.md`: "the output schema renames heldoutOwn so nothing called 'biological' holds rewired results"). */
export interface RepertoireCellResult {
  readonly cell: number;
  readonly quality: number;
  readonly coverage: number;
  readonly turning: number;
  readonly heldoutOwn: readonly Metrics[];
}

export interface RepertoireEvaluatedEntry {
  readonly graphId: string;
  readonly arm: ArmName;
  readonly rewiringSeed: number | null;
  readonly searchSeed: number;
  readonly gpuArchiveSize: number;
  readonly occupied: number;
  readonly collisions: number;
  readonly cells: readonly RepertoireCellResult[];
}

export type RepertoireWorkerMessage =
  | { readonly type: 'result'; readonly graphId: string; readonly results: readonly [RepertoireEvaluatedEntry] }
  | { readonly type: 'error'; readonly graphId: string; readonly message: string };

export const runRepertoireTask = async (task: Readonly<RepertoireWorkerTask>): Promise<RepertoireEvaluatedEntry> => {
  const result = await verifyAndEvaluateSearchGraph(task.searchPath, task.expected);
  const cells: RepertoireCellResult[] = result.cells.map((cell) => {
    const heldoutOwn = cell.heldout.biological;
    if (!heldoutOwn) {
      throw new Error(`repertoire-task: ${task.graphId} cell ${cell.cell} has no held-out-own evaluation`);
    }
    return { cell: cell.cell, quality: cell.quality, coverage: cell.coverage, turning: cell.turning, heldoutOwn };
  });
  return {
    graphId: task.graphId,
    arm: task.arm,
    rewiringSeed: task.rewiringSeed,
    searchSeed: task.searchSeed,
    gpuArchiveSize: result.gpuArchiveSize,
    occupied: result.cells.length,
    collisions: result.collisions,
    cells
  };
};
