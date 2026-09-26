import type { Metrics } from '../../src/lib/atlas/types';
import type { ArmName } from '../training/arms';
import { verifyAndEvaluateSearchGraph, type ExpectedGraphIdentity, type ExpectedSearchOptions } from './verify-search-graph';

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
  /**
   * The composite wire/scheduling key (`repertoire-plan.ts`'s `planEntryKey`,
   * e.g. `"rewired-3@1730"`) -- unique per task, used only for IPC
   * message-matching and the scheduler's results map. Never itself persisted
   * into `RepertoireEvaluatedEntry`; see `graphId` below for the field that
   * is (a dual-review finding: an earlier version copied this composite
   * string into the persisted entry's own `graphId`, which is supposed to
   * mean the same thing as `RepertoirePlanEntry.graphId` -- the plain graph
   * identity, with no seed baked in -- for a downstream per-graph grouping
   * consumer, e.g. WP3's report).
   */
  readonly key: string;
  /** `biological` | `disconnected` | `rewired-<N>` -- no search seed baked in. Passed straight through to `RepertoireEvaluatedEntry.graphId`. */
  readonly graphId: string;
  readonly searchPath: string;
  readonly expected: ExpectedGraphIdentity;
  /** The plan's expected `(seed, population, generations, ticks)` for this task, checked against the search file's own recorded `options` before it is trusted (see `verifyAndEvaluateSearchGraph`'s `expectedOptions` parameter). Catches a misnamed, duplicated, or under-budget search file standing in for this task -- graph identity alone is shared across every search seed of the same graph, so it cannot tell them apart. */
  readonly expectedOptions: ExpectedSearchOptions;
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
  /** The plain graph identity (`"biological"`, `"disconnected"`, `"rewired-<N>"`) -- matches `RepertoirePlanEntry.graphId`, never the composite `key`. */
  readonly graphId: string;
  readonly arm: ArmName;
  readonly rewiringSeed: number | null;
  readonly searchSeed: number;
  /** The verified search options this entry was actually searched and re-evaluated with -- an audit echo of `RepertoireWorkerTask.expectedOptions`, recorded here because `verifyAndEvaluateSearchGraph` has already proven the search file's own `options` equal it. */
  readonly searchOptions: ExpectedSearchOptions;
  readonly gpuArchiveSize: number;
  readonly occupied: number;
  readonly collisions: number;
  readonly cells: readonly RepertoireCellResult[];
}

export type RepertoireWorkerMessage =
  | { readonly type: 'result'; readonly key: string; readonly results: readonly [RepertoireEvaluatedEntry] }
  | { readonly type: 'error'; readonly key: string; readonly message: string };

/** Every held-out score `evaluateBehavior` reports; a search re-evaluation with an empty or non-finite `heldoutOwn` would otherwise let a NaN/Infinity silently propagate into `repertoire-metrics.ts`'s `heldoutOwnMedian` (where it would make the reported median depend on JS's sort order instead of throwing) and into published statistics further downstream -- fail here, at the one point that still knows which task produced it (mirrors `scripts/null/null-worker-shared.ts`'s `assertFiniteScores` convention). */
const HELDOUT_SEED_COUNT = 12;

const assertHeldoutOwnIsSound = (task: Readonly<RepertoireWorkerTask>, cell: Readonly<RepertoireCellResult>): void => {
  if (cell.heldoutOwn.length !== HELDOUT_SEED_COUNT) {
    throw new Error(
      `repertoire-task: ${task.key} cell ${cell.cell} has ${cell.heldoutOwn.length} held-out-own scores, expected ${HELDOUT_SEED_COUNT}`
    );
  }
  for (const metrics of cell.heldoutOwn) {
    if (![metrics.movementScore, metrics.foodPickups, metrics.hazardContacts].every(Number.isFinite)) {
      throw new Error(`repertoire-task: ${task.key} cell ${cell.cell} produced a non-finite held-out-own score`);
    }
  }
};

export const runRepertoireTask = async (task: Readonly<RepertoireWorkerTask>): Promise<RepertoireEvaluatedEntry> => {
  const result = await verifyAndEvaluateSearchGraph(task.searchPath, task.expected, undefined, task.expectedOptions);
  const cells: RepertoireCellResult[] = result.cells.map((cell) => {
    const heldoutOwn = cell.heldout.biological;
    if (!heldoutOwn) {
      throw new Error(`repertoire-task: ${task.key} cell ${cell.cell} has no held-out-own evaluation`);
    }
    const cellResult = { cell: cell.cell, quality: cell.quality, coverage: cell.coverage, turning: cell.turning, heldoutOwn };
    assertHeldoutOwnIsSound(task, cellResult);
    return cellResult;
  });
  return {
    graphId: task.graphId,
    arm: task.arm,
    rewiringSeed: task.rewiringSeed,
    searchSeed: task.searchSeed,
    searchOptions: task.expectedOptions,
    gpuArchiveSize: result.gpuArchiveSize,
    occupied: result.cells.length,
    collisions: result.collisions,
    cells
  };
};
