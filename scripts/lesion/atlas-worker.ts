import { parseGraphBinary } from '../../src/lib/connectome/format';
import { runEpisode } from '../training/episode';
import { assertFiniteScores, loadVerifiedGraphBinary, runWorkerMain } from '../null/null-worker-shared';
import type { AtlasSeedResult, AtlasWorkerTask } from './atlas-evaluate';

/**
 * `atlas-evaluate.ts`'s child process, `.agents/plans/lesion-atlas/02-atlas-computation.md`'s
 * WP2 worker: `node:child_process.fork`ed once per shard by the parent's
 * `runShardedEvaluation` (`scripts/null/null-evaluate.ts`'s generic
 * `runShardedEvaluation<Task, Result, Message>`, called directly by
 * `atlas-evaluate.ts` -- no adapter needed). The `process.on('message',
 * ...)`/`process.send` IPC wiring and the non-finite-score guard are shared
 * with `null-worker.ts`/`null-trained-worker.ts` via
 * `scripts/null/null-worker-shared.ts`'s `runWorkerMain`/`assertFiniteScores`,
 * not hand-duplicated.
 *
 * Unlike `scripts/null/null-worker.ts`, this worker never derives a graph
 * mode (no `disconnected` control, no rewiring-at-runtime): the atlas has
 * exactly two graphs, both already-compiled artifacts on disk
 * (`malecns-arena-v1.bin.gz` and `malecns-arena-v1-rewired-seed0.bin.gz`),
 * so a task's `path` is loaded and parsed directly.
 *
 * Every task covers *all* of `task.heldOutSeeds` for one `(graph,
 * lesionIndex | baseline)` pair (`task.lesionIndex === null` means the
 * unlesioned baseline for that graph): `01-lesion-episodes.md`'s
 * `AgentEpisodeConfig.lesion` contract requires a sorted, unique,
 * in-range `Int32Array`, which a single-element `Int32Array.of(index)`
 * trivially satisfies; the baseline task simply omits `lesion` (same
 * numeric result as an empty lesion, since `runLesionedSubsteps`'s zeroing
 * loops are no-ops at length zero -- see `episode.ts`'s `EMPTY_LESION`
 * doc comment).
 */

const runTask = (task: AtlasWorkerTask): readonly AtlasSeedResult[] => {
  const graphBinary = loadVerifiedGraphBinary('atlas-worker', task.path, task.expectedSha256);
  const graph = parseGraphBinary(graphBinary);
  const lesion = task.lesionIndex === null ? undefined : Int32Array.of(task.lesionIndex);
  const lesionLabel = task.lesionIndex === null ? 'baseline' : task.lesionIndex;

  return task.heldOutSeeds.map((seed) => {
    let result;
    try {
      result = runEpisode({
        seed,
        ticks: task.ticks,
        left: { decoder: 'authored', graph, lesion },
        right: { decoder: 'parked' }
      });
    } catch (error) {
      // Any runEpisode failure (a validation throw, an unexpected runtime
      // error) would otherwise surface with whatever message runEpisode
      // itself produced, with no indication of which graph/lesion/seed in
      // a ~200,000-episode run hit it -- re-thrown here naming all three.
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`atlas-worker: graph "${task.graphId}" lesion=${lesionLabel} seed ${seed}: ${message}`);
    }
    const { movementScore, foodPickups, hazardContacts } = result.left;
    assertFiniteScores('atlas-worker', task.graphId, seed, { movementScore, foodPickups, hazardContacts });
    return { seed, movementScore, foodPickups, hazardContacts };
  });
};

runWorkerMain(runTask);
