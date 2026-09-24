import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';

import { parseGraphBinary } from '../../src/lib/connectome/format';
import { runEpisode } from '../training/episode';
import { sha256Hex } from '../training/fsio';
import type { AtlasSeedResult, AtlasWorkerMessage, AtlasWorkerTask } from './atlas-evaluate';

/**
 * `atlas-evaluate.ts`'s child process, `.agents/plans/lesion-atlas/02-atlas-computation.md`'s
 * WP2 worker: `node:child_process.fork`ed once per shard by the parent's
 * `runShardedEvaluation` (reused, not duplicated -- see `atlas-evaluate.ts`'s
 * import of it and the `AtlasWorkerTask` doc comment for why that reuse is
 * safe despite `runShardedEvaluation` being written against `NullWorkerTask`/
 * `NullSeedResult`/`NullWorkerMessage`, not a generic).
 *
 * Unlike `scripts/null/null-worker.ts`, this worker never derives a graph
 * mode (no `disconnected` control, no rewiring-at-runtime): the atlas has
 * exactly two graphs, both already-compiled artifacts on disk
 * (`malecns-arena-v1.bin.gz` and `malecns-arena-v1-rewired-seed0.bin.gz`),
 * so a task's `path` is loaded and parsed directly. `task.mode` is carried
 * along purely so `AtlasWorkerTask` satisfies `NullWorkerTask`'s shape for
 * `runShardedEvaluation`'s type signature -- this file never reads it.
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

/**
 * Read, decompress, and sha256-verify a graph binary against
 * `expectedSha256` before it is ever fed into an episode -- the same
 * "verify what this process actually loaded" second check
 * `null-worker.ts`'s identically-purposed (but not exported, hence this
 * small independent copy) helper performs, on top of `atlas-evaluate.ts`'s
 * own up-front verification of every graph file before any shard is
 * forked.
 */
const loadVerifiedGraphBinary = (path: string, expectedSha256: string): ArrayBuffer => {
  const gzipBytes = readFileSync(path);
  const binary = gunzipSync(gzipBytes);
  const actualSha256 = sha256Hex(binary);
  if (actualSha256 !== expectedSha256) {
    throw new Error(
      `atlas-worker: ${path} decompressed sha256 ${actualSha256} does not match expected ${expectedSha256}`
    );
  }
  return binary.buffer.slice(binary.byteOffset, binary.byteOffset + binary.byteLength);
};

const runTask = (task: AtlasWorkerTask): readonly AtlasSeedResult[] => {
  const graphBinary = loadVerifiedGraphBinary(task.path, task.expectedSha256);
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
      // error) previously surfaced with whatever message runEpisode itself
      // produced, with no indication of which graph/lesion/seed in a
      // ~200,000-episode run hit it -- re-thrown here naming all three, the
      // same as the non-finite-score check below already does (a
      // dual-review finding).
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`atlas-worker: graph "${task.graphId}" lesion=${lesionLabel} seed ${seed}: ${message}`);
    }
    const { movementScore, foodPickups, hazardContacts } = result.left;
    // Same reasoning as null-worker.ts: a NaN/Infinity score would
    // silently become `null` under JSON.stringify and then `0` wherever
    // this is later summed (pairedStats' diffs, the bootstrap resamples) --
    // fail here, where the graph, lesion index, and seed that produced it
    // are still known, naming all three per this WP's own acceptance
    // criterion.
    if (![movementScore, foodPickups, hazardContacts].every(Number.isFinite)) {
      throw new Error(
        `atlas-worker: graph "${task.graphId}" lesion=${lesionLabel} ` +
          `seed ${seed} produced a non-finite score (movementScore=${movementScore}, ` +
          `foodPickups=${foodPickups}, hazardContacts=${hazardContacts})`
      );
    }
    return { seed, movementScore, foodPickups, hazardContacts };
  });
};

process.on('message', (task: AtlasWorkerTask) => {
  try {
    // Defends this process against receiving a task shaped like a plain
    // `NullWorkerTask` (no `lesionIndex` field) instead of a real
    // `AtlasWorkerTask` -- e.g. a future refactor that has this same
    // `process.on('message', ...)` handler double as a generic worker, or a
    // caller that constructs a task by hand and gets the shape wrong. Note
    // what this does *not* catch: `atlas-evaluate.ts`'s `workerPath`
    // pointed at `null-worker.ts` by mistake never reaches this file at
    // all -- that would fork `null-worker.ts`'s own process and handler
    // instead, which has no equivalent check (a round-2 dual-review
    // finding: an earlier version of this comment claimed that scenario
    // incorrectly). Cast through `Record<string, unknown>` rather than
    // checking `task` (typed `AtlasWorkerTask`, which always statically has
    // `lesionIndex`) directly: TypeScript would otherwise narrow the
    // negative branch to `never`, since nothing at compile time can make
    // this IPC payload actually be something else -- the whole point is to
    // check what a real, untyped `process.on('message', ...)` payload has
    // at runtime.
    const rawTask = task as unknown as Record<string, unknown>;
    if (!('lesionIndex' in rawTask)) {
      throw new Error(`atlas-worker: task "${String(rawTask.graphId)}" has no lesionIndex -- not an atlas task (wrong workerPath?)`);
    }
    const results = runTask(task);
    const message: AtlasWorkerMessage = { type: 'result', graphId: task.graphId, results };
    process.send?.(message);
  } catch (error) {
    const message: AtlasWorkerMessage = {
      type: 'error',
      graphId: task.graphId,
      message: error instanceof Error ? error.message : String(error)
    };
    process.send?.(message);
  }
});
