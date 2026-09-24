import { readFileSync } from 'node:fs';

import { computeArmBundleSha256, deserializeArmBundle, type SerializedArmBundle } from '../training/export-arms';
import { runEpisode } from '../training/episode';
import { readRunDir } from '../training/run-dir';
import type { NullSeedResult, NullWorkerMessage } from './null-worker';

/**
 * `.agents/plans/rewiring-null/03-trained-sample.md`'s WP3 child process:
 * `null-trained-evaluate.ts`'s `node:child_process.fork`ed worker. Scores
 * one CEM-trained readout — a `run-dir.ts` run directory's
 * `theta_final.npy`, loaded against its own `export-arms.ts` bundle graph —
 * with decoder `trained`, opponent parked, over every held-out seed.
 *
 * Deliberately reuses `null-worker.ts`'s `NullSeedResult`/`NullWorkerMessage`
 * types directly rather than redeclaring near-identical ones: both workers
 * report the same per-seed shape (`seed`/`movementScore`/`foodPickups`/
 * `hazardContacts`) over the same `{type: 'result'|'error', graphId, ...}`
 * wire protocol, so `null-evaluate.ts`'s generic `runShardedEvaluation` can
 * drive this worker with the exact same sharding/fork/failure-handling
 * mechanism it already uses for `null-worker.ts` (see that function's doc
 * comment) — only the *task* shape below differs (a run directory + arm
 * bundle path, not a gzip graph path + sha256).
 */

export interface NullTrainedWorkerTask {
  readonly graphId: string;
  /** A `run-dir.ts` run directory: `config.json` + `theta_final.npy` (+ optional `env.json`/`generations.csv`). */
  readonly runDir: string;
  /** The `export-arms.ts` bundle this run was trained against (a `.../rewired.json` or `.../biological.json`). */
  readonly armBundlePath: string;
  readonly heldOutSeeds: readonly number[];
  readonly ticks: number;
}

/**
 * Load and integrity-check one arm bundle exactly as `evaluate.ts`'s
 * `loadArmGraphs` does: recompute its self-certifying sha256 over its own
 * content and reject a tampered/stale file, rather than trusting the
 * bundle's own declared `sha256` field.
 */
const loadVerifiedBundle = (path: string): SerializedArmBundle => {
  const bundle = JSON.parse(readFileSync(path, 'utf8')) as SerializedArmBundle;
  const { sha256, ...withoutHash } = bundle;
  if (computeArmBundleSha256(withoutHash) !== sha256) {
    throw new Error(`null-trained-worker: ${path}'s sha256 does not match its content (tampered or stale file?)`);
  }
  return bundle;
};

const runTask = (task: NullTrainedWorkerTask): readonly NullSeedResult[] => {
  const run = readRunDir(task.runDir);
  const bundle = loadVerifiedBundle(task.armBundlePath);

  // `config.armBundleSha256` is required present for every run this study
  // trains (flyarena-train always writes it -- REQUIRED_BUNDLE_FIELDS in
  // training/src/flyarena_training/cli.py gates every bundle load). Unlike
  // evaluate.ts's `validateRunArmBundleSha256` (which treats this field as
  // optional, for older run directories that predate it), a missing value
  // here means this run directory does not belong to this study -- fail
  // loudly rather than silently skip the integrity check.
  if (!run.config.armBundleSha256) {
    throw new Error(`null-trained-worker: ${task.runDir}/config.json has no armBundleSha256`);
  }
  if (run.config.armBundleSha256 !== bundle.sha256) {
    throw new Error(
      `null-trained-worker: ${task.runDir}/config.json was trained against arm bundle sha256 ` +
        `${run.config.armBundleSha256}, but ${task.armBundlePath} is ${bundle.sha256}`
    );
  }

  const graph = deserializeArmBundle(bundle);

  return task.heldOutSeeds.map((seed) => {
    const result = runEpisode({
      seed,
      ticks: task.ticks,
      left: { decoder: 'trained', graph, weights: run.weights },
      right: { decoder: 'parked' }
    });
    const { movementScore, foodPickups, hazardContacts } = result.left;
    // See null-worker.ts's identical check: a NaN/Infinity score would
    // silently round-trip through JSON as null/be summed as 0 downstream.
    if (![movementScore, foodPickups, hazardContacts].every(Number.isFinite)) {
      throw new Error(
        `null-trained-worker: ${task.graphId} seed ${seed} produced a non-finite score ` +
          `(movementScore=${movementScore}, foodPickups=${foodPickups}, hazardContacts=${hazardContacts})`
      );
    }
    return { seed, movementScore, foodPickups, hazardContacts };
  });
};

process.on('message', (task: NullTrainedWorkerTask) => {
  try {
    const results = runTask(task);
    const message: NullWorkerMessage = { type: 'result', graphId: task.graphId, results };
    process.send?.(message);
  } catch (error) {
    const message: NullWorkerMessage = {
      type: 'error',
      graphId: task.graphId,
      message: error instanceof Error ? error.message : String(error)
    };
    process.send?.(message);
  }
});
