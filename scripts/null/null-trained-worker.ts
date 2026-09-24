import { readFileSync } from 'node:fs';

import type { ArmName } from '../training/arms';
import { computeArmBundleSha256, deserializeArmBundle, type SerializedArmBundle } from '../training/export-arms';
import { runEpisode } from '../training/episode';
import { readRunDir, type LoadedRun } from '../training/run-dir';
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
  /**
   * Structural identity this run directory must actually have, verified in
   * `runTask` before it is scored (mirroring `evaluate.ts`'s
   * `validateRunSubsteps`/`validateRunArmBundleSha256`, which this script
   * previously did not replicate — a dual-review finding). A run directory
   * this script silently trusted without these checks could be scored
   * against the wrong trainer seed, arm, substep count, or hidden size with
   * no error anywhere: `runEpisode` happily runs `trained` at whatever
   * `substeps` it is given (defaulting to `NEURAL_SUBSTEPS_PER_TICK` if
   * omitted — `episode.ts`), so a stale/mismatched run would be scored
   * "successfully" at the wrong K rather than failing loudly.
   */
  readonly expectedArm: ArmName;
  readonly expectedTrainerSeed: number;
  readonly expectedSubsteps: number;
  readonly expectedHiddenSize: number;
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

/**
 * Every field a run directory's `config.json` must actually have for this
 * task to be the run this script's caller believes it is scoring. Unlike
 * `reconcileCemConfig` in `null-trained-evaluate.ts` (which only *warns* on
 * a CEM-hyperparameter mismatch, matching `run-dir.ts`'s existing
 * `deriveTrainingBlock` precedent for informational reconciliation), a
 * mismatch on any of these four fields means this run directory is not
 * structurally what it was asked to be -- scoring it anyway would silently
 * publish a number for the wrong arm/trainer-seed/substep-count/hidden-size,
 * so each one is a hard failure, not a warning.
 */
const assertRunMatchesExpectedIdentity = (run: Readonly<LoadedRun>, task: Readonly<NullTrainedWorkerTask>): void => {
  if (run.config.arm !== task.expectedArm) {
    throw new Error(
      `null-trained-worker: ${task.runDir}/config.json has arm "${run.config.arm}", expected "${task.expectedArm}"`
    );
  }
  if (run.config.trainerSeed !== task.expectedTrainerSeed) {
    throw new Error(
      `null-trained-worker: ${task.runDir}/config.json has trainerSeed ${run.config.trainerSeed}, expected ${task.expectedTrainerSeed}`
    );
  }
  if (run.config.substeps !== task.expectedSubsteps) {
    throw new Error(
      `null-trained-worker: ${task.runDir}/config.json was trained at substeps=${run.config.substeps}, but this ` +
        `evaluation expects substeps=${task.expectedSubsteps} (NEURAL_SUBSTEPS_PER_TICK) -- a rerun trained at a ` +
        'different substep count would otherwise be silently scored as if it were a like-for-like comparison'
    );
  }
  if (run.config.H !== task.expectedHiddenSize) {
    throw new Error(
      `null-trained-worker: ${task.runDir}/config.json has H=${run.config.H}, expected ${task.expectedHiddenSize}`
    );
  }
};

/** Exported so tests can exercise the run-directory/bundle validation and scoring logic directly, without going through `node:child_process.fork`'s IPC wire protocol. */
export const runTask = (task: NullTrainedWorkerTask): readonly NullSeedResult[] => {
  const run = readRunDir(task.runDir);
  const bundle = loadVerifiedBundle(task.armBundlePath);

  assertRunMatchesExpectedIdentity(run, task);

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
      // Explicit, not omitted-and-defaulted: `episode.ts` defaults
      // `substeps` to `NEURAL_SUBSTEPS_PER_TICK` when omitted, which would
      // happen to be correct for every run this study trains but would mask
      // the very mismatch `assertRunMatchesExpectedIdentity` above already
      // caught -- passing it explicitly means a future change to the
      // default can never silently change what this scores at.
      substeps: task.expectedSubsteps,
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
