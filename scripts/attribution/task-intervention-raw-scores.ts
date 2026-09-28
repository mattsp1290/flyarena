import { readFileSync } from 'node:fs';

import { copyVerbatim } from './copy-verbatim';
import type { ArmBundleCheckable, RawInterventionRun } from './raw-intervention-scores';

/**
 * `training/archive/task-intervention-trained-raw-<arenaTask>-v1.json`
 * (WP1b, `.agents/plans/readout-attribution/01-archive-and-types.md`): a
 * byte-for-byte copy of one arena task's own `null-trained-evaluate.ts
 * --graph-list` rescore output -- the exact `flyarena-s8z8` worktree
 * `training/runs/tasks/<task>/trained.json`, one file per task (never
 * merged into one file: each is independently a verbatim copy, so each
 * one's own committed sha256 stays checkable against the source file it was
 * copied from, matching `raw-intervention-scores.ts`'s WP1 precedent for
 * `intervention-trained-raw-v1.json`).
 *
 * Reuses `raw-intervention-scores.ts`'s `RawInterventionRun`/
 * `ArmBundleCheckable` shapes rather than redeclaring near-identical ones:
 * `null-trained-evaluate-graph-list.ts`'s per-run row (`id`, `trainerSeed`,
 * `gzipSha256`, `armBundleSha256`, `movementScore`, ...) is structurally the
 * same whether it came from the default task's `--graph-list` rescore
 * (WP1's `intervention-trained-raw-v1.json`) or a per-task one -- only the
 * *file* now also carries a top-level `arenaTask`, which this module checks
 * explicitly (`RawInterventionRun`'s own validation, reused via
 * `readRawInterventionScores`'s per-run field checks, still applies to each
 * row).
 */

export interface TaskInterventionRawFile {
  readonly arenaTask: string;
  readonly arenaTaskFingerprint?: string;
  readonly graphListSha256: string;
  readonly runs: readonly RawInterventionRun[];
}

/**
 * Read+validate (not copy) -- split out so `archive-readouts.ts` can
 * cross-check `--source` task-intervention additions against this file's
 * `(id, trainerSeed) -> armBundleSha256` map, and its `graphListSha256`
 * against the intervention index actually in use for this invocation,
 * before either is written (mirrors `raw-intervention-scores.ts`'s
 * `readRawInterventionScores` split).
 */
export const readTaskInterventionRawScores = (
  sourcePath: string,
  expectedArenaTask: string
): { readonly rawBytes: Buffer; readonly parsed: TaskInterventionRawFile } => {
  const rawBytes = readFileSync(sourcePath);
  const parsed = JSON.parse(rawBytes.toString('utf8')) as Partial<TaskInterventionRawFile>;
  if (parsed.arenaTask !== expectedArenaTask) {
    throw new Error(
      `archive-readouts: ${sourcePath} has arenaTask ${JSON.stringify(parsed.arenaTask)}, expected ` +
        `${JSON.stringify(expectedArenaTask)} (the --task-intervention-raw-scores label)`
    );
  }
  if (typeof parsed.graphListSha256 !== 'string' || parsed.graphListSha256.length === 0) {
    throw new Error(`archive-readouts: ${sourcePath} has no "graphListSha256"`);
  }
  if (!Array.isArray(parsed.runs) || parsed.runs.length === 0) {
    throw new Error(`archive-readouts: ${sourcePath} has no "runs" array`);
  }
  for (const run of parsed.runs) {
    if (
      typeof run.id !== 'string' ||
      run.id.length === 0 ||
      typeof run.trainerSeed !== 'number' ||
      !Array.isArray(run.movementScore)
    ) {
      throw new Error(`archive-readouts: ${sourcePath} has a malformed run entry: ${JSON.stringify(run)}`);
    }
  }
  return { rawBytes, parsed: parsed as TaskInterventionRawFile };
};

export const copyTaskInterventionRawScores = (sourcePath: string, outPath: string, expectedArenaTask: string): void => {
  const { rawBytes } = readTaskInterventionRawScores(sourcePath, expectedArenaTask);
  copyVerbatim(sourcePath, outPath, rawBytes);
};

export interface TaskArmBundleCheckable extends ArmBundleCheckable {
  readonly arenaTask: string;
}

/**
 * Same cross-check as `raw-intervention-scores.ts`'s
 * `assertArmBundlesMatchRawScores`, generalized to `kind: "task-intervention"`
 * entries filtered to one `arenaTask` at a time (one raw-scores file only
 * ever covers one task -- see this module's own doc comment). Silently
 * skips an addition with no matching raw entry, matching the WP1 precedent
 * (a bigq/default-task-intervention addition, or a different task's
 * addition, is simply not this file's concern).
 */
export const assertTaskArmBundlesMatchRawScores = (
  additions: readonly TaskArmBundleCheckable[],
  arenaTask: string,
  rawRuns: readonly RawInterventionRun[]
): void => {
  const rawByKey = new Map(rawRuns.map((run) => [`${run.id}\u0000${run.trainerSeed}`, run]));
  for (const entry of additions) {
    if (entry.kind !== 'task-intervention' || entry.arenaTask !== arenaTask) continue;
    const match = rawByKey.get(`${entry.graphId}\u0000${entry.trainerSeed}`);
    if (match && typeof match.armBundleSha256 === 'string' && match.armBundleSha256 !== entry.armBundleSha256) {
      throw new Error(
        `archive-readouts: --source ${entry.graphId} (trainerSeed ${entry.trainerSeed}, arenaTask ` +
          `"${arenaTask}") has armBundleSha256 ${entry.armBundleSha256}, but the raw task-intervention scores ` +
          `file records ${match.armBundleSha256} for id "${entry.graphId}" -- mismatched --source label, or ` +
          'wrong run directory?'
      );
    }
  }
};
