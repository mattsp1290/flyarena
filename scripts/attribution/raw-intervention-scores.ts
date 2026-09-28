import { readFileSync } from 'node:fs';

import { copyVerbatim } from './copy-verbatim';

/**
 * `training/archive/intervention-trained-raw-v1.json` (WP1) and
 * `training/archive/task-intervention-trained-raw-<arenaTask>-v1.json`
 * (WP1b): a byte-for-byte copy of `null-trained-evaluate-graph-list.ts`'s
 * `NullTrainedInterventionEvaluationRaw` (id-keyed `runs`) -- the exact raw
 * output of a `--graph-list`/`--runs` rescore, whether the default task's
 * (the hbru worktree's `trained.json`, WP1) or one non-default arena task's
 * (a `flyarena-s8z8` worktree `trained.json`, WP1b, one file per task --
 * never merged into one file, so each stays independently checkable against
 * its own recorded sha256). This is the only place the published
 * `pathway-interventions-v1.json` artifact's sorted, unlabeled C/M score
 * arrays (WP1) or a task's per-run means (WP1b) can be traced back to a
 * specific `(id, trainerSeed)` pair. Copied (not re-derived) so that id
 * mapping survives the source worktree being pruned.
 *
 * Split out of `archive-readouts.ts` (a thermo-maintainability review
 * finding: that file bundled four independent artifact-builders behind one
 * 806-line CLI) into its own module, since this concern doesn't depend on
 * `ArchivedReadout`'s schema and is already independently tested/importable.
 *
 * WP1b originally added a second, near-duplicate module
 * (`task-intervention-raw-scores.ts`) for the per-task variant -- a
 * thermo-maintainability review finding: ~80% of that module was a
 * byte-for-byte copy of this one's own validation loop and cross-check,
 * differing only in the extra top-level `arenaTask`/`graphListSha256`
 * checks and the `kind`/`arenaTask` filter predicate. Both concerns are one
 * module, parameterized by `RawScoresFileOptions`/`ArmBundleMatchOptions`
 * below, not two.
 */

/**
 * `gzipSha256` is present on every real `null-trained-evaluate-graph-list.ts`
 * row (`NullTrainedInterventionGraphRaw`), declared optional here only
 * because WP1's original default-task raw-scores validation never checked
 * for it (a real gap the WP1b variant closes with `expectedArenaTask` set --
 * see `readRawInterventionScores`'s doc comment).
 */
export interface RawInterventionRun {
  readonly id: string;
  readonly trainerSeed: number;
  readonly armBundleSha256?: string;
  readonly gzipSha256?: string;
  readonly movementScore: readonly number[];
}

/**
 * The raw-scores file's own top-level shape. `arenaTask`/`arenaTaskFingerprint`/
 * `graphListSha256` are present only on a WP1b per-task file (absent on
 * WP1's default-task `intervention-trained-raw-v1.json`, which predates
 * arena tasks and per-run graph-list identity at this file's top level).
 */
export interface RawScoresFile {
  readonly arenaTask?: string;
  readonly arenaTaskFingerprint?: string;
  readonly graphListSha256?: string;
  readonly runs: readonly RawInterventionRun[];
}

export interface RawScoresFileOptions {
  /**
   * Present only for a WP1b per-task raw-scores file: the top-level
   * `arenaTask` field must equal this value, `graphListSha256` must be a
   * non-empty string, and every row must carry a non-empty
   * `armBundleSha256`/`gzipSha256` (a thermo-provenance review finding --
   * tighter than the WP1 default-task file's validation below, which stays
   * loose to match what WP1 already committed and tested: `armBundleSha256`
   * only checked when present, `gzipSha256` not checked at all). Absent
   * (the default) means this is a WP1 default-task file, and none of the
   * above is checked.
   */
  readonly expectedArenaTask?: string;
}

/**
 * Read+validate (not copy) -- split out from `copyRawInterventionScores` so
 * `runArchiveReadouts` can cross-check `--source` additions against this
 * file's `(id, trainerSeed) -> armBundleSha256` map (and, for a per-task
 * file, its `graphListSha256` against the intervention index actually in
 * use for this invocation) in the same invocation, before either is
 * written.
 */
export const readRawInterventionScores = (
  sourcePath: string,
  options: RawScoresFileOptions = {}
): { readonly rawBytes: Buffer; readonly runs: readonly RawInterventionRun[]; readonly parsed: RawScoresFile } => {
  const rawBytes = readFileSync(sourcePath);
  const parsed = JSON.parse(rawBytes.toString('utf8')) as Partial<RawScoresFile>;
  const { expectedArenaTask } = options;

  if (expectedArenaTask !== undefined) {
    if (parsed.arenaTask !== expectedArenaTask) {
      throw new Error(
        `archive-readouts: ${sourcePath} has arenaTask ${JSON.stringify(parsed.arenaTask)}, expected ` +
          `${JSON.stringify(expectedArenaTask)} (the --task-intervention-raw-scores label)`
      );
    }
    if (typeof parsed.graphListSha256 !== 'string' || parsed.graphListSha256.length === 0) {
      throw new Error(`archive-readouts: ${sourcePath} has no "graphListSha256"`);
    }
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
    if (expectedArenaTask !== undefined) {
      // A thermo-provenance review finding: a per-task raw-scores file's
      // rows are what `assertArmBundlesMatchRawScores`'s cross-check and
      // `archive-gate1-task.test.ts`'s per-row identity checks depend on --
      // a row silently missing `armBundleSha256`/`gzipSha256` would
      // silently lose cross-check coverage for that row rather than fail
      // loudly here, at read time.
      if (typeof run.armBundleSha256 !== 'string' || run.armBundleSha256.length === 0) {
        throw new Error(
          `archive-readouts: ${sourcePath} run "${run.id}" (trainerSeed ${run.trainerSeed}) has no armBundleSha256`
        );
      }
      if (typeof run.gzipSha256 !== 'string' || run.gzipSha256.length === 0) {
        throw new Error(
          `archive-readouts: ${sourcePath} run "${run.id}" (trainerSeed ${run.trainerSeed}) has no gzipSha256`
        );
      }
    }
  }
  return { rawBytes, runs: parsed.runs, parsed: parsed as RawScoresFile };
};

export const copyRawInterventionScores = (sourcePath: string, outPath: string, options: RawScoresFileOptions = {}): void => {
  const { rawBytes } = readRawInterventionScores(sourcePath, options);
  copyVerbatim(sourcePath, outPath, rawBytes);
};

/**
 * Cross-check every addition's `armBundleSha256` against the raw scores
 * file's own `(graphId, trainerSeed) -> armBundleSha256` mapping -- a
 * stronger check than `assertArmMatchesGraphId` alone (which cannot
 * distinguish e.g. a `C000` label from a `C001` run, since both are
 * `arm: "rewired"`), available whenever `--source` and
 * `--raw-intervention-scores`/`--task-intervention-raw-scores` are supplied
 * together in one invocation (a dual-review finding). Silently skips an
 * addition with no matching raw entry (e.g. a bigq addition, or an id the
 * raw file doesn't cover) rather than requiring full coverage in either
 * direction.
 *
 * Generic over the addition shape (`readonly {kind, graphId, trainerSeed,
 * armBundleSha256, arenaTask?}[]`) rather than importing `ArchivedReadout`
 * from `archive-readouts.ts`, so this module has no dependency on that one
 * (it would otherwise be the only thing pulling the full readout schema
 * into a module whose own concern is the raw-scores file).
 */
export interface ArmBundleCheckable {
  readonly kind: string;
  readonly graphId: string;
  readonly trainerSeed: number;
  readonly armBundleSha256: string;
  readonly arenaTask?: string;
}

export interface ArmBundleMatchOptions {
  /** Which `ArchivedReadout.kind` this call is cross-checking. Default `'intervention'` (WP1's default-task case). */
  readonly kind?: string;
  /**
   * When set (WP1b's per-task case, `kind: 'task-intervention'`), restrict
   * the cross-check to additions whose own `arenaTask` equals this -- one
   * raw-scores file only ever covers one arena task (this module's own doc
   * comment). Additions for a DIFFERENT `arenaTask` are treated exactly
   * like a `kind` mismatch: silently skipped, not cross-checked by this
   * call (a separate call, with this file's own `arenaTask`, covers them).
   */
  readonly arenaTask?: string;
}

export const assertArmBundlesMatchRawScores = (
  additions: readonly ArmBundleCheckable[],
  rawRuns: readonly RawInterventionRun[],
  options: ArmBundleMatchOptions = {}
): void => {
  const kind = options.kind ?? 'intervention';
  const rawByKey = new Map(rawRuns.map((run) => [`${run.id}\u0000${run.trainerSeed}`, run]));
  for (const entry of additions) {
    if (entry.kind !== kind) continue;
    if (options.arenaTask !== undefined && entry.arenaTask !== options.arenaTask) continue;
    const match = rawByKey.get(`${entry.graphId}\u0000${entry.trainerSeed}`);
    if (match && typeof match.armBundleSha256 === 'string' && match.armBundleSha256 !== entry.armBundleSha256) {
      const taskContext = options.arenaTask !== undefined ? `, arenaTask "${options.arenaTask}"` : '';
      throw new Error(
        `archive-readouts: --source ${entry.graphId} (trainerSeed ${entry.trainerSeed}${taskContext}) has ` +
          `armBundleSha256 ${entry.armBundleSha256}, but the raw ${kind === 'task-intervention' ? 'task-intervention' : 'intervention'} ` +
          `scores file records ${match.armBundleSha256} for id "${entry.graphId}" -- mismatched --source label, ` +
          'or wrong run directory?'
      );
    }
  }
};
