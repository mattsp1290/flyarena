import { readFileSync } from 'node:fs';

import { copyVerbatim } from './copy-verbatim';

/**
 * `training/archive/intervention-trained-raw-v1.json`: a byte-for-byte copy
 * of `null-trained-evaluate-graph-list.ts`'s
 * `NullTrainedInterventionEvaluationRaw` (id-keyed `runs`), the exact raw
 * output of the hbru worktree's `--graph-list`/`--runs` rescore -- the only
 * place the published `pathway-interventions-v1.json` artifact's sorted,
 * unlabeled C/M score arrays can be traced back to a specific
 * `(id, trainerSeed)` pair. Copied (not re-derived) so that id mapping
 * survives the hbru worktree being pruned. Only lightly validated here (has
 * a non-empty `runs` array of well-shaped entries) -- this is a copy of an
 * already-produced, already cross-checked artifact, not a second
 * independent computation.
 *
 * Split out of `archive-readouts.ts` (a thermo-maintainability review
 * finding: that file bundled four independent artifact-builders behind one
 * 806-line CLI) into its own module, since this concern doesn't depend on
 * `ArchivedReadout`'s schema and is already independently tested/importable.
 */
export interface RawInterventionRun {
  readonly id: string;
  readonly trainerSeed: number;
  readonly armBundleSha256?: string;
  readonly movementScore: readonly number[];
}

/** Read+validate (not copy) -- split out from `copyRawInterventionScores` so `runArchiveReadouts` can cross-check `--source` additions against this file's `(id, trainerSeed) -> armBundleSha256` map in the same invocation, before either is written. */
export const readRawInterventionScores = (
  sourcePath: string
): { readonly rawBytes: Buffer; readonly runs: readonly RawInterventionRun[] } => {
  const rawBytes = readFileSync(sourcePath);
  const parsed = JSON.parse(rawBytes.toString('utf8')) as { readonly runs?: readonly RawInterventionRun[] };
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
  return { rawBytes, runs: parsed.runs };
};

export const copyRawInterventionScores = (sourcePath: string, outPath: string): void => {
  const { rawBytes } = readRawInterventionScores(sourcePath);
  copyVerbatim(sourcePath, outPath, rawBytes);
};

/**
 * Cross-check every `kind: "intervention"` addition's `armBundleSha256`
 * against the raw scores file's own `(graphId, trainerSeed) -> armBundleSha256`
 * mapping -- a stronger check than `assertArmMatchesGraphId` alone (which
 * cannot distinguish e.g. a `C000` label from a `C001` run, since both are
 * `arm: "rewired"`), available whenever `--source` and
 * `--raw-intervention-scores` are supplied together in one invocation (a
 * dual-review finding). Silently skips an addition with no matching raw
 * entry (e.g. a bigq addition, or an intervention id the raw file doesn't
 * cover) rather than requiring full coverage in either direction.
 *
 * Generic over the addition shape (`readonly {kind, graphId, trainerSeed,
 * armBundleSha256}[]`) rather than importing `ArchivedReadout` from
 * `archive-readouts.ts`, so this module has no dependency on that one (it
 * would otherwise be the only thing pulling the full readout schema into a
 * module whose own concern is the raw-scores file).
 */
export interface ArmBundleCheckable {
  readonly kind: string;
  readonly graphId: string;
  readonly trainerSeed: number;
  readonly armBundleSha256: string;
}

export const assertArmBundlesMatchRawScores = (
  additions: readonly ArmBundleCheckable[],
  rawRuns: readonly RawInterventionRun[]
): void => {
  const rawByKey = new Map(rawRuns.map((run) => [`${run.id}\u0000${run.trainerSeed}`, run]));
  for (const entry of additions) {
    if (entry.kind !== 'intervention') continue;
    const match = rawByKey.get(`${entry.graphId}\u0000${entry.trainerSeed}`);
    if (match && typeof match.armBundleSha256 === 'string' && match.armBundleSha256 !== entry.armBundleSha256) {
      throw new Error(
        `archive-readouts: --source ${entry.graphId} (trainerSeed ${entry.trainerSeed}) has armBundleSha256 ` +
          `${entry.armBundleSha256}, but the raw intervention scores file records ${match.armBundleSha256} for ` +
          `id "${entry.graphId}" -- mismatched --source label, or wrong run directory?`
      );
    }
  }
};
