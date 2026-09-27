import { readFileSync } from 'node:fs';
import { dirname } from 'node:path';

import { readGraphListIndex, type GraphListIndex } from '../null/graph-list-index';
import { copyVerbatim } from './copy-verbatim';

/**
 * `training/archive/intervention-index-v1.json`: `graph-list-index.ts`'s
 * `GraphListIndex` (`scripts/analysis/interventions.py`'s `index.json`),
 * copied byte-for-byte (never re-serialized) so this file's own sha256 stays
 * equal to `pathway-interventions-v1.json`'s recorded `sources.indexSha` --
 * the one independent, already-published check this archive's copy can be
 * verified against -- and so WP2's regenerated intervention graphs can be
 * sha-checked against this repository's own committed record, without
 * depending on any worktree.
 *
 * Split out of `archive-readouts.ts` (a thermo-maintainability review
 * finding) into its own module.
 *
 * Validated with `graph-list-index.ts`'s own `readGraphListIndex` --
 * `null-evaluate.ts`'s `--graph-list` mode already reads and validates
 * exactly this file format (entry shape, reserved-id rejection, duplicate-id
 * rejection, path-traversal rejection), so this module reuses that canonical
 * parser rather than a second, weaker, ad hoc check (a thermo-maintainability
 * review finding: an earlier version of this file only checked that
 * `sourceSha256` was a string and `entries` was non-empty, missing all four
 * of the above).
 */

/** Read+validate (not copy) -- split out so `runArchiveReadouts` can validate this input before writing anything, and so `graph-identity.ts` can resolve individual entries' shas from the same parsed index, before either is written. */
export const readAndValidateInterventionIndex = (
  sourcePath: string
): { readonly rawBytes: Buffer; readonly index: GraphListIndex; readonly indexDir: string } => {
  const index = readGraphListIndex(sourcePath); // throws on any structural/id/path-traversal problem
  const rawBytes = readFileSync(sourcePath);
  return { rawBytes, index, indexDir: dirname(sourcePath) };
};

export const copyInterventionIndex = (sourcePath: string, outPath: string): void => {
  const { rawBytes } = readAndValidateInterventionIndex(sourcePath);
  copyVerbatim(sourcePath, outPath, rawBytes);
};
