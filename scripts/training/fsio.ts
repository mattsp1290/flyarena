import { execFileSync } from 'node:child_process';
import { createHash, randomBytes } from 'node:crypto';
import { renameSync, unlinkSync, writeFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';

/**
 * Small filesystem/hashing/git helpers shared by `scripts/training/evaluate.ts`
 * and every script under `scripts/null/`. `sha256Hex` and
 * `atomicWriteFileSync` had drifted into multiple near-identical per-file
 * copies within the null-evaluation pipeline (the four `sha256Hex`s in
 * `evaluate.ts`/`null-evaluate.ts`/`null-worker.ts`/`null-report.ts`, plus
 * `null-report.ts`'s own `atomicWriteFileSync`) — consolidated here per the
 * round-2 thermo-maintainability review, matching the precedent
 * `scripts/training/cli.ts` already set for shared argv helpers.
 *
 * Two further `sha256Hex` copies exist outside this review's scope
 * (`scripts/training/export-arms.ts`, `scripts/training/run-dir.ts`) and
 * were deliberately left as-is here — not migrated to this module — since
 * consolidating them wasn't part of the finding this module was created to
 * address. A future pass touching either file should prefer importing
 * `sha256Hex` from here over adding a fifth/sixth copy.
 *
 * `gitRev` moved here from `scripts/null/null-trained-evaluate.ts` (WP4 of
 * `.agents/plans/pathway-interventions`, thermo-methodology review I2): the
 * authored side (`null-evaluate.ts`) needed the identical helper to stamp
 * `evaluatorGitRev` on its own `--graph-list` output, but `null-evaluate.ts`
 * is imported *by* `null-trained-evaluate.ts` (`import { runCliMain, ... }
 * from './null-evaluate'`), so importing `gitRev` the other way would have
 * created a load-time circular import. This module has no dependency on
 * either null-evaluation file, so both can import it without a cycle.
 */

/** sha256 of `data`, hex-encoded. Accepts either raw bytes (a graph binary, a JSON artifact's UTF-8 bytes) or a string directly. */
export const sha256Hex = (data: Uint8Array | string): string => createHash('sha256').update(data).digest('hex');

/**
 * Write `contents` to `path` via a same-directory temp file + `renameSync`
 * (POSIX rename is atomic), so a process killed mid-write leaves either the
 * previous file or nothing — never a truncated one. Matches
 * `scripts/data/fsutil.py`'s `atomic_write_text` convention, which
 * `positions.py`/`rewire_batch.py` already use for the files this pipeline's
 * output sits alongside.
 */
export const atomicWriteFileSync = (path: string, contents: string): void => {
  const tmpPath = resolve(dirname(path), `.${basename(path)}.${randomBytes(6).toString('hex')}.tmp`);
  try {
    writeFileSync(tmpPath, contents);
    renameSync(tmpPath, path);
  } catch (error) {
    try {
      unlinkSync(tmpPath);
    } catch {
      // tmpPath was never created, or was already cleaned up — nothing more to do.
    }
    throw error;
  }
};

/**
 * `git rev-parse HEAD` run in `cwd`, `null` on any failure (not a git
 * checkout, `git` missing, a dirty/detached edge case `execFileSync` itself
 * rejects) — informational, matches `training/src/flyarena_training/cli.py`'s
 * `_git_rev` convention. Takes `cwd` explicitly (rather than closing over a
 * module-private `repoRoot`, `null-trained-evaluate.ts`'s original shape)
 * so every caller in this shared module passes its own repo root, with no
 * implicit dependency on which file happened to define it first.
 */
export const gitRev = (cwd: string): string | null => {
  try {
    return execFileSync('git', ['rev-parse', 'HEAD'], { cwd, encoding: 'utf8' }).trim();
  } catch {
    return null;
  }
};
