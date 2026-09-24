import { createHash, randomBytes } from 'node:crypto';
import { renameSync, unlinkSync, writeFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';

/**
 * Small filesystem/hashing helpers shared by `scripts/training/evaluate.ts`
 * and every script under `scripts/null/`. Both `sha256Hex` and
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
