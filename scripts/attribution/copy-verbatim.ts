import { randomBytes } from 'node:crypto';
import { mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';

import { sha256Hex } from '../training/fsio';

/**
 * Shared between `raw-intervention-scores.ts` and `intervention-index.ts`:
 * both copy an externally-produced JSON artifact into `training/archive/`
 * byte-for-byte, so each one's own committed sha256 continues to match any
 * externally-recorded sha (e.g. `pathway-interventions-v1.json`'s
 * `sources.indexSha`) -- a round-1 dual-review finding.
 *
 * `atomicWriteBufferSync` is a LOCAL copy of `../training/fsio.ts`'s
 * `atomicWriteFileSync` contract (same-directory temp file + `renameSync`),
 * over a `Buffer` rather than a `string`, NOT a change to `fsio.ts`'s own
 * exported signature: `fsio.ts` is a dependency (via
 * `collectRepoRelativeDependencies`) of several already-published artifacts'
 * own `producer.sourceSha256` (e.g. `repertoire-report.ts`'s
 * `repertoireNullProducer`), and editing it -- even a purely additive
 * signature widening -- changes those artifacts' recomputed source identity
 * and spuriously fails `tests/unit/repertoire-report.test.ts`'s "the
 * committed artifact was produced by the producer at HEAD" check, which has
 * nothing to do with this WP (verified empirically while developing this
 * diff).
 */
const atomicWriteBufferSync = (path: string, contents: Buffer): void => {
  const tmpPath = resolve(dirname(path), `.${basename(path)}.${randomBytes(6).toString('hex')}.tmp`);
  try {
    writeFileSync(tmpPath, contents);
    renameSync(tmpPath, path);
  } catch (error) {
    try {
      unlinkSync(tmpPath);
    } catch {
      // tmpPath was never created, or was already cleaned up -- nothing more to do.
    }
    throw error;
  }
};

/**
 * Write `bytes` to `outPath` unmodified, then re-read and re-hash the
 * written file to confirm the write itself didn't alter anything -- a
 * byte-for-byte copy of an externally-produced artifact (this file's whole
 * purpose is preserving its exact bytes against worktree pruning, so a
 * `string` decode/re-encode round trip, or any other silent reformatting,
 * would defeat it -- a dual-review finding).
 */
export const copyVerbatim = (sourcePath: string, outPath: string, bytes: Buffer): void => {
  mkdirSync(dirname(outPath), { recursive: true });
  atomicWriteBufferSync(outPath, bytes);
  const expected = sha256Hex(bytes);
  const actual = sha256Hex(readFileSync(outPath));
  if (actual !== expected) {
    throw new Error(`copy-verbatim: ${outPath} sha256 ${actual} does not match ${sourcePath}'s ${expected} after copying`);
  }
};
