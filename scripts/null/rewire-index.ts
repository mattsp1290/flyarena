import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { gunzipSync } from 'node:zlib';

import { sha256Hex } from '../training/fsio';

/**
 * `rewire_batch.py`'s `index.json` parsing/verification, plus biological
 * source verification — extracted out of `scripts/null/null-evaluate.ts`
 * (a thermo-maintainability review finding: the file crossed the 1000-line
 * threshold, and this self-contained, already-delimited section, mirroring
 * `graph-list-index.ts`'s own section in shape, was the natural first
 * extraction). Re-exported from `null-evaluate.ts` unchanged, so every
 * existing importer (`regime-check.ts`, `lookup-rewired-artifact.ts`,
 * `null-report.ts`, and this branch's own tests) keeps importing from
 * `./null-evaluate` with no changes required.
 */

export interface RewireIndexSeedEntry {
  readonly seed: number;
  readonly artifact: string;
  readonly binarySha256: string;
  readonly binaryBytes: number;
  readonly gzipSha256: string;
  readonly gzipBytes: number;
  readonly stats: {
    readonly acceptedSwaps: number;
    readonly attempts: number;
  };
}

export interface RewireIndex {
  readonly sourceArtifact: string;
  readonly sourceSha256: string;
  readonly rewireSourceSha256: string;
  readonly seeds: readonly RewireIndexSeedEntry[];
}

/**
 * `index.seeds` sorted by numeric seed ascending — the single source of
 * `null-evaluate.ts`'s "canonical task order" invariant (see that module's
 * own doc comment). `buildTasks` and `assembleRaw` both need this exact
 * order for the same reason (byte-identical output independent of shard
 * count/timing); previously each independently re-sorted, which let the
 * two copies drift out of agreement by convention alone. Never a sort over
 * `graphId` strings, which would put `rewired-10` before `rewired-2`.
 */
export const sortedRewireSeeds = (index: Readonly<RewireIndex>): readonly RewireIndexSeedEntry[] =>
  [...index.seeds].sort((a, b) => a.seed - b.seed);

/**
 * Parse and lightly validate `rewire_batch.py`'s `index.json`. Deliberately
 * tolerant of extra/missing provenance fields beyond the ones this script
 * reads (`binfmtSourceSha256`/`numpyVersion`/`params` are recorded by
 * `rewire_batch.py` but not consumed here) — this script only depends on
 * the fields it actually uses.
 */
export const readRewireIndex = (path: string): RewireIndex => {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as Partial<RewireIndex>;
  if (typeof parsed.sourceArtifact !== 'string' || typeof parsed.sourceSha256 !== 'string') {
    throw new Error(`null-evaluate: ${path} is missing sourceArtifact/sourceSha256`);
  }
  if (typeof parsed.rewireSourceSha256 !== 'string') {
    throw new Error(`null-evaluate: ${path} is missing rewireSourceSha256`);
  }
  if (!Array.isArray(parsed.seeds) || parsed.seeds.length === 0) {
    throw new Error(`null-evaluate: ${path} has no seeds`);
  }
  const seenSeeds = new Set<number>();
  for (const entry of parsed.seeds) {
    if (
      typeof entry.seed !== 'number' ||
      !Number.isInteger(entry.seed) ||
      entry.seed < 0 ||
      typeof entry.artifact !== 'string' ||
      typeof entry.binarySha256 !== 'string' ||
      typeof entry.gzipSha256 !== 'string' ||
      typeof entry.gzipBytes !== 'number' ||
      typeof entry.stats?.acceptedSwaps !== 'number' ||
      typeof entry.stats?.attempts !== 'number'
    ) {
      throw new Error(`null-evaluate: ${path} has a malformed seed entry: ${JSON.stringify(entry)}`);
    }
    // A duplicate seed would create two tasks with the same graphId
    // (`rewired-${seed}`); which result "wins" then depends on completion
    // order, which is exactly what the shard byte-identity guarantee
    // promises can never happen — reject it here instead (a dual-review
    // finding; `rewire_batch.py` itself can't produce this, since it
    // iterates a `range`, but a hand-merged or hand-edited index.json can).
    if (seenSeeds.has(entry.seed)) {
      throw new Error(`null-evaluate: ${path} lists rewiring seed ${entry.seed} more than once`);
    }
    seenSeeds.add(entry.seed);
  }
  return parsed as RewireIndex;
};

/**
 * Verify every rewired file's raw gzip bytes against `index.json` before
 * any shard is forked, so a corrupted or stale batch (a smaller local disk
 * problem, a partially-copied directory) fails in seconds rather than
 * after however much of an 8-hour run has already completed. `null-worker.ts`
 * independently re-verifies the *decompressed* sha256 of whatever file it
 * actually loads, closer to where scoring happens.
 */
export const verifyRewiredFiles = (index: Readonly<RewireIndex>, graphsDir: string): void => {
  const mismatches: string[] = [];
  for (const entry of index.seeds) {
    const path = resolve(graphsDir, entry.artifact);
    let bytes: Buffer;
    try {
      bytes = readFileSync(path);
    } catch (error) {
      mismatches.push(`seed ${entry.seed}: cannot read ${path} (${error instanceof Error ? error.message : String(error)})`);
      continue;
    }
    if (bytes.byteLength !== entry.gzipBytes) {
      mismatches.push(`seed ${entry.seed}: ${path} is ${bytes.byteLength} bytes, index.json expects ${entry.gzipBytes}`);
      continue;
    }
    const actual = sha256Hex(bytes);
    if (actual !== entry.gzipSha256) {
      mismatches.push(`seed ${entry.seed}: ${path} gzip sha256 ${actual} does not match index.json (${entry.gzipSha256})`);
    }
  }
  if (mismatches.length > 0) {
    throw new Error(`null-evaluate: ${mismatches.length} rewired file(s) failed verification:\n${mismatches.join('\n')}`);
  }
};

/**
 * Verify the biological source graph's decompressed sha256 against
 * `index.json`'s own `sourceSha256`. Exported (a thermo-maintainability
 * review finding): `regime-check.ts` (WP2) needs the exact same check and
 * previously carried a hand-duplicated copy because this was private.
 * `source` is the calling CLI's own name (`"null-evaluate"`/
 * `"regime-check"`), matching `null-worker-shared.ts`'s
 * `assertFiniteScores`/`loadVerifiedGraphBinary` `source`-prefixed-message
 * convention, so the thrown message still identifies which CLI raised it.
 */
export const verifyBiologicalSource = (source: string, path: string, expectedSha256: string): void => {
  const gzipBytes = readFileSync(path);
  const binary = gunzipSync(gzipBytes);
  const actual = sha256Hex(binary);
  if (actual !== expectedSha256) {
    throw new Error(`${source}: ${path} decompressed sha256 ${actual} does not match index.json's sourceSha256 (${expectedSha256})`);
  }
};
