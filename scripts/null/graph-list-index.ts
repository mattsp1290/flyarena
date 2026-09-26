import { readFileSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve, sep } from 'node:path';

import { sha256Hex } from '../training/fsio';

/**
 * `scripts/analysis/interventions.py`'s `index.json` parsing/verification —
 * extracted out of `scripts/null/null-evaluate.ts` (a thermo-maintainability
 * review finding: the file crossed the 1000-line threshold after this
 * `--graph-list` mode was added; this section — roughly 150 lines mirroring
 * `rewire-index.ts`'s own section in shape and purpose — was the largest
 * single addition). Re-exported from `null-evaluate.ts` unchanged, so every
 * existing importer keeps importing from `./null-evaluate` with no changes
 * required.
 *
 * `.agents/plans/pathway-interventions/03-evaluation.md`'s WP2 `--graph-list`
 * mode: one entry per predeclared intervention/control graph
 * (`scripts/analysis/interventions.py`'s `index.json` — P, Q, C000..C099,
 * M1000..M1099, MQ2000..MQ2099). The plan states the entry contract as
 * `{ id, path, gzipSha256 }`; `binarySha256` is additionally required here
 * (present on every real `interventions.py` entry) so this mode gets the
 * same two-layer verification `--rewired-index` mode already has: the gzip
 * bytes checked up front by `verifyGraphListFiles` (mirroring
 * `verifyRewiredFiles`), and the *decompressed* bytes independently
 * re-checked inside the worker by `loadVerifiedGraphBinary` (mirroring every
 * `RewireIndexSeedEntry` task's `expectedSha256`) — never trusting a single
 * check across the parent/worker process boundary.
 */
export interface GraphListEntry {
  readonly id: string;
  /** Path to the gzip-compressed graph binary, relative to the graph-list index.json's own directory. */
  readonly path: string;
  readonly gzipSha256: string;
  /** Expected sha256 of the *decompressed* binary — see this interface's doc comment. */
  readonly binarySha256: string;
}

export interface GraphListIndex {
  readonly sourceArtifact: string;
  readonly sourceSha256: string;
  readonly entries: readonly GraphListEntry[];
}

/**
 * `buildGraphListTasks` (`null-evaluate.ts`) synthesizes tasks with
 * `graphId: 'biological'`/`'disconnected'` when `--biological` is set (see
 * `biologicalAndDisconnectedTasks`). A graph-list entry using either id
 * would collide with those tasks: `runShardedEvaluation` keeps whichever
 * result arrives first for a given `graphId` and silently discards the
 * other, so which graph's scores end up in the output (and in the
 * reproduction check `intervention-report.ts` depends on) would depend on
 * shard completion timing — the exact determinism failure `readGraphListIndex`'s
 * duplicate-id check exists to prevent, just via a different id source (a
 * dual-review finding).
 */
const RESERVED_GRAPH_IDS: ReadonlySet<string> = new Set(['biological', 'disconnected']);

/**
 * Parse and lightly validate `scripts/analysis/interventions.py`'s
 * `index.json`. Deliberately tolerant of extra fields beyond the ones this
 * script reads (`kind`, `swaps`, `targetReached`, `transfer`, `producer`,
 * `kP`/`kQ`/`maxSwaps`/`controlCount` are all recorded by `interventions.py`
 * but not consumed here) — the same tolerance `readRewireIndex` already
 * applies to `rewire_batch.py`'s own index.json.
 */
export const readGraphListIndex = (path: string): GraphListIndex => {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as Partial<GraphListIndex>;
  if (typeof parsed.sourceArtifact !== 'string' || typeof parsed.sourceSha256 !== 'string') {
    throw new Error(`null-evaluate: ${path} is missing sourceArtifact/sourceSha256`);
  }
  if (!Array.isArray(parsed.entries) || parsed.entries.length === 0) {
    throw new Error(`null-evaluate: ${path} has no entries`);
  }
  const seenIds = new Set<string>();
  for (const entry of parsed.entries) {
    if (
      typeof entry.id !== 'string' ||
      entry.id.length === 0 ||
      typeof entry.path !== 'string' ||
      typeof entry.gzipSha256 !== 'string' ||
      typeof entry.binarySha256 !== 'string'
    ) {
      throw new Error(`null-evaluate: ${path} has a malformed graph-list entry: ${JSON.stringify(entry)}`);
    }
    if (RESERVED_GRAPH_IDS.has(entry.id)) {
      throw new Error(
        `null-evaluate: ${path} uses reserved graph id "${entry.id}" (reserved for --biological's own tasks)`
      );
    }
    // A path escaping index.json's own directory (an absolute path, or a
    // "../" traversal) would break this mode's documented contract ("every
    // entry's path is relative to the graph-list index.json's own
    // directory" — see `GraphListEntry.path`'s doc comment) and let a
    // mis-generated index silently read a file outside the run's own graph
    // set. Content integrity is still independently enforced by both sha
    // layers either way, so this is a fail-fast/contract check, not the
    // primary integrity guard. Resolution-based (not a raw string check on
    // `entry.path` for a leading `/` or a `..` path segment): a string check
    // both under- and over-rejects -- it would miss a Windows-style
    // drive-absolute path, and it would reject an in-bounds path like
    // `"graphs/../graphs/P.bin.gz"` that a resolution-based check correctly
    // allows (a dual-review finding).
    const indexDir = dirname(path);
    const relativeToIndexDir = relative(indexDir, resolve(indexDir, entry.path));
    if (
      isAbsolute(entry.path) ||
      relativeToIndexDir === '' ||
      relativeToIndexDir === '..' ||
      relativeToIndexDir.startsWith(`..${sep}`) ||
      isAbsolute(relativeToIndexDir)
    ) {
      throw new Error(
        `null-evaluate: ${path} entry "${entry.id}" has a path outside index.json's own directory: "${entry.path}"`
      );
    }
    // Two tasks sharing the same graphId would let whichever result arrives
    // last silently win (the same reasoning as readRewireIndex's duplicate-
    // seed rejection above) — reject it here rather than at result-assembly
    // time.
    if (seenIds.has(entry.id)) {
      throw new Error(`null-evaluate: ${path} lists graph id "${entry.id}" more than once`);
    }
    seenIds.add(entry.id);
  }
  return parsed as GraphListIndex;
};

/**
 * `entries` sorted by `id` ascending (plain string comparison) — the single
 * source of this mode's "results keyed by id, sorted" output guarantee
 * (`03-evaluation.md`'s own wording), matching `sortedRewireSeeds`'s role
 * for `--rewired-index` mode. Never a sort over collected results or
 * completion order.
 */
export const sortedGraphListEntries = (index: Readonly<GraphListIndex>): readonly GraphListEntry[] =>
  [...index.entries].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));

/**
 * Verify every graph-list file's raw gzip bytes against `index.json` before
 * any shard is forked — mirrors `verifyRewiredFiles`'s reasoning exactly
 * (fail in seconds on a corrupted/stale/partial copy, not hours into a run).
 * `indexDir` is the graph-list index.json's own directory: every entry's
 * `path` is relative to it (e.g. `"graphs/P.bin.gz"`), not to `--graphs-dir`
 * (there is no separate `--graphs-dir` in this mode).
 */
export const verifyGraphListFiles = (index: Readonly<GraphListIndex>, indexDir: string): void => {
  const mismatches: string[] = [];
  for (const entry of index.entries) {
    const path = resolve(indexDir, entry.path);
    let bytes: Buffer;
    try {
      bytes = readFileSync(path);
    } catch (error) {
      mismatches.push(`id ${entry.id}: cannot read ${path} (${error instanceof Error ? error.message : String(error)})`);
      continue;
    }
    const actual = sha256Hex(bytes);
    if (actual !== entry.gzipSha256) {
      mismatches.push(`id ${entry.id}: ${path} gzip sha256 ${actual} does not match index.json (${entry.gzipSha256})`);
    }
  }
  if (mismatches.length > 0) {
    throw new Error(`null-evaluate: ${mismatches.length} graph-list file(s) failed verification:\n${mismatches.join('\n')}`);
  }
};
