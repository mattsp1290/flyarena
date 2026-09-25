import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { sha256Hex } from '../training/fsio';

/**
 * `.agents/plans/pathway-interventions/03-evaluation.md`'s WP3
 * (`train-sample.sh --graph-list/--ids`, `null-trained-evaluate.ts
 * --graph-list/--trained-dir`): a small, self-contained reader/verifier for
 * `scripts/analysis/interventions.py`'s `index.json` (WP1's output --
 * `training/runs/interventions/run1/index.json` in this study), the
 * predeclared intervention/control graph list (`P`, `Q`, `C000..C099`,
 * `M1000..M1099`, `MQ2000..MQ2099`, and `R` only when non-empty).
 *
 * Deliberately its OWN module, not an addition to `null-evaluate.ts`'s
 * `RewireIndex`/`readRewireIndex` (a *different* index format -- seed-keyed
 * numeric entries from `rewire_batch.py`, not id-keyed entries from
 * `interventions.py`) -- and not an edit to `null-evaluate.ts`/
 * `null-worker.ts` at all: a concurrent bean is adding `null-evaluate.ts`'s
 * own `--graph-list` mode (WP2) to those exact two files, so this module
 * only ever imports their already-published exports, never edits them,
 * avoiding a merge collision over the same lines.
 *
 * Tolerant of extra fields (`kind`, `swaps`, `targetReached`, `transfer`,
 * `binarySha256`, `kP`, `kQ`, `maxSwaps`, `producer`, `sourceArtifact`,
 * `sourceSha256`, `version`, `controlCount`) beyond the three this module
 * reads (`id`/`path`/`gzipSha256`) -- this module only depends on what it
 * actually uses, matching `null-evaluate.ts`'s `readRewireIndex` doc
 * comment's own stated convention.
 */

export interface InterventionIndexEntry {
  readonly id: string;
  readonly path: string;
  readonly gzipSha256: string;
}

export interface InterventionIndex {
  readonly entries: readonly InterventionIndexEntry[];
}

/** Parse and lightly validate `interventions.py`'s `index.json`. */
export const readInterventionIndex = (path: string): InterventionIndex => {
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(path, 'utf8'));
  } catch (error) {
    throw new Error(
      `lookup-intervention-graph: cannot read/parse ${path}: ${error instanceof Error ? error.message : String(error)}`
    );
  }
  const record = parsed as { entries?: unknown };
  if (!Array.isArray(record.entries) || record.entries.length === 0) {
    throw new Error(`lookup-intervention-graph: ${path} has no "entries"`);
  }
  const seenIds = new Set<string>();
  for (const entry of record.entries as readonly unknown[]) {
    const candidate = entry as Partial<InterventionIndexEntry>;
    if (
      typeof candidate.id !== 'string' ||
      candidate.id.length === 0 ||
      typeof candidate.path !== 'string' ||
      typeof candidate.gzipSha256 !== 'string'
    ) {
      throw new Error(`lookup-intervention-graph: ${path} has a malformed entry: ${JSON.stringify(entry)}`);
    }
    // A duplicate id would make `resolveVerifiedInterventionGraphPath`'s
    // `find` silently pick whichever one happens to come first -- reject the
    // index outright instead (`interventions.py` itself can't produce this,
    // since every id it writes is unique by construction, but a hand-merged
    // or hand-edited index.json can).
    if (seenIds.has(candidate.id)) {
      throw new Error(`lookup-intervention-graph: ${path} lists id "${candidate.id}" more than once`);
    }
    seenIds.add(candidate.id);
  }
  return parsed as InterventionIndex;
};

/**
 * Resolve `id`'s graph path (relative to `indexPath`'s own directory, the
 * same convention `interventions.py` itself uses for `entries[].path`) and
 * verify its raw gzip bytes' sha256 against the index's recorded
 * `gzipSha256` -- so a corrupted, stale, or mismatched-provenance graph
 * fails loudly here, before any `export-arms`/`flyarena-train` GPU time is
 * spent, rather than surfacing only as a much-later, much-less-specific
 * training or scoring failure.
 */
export const resolveVerifiedInterventionGraphPath = (indexPath: string, id: string): string => {
  const index = readInterventionIndex(indexPath);
  const entry = index.entries.find((candidate) => candidate.id === id);
  if (!entry) {
    throw new Error(`lookup-intervention-graph: id "${id}" not found in ${indexPath}`);
  }
  const graphPath = resolve(dirname(indexPath), entry.path);
  let bytes: Buffer;
  try {
    bytes = readFileSync(graphPath);
  } catch (error) {
    throw new Error(
      `lookup-intervention-graph: id "${id}": cannot read ${graphPath}: ${error instanceof Error ? error.message : String(error)}`
    );
  }
  const actual = sha256Hex(bytes);
  if (actual !== entry.gzipSha256) {
    throw new Error(
      `lookup-intervention-graph: id "${id}": ${graphPath} gzip sha256 ${actual} does not match index.json (${entry.gzipSha256})`
    );
  }
  return graphPath;
};

/**
 * Usage: `tsx scripts/null/lookup-intervention-graph.ts <index.json> <id>`
 * -- prints the resolved, sha256-verified graph path to stdout (nothing
 * else) on success, or a one-line error to stderr and exits 1 on failure.
 * Mirrors `lookup-rewired-artifact.ts`'s CLI shape so
 * `train-sample.sh`'s `--graph-list`/`--ids` mode can shell out to this the
 * same way its existing seed-based mode shells out to that one.
 */
const main = (): void => {
  const [indexPath, id] = process.argv.slice(2);
  if (!indexPath || !id) {
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error('Usage: lookup-intervention-graph.ts <index.json> <id>');
    process.exit(1);
  }
  try {
    const graphPath = resolveVerifiedInterventionGraphPath(indexPath, id);
    process.stdout.write(`${graphPath}\n`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(message);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
