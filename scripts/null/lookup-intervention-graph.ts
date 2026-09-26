import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { readGraphListIndex, verifyGraphListFiles } from './graph-list-index';

/**
 * `.agents/plans/pathway-interventions/03-evaluation.md`'s WP3
 * (`train-sample.sh --graph-list/--ids`): a thin CLI wrapper that resolves
 * one id to its verified graph path, for `train-sample.sh`'s bash callers
 * (which cannot import a TS module's functions directly and need a
 * subprocess entrypoint the way `lookup-rewired-artifact.ts` already
 * provides for the seed-keyed mode).
 *
 * Reuses `graph-list-index.ts`'s `readGraphListIndex`/`verifyGraphListFiles`
 * -- the exact same validated, path-traversal-safe, two-layer-gzip/binary-
 * sha-aware reader `null-evaluate.ts`'s own `--graph-list` mode (WP2) uses
 * for `scripts/analysis/interventions.py`'s id-keyed `index.json` (WP1's
 * output: `P`, `Q`, `C000..C099`, `M1000..M1099`, `MQ2000..MQ2099`, and `R`
 * only when non-empty) -- rather than a second, independently-drifting
 * parser (a thermo-maintainability review finding: an earlier version of
 * this file had its own parallel `index.json` reader/verifier with a
 * strictly weaker contract -- no path-traversal containment, and only
 * `gzipSha256`, not `graph-list-index.ts`'s two-layer gzip/binary sha
 * contract). This module now only ever imports `graph-list-index.ts`'s
 * already-published exports, never redeclares its own copy of that parsing
 * logic, so `train-sample.sh`'s `--graph-list`/`--ids` mode and
 * `null-evaluate.ts`'s `--graph-list` mode can never again silently diverge
 * on what counts as a valid `index.json`. `readGraphListIndex`/
 * `verifyGraphListFiles`'s own test coverage (`tests/unit/null-evaluate.test.ts`)
 * is this module's coverage too; only the "resolve one id, CLI wrapper"
 * behavior below is specific to this file.
 */
export const resolveVerifiedInterventionGraphPath = (indexPath: string, id: string): string => {
  const index = readGraphListIndex(indexPath);
  const entry = index.entries.find((candidate) => candidate.id === id);
  if (!entry) {
    throw new Error(`lookup-intervention-graph: id "${id}" not found in ${indexPath}`);
  }
  // Verifies every entry's gzip bytes (not just `id`'s) -- the same
  // whole-index check `null-evaluate.ts`'s own `--graph-list` mode performs
  // once up front, reused here rather than adding a single-entry variant
  // `graph-list-index.ts` doesn't otherwise need.
  verifyGraphListFiles(index, dirname(indexPath));
  return resolve(dirname(indexPath), entry.path);
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
