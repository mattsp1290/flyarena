import { readFileSync } from 'node:fs';

import { parseGraphBinary } from '../../src/lib/connectome/format';
import { loadVerifiedGraphBinary } from '../null/null-worker-shared';
import { verifyAndEvaluateSearchGraph, type ExpectedGraphIdentity } from '../atlas/verify-search-graph';

/**
 * `entry-atlas-reeval.mjs`: re-evaluates a GPU atlas search's candidates in
 * TypeScript (`heldout: 'own'`, `requireDiversity: false`, matching
 * `.agents/plans/graph-lab/02-job-engines.md`'s atlas engine description)
 * via `verify-search-graph.ts`'s existing `verifyAndEvaluateSearchGraph` --
 * unmodified, just called with an explicit `dataDir`-derived graph path
 * instead of any `import.meta.url`/cwd-relative default.
 *
 * **Not in `bundle.mjs`'s entry list yet -- confirmed unsafe to
 * single-file-bundle with today's sources; report this instead of editing
 * around it (this WP's scope note: "if bundling requires a source change,
 * report it").** `verifyAndEvaluateSearchGraph` imports
 * `scripts/atlas/publish.ts`, which unconditionally imports
 * `scripts/training/export-arms.ts` (for `computeArmBundleSha256`/
 * `deserializeArmBundle`), which unconditionally imports
 * `scripts/training/export-traces.ts`. Both `export-arms.ts` and
 * `export-traces.ts` end with a top-level
 * `if (process.argv[1] === fileURLToPath(import.meta.url)) main();` guard
 * that invokes *their own* CLI parser. Verified empirically: once esbuild
 * bundles this file with `--bundle` into one output, every inlined
 * module's `import.meta.url` collapses to that single output file's own
 * URL (there is only one real ES module left at runtime), so both guards'
 * comparisons against `process.argv[1]` (this entry's own invocation)
 * evaluate `true`, and the bundle crashes on startup trying to parse
 * `entry-atlas-reeval`'s own args as `export-arms`'s or `export-traces`'s
 * CLI flags (reproduced: `export-traces failed: Unknown argument:
 * <args-file-path>`). Because the guard is a bare top-level statement (a
 * side effect, not a declaration), esbuild's tree-shaking cannot remove it
 * merely because none of `export-arms.ts`'s *exports* are used -- the
 * import itself forces the whole module body to run. `publish.ts` is
 * therefore not currently safe to import into any single-file esbuild
 * bundle, regardless of which of its exports a caller actually needs.
 *
 * This file's logic is otherwise real and correct (type-checked, and
 * calling exactly the function WP2's plan names) -- it is simply not
 * wired into `bundle.mjs` until WP2 either resolves the guard hazard
 * upstream or changes how this entry composes with `publish.ts`. `jobs.py`
 * does not dispatch `kind: "atlas"` to it either (501, see
 * `service.py`'s `default_runner`).
 */
interface AtlasReevalArgs {
  readonly dataDir: string;
  readonly searchPath: string;
  readonly expected: ExpectedGraphIdentity;
  readonly biologicalGraphPath: string;
  readonly biologicalExpectedSha256: string;
}

const printResult = (result: unknown): void => {
  process.stdout.write(`${JSON.stringify({ type: 'result', result })}\n`);
};

const printError = (message: string): void => {
  process.stdout.write(`${JSON.stringify({ type: 'error', message })}\n`);
};

const main = async (): Promise<void> => {
  const argsPath = process.argv[2];
  if (!argsPath) throw new Error('entry-atlas-reeval: missing required args-file argument');
  const args: AtlasReevalArgs = JSON.parse(readFileSync(argsPath, 'utf8'));

  const biologicalBinary = loadVerifiedGraphBinary(
    'entry-atlas-reeval',
    args.biologicalGraphPath,
    args.biologicalExpectedSha256
  );
  const biologicalGraph = parseGraphBinary(biologicalBinary.slice(0));

  const evaluation = await verifyAndEvaluateSearchGraph(args.searchPath, args.expected, biologicalGraph);

  printResult({
    dataDir: args.dataDir,
    host: { arch: process.arch, node: process.version },
    label: 'Computed on DGX (private, not published)',
    ...evaluation
  });
};

main().catch((error: unknown) => {
  printError(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
