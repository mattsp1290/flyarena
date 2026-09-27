import { parseExportArmsArgs, runExportArms } from './export-arms';

/**
 * `export-arms.ts`'s CLI entry point, split into its own file so
 * `export-arms.ts` itself has no top-level `if (process.argv[1] === ...)
 * main();` side effect (`.agents/plans/graph-lab/02-job-engines.md`'s
 * atlas engine needs to esbuild-`--bundle` `export-arms.ts`'s exports
 * transitively via `scripts/atlas/publish.ts`/`verify-search-graph.ts`,
 * and a bundled CLI guard misfires -- see `export-arms.ts`'s own doc
 * comment on this split, and `entry-atlas-reeval.ts`'s doc comment for the
 * original reproduced failure). This file is never imported by anything
 * else; it is only ever run directly (`npm run training:export-arms`, i.e.
 * `tsx scripts/training/export-arms-cli.ts`), so it keeps its own
 * unconditional `main()` call rather than a guard -- there is no bundling
 * hazard here (this file is never bundled) and no test imports it either.
 */
const main = (): void => {
  const args = parseExportArmsArgs(process.argv.slice(2));
  const { outDir, written, d } = runExportArms(args);
  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
  console.log(`export-arms: wrote ${written.length} bundle(s) to ${outDir} (D=${d}): ${written.join(', ')}`);
};

// try/catch wraps the top-level `main()` call (not `main`'s own body) --
// matches `export-traces-cli.ts`'s identical shape (a thermo-review
// suggestion: the two CLI files this WP split out were inconsistent here,
// both a faithful move of the pre-split code but not aligned with each
// other).
try {
  main();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
  console.error(`export-arms failed: ${message}`);
  process.exit(1);
}
