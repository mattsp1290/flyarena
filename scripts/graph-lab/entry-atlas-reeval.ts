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
 * **Now included in `bundle.mjs`'s entry list (WP2).** WP1 originally
 * excluded this file: `verifyAndEvaluateSearchGraph` imports
 * `scripts/atlas/publish.ts`, which unconditionally imports
 * `scripts/training/export-arms.ts` (for `computeArmBundleSha256`/
 * `deserializeArmBundle`), which unconditionally imports
 * `scripts/training/export-traces.ts`. Both `export-arms.ts` and
 * `export-traces.ts` used to end with a top-level
 * `if (process.argv[1] === fileURLToPath(import.meta.url)) main();` guard
 * that invoked *their own* CLI parser. Verified empirically at the time:
 * once esbuild bundled this file with `--bundle` into one output, every
 * inlined module's `import.meta.url` collapsed to that single output
 * file's own URL (there is only one real ES module left at runtime), so
 * both guards' comparisons against `process.argv[1]` (this entry's own
 * invocation) evaluated `true`, and the bundle crashed on startup trying
 * to parse `entry-atlas-reeval`'s own args as `export-arms`'s or
 * `export-traces`'s CLI flags (reproduced: `export-traces failed: Unknown
 * argument: <args-file-path>`).
 *
 * WP2 resolved this at the source rather than editing around it: both
 * guards moved into their own thin `export-arms-cli.ts`/
 * `export-traces-cli.ts` files (`package.json`'s `training:export-arms`/
 * `training:traces` scripts repointed at them), so `export-arms.ts`/
 * `export-traces.ts` themselves have no top-level side effect and are
 * safe to bundle. `service.py`'s `default_runner` dispatches `kind:
 * "atlas"` to `graph_lab.engine_atlas`, which runs this entry (bundled as
 * `entry-atlas-reeval.mjs`) as its final step.
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
