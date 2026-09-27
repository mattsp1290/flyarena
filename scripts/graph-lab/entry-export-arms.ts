import { readFileSync } from 'node:fs';

import { parseExportArmsArgs, runExportArms } from '../training/export-arms';

/**
 * `entry-export-arms.mjs`: the atlas engine's first step
 * (`.agents/plans/graph-lab/02-job-engines.md`'s atlas engine description,
 * "export the arm bundle with the existing `export-arms` logic, bundled
 * as a JS entry, into tmpfs"). `backend/graph_lab/engine_atlas.py` runs
 * this once per atlas job, pointed at the job's sha-verified graph binary
 * (`biological`, a regenerated `rewired:<seed>`, or `disconnected`), then
 * feeds the written arm bundle straight into
 * `python -m flyarena_training.atlas_cli --graph <bundle>`.
 *
 * Deliberately reuses `export-arms.ts`'s own `parseExportArmsArgs`/
 * `runExportArms` unchanged -- never a graph-lab-local reimplementation
 * of arm-bundle export -- so the GPU search sees exactly the same bundle
 * shape (`SerializedArmBundle`, sha-self-certifying) that
 * `training:export-arms`/`atlas:publish` produce everywhere else. This
 * file's only job is the args-file/stdout-JSON-line protocol every other
 * graph-lab entry uses (`entry-lesion.ts`/`entry-swapset.ts`), so
 * `jobs.py`'s child-process supervision (stdout draining, cancellation,
 * timeouts) works identically for this step too.
 *
 * Unlike `entry-atlas-reeval.ts` (whose transitive import of this same
 * `export-arms.ts` module was the reason that file couldn't be bundled
 * until WP2 moved `export-arms.ts`'s/`export-traces.ts`'s own CLI guards
 * into separate `-cli.ts` files -- see `bundle.mjs`'s doc comment), this
 * entry imports `export-arms.ts` directly and is itself one of the
 * bundle's entry points, so it is proof, not just an assertion, that the
 * guard split actually closed the bundling hazard.
 */
interface ExportArmsEntryArgs {
  readonly graphPath: string;
  readonly rewiredPath?: string;
  readonly outDir: string;
}

const printResult = (result: unknown): void => {
  process.stdout.write(`${JSON.stringify({ type: 'result', result })}\n`);
};

const printError = (message: string): void => {
  process.stdout.write(`${JSON.stringify({ type: 'error', message })}\n`);
};

const main = (): void => {
  const argsPath = process.argv[2];
  if (!argsPath) throw new Error('entry-export-arms: missing required args-file argument');
  const args: ExportArmsEntryArgs = JSON.parse(readFileSync(argsPath, 'utf8'));

  const parsed = parseExportArmsArgs([
    '--graph',
    args.graphPath,
    ...(args.rewiredPath ? ['--rewired', args.rewiredPath] : []),
    '--out',
    args.outDir
  ]);
  const { outDir, written, d } = runExportArms(parsed);
  printResult({ outDir, written, d });
};

try {
  main();
} catch (error: unknown) {
  printError(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
