import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

import {
  DEFAULT_GRAPH_ID,
  buildGoldenFiles,
  buildTaskGoldenFiles,
  graphIdFromPath,
  loadGraphArtifact,
  parseArgs
} from './export-traces';
import { resolveArenaTask } from '../../src/lib/arena/tasks';
import { createTraceGraph } from '../../tests/fixtures/trace-graph';
import { validateGraph } from '../../src/lib/connectome/format';

/**
 * `export-traces.ts`'s CLI entry point, split into its own file for the
 * same reason `export-arms.ts`'s `main`/guard moved to
 * `export-arms-cli.ts` (see that file's doc comment): a bundled top-level
 * `if (process.argv[1] === ...) main();` in `export-traces.ts` itself
 * misfired once `.agents/plans/graph-lab/02-job-engines.md`'s atlas
 * engine esbuild-`--bundle`d `export-arms.ts` (which unconditionally
 * imports this file). This file is never imported by anything else; it is
 * only ever run directly (`npm run training:traces`, i.e.
 * `tsx scripts/training/export-traces-cli.ts`), so it keeps its own
 * unconditional `main()` call rather than a guard.
 */
const writeJson = (path: string, value: unknown): number => {
  mkdirSync(dirname(path), { recursive: true });
  const contents = JSON.stringify(value);
  writeFileSync(path, contents);
  return Buffer.byteLength(contents);
};

const main = (): void => {
  const args = parseArgs(process.argv.slice(2));

  const graph = args.graphPath ? loadGraphArtifact(args.graphPath) : createTraceGraph();
  validateGraph(graph);
  const graphId = args.graphPath ? graphIdFromPath(args.graphPath) : DEFAULT_GRAPH_ID;

  const outDir = resolve(process.cwd(), args.outDir);
  const isNonDefaultTask = args.arenaTask !== undefined && args.arenaTask !== 'default';
  const files = isNonDefaultTask
    ? buildTaskGoldenFiles(graph, graphId, args.substeps, resolveArenaTask(args.arenaTask).config)
    : buildGoldenFiles(graph, graphId, args.substeps, { includeWorld: args.includeWorld });

  let totalBytes = 0;
  for (const { fileName, value } of files) {
    totalBytes += writeJson(resolve(outDir, fileName), value);
  }

  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
  console.log(
    `Wrote ${files.length} file(s) to ${outDir} (${totalBytes} bytes total): ` +
      files.map((f) => f.fileName).join(', ')
  );
};

try {
  main();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
  console.error(`export-traces failed: ${message}`);
  process.exit(1);
}
