// Tiny CLI shim for `tests_python/test_ts_import_graph_cross_check.py`:
// reads a JSON object `{ entryFile: string, repoRoot: string }` from stdin,
// runs it through the REAL `scripts/lib/import-graph.ts`
// `collectRepoRelativeDependencies`/`computeSourceIdentitySha256` (not a
// reimplementation), and writes `{ dependencies, sha256 }` as JSON to
// stdout. Run via `node_modules/.bin/tsx` -- no build step, no test
// framework, just this file's own process boundary. Same pattern as
// `tests_python/fixtures/null_stats_cross_check.ts` (this study's existing
// "cross-check the real TS implementation from Python" convention).

import { collectRepoRelativeDependencies, computeSourceIdentitySha256 } from '../../scripts/lib/import-graph';

interface Input {
  readonly entryFile: string;
  readonly repoRoot: string;
}

const readStdin = async (): Promise<string> => {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(Buffer.from(chunk));
  return Buffer.concat(chunks).toString('utf8');
};

const main = async (): Promise<void> => {
  const raw = await readStdin();
  const input = JSON.parse(raw) as Input;

  const dependencies = collectRepoRelativeDependencies(input.entryFile, input.repoRoot);
  const sha256 = computeSourceIdentitySha256(input.repoRoot, dependencies);

  process.stdout.write(JSON.stringify({ dependencies, sha256 }));
};

void main();
