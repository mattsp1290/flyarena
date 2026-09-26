#!/usr/bin/env node
// esbuild bundles each graph-lab entry and each worker as its own entry
// point (`.agents/plans/graph-lab/01-service-and-container.md`'s
// `bundle.mjs` row): every one of them can be `fork()`ed or `node`-invoked
// standalone, and each entry computes its own sibling worker path via
// `new URL('./worker-*.mjs', import.meta.url)` rather than assuming a
// shared runtime. Output goes to `backend/graph_lab/js/` (gitignored --
// this script is what regenerates it, and the Dockerfile copies its
// output into the image). `bundle.json` records every output file's own
// sha256 plus `bundleSha256`, the sha256 of `bundle.json`'s own canonical
// (sorted-key) file-list -- the single value `/health` and every job
// result report, covering all graph-lab JS code as one unit.
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { build } from 'esbuild';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');
const outdir = resolve(repoRoot, 'backend/graph_lab/js');

// `entry-atlas-reeval` is deliberately excluded: it transitively imports
// `scripts/training/export-arms.ts` and `scripts/training/export-traces.ts`,
// both of which end with a top-level
// `if (process.argv[1] === fileURLToPath(import.meta.url)) main();` guard.
// Bundling collapses every inlined module's `import.meta.url` to this
// output file's own URL, so those guards misfire and invoke the wrong CLI
// parser against this entry's own argv (confirmed empirically -- see
// `entry-atlas-reeval.ts`'s own doc comment for the full analysis and the
// reproduced error). Reported as a plan deviation rather than bundling it
// anyway or editing the guarded files (out of this WP's change surface).
const ENTRY_NAMES = ['entry-lesion', 'entry-swapset', 'worker-lesion', 'worker-score'];

const sha256Hex = (bytes) => createHash('sha256').update(bytes).digest('hex');

const main = async () => {
  mkdirSync(outdir, { recursive: true });

  await build({
    entryPoints: ENTRY_NAMES.map((name) => resolve(here, `${name}.ts`)),
    outdir,
    outExtension: { '.js': '.mjs' },
    bundle: true,
    platform: 'node',
    format: 'esm',
    target: 'node22',
    sourcemap: false,
    logLevel: 'info'
  });

  const files = {};
  for (const name of ENTRY_NAMES) {
    const outPath = resolve(outdir, `${name}.mjs`);
    files[`${name}.mjs`] = sha256Hex(readFileSync(outPath));
  }
  // Sorted keys: deterministic across Node/esbuild's own directory-listing
  // order, and across runs, so two builds of the same source produce a
  // byte-identical `bundle.json` and therefore the same `bundleSha256`
  // (WP1's own acceptance: "two runs give identical sha256 values").
  const sortedFiles = Object.fromEntries(Object.keys(files).sort().map((key) => [key, files[key]]));
  const bundleSha256 = sha256Hex(JSON.stringify(sortedFiles));
  const bundleJson = { files: sortedFiles, bundleSha256 };

  writeFileSync(resolve(outdir, 'bundle.json'), `${JSON.stringify(bundleJson, null, 2)}\n`);
  // eslint-disable-next-line no-console -- CLI tool: user-facing summary.
  console.log(`graph-lab bundle.mjs: wrote ${ENTRY_NAMES.length} bundles to ${outdir} (bundleSha256=${bundleSha256})`);
};

main().catch((error) => {
  console.error('graph-lab bundle.mjs failed:', error);
  process.exit(1);
});
