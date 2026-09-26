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

/** @param {Uint8Array | string} bytes */
const sha256Hex = (bytes) => createHash('sha256').update(bytes).digest('hex');

/**
 * Exported (not just used by this file's own CLI `main`) so
 * `tests/unit/graph-lab-entries.test.ts` can build into its own temp
 * directory in a `beforeAll` rather than depending on
 * `backend/graph_lab/js/` already existing -- that directory is
 * gitignored and produced only by `npm run graph-lab:bundle`, which CI's
 * `npm run test:unit` never runs on its own; a test that silently skipped
 * itself (or worse, silently passed against a stale local bundle left
 * over from a developer's own machine) when it was missing would be
 * worse than a build step embedded in the test itself.
 * @param {string} targetOutdir
 */
export const runBundle = async (targetOutdir) => {
  mkdirSync(targetOutdir, { recursive: true });

  await build({
    entryPoints: ENTRY_NAMES.map((name) => resolve(here, `${name}.ts`)),
    outdir: targetOutdir,
    // Pinned to the repo root, not left to default to `process.cwd()`:
    // esbuild embeds `absWorkingDir`-relative path *comments* for each
    // bundled module's source origin even with `sourcemap: false`, so two
    // invocations of this exact same script from different working
    // directories (`npm run graph-lab:bundle` from the repo root vs.
    // `node bundle.mjs` from inside `scripts/graph-lab/`) produced
    // byte-different output and therefore a different `bundleSha256` for
    // identical source -- verified by diffing both outputs before this
    // was added. `bundleSha256` is the one value `/health` and every job
    // result report as covering "all graph-lab code"; it must depend only
    // on that code, never on how the build happened to be invoked.
    absWorkingDir: repoRoot,
    outExtension: { '.js': '.mjs' },
    bundle: true,
    platform: 'node',
    format: 'esm',
    target: 'node22',
    sourcemap: false,
    logLevel: 'info'
  });

  /** @type {Record<string, string>} */
  const files = {};
  for (const name of ENTRY_NAMES) {
    const outPath = resolve(targetOutdir, `${name}.mjs`);
    files[`${name}.mjs`] = sha256Hex(readFileSync(outPath));
  }
  // Sorted keys: deterministic across Node/esbuild's own directory-listing
  // order, and across runs, so two builds of the same source produce a
  // byte-identical `bundle.json` and therefore the same `bundleSha256`
  // (WP1's own acceptance: "two runs give identical sha256 values").
  const sortedFiles = Object.fromEntries(Object.keys(files).sort().map((key) => [key, files[key]]));
  const bundleSha256 = sha256Hex(JSON.stringify(sortedFiles));
  const bundleJson = { files: sortedFiles, bundleSha256 };

  writeFileSync(resolve(targetOutdir, 'bundle.json'), `${JSON.stringify(bundleJson, null, 2)}\n`);
  return { outdir: targetOutdir, bundleSha256, files: sortedFiles };
};

const main = async () => {
  const result = await runBundle(outdir);
  // eslint-disable-next-line no-console -- CLI tool: user-facing summary.
  console.log(
    `graph-lab bundle.mjs: wrote ${ENTRY_NAMES.length} bundles to ${result.outdir} (bundleSha256=${result.bundleSha256})`
  );
};

// Guarded (unlike a bundled entry, this file is only ever imported by
// Vite/vitest's own per-module transform, which keeps each module's
// `import.meta.url` accurate -- there is no esbuild `--bundle` collapse
// hazard here, so this guard is safe): running this file directly (the
// npm script, or the Dockerfile's build step) hits `main()`; importing
// `runBundle` from the test file does not also trigger the CLI's own
// default-outdir build as an import side effect.
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error('graph-lab bundle.mjs failed:', error);
    process.exit(1);
  });
}
