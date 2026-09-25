import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { dirname, relative, resolve, sep } from 'node:path';

/**
 * Real-import-graph code-identity provenance, shared by every TypeScript
 * producer that stamps a `producer.sourceSha256` (currently just
 * `scripts/null/regime-check.ts`'s `regimeProducer()`) so `scripts/
 * analysis/explain_provenance.py`'s Python-side `ts_import_graph.py` can
 * independently recompute the identical dependency set and hash (a
 * cross-language cross-check, the same "implement the identical scheme
 * twice, once per language, and assert they agree" pattern `graph_io.py`'s
 * `source_identity_sha256`/`scripts/data/compile.py`'s
 * `compiler_source_sha256` already use for the hashing half of this
 * problem).
 *
 * Replaces the hand-maintained flat filename lists (e.g. the old
 * `REGIME_SOURCE_FILENAMES = ['regime-check.ts', 'regime-task.ts',
 * 'regime-worker.ts', 'null-worker-shared.ts']`) a thermo-fix-verification
 * review finding showed were silently missing the actual numerically
 * relevant code: `regime-task.ts` dispatches into `scripts/training/
 * episode.ts`, which in turn drives `src/lib/connectome/model.ts`'s
 * `runSubsteps`/`runLesionedSubsteps` -- the substep execution the entire
 * null-explanation study is about -- and none of those files (nor
 * `readout.ts`/`world.ts`/`sensors.ts`/`actions.ts`/`bindings.ts`) were in
 * the hashed set. A hand-maintained list is guaranteed to drift again as
 * those files grow their own imports; walking the real graph removes that
 * "did the author forget a file" risk category rather than just enlarging
 * the list once.
 */

//: TypeScript source extensions this repo's producer files resolve a
//: relative import against, tried in this order (matches this repo's
//: `moduleResolution: "Bundler"` `tsconfig.json` behavior for the plain
//: `import ... from './x'`, no-extension style every file under `scripts/`/
//: `src/lib/` already uses -- confirmed by grep, no `.js`-suffixed relative
//: specifier and no `.svelte`/`.svelte.ts` import appears anywhere in any
//: producer's real dependency closure). `ts_import_graph.py`'s
//: `_resolve_relative_import` tries the exact same candidates in the exact
//: same order, so the two languages' walkers agree file-for-file.
const RESOLUTION_CANDIDATES = (base: string): readonly string[] => [
  `${base}.ts`,
  `${base}.tsx`,
  resolve(base, 'index.ts'),
  resolve(base, 'index.tsx'),
  base // the specifier already carried its own extension (not used today, kept for safety)
];

//: Matches a relative (`.`/`..`-prefixed) import specifier in either
//: `import ... from '...'` / `export ... from '...'` (including multi-line
//: `import {\n  a,\n  b\n} from '...'` -- the `from '...'` clause always
//: ends the statement on its own, so a whole-file regex scan finds it
//: regardless of how many lines the specifier list spans), a dynamic
//: `import('...')`, or a `new URL('...', import.meta.url)` (the pattern
//: `scripts/null/regime-check.ts`'s own `runRegimeCheck` uses to locate
//: `regime-worker.ts` -- its forked-child-process worker script -- since a
//: worker is spawned by file path/URL, never a static `import`; without
//: this third alternative, a worker script forked this way would be
//: invisible to a purely import-based walker even though its bytes
//: genuinely execute as part of producing this driver's output, which is
//: exactly the failure mode this whole fix exists to close). Deliberately
//: does not special-case `import type`/`export type` -- included for
//: simplicity/safety, since a `type`-only import still names a file whose
//: *shape* the producer's compiled output implicitly depends on, and
//: erring toward pinning one extra file is far cheaper than erring toward
//: silently missing one.
const RELATIVE_IMPORT_RE =
  /\bfrom\s+['"](\.[^'"]+)['"]|\bimport\(\s*['"](\.[^'"]+)['"]\s*\)|\bnew\s+URL\(\s*['"](\.[^'"]+)['"]/g;

//: Every producer file this walker scans carries this repo's extremely
//: verbose doc-comment convention, which routinely quotes example import
//: syntax in prose (e.g. this very module's own comments say things like
//: "`import ... from './null-evaluate'`") -- scanning raw, un-stripped file
//: text against `RELATIVE_IMPORT_RE` picks those up as if they were real
//: import statements (confirmed empirically: this walker's own test suite
//: failed on itself until this stripping step was added). Block (`/* ... */`)
//: and line (`// ...`) comments are stripped before the specifier scan, not
//: after, so a comment can never masquerade as a real dependency. This is a
//: deliberately naive text-level strip (it does not understand string
//: literals that happen to contain `//`), acceptable here because the files
//: this walker runs against are a known, small set of producer scripts, not
//: arbitrary third-party source.
const stripComments = (source: string): string =>
  source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');

const extractRelativeImportSpecifiers = (source: string): string[] => {
  const specifiers: string[] = [];
  for (const match of stripComments(source).matchAll(RELATIVE_IMPORT_RE)) {
    const specifier = match[1] ?? match[2] ?? match[3];
    if (specifier) specifiers.push(specifier);
  }
  return specifiers;
};

const isFile = (path: string): boolean => {
  try {
    return existsSync(path) && statSync(path).isFile();
  } catch {
    return false;
  }
};

const resolveRelativeImport = (fromFile: string, specifier: string): string => {
  const base = resolve(dirname(fromFile), specifier);
  for (const candidate of RESOLUTION_CANDIDATES(base)) {
    if (isFile(candidate)) return candidate;
  }
  throw new Error(`import-graph: cannot resolve "${specifier}" (imported from ${fromFile})`);
};

/**
 * Walks every relative (`.`/`..`-prefixed) `import`/`export ... from`/
 * dynamic `import()` specifier reachable from `entryFile`, recursively,
 * restricted to files that resolve on disk (bare specifiers -- `'vitest'`,
 * `'node:fs'`, any `node_modules` package -- are never followed, since they
 * never match the leading-`.` pattern `RELATIVE_IMPORT_RE` requires; this
 * is what "exclude node_modules/stdlib" means here, structurally, rather
 * than as a maintained exclusion list). Returns every file in the closure
 * -- `entryFile` itself included -- as a sorted, forward-slash-separated
 * path relative to `repoRoot`, so `computeSourceIdentitySha256` below (and
 * its Python mirror, `graph_io.source_identity_sha256`) can hash a stable,
 * platform-independent path set.
 */
export const collectRepoRelativeDependencies = (entryFile: string, repoRoot: string): string[] => {
  const visited = new Set<string>();
  const stack = [resolve(entryFile)];
  while (stack.length > 0) {
    const current = stack.pop() as string;
    if (visited.has(current)) continue;
    visited.add(current);
    const source = readFileSync(current, 'utf-8');
    for (const specifier of extractRelativeImportSpecifiers(source)) {
      const resolved = resolveRelativeImport(current, specifier);
      if (!visited.has(resolved)) stack.push(resolved);
    }
  }
  return Array.from(visited)
    .map((absolute) => relative(repoRoot, absolute).split(sep).join('/'))
    .sort();
};

/**
 * sha256 over a set of source files: sorted repo-relative paths, each
 * contributing the path (UTF-8 bytes, forward-slash separated), then a
 * single NUL byte, then the file's raw bytes, into one hasher -- the exact
 * scheme `scripts/data/compile.py`'s `compiler_source_sha256()` and
 * `scripts/analysis/graph_io.py`'s `source_identity_sha256()` already use,
 * reused here unchanged (a third, TS-side implementation of the identical
 * scheme, not a new one) so `explain_provenance.py`'s `current_regime_
 * source_sha256` can recompute this exact hash from raw file bytes, with
 * no TypeScript execution required. Repo-relative (not bare-filename)
 * paths mean two identically-named files in different directories can
 * never collide.
 */
export const computeSourceIdentitySha256 = (repoRoot: string, repoRelativePaths: readonly string[]): string => {
  const hash = createHash('sha256');
  for (const relativePath of [...repoRelativePaths].sort()) {
    hash.update(relativePath, 'utf-8');
    hash.update(Buffer.from([0]));
    hash.update(readFileSync(resolve(repoRoot, relativePath)));
  }
  return hash.digest('hex');
};
