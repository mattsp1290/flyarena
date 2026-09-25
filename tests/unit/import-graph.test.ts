// @vitest-environment node
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { collectRepoRelativeDependencies, computeSourceIdentitySha256 } from '../../scripts/lib/import-graph';

/**
 * `scripts/lib/import-graph.ts`'s real-import-graph code-identity walker --
 * the structural fix for a thermo-fix-verification review finding (`01-
 * critical-and-important.md`, finding 1): the old hand-maintained
 * `REGIME_SOURCE_FILENAMES` flat list silently omitted `scripts/training/
 * episode.ts`, `src/lib/connectome/model.ts`/`readout.ts`, and `src/lib/
 * arena/world.ts`/`sensors.ts`/`actions.ts` -- the actual episode/substep/
 * physics code `regime.json`'s numbers depend on. This suite covers two
 * things: (1) against the real repository, `scripts/null/regime-check.ts`'s
 * walked closure genuinely includes those previously-missing files (not a
 * synthetic stand-in -- a synthetic fixture could pass while the real
 * driver's import graph still didn't resolve the way the walker assumes),
 * and (2) against small, hand-built fixture trees, the walker's resolution
 * rules (bare-specifier `.ts` resolution, `index.ts` resolution, import
 * cycles) and `computeSourceIdentitySha256`'s content-sensitivity (the
 * mutation-test requirement) in isolation.
 */

const REPO_ROOT = resolve(__dirname, '../..');

describe('collectRepoRelativeDependencies (real repository)', () => {
  it("regime-check.ts's closure includes the episode/model/physics code the study characterizes", () => {
    const entry = resolve(REPO_ROOT, 'scripts/null/regime-check.ts');
    const dependencies = collectRepoRelativeDependencies(entry, REPO_ROOT);

    // The entry file itself, per the existing convention (the flat lists
    // this replaced always listed the producer script first).
    expect(dependencies).toContain('scripts/null/regime-check.ts');

    // Previously-missing files a thermo-fix-verification review finding
    // named explicitly: the episode driver and the substep/physics/arena
    // code it actually calls into.
    expect(dependencies).toContain('scripts/training/episode.ts');
    expect(dependencies).toContain('src/lib/connectome/model.ts');
    expect(dependencies).toContain('src/lib/connectome/readout.ts');
    expect(dependencies).toContain('src/lib/connectome/constants.ts');
    expect(dependencies).toContain('src/lib/arena/world.ts');
    expect(dependencies).toContain('src/lib/arena/sensors.ts');
    expect(dependencies).toContain('src/lib/arena/actions.ts');

    // The files the old flat list already had.
    expect(dependencies).toContain('scripts/null/regime-task.ts');
    expect(dependencies).toContain('scripts/null/regime-worker.ts');
    expect(dependencies).toContain('scripts/null/null-worker-shared.ts');

    // Sorted, de-duplicated, repo-relative, forward-slash-separated.
    expect(dependencies).toEqual([...new Set(dependencies)].sort());
    for (const dep of dependencies) {
      expect(dep).not.toMatch(/\\/);
      expect(dep.startsWith('.')).toBe(false);
    }
  });

  it('produces a stable, non-empty closure across repeated calls (deterministic walk)', () => {
    const entry = resolve(REPO_ROOT, 'scripts/null/regime-check.ts');
    const first = collectRepoRelativeDependencies(entry, REPO_ROOT);
    const second = collectRepoRelativeDependencies(entry, REPO_ROOT);
    expect(second).toEqual(first);
    expect(first.length).toBeGreaterThan(10);
  });
});

describe('collectRepoRelativeDependencies (synthetic fixture trees)', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'import-graph-test-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('resolves a bare relative specifier to its .ts file', () => {
    writeFileSync(join(root, 'a.ts'), "import { b } from './b';\nexport const a = b;\n");
    writeFileSync(join(root, 'b.ts'), 'export const b = 1;\n');

    const deps = collectRepoRelativeDependencies(join(root, 'a.ts'), root);
    expect(deps).toEqual(['a.ts', 'b.ts']);
  });

  it('resolves a directory specifier to its index.ts', () => {
    mkdirSync(join(root, 'dir'));
    writeFileSync(join(root, 'a.ts'), "import { x } from './dir';\nexport const a = x;\n");
    writeFileSync(join(root, 'dir', 'index.ts'), 'export const x = 1;\n');

    const deps = collectRepoRelativeDependencies(join(root, 'a.ts'), root);
    expect(deps.sort()).toEqual(['a.ts', 'dir/index.ts']);
  });

  it('follows a multi-line import statement (specifier list spans several lines)', () => {
    writeFileSync(
      join(root, 'a.ts'),
      "import {\n  b,\n  c\n} from './b';\nexport const a = b + c;\n"
    );
    writeFileSync(join(root, 'b.ts'), 'export const b = 1;\nexport const c = 2;\n');

    const deps = collectRepoRelativeDependencies(join(root, 'a.ts'), root);
    expect(deps).toEqual(['a.ts', 'b.ts']);
  });

  it('follows a type-only import (included for safety, per this walker\'s policy)', () => {
    writeFileSync(join(root, 'a.ts'), "import type { B } from './b';\nexport const a: B = 1;\n");
    writeFileSync(join(root, 'b.ts'), 'export type B = number;\n');

    const deps = collectRepoRelativeDependencies(join(root, 'a.ts'), root);
    expect(deps).toEqual(['a.ts', 'b.ts']);
  });

  it('follows an export-from re-export', () => {
    writeFileSync(join(root, 'a.ts'), "export { b } from './b';\n");
    writeFileSync(join(root, 'b.ts'), 'export const b = 1;\n');

    const deps = collectRepoRelativeDependencies(join(root, 'a.ts'), root);
    expect(deps).toEqual(['a.ts', 'b.ts']);
  });

  it('never follows a bare (non-relative) specifier -- node_modules/stdlib excluded structurally', () => {
    writeFileSync(join(root, 'a.ts'), "import { z } from 'node:zlib';\nimport { v } from 'vitest';\nexport const a = 1;\n");

    const deps = collectRepoRelativeDependencies(join(root, 'a.ts'), root);
    expect(deps).toEqual(['a.ts']);
  });

  it('terminates on an import cycle instead of looping forever', () => {
    writeFileSync(join(root, 'a.ts'), "import { b } from './b';\nexport const a = 1;\n");
    writeFileSync(join(root, 'b.ts'), "import { a } from './a';\nexport const b = 1;\n");

    const deps = collectRepoRelativeDependencies(join(root, 'a.ts'), root);
    expect(deps.sort()).toEqual(['a.ts', 'b.ts']);
  });

  it('throws a clear error for an unresolvable relative specifier', () => {
    writeFileSync(join(root, 'a.ts'), "import { missing } from './does-not-exist';\nexport const a = missing;\n");

    expect(() => collectRepoRelativeDependencies(join(root, 'a.ts'), root)).toThrow(/cannot resolve/);
  });
});

describe('computeSourceIdentitySha256', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'import-graph-hash-test-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('is independent of the input path array order (sorts internally)', () => {
    writeFileSync(join(root, 'a.ts'), 'export const a = 1;\n');
    writeFileSync(join(root, 'b.ts'), 'export const b = 2;\n');

    const forward = computeSourceIdentitySha256(root, ['a.ts', 'b.ts']);
    const reverse = computeSourceIdentitySha256(root, ['b.ts', 'a.ts']);
    expect(forward).toBe(reverse);
  });

  it('changes when a dependency file\'s bytes change -- the mutation-test requirement', () => {
    writeFileSync(join(root, 'a.ts'), 'export const a = 1;\n');
    writeFileSync(join(root, 'model.ts'), 'export const runSubsteps = () => 1;\n');
    const paths = ['a.ts', 'model.ts'];

    const before = computeSourceIdentitySha256(root, paths);
    writeFileSync(join(root, 'model.ts'), 'export const runSubsteps = () => 2;\n'); // simulates a `model.ts` edit
    const after = computeSourceIdentitySha256(root, paths);

    expect(after).not.toBe(before);
  });

  it('does not change when an out-of-set file changes (only the pinned closure matters)', () => {
    writeFileSync(join(root, 'a.ts'), 'export const a = 1;\n');
    writeFileSync(join(root, 'unrelated.ts'), 'export const u = 1;\n');
    const paths = ['a.ts'];

    const before = computeSourceIdentitySha256(root, paths);
    writeFileSync(join(root, 'unrelated.ts'), 'export const u = 999;\n');
    const after = computeSourceIdentitySha256(root, paths);

    expect(after).toBe(before);
  });

  it("distinguishes two identically-named files in different directories (repo-relative paths, not bare filenames)", () => {
    mkdirSync(join(root, 'dir1'));
    mkdirSync(join(root, 'dir2'));
    writeFileSync(join(root, 'dir1', 'x.ts'), 'export const x = 1;\n');
    writeFileSync(join(root, 'dir2', 'x.ts'), 'export const x = 1;\n'); // same bytes, different path

    const sha1 = computeSourceIdentitySha256(root, ['dir1/x.ts']);
    const sha2 = computeSourceIdentitySha256(root, ['dir2/x.ts']);
    expect(sha1).not.toBe(sha2);
  });
});
