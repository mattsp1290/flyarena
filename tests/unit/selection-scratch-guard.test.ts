// @vitest-environment node
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { guardSelectionScratchTarget } from '../../scripts/null/selection-scratch-guard';

/**
 * Coverage for `scripts/null/selection-scratch-guard.ts` -- extracted out of
 * `null-report.ts`'s `guardSelectionScratchTarget` and
 * `intervention-report-run-mode.ts`'s `guardSelectionScratchOut` (a
 * thermo-maintainability review finding: both files implemented the
 * identical algorithm). `null-report.test.ts` and
 * `intervention-report-arena-task.test.ts` keep their own call-site-level
 * integration coverage (through `runNullReport`/`runInterventionReport`);
 * this file covers the shared guard's own behavior directly, including the
 * corrupt-shipped-manifest fail-safe case neither of those two files
 * exercised before this branch (a thermo-methodology review finding: the
 * `null-report.ts`/`explain_selection_mode.py` copies only handled a
 * *missing* shipped manifest, not a present-but-corrupt one).
 */
describe('guardSelectionScratchTarget', () => {
  let root: string;
  let pub: string;
  let docs: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'selection-scratch-guard-'));
    pub = join(root, 'public', 'data');
    docs = join(root, 'docs');
    mkdirSync(pub, { recursive: true });
    mkdirSync(docs, { recursive: true });
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('allows a write under public/ when sha matches the shipped manifest', () => {
    writeFileSync(join(pub, 'malecns-arena-v1.manifest.json'), JSON.stringify({ binarySha256: 'a'.repeat(64) }));
    expect(() => guardSelectionScratchTarget(join(pub, 'x.json'), '--out', 'a'.repeat(64), pub, docs)).not.toThrow();
  });

  it('refuses a write under public/ when sha does not match the shipped manifest', () => {
    writeFileSync(join(pub, 'malecns-arena-v1.manifest.json'), JSON.stringify({ binarySha256: 'a'.repeat(64) }));
    expect(() => guardSelectionScratchTarget(join(pub, 'x.json'), '--out', 'b'.repeat(64), pub, docs)).toThrow(
      /resolves under/
    );
  });

  it('refuses a write under docs/ when sha does not match the shipped manifest', () => {
    writeFileSync(join(pub, 'malecns-arena-v1.manifest.json'), JSON.stringify({ binarySha256: 'a'.repeat(64) }));
    expect(() => guardSelectionScratchTarget(join(docs, 'x.md'), '--out', 'b'.repeat(64), pub, docs)).toThrow(
      /resolves under/
    );
  });

  it('never blocks a sibling directory that merely shares a string prefix with public/ or docs/', () => {
    writeFileSync(join(pub, 'malecns-arena-v1.manifest.json'), JSON.stringify({ binarySha256: 'a'.repeat(64) }));
    expect(() =>
      guardSelectionScratchTarget(join(root, 'public-old', 'x.json'), '--out', 'b'.repeat(64), pub, docs)
    ).not.toThrow();
    expect(() =>
      guardSelectionScratchTarget(join(root, 'docs2', 'x.md'), '--out', 'b'.repeat(64), pub, docs)
    ).not.toThrow();
  });

  it('fails safe (refuses) when the shipped manifest is missing entirely', () => {
    // No manifest written at all -- `pub` exists but is empty.
    expect(() => guardSelectionScratchTarget(join(pub, 'x.json'), '--out', 'b'.repeat(64), pub, docs)).toThrow(
      /resolves under/
    );
  });

  it('fails safe (refuses, does not throw a raw JSON.parse SyntaxError) when the shipped manifest is corrupt', () => {
    writeFileSync(join(pub, 'malecns-arena-v1.manifest.json'), '{not valid json');
    expect(() => guardSelectionScratchTarget(join(pub, 'x.json'), '--out', 'b'.repeat(64), pub, docs)).toThrow(
      /resolves under/
    );
  });
});
