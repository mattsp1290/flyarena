// @vitest-environment node
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { resolveVerifiedInterventionGraphPath } from '../../scripts/null/lookup-intervention-graph';

/**
 * Coverage for `scripts/null/lookup-intervention-graph.ts` --
 * `train-sample.sh`'s `--graph-list`/`--ids` mode's thin CLI wrapper
 * (`.agents/plans/pathway-interventions/03-evaluation.md`'s WP3). This
 * module now composes `graph-list-index.ts`'s `readGraphListIndex`/
 * `verifyGraphListFiles` (a thermo-maintainability review finding: an
 * earlier version had its own parallel index.json parser/verifier) --
 * `readGraphListIndex`'s own malformed-entry/duplicate-id/path-traversal
 * rejection and `verifyGraphListFiles`'s gzip-sha mismatch detection are
 * already covered by `tests/unit/null-evaluate.test.ts`, so this suite only
 * covers what is specific to THIS module: resolving one id to its verified
 * path, and the CLI entrypoint's stdout/stderr/exit-code contract.
 */

const sha256Hex = (data: Buffer | string) => createHash('sha256').update(data).digest('hex');

const runCli = (args: readonly string[]) =>
  spawnSync(process.execPath, ['--import', 'tsx', 'scripts/null/lookup-intervention-graph.ts', ...args], {
    encoding: 'utf8',
    timeout: 30_000
  });

/** A valid `graph-list-index.ts` `GraphListIndex`, written to `root/index.json`, with `P.bin.gz`/`C000.bin.gz` graph files alongside it. */
const buildValidGraphList = (root: string): { indexPath: string; pBytes: Buffer; c000Bytes: Buffer } => {
  const indexPath = join(root, 'index.json');
  const pBytes = Buffer.from('P graph bytes');
  const c000Bytes = Buffer.from('C000 graph bytes');
  writeFileSync(join(root, 'P.bin.gz'), pBytes);
  writeFileSync(join(root, 'C000.bin.gz'), c000Bytes);
  writeFileSync(
    indexPath,
    JSON.stringify({
      sourceArtifact: 'malecns-arena-v1.bin.gz',
      sourceSha256: 'a'.repeat(64),
      entries: [
        { id: 'P', kind: 'P', path: 'P.bin.gz', gzipSha256: sha256Hex(pBytes), binarySha256: 'b'.repeat(64), swaps: 6 },
        { id: 'C000', kind: 'C', path: 'C000.bin.gz', gzipSha256: sha256Hex(c000Bytes), binarySha256: 'c'.repeat(64) }
      ],
      version: 1
    })
  );
  return { indexPath, pBytes, c000Bytes };
};

describe('resolveVerifiedInterventionGraphPath', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'lookup-intervention-graph-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('resolves an id to its verified, sha256-matching graph path', () => {
    const { indexPath } = buildValidGraphList(root);
    expect(resolveVerifiedInterventionGraphPath(indexPath, 'C000')).toBe(join(root, 'C000.bin.gz'));
    expect(resolveVerifiedInterventionGraphPath(indexPath, 'P')).toBe(join(root, 'P.bin.gz'));
  });

  it('throws when the id is not present', () => {
    const { indexPath } = buildValidGraphList(root);
    expect(() => resolveVerifiedInterventionGraphPath(indexPath, 'M1000')).toThrow(/id "M1000" not found/);
  });

  it('propagates verifyGraphListFiles\'s gzip sha256 mismatch (tampered/stale graph)', () => {
    const { indexPath } = buildValidGraphList(root);
    writeFileSync(join(root, 'P.bin.gz'), Buffer.from('tampered bytes'));
    expect(() => resolveVerifiedInterventionGraphPath(indexPath, 'P')).toThrow(/gzip sha256 .* does not match/);
  });

  it('propagates readGraphListIndex\'s own validation on a malformed index.json', () => {
    const indexPath = join(root, 'index.json');
    writeFileSync(
      indexPath,
      JSON.stringify({ sourceArtifact: 'malecns-arena-v1.bin.gz', sourceSha256: 'a'.repeat(64), entries: [] })
    );
    expect(() => resolveVerifiedInterventionGraphPath(indexPath, 'P')).toThrow(/has no entries/);
  });
});

describe('lookup-intervention-graph.ts CLI', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'lookup-intervention-graph-cli-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('prints the resolved graph path to stdout and exits 0 on success', () => {
    const { indexPath } = buildValidGraphList(root);
    const result = runCli([indexPath, 'C000']);
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout.trim()).toBe(join(root, 'C000.bin.gz'));
  });

  it('exits 1 with a stderr message when the id is missing', () => {
    const { indexPath } = buildValidGraphList(root);
    const result = runCli([indexPath, 'does-not-exist']);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/id "does-not-exist" not found/);
    expect(result.stdout).toBe('');
  });

  it('exits 1 with a usage message when called with too few arguments', () => {
    const { indexPath } = buildValidGraphList(root);
    const result = runCli([indexPath]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/Usage:/);
  });
});
