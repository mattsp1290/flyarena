// @vitest-environment node
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  readInterventionIndex,
  resolveVerifiedInterventionGraphPath
} from '../../scripts/null/lookup-intervention-graph';

/**
 * Coverage for `scripts/null/lookup-intervention-graph.ts` --
 * `train-sample.sh`'s `--graph-list`/`--ids` mode's index reader/verifier
 * (`.agents/plans/pathway-interventions/03-evaluation.md`'s WP3). Mirrors
 * `lookup-rewired-artifact.test.ts`'s two-half shape (the exported function
 * in-process, then the CLI entrypoint via `spawnSync`), plus coverage this
 * module's own gzip-sha256 verification needs that the rewired-seed lookup
 * doesn't (that one never reads the artifact's bytes at all).
 */

const sha256Hex = (data: Buffer | string) => createHash('sha256').update(data).digest('hex');

const runCli = (args: readonly string[]) =>
  spawnSync(process.execPath, ['--import', 'tsx', 'scripts/null/lookup-intervention-graph.ts', ...args], {
    encoding: 'utf8',
    timeout: 30_000
  });

describe('readInterventionIndex / resolveVerifiedInterventionGraphPath', () => {
  let root: string;
  let indexPath: string;
  let pBytes: Buffer;
  let cBytes: Buffer;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'lookup-intervention-graph-'));
    indexPath = join(root, 'index.json');
    pBytes = Buffer.from('P graph bytes');
    cBytes = Buffer.from('C000 graph bytes');
    writeFileSync(join(root, 'P.bin.gz'), pBytes);
    writeFileSync(join(root, 'C000.bin.gz'), cBytes);
    writeFileSync(
      indexPath,
      JSON.stringify({
        entries: [
          { id: 'P', kind: 'P', path: 'P.bin.gz', gzipSha256: sha256Hex(pBytes), swaps: 6 },
          { id: 'C000', kind: 'C', path: 'C000.bin.gz', gzipSha256: sha256Hex(cBytes), swaps: 6 }
        ],
        sourceArtifact: 'malecns-arena-v1.bin.gz',
        version: 1
      })
    );
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('parses a well-formed index and preserves extra fields on the entry', () => {
    const index = readInterventionIndex(indexPath);
    expect(index.entries).toHaveLength(2);
    expect(index.entries[0].id).toBe('P');
  });

  it('resolves an id to its verified, sha256-matching graph path', () => {
    const resolved = resolveVerifiedInterventionGraphPath(indexPath, 'C000');
    expect(resolved).toBe(join(root, 'C000.bin.gz'));
  });

  it('throws when the id is not present', () => {
    expect(() => resolveVerifiedInterventionGraphPath(indexPath, 'M1000')).toThrow(/id "M1000" not found/);
  });

  it('throws on a gzip sha256 mismatch', () => {
    writeFileSync(join(root, 'P.bin.gz'), Buffer.from('tampered bytes'));
    expect(() => resolveVerifiedInterventionGraphPath(indexPath, 'P')).toThrow(/gzip sha256 .* does not match/);
  });

  it('throws when the referenced graph file is missing', () => {
    writeFileSync(
      indexPath,
      JSON.stringify({ entries: [{ id: 'ghost', path: 'ghost.bin.gz', gzipSha256: 'x'.repeat(64) }] })
    );
    expect(() => resolveVerifiedInterventionGraphPath(indexPath, 'ghost')).toThrow(/cannot read/);
  });

  it('throws on a malformed entry', () => {
    writeFileSync(indexPath, JSON.stringify({ entries: [{ id: 'P' }] }));
    expect(() => readInterventionIndex(indexPath)).toThrow(/malformed entry/);
  });

  it('throws on a duplicate id', () => {
    writeFileSync(
      indexPath,
      JSON.stringify({
        entries: [
          { id: 'P', path: 'P.bin.gz', gzipSha256: sha256Hex(pBytes) },
          { id: 'P', path: 'C000.bin.gz', gzipSha256: sha256Hex(cBytes) }
        ]
      })
    );
    expect(() => readInterventionIndex(indexPath)).toThrow(/lists id "P" more than once/);
  });

  it('throws when index.json does not exist', () => {
    expect(() => readInterventionIndex(join(root, 'does-not-exist.json'))).toThrow(/cannot read\/parse/);
  });

  it('throws when index.json has no entries', () => {
    writeFileSync(indexPath, JSON.stringify({ entries: [] }));
    expect(() => readInterventionIndex(indexPath)).toThrow(/has no "entries"/);
  });
});

describe('lookup-intervention-graph.ts CLI', () => {
  let root: string;
  let indexPath: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'lookup-intervention-graph-cli-'));
    indexPath = join(root, 'index.json');
    const bytes = Buffer.from('M1000 graph bytes');
    writeFileSync(join(root, 'M1000.bin.gz'), bytes);
    writeFileSync(
      indexPath,
      JSON.stringify({ entries: [{ id: 'M1000', path: 'M1000.bin.gz', gzipSha256: sha256Hex(bytes) }] })
    );
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('prints the resolved graph path to stdout and exits 0 on success', () => {
    const result = runCli([indexPath, 'M1000']);
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout.trim()).toBe(join(root, 'M1000.bin.gz'));
  });

  it('exits 1 with a stderr message when the id is missing', () => {
    const result = runCli([indexPath, 'does-not-exist']);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/id "does-not-exist" not found/);
    expect(result.stdout).toBe('');
  });

  it('exits 1 with a usage message when called with too few arguments', () => {
    const result = runCli([indexPath]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/Usage:/);
  });
});
