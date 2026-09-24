// @vitest-environment node
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { lookupRewiredArtifact } from '../../scripts/null/lookup-rewired-artifact';

/**
 * Coverage for `scripts/null/lookup-rewired-artifact.ts` -- the TS CLI
 * `train-sample.sh`'s `rewired_artifact_for_seed` shells out to instead of
 * its own ad hoc, unvalidated `python3` heredoc (a thermo-maintainability
 * review finding: reuse `null-evaluate.ts`'s already-validated
 * `readRewireIndex`/`RewireIndex` schema rather than a second,
 * disconnected implementation of the same lookup). Two halves: the
 * exported `lookupRewiredArtifact` function (in-process), and the CLI
 * entrypoint itself (`spawnSync`, matching `null-evaluate.test.ts`'s own
 * CLI-testing convention) so `train-sample.sh`'s actual invocation shape
 * (stdout-only success, stderr + exit 1 on failure) is proven end-to-end.
 */

const runCli = (args: readonly string[]) =>
  spawnSync(process.execPath, ['--import', 'tsx', 'scripts/null/lookup-rewired-artifact.ts', ...args], {
    encoding: 'utf8',
    timeout: 30_000
  });

const seedEntry = (seed: number, artifact: string) => ({
  seed,
  artifact,
  binarySha256: 'a'.repeat(64),
  binaryBytes: 1,
  gzipSha256: 'b'.repeat(64),
  gzipBytes: 1,
  stats: { acceptedSwaps: 1, attempts: 1 }
});

const buildIndex = (seeds: readonly { seed: number; artifact: string }[]) => ({
  sourceArtifact: 'src.bin.gz',
  sourceSha256: 'c'.repeat(64),
  rewireSourceSha256: 'd'.repeat(64),
  seeds: seeds.map((s) => seedEntry(s.seed, s.artifact))
});

describe('lookupRewiredArtifact', () => {
  let root: string;
  let indexPath: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'lookup-rewired-artifact-'));
    indexPath = join(root, 'index.json');
    writeFileSync(
      indexPath,
      JSON.stringify(
        buildIndex([
          { seed: 0, artifact: 'seed0.bin.gz' },
          { seed: 5, artifact: 'seed5.bin.gz' }
        ])
      )
    );
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('returns the matching entry\'s artifact field', () => {
    expect(lookupRewiredArtifact(indexPath, 5)).toBe('seed5.bin.gz');
    expect(lookupRewiredArtifact(indexPath, 0)).toBe('seed0.bin.gz');
  });

  it('throws a clear error when the seed is not present', () => {
    expect(() => lookupRewiredArtifact(indexPath, 99)).toThrow(/seed 99 not found/);
  });

  it('propagates readRewireIndex\'s schema validation on a malformed index.json', () => {
    const badIndexPath = join(root, 'bad-index.json');
    writeFileSync(
      badIndexPath,
      JSON.stringify({
        sourceArtifact: 'src.bin.gz',
        sourceSha256: 'c'.repeat(64),
        rewireSourceSha256: 'd'.repeat(64),
        seeds: [{ seed: 1.5, artifact: 'x.bin.gz' }] // non-integer seed, missing required fields
      })
    );
    expect(() => lookupRewiredArtifact(badIndexPath, 1)).toThrow(/malformed seed entry/);
  });

  it('throws when index.json does not exist', () => {
    expect(() => lookupRewiredArtifact(join(root, 'does-not-exist.json'), 0)).toThrow();
  });
});

describe('lookup-rewired-artifact.ts CLI', () => {
  let root: string;
  let indexPath: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'lookup-rewired-artifact-cli-'));
    indexPath = join(root, 'index.json');
    writeFileSync(indexPath, JSON.stringify(buildIndex([{ seed: 7, artifact: 'seed7.bin.gz' }])));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('prints the artifact filename to stdout and exits 0 on success', () => {
    const result = runCli([indexPath, '7']);
    expect(result.status, result.stderr).toBe(0);
    expect(result.stdout.trim()).toBe('seed7.bin.gz');
  });

  it('exits 1 with a stderr message when the seed is missing', () => {
    const result = runCli([indexPath, '999']);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/seed 999 not found/);
    expect(result.stdout).toBe('');
  });

  it('exits 1 with a usage message when called with too few arguments', () => {
    const result = runCli([indexPath]);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/Usage:/);
  });
});
