// @vitest-environment node
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

/**
 * Coverage for `scripts/null/train-sample.sh`'s CEM-config manifest
 * pre-flight (`read_manifest_cem_config`, run before any
 * export-arms/flyarena-train call -- see the script's own module doc
 * comment) -- a thermo-architecture/thermo-maintainability review finding:
 * the script previously hard-coded its CEM defaults as literals with a
 * one-time, manual "verified byte-for-byte against the manifest" comment
 * and no repeatable check. `TRAIN_SAMPLE_MANIFEST_PATH` (an env var the
 * script documents as existing "for this script's own tests") lets this
 * suite point the pre-flight at fixture manifests without touching the
 * real, shipped `public/data/trained-readout-v1.manifest.json`.
 *
 * Every case here runs with `--dry-run-fixture` and small overrides so
 * nothing here ever spins up `flyarena-train`/the Python training
 * toolchain: a bad manifest must fail during the pre-flight itself, before
 * `train-sample.sh` ever prints its "exporting fixture-rewired arms"
 * message (the first line the seed loop prints) -- asserting that message
 * never appears is how each failure case below proves the pre-flight
 * rejected the manifest before any work started, not merely that the whole
 * script eventually exited non-zero for some unrelated reason.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const scriptPath = resolve(repoRoot, 'scripts/null/train-sample.sh');

/**
 * `--arms-out`/`--trained-out` point inside this test's own tmp `root`
 * (never the script's real default `training/runs/null/*-dry-run-fixture`
 * directories) so this suite is isolated from, and never pollutes, the
 * shared gitignored `training/runs/null/` tree -- and so repeated runs are
 * never short-circuited by `train-sample.sh`'s own resumability check
 * (`<trained-out>/seed<s>/config.json already exists -- skipping`) finding
 * a leftover directory from a previous test run.
 */
const runPreflight = (root: string, manifestPath: string, timeout = 30_000) =>
  spawnSync(
    'bash',
    [
      scriptPath,
      '--dry-run-fixture',
      '--seed-count',
      '1',
      '--generations',
      '1',
      '--population',
      '1',
      '--elites',
      '1',
      '--train-seeds-per-generation',
      '1',
      '--arms-out',
      join(root, 'arms'),
      '--trained-out',
      join(root, 'trained')
    ],
    {
      encoding: 'utf8',
      cwd: repoRoot,
      timeout,
      env: { ...process.env, TRAIN_SAMPLE_MANIFEST_PATH: manifestPath }
    }
  );

describe('train-sample.sh manifest pre-flight', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'train-sample-preflight-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('fails fast, before any export-arms work, when the manifest file does not exist', () => {
    const result = runPreflight(root, join(root, 'does-not-exist.json'));
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/cannot read manifest/);
    expect(result.stdout).not.toContain('exporting fixture-rewired arms');
  });

  it('fails fast when the manifest is not valid JSON', () => {
    const manifestPath = join(root, 'invalid.json');
    writeFileSync(manifestPath, '{ not valid json');
    const result = runPreflight(root, manifestPath);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/is not valid JSON/);
    expect(result.stdout).not.toContain('exporting fixture-rewired arms');
  });

  it('fails fast when the manifest has no "training" object', () => {
    const manifestPath = join(root, 'no-training.json');
    writeFileSync(manifestPath, JSON.stringify({ H: 16 }));
    const result = runPreflight(root, manifestPath);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/has no "training" object/);
    expect(result.stdout).not.toContain('exporting fixture-rewired arms');
  });

  it('fails fast when a required training field is missing', () => {
    const manifestPath = join(root, 'missing-field.json');
    writeFileSync(
      manifestPath,
      JSON.stringify({
        H: 16,
        training: { population: 128, elites: 32, generations: 150, alpha: 0.7, stdFloor: 0.02, initStd: 0.5 }
        // trainingSeedsPerGeneration deliberately omitted
      })
    );
    const result = runPreflight(root, manifestPath);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/training\.trainingSeedsPerGeneration.*missing or not an integer/);
    expect(result.stdout).not.toContain('exporting fixture-rewired arms');
  });

  it('fails fast when a required training field is non-numeric', () => {
    const manifestPath = join(root, 'bad-type.json');
    writeFileSync(
      manifestPath,
      JSON.stringify({
        H: 16,
        training: {
          population: 128,
          elites: 32,
          generations: 150,
          alpha: 0.7,
          stdFloor: 0.02,
          initStd: 0.5,
          trainingSeedsPerGeneration: 'sixteen'
        }
      })
    );
    const result = runPreflight(root, manifestPath);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/training\.trainingSeedsPerGeneration.*missing or not an integer/);
    expect(result.stdout).not.toContain('exporting fixture-rewired arms');
  });

  it('fails fast when the top-level H field is missing', () => {
    const manifestPath = join(root, 'no-h.json');
    writeFileSync(
      manifestPath,
      JSON.stringify({
        training: {
          population: 128,
          elites: 32,
          generations: 150,
          alpha: 0.7,
          stdFloor: 0.02,
          initStd: 0.5,
          trainingSeedsPerGeneration: 16
        }
      })
    );
    const result = runPreflight(root, manifestPath);
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/'s H is missing or not an integer/);
    expect(result.stdout).not.toContain('exporting fixture-rewired arms');
  });

  it('reaches the seed loop (passes the pre-flight) with a valid manifest, without waiting for training to finish', { timeout: 15_000 }, () => {
    const manifestPath = join(root, 'valid.json');
    writeFileSync(
      manifestPath,
      JSON.stringify({
        H: 16,
        training: {
          population: 128,
          elites: 32,
          generations: 150,
          alpha: 0.7,
          stdFloor: 0.02,
          initStd: 0.5,
          trainingSeedsPerGeneration: 16
        }
      })
    );
    // Every CEM flag is explicitly overridden by this call's own argv, so
    // this only proves the pre-flight itself didn't reject a valid
    // manifest -- a short timeout kills the script (SIGTERM) once it has
    // reached export-arms, well before the real Python/flyarena-train
    // toolchain would finish (exercised separately by this repo's manual
    // `--dry-run-fixture` verification, not this fast unit test).
    const result = runPreflight(root, manifestPath, 8_000);
    expect(result.stdout).toContain('exporting fixture-rewired arms');
  });
});
