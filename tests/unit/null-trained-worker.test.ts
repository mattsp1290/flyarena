import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { resolveArenaTask } from '../../src/lib/arena/tasks';
import { NEURAL_SUBSTEPS_PER_TICK } from '../../src/lib/connectome/constants';
import { runExportArms } from '../../scripts/training/export-arms';
import { runTask, type NullTrainedWorkerTask } from '../../scripts/null/null-trained-worker';
import { writeTinyRunDir } from '../fixtures/trained-readout-run';

/**
 * Coverage for `scripts/null/null-trained-worker.ts`'s `runTask` — a
 * dual-review finding: the first version of this branch had no direct test
 * of the worker's run-directory/arm-bundle integrity checks
 * (`assertRunMatchesExpectedIdentity`, the arm-bundle sha256 check), only
 * an end-to-end manual `--dry-run-fixture` run. Fixtures use the same real
 * `runExportArms`/`writeTinyRunDir` helpers `tests/unit/null-trained-evaluate.test.ts`
 * uses.
 */

describe('runTask', () => {
  let root: string;
  let runDir: string;
  let armBundlePath: string;
  let d: number;
  let bundleSha256: string;

  const baseTask = (overrides: Partial<NullTrainedWorkerTask> = {}): NullTrainedWorkerTask => ({
    graphId: 'test',
    runDir,
    armBundlePath,
    heldOutSeeds: [30001, 30002],
    ticks: 20,
    expectedArm: 'rewired',
    expectedTrainerSeed: 101,
    expectedSubsteps: NEURAL_SUBSTEPS_PER_TICK,
    expectedHiddenSize: 4,
    expectedArenaTaskFingerprint: resolveArenaTask().fingerprint,
    ...overrides
  });

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'null-trained-worker-'));
    const exported = runExportArms({ fixtureRewire: true, fixtureRewireSeed: 0, outDir: join(root, 'arms') });
    const bundlePath = resolve(exported.outDir, 'rewired.json');
    const bundle = JSON.parse(readFileSync(bundlePath, 'utf8')) as { sha256: string; D: number };
    armBundlePath = bundlePath;
    d = bundle.D;
    bundleSha256 = bundle.sha256;

    runDir = join(root, 'run');
    writeTinyRunDir({
      dir: runDir,
      arm: 'rewired',
      trainerSeed: 101,
      D: d,
      H: 4,
      substeps: NEURAL_SUBSTEPS_PER_TICK,
      weightSeed: 7,
      armBundleSha256: bundleSha256
    });
  });

  afterEach(() => rmSync(root, { recursive: true, force: true }));

  it('scores every held-out seed and returns finite results', () => {
    const results = runTask(baseTask());
    expect(results).toHaveLength(2);
    expect(results.map((r) => r.seed)).toEqual([30001, 30002]);
    for (const r of results) {
      expect(Number.isFinite(r.movementScore)).toBe(true);
      expect(Number.isFinite(r.foodPickups)).toBe(true);
      expect(Number.isFinite(r.hazardContacts)).toBe(true);
    }
  });

  it('throws when the run was trained for a different arm', () => {
    expect(() => runTask(baseTask({ expectedArm: 'biological' }))).toThrow(/has arm "rewired", expected "biological"/);
  });

  it('throws when the run was trained at a different trainer seed', () => {
    expect(() => runTask(baseTask({ expectedTrainerSeed: 202 }))).toThrow(
      /has trainerSeed 101, expected 202/
    );
  });

  it('throws when the run was trained at a different substep count', () => {
    expect(() => runTask(baseTask({ expectedSubsteps: NEURAL_SUBSTEPS_PER_TICK + 1 }))).toThrow(
      /was trained at substeps=/
    );
  });

  it('throws when the run was trained with a different hidden size', () => {
    expect(() => runTask(baseTask({ expectedHiddenSize: 999 }))).toThrow(/has H=4, expected 999/);
  });

  it('throws when config.json has no armBundleSha256', () => {
    const configPath = join(runDir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    delete config.armBundleSha256;
    writeFileSync(configPath, JSON.stringify(config));
    expect(() => runTask(baseTask())).toThrow(/has no armBundleSha256/);
  });

  it('throws when config.json was trained against a different arm bundle sha256', () => {
    const configPath = join(runDir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.armBundleSha256 = 'f'.repeat(64);
    writeFileSync(configPath, JSON.stringify(config));
    expect(() => runTask(baseTask())).toThrow(/was trained against arm bundle sha256/);
  });

  it('throws when the arm bundle file has been tampered with (sha256 no longer matches its content)', () => {
    const bundle = JSON.parse(readFileSync(armBundlePath, 'utf8')) as Record<string, unknown>;
    writeFileSync(armBundlePath, JSON.stringify({ ...bundle, D: (bundle.D as number) + 1 }));
    expect(() => runTask(baseTask())).toThrow(/sha256 does not match its content/);
  });

  it('throws when the run was trained for a different arena task (fifth identity check)', () => {
    expect(() => runTask(baseTask({ expectedArenaTaskFingerprint: resolveArenaTask('hazard-heavy').fingerprint }))).toThrow(
      /has arenaTaskFingerprint/
    );
  });

  it('throws when config.json predates arena tasks (no recorded arenaTaskFingerprint at all)', () => {
    const configPath = join(runDir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    delete config.arenaTask;
    delete config.arenaTaskFingerprint;
    writeFileSync(configPath, JSON.stringify(config));
    expect(() => runTask(baseTask())).toThrow(/has arenaTaskFingerprint undefined/);
  });

  it('throws (readRunDir type validation) when config.json has a non-string arenaTaskFingerprint', () => {
    const configPath = join(runDir, 'config.json');
    const config = JSON.parse(readFileSync(configPath, 'utf8')) as Record<string, unknown>;
    config.arenaTaskFingerprint = 12345;
    writeFileSync(configPath, JSON.stringify(config));
    expect(() => runTask(baseTask())).toThrow(/"arenaTaskFingerprint" must be a string when present/);
  });
});
