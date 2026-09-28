import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  assertTaskArmBundlesMatchRawScores,
  copyTaskInterventionRawScores,
  readTaskInterventionRawScores
} from '../../scripts/attribution/task-intervention-raw-scores';

/**
 * Coverage for `scripts/attribution/task-intervention-raw-scores.ts` (WP1b,
 * `.agents/plans/readout-attribution/01-archive-and-types.md`), mirroring
 * `raw-intervention-scores.test.ts`'s own coverage pattern for its WP1
 * sibling module.
 */

let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'task-intervention-raw-scores-'));
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

const wellFormedPayload = (arenaTask: string) => ({
  version: 1,
  arenaTask,
  arenaTaskFingerprint: `fingerprint-for-${arenaTask}`,
  graphListSha256: 'a'.repeat(64),
  runs: [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'sha', movementScore: [1, 2, 3] }]
});

describe('readTaskInterventionRawScores', () => {
  it('reads a well-formed file for the expected arena task', () => {
    const src = resolve(root, 'trained.json');
    writeFileSync(src, JSON.stringify(wellFormedPayload('hazard-heavy')));
    const { parsed } = readTaskInterventionRawScores(src, 'hazard-heavy');
    expect(parsed.arenaTask).toBe('hazard-heavy');
    expect(parsed.runs).toHaveLength(1);
  });

  it('refuses a file whose arenaTask does not match the expected label', () => {
    const src = resolve(root, 'trained.json');
    writeFileSync(src, JSON.stringify(wellFormedPayload('hazard-heavy')));
    expect(() => readTaskInterventionRawScores(src, 'sparse-food')).toThrow(/arenaTask "hazard-heavy", expected "sparse-food"/);
  });

  it('refuses a file with no graphListSha256', () => {
    const src = resolve(root, 'bad.json');
    const payload = wellFormedPayload('crowded') as Record<string, unknown>;
    delete payload.graphListSha256;
    writeFileSync(src, JSON.stringify(payload));
    expect(() => readTaskInterventionRawScores(src, 'crowded')).toThrow(/no "graphListSha256"/);
  });

  it('refuses a file with no "runs" array', () => {
    const src = resolve(root, 'bad.json');
    writeFileSync(src, JSON.stringify({ arenaTask: 'crowded', graphListSha256: 'a'.repeat(64) }));
    expect(() => readTaskInterventionRawScores(src, 'crowded')).toThrow(/no "runs" array/);
  });

  it('refuses a malformed run entry', () => {
    const src = resolve(root, 'bad-run.json');
    writeFileSync(
      src,
      JSON.stringify({ arenaTask: 'crowded', graphListSha256: 'a'.repeat(64), runs: [{ id: 'C000' }] })
    );
    expect(() => readTaskInterventionRawScores(src, 'crowded')).toThrow(/malformed run entry/);
  });
});

describe('copyTaskInterventionRawScores', () => {
  it('copies a well-formed file byte-for-byte, not re-encoded', () => {
    const src = resolve(root, 'trained.json');
    const out = resolve(root, 'archive', 'task-intervention-trained-raw-no-movement-v1.json');
    const text = `${JSON.stringify(wellFormedPayload('no-movement'), null, 2)}\n`;
    writeFileSync(src, text);

    copyTaskInterventionRawScores(src, out, 'no-movement');
    expect(readFileSync(out)).toEqual(readFileSync(src));
  });

  it('refuses to copy when the arenaTask label disagrees', () => {
    const src = resolve(root, 'trained.json');
    writeFileSync(src, JSON.stringify(wellFormedPayload('no-movement')));
    expect(() => copyTaskInterventionRawScores(src, resolve(root, 'out.json'), 'crowded')).toThrow(/expected "crowded"/);
  });
});

describe('assertTaskArmBundlesMatchRawScores', () => {
  const arenaTask = 'hazard-heavy';

  it('does not throw when armBundleSha256 values agree', () => {
    const additions = [{ kind: 'task-intervention', graphId: 'C000', trainerSeed: 101, armBundleSha256: 'shared-sha', arenaTask }];
    const rawRuns = [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'shared-sha', movementScore: [] }];
    expect(() => assertTaskArmBundlesMatchRawScores(additions, arenaTask, rawRuns)).not.toThrow();
  });

  it('throws when a task-intervention addition\'s armBundleSha256 disagrees with the raw file (mislabeled --source)', () => {
    const additions = [{ kind: 'task-intervention', graphId: 'C000', trainerSeed: 101, armBundleSha256: 'wrong-sha', arenaTask }];
    const rawRuns = [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'correct-sha', movementScore: [] }];
    expect(() => assertTaskArmBundlesMatchRawScores(additions, arenaTask, rawRuns)).toThrow(/mismatched --source label/);
  });

  it('ignores a task-intervention addition for a DIFFERENT arena task (each raw file only covers one task)', () => {
    const additions = [
      { kind: 'task-intervention', graphId: 'C000', trainerSeed: 101, armBundleSha256: 'wrong-sha', arenaTask: 'sparse-food' }
    ];
    const rawRuns = [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'correct-sha', movementScore: [] }];
    expect(() => assertTaskArmBundlesMatchRawScores(additions, arenaTask, rawRuns)).not.toThrow();
  });

  it('ignores a default-task intervention/bigq addition (never cross-checked against a task raw file)', () => {
    const additions = [{ kind: 'intervention', graphId: 'C000', trainerSeed: 101, armBundleSha256: 'anything', arenaTask: 'default' }];
    const rawRuns = [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'something-else', movementScore: [] }];
    expect(() => assertTaskArmBundlesMatchRawScores(additions, arenaTask, rawRuns)).not.toThrow();
  });

  it('ignores an addition with no matching raw entry', () => {
    const additions = [{ kind: 'task-intervention', graphId: 'C099', trainerSeed: 101, armBundleSha256: 'x', arenaTask }];
    expect(() => assertTaskArmBundlesMatchRawScores(additions, arenaTask, [])).not.toThrow();
  });
});
