import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  assertArmBundlesMatchRawScores,
  copyRawInterventionScores,
  readRawInterventionScores
} from '../../scripts/attribution/raw-intervention-scores';

/**
 * Coverage for `scripts/attribution/raw-intervention-scores.ts` (split out
 * of `archive-readouts.ts` -- a thermo-maintainability review finding).
 *
 * This file also covers the WP1b per-task variant (`RawScoresFileOptions.
 * expectedArenaTask`, and `ArmBundleMatchOptions.kind`/`arenaTask` on
 * `assertArmBundlesMatchRawScores`) -- WP1b originally split that variant
 * into its own near-duplicate module/test file
 * (`task-intervention-raw-scores.ts`/`.test.ts`), which a
 * thermo-maintainability review found to be ~80% copy-pasted from this
 * module. Both concerns were merged into one parameterized module; these
 * tests were folded in alongside it rather than left in a separate file.
 */

let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'raw-intervention-scores-'));
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

const wellFormedTaskPayload = (arenaTask: string) => ({
  version: 1,
  arenaTask,
  arenaTaskFingerprint: `fingerprint-for-${arenaTask}`,
  graphListSha256: 'a'.repeat(64),
  runs: [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'sha', gzipSha256: 'gzip-sha', movementScore: [1, 2, 3] }]
});

describe('readRawInterventionScores (default-task mode, no options)', () => {
  it('reads a well-formed default-task file', () => {
    const src = resolve(root, 'trained.json');
    writeFileSync(src, JSON.stringify({ version: 1, runs: [{ id: 'C000', trainerSeed: 101, movementScore: [1, 2, 3] }] }));
    const { runs, parsed } = readRawInterventionScores(src);
    expect(runs).toHaveLength(1);
    expect(parsed.runs).toBe(runs);
  });

  it('refuses a file with no "runs" array', () => {
    const src = resolve(root, 'bad.json');
    writeFileSync(src, JSON.stringify({ version: 1 }));
    expect(() => readRawInterventionScores(src)).toThrow(/no "runs" array/);
  });

  it('refuses a malformed run entry', () => {
    const src = resolve(root, 'bad-run.json');
    writeFileSync(src, JSON.stringify({ runs: [{ id: 'C000' }] }));
    expect(() => readRawInterventionScores(src)).toThrow(/malformed run entry/);
  });

  it('does NOT require armBundleSha256/gzipSha256 per row (WP1 back-compat: no expectedArenaTask option)', () => {
    const src = resolve(root, 'no-arm-bundle.json');
    writeFileSync(src, JSON.stringify({ runs: [{ id: 'C000', trainerSeed: 101, movementScore: [1] }] }));
    expect(() => readRawInterventionScores(src)).not.toThrow();
  });
});

describe('readRawInterventionScores (task mode, expectedArenaTask option)', () => {
  it('reads a well-formed file for the expected arena task', () => {
    const src = resolve(root, 'trained.json');
    writeFileSync(src, JSON.stringify(wellFormedTaskPayload('hazard-heavy')));
    const { parsed } = readRawInterventionScores(src, { expectedArenaTask: 'hazard-heavy' });
    expect(parsed.arenaTask).toBe('hazard-heavy');
    expect(parsed.runs).toHaveLength(1);
  });

  it('refuses a file whose arenaTask does not match the expected label', () => {
    const src = resolve(root, 'trained.json');
    writeFileSync(src, JSON.stringify(wellFormedTaskPayload('hazard-heavy')));
    expect(() => readRawInterventionScores(src, { expectedArenaTask: 'sparse-food' })).toThrow(
      /arenaTask "hazard-heavy", expected "sparse-food"/
    );
  });

  it('refuses a file with no graphListSha256', () => {
    const src = resolve(root, 'bad.json');
    const payload = wellFormedTaskPayload('crowded') as Record<string, unknown>;
    delete payload.graphListSha256;
    writeFileSync(src, JSON.stringify(payload));
    expect(() => readRawInterventionScores(src, { expectedArenaTask: 'crowded' })).toThrow(/no "graphListSha256"/);
  });

  it('refuses a file with no "runs" array', () => {
    const src = resolve(root, 'bad.json');
    writeFileSync(src, JSON.stringify({ arenaTask: 'crowded', graphListSha256: 'a'.repeat(64) }));
    expect(() => readRawInterventionScores(src, { expectedArenaTask: 'crowded' })).toThrow(/no "runs" array/);
  });

  it('refuses a malformed run entry', () => {
    const src = resolve(root, 'bad-run.json');
    writeFileSync(src, JSON.stringify({ arenaTask: 'crowded', graphListSha256: 'a'.repeat(64), runs: [{ id: 'C000' }] }));
    expect(() => readRawInterventionScores(src, { expectedArenaTask: 'crowded' })).toThrow(/malformed run entry/);
  });

  it('refuses a row with no armBundleSha256 (a thermo-provenance review finding: tighter than default-task mode)', () => {
    const src = resolve(root, 'no-arm-bundle.json');
    const payload = wellFormedTaskPayload('crowded');
    writeFileSync(
      src,
      JSON.stringify({ ...payload, runs: [{ id: 'C000', trainerSeed: 101, gzipSha256: 'g', movementScore: [1] }] })
    );
    expect(() => readRawInterventionScores(src, { expectedArenaTask: 'crowded' })).toThrow(
      /run "C000" \(trainerSeed 101\) has no armBundleSha256/
    );
  });

  it('refuses a row with no gzipSha256 (a thermo-provenance review finding)', () => {
    const src = resolve(root, 'no-gzip.json');
    const payload = wellFormedTaskPayload('crowded');
    writeFileSync(
      src,
      JSON.stringify({ ...payload, runs: [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'a', movementScore: [1] }] })
    );
    expect(() => readRawInterventionScores(src, { expectedArenaTask: 'crowded' })).toThrow(
      /run "C000" \(trainerSeed 101\) has no gzipSha256/
    );
  });
});

describe('copyRawInterventionScores', () => {
  it('copies a well-formed raw trained.json', () => {
    const src = resolve(root, 'trained.json');
    const out = resolve(root, 'archive', 'intervention-trained-raw-v1.json');
    const payload = { version: 1, runs: [{ id: 'C000', trainerSeed: 101, movementScore: [1, 2, 3] }] };
    writeFileSync(src, JSON.stringify(payload));

    copyRawInterventionScores(src, out);
    const written = JSON.parse(readFileSync(out, 'utf8'));
    expect(written).toEqual(payload);
  });

  it('copies non-canonical (pretty-printed) input byte-for-byte, not re-encoded', () => {
    const src = resolve(root, 'pretty-trained.json');
    const out = resolve(root, 'archive', 'pretty-out.json');
    const text = `${JSON.stringify({ version: 1, runs: [{ id: 'C000', trainerSeed: 101, movementScore: [1] }] }, null, 2)}\n`;
    writeFileSync(src, text);

    copyRawInterventionScores(src, out);
    // Byte identity, not merely structural equality: a JSON.stringify(parsed)
    // re-encode would silently collapse this back to compact form.
    expect(readFileSync(out)).toEqual(readFileSync(src));
  });

  it('refuses a file with no "runs" array', () => {
    const src = resolve(root, 'bad.json');
    writeFileSync(src, JSON.stringify({ version: 1 }));
    expect(() => copyRawInterventionScores(src, resolve(root, 'out.json'))).toThrow(/no "runs" array/);
  });

  it('refuses a malformed run entry', () => {
    const src = resolve(root, 'bad-run.json');
    writeFileSync(src, JSON.stringify({ runs: [{ id: 'C000' }] }));
    expect(() => copyRawInterventionScores(src, resolve(root, 'out.json'))).toThrow(/malformed run entry/);
  });

  it('copies a well-formed task file byte-for-byte, honoring the expectedArenaTask option', () => {
    const src = resolve(root, 'trained.json');
    const out = resolve(root, 'archive', 'task-intervention-trained-raw-no-movement-v1.json');
    const text = `${JSON.stringify(wellFormedTaskPayload('no-movement'), null, 2)}\n`;
    writeFileSync(src, text);

    copyRawInterventionScores(src, out, { expectedArenaTask: 'no-movement' });
    expect(readFileSync(out)).toEqual(readFileSync(src));
  });

  it('refuses to copy a task file when the arenaTask label disagrees', () => {
    const src = resolve(root, 'trained.json');
    writeFileSync(src, JSON.stringify(wellFormedTaskPayload('no-movement')));
    expect(() => copyRawInterventionScores(src, resolve(root, 'out.json'), { expectedArenaTask: 'crowded' })).toThrow(
      /expected "crowded"/
    );
  });
});

describe('assertArmBundlesMatchRawScores (default options: kind "intervention")', () => {
  it('does not throw when armBundleSha256 values agree', () => {
    const additions = [{ kind: 'intervention', graphId: 'C000', trainerSeed: 101, armBundleSha256: 'shared-sha' }];
    const rawRuns = [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'shared-sha', movementScore: [] }];
    expect(() => assertArmBundlesMatchRawScores(additions, rawRuns)).not.toThrow();
  });

  it('throws when an intervention addition\'s armBundleSha256 disagrees with the raw file (mislabeled --source)', () => {
    const additions = [{ kind: 'intervention', graphId: 'C000', trainerSeed: 101, armBundleSha256: 'wrong-sha' }];
    const rawRuns = [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'correct-sha', movementScore: [] }];
    expect(() => assertArmBundlesMatchRawScores(additions, rawRuns)).toThrow(/mismatched --source label/);
  });

  it('ignores a bigq addition (never cross-checked against the intervention raw file)', () => {
    const additions = [{ kind: 'bigq', graphId: 'biological', trainerSeed: 101, armBundleSha256: 'anything' }];
    const rawRuns = [{ id: 'biological', trainerSeed: 101, armBundleSha256: 'something-else', movementScore: [] }];
    expect(() => assertArmBundlesMatchRawScores(additions, rawRuns)).not.toThrow();
  });

  it('ignores an addition with no matching raw entry', () => {
    const additions = [{ kind: 'intervention', graphId: 'C099', trainerSeed: 101, armBundleSha256: 'x' }];
    expect(() => assertArmBundlesMatchRawScores(additions, [])).not.toThrow();
  });
});

describe('assertArmBundlesMatchRawScores (task options: kind "task-intervention" + arenaTask)', () => {
  const arenaTask = 'hazard-heavy';
  const options = { kind: 'task-intervention', arenaTask } as const;

  it('does not throw when armBundleSha256 values agree', () => {
    const additions = [{ kind: 'task-intervention', graphId: 'C000', trainerSeed: 101, armBundleSha256: 'shared-sha', arenaTask }];
    const rawRuns = [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'shared-sha', movementScore: [] }];
    expect(() => assertArmBundlesMatchRawScores(additions, rawRuns, options)).not.toThrow();
  });

  it('throws when a task-intervention addition\'s armBundleSha256 disagrees with the raw file (mislabeled --source)', () => {
    const additions = [{ kind: 'task-intervention', graphId: 'C000', trainerSeed: 101, armBundleSha256: 'wrong-sha', arenaTask }];
    const rawRuns = [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'correct-sha', movementScore: [] }];
    expect(() => assertArmBundlesMatchRawScores(additions, rawRuns, options)).toThrow(/mismatched --source label/);
  });

  it('ignores a task-intervention addition for a DIFFERENT arena task (each raw file only covers one task)', () => {
    const additions = [
      { kind: 'task-intervention', graphId: 'C000', trainerSeed: 101, armBundleSha256: 'wrong-sha', arenaTask: 'sparse-food' }
    ];
    const rawRuns = [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'correct-sha', movementScore: [] }];
    expect(() => assertArmBundlesMatchRawScores(additions, rawRuns, options)).not.toThrow();
  });

  it('ignores a default-task intervention/bigq addition (kind filter excludes it)', () => {
    const additions = [{ kind: 'intervention', graphId: 'C000', trainerSeed: 101, armBundleSha256: 'anything', arenaTask: 'default' }];
    const rawRuns = [{ id: 'C000', trainerSeed: 101, armBundleSha256: 'something-else', movementScore: [] }];
    expect(() => assertArmBundlesMatchRawScores(additions, rawRuns, options)).not.toThrow();
  });

  it('ignores an addition with no matching raw entry', () => {
    const additions = [{ kind: 'task-intervention', graphId: 'C099', trainerSeed: 101, armBundleSha256: 'x', arenaTask }];
    expect(() => assertArmBundlesMatchRawScores(additions, [], options)).not.toThrow();
  });
});
