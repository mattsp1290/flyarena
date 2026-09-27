import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { assertArmBundlesMatchRawScores, copyRawInterventionScores } from '../../scripts/attribution/raw-intervention-scores';

/**
 * Coverage for `scripts/attribution/raw-intervention-scores.ts` (split out
 * of `archive-readouts.ts` -- a thermo-maintainability review finding).
 */

let root: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'raw-intervention-scores-'));
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
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
});

describe('assertArmBundlesMatchRawScores', () => {
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
