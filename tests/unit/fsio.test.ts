// @vitest-environment node
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { gitRev } from '../../scripts/training/fsio';

/**
 * Coverage for `gitRev` — moved here from `scripts/null/null-trained-evaluate.ts`
 * (WP4 of `.agents/plans/pathway-interventions`, thermo-methodology review
 * I2) so `scripts/null/null-evaluate.ts` can use the identical helper
 * without a circular import. Was previously untested in its original home;
 * added here as part of the move.
 */
describe('gitRev', () => {
  it('returns a 40-character hex commit sha when run inside this repo checkout', () => {
    const rev = gitRev(process.cwd());
    expect(rev).toMatch(/^[0-9a-f]{40}$/);
  });

  let scratchDir: string;
  beforeEach(() => {
    scratchDir = mkdtempSync(join(tmpdir(), 'gitrev-not-a-repo-'));
  });
  afterEach(() => {
    rmSync(scratchDir, { recursive: true, force: true });
  });

  it('returns null (never throws) when cwd is not a git checkout', () => {
    expect(gitRev(scratchDir)).toBeNull();
  });
});
