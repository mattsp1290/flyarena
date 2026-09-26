// @vitest-environment node
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { resolveArenaTask } from '../../src/lib/arena/tasks';
import { assembleGraphListRaw, assembleRaw, buildTasks, runNullEvaluate } from '../../scripts/null/null-evaluate';

/**
 * Extracted from `null-evaluate.test.ts` (which was already at the repo's
 * 1000-line review-blocker threshold before this WP's own additions — see
 * `null-worker-arena-task.test.ts` for the same reasoning). Covers
 * `assembleRaw`/`assembleGraphListRaw`'s `--arena-task` byte-identity gate:
 * `arenaTask`/`arenaTaskFingerprint` must be entirely absent from the raw
 * output when no `--arena-task` was passed, and present (matching
 * `resolveArenaTask`) when it was.
 */

const rewireIndex = {
  sourceArtifact: 'src.bin.gz',
  sourceSha256: 'c'.repeat(64),
  rewireSourceSha256: 'd'.repeat(64),
  seeds: [
    {
      seed: 0,
      artifact: 'rewired-0.bin.gz',
      binarySha256: '0'.padStart(64, '0'),
      binaryBytes: 1,
      gzipSha256: '0'.padStart(64, '0'),
      gzipBytes: 1,
      stats: { acceptedSwaps: 1, attempts: 1 }
    }
  ]
};
const graphListIndex = {
  sourceArtifact: 'src.bin.gz',
  sourceSha256: 'c'.repeat(64),
  entries: [{ id: 'P', path: 'p.bin.gz', gzipSha256: 'a'.repeat(64), binarySha256: '1'.repeat(64) }]
};
const baseArgs = () => ({
  biological: false,
  graph: undefined,
  rewiredIndex: 'unused',
  graphsDir: 'unused',
  graphList: undefined as string | undefined,
  heldOutStart: 30001,
  heldOutCount: 1,
  ticks: 20,
  shards: 1,
  out: 'unused.json',
  decoder: 'authored' as const,
  rewiredSeeds: undefined as { start: number; end: number } | undefined,
  arenaTask: undefined as string | undefined
});
const oneResult = new Map([['rewired-0', [{ seed: 30001, movementScore: 1, foodPickups: 0, hazardContacts: 0 }]]]);
const oneGraphListResult = new Map([['P', [{ seed: 30001, movementScore: 1, foodPickups: 0, hazardContacts: 0 }]]]);

describe('assembleRaw / assembleGraphListRaw: --arena-task byte-identity gate', () => {
  it('assembleRaw: no arenaTask/arenaTaskFingerprint key when --arena-task is omitted (byte-identity gate)', () => {
    const raw = assembleRaw(rewireIndex, baseArgs(), oneResult);
    expect(raw.arenaTask).toBeUndefined();
    expect(raw.arenaTaskFingerprint).toBeUndefined();
    expect(JSON.stringify(raw)).not.toContain('arenaTask');
  });

  it('assembleRaw: records the requested arena task id/fingerprint when --arena-task is passed', () => {
    const raw = assembleRaw(rewireIndex, { ...baseArgs(), arenaTask: 'no-movement' }, oneResult);
    expect(raw.arenaTask).toBe('no-movement');
    expect(raw.arenaTaskFingerprint).toBe(resolveArenaTask('no-movement').fingerprint);
  });

  it('assembleGraphListRaw: no arenaTask/arenaTaskFingerprint key when --arena-task is omitted (byte-identity gate)', () => {
    const raw = assembleGraphListRaw(graphListIndex, baseArgs(), oneGraphListResult);
    expect(raw.arenaTask).toBeUndefined();
    expect(raw.arenaTaskFingerprint).toBeUndefined();
    expect(JSON.stringify(raw)).not.toContain('arenaTask');
  });

  it('assembleGraphListRaw: records the requested arena task id/fingerprint when --arena-task is passed', () => {
    const raw = assembleGraphListRaw(graphListIndex, { ...baseArgs(), arenaTask: 'crowded' }, oneGraphListResult);
    expect(raw.arenaTask).toBe('crowded');
    expect(raw.arenaTaskFingerprint).toBe(resolveArenaTask('crowded').fingerprint);
  });

  it('buildTasks: every task carries the requested arena task, undefined when omitted', () => {
    const variantArgs = { ...baseArgs(), decoder: 'authored-flip-both' as const, arenaTask: 'hazard-heavy' as const };
    for (const task of buildTasks(rewireIndex, variantArgs, 'bio.bin.gz')) {
      expect(task.decoder).toBe('authored-flip-both');
      expect(task.arenaTask).toBe('hazard-heavy');
    }
    for (const task of buildTasks(rewireIndex, baseArgs(), 'bio.bin.gz')) {
      expect(task.arenaTask).toBeUndefined();
    }
  });
});

describe('runNullEvaluate: guardCanonicalOutDefault treats any --arena-task as non-canonical', () => {
  // Mirrors null-evaluate.test.ts's own "refuses to overwrite the canonical
  // default --out" suite, scoped to the --arena-task branch specifically
  // (`args.arenaTask !== undefined` in guardCanonicalOutDefault).
  const argsWithDefaultOut = (arenaTask: string | undefined) => ({
    biological: true,
    graph: undefined,
    rewiredIndex: '/nonexistent/index.json',
    graphsDir: '/nonexistent/graphs',
    heldOutStart: 30001,
    heldOutCount: 100,
    ticks: 1800,
    shards: 1,
    out: resolve(process.cwd(), 'training/runs/null/authored.json'),
    decoder: 'authored' as const,
    rewiredSeeds: undefined as { start: number; end: number } | undefined,
    arenaTask
  });

  it('throws for a non-default --arena-task run writing to the default --out', async () => {
    await expect(runNullEvaluate(argsWithDefaultOut('hazard-heavy'))).rejects.toThrow(
      /refusing to write a non-canonical run/
    );
  });

  it("throws even for --arena-task 'default' explicitly passed: it still adds arenaTask/arenaTaskFingerprint keys arena-task-fields.ts omits for a true no-flag run, so it is not byte-identical to the canonical default", async () => {
    await expect(runNullEvaluate(argsWithDefaultOut('default'))).rejects.toThrow(/refusing to write a non-canonical run/);
  });

  it('does not throw this guard when --arena-task is omitted entirely (fails later, on the nonexistent index instead)', async () => {
    await expect(runNullEvaluate(argsWithDefaultOut(undefined))).rejects.not.toThrow(
      /refusing to write a non-canonical run/
    );
  });
});
