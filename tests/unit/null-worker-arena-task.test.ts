// @vitest-environment node
import { createHash } from 'node:crypto';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';

import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { runTask, type NullSeedResult, type NullWorkerTask } from '../../scripts/null/null-worker';
import { createTraceGraph } from '../fixtures/trace-graph';

/**
 * Extracted from `null-evaluate.test.ts` (which was already at the repo's
 * 1000-line review-blocker threshold before this WP's own additions —
 * keeping this coverage in its own file avoids pushing that file further
 * over it).
 *
 * `.agents/plans/task-generality/01-task-plumbing.md`'s WP1 acceptance:
 * "the worker-side createWorld receives the variant config ... and the
 * scores differ from the default on a fixture. Checking only the output
 * metadata is not enough." Calls `null-worker.ts`'s own `runTask` directly
 * (bypassing `fork`/IPC entirely, which a spy could never see across a
 * real child process boundary) and checks the actual computed
 * `movementScore`, not a label — `no-movement`'s `movementScorePerUnit: 0`
 * severs the distance-to-score channel entirely, so an unchanged (still
 * `ARENA_CONFIG`-driven) world would produce numerically identical scores
 * regardless of the requested task.
 */

const sha256Hex = (data: Uint8Array): string => createHash('sha256').update(data).digest('hex');

describe('null-worker runTask: --arena-task propagation (worker-side config spy)', () => {
  const writeFixtureGzip = (dir: string): { path: string; expectedSha256: string } => {
    const graph = createTraceGraph();
    const binary = Buffer.from(encodeGraphBinary(graph));
    const path = join(dir, 'trace-graph.bin.gz');
    writeFileSync(path, gzipSync(binary));
    return { path, expectedSha256: sha256Hex(binary) };
  };

  it('scores differ between the default task and no-movement on the same graph/seeds', () => {
    const root = mkdtempSync(join(tmpdir(), 'null-worker-arena-task-'));
    try {
      const { path, expectedSha256 } = writeFixtureGzip(root);
      const baseTask: NullWorkerTask = {
        graphId: 'trace-graph',
        mode: 'biological',
        path,
        expectedSha256,
        heldOutSeeds: [1, 2, 3],
        ticks: 30
      };
      const sumMovement = (results: readonly NullSeedResult[]) => results.reduce((sum, r) => sum + r.movementScore, 0);

      const defaultTotal = sumMovement(runTask(baseTask));
      const noMovementTotal = sumMovement(runTask({ ...baseTask, arenaTask: 'no-movement' }));

      expect(defaultTotal).not.toBe(0); // sanity: the fixture actually moves under the default task
      expect(noMovementTotal).toBe(0); // movementScorePerUnit: 0 zeroes this channel entirely
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });

  it('rejects an unknown arena task at the IPC boundary', () => {
    const root = mkdtempSync(join(tmpdir(), 'null-worker-arena-task-bad-'));
    try {
      const { path, expectedSha256 } = writeFixtureGzip(root);
      const task: NullWorkerTask = {
        graphId: 'trace-graph',
        mode: 'biological',
        path,
        expectedSha256,
        heldOutSeeds: [1],
        ticks: 10,
        arenaTask: 'not-a-real-task'
      };
      expect(() => runTask(task)).toThrow(/unknown arena task/);
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  });
});
