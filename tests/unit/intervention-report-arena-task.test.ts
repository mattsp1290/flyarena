// @vitest-environment node
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  DEFAULT_OUT,
  runInterventionReport,
  type InterventionReportArgs
} from '../../scripts/null/intervention-report';
import type { NullGraphListEvaluationRaw } from '../../scripts/null/null-evaluate';
import { resolveArenaTask } from '../../src/lib/arena/tasks';

/**
 * Extracted from `intervention-report.test.ts` (which was already at the
 * repo's 1000-line review-blocker threshold before this WP's own
 * additions — keeping this coverage in its own file avoids pushing that
 * file further over it, matching the same split
 * `null-worker-arena-task.test.ts` already did for `null-evaluate.test.ts`).
 *
 * Covers `runInterventionReport`'s `--arena-task` (WP1,
 * `.agents/plans/task-generality/01-task-plumbing.md`) and `--stats-only`
 * (WP2, `.agents/plans/task-generality/02-authored-runs.md`) behavior —
 * the `writeFixtureFiles`/`argsFor` setup below is a duplicate of
 * `intervention-report.test.ts`'s own (small enough that duplicating it
 * here is cheaper than sharing it across two files for one describe
 * block's worth of tests).
 */

describe('runInterventionReport: --arena-task / --stats-only (task-generality)', () => {
  const seeds = [30001, 30002, 30003];
  const SOURCE_SHA = 'x'.repeat(64);
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'intervention-report-arena-task-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  const writeFixtureFiles = (biologicalScore: number, publishedBiologicalScore: number) => {
    const graphs = [
      { id: 'P', kind: 'P', score: 10 },
      { id: 'Q', kind: 'Q', score: 8 },
      { id: 'C000', kind: 'C', score: 1 },
      { id: 'M1000', kind: 'M', score: 2 },
      { id: 'MQ2000', kind: 'MQ', score: 1 }
    ];
    const authored: NullGraphListEvaluationRaw = {
      version: 1,
      sourceGraphSha256: SOURCE_SHA,
      seeds: { start: 30001, count: seeds.length },
      ticks: 20,
      substeps: 4,
      decoder: 'authored',
      biological: {
        heldOutSeeds: seeds,
        movementScore: seeds.map(() => biologicalScore),
        foodPickups: [0, 0, 0],
        hazardContacts: [0, 0, 0]
      },
      graphs: graphs.map((g) => ({
        id: g.id,
        gzipSha256: `gz-${g.id}`.padEnd(64, '0'),
        binarySha256: 'z'.repeat(64),
        heldOutSeeds: seeds,
        movementScore: seeds.map(() => g.score),
        foodPickups: [0, 0, 0],
        hazardContacts: [0, 0, 0]
      })),
      host: { arch: 'arm64', node: 'v22.0.0' },
      evaluatorGitRev: null
    };
    writeFileSync(join(root, 'authored.json'), JSON.stringify(authored));

    const index = {
      sourceArtifact: 'src.bin.gz',
      sourceSha256: SOURCE_SHA,
      controlCount: 1,
      entries: graphs.map((g) => ({ id: g.id, kind: g.kind, gzipSha256: `gz-${g.id}`.padEnd(64, '0') }))
    };
    writeFileSync(join(root, 'index.json'), JSON.stringify(index));

    const publishedNull = {
      biological: { score: publishedBiologicalScore },
      rewired: Array.from({ length: 20 }, (_, i) => ({ score: -5 + i * 0.1 })),
      sourceGraphSha256: SOURCE_SHA,
      seeds: { start: 30001, count: seeds.length },
      ticks: 20,
      substeps: 4
    };
    writeFileSync(join(root, 'null.json'), JSON.stringify(publishedNull));
  };

  const argsFor = (overrides: Partial<InterventionReportArgs> = {}): InterventionReportArgs => ({
    authored: join(root, 'authored.json'),
    index: join(root, 'index.json'),
    publishedNull: join(root, 'null.json'),
    out: join(root, 'nested', 'statistics.json'),
    bootstrapSeed: 42,
    bootstrapResamples: 200,
    allowReproductionMismatch: false,
    statsOnly: false,
    ...overrides
  });

  const patchAuthoredArenaTask = (id: string): void => {
    const path = join(root, 'authored.json');
    const authored = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>;
    authored.arenaTask = id;
    authored.arenaTaskFingerprint = resolveArenaTask(id).fingerprint;
    writeFileSync(path, JSON.stringify(authored));
  };

  const patchPublishedNullArenaTask = (id: string): void => {
    const path = join(root, 'null.json');
    const publishedNull = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>;
    publishedNull.arenaTask = { id, fingerprint: resolveArenaTask(id).fingerprint };
    writeFileSync(path, JSON.stringify(publishedNull));
  };

  describe('--arena-task (task-generality WP1)', () => {
    it('labels the output when --arena-task matches authored.json\'s recorded task', () => {
      writeFixtureFiles(0, 0);
      patchAuthoredArenaTask('hazard-heavy');
      const { statistics } = runInterventionReport(argsFor({ arenaTask: 'hazard-heavy' }));
      expect(statistics.arenaTask).toEqual({ id: 'hazard-heavy', fingerprint: resolveArenaTask('hazard-heavy').fingerprint });
    });

    it('throws when --arena-task disagrees with authored.json\'s recorded task', () => {
      writeFixtureFiles(0, 0);
      patchAuthoredArenaTask('hazard-heavy');
      expect(() => runInterventionReport(argsFor({ arenaTask: 'crowded' }))).toThrow(
        /does not match authored\.json's recorded arena task fingerprint/
      );
    });

    it('throws when --arena-task is passed but authored.json was scored under the default task', () => {
      writeFixtureFiles(0, 0);
      expect(() => runInterventionReport(argsFor({ arenaTask: 'hazard-heavy' }))).toThrow(
        /does not match authored\.json's recorded arena task fingerprint/
      );
    });

    it('does not add an arenaTask key when --arena-task is omitted and authored.json is the default task (byte-identity gate)', () => {
      writeFixtureFiles(0, 0);
      const { statistics } = runInterventionReport(argsFor());
      expect(statistics.arenaTask).toBeUndefined();
      expect(JSON.stringify(statistics)).not.toContain('arenaTask');
    });

    it('refuses to write a non-default arena-task result to the default --out', () => {
      writeFixtureFiles(0, 0);
      patchAuthoredArenaTask('hazard-heavy');
      const args = { ...argsFor({ arenaTask: 'hazard-heavy' }), out: DEFAULT_OUT };
      expect(() => runInterventionReport(args)).toThrow(/refusing to write a --arena-task "hazard-heavy" result/);
    });
  });

  describe('--stats-only (task-generality WP2)', () => {
    it('marks the output statsOnly when the flag was passed and --null is scored under the same task', () => {
      writeFixtureFiles(0, 0);
      patchAuthoredArenaTask('hazard-heavy');
      patchPublishedNullArenaTask('hazard-heavy');
      const { statistics } = runInterventionReport(argsFor({ arenaTask: 'hazard-heavy', statsOnly: true }));
      expect(statistics.statsOnly).toBe(true);
    });

    it('does not add a statsOnly key when the flag is omitted (byte-identity gate)', () => {
      writeFixtureFiles(0, 0);
      const { statistics } = runInterventionReport(argsFor());
      expect(statistics.statsOnly).toBeUndefined();
      expect(JSON.stringify(statistics)).not.toContain('statsOnly');
    });

    it('refuses --stats-only when --null has no recorded arena task (the task-independent default null)', () => {
      writeFixtureFiles(0, 0);
      patchAuthoredArenaTask('hazard-heavy');
      expect(() => runInterventionReport(argsFor({ arenaTask: 'hazard-heavy', statsOnly: true }))).toThrow(
        /requires --null to be a per-task null.*no recorded arena task/
      );
    });

    it("refuses --stats-only when --null was scored under a different task than requested", () => {
      writeFixtureFiles(0, 0);
      patchAuthoredArenaTask('hazard-heavy');
      patchPublishedNullArenaTask('crowded');
      expect(() => runInterventionReport(argsFor({ arenaTask: 'hazard-heavy', statsOnly: true }))).toThrow(
        /requires --null to be a per-task null.*is recorded under arena task "crowded"/
      );
    });

    it('does NOT require a task-matched --null for a plain --arena-task run without --stats-only (WP1 behavior unchanged)', () => {
      writeFixtureFiles(0, 0);
      patchAuthoredArenaTask('hazard-heavy');
      // --null has no arenaTask recorded at all here, yet this must still succeed.
      const { statistics } = runInterventionReport(argsFor({ arenaTask: 'hazard-heavy' }));
      expect(statistics.arenaTask).toEqual({ id: 'hazard-heavy', fingerprint: resolveArenaTask('hazard-heavy').fingerprint });
    });
  });
});
