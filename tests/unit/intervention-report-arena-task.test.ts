// @vitest-environment node
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  DEFAULT_OUT,
  runInterventionReport,
  type InterventionReportArgs
} from '../../scripts/null/intervention-report';
import { guardSelectionScratchOut } from '../../scripts/null/intervention-report-run-mode';
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

  /**
   * `writeFixtureFiles`'s single-entry-per-arm fixture is fine for label/
   * fingerprint plumbing tests, but a `controlCount: 1` arm is *always*
   * IQR-0/degenerate by construction -- useless for exercising the
   * degenerate guard's non-degenerate path. This variant takes explicit
   * per-arm score arrays (all the same length = `controlCount`) so a test
   * can construct a clean arm, a near-point-mass arm (like the real
   * `no-movement` C arm), or anything in between.
   */
  const writeFixtureFilesWithArms = (options: {
    readonly biologicalScore: number;
    readonly publishedBiologicalScore: number;
    readonly pScore: number;
    readonly qScore: number;
    readonly cScores: readonly number[];
    readonly mScores: readonly number[];
    readonly mqScores: readonly number[];
  }): void => {
    const { biologicalScore, publishedBiologicalScore, pScore, qScore, cScores, mScores, mqScores } = options;
    const controlCount = cScores.length;
    if (mScores.length !== controlCount || mqScores.length !== controlCount) {
      throw new Error('test fixture: cScores/mScores/mqScores must all have the same length (controlCount)');
    }
    const graphs = [
      { id: 'P', kind: 'P', score: pScore },
      { id: 'Q', kind: 'Q', score: qScore },
      ...cScores.map((score, i) => ({ id: `C${String(i).padStart(3, '0')}`, kind: 'C', score })),
      ...mScores.map((score, i) => ({ id: `M${1000 + i}`, kind: 'M', score })),
      ...mqScores.map((score, i) => ({ id: `MQ${2000 + i}`, kind: 'MQ', score }))
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
      controlCount,
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

    it('carries both statsOnly:true and diagnosticOnly:true when --stats-only and --allow-reproduction-mismatch are combined', () => {
      writeFixtureFilesWithArms({
        biologicalScore: 0,
        publishedBiologicalScore: 999, // deliberate mismatch
        pScore: 10,
        qScore: 8,
        cScores: [1, 2, 3, 4, 5],
        mScores: [2, 3, 4, 5, 6],
        mqScores: [1, 2, 3, 4, 5]
      });
      patchAuthoredArenaTask('hazard-heavy');
      patchPublishedNullArenaTask('hazard-heavy');
      const { statistics } = runInterventionReport(
        argsFor({ arenaTask: 'hazard-heavy', statsOnly: true, allowReproductionMismatch: true })
      );
      expect(statistics.statsOnly).toBe(true);
      expect(statistics.diagnosticOnly).toBe(true);
    });

    // `.agents/plans/selection-robustness/02-per-selection-chain.md` WP2:
    // a per-selection chain's own null is itself a default-task run (no
    // recorded `arenaTask`, same as the shipped default null) -- `--null`
    // must not be refused just because it carries no stamp when the
    // *requested* task is explicitly `'default'`.
    it('accepts an unstamped (default-task) --null when --arena-task default is explicitly requested', () => {
      writeFixtureFiles(0, 0); // authored.json and null.json both default-task, no arenaTask field on either
      const { statistics } = runInterventionReport(argsFor({ arenaTask: 'default', statsOnly: true }));
      expect(statistics.statsOnly).toBe(true);
      expect(statistics.arenaTask).toEqual({ id: 'default', fingerprint: resolveArenaTask('default').fingerprint });
    });

    it('still refuses --stats-only --arena-task default when --null is stamped under a non-default task', () => {
      writeFixtureFiles(0, 0);
      patchPublishedNullArenaTask('hazard-heavy');
      expect(() => runInterventionReport(argsFor({ arenaTask: 'default', statsOnly: true }))).toThrow(
        /requires --null to be a per-task null.*is recorded under arena task "hazard-heavy"/
      );
    });
  });

  describe('guardSelectionScratchOut (selection-robustness WP2)', () => {
    // Every fixture in this describe block uses SOURCE_SHA ('x'.repeat(64)),
    // which never matches the real shipped manifest's binarySha256 -- so
    // every path here is the "non-shipped graph" case. Only refusal is
    // asserted (never a successful write into the real public/data or docs
    // trees), matching `null-report.test.ts`'s own discipline for its
    // sibling guard.
    it('refuses an explicit --out under the real public/data tree for a non-shipped-graph run', () => {
      writeFixtureFiles(0, 0);
      // dirname(DEFAULT_OUT) is <repoRoot>/training/runs/interventions -- three
      // levels up is <repoRoot>, matching intervention-report-run-mode.ts's own
      // `repoRoot` derivation.
      const repoRoot = join(dirname(DEFAULT_OUT), '..', '..', '..');
      const otherPublicPath = join(repoRoot, 'public', 'data', 'intervention-stats-selection-larger.json');
      const args = argsFor({ out: otherPublicPath });
      expect(() => runInterventionReport(args)).toThrow(/resolves under/);
      expect(existsSync(otherPublicPath)).toBe(false);
    });

    it('does not refuse a scratch --out outside public/ and docs/ (the normal selection-scratch case)', () => {
      writeFixtureFiles(0, 0);
      expect(() => runInterventionReport(argsFor())).not.toThrow();
    });

    // A dual-review gap: every test above only exercises refusal against
    // the real repo tree. `guardSelectionScratchOut` takes injectable
    // `publicDataDir`/`docsDir`, so the allow-path and the sibling-prefix
    // boundary can be checked directly, without a real repo write.
    it('guardSelectionScratchOut: allows public/ when sha matches shipped; never blocks a sibling prefix', () => {
      const root = mkdtempSync(join(tmpdir(), 'guard-selection-scratch-out-'));
      try {
        const pub = join(root, 'public', 'data');
        mkdirSync(pub, { recursive: true });
        writeFileSync(join(pub, 'malecns-arena-v1.manifest.json'), JSON.stringify({ binarySha256: 'a'.repeat(64) }));
        const docs = join(root, 'docs');
        mkdirSync(docs, { recursive: true });

        expect(() => guardSelectionScratchOut(join(pub, 'x.json'), 'a'.repeat(64), pub, docs)).not.toThrow();
        expect(() => guardSelectionScratchOut(join(pub, 'x.json'), 'b'.repeat(64), pub, docs)).toThrow(/resolves under/);
        expect(() => guardSelectionScratchOut(join(root, 'public-old', 'x.json'), 'b'.repeat(64), pub, docs)).not.toThrow();
        expect(() => guardSelectionScratchOut(join(root, 'docs2', 'x.md'), 'b'.repeat(64), pub, docs)).not.toThrow();
        expect(() => guardSelectionScratchOut(pub, 'b'.repeat(64), pub, docs)).toThrow(/resolves under/);
      } finally {
        rmSync(root, { recursive: true, force: true });
      }
    });
  });

  describe('degenerate guard (00-overview.md, applied only in --stats-only mode)', () => {
    it('overrides p.category to "degenerate" when the C arm is IQR-0, without affecting channelSpecific (MQ/null are clean)', () => {
      writeFixtureFilesWithArms({
        biologicalScore: 0,
        publishedBiologicalScore: 0,
        pScore: 10,
        qScore: 8,
        cScores: [1, 1, 1, 1, 1], // IQR 0 -- the real no-movement shape, simplified
        mScores: [2, 3, 4, 5, 6],
        mqScores: [1, 2, 3, 4, 5]
      });
      patchAuthoredArenaTask('hazard-heavy');
      patchPublishedNullArenaTask('hazard-heavy');
      const { statistics } = runInterventionReport(argsFor({ arenaTask: 'hazard-heavy', statsOnly: true }));
      expect(statistics.armDegeneracy?.cArm.degenerate).toBe(true);
      expect(statistics.armDegeneracy?.categoryDegenerate).toBe(true);
      expect(statistics.p.category).toBe('degenerate');
      // channelSpecific is unaffected: null and MQ are both clean here.
      expect(statistics.armDegeneracy?.channelSpecificDegenerate).toBe(false);
      expect(statistics.q.channelSpecific).toBe(true);
    });

    it('overrides q.channelSpecific to "degenerate" when the MQ arm is IQR-0, without affecting p.category (C/M/null are clean)', () => {
      writeFixtureFilesWithArms({
        biologicalScore: 0,
        publishedBiologicalScore: 0,
        pScore: 10,
        qScore: 8,
        cScores: [1, 2, 3, 4, 5],
        mScores: [2, 3, 4, 5, 6],
        mqScores: [1, 1, 1, 1, 1] // IQR 0
      });
      patchAuthoredArenaTask('crowded');
      patchPublishedNullArenaTask('crowded');
      const { statistics } = runInterventionReport(argsFor({ arenaTask: 'crowded', statsOnly: true }));
      expect(statistics.armDegeneracy?.mqArm.degenerate).toBe(true);
      expect(statistics.armDegeneracy?.channelSpecificDegenerate).toBe(true);
      expect(statistics.q.channelSpecific).toBe('degenerate');
      // p.category is unaffected: C/M/null are all clean here.
      expect(statistics.armDegeneracy?.categoryDegenerate).toBe(false);
      expect(statistics.p.category).toBe('pathway-supported');
    });

    it('reports armDegeneracy with every flag false, and normal (non-string) category/channelSpecific, when every arm is clean', () => {
      writeFixtureFilesWithArms({
        biologicalScore: 0,
        publishedBiologicalScore: 0,
        pScore: 10,
        qScore: 8,
        cScores: [1, 2, 3, 4, 5],
        mScores: [2, 3, 4, 5, 6],
        mqScores: [1, 2, 3, 4, 5]
      });
      patchAuthoredArenaTask('sparse-food');
      patchPublishedNullArenaTask('sparse-food');
      const { statistics } = runInterventionReport(argsFor({ arenaTask: 'sparse-food', statsOnly: true }));
      expect(statistics.armDegeneracy).toBeDefined();
      expect(statistics.armDegeneracy?.categoryDegenerate).toBe(false);
      expect(statistics.armDegeneracy?.channelSpecificDegenerate).toBe(false);
      expect(statistics.p.category).toBe('pathway-supported');
      expect(statistics.q.channelSpecific).toBe(true);
    });

    it('does not add an armDegeneracy key when --stats-only is omitted (byte-identity gate)', () => {
      writeFixtureFilesWithArms({
        biologicalScore: 0,
        publishedBiologicalScore: 0,
        pScore: 10,
        qScore: 8,
        cScores: [1, 1, 1, 1, 1], // degenerate, but --stats-only is not set
        mScores: [2, 3, 4, 5, 6],
        mqScores: [1, 2, 3, 4, 5]
      });
      const { statistics } = runInterventionReport(argsFor());
      expect(statistics.armDegeneracy).toBeUndefined();
      expect(statistics.p.category).not.toBe('degenerate');
      expect(JSON.stringify(statistics)).not.toContain('armDegeneracy');
    });
  });
});
