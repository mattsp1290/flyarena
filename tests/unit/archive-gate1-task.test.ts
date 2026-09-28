import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { sha256Hex } from '../../scripts/training/fsio';
import { arenaTaskFingerprintOf, type ArchivedReadout, type TrainedReadoutArchive } from '../../scripts/attribution/archive-readouts';

/**
 * `.agents/plans/readout-attribution/01-archive-and-types.md`'s WP1b Gate 1
 * (sibling to `archive-gate1.test.ts`'s WP1 coverage, split out rather than
 * appended -- WP1b's source data is four separate per-task raw-scores files,
 * not one raw-scores file, so its own `describe` block reads a different
 * fixture set): a permanent, committed-data-only regression test that the
 * archived per-task ("task-intervention") readouts reproduce
 * `flyarena-s8z8`'s own per-task `trained.json` exactly.
 *
 * The actual TS rescore (decoder `trained`, held-out seeds 30001-30100, T
 * 1800, `--arena-task <task>`) was run once by hand when this archive was
 * built, against every one of the 52 archived run directories
 * (`null-trained-evaluate.ts --graph-list`), and its output matched
 * `flyarena-s8z8`'s committed `training/runs/tasks/<task>/trained.json`
 * bit-for-bit -- every `movementScore` array, not merely its mean -- for
 * all 52 runs across all 4 tasks (verified against the exact trained.json
 * shas the bead recorded, pinned below as `EXPECTED_RAW_FILE_SHA256`). That
 * full rescore is too expensive to re-run as a unit test (5,200 episodes);
 * this file is the static half that would have caught a hand-edit or a
 * future regression in `archive-readouts.ts` (or in the committed
 * `task-intervention-trained-raw-*-v1.json` files) immediately -- it reads
 * only files this repository already commits, no worktree, no episode
 * simulation.
 *
 * A thermo-maintainability review found an earlier version of this file
 * pinned 52 hand-computed per-run mean `movementScore` floats
 * (`EXPECTED_MEANS`) instead of the four raw files' own sha256 -- strictly
 * weaker (protects only the mean of one field, not the file's other bytes:
 * `heldOutSeeds`, `gzipSha256`, `armBundleSha256`, `ticks`, ...) and noisier
 * (52 ~15-significant-digit constants vs. 4 hex strings that were already
 * sitting in this doc comment as the verified-rescore provenance). Pinning
 * each raw file's own sha256 -- exactly what `copyVerbatim`'s whole purpose
 * already is elsewhere in this archive -- catches corruption of ANY byte in
 * the file, is smaller, and needs no new information: `EXPECTED_MEANS` is
 * gone.
 */

const repoRoot = resolve(__dirname, '../..');
const readJson = <T>(relPath: string): T => JSON.parse(readFileSync(resolve(repoRoot, relPath), 'utf8')) as T;
const readBytes = (relPath: string): Buffer => readFileSync(resolve(repoRoot, relPath));

const archive = readJson<TrainedReadoutArchive>('training/archive/trained-readouts-v1.json');
const taskIntervention = archive.readouts.filter((r): r is ArchivedReadout => r.kind === 'task-intervention');

const TASKS = ['hazard-heavy', 'sparse-food', 'no-movement', 'crowded'] as const;

interface RawTaskRun {
  readonly id: string;
  readonly trainerSeed: number;
  readonly gzipSha256: string;
  readonly armBundleSha256: string;
  readonly heldOutSeeds: readonly number[];
  readonly movementScore: readonly number[];
}
interface RawTaskFile {
  readonly arenaTask: string;
  readonly arenaTaskFingerprint: string;
  readonly graphListSha256: string;
  readonly ticks: number;
  readonly seeds: { readonly start: number; readonly count: number };
  readonly runs: readonly RawTaskRun[];
}
const rawByTask = new Map<string, RawTaskFile>(
  TASKS.map((task) => [task, readJson<RawTaskFile>(`training/archive/task-intervention-trained-raw-${task}-v1.json`)])
);

const goldenTasks = readJson<Record<string, string>>('tests/fixtures/golden/tasks.json');

const interventionEntries = archive.readouts.filter((r) => r.kind === 'intervention');

/**
 * The verified rescore's own provenance (this file's doc comment): each
 * task's raw-scores file's sha256, exactly as `flyarena-s8z8`'s bead
 * recorded it. A thermo-maintainability-review suggestion, replacing a
 * previous 52-entry hand-computed mean table -- see the doc comment above.
 */
const EXPECTED_RAW_FILE_SHA256: Readonly<Record<(typeof TASKS)[number], string>> = {
  'hazard-heavy': '10006b64a8dd3f7a25f3bdebb023d1d1c4a99c804a47aa1d3a6b9761f8b8832d',
  'sparse-food': '6fbae4e239ce4ef0afce49330faf51244706d77b6b8ca7ede4f76aa58377cbb9',
  'no-movement': '78bdaf1b212fa6f044a4447eb34302989d87d97cd31602e658b928bc61967402',
  crowded: '98c19e06b7678c0096739502d35b50168a114eb23befbbc356a481b4382798d1'
};

describe('WP1b archive vs flyarena-s8z8 per-task trained.json (Gate 1, task-intervention identity/score half)', () => {
  it('has exactly 52 task-intervention entries (13 per task x 4 tasks)', () => {
    expect(taskIntervention).toHaveLength(52);
    for (const task of TASKS) {
      expect(taskIntervention.filter((r) => r.arenaTask === task)).toHaveLength(13);
    }
  });

  it('every task-intervention entry\'s id is unique and distinct from every other archived entry\'s id', () => {
    const ids = archive.readouts.map((r) => r.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(archive.readouts).toHaveLength(75); // 10 bigq + 13 intervention + 52 task-intervention
  });

  it('every task-intervention entry\'s arenaTaskFingerprintOf(entry) matches tests/fixtures/golden/tasks.json', () => {
    expect(taskIntervention.length).toBeGreaterThan(0);
    for (const entry of taskIntervention) {
      const expected = goldenTasks[entry.arenaTask];
      expect(expected, `no golden fixture for arenaTask "${entry.arenaTask}"`).toBeDefined();
      // The sanctioned read path (a thermo-maintainability review finding),
      // not the raw `entry.arenaTaskFingerprint` field directly.
      expect(arenaTaskFingerprintOf(entry)).toBe(expected);
    }
  });

  it('no bigq/default-task-intervention entry carries a STORED arenaTaskFingerprint (only task-intervention entries do), but arenaTaskFingerprintOf still resolves the default task\'s real fingerprint for them', () => {
    for (const entry of archive.readouts.filter((r) => r.kind !== 'task-intervention')) {
      // Testing the storage/optionality mechanic itself here -- the raw
      // field, not the accessor, is the thing under test in this assertion.
      expect(entry.arenaTaskFingerprint).toBeUndefined();
      // But the accessor never returns "any"/"unknown": a default entry
      // resolves to the DEFAULT task's own fingerprint, matching the golden
      // fixture's own "default" entry.
      expect(arenaTaskFingerprintOf(entry)).toBe(goldenTasks.default);
    }
  });

  it('every task-intervention entry\'s theta decodes to bytes matching its own weightsSha256', () => {
    for (const entry of taskIntervention) {
      const decoded = Buffer.from(entry.theta, 'base64');
      expect(sha256Hex(decoded), `${entry.id}'s theta does not hash to its own weightsSha256`).toBe(entry.weightsSha256);
    }
  });

  it('every archived entry\'s weightsSha256 is unique across the whole archive (75 entries, no per-task retrain collided with another run)', () => {
    const byWeights = new Map<string, string>();
    for (const entry of archive.readouts) {
      const priorId = byWeights.get(entry.weightsSha256);
      expect(priorId, `${entry.id} shares weightsSha256 with ${String(priorId)}`).toBeUndefined();
      byWeights.set(entry.weightsSha256, entry.id);
    }
    expect(byWeights.size).toBe(archive.readouts.length);
  });

  it('each task\'s raw-scores file graphListSha256 equals the committed intervention-index-v1.json\'s own sha256 (same intervention graphs as the default task)', () => {
    const indexSha256 = sha256Hex(readBytes('training/archive/intervention-index-v1.json'));
    for (const task of TASKS) {
      const raw = rawByTask.get(task)!;
      expect(raw.graphListSha256, `${task}'s raw scores file`).toBe(indexSha256);
    }
  });

  for (const task of TASKS) {
    it(`every "${task}" task-intervention entry's armBundleSha256 matches its raw scores file's per-id record`, () => {
      const raw = rawByTask.get(task)!;
      const rawByKey = new Map(raw.runs.map((r) => [`${r.id}\u0000${r.trainerSeed}`, r]));
      const entries = taskIntervention.filter((r) => r.arenaTask === task);
      expect(entries).toHaveLength(13);
      for (const entry of entries) {
        const raw = rawByKey.get(`${entry.graphId}\u0000${entry.trainerSeed}`);
        expect(raw, `no raw entry for ${entry.id}`).toBeDefined();
        expect(raw?.armBundleSha256).toBe(entry.armBundleSha256);
      }
    });

    it(`every "${task}" task-intervention entry's graph matches the default-task intervention entry's graph for the same graphId`, () => {
      const entries = taskIntervention.filter((r) => r.arenaTask === task);
      expect(entries).toHaveLength(13);
      for (const entry of entries) {
        const defaultEntry = interventionEntries.find((r) => r.graphId === entry.graphId);
        expect(defaultEntry, `no default-task intervention entry for graphId "${entry.graphId}"`).toBeDefined();
        expect(entry.graphGzipSha256).toBe(defaultEntry?.graphGzipSha256);
        expect(entry.graphBinarySha256).toBe(defaultEntry?.graphBinarySha256);
      }
    });
  }

  it('every task\'s raw-scores file matches its verified-rescore sha256 exactly (protects every byte, not just movementScore\'s mean)', () => {
    for (const task of TASKS) {
      const actual = sha256Hex(readBytes(`training/archive/task-intervention-trained-raw-${task}-v1.json`));
      expect(actual, `${task}'s raw-scores file`).toBe(EXPECTED_RAW_FILE_SHA256[task]);
    }
  });

  it('every archived task-intervention entry has a corresponding run in its task\'s raw scores file with 100 held-out movement scores', () => {
    for (const entry of taskIntervention) {
      const raw = rawByTask.get(entry.arenaTask)!;
      const rawRun = raw.runs.find((r) => r.id === entry.graphId && r.trainerSeed === entry.trainerSeed);
      expect(rawRun, `no raw run for ${entry.id}`).toBeDefined();
      expect(rawRun?.movementScore).toHaveLength(100);
    }
  });

  it('every task\'s raw scores file used held-out seeds 30001-30100 and ticks 1800 (the gate\'s own seed/tick contract)', () => {
    const expectedSeeds = Array.from({ length: 100 }, (_, i) => 30001 + i);
    for (const task of TASKS) {
      const raw = rawByTask.get(task)!;
      expect(raw.ticks).toBe(1800);
      expect(raw.seeds).toEqual({ start: 30001, count: 100 });
      for (const run of raw.runs) {
        expect(run.heldOutSeeds).toEqual(expectedSeeds);
      }
    }
  });
});
