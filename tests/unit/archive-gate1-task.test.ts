import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { sha256Hex } from '../../scripts/training/fsio';
import type { ArchivedReadout, TrainedReadoutArchive } from '../../scripts/attribution/archive-readouts';

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
 * shas the bead recorded: hazard-heavy 10006b64…, sparse-food 6fbae4e2…,
 * no-movement 78bdaf1b…, crowded 98c19e06…). That full rescore is too
 * expensive to re-run as a unit test (5,200 episodes); this file is the
 * static half that would have caught a hand-edit or a future regression in
 * `archive-readouts.ts` (or in the committed `task-intervention-trained-raw-
 * *-v1.json` files) immediately -- it reads only files this repository
 * already commits, no worktree, no episode simulation. `EXPECTED_MEANS`
 * below are the exact per-run means from that verified rescore (computed
 * here from the committed raw `movementScore` arrays via plain arithmetic,
 * not re-simulated), so a future edit that silently swapped a raw-scores
 * file or an archive entry would fail this test even if every other
 * cross-reference happened to still line up.
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

const mean = (values: readonly number[]): number => values.reduce((sum, v) => sum + v, 0) / values.length;

/**
 * The exact per-run mean movement score from the verified rescore (this
 * file's own doc comment) -- `<graphId>-seed<trainerSeed>-<arenaTask>`,
 * matching `ArchivedReadout.id`'s own format.
 */
const EXPECTED_MEANS: ReadonlyMap<string, number> = new Map([
  ['C000-seed101-crowded', 95.19914927275973],
  ['C000-seed101-hazard-heavy', 26.0725327620016],
  ['C000-seed101-no-movement', 51.7],
  ['C000-seed101-sparse-food', 44.466663647620756],
  ['C001-seed101-crowded', 92.42038888192845],
  ['C001-seed101-hazard-heavy', 52.30779569585795],
  ['C001-seed101-no-movement', 55.3],
  ['C001-seed101-sparse-food', 43.928357644156286],
  ['C002-seed101-crowded', 112.02450993084136],
  ['C002-seed101-hazard-heavy', 35.467567745665264],
  ['C002-seed101-no-movement', 58.42],
  ['C002-seed101-sparse-food', 44.12507647495929],
  ['C003-seed101-crowded', 111.67692900726094],
  ['C003-seed101-hazard-heavy', 50.22682831796527],
  ['C003-seed101-no-movement', 54.6],
  ['C003-seed101-sparse-food', 43.249659030448385],
  ['C004-seed101-crowded', 109.35731776818056],
  ['C004-seed101-hazard-heavy', 55.81700065017724],
  ['C004-seed101-no-movement', 55.66],
  ['C004-seed101-sparse-food', 41.81708264925878],
  ['M1000-seed101-crowded', 95.15836881361496],
  ['M1000-seed101-hazard-heavy', 47.55037668662937],
  ['M1000-seed101-no-movement', 48.24],
  ['M1000-seed101-sparse-food', 43.67897797046632],
  ['M1001-seed101-crowded', 93.33902111733344],
  ['M1001-seed101-hazard-heavy', 49.94973320887546],
  ['M1001-seed101-no-movement', 29.3],
  ['M1001-seed101-sparse-food', 45.55990603102984],
  ['M1002-seed101-crowded', 114.1691926748693],
  ['M1002-seed101-hazard-heavy', 49.68896438477141],
  ['M1002-seed101-no-movement', 56.04],
  ['M1002-seed101-sparse-food', 44.003700492192976],
  ['M1003-seed101-crowded', 108.85213170719247],
  ['M1003-seed101-hazard-heavy', 49.076932496726215],
  ['M1003-seed101-no-movement', 59.62],
  ['M1003-seed101-sparse-food', 45.074270465629425],
  ['M1004-seed101-crowded', 86.58248588702449],
  ['M1004-seed101-hazard-heavy', 37.410368494934474],
  ['M1004-seed101-no-movement', 69.14],
  ['M1004-seed101-sparse-food', 43.9359035213899],
  ['P-seed101-crowded', 92.7795677997876],
  ['P-seed101-hazard-heavy', 37.918642167784135],
  ['P-seed101-no-movement', 56.26],
  ['P-seed101-sparse-food', 39.65600048306926],
  ['P-seed202-crowded', 93.56504615995198],
  ['P-seed202-hazard-heavy', 59.087402450365644],
  ['P-seed202-no-movement', 56.2],
  ['P-seed202-sparse-food', 39.47761767478665],
  ['P-seed303-crowded', 114.79844118015906],
  ['P-seed303-hazard-heavy', 51.462333674606654],
  ['P-seed303-no-movement', 54.68],
  ['P-seed303-sparse-food', 41.15037294764599]
]);

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

  it('every task-intervention entry\'s arenaTaskFingerprint matches tests/fixtures/golden/tasks.json', () => {
    expect(taskIntervention.length).toBeGreaterThan(0);
    for (const entry of taskIntervention) {
      const expected = goldenTasks[entry.arenaTask];
      expect(expected, `no golden fixture for arenaTask "${entry.arenaTask}"`).toBeDefined();
      expect(entry.arenaTaskFingerprint).toBe(expected);
    }
  });

  it('no bigq/default-task-intervention entry carries an arenaTaskFingerprint (only task-intervention entries do)', () => {
    for (const entry of archive.readouts.filter((r) => r.kind !== 'task-intervention')) {
      expect(entry.arenaTaskFingerprint).toBeUndefined();
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

  it('every archived task-intervention entry\'s per-run mean (from its task\'s raw scores file) matches the verified rescore exactly', () => {
    expect(EXPECTED_MEANS.size).toBe(52);
    for (const entry of taskIntervention) {
      const raw = rawByTask.get(entry.arenaTask)!;
      const rawRun = raw.runs.find((r) => r.id === entry.graphId && r.trainerSeed === entry.trainerSeed);
      expect(rawRun, `no raw run for ${entry.id}`).toBeDefined();
      expect(rawRun?.movementScore).toHaveLength(100);
      const expected = EXPECTED_MEANS.get(entry.id);
      expect(expected, `no EXPECTED_MEANS entry for ${entry.id}`).toBeDefined();
      expect(mean(rawRun!.movementScore)).toBe(expected);
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
