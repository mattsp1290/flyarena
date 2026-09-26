import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { ARENA_CONFIG, retainArenaConfig } from '../../src/lib/arena/config';
import { ARENA_TASK_IDS, ARENA_TASKS, resolveArenaTask } from '../../src/lib/arena/tasks';

/**
 * Coverage for `src/lib/arena/tasks.ts`
 * (`.agents/plans/task-generality/01-task-plumbing.md`'s WP1). Every task
 * must pass `retainArenaConfig` (already proven at module load — this test
 * re-proves it explicitly so a future change that bypasses that validation
 * fails loudly here) and its fingerprint must equal the committed
 * `tests/fixtures/golden/tasks.json` (the cross-language source of truth
 * `training/tests/test_tasks.py` reads independently).
 */

const TASKS_JSON_PATH = resolve(dirname(fileURLToPath(import.meta.url)), '../fixtures/golden/tasks.json');
const committedFingerprints = JSON.parse(readFileSync(TASKS_JSON_PATH, 'utf8')) as Record<string, string>;

describe('ARENA_TASKS', () => {
  it('default is ARENA_CONFIG itself, unchanged', () => {
    expect(ARENA_TASKS.default).toBe(ARENA_CONFIG);
  });

  it.each(ARENA_TASK_IDS)('%s passes retainArenaConfig', (id) => {
    expect(() => retainArenaConfig(ARENA_TASKS[id])).not.toThrow();
  });

  it.each(ARENA_TASK_IDS)('%s fingerprint matches the committed tasks.json', (id) => {
    expect(resolveArenaTask(id).fingerprint).toBe(committedFingerprints[id]);
  });

  it('tasks.json has no extra or missing ids', () => {
    expect(Object.keys(committedFingerprints).sort()).toEqual([...ARENA_TASK_IDS].sort());
  });

  it('hazard-heavy overrides only hazardCount/hazardPenalty', () => {
    const task = ARENA_TASKS['hazard-heavy'];
    expect(task.hazardCount).toBe(4);
    expect(task.hazardPenalty).toBe(6);
    expect(task.halfWidth).toBe(ARENA_CONFIG.halfWidth);
    expect(task.foodCount).toBe(ARENA_CONFIG.foodCount);
  });

  it('sparse-food overrides only foodCount/halfWidth/halfDepth', () => {
    const task = ARENA_TASKS['sparse-food'];
    expect(task.foodCount).toBe(1);
    expect(task.halfWidth).toBe(18);
    expect(task.halfDepth).toBe(12);
    expect(task.hazardCount).toBe(ARENA_CONFIG.hazardCount);
  });

  it('no-movement overrides only movementScorePerUnit', () => {
    const task = ARENA_TASKS['no-movement'];
    expect(task.movementScorePerUnit).toBe(0);
    expect(task.foodCount).toBe(ARENA_CONFIG.foodCount);
    expect(task.halfWidth).toBe(ARENA_CONFIG.halfWidth);
  });

  it('crowded overrides only halfWidth/halfDepth', () => {
    const task = ARENA_TASKS.crowded;
    expect(task.halfWidth).toBe(8);
    expect(task.halfDepth).toBe(5.5);
    expect(task.foodCount).toBe(ARENA_CONFIG.foodCount);
  });
});

describe('resolveArenaTask', () => {
  it('undefined resolves to default', () => {
    const resolved = resolveArenaTask(undefined);
    expect(resolved.id).toBe('default');
    expect(resolved.config).toBe(ARENA_CONFIG);
  });

  it('resolves every known id', () => {
    for (const id of ARENA_TASK_IDS) {
      expect(resolveArenaTask(id).id).toBe(id);
    }
  });

  it('throws on an unknown id', () => {
    expect(() => resolveArenaTask('not-a-real-task')).toThrow(/unknown arena task id/);
  });
});
