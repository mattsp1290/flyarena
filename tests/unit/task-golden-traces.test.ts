import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import { ARENA_TASK_IDS, resolveArenaTask } from '../../src/lib/arena/tasks';
import { buildTaskGoldenFiles, DEFAULT_GRAPH_ID, TRACE_SUBSTEPS } from '../../scripts/training/export-traces';
import { createTraceGraph } from '../fixtures/trace-graph';
import { diffCloseEnough, GOLDEN_GENERATING_ARCH, MAX_INEXACT_LEAVES } from '../fixtures/cross-arch-tolerance';

/**
 * `.agents/plans/task-generality/01-task-plumbing.md`'s WP1: per-task golden
 * traces (`tests/fixtures/golden/tasks/<id>/`), one per non-`default` arena
 * task, regenerated in-process and compared against the committed bytes —
 * mirrors `golden-traces.test.ts`'s own regeneration pattern for the default
 * export, including its cross-architecture handling (see that file's module
 * doc comment): byte-for-byte on `GOLDEN_GENERATING_ARCH`, tolerance- plus
 * leaf-budget-based everywhere else. Never byte-compares arm64-only numbers
 * on another architecture's CI run.
 */

const GOLDEN_TASKS_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../fixtures/golden/tasks');
const NON_DEFAULT_TASK_IDS = ARENA_TASK_IDS.filter((id) => id !== 'default');

const readGoldenText = (id: string, fileName: string): string => readFileSync(resolve(GOLDEN_TASKS_DIR, id, fileName), 'utf8');

describe('per-task golden trace regeneration', () => {
  it.each(NON_DEFAULT_TASK_IDS)('regenerates %s from a fresh build', (id) => {
    const graph = createTraceGraph();
    const { config } = resolveArenaTask(id);
    const files = buildTaskGoldenFiles(graph, DEFAULT_GRAPH_ID, TRACE_SUBSTEPS, config);
    expect(files.length).toBe(2);

    if (process.arch === GOLDEN_GENERATING_ARCH) {
      for (const { fileName, value } of files) {
        expect(JSON.stringify(value)).toBe(readGoldenText(id, fileName));
      }
    } else {
      const allMismatches: string[] = [];
      let overBudget = false;
      for (const { fileName, value } of files) {
        const expected = JSON.parse(readGoldenText(id, fileName));
        const actual = JSON.parse(JSON.stringify(value));
        const { mismatches, inexactLeaves } = diffCloseEnough(expected, actual, `${id}/${fileName}`);
        allMismatches.push(...mismatches);
        if (inexactLeaves > MAX_INEXACT_LEAVES) overBudget = true;
      }
      expect(allMismatches, `${id}: golden fixture differs from a fresh build beyond cross-arch tolerance`).toEqual([]);
      expect(overBudget, `${id}: too many above-noise-floor inexact leaves (budget ${MAX_INEXACT_LEAVES})`).toBe(false);
    }
  });

  it('every committed per-task trace records that task\'s own configFingerprint', () => {
    for (const id of NON_DEFAULT_TASK_IDS) {
      const trace = JSON.parse(readGoldenText(id, `${DEFAULT_GRAPH_ID}-seed-1.json`)) as { configFingerprint: string };
      expect(trace.configFingerprint).toBe(resolveArenaTask(id).fingerprint);
    }
  });
});
