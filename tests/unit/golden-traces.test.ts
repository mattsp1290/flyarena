import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  buildGoldenFiles,
  buildSeedTrace,
  DEFAULT_GRAPH_ID,
  TRACE_SEEDS,
  TRACE_SUBSTEPS,
  TRACE_TICKS
} from '../../scripts/training/export-traces';
import { createTraceGraph } from '../fixtures/trace-graph';

/**
 * Golden-trace regression: `scripts/training/export-traces.ts` is TypeScript
 * behavior frozen into committed JSON so a later PyTorch port can be checked
 * against it (`.agents/plans/trained-readout/02-gpu-port-and-parity.md`). If
 * this test fails, the arena/rate-model behavior changed; regenerate the
 * fixtures deliberately with `npm run training:traces` and review the diff
 * before committing it.
 */

const GOLDEN_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../fixtures/golden');
const GOLDEN_BUDGET_BYTES = 200_000;

const readGoldenText = (fileName: string): string =>
  readFileSync(resolve(GOLDEN_DIR, fileName), 'utf8');

describe('golden trace regeneration', () => {
  it('regenerates every committed golden file byte-for-byte from a fresh build', () => {
    // Comparing serialized *text* (not parsed objects with `toEqual`) is
    // deliberate: `JSON.stringify` writes `-0` as `"0"`, but a parsed `-0`
    // (which `decodeAction`'s clamp can produce, e.g. `Math.max(-1, -0)`)
    // is distinct from `0` under `toEqual`'s `Object.is`-based comparison.
    // Comparing text sidesteps that entirely and is a strictly stronger
    // check: it is exactly "byte-identical to what's committed."
    const graph = createTraceGraph();
    const files = buildGoldenFiles(graph, DEFAULT_GRAPH_ID, TRACE_SUBSTEPS);

    expect(files.length).toBeGreaterThan(0);
    for (const { fileName, value } of files) {
      expect(JSON.stringify(value)).toBe(readGoldenText(fileName));
    }

    // No orphaned committed file that buildGoldenFiles no longer produces.
    const committedNames = readdirSync(GOLDEN_DIR).filter((name) => name.endsWith('.json'));
    expect(committedNames.sort()).toEqual(files.map((f) => f.fileName).sort());
  });

  it('is deterministic: two in-process runs produce identical output for every committed seed', () => {
    const graph = createTraceGraph();
    for (const seed of TRACE_SEEDS) {
      const first = buildSeedTrace(graph, DEFAULT_GRAPH_ID, seed, TRACE_TICKS, TRACE_SUBSTEPS);
      const second = buildSeedTrace(graph, DEFAULT_GRAPH_ID, seed, TRACE_TICKS, TRACE_SUBSTEPS);
      expect(second).toEqual(first);
    }
  });

  it('has a non-zero decoded thrust, yaw, and brake on at least one tick of each committed trace', () => {
    for (const seed of TRACE_SEEDS) {
      const trace = JSON.parse(readGoldenText(`trace-graph-seed-${seed}.json`)) as {
        actions: number[][];
      };
      expect(trace.actions.some((action) => action[0] !== 0)).toBe(true);
      expect(trace.actions.some((action) => action[1] !== 0)).toBe(true);
      expect(trace.actions.some((action) => action[2] !== 0)).toBe(true);
    }
  });

  it('stays inside the committed byte budget', () => {
    const names = readdirSync(GOLDEN_DIR).filter((name) => name.endsWith('.json'));
    const totalBytes = names.reduce(
      (sum, name) => sum + statSync(resolve(GOLDEN_DIR, name)).size,
      0
    );
    expect(
      totalBytes,
      `Committed golden fixtures are ${totalBytes} bytes, over the ${GOLDEN_BUDGET_BYTES}-byte ` +
        'budget. See TRACE_TICKS\'s doc comment in scripts/training/export-traces.ts for how to ' +
        'free budget (e.g. reduce TRACE_TICKS or TRACE_SEEDS) before regenerating and committing.'
    ).toBeLessThanOrEqual(GOLDEN_BUDGET_BYTES);
  });
});
