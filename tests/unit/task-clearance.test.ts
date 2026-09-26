// @vitest-environment node
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  buildTaskClearanceReport,
  CLEARANCE_CHANNELS,
  collectClearanceSamples,
  computeChannelPercentiles,
  parseTaskClearanceArgs,
  runTaskClearance,
  type TaskClearanceArgs
} from '../../scripts/null/task-clearance';
import { dirname, resolve as resolvePath } from 'node:path';
import { fileURLToPath } from 'node:url';
import { sha256Hex } from '../../scripts/training/fsio';
import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { createTraceGraph } from '../fixtures/trace-graph';
import { resolveArenaTask } from '../../src/lib/arena/tasks';

const here = dirname(fileURLToPath(import.meta.url));
const REAL_ARTIFACT = resolvePath(here, '../../public/data/malecns-arena-v1.bin.gz');

/**
 * Coverage for `scripts/null/task-clearance.ts`
 * (`.agents/plans/task-generality/02-authored-runs.md`'s WP2 step 3, the
 * clearance measurement `export-traces.ts --arena-task` cannot itself
 * produce — see that script's own doc comment for the plan deviation).
 * Uses `tests/fixtures/trace-graph.ts`'s small deterministic fixture, never
 * the real biological connectome, so this suite stays fast and has no
 * dependency on `public/data/malecns-arena-v1.bin.gz`.
 */

describe('computeChannelPercentiles', () => {
  it('computes p5/p50/p95/max/n over a sorted 20-length sample', () => {
    const values = Array.from({ length: 20 }, (_, i) => i); // 0..19
    const stats = computeChannelPercentiles(values);
    expect(stats.n).toBe(20);
    expect(stats.max).toBe(19);
    expect(stats.p50).toBe(10);
  });

  it('is order-independent (sorts its input)', () => {
    const ascending = computeChannelPercentiles([1, 2, 3, 4, 5]);
    const shuffled = computeChannelPercentiles([5, 1, 4, 2, 3]);
    expect(shuffled).toEqual(ascending);
  });

  it('throws on an empty array', () => {
    expect(() => computeChannelPercentiles([])).toThrow(/at least one value/);
  });

  it('throws on a non-finite sample instead of silently corrupting the sort', () => {
    expect(() => computeChannelPercentiles([0.1, Number.NaN, 0.5])).toThrow(/values\[1\] is not a finite number/);
    expect(() => computeChannelPercentiles([0.1, Number.POSITIVE_INFINITY])).toThrow(/values\[1\] is not a finite number/);
  });

  it('computes fractionSaturated as the fraction of samples exactly at 1', () => {
    expect(computeChannelPercentiles([1, 1, 1, 1]).fractionSaturated).toBe(1);
    expect(computeChannelPercentiles([0, 0.5, 0.9]).fractionSaturated).toBe(0);
    expect(computeChannelPercentiles([0, 1, 1, 1]).fractionSaturated).toBe(0.75);
    // A value merely close to 1 (not clamped to it) must not count as saturated.
    expect(computeChannelPercentiles([0.999999, 1]).fractionSaturated).toBe(0.5);
  });
});

describe('collectClearanceSamples', () => {
  const graph = createTraceGraph();

  it('collects exactly seedCount * ticks samples per channel', () => {
    const samples = collectClearanceSamples(graph, resolveArenaTask('default'), 1, 3, 20, 4);
    for (const channel of CLEARANCE_CHANNELS) {
      expect(samples[channel]).toHaveLength(3 * 20);
    }
  });

  it('produces channel values within the documented 0..1 observation range', () => {
    const samples = collectClearanceSamples(graph, resolveArenaTask('crowded'), 1, 2, 30, 4);
    for (const channel of CLEARANCE_CHANNELS) {
      for (const value of samples[channel]) {
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThanOrEqual(1);
      }
    }
  });

  it('is deterministic given the same seed range', () => {
    const a = collectClearanceSamples(graph, resolveArenaTask('sparse-food'), 1, 2, 15, 4);
    const b = collectClearanceSamples(graph, resolveArenaTask('sparse-food'), 1, 2, 15, 4);
    expect(a).toEqual(b);
  });

  it('actually measures the requested arena task, not the default (regression: a dropped {arenaConfig} would pass every other check here)', () => {
    const defaultSamples = collectClearanceSamples(graph, resolveArenaTask('default'), 1, 3, 30, 4);
    const crowdedSamples = collectClearanceSamples(graph, resolveArenaTask('crowded'), 1, 3, 30, 4);
    // crowded shrinks halfWidth/halfDepth substantially (12x8 -> 8x5.5), so
    // the pooled clearance samples must differ from the default arena's.
    expect(crowdedSamples.forwardClearance).not.toEqual(defaultSamples.forwardClearance);
  });

  it('throws when a trace records a different config fingerprint than requested (defense in depth against a dropped arenaConfig)', () => {
    const wrong = { config: resolveArenaTask('default').config, fingerprint: 'not-a-real-fingerprint' };
    expect(() => collectClearanceSamples(graph, wrong, 1, 1, 5, 4)).toThrow(
      /trace was recorded under a different arena config than requested/
    );
  });
});

describe('buildTaskClearanceReport', () => {
  const graph = createTraceGraph();
  const graphSha256 = 'f'.repeat(64);

  it('records the resolved arena task id/fingerprint and every clearance channel', () => {
    const report = buildTaskClearanceReport(graph, '/fixture/graph.bin.gz', graphSha256, 'hazard-heavy', 1, 4, 25, 4);
    expect(report.version).toBe(1);
    expect(report.arenaTask).toEqual({ id: 'hazard-heavy', fingerprint: resolveArenaTask('hazard-heavy').fingerprint });
    expect(report.graph).toEqual({ path: '/fixture/graph.bin.gz', sha256: graphSha256 });
    expect(report.seeds).toEqual({ start: 1, count: 4 });
    expect(report.ticks).toBe(25);
    expect(report.substeps).toBe(4);
    expect(report.sensorRange).toBe(resolveArenaTask('hazard-heavy').config.sensorRange);
    for (const channel of CLEARANCE_CHANNELS) {
      expect(report.channels[channel].n).toBe(4 * 25);
    }
  });

  it('defaults to the default task when arenaTask is omitted', () => {
    const report = buildTaskClearanceReport(graph, '/fixture/graph.bin.gz', graphSha256, undefined, 1, 2, 10, 4);
    expect(report.arenaTask.id).toBe('default');
  });

  it('is byte-identical across two runs against the same inputs (JSON.stringify)', () => {
    const first = buildTaskClearanceReport(graph, '/fixture/graph.bin.gz', graphSha256, 'no-movement', 1, 3, 20, 4);
    const second = buildTaskClearanceReport(graph, '/fixture/graph.bin.gz', graphSha256, 'no-movement', 1, 3, 20, 4);
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  });

  it('rejects an unknown arena task id', () => {
    expect(() => buildTaskClearanceReport(graph, '/fixture/graph.bin.gz', graphSha256, 'bogus-task', 1, 2, 10, 4)).toThrow(
      /unknown arena task id/
    );
  });
});

describe('parseTaskClearanceArgs', () => {
  it('applies defaults (seed-start 1, seed-count 10, ticks 1800)', () => {
    const args = parseTaskClearanceArgs([]);
    expect(args.seedStart).toBe(1);
    expect(args.seedCount).toBe(10);
    expect(args.ticks).toBe(1800);
    expect(args.arenaTask).toBeUndefined();
  });

  it('derives training/runs/tasks/<id>/clearance.json when --out is omitted', () => {
    const args = parseTaskClearanceArgs(['--arena-task', 'crowded']);
    expect(args.out.replaceAll('\\', '/')).toMatch(/training\/runs\/tasks\/crowded\/clearance\.json$/);
  });

  it('derives the default task path when --arena-task is omitted', () => {
    const args = parseTaskClearanceArgs([]);
    expect(args.out.replaceAll('\\', '/')).toMatch(/training\/runs\/tasks\/default\/clearance\.json$/);
  });

  it('does not override an explicit --out', () => {
    const args = parseTaskClearanceArgs(['--arena-task', 'crowded', '--out', 'custom.json']);
    expect(args.out).toContain('custom.json');
  });

  it('rejects an unknown arena task id', () => {
    expect(() => parseTaskClearanceArgs(['--arena-task', 'bogus'])).toThrow(/unknown arena task id/);
  });

  it('rejects an unknown flag', () => {
    expect(() => parseTaskClearanceArgs(['--bogus'])).toThrow(/Unknown argument/);
  });
});

describe('runTaskClearance (file I/O)', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'task-clearance-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  const writeFixtureGraph = (): string => {
    const graphPath = join(root, 'graph.bin.gz');
    const binary = Buffer.from(encodeGraphBinary(createTraceGraph()));
    writeFileSync(graphPath, gzipSync(binary));
    return graphPath;
  };

  it('writes the report to --out, with graph.sha256 matching the actual file bytes', () => {
    const graphPath = writeFixtureGraph();
    const out = join(root, 'clearance.json');
    const args: TaskClearanceArgs = {
      graph: graphPath,
      arenaTask: 'crowded',
      seedStart: 1,
      seedCount: 2,
      ticks: 10,
      substeps: 4,
      out
    };
    const { report } = runTaskClearance(args);
    expect(existsSync(out)).toBe(true);
    const writtenBytes = readFileSync(out);
    expect(JSON.parse(writtenBytes.toString('utf8'))).toEqual(report);
    expect(report.graph.sha256).toBe(sha256Hex(readFileSync(graphPath)));
  });

  it('records the graph path relative to the repo root when the graph lives inside it (reproducible across checkouts)', () => {
    const out = join(root, 'clearance.json');
    const { report } = runTaskClearance({
      graph: REAL_ARTIFACT,
      arenaTask: 'crowded',
      seedStart: 1,
      seedCount: 1,
      ticks: 3,
      substeps: 4,
      out
    });
    expect(report.graph.path).toBe('public/data/malecns-arena-v1.bin.gz');
  });

  it('keeps an out-of-repo graph path absolute', () => {
    const graphPath = writeFixtureGraph(); // under the tmpdir root, outside the repo
    const out = join(root, 'clearance.json');
    const { report } = runTaskClearance({
      graph: graphPath,
      arenaTask: 'crowded',
      seedStart: 1,
      seedCount: 1,
      ticks: 5,
      substeps: 4,
      out
    });
    expect(report.graph.path).toBe(graphPath);
  });

  it('creates the output directory if it does not exist (mkdir before write)', () => {
    const graphPath = writeFixtureGraph();
    const out = join(root, 'nested', 'deeper', 'clearance.json');
    runTaskClearance({
      graph: graphPath,
      arenaTask: 'crowded',
      seedStart: 1,
      seedCount: 1,
      ticks: 5,
      substeps: 4,
      out
    });
    expect(existsSync(out)).toBe(true);
  });

  it('produces byte-identical output across two runs against the same inputs (the 02-authored-runs.md acceptance rule)', () => {
    const graphPath = writeFixtureGraph();
    const args: TaskClearanceArgs = {
      graph: graphPath,
      arenaTask: 'no-movement',
      seedStart: 1,
      seedCount: 2,
      ticks: 10,
      substeps: 4,
      out: join(root, 'run1.json')
    };
    runTaskClearance(args);
    runTaskClearance({ ...args, out: join(root, 'run2.json') });
    expect(readFileSync(join(root, 'run1.json'), 'utf8')).toBe(readFileSync(join(root, 'run2.json'), 'utf8'));
  });
});
