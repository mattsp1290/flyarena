import { execFileSync } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

import { describe, expect, it } from 'vitest';

import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { sha256Hex } from '../../scripts/training/fsio';
import { createTinyGraph } from '../fixtures/tiny-graph';

/**
 * `.agents/plans/graph-lab/01-service-and-container.md`'s `bundle.mjs` row:
 * "A test runs the built .mjs bundles from an unrelated cwd, with the data
 * in a temp directory, and checks that they produce the fixture results."
 * Requires `npm run graph-lab:bundle` to have already produced
 * `backend/graph_lab/js/*.mjs` (gitignored, not committed) -- run it first
 * if this suite reports the bundle missing.
 *
 * Runs each entry as a real, separate `node` child process (not an
 * in-process `import()`), from a cwd unrelated to both the repo and the
 * fixture data, proving the entries never rely on a cwd-relative default
 * (`PUBLIC_DATA_DIR`/`'public/data'`-shaped assumptions this WP's plan
 * explicitly calls out as the bug class to avoid) -- every path they touch
 * comes from the args file this test writes into its own temp directory.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');
const bundleDir = resolve(repoRoot, 'backend/graph_lab/js');

const unrelatedCwd = mkdtempSync(join(tmpdir(), 'graph-lab-cwd-'));

const parseLines = (output: string): { type: string; result?: unknown; message?: string }[] =>
  output
    .trim()
    .split('\n')
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as { type: string; result?: unknown; message?: string });

const runEntry = (name: string, argsPath: string): unknown => {
  let output: string;
  try {
    output = execFileSync('node', [resolve(bundleDir, name), argsPath], {
      cwd: unrelatedCwd,
      env: { PATH: process.env.PATH ?? '', HOME: unrelatedCwd },
      encoding: 'utf8'
    });
  } catch (error) {
    // `entry-*.ts`'s own catch-all prints its structured `{"type":"error",
    // ...}` line to stdout and then `process.exit(1)` -- `execFileSync`
    // throws on that nonzero exit before this function's own parsing runs,
    // but still attaches the child's stdout to the thrown error.
    const stdout = typeof (error as { stdout?: unknown }).stdout === 'string' ? (error as { stdout: string }).stdout : '';
    const errorLine = parseLines(stdout).find((line) => line.type === 'error');
    if (errorLine) throw new Error(`entry ${name} reported an error: ${errorLine.message}`);
    throw error;
  }
  const lines = parseLines(output);
  const resultLine = lines.find((line) => line.type === 'result');
  const errorLine = lines.find((line) => line.type === 'error');
  if (errorLine) throw new Error(`entry ${name} reported an error: ${errorLine.message}`);
  if (!resultLine) throw new Error(`entry ${name} produced no result line (raw output: ${output})`);
  return resultLine.result;
};

const writeFixtureGraph = (dataDir: string): { path: string; sha256: string } => {
  const graph = createTinyGraph();
  const binary = new Uint8Array(encodeGraphBinary(graph));
  const sha256 = sha256Hex(binary);
  const gzipped = gzipSync(binary);
  const path = join(dataDir, 'tiny.bin.gz');
  writeFileSync(path, gzipped);
  return { path, sha256 };
};

describe('graph-lab entry-lesion.mjs (built bundle, unrelated cwd, temp-directory data)', () => {
  it('lesioning the sole output neuron changes the movement score versus baseline', () => {
    const dataDir = mkdtempSync(join(tmpdir(), 'graph-lab-data-'));
    const { path, sha256 } = writeFixtureGraph(dataDir);

    const args = {
      dataDir,
      mode: 'biological',
      graphPath: path,
      expectedSha256: sha256,
      sets: [[2]], // the tiny graph's sole "output population 0" neuron
      seedStart: 1,
      seedCount: 4,
      ticks: 20,
      shards: 1,
      bootstrapResamples: 200
    };
    const argsPath = join(dataDir, 'args.json');
    writeFileSync(argsPath, JSON.stringify(args));

    const result = runEntry('entry-lesion.mjs', argsPath) as {
      graph: string;
      graphSha256: string;
      label: string;
      baseline: { n: number; mean: number };
      sets: { indices: number[]; bodyIds: string[]; effect: { n: number; meanDifference: number }; n: number }[];
    };

    expect(result.graph).toBe('biological');
    expect(result.graphSha256).toBe(sha256);
    expect(result.label).toBe('Computed on DGX (private, not published)');
    expect(result.baseline.n).toBe(4);
    expect(result.sets).toHaveLength(1);
    expect(result.sets[0].indices).toEqual([2]);
    expect(result.sets[0].bodyIds).toHaveLength(1);
    expect(result.sets[0].n).toBe(4);
    // Nontriviality guard (matching `tests/unit/episode-lesion.test.ts`'s
    // own convention): lesioning the output neuron must actually move the
    // score away from baseline, or a bug that silently ignored `sets`
    // could still pass.
    expect(result.sets[0].effect.meanDifference).not.toBe(0);
  });

  it('produces byte-identical results across two independent runs (deterministic scoring)', () => {
    const dataDir = mkdtempSync(join(tmpdir(), 'graph-lab-data-'));
    const { path, sha256 } = writeFixtureGraph(dataDir);
    const args = {
      dataDir,
      mode: 'biological',
      graphPath: path,
      expectedSha256: sha256,
      sets: [[2]],
      seedStart: 1,
      seedCount: 4,
      ticks: 20,
      shards: 1,
      bootstrapResamples: 200
    };
    const argsPath = join(dataDir, 'args.json');
    writeFileSync(argsPath, JSON.stringify(args));

    const first = runEntry('entry-lesion.mjs', argsPath);
    const second = runEntry('entry-lesion.mjs', argsPath);
    expect(second).toEqual(first);
  });

  it('rejects a graph binary whose sha256 does not match (tamper/corruption guard)', () => {
    const dataDir = mkdtempSync(join(tmpdir(), 'graph-lab-data-'));
    const { path } = writeFixtureGraph(dataDir);
    const args = {
      dataDir,
      mode: 'biological',
      graphPath: path,
      expectedSha256: '0'.repeat(64),
      sets: [[2]],
      seedStart: 1,
      seedCount: 4,
      ticks: 20,
      shards: 1
    };
    const argsPath = join(dataDir, 'args.json');
    writeFileSync(argsPath, JSON.stringify(args));

    expect(() => runEntry('entry-lesion.mjs', argsPath)).toThrow(/does not match expected|reported an error/);
  });
});

describe('graph-lab entry-swapset.mjs (built bundle, unrelated cwd, temp-directory data)', () => {
  it('scores two graphs and computes a paired diff against a named baseline', () => {
    const dataDir = mkdtempSync(join(tmpdir(), 'graph-lab-data-'));
    const baseline = writeFixtureGraph(dataDir);

    const args = {
      dataDir,
      graphs: [
        { graphId: 'biological', path: baseline.path, expectedSha256: baseline.sha256 },
        { graphId: 'candidate', path: baseline.path, expectedSha256: baseline.sha256 }
      ],
      baselineGraphId: 'biological',
      seedStart: 1,
      seedCount: 4,
      ticks: 20,
      shards: 1,
      bootstrapResamples: 200
    };
    const argsPath = join(dataDir, 'args.json');
    writeFileSync(argsPath, JSON.stringify(args));

    const result = runEntry('entry-swapset.mjs', argsPath) as {
      scores: { graphId: string; n: number; mean: number }[];
      paired: { graphId: string; effect: { meanDifference: number } }[];
    };
    expect(result.scores).toHaveLength(2);
    expect(result.paired).toHaveLength(1);
    expect(result.paired[0].graphId).toBe('candidate');
    // Same graph scored twice under the same seeds: the paired difference
    // must be exactly zero (a real determinism check, not a placeholder).
    expect(result.paired[0].effect.meanDifference).toBe(0);
  });
});

// `entry-atlas-reeval.mjs` has no bundle to test here: it is deliberately
// excluded from `bundle.mjs`'s entry list (see its own doc comment and
// `bundle.mjs`'s) because its transitive dependency on
// `scripts/training/export-arms.ts`/`export-traces.ts` is not safe to
// single-file-bundle with today's sources (confirmed empirically -- both
// files end with a top-level CLI-invocation guard keyed on
// `import.meta.url`, which misfires once bundled).
