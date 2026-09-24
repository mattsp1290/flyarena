// @vitest-environment node
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';

import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { encodeGraphBinary } from '../../src/lib/connectome/format';
import { createTraceGraph } from '../fixtures/trace-graph';
import { createFixtureRewiredTraceGraph } from '../fixtures/trace-graph-rewire';
import { parseNullEvaluateArgs, readRewireIndex, runShardedEvaluation } from '../../scripts/null/null-evaluate';
import type { NullWorkerTask } from '../../scripts/null/null-worker';

/**
 * Coverage for `scripts/null/null-evaluate.ts`. Two halves:
 *
 * - `parseNullEvaluateArgs`: pure argv parsing, in-process.
 * - Shard determinism: the plan's WP2 stop/go gate #2 ("sharded evaluation
 *   of 5 graphs is byte-identical to the single-process result") and
 *   `02-authored-null-evaluation.md`'s own acceptance criterion ("`--shards 1`
 *   and `--shards 3` produce byte-identical output"). This exercises the
 *   real `node:child_process.fork` sharding mechanism end-to-end (not
 *   mocked) via the actual CLI, over `tests/fixtures/trace-graph.ts`'s tiny
 *   24-neuron graph and `tests/fixtures/trace-graph-rewire.ts`'s
 *   fixture-only rewiring — proving sharding and I/O determinism, not the
 *   production rewiring algorithm (`tests_python` covers that).
 */

describe('parseNullEvaluateArgs', () => {
  it('applies defaults and requires --rewired-index/--graphs-dir', () => {
    const args = parseNullEvaluateArgs(['--rewired-index', 'i.json', '--graphs-dir', 'g']);
    expect(args.biological).toBe(false);
    expect(args.heldOutStart).toBe(30001);
    expect(args.heldOutCount).toBe(100);
    expect(args.ticks).toBe(1800);
    expect(args.shards).toBe(18);
    expect(args.rewiredIndex).toBe(resolve(process.cwd(), 'i.json'));
    expect(args.graphsDir).toBe(resolve(process.cwd(), 'g'));
  });

  it('throws without --rewired-index', () => {
    expect(() => parseNullEvaluateArgs(['--graphs-dir', 'g'])).toThrow(/--rewired-index/);
  });

  it('throws without --graphs-dir', () => {
    expect(() => parseNullEvaluateArgs(['--rewired-index', 'i.json'])).toThrow(/--graphs-dir/);
  });

  it('parses --biological and numeric overrides', () => {
    const args = parseNullEvaluateArgs([
      '--biological',
      '--rewired-index',
      'i.json',
      '--graphs-dir',
      'g',
      '--held-out-start',
      '5',
      '--held-out-count',
      '3',
      '--ticks',
      '20',
      '--shards',
      '2'
    ]);
    expect(args.biological).toBe(true);
    expect(args.heldOutStart).toBe(5);
    expect(args.heldOutCount).toBe(3);
    expect(args.ticks).toBe(20);
    expect(args.shards).toBe(2);
  });

  it('rejects --graph without --biological', () => {
    expect(() =>
      parseNullEvaluateArgs(['--graph', 'x.bin.gz', '--rewired-index', 'i.json', '--graphs-dir', 'g'])
    ).toThrow(/--graph requires --biological/);
  });

  it('rejects an unknown flag', () => {
    expect(() => parseNullEvaluateArgs(['--rewired-index', 'i.json', '--graphs-dir', 'g', '--bogus'])).toThrow(
      /Unknown argument/
    );
  });
});

const sha256Hex = (data: Uint8Array): string => createHash('sha256').update(data).digest('hex');

describe('null-evaluate CLI: shard determinism (trace-graph fixture)', () => {
  let root: string;
  let graphsDir: string;
  let indexPath: string;
  let bioGzipPath: string;

  const REWIRED_SEEDS = [0, 1, 2, 3, 4];
  const HELD_OUT_START = 30001;
  const HELD_OUT_COUNT = 3;
  const TICKS = 20;

  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'null-evaluate-fixture-'));
    graphsDir = join(root, 'graphs');
    mkdirSync(graphsDir, { recursive: true });

    const bioGraph = createTraceGraph();
    const bioBinary = Buffer.from(encodeGraphBinary(bioGraph));
    const bioSha256 = sha256Hex(bioBinary);
    bioGzipPath = join(root, 'trace-graph.bin.gz');
    writeFileSync(bioGzipPath, gzipSync(bioBinary));

    const seedEntries = REWIRED_SEEDS.map((seed) => {
      const rewired = createFixtureRewiredTraceGraph(bioGraph, seed);
      const binary = Buffer.from(encodeGraphBinary(rewired));
      const binarySha256 = sha256Hex(binary);
      const gzipBytes = gzipSync(binary);
      const artifact = `trace-graph-rewired-seed${seed}.bin.gz`;
      writeFileSync(join(graphsDir, artifact), gzipBytes);
      return {
        seed,
        artifact,
        binarySha256,
        binaryBytes: binary.byteLength,
        gzipSha256: sha256Hex(gzipBytes),
        gzipBytes: gzipBytes.byteLength,
        stats: { acceptedSwaps: 42, attempts: 100, seed }
      };
    });

    const index = {
      sourceArtifact: 'trace-graph.bin.gz',
      sourceSha256: bioSha256,
      rewireSourceSha256: sha256Hex(Buffer.from('fixture-only-rewiring, not scripts/data/rewire.py')),
      seeds: seedEntries
    };
    indexPath = join(root, 'index.json');
    writeFileSync(indexPath, JSON.stringify(index, null, 2));
  });

  afterAll(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  const runCli = (shards: number, out: string) =>
    spawnSync(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/null/null-evaluate.ts',
        '--biological',
        '--graph',
        bioGzipPath,
        '--rewired-index',
        indexPath,
        '--graphs-dir',
        graphsDir,
        '--held-out-start',
        String(HELD_OUT_START),
        '--held-out-count',
        String(HELD_OUT_COUNT),
        '--ticks',
        String(TICKS),
        '--shards',
        String(shards),
        '--out',
        out
      ],
      { encoding: 'utf8', timeout: 60_000 }
    );

  it('--shards 1 and --shards 3 produce byte-identical authored.json', () => {
    const out1 = join(root, 'authored-shards1.json');
    const out3 = join(root, 'authored-shards3.json');

    const result1 = runCli(1, out1);
    expect(result1.status, result1.stderr).toBe(0);

    const result3 = runCli(3, out3);
    expect(result3.status, result3.stderr).toBe(0);

    const bytes1 = readFileSync(out1);
    const bytes3 = readFileSync(out3);
    expect(bytes3.equals(bytes1)).toBe(true);

    const parsed = JSON.parse(bytes1.toString('utf8'));
    expect(parsed.version).toBe(1);
    expect(parsed.ticks).toBe(TICKS);
    expect(parsed.seeds).toEqual({ start: HELD_OUT_START, count: HELD_OUT_COUNT });
    expect(parsed.biological.movementScore).toHaveLength(HELD_OUT_COUNT);
    expect(parsed.disconnected.movementScore).toHaveLength(HELD_OUT_COUNT);
    expect(parsed.rewired).toHaveLength(REWIRED_SEEDS.length);
    expect(parsed.rewired.map((r: { seed: number }) => r.seed)).toEqual(REWIRED_SEEDS);
    for (const entry of parsed.rewired) {
      expect(entry.movementScore).toHaveLength(HELD_OUT_COUNT);
      for (const score of entry.movementScore) expect(Number.isFinite(score)).toBe(true);
    }
  });

  it('rejects a rewired file whose bytes do not match index.json', () => {
    const tamperedDir = join(root, 'graphs-tampered');
    mkdirSync(tamperedDir, { recursive: true });
    cpSync(graphsDir, tamperedDir, { recursive: true });
    const victim = join(tamperedDir, 'trace-graph-rewired-seed0.bin.gz');
    writeFileSync(victim, Buffer.concat([readFileSync(victim), Buffer.from([0])]));

    const out = join(root, 'authored-tampered.json');
    const result = spawnSync(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/null/null-evaluate.ts',
        '--rewired-index',
        indexPath,
        '--graphs-dir',
        tamperedDir,
        '--held-out-count',
        '1',
        '--ticks',
        String(TICKS),
        '--shards',
        '1',
        '--out',
        out
      ],
      { encoding: 'utf8', timeout: 60_000 }
    );
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/failed verification|gzip sha256/);
  });

  it('rejects a rewired file with a flipped byte (same length, wrong gzip sha256)', () => {
    // Appending a byte (the case above) only ever reaches the gzipBytes
    // length check in verifyRewiredFiles; flipping a byte in place keeps
    // the length the same and actually exercises the sha256 comparison.
    const tamperedDir = join(root, 'graphs-tampered-flip');
    mkdirSync(tamperedDir, { recursive: true });
    cpSync(graphsDir, tamperedDir, { recursive: true });
    const victim = join(tamperedDir, 'trace-graph-rewired-seed1.bin.gz');
    const bytes = readFileSync(victim);
    bytes[bytes.length - 5] ^= 0xff;
    writeFileSync(victim, bytes);

    const out = join(root, 'authored-tampered-flip.json');
    const result = spawnSync(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/null/null-evaluate.ts',
        '--rewired-index',
        indexPath,
        '--graphs-dir',
        tamperedDir,
        '--held-out-count',
        '1',
        '--ticks',
        String(TICKS),
        '--shards',
        '1',
        '--out',
        out
      ],
      { encoding: 'utf8', timeout: 60_000 }
    );
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/gzip sha256/);
  });
});

describe('readRewireIndex: duplicate/malformed seed rejection', () => {
  it('rejects a duplicate rewiring seed', () => {
    const root = mkdtempSync(join(tmpdir(), 'null-evaluate-dup-seed-'));
    const indexPath = join(root, 'index.json');
    const seedEntry = {
      seed: 0,
      artifact: 'x.bin.gz',
      binarySha256: 'a'.repeat(64),
      binaryBytes: 1,
      gzipSha256: 'b'.repeat(64),
      gzipBytes: 1,
      stats: { acceptedSwaps: 1, attempts: 1 }
    };
    writeFileSync(
      indexPath,
      JSON.stringify({
        sourceArtifact: 'src.bin.gz',
        sourceSha256: 'c'.repeat(64),
        rewireSourceSha256: 'd'.repeat(64),
        seeds: [seedEntry, { ...seedEntry }]
      })
    );
    expect(() => readRewireIndex(indexPath)).toThrow(/more than once/);
    rmSync(root, { recursive: true, force: true });
  });

  it('rejects a non-integer seed', () => {
    const root = mkdtempSync(join(tmpdir(), 'null-evaluate-bad-seed-'));
    const indexPath = join(root, 'index.json');
    writeFileSync(
      indexPath,
      JSON.stringify({
        sourceArtifact: 'src.bin.gz',
        sourceSha256: 'c'.repeat(64),
        rewireSourceSha256: 'd'.repeat(64),
        seeds: [
          {
            seed: 1.5,
            artifact: 'x.bin.gz',
            binarySha256: 'a'.repeat(64),
            binaryBytes: 1,
            gzipSha256: 'b'.repeat(64),
            gzipBytes: 1,
            stats: { acceptedSwaps: 1, attempts: 1 }
          }
        ]
      })
    );
    expect(() => readRewireIndex(indexPath)).toThrow(/malformed seed entry/);
    rmSync(root, { recursive: true, force: true });
  });
});

describe('runShardedEvaluation: failure/abort paths (stub worker)', () => {
  const stubWorkerPath = resolve(dirname(fileURLToPath(import.meta.url)), '../fixtures/null-stub-worker.mjs');

  /** `delayMs` is a stub-worker-only extension of the wire protocol, not part of the real `NullWorkerTask`. */
  const task = (graphId: string, delayMs = 0): NullWorkerTask & { delayMs: number } => ({
    graphId,
    mode: 'rewired',
    path: 'unused',
    expectedSha256: 'unused',
    heldOutSeeds: [],
    ticks: 0,
    delayMs
  });

  it('collects every result regardless of which shard finishes which task first', async () => {
    // Reverse-order completion: task 0 is slowest, task 4 is fastest.
    const tasks = [0, 1, 2, 3, 4].map((i) => task(`t${i}`, (5 - i) * 15));
    const results = await runShardedEvaluation(tasks, 5, stubWorkerPath);
    expect([...results.keys()].sort()).toEqual(['t0', 't1', 't2', 't3', 't4']);
  });

  it('a task-level error aborts the whole run quickly, not after the full queue drains', async () => {
    const tasks = [task('err'), ...Array.from({ length: 8 }, (_, i) => task(`t${i}`, 300))];
    const started = Date.now();
    await expect(runShardedEvaluation(tasks, 2, stubWorkerPath)).rejects.toThrow(/stub-induced failure/);
    const elapsedMs = Date.now() - started;
    // If the other shard kept draining its half of the queue (4 tasks x 300ms
    // each), this would take >= 1200ms. Aborting promptly keeps it well under.
    expect(elapsedMs).toBeLessThan(900);
  });

  it('a worker killed by a signal is reported as a failure, not treated as a clean exit', async () => {
    const tasks = [task('kill'), task('t1', 50), task('t2', 50)];
    await expect(runShardedEvaluation(tasks, 3, stubWorkerPath)).rejects.toThrow(/exited unexpectedly/);
  });
});
