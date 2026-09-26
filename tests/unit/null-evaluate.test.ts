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
import {
  buildGraphListTasks,
  buildTasks,
  parseNullEvaluateArgs,
  readGraphListIndex,
  readRewireIndex,
  runNullEvaluate,
  runShardedEvaluation
} from '../../scripts/null/null-evaluate';
import type { NullSeedResult, NullWorkerMessage, NullWorkerTask } from '../../scripts/null/null-worker';

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
    expect(args.decoder).toBe('authored');
    expect(args.rewiredSeeds).toBeUndefined();
  });

  it('parses --decoder for each accepted variant', () => {
    for (const decoder of ['authored', 'authored-flip-thrust', 'authored-flip-yaw', 'authored-flip-both'] as const) {
      const args = parseNullEvaluateArgs(['--rewired-index', 'i.json', '--graphs-dir', 'g', '--decoder', decoder]);
      expect(args.decoder).toBe(decoder);
    }
  });

  it('rejects an unknown --decoder value', () => {
    // 'trained'/'silenced'/'parked' are real EpisodeDecoderKind values --
    // just never valid for null-evaluate.ts, which always drives the left
    // agent through the authored family against a parked opponent.
    for (const bogus of ['trained', 'silenced', 'parked', 'authored-flip-brake', '']) {
      expect(() =>
        parseNullEvaluateArgs(['--rewired-index', 'i.json', '--graphs-dir', 'g', '--decoder', bogus])
      ).toThrow(/--decoder must be one of/);
    }
  });

  it('parses --rewired-seeds START:END', () => {
    const args = parseNullEvaluateArgs([
      '--rewired-index',
      'i.json',
      '--graphs-dir',
      'g',
      '--rewired-seeds',
      '0:5'
    ]);
    expect(args.rewiredSeeds).toEqual({ start: 0, end: 5 });
  });

  it('rejects a malformed --rewired-seeds value', () => {
    expect(() =>
      parseNullEvaluateArgs(['--rewired-index', 'i.json', '--graphs-dir', 'g', '--rewired-seeds', '5'])
    ).toThrow(/--rewired-seeds must be START:END/);
  });

  it('rejects --rewired-seeds with end <= start', () => {
    expect(() =>
      parseNullEvaluateArgs(['--rewired-index', 'i.json', '--graphs-dir', 'g', '--rewired-seeds', '5:5'])
    ).toThrow(/end must be greater than start/);
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

  it('parses --graph-list without --rewired-index/--graphs-dir', () => {
    const args = parseNullEvaluateArgs(['--graph-list', 'list.json']);
    expect(args.graphList).toBe(resolve(process.cwd(), 'list.json'));
    expect(args.rewiredIndex).toBeUndefined();
    expect(args.graphsDir).toBeUndefined();
  });

  it('rejects --graph-list combined with --rewired-index', () => {
    expect(() =>
      parseNullEvaluateArgs(['--graph-list', 'list.json', '--rewired-index', 'i.json'])
    ).toThrow(/--graph-list is mutually exclusive with --rewired-index\/--graphs-dir/);
  });

  it('rejects --graph-list combined with --graphs-dir', () => {
    expect(() =>
      parseNullEvaluateArgs(['--graph-list', 'list.json', '--graphs-dir', 'g'])
    ).toThrow(/--graph-list is mutually exclusive with --rewired-index\/--graphs-dir/);
  });

  it('rejects neither --graph-list nor --rewired-index/--graphs-dir', () => {
    expect(() => parseNullEvaluateArgs([])).toThrow(/--rewired-index is required/);
  });

  it('rejects --graph-list combined with --rewired-seeds (a silent no-op otherwise: --graph-list ignores it)', () => {
    expect(() =>
      parseNullEvaluateArgs(['--graph-list', 'list.json', '--rewired-seeds', '0:5'])
    ).toThrow(/--rewired-seeds only applies to --rewired-index mode/);
  });

  it('rejects --out without a .json extension', () => {
    // Regression test for a dual-review finding: runNullEvaluate derives its
    // <out>.run.json sidecar path by stripping a trailing ".json" off --out;
    // without this guard, a non-.json --out would make the sidecar path
    // collide with --out itself, and writing the sidecar would silently
    // overwrite the just-written authored.json.
    expect(() =>
      parseNullEvaluateArgs(['--rewired-index', 'i.json', '--graphs-dir', 'g', '--out', 'authored'])
    ).toThrow(/--out must end with "\.json"/);
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

  it('--rewired-seeds 0:2 evaluates only those seeds end-to-end (buildTasks and assembleRaw agree)', () => {
    // Regression test: an earlier version filtered buildTasks's task list by
    // --rewired-seeds but assembleRaw still iterated the full, unfiltered
    // index when reassembling output, throwing "missing results for
    // rewired-2" (the first seed outside the requested range) instead of
    // producing a 2-seed authored.json.
    const out = join(root, 'authored-rewired-seeds-0-2.json');
    const result = spawnSync(
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
        '2',
        '--rewired-seeds',
        '0:2',
        '--decoder',
        'authored',
        '--out',
        out
      ],
      { encoding: 'utf8', timeout: 60_000 }
    );
    expect(result.status, result.stderr).toBe(0);
    const parsed = JSON.parse(readFileSync(out, 'utf8'));
    expect(parsed.decoder).toBe('authored');
    expect(parsed.rewired.map((r: { seed: number }) => r.seed)).toEqual([0, 1]);
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

describe('null-evaluate CLI: --graph-list mode (trace-graph fixture)', () => {
  // `.agents/plans/pathway-interventions/03-evaluation.md`'s WP2 tests:
  // "--graph-list on fixture graphs: 1 vs 3 shards give identical output; a
  // sha mismatch fails; --graph-list with --rewired-index fails". This
  // mirrors the `--rewired-index` shard-determinism block above exactly,
  // over a `scripts/analysis/interventions.py`-shaped index.json (id/path/
  // gzipSha256/binarySha256 entries, keyed by an arbitrary string id) instead
  // of a `rewire_batch.py`-shaped one (numeric seed entries).
  let root: string;
  let graphListPath: string;
  let bioGzipPath: string;

  const GRAPH_IDS = ['P', 'C000', 'C001', 'MQ2000'];
  const HELD_OUT_START = 30001;
  const HELD_OUT_COUNT = 3;
  const TICKS = 20;

  beforeAll(() => {
    root = mkdtempSync(join(tmpdir(), 'null-evaluate-graph-list-fixture-'));

    const bioGraph = createTraceGraph();
    const bioBinary = Buffer.from(encodeGraphBinary(bioGraph));
    const bioSha256 = sha256Hex(bioBinary);
    bioGzipPath = join(root, 'trace-graph.bin.gz');
    writeFileSync(bioGzipPath, gzipSync(bioBinary));

    const entries = GRAPH_IDS.map((id, i) => {
      const rewired = createFixtureRewiredTraceGraph(bioGraph, i);
      const binary = Buffer.from(encodeGraphBinary(rewired));
      const binarySha256 = sha256Hex(binary);
      const gzipBytes = gzipSync(binary);
      const relativePath = `graphs/${id}.bin.gz`;
      mkdirSync(join(root, 'graphs'), { recursive: true });
      writeFileSync(join(root, relativePath), gzipBytes);
      return { id, path: relativePath, gzipSha256: sha256Hex(gzipBytes), binarySha256 };
    });

    const index = {
      sourceArtifact: 'trace-graph.bin.gz',
      sourceSha256: bioSha256,
      entries
    };
    graphListPath = join(root, 'index.json');
    writeFileSync(graphListPath, JSON.stringify(index, null, 2));
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
        '--graph-list',
        graphListPath,
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

  it('--shards 1 and --shards 3 produce byte-identical output, keyed by id and sorted', () => {
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
    expect(parsed.graphs).toHaveLength(GRAPH_IDS.length);
    // Sorted by id ascending (plain string comparison): 'C000' < 'C001' < 'MQ2000' < 'P'.
    expect(parsed.graphs.map((g: { id: string }) => g.id)).toEqual(['C000', 'C001', 'MQ2000', 'P']);
    for (const entry of parsed.graphs) {
      expect(entry.movementScore).toHaveLength(HELD_OUT_COUNT);
      for (const score of entry.movementScore) expect(Number.isFinite(score)).toBe(true);
    }
  });

  it('rejects a graph-list file whose bytes do not match index.json (sha mismatch)', () => {
    const tamperedRoot = mkdtempSync(join(tmpdir(), 'null-evaluate-graph-list-tampered-'));
    cpSync(root, tamperedRoot, { recursive: true });
    const victim = join(tamperedRoot, 'graphs', 'C000.bin.gz');
    const bytes = readFileSync(victim);
    bytes[bytes.length - 5] ^= 0xff;
    writeFileSync(victim, bytes);

    const out = join(tamperedRoot, 'authored-tampered.json');
    const result = spawnSync(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/null/null-evaluate.ts',
        '--graph-list',
        join(tamperedRoot, 'index.json'),
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
    rmSync(tamperedRoot, { recursive: true, force: true });
  });

  it('rejects a graph-list entry whose binarySha256 does not match the decompressed bytes (worker-side check, correct gzipSha256)', () => {
    // Regression test for the worker-side (decompressed) sha layer actually
    // being enforced, not merely declared: a regression that accidentally
    // wired `expectedSha256: entry.gzipSha256` (instead of `binarySha256`)
    // into the task would pass every other test in this suite, since the
    // gzip bytes themselves are untouched here -- only the index's recorded
    // binarySha256 is corrupted, so verifyGraphListFiles's gzip-level check
    // (parent, pre-fork) still passes, and only null-worker.ts's
    // loadVerifiedGraphBinary (decompressed, in-worker) can catch this.
    const tamperedRoot = mkdtempSync(join(tmpdir(), 'null-evaluate-graph-list-binary-sha-'));
    cpSync(root, tamperedRoot, { recursive: true });
    const indexPath = join(tamperedRoot, 'index.json');
    const index = JSON.parse(readFileSync(indexPath, 'utf8'));
    index.entries[0].binarySha256 = '0'.repeat(64);
    writeFileSync(indexPath, JSON.stringify(index));

    const out = join(tamperedRoot, 'authored-binary-sha-tampered.json');
    const result = spawnSync(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/null/null-evaluate.ts',
        '--graph-list',
        indexPath,
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
    expect(result.stderr).toMatch(/decompressed sha256/);
    rmSync(tamperedRoot, { recursive: true, force: true });
  });

  it('--graph-list combined with --rewired-index fails at the CLI', () => {
    const out = join(root, 'authored-mutex.json');
    const result = spawnSync(
      process.execPath,
      [
        '--import',
        'tsx',
        'scripts/null/null-evaluate.ts',
        '--graph-list',
        graphListPath,
        '--rewired-index',
        graphListPath,
        '--graphs-dir',
        root,
        '--out',
        out
      ],
      { encoding: 'utf8', timeout: 60_000 }
    );
    expect(result.status).not.toBe(0);
    expect(result.stderr).toMatch(/--graph-list is mutually exclusive with --rewired-index\/--graphs-dir/);
  });
});

describe('readGraphListIndex: duplicate/malformed entry rejection', () => {
  it('rejects a duplicate graph id', () => {
    const root = mkdtempSync(join(tmpdir(), 'null-evaluate-graph-list-dup-id-'));
    const indexPath = join(root, 'index.json');
    const entry = { id: 'P', path: 'x.bin.gz', gzipSha256: 'a'.repeat(64), binarySha256: 'b'.repeat(64) };
    writeFileSync(
      indexPath,
      JSON.stringify({
        sourceArtifact: 'src.bin.gz',
        sourceSha256: 'c'.repeat(64),
        entries: [entry, { ...entry }]
      })
    );
    expect(() => readGraphListIndex(indexPath)).toThrow(/more than once/);
    rmSync(root, { recursive: true, force: true });
  });

  it('rejects a malformed entry', () => {
    const root = mkdtempSync(join(tmpdir(), 'null-evaluate-graph-list-bad-entry-'));
    const indexPath = join(root, 'index.json');
    writeFileSync(
      indexPath,
      JSON.stringify({
        sourceArtifact: 'src.bin.gz',
        sourceSha256: 'c'.repeat(64),
        entries: [{ id: 'P', path: 'x.bin.gz', gzipSha256: 'a'.repeat(64) }] // missing binarySha256
      })
    );
    expect(() => readGraphListIndex(indexPath)).toThrow(/malformed graph-list entry/);
    rmSync(root, { recursive: true, force: true });
  });

  it.each(['biological', 'disconnected'])('rejects an entry using the reserved id "%s"', (reservedId) => {
    const root = mkdtempSync(join(tmpdir(), 'null-evaluate-graph-list-reserved-id-'));
    const indexPath = join(root, 'index.json');
    writeFileSync(
      indexPath,
      JSON.stringify({
        sourceArtifact: 'src.bin.gz',
        sourceSha256: 'c'.repeat(64),
        entries: [{ id: reservedId, path: 'x.bin.gz', gzipSha256: 'a'.repeat(64), binarySha256: 'b'.repeat(64) }]
      })
    );
    expect(() => readGraphListIndex(indexPath)).toThrow(/reserved graph id/);
    rmSync(root, { recursive: true, force: true });
  });

  it('rejects an entry path that escapes index.json\'s own directory (absolute path)', () => {
    const root = mkdtempSync(join(tmpdir(), 'null-evaluate-graph-list-abs-path-'));
    const indexPath = join(root, 'index.json');
    writeFileSync(
      indexPath,
      JSON.stringify({
        sourceArtifact: 'src.bin.gz',
        sourceSha256: 'c'.repeat(64),
        entries: [{ id: 'P', path: '/etc/passwd', gzipSha256: 'a'.repeat(64), binarySha256: 'b'.repeat(64) }]
      })
    );
    expect(() => readGraphListIndex(indexPath)).toThrow(/path outside index\.json's own directory/);
    rmSync(root, { recursive: true, force: true });
  });

  it('rejects an entry path that traverses above index.json\'s own directory ("..")', () => {
    const root = mkdtempSync(join(tmpdir(), 'null-evaluate-graph-list-traversal-'));
    const indexPath = join(root, 'index.json');
    writeFileSync(
      indexPath,
      JSON.stringify({
        sourceArtifact: 'src.bin.gz',
        sourceSha256: 'c'.repeat(64),
        entries: [{ id: 'P', path: '../outside.bin.gz', gzipSha256: 'a'.repeat(64), binarySha256: 'b'.repeat(64) }]
      })
    );
    expect(() => readGraphListIndex(indexPath)).toThrow(/path outside index\.json's own directory/);
    rmSync(root, { recursive: true, force: true });
  });
});

describe('buildGraphListTasks: --decoder propagation', () => {
  it('every task (biological, disconnected, and each graph-list entry) carries the requested decoder, sorted by id', () => {
    const index = {
      sourceArtifact: 'src.bin.gz',
      sourceSha256: 'c'.repeat(64),
      entries: [
        { id: 'P', path: 'p.bin.gz', gzipSha256: 'a'.repeat(64), binarySha256: '1'.repeat(64) },
        { id: 'C000', path: 'c000.bin.gz', gzipSha256: 'b'.repeat(64), binarySha256: '2'.repeat(64) }
      ]
    };
    const args = {
      biological: true,
      graph: undefined,
      rewiredIndex: undefined,
      graphsDir: undefined,
      graphList: 'unused',
      heldOutStart: 30001,
      heldOutCount: 2,
      ticks: 20,
      shards: 1,
      out: 'unused.json',
      decoder: 'authored-flip-both' as const,
      rewiredSeeds: undefined
    };
    const tasks = buildGraphListTasks(index, args, '/graphs-dir', 'bio.bin.gz');
    expect(tasks.length).toBeGreaterThan(0);
    for (const task of tasks) expect(task.decoder).toBe('authored-flip-both');
    const graphListGraphIds = tasks.filter((t) => t.mode === 'rewired').map((t) => t.graphId);
    expect(graphListGraphIds).toEqual(['C000', 'P']);
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

describe('buildTasks: --decoder propagation and --rewired-seeds filtering', () => {
  const indexFor = (seeds: readonly number[]) => ({
    sourceArtifact: 'src.bin.gz',
    sourceSha256: 'c'.repeat(64),
    rewireSourceSha256: 'd'.repeat(64),
    seeds: seeds.map((seed) => ({
      seed,
      artifact: `rewired-${seed}.bin.gz`,
      binarySha256: `${seed}`.padStart(64, '0'),
      binaryBytes: 1,
      gzipSha256: `${seed}`.padStart(64, '0'),
      gzipBytes: 1,
      stats: { acceptedSwaps: 1, attempts: 1 }
    }))
  });

  const baseArgs = () => ({
    biological: true,
    graph: undefined,
    rewiredIndex: 'unused',
    graphsDir: 'unused',
    heldOutStart: 30001,
    heldOutCount: 2,
    ticks: 20,
    shards: 1,
    out: 'unused.json',
    decoder: 'authored' as const,
    rewiredSeeds: undefined as { start: number; end: number } | undefined
  });

  it('--rewired-seeds 0:2 evaluates only seeds 0 and 1', () => {
    const index = indexFor([0, 1, 2, 3, 4]);
    const args = { ...baseArgs(), rewiredSeeds: { start: 0, end: 2 } };
    const tasks = buildTasks(index, args, 'bio.bin.gz');
    const rewiredGraphIds = tasks.filter((t) => t.mode === 'rewired').map((t) => t.graphId);
    expect(rewiredGraphIds).toEqual(['rewired-0', 'rewired-1']);
  });

  it('omitting --rewired-seeds evaluates every seed in the index', () => {
    const index = indexFor([0, 1, 2]);
    const tasks = buildTasks(index, baseArgs(), 'bio.bin.gz');
    const rewiredGraphIds = tasks.filter((t) => t.mode === 'rewired').map((t) => t.graphId);
    expect(rewiredGraphIds).toEqual(['rewired-0', 'rewired-1', 'rewired-2']);
  });

  it('every task (biological, disconnected, and each rewired seed) carries the requested decoder', () => {
    const index = indexFor([0, 1]);
    const args = { ...baseArgs(), decoder: 'authored-flip-both' as const };
    const tasks = buildTasks(index, args, 'bio.bin.gz');
    expect(tasks.length).toBeGreaterThan(0);
    for (const task of tasks) expect(task.decoder).toBe('authored-flip-both');
  });

  it('throws when --rewired-seeds matches no seeds in the index (regression: used to silently produce an empty rewired list)', () => {
    const index = indexFor([0, 1, 2]);
    const args = { ...baseArgs(), rewiredSeeds: { start: 10, end: 12 } };
    expect(() => buildTasks(index, args, 'bio.bin.gz')).toThrow(/missing 2: 10, 11/);
  });

  it('throws when --rewired-seeds only partially matches the index (regression: used to silently score fewer seeds than requested)', () => {
    const index = indexFor([0, 1, 2]); // seed 3 is absent
    const args = { ...baseArgs(), rewiredSeeds: { start: 1, end: 4 } };
    expect(() => buildTasks(index, args, 'bio.bin.gz')).toThrow(/requested 3 seed\(s\).*missing 1: 3/s);
  });
});

describe('runNullEvaluate: refuses to overwrite the canonical default --out with a non-canonical run', () => {
  // Regression test for a dual-review finding: the plan's own reproduction-
  // gate example command (`--rewired-seeds 0:5 --decoder authored`, no
  // --out) would otherwise silently overwrite the canonical, hours-long
  // full-index run at the default --out path. The guard must fire before
  // any file is read (readRewireIndex would throw on the bogus paths below
  // first if it ran), so a non-matching error message here would mean the
  // guard isn't actually first.
  const nonCanonicalArgsWithDefaultOut = (overrides: Partial<Parameters<typeof runNullEvaluate>[0]>) => ({
    biological: true,
    graph: undefined,
    rewiredIndex: '/nonexistent/index.json',
    graphsDir: '/nonexistent/graphs',
    heldOutStart: 30001,
    heldOutCount: 100,
    ticks: 1800,
    shards: 1,
    out: resolve(process.cwd(), 'training/runs/null/authored.json'),
    decoder: 'authored' as const,
    rewiredSeeds: undefined as { start: number; end: number } | undefined,
    ...overrides
  });

  it('throws for a non-authored decoder writing to the default --out', async () => {
    await expect(
      runNullEvaluate(nonCanonicalArgsWithDefaultOut({ decoder: 'authored-flip-both' }))
    ).rejects.toThrow(/refusing to write a non-canonical run/);
  });

  it("throws for the reproduction gate's own --rewired-seeds-restricted authored run writing to the default --out", async () => {
    await expect(
      runNullEvaluate(nonCanonicalArgsWithDefaultOut({ rewiredSeeds: { start: 0, end: 5 } }))
    ).rejects.toThrow(/refusing to write a non-canonical run/);
  });

  it('does not throw this guard for a canonical (authored, unfiltered) run at the default --out (fails later, on the nonexistent index instead)', async () => {
    // Proves the guard is scoped correctly: a plain authored run targeting
    // the default --out is legitimate and must not be blocked by this
    // check. It still fails -- just for an unrelated, expected reason (the
    // fixture's rewiredIndex path does not exist).
    await expect(runNullEvaluate(nonCanonicalArgsWithDefaultOut({}))).rejects.not.toThrow(
      /refusing to write a non-canonical run/
    );
  });

  it('throws for a --graph-list run writing to the default (--rewired-index-shaped) --out', async () => {
    await expect(
      runNullEvaluate(
        nonCanonicalArgsWithDefaultOut({
          rewiredIndex: undefined,
          graphsDir: undefined,
          graphList: '/nonexistent/graph-list.json'
        })
      )
    ).rejects.toThrow(/refusing to write a non-canonical run/);
  });
});

describe('runNullEvaluate: --graph-list mutual exclusion (direct call, bypassing the CLI parser)', () => {
  // parseNullEvaluateArgs already rejects this combination (see the CLI-level
  // test above), but runNullEvaluate is also called directly by any caller
  // that builds its own NullEvaluateArgs -- this proves the function itself
  // re-checks rather than trusting every caller routes through the parser.
  it('rejects --graph-list combined with --rewired-index/--graphs-dir', async () => {
    await expect(
      runNullEvaluate({
        biological: false,
        graph: undefined,
        rewiredIndex: '/nonexistent/index.json',
        graphsDir: '/nonexistent/graphs',
        graphList: '/nonexistent/graph-list.json',
        heldOutStart: 30001,
        heldOutCount: 1,
        ticks: 20,
        shards: 1,
        out: resolve(process.cwd(), 'training/runs/interventions/scratch.json'),
        decoder: 'authored' as const,
        rewiredSeeds: undefined
      })
    ).rejects.toThrow(/--graph-list is mutually exclusive with --rewired-index\/--graphs-dir/);
  });

  it('rejects --graph-list combined with --rewired-seeds', async () => {
    await expect(
      runNullEvaluate({
        biological: false,
        graph: undefined,
        rewiredIndex: undefined,
        graphsDir: undefined,
        graphList: '/nonexistent/graph-list.json',
        heldOutStart: 30001,
        heldOutCount: 1,
        ticks: 20,
        shards: 1,
        out: resolve(process.cwd(), 'training/runs/interventions/scratch.json'),
        decoder: 'authored' as const,
        rewiredSeeds: { start: 0, end: 5 }
      })
    ).rejects.toThrow(/--rewired-seeds only applies to --rewired-index mode/);
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
    const results = await runShardedEvaluation<NullWorkerTask, NullSeedResult, NullWorkerMessage>(
      tasks,
      5,
      stubWorkerPath
    );
    expect([...results.keys()].sort()).toEqual(['t0', 't1', 't2', 't3', 't4']);
  });

  it('a task-level error aborts the whole run quickly, not after the full queue drains', async () => {
    // Self-calibrating regression floor rather than a tight fixed-ms bound
    // (flagged as a small flakiness risk on slow/loaded CI runners by the
    // thermo-nuclear maintainability review): with 2 shards and `err` as
    // the very first dispatched task, a correct abort only ever lets at
    // most one delayed task per shard start (each killed via SIGTERM
    // mid-busy-wait, so `delayMs` never elapses for it) before `abortAll`
    // fires -- wall time for the *passing* path is dominated by fork()/IPC
    // overhead alone, independent of `remainingTaskCount`/`delayMs`. The
    // pre-fix ("keep draining after an error") behavior instead splits the
    // remaining tasks across both shards and finishes no faster than
    // `regressionFloorMs`. Scaling both up (more remaining tasks, same
    // per-task delay) widens the gap between "fast, correct" and "slow,
    // regressed" far beyond any plausible fork/IPC jitter, so the assertion
    // below can use a generous threshold without losing the ability to
    // catch the regression.
    const delayMs = 300;
    const remainingTaskCount = 24;
    const tasks = [task('err'), ...Array.from({ length: remainingTaskCount }, (_, i) => task(`t${i}`, delayMs))];
    const regressionFloorMs = (remainingTaskCount / 2) * delayMs; // 3600ms
    const started = Date.now();
    await expect(
      runShardedEvaluation<NullWorkerTask, NullSeedResult, NullWorkerMessage>(tasks, 2, stubWorkerPath)
    ).rejects.toThrow(/stub-induced failure/);
    const elapsedMs = Date.now() - started;
    expect(elapsedMs).toBeLessThan(regressionFloorMs / 2); // generous 1800ms bound, still well below the 3600ms floor
  });

  it('a worker killed by a signal is reported as a failure, not treated as a clean exit', async () => {
    const tasks = [task('kill'), task('t1', 50), task('t2', 50)];
    await expect(
      runShardedEvaluation<NullWorkerTask, NullSeedResult, NullWorkerMessage>(tasks, 3, stubWorkerPath)
    ).rejects.toThrow(/exited unexpectedly/);
  });
});
