import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import type { NullEvaluationRaw, NullGraphRaw } from '../../scripts/null/null-evaluate';
import type { NullTrainedEvaluationRaw } from '../../scripts/null/null-trained-evaluate';
import {
  DEFAULT_MANIFEST,
  DEFAULT_OUT,
  DEFAULT_REPORT_MD,
  buildArtifact,
  parseNullReportArgs,
  resolveRunMeta,
  runNullReport,
  updateManifestWithRewiringNull,
  verifyManifestRoundTrips,
  type NullReportArgs
} from '../../scripts/null/null-report';

/**
 * Coverage for `scripts/null/null-report.ts` — a dual-review pass on the
 * first version of this branch found this file had no tests at all,
 * including the plan's own "running null:report twice gives byte-identical
 * output" requirement, which had only been checked by hand. Everything
 * here runs against small, hand-built `NullEvaluationRaw` fixtures (never
 * a real evaluation run), so the whole suite is fast.
 */

const HEX64 = (fill: string): string => fill.repeat(64);

/** A small, internally-consistent `NullEvaluationRaw` — 5 rewired seeds (0..4), enough to exercise every code path without a real evaluation run. */
const buildRaw = (overrides: Partial<NullEvaluationRaw> = {}): NullEvaluationRaw => {
  const heldOutSeeds = [30001, 30002, 30003];
  const graph = (base: number): NullGraphRaw => ({
    heldOutSeeds,
    movementScore: [base, base + 1, base + 2],
    foodPickups: [1, 2, 3],
    hazardContacts: [0, 0, 1]
  });

  return {
    version: 1,
    sourceGraphSha256: HEX64('a'),
    rewireSourceSha256: HEX64('b'),
    seeds: { start: 30001, count: 3 },
    ticks: 20,
    substeps: 4,
    biological: graph(1),
    disconnected: graph(-1),
    rewired: [0, 1, 2, 3, 4].map((seed) => ({
      seed,
      gzipSha256: HEX64(String(seed)),
      acceptedSwaps: 10 + seed,
      attempts: 20 + seed,
      ...graph(seed)
    })),
    host: { arch: 'arm64', node: 'v22.22.3' },
    ...overrides
  };
};

/** Alphabetically-keyed by construction, so it already matches `sortKeysDeep`'s output — `updateManifestWithRewiringNull`'s round-trip safety check requires this. */
const writeTestManifest = (path: string, binarySha256: string): void => {
  const manifest = { artifact: 'test.bin.gz', binarySha256, note: 'test fixture, not the real manifest' };
  writeFileSync(path, `${JSON.stringify(manifest, null, 2)}\n`);
};

/** A small, internally-consistent `NullTrainedEvaluationRaw` (WP3) — 4 rewired seeds (0..3) and the three biological trainer seeds, enough to exercise `buildTrainedSection`/the merge path without a real training run. */
const buildTrainedRaw = (overrides: Partial<NullTrainedEvaluationRaw> = {}): NullTrainedEvaluationRaw => {
  const heldOutSeeds = [30001, 30002, 30003];
  const graph = (base: number) => ({
    heldOutSeeds,
    movementScore: [base, base + 1, base + 2],
    foodPickups: [1, 2, 3],
    hazardContacts: [0, 0, 1]
  });
  return {
    version: 1,
    seeds: { start: 30001, count: 3 },
    ticks: 20,
    substeps: 4,
    replicaSeed: 101,
    rewired: [0, 1, 2, 3].map((seed) => ({ seed, ...graph(seed) })),
    biological: [101, 202, 303].map((trainerSeed) => ({ trainerSeed, ...graph(trainerSeed / 100) })),
    host: { arch: 'arm64', node: 'v22.22.3' },
    d: 48,
    bigqMergeCommit: '69b610d4a9da11b12a7ac180997e702cf9fd2a4f',
    evaluatorGitRev: 'deadbeefdeadbeefdeadbeefdeadbeefdeadbeef',
    cemConfig: { population: 128, elites: 32, generations: 150, alpha: 0.7, stdFloor: 0.02, initStd: 0.5, trainingSeedsPerGeneration: 16 },
    cemConfigWarnings: [],
    ...overrides
  };
};

const writeTrainedReadoutManifest = (path: string, gpuRerunFitnessDelta = 6.354025749714424): void => {
  writeFileSync(path, JSON.stringify({ gpuRerunFitnessDelta }));
};

describe('parseNullReportArgs', () => {
  it('applies defaults, with shards left undefined (resolved later from the run-meta sidecar)', () => {
    const args = parseNullReportArgs([]);
    expect(args.shards).toBeUndefined();
    expect(args.bootstrapResamples).toBe(10000);
    expect(args.histogramBins).toBe(30);
  });

  it('parses an explicit --shards override', () => {
    const args = parseNullReportArgs(['--shards', '12']);
    expect(args.shards).toBe(12);
  });

  it('rejects an unknown flag', () => {
    expect(() => parseNullReportArgs(['--bogus'])).toThrow(/Unknown argument/);
  });

  it('rejects --authored without a .json extension', () => {
    // Regression test for a dual-review finding, mirroring null-evaluate.ts's
    // equivalent --out guard: resolveRunMeta derives its sidecar path by
    // stripping a trailing ".json" off --authored.
    expect(() => parseNullReportArgs(['--authored', 'authored'])).toThrow(/--authored must end with "\.json"/);
  });
});

describe('resolveRunMeta', () => {
  let root: string;
  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'null-report-runmeta-'));
  });
  afterEach(() => rmSync(root, { recursive: true, force: true }));

  const baseArgs = (authored: string, shards?: number): NullReportArgs => ({
    authored,
    trained: join(root, 'trained.json'),
    trainedReadoutManifest: join(root, 'trained-readout-manifest.json'),
    out: join(root, 'out.json'),
    reportMd: join(root, 'out.md'),
    manifest: join(root, 'manifest.json'),
    bootstrapSeed: 1,
    bootstrapResamples: 200,
    histogramBins: 5,
    shards
  });

  it('throws when no --shards is given and no sidecar exists', () => {
    const authored = join(root, 'authored.json');
    writeFileSync(authored, '{}');
    expect(() => resolveRunMeta(baseArgs(authored))).toThrow(/cannot determine shard count/);
  });

  it('reads shards and timing from the <authored>.run.json sidecar', () => {
    const authored = join(root, 'authored.json');
    writeFileSync(authored, '{}');
    writeFileSync(join(root, 'authored.run.json'), JSON.stringify({ shards: 7, elapsedMs: 1234, perEpisodeMs: 5.6 }));
    const meta = resolveRunMeta(baseArgs(authored));
    expect(meta).toEqual({ shards: 7, elapsedMs: 1234, perEpisodeMs: 5.6 });
  });

  it('an explicit --shards overrides the sidecar shard count but keeps its timing', () => {
    const authored = join(root, 'authored.json');
    writeFileSync(authored, '{}');
    writeFileSync(join(root, 'authored.run.json'), JSON.stringify({ shards: 7, elapsedMs: 1234, perEpisodeMs: 5.6 }));
    const meta = resolveRunMeta(baseArgs(authored, 3));
    expect(meta).toEqual({ shards: 3, elapsedMs: 1234, perEpisodeMs: 5.6 });
  });
});

describe('buildArtifact', () => {
  const args: NullReportArgs = {
    authored: 'authored.json',
    trained: 'trained.json',
    trainedReadoutManifest: 'trained-readout-manifest.json',
    out: 'out.json',
    reportMd: 'out.md',
    manifest: 'manifest.json',
    bootstrapSeed: 1,
    bootstrapResamples: 200,
    histogramBins: 5,
    shards: 3
  };
  const runMeta = { shards: 3 };

  it('throws on an unsupported version', () => {
    expect(() => buildArtifact(buildRaw({ version: 2 as 1 }), args, runMeta)).toThrow(/unsupported version/);
  });

  it('throws when biological/disconnected are missing', () => {
    const raw = buildRaw();
    // @ts-expect-error -- deliberately constructing an invalid raw file (calibration-only, no --biological).
    delete raw.biological;
    expect(() => buildArtifact(raw, args, runMeta)).toThrow(/no biological\/disconnected section/);
  });

  it('throws when a graph was scored on different held-out seeds than biological', () => {
    const raw = buildRaw();
    const tampered = { ...raw, disconnected: { ...raw.disconnected!, heldOutSeeds: [1, 2, 3] } };
    expect(() => buildArtifact(tampered, args, runMeta)).toThrow(/different held-out seeds/);
  });

  it('throws on a non-finite score anywhere in a hand-edited authored.json', () => {
    // Regression test for a dual-review finding: null-worker.ts refuses to
    // *produce* a non-finite score, but authored.json is a plain file that
    // could be hand-edited (or merged from an older/buggy evaluator) after
    // the fact -- without this check, a NaN/Infinity would round-trip
    // through JSON.stringify as `null` and then be silently summed as 0.
    const raw = buildRaw();
    const tampered = {
      ...raw,
      rewired: raw.rewired.map((entry) => (entry.seed === 2 ? { ...entry, movementScore: [1, NaN, 3] } : entry))
    };
    expect(() => buildArtifact(tampered, args, runMeta)).toThrow(/rewired-2\.movementScore\[1\] is not a finite number/);
  });

  it('throws an actionable error when rewired seed 0 is missing', () => {
    const raw = buildRaw();
    const withoutSeed0 = { ...raw, rewired: raw.rewired.filter((entry) => entry.seed !== 0) };
    expect(() => buildArtifact(withoutSeed0, args, runMeta)).toThrow(/rewired seed 0 is missing.*found 4 seed\(s\)/);
  });

  it("the histogram's bin counts sum to the null set size, not the null set plus biological/disconnected", () => {
    // Regression test for a dual-review finding: an earlier version counted
    // biological and disconnected into `bins.counts` too (sum === 502 for
    // the real 500-graph run instead of 500).
    const artifact = buildArtifact(buildRaw(), args, runMeta);
    const total = artifact.bins.counts.reduce((sum, c) => sum + c, 0);
    expect(total).toBe(artifact.rewired.length);
    expect(total).toBe(5);
  });

  it('shards comes from runMeta, not from a hard-coded default', () => {
    const artifact = buildArtifact(buildRaw(), args, { shards: 42 });
    expect(artifact.shards).toBe(42);
  });

  it('includes a timing section only when runMeta provides it', () => {
    const withoutTiming = buildArtifact(buildRaw(), args, { shards: 1 });
    expect(withoutTiming.timing).toBeUndefined();
    const withTiming = buildArtifact(buildRaw(), args, { shards: 1, elapsedMs: 100, perEpisodeMs: 2 });
    expect(withTiming.timing).toEqual({ elapsedMs: 100, perEpisodeMs: 2 });
  });
});

describe('runNullReport', () => {
  let root: string;
  let authoredPath: string;
  let manifestPath: string;
  let outPath: string;
  let reportMdPath: string;
  let args: NullReportArgs;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'null-report-run-'));
    authoredPath = join(root, 'authored.json');
    manifestPath = join(root, 'manifest.json');
    outPath = join(root, 'rewiring-null-v1.json');
    reportMdPath = join(root, 'rewiring-null-report.md');

    const raw = buildRaw();
    writeFileSync(authoredPath, JSON.stringify(raw));
    writeTestManifest(manifestPath, raw.sourceGraphSha256);

    args = {
      authored: authoredPath,
      trained: join(root, 'trained.json'), // does not exist -> --trained is a no-op
      trainedReadoutManifest: join(root, 'trained-readout-manifest.json'), // only read when --trained's file exists
      out: outPath,
      reportMd: reportMdPath,
      manifest: manifestPath,
      bootstrapSeed: 1,
      bootstrapResamples: 200,
      histogramBins: 5,
      shards: 4
    };
  });

  afterEach(() => rmSync(root, { recursive: true, force: true }));

  it('running it twice produces byte-identical artifact, markdown, and manifest', () => {
    const first = runNullReport(args);
    const artifactAfterFirst = readFileSync(outPath);
    const mdAfterFirst = readFileSync(reportMdPath);
    const manifestAfterFirst = readFileSync(manifestPath);

    const second = runNullReport(args);
    expect(readFileSync(outPath).equals(artifactAfterFirst)).toBe(true);
    expect(readFileSync(reportMdPath).equals(mdAfterFirst)).toBe(true);
    expect(readFileSync(manifestPath).equals(manifestAfterFirst)).toBe(true);
    expect(second.artifactSha256).toBe(first.artifactSha256);
  });

  it('the manifest diff touches only the rewiringNull key', () => {
    const before = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
    runNullReport(args);
    const after = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
    const { rewiringNull, ...afterWithoutNewKey } = after;
    expect(rewiringNull).toBeDefined();
    expect(afterWithoutNewKey).toEqual(before);
  });

  it("refuses to overwrite a default (shipped) path when the input isn't a real 500-graph run", () => {
    expect(() => runNullReport({ ...args, out: DEFAULT_OUT })).toThrow(/refusing to overwrite the shipped/);
    expect(() => runNullReport({ ...args, reportMd: DEFAULT_REPORT_MD })).toThrow(/refusing to overwrite the shipped/);
    expect(() => runNullReport({ ...args, manifest: DEFAULT_MANIFEST })).toThrow(/refusing to overwrite the shipped/);
  });

  it('refuses to publish when the authored graph sha does not match the manifest being updated', () => {
    writeTestManifest(manifestPath, HEX64('f')); // a different graph than raw.sourceGraphSha256
    expect(() => runNullReport(args)).toThrow(/was scored against a graph with sha256/);
  });

  it('refuses to publish (before writing anything) when the manifest cannot round-trip byte-identically', () => {
    // Regression test for a dual-review finding: the manifest round-trip
    // safety check must run as a preflight, before the artifact is written
    // -- not only inside updateManifestWithRewiringNull (called last), which
    // would leave a new rewiring-null-v1.json on disk with no manifest entry
    // pointing at it if the check failed there instead.
    // Hand-written text (not JSON.stringify'd): a JS number can't hold the
    // distinction between "1" and "1.0", so this has to be literal text on
    // disk to reproduce the Python-vs-JS formatting mismatch the check guards against.
    writeFileSync(
      manifestPath,
      `{\n  "artifact": "test.bin.gz",\n  "binarySha256": "${buildRaw().sourceGraphSha256}",\n  "value": 1.0\n}\n`
    );
    expect(() => runNullReport(args)).toThrow(/produced different bytes/);
    expect(() => readFileSync(outPath)).toThrow(); // nothing was written
  });

  describe('with --trained (WP3 trained section)', () => {
    let trainedArgs: NullReportArgs;

    beforeEach(() => {
      writeFileSync(args.trained, JSON.stringify(buildTrainedRaw()));
      writeTrainedReadoutManifest(args.trainedReadoutManifest);
      trainedArgs = args;
    });

    it('merges a trained section into the published artifact and report', () => {
      const result = runNullReport(trainedArgs);
      expect(result.artifact.trained).toBeDefined();
      const trained = result.artifact.trained!;
      expect(trained.rewired).toHaveLength(4);
      expect(trained.biological).toHaveLength(3);
      expect(trained.replicaSeed).toBe(101);
      expect(trained.d).toBe(48);
      expect(trained.bigqMergeCommit).toBe('69b610d4a9da11b12a7ac180997e702cf9fd2a4f');
      expect(trained.percentileResolution).toBeCloseTo(0.25, 12); // 1/4 rewired replicas in this fixture
      expect(trained.bigqGpuRerunFitnessDelta).toBeCloseTo(6.354025749714424, 10);
      expect(trained.bioTrainerSeedSpread.label).toMatch(/trainer-noise variance/);

      const reportMd = readFileSync(reportMdPath, 'utf8');
      expect(reportMd).toContain('## Trained-readout sample');
      expect(reportMd).toContain('69b610d4a9da11b12a7ac180997e702cf9fd2a4f');
    });

    it('running it twice with --trained present is still byte-identical', () => {
      const first = runNullReport(trainedArgs);
      const artifactAfterFirst = readFileSync(outPath);
      const mdAfterFirst = readFileSync(reportMdPath);

      const second = runNullReport(trainedArgs);
      expect(readFileSync(outPath).equals(artifactAfterFirst)).toBe(true);
      expect(readFileSync(reportMdPath).equals(mdAfterFirst)).toBe(true);
      expect(second.artifactSha256).toBe(first.artifactSha256);
    });

    it('omits the trained section entirely when --trained does not exist (unchanged from before WP3)', () => {
      const noTrained = { ...trainedArgs, trained: join(root, 'does-not-exist.json') };
      const result = runNullReport(noTrained);
      expect(result.artifact.trained).toBeUndefined();
      expect(readFileSync(reportMdPath, 'utf8')).not.toContain('## Trained-readout sample');
    });
  });

  it('does not write authored.run.json as part of publishing (that sidecar is null-evaluate.ts\'s output)', () => {
    runNullReport(args);
    expect(() => readFileSync(join(root, 'authored.run.json'))).toThrow();
  });
});

describe('verifyManifestRoundTrips: round-trip safety', () => {
  // `updateManifestWithRewiringNull` no longer calls `verifyManifestRoundTrips`
  // itself -- `runNullReport` already runs it as a preflight, before
  // anything is written (see the "refuses to publish (before writing
  // anything) when the manifest cannot round-trip byte-identically" test
  // above), so calling it again inside `updateManifestWithRewiringNull` was
  // pure duplicated work on every successful publish (a thermo-nuclear
  // maintainability finding). This block now tests the exported
  // `verifyManifestRoundTrips` function directly, which is the contract any
  // other caller of `updateManifestWithRewiringNull` must uphold itself.
  it('refuses (throws) if re-serializing the unmodified manifest is not byte-identical to the file on disk', () => {
    const root = mkdtempSync(join(tmpdir(), 'null-report-manifest-safety-'));
    const manifestPath = join(root, 'manifest.json');
    // A float value: JS's JSON.stringify writes `1` for `1.0`, which
    // Python's json.dumps would not -- this manifest was NOT written by
    // this module's own sortKeysDeep+stringify convention, so the round
    // trip must fail closed rather than silently rewrite it.
    writeFileSync(manifestPath, '{\n  "binarySha256": "abc",\n  "value": 1.0\n}\n');
    expect(() => verifyManifestRoundTrips(manifestPath)).toThrow(/re-serializing .* produced different bytes/);
    rmSync(root, { recursive: true, force: true });
  });

  it('does not throw when the manifest already matches the sorted-keys/2-space-indent convention', () => {
    const root = mkdtempSync(join(tmpdir(), 'null-report-manifest-ok-'));
    const manifestPath = join(root, 'manifest.json');
    writeFileSync(manifestPath, '{\n  "artifact": "x.bin.gz",\n  "binarySha256": "abc"\n}\n');
    expect(() => verifyManifestRoundTrips(manifestPath)).not.toThrow();
    rmSync(root, { recursive: true, force: true });
  });

  it('updateManifestWithRewiringNull itself writes normally given an already-conformant manifest (no preflight call of its own)', () => {
    const root = mkdtempSync(join(tmpdir(), 'null-report-manifest-update-'));
    const manifestPath = join(root, 'manifest.json');
    writeFileSync(manifestPath, '{\n  "artifact": "x.bin.gz",\n  "binarySha256": "abc"\n}\n');
    updateManifestWithRewiringNull(manifestPath, { artifact: 'x.json', sha256: 'y' });
    const written = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
    expect(written.rewiringNull).toEqual({ artifact: 'x.json', sha256: 'y' });
    rmSync(root, { recursive: true, force: true });
  });

  it('updateManifestWithRewiringNull called directly (bypassing the caller-owned preflight) does NOT throw on a non-round-trippable manifest -- it trusts the caller per its doc comment', () => {
    // Locks in the contract this function's doc comment now states: unlike
    // before, this function no longer independently guards against
    // rewriting unrelated bytes -- any caller that skips
    // `verifyManifestRoundTrips` itself (as `runNullReport` no longer does,
    // since it already ran the check as its own preflight) gets a silent
    // reformat of the whole file, not a thrown error. This test exists so a
    // future accidental re-introduction of the internal check (undoing this
    // maintainability fix, or silently changing this function's contract
    // back) shows up as a failing assertion rather than passing unnoticed.
    const root = mkdtempSync(join(tmpdir(), 'null-report-manifest-unguarded-'));
    const manifestPath = join(root, 'manifest.json');
    const originalText = '{\n  "binarySha256": "abc",\n  "value": 1.0\n}\n';
    writeFileSync(manifestPath, originalText);
    expect(() => updateManifestWithRewiringNull(manifestPath, { artifact: 'x.json', sha256: 'y' })).not.toThrow();
    const rewritten = readFileSync(manifestPath, 'utf8');
    // The float `1.0` was silently reformatted to `1` (JS's JSON.stringify
    // cannot preserve the distinction) -- exactly the "silently rewrite
    // unrelated manifest bytes" risk `verifyManifestRoundTrips`'s doc
    // comment warns about, now only prevented by callers that check first.
    expect(rewritten).not.toBe(originalText);
    const written = JSON.parse(rewritten) as Record<string, unknown>;
    expect(written.rewiringNull).toEqual({ artifact: 'x.json', sha256: 'y' });
    expect(written.value).toBe(1);
    rmSync(root, { recursive: true, force: true });
  });
});
