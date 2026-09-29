import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
  buildSelectionRobustnessArtifact,
  parseSelectionReportArgs,
  runSelectionReport,
  SELECTION_IDS,
  selectionRobustnessProducer,
  type SelectionId
} from '../../scripts/selections/selection-report';

/**
 * WP3 of `.agents/plans/selection-robustness`: unit coverage for
 * `scripts/selections/selection-report.ts`, mirroring
 * `tests/unit/task-generality-report.test.ts`'s own "synthetic, fully
 * self-contained fixture world" discipline -- every input is a real file
 * written to a temp directory, not a hand-parsed in-memory object, so the
 * producer's own `readFileSync`/`JSON.parse` calls are exercised for real.
 *
 * `EXPECTED_GRAPH_SHA_PREFIX` (this module's own safety-net constant) means
 * every fixture's `binarySha256` must start with the real, pinned prefix
 * for that selection id -- the fixtures below use that exact prefix
 * followed by zero-padding, never an arbitrary sha, for exactly that
 * reason.
 */

/** `sorted(rewired scores)[quantileIndex(8, 0.25)] === sorted[2]`, for the fixed 8-point fixture below -- `scripts/null/null-stats.ts`'s own low-tail-floor convention. */
const REWIRED_SCORES = [1, 2, 3, 4, 5, 6, 7, 8];
const FIXTURE_P25 = 3;

const GRAPH_SHA_PREFIX: Readonly<Record<SelectionId, string>> = {
  larger: '6a6304aa',
  smaller: '47992d46',
  'random-bridge': '34076335',
  'alt-sensory-mapping': '25d48a35'
};

interface SelectionFixtureOptions {
  readonly bioScore?: number;
  readonly nullDegenerate?: boolean;
  readonly replicates?: boolean;
  readonly structuralReplicates?: boolean;
  readonly pScore?: number;
  readonly cP95?: number;
  readonly mP95?: number;
  readonly cDegenerate?: boolean;
  readonly mDegenerate?: boolean;
  readonly k?: number;
  readonly maxSwaps?: number;
  readonly targetReached?: boolean;
  readonly coverageFlagged?: boolean;
  readonly withSingleAxis?: boolean;
}

/** Every field an "unremarkable, everything holds/replicates/supports" selection needs -- individual tests override just the fields relevant to what they're probing. */
const defaultOptions = (): Required<Omit<SelectionFixtureOptions, 'withSingleAxis'>> & { withSingleAxis: boolean } => ({
  bioScore: 0,
  nullDegenerate: false,
  replicates: true,
  structuralReplicates: true,
  pScore: 10,
  cP95: 1,
  mP95: 1,
  cDegenerate: false,
  mDegenerate: false,
  k: 2,
  maxSwaps: 200,
  targetReached: true,
  coverageFlagged: false,
  withSingleAxis: false
});

/** Writes one selection's complete, internally-consistent set of WP2 outputs under `dir/<id>/`. */
const writeSelectionFixture = (dir: string, id: SelectionId, options: SelectionFixtureOptions = {}): void => {
  const opts = { ...defaultOptions(), ...options };
  const selDir = join(dir, id);
  mkdirSync(join(selDir, 'interventions'), { recursive: true });

  const graphSha = `${GRAPH_SHA_PREFIX[id]}${'0'.repeat(64 - GRAPH_SHA_PREFIX[id].length)}`;

  const ledger: Record<string, unknown> = {
    binarySha256: graphSha,
    selection: { id, params: { seed: 1 } },
    selectionCounts: { bridgeCandidateCount: 2557, bridgeSelectedCount: 800 },
    dynamics: { globalGain: 0.0005 },
    compiledFromGitRevision: 'f'.repeat(40),
    compilerSourceSha256: 'a'.repeat(64)
  };
  if (id === 'smaller') {
    ledger.sourceFiles = [{ sha256: 'b'.repeat(64) }, { sha256: 'c'.repeat(64) }, { sha256: 'd'.repeat(64) }];
  }
  writeFileSync(join(selDir, `malecns-arena-${id}.ledger.json`), JSON.stringify(ledger));

  const rewiringNull = {
    rewired: REWIRED_SCORES.map((score) => ({ score })),
    null: { degenerate: opts.nullDegenerate },
    biological: { score: opts.bioScore },
    bioPercentile: opts.bioScore < FIXTURE_P25 ? 0.1 : 0.9
  };
  writeFileSync(join(selDir, 'rewiring-null.json'), JSON.stringify(rewiringNull));

  const metrics = [
    {
      name: 'T:rightClearance->thrust',
      spearman: opts.replicates ? 0.5 : 0.1,
      bioPercentile: opts.replicates ? 0.01 : 0.5
    },
    { name: 'T:forwardClearance->thrust', spearman: 0.1, bioPercentile: 0.5 },
    {
      name: 'weightedInDegree:thrust',
      spearman: opts.structuralReplicates ? 0.4 : 0.1,
      bioPercentile: opts.structuralReplicates ? 0.01 : 0.5
    }
  ];
  const variants: Record<string, unknown> = { flipBoth: { bioPercentile: 0.1 } };
  if (opts.withSingleAxis) {
    variants.flipThrust = { bioPercentile: 0.94 };
    variants.flipYaw = { bioPercentile: 0.81 };
  }
  const nullExplanation = {
    metrics,
    thresholds: { spearmanRho: 0.3 },
    variants,
    exploratoryOmittedReason: 'selection-mode has no exploratory-unrestricted counterpart'
  };
  writeFileSync(join(selDir, 'null-explanation.json'), JSON.stringify(nullExplanation));

  const coverage = { perChannel: [], perPopulation: [], flagged: opts.coverageFlagged };
  writeFileSync(join(selDir, 'coverage.json'), JSON.stringify(coverage));

  const interventionStats = {
    p: { score: opts.pScore },
    q: { score: opts.pScore, channelSpecific: true },
    controls: { C: { p95: opts.cP95 }, M: { p95: opts.mP95 } },
    armDegeneracy: {
      cArm: { degenerate: opts.cDegenerate },
      mArm: { degenerate: opts.mDegenerate },
      mqArm: { degenerate: false }
    },
    publishedNullFloor: FIXTURE_P25
  };
  writeFileSync(join(selDir, 'intervention-stats.json'), JSON.stringify(interventionStats));

  const interventionsIndex = {
    kP: opts.k,
    kQ: opts.k,
    maxSwaps: opts.maxSwaps,
    entries: [
      { id: 'P', swaps: opts.k, targetReached: opts.targetReached },
      { id: 'Q', swaps: opts.k, targetReached: opts.targetReached }
    ]
  };
  writeFileSync(join(selDir, 'interventions', 'index.json'), JSON.stringify(interventionsIndex));
};

/** Writes an "everything holds/replicates/is supported" world for all four selections -- the baseline every individual test overrides one selection's options against. */
const writeAllUnremarkable = (dir: string): void => {
  for (const id of SELECTION_IDS) writeSelectionFixture(dir, id);
};

describe('selectionRobustnessProducer', () => {
  it('is deterministic across two calls', () => {
    expect(selectionRobustnessProducer().sourceSha256).toBe(selectionRobustnessProducer().sourceSha256);
  });

  it('includes this file and the markdown renderer in its own dependency closure', () => {
    const deps = selectionRobustnessProducer().dependencies;
    expect(deps).toContain('scripts/selections/selection-report.ts');
    expect(deps).toContain('scripts/selections/selection-report-markdown.ts');
    expect(deps).toContain('scripts/null/null-stats.ts');
  });
});

describe('buildSelectionRobustnessArtifact (pure builder, against a synthetic fixture world)', () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'selection-report-'));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it('is byte-identical across two calls against the same inputs', () => {
    writeAllUnremarkable(dir);
    const first = JSON.stringify(buildSelectionRobustnessArtifact({ selectionsDir: dir, defaultGraphSha: 'z'.repeat(64) }));
    const second = JSON.stringify(buildSelectionRobustnessArtifact({ selectionsDir: dir, defaultGraphSha: 'z'.repeat(64) }));
    expect(second).toBe(first);
  });

  it('an all-unremarkable world is robust to size, robust to method, and mapping holds', () => {
    writeAllUnremarkable(dir);
    const artifact = buildSelectionRobustnessArtifact({ selectionsDir: dir, defaultGraphSha: 'z'.repeat(64) });
    expect(artifact.overall.robustToSize.verdict).toBe(true);
    expect(artifact.overall.robustToMethod.verdict).toBe(true);
    expect(artifact.overall.mapping.verdict).toBe(true);
    for (const selection of artifact.selections) {
      expect(selection.categorized).toBe(true);
      expect(selection.null.holds).toBe(true);
      expect(selection.explanation.replicates).toBe(true);
      expect(selection.pathway.supported).toBe(true);
    }
  });

  it("random-bridge's explanation failing alone makes robust-to-method false, even though its pathway stays categorized", () => {
    writeAllUnremarkable(dir);
    writeSelectionFixture(dir, 'random-bridge', { replicates: false });
    const artifact = buildSelectionRobustnessArtifact({ selectionsDir: dir, defaultGraphSha: 'z'.repeat(64) });
    const randomBridge = artifact.selections.find((s) => s.id === 'random-bridge');
    expect(randomBridge?.explanation.replicates).toBe(false);
    expect(randomBridge?.categorized).toBe(true);
    expect(artifact.overall.robustToMethod.verdict).toBe(false);
    expect(artifact.overall.robustToMethod.reason).toContain('explanation does not replicate');
  });

  it('a degenerate pathway (cDegenerate) makes the selection uncategorized, with the mechanism stated, and the axis result indeterminate when nothing else fails', () => {
    writeAllUnremarkable(dir);
    writeSelectionFixture(dir, 'random-bridge', { cDegenerate: true, mDegenerate: true, k: 0 });
    const artifact = buildSelectionRobustnessArtifact({ selectionsDir: dir, defaultGraphSha: 'z'.repeat(64) });
    const randomBridge = artifact.selections.find((s) => s.id === 'random-bridge');
    expect(randomBridge?.categorized).toBe(false);
    expect(randomBridge?.categorizedReason).toContain('0 swaps');
    expect(randomBridge?.pathway.degenerateMechanism).toContain('0 swaps');
    expect(artifact.overall.robustToMethod.verdict).toBe('indeterminate');
  });

  it("alt-sensory-mapping's pathway failing alone (not-supported) makes the mapping verdict false", () => {
    writeAllUnremarkable(dir);
    writeSelectionFixture(dir, 'alt-sensory-mapping', { pScore: 0 });
    const artifact = buildSelectionRobustnessArtifact({ selectionsDir: dir, defaultGraphSha: 'z'.repeat(64) });
    const altSensoryMapping = artifact.selections.find((s) => s.id === 'alt-sensory-mapping');
    expect(altSensoryMapping?.pathway.supported).toBe(false);
    expect(altSensoryMapping?.categorized).toBe(true);
    expect(artifact.overall.mapping.verdict).toBe(false);
  });

  it('a P that hits the swap cap without reaching its target is reported search-limited, never folded into "not-supported"', () => {
    writeAllUnremarkable(dir);
    writeSelectionFixture(dir, 'larger', { k: 200, maxSwaps: 200, targetReached: false, pScore: 0 });
    const artifact = buildSelectionRobustnessArtifact({ selectionsDir: dir, defaultGraphSha: 'z'.repeat(64) });
    const larger = artifact.selections.find((s) => s.id === 'larger');
    expect(larger?.pathway.searchLimited).toBe(true);
    expect(larger?.pathway.targetReached).toBe(false);
  });

  it("random-bridge's single-axis variants are carried through only when present, null otherwise", () => {
    writeAllUnremarkable(dir);
    writeSelectionFixture(dir, 'random-bridge', { withSingleAxis: true });
    const artifact = buildSelectionRobustnessArtifact({ selectionsDir: dir, defaultGraphSha: 'z'.repeat(64) });
    const randomBridge = artifact.selections.find((s) => s.id === 'random-bridge');
    const larger = artifact.selections.find((s) => s.id === 'larger');
    expect(randomBridge?.explanation.singleAxis).toEqual({ flipThrust: 0.94, flipYaw: 0.81 });
    expect(larger?.explanation.singleAxis).toBeNull();
  });

  it('a coverage-flagged selection is not categorized', () => {
    writeAllUnremarkable(dir);
    writeSelectionFixture(dir, 'smaller', { coverageFlagged: true });
    const artifact = buildSelectionRobustnessArtifact({ selectionsDir: dir, defaultGraphSha: 'z'.repeat(64) });
    const smaller = artifact.selections.find((s) => s.id === 'smaller');
    expect(smaller?.categorized).toBe(false);
    expect(smaller?.categorizedReason).toMatch(/zero bridge coverage/);
  });

  it('refuses to publish when a selection\'s graph sha does not start with the expected pinned prefix', () => {
    writeAllUnremarkable(dir);
    const ledgerPath = join(dir, 'larger', 'malecns-arena-larger.ledger.json');
    const ledger = JSON.parse(readFileSync(ledgerPath, 'utf8')) as Record<string, unknown>;
    ledger.binarySha256 = 'ffffffff' + '0'.repeat(56);
    writeFileSync(ledgerPath, JSON.stringify(ledger));
    expect(() => buildSelectionRobustnessArtifact({ selectionsDir: dir, defaultGraphSha: 'z'.repeat(64) })).toThrow(
      /does not start with the expected/
    );
  });

  it("refuses to publish when the recomputed p25 disagrees with intervention-stats.json's publishedNullFloor", () => {
    writeAllUnremarkable(dir);
    const statsPath = join(dir, 'larger', 'intervention-stats.json');
    const stats = JSON.parse(readFileSync(statsPath, 'utf8')) as Record<string, unknown>;
    stats.publishedNullFloor = 999;
    writeFileSync(statsPath, JSON.stringify(stats));
    expect(() => buildSelectionRobustnessArtifact({ selectionsDir: dir, defaultGraphSha: 'z'.repeat(64) })).toThrow(
      /disagrees with intervention-stats\.json's publishedNullFloor/
    );
  });
});

describe('parseSelectionReportArgs', () => {
  it('refuses --out and --report-md pointing at the same path', () => {
    expect(() => parseSelectionReportArgs(['--out', '/tmp/x.json', '--report-md', '/tmp/x.json'])).toThrow(
      /--out and --report-md must not be the same path/
    );
  });

  it('rejects an unknown flag', () => {
    expect(() => parseSelectionReportArgs(['--bogus', 'value'])).toThrow(/Unknown argument/);
  });
});

describe('runSelectionReport (CLI layer)', () => {
  let dir: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'selection-report-cli-'));
  });
  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  const writeManifest = (): string => {
    const manifestPath = join(dir, 'manifest.json');
    writeFileSync(manifestPath, JSON.stringify({ binarySha256: 'z'.repeat(64) }));
    return manifestPath;
  };

  it('writes a byte-identical artifact across two runs, and a report containing the predeclared rules verbatim', () => {
    const selectionsDir = join(dir, 'selections');
    mkdirSync(selectionsDir, { recursive: true });
    writeAllUnremarkable(selectionsDir);
    const manifestPath = writeManifest();

    const args = parseSelectionReportArgs([
      '--selections-dir',
      selectionsDir,
      '--manifest',
      manifestPath,
      '--out',
      join(dir, 'out1.json'),
      '--report-md',
      join(dir, 'report1.md'),
      '--skip-manifest-update'
    ]);
    const first = runSelectionReport(args);

    const secondArgs = parseSelectionReportArgs([
      '--selections-dir',
      selectionsDir,
      '--manifest',
      manifestPath,
      '--out',
      join(dir, 'out2.json'),
      '--report-md',
      join(dir, 'report2.md'),
      '--skip-manifest-update'
    ]);
    const second = runSelectionReport(secondArgs);

    expect(readFileSync(first.out, 'utf8')).toBe(readFileSync(second.out, 'utf8'));
    expect(first.artifactSha256).toBe(second.artifactSha256);
    const report = readFileSync(first.reportMdPath, 'utf8');
    expect(report).toContain('biological is below that selection\'s rewired-null 25th percentile');
    expect(report).toContain('Search-budget disclosure');
    expect(report).toMatch(/no biological claim/);
  });

  it('updates the manifest in place with the artifact entry, unless --skip-manifest-update is given', () => {
    const selectionsDir = join(dir, 'selections');
    mkdirSync(selectionsDir, { recursive: true });
    writeAllUnremarkable(selectionsDir);
    const manifestPath = writeManifest();
    // `verifyManifestRoundTrips` requires the file to already be in the
    // sortKeysDeep + 2-space-indent + trailing-newline form.
    writeFileSync(manifestPath, `${JSON.stringify({ binarySha256: 'z'.repeat(64) }, null, 2)}\n`);

    const args = parseSelectionReportArgs([
      '--selections-dir',
      selectionsDir,
      '--manifest',
      manifestPath,
      '--out',
      join(dir, 'out.json'),
      '--report-md',
      join(dir, 'report.md')
    ]);
    const result = runSelectionReport(args);

    const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as { selectionRobustness?: { artifact: string; sha256: string } };
    expect(manifest.selectionRobustness).toEqual({ artifact: 'out.json', sha256: result.artifactSha256 });
  });
});
