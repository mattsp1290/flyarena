// @vitest-environment node
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

const here = dirname(fileURLToPath(import.meta.url));

import {
  buildRepertoireNullArtifact,
  parseRepertoireReportArgs,
  renderRepertoireNullReportMarkdown,
  runRepertoireReport,
  repertoireNullProducer,
  type BuildArtifactInputs,
  type RepertoireNullArtifact
} from '../../scripts/atlas/repertoire-report';
import { occupied, qd, rewiredDistribution, metricVerdict, categorize, robustness } from '../../scripts/atlas/repertoire-metrics';
import { PRIMARY_SEARCH_SEED, EXTRA_SEARCH_SEEDS, REWIRED_COUNT, REWIRED_SEED_MATCHED_COUNT } from '../../scripts/atlas/repertoire-plan';
import type { RepertoireEvaluatedArtifact } from '../../scripts/atlas/repertoire-evaluate';
import type { RepertoireEvaluatedEntry } from '../../scripts/atlas/repertoire-task';
import { sha256Hex } from '../../scripts/training/fsio';
import { CELL_COUNT, DISCOVERY_SEEDS, HELDOUT_SEEDS, type SearchArtifact } from '../../src/lib/atlas/types';

/**
 * Coverage for `scripts/atlas/repertoire-report.ts` -- WP3's top-level
 * artifact/report/manifest producer. `buildRepertoireNullArtifact` is
 * exercised directly against a small synthetic `evaluated.json`-shaped
 * fixture (no filesystem, no GPU search) covering the exact planned 46
 * `(graph, seed)` pairs (`repertoire-plan.ts`'s own enumeration); the CLI
 * layer (`runRepertoireReport`) is exercised against a real scratch
 * directory, mirroring `tests/unit/intervention-artifact.test.ts`'s own
 * "CLI layer against a real scratch directory" pattern -- neither depends
 * on the real, gitignored `training/runs/repertoire/evaluated.json` (a
 * multi-hour GPU-search product this project's own CI never has), so this
 * suite runs in a fresh checkout with no GPU inputs at all.
 */

const SHA = (label: string): string => label.padEnd(64, '0');

const SEARCH_OPTIONS = { population: 4, generations: 1, ticks: 30 };

const makeCell = (cell: number, quality: number, heldoutMean: number) => ({
  cell,
  quality,
  coverage: 0,
  turning: 0,
  heldoutOwn: [{ foodPickups: 0, hazardContacts: 0, distanceTravelled: 0, movementScore: heldoutMean, coverage: 0, turning: 0 }]
});

/** `qd = sum(max(0, quality))` -- fixing every cell's quality at 10 makes `qd = occupiedCount * 10`, simple enough to hand-verify without leaning on `repertoire-metrics.ts`'s own (separately tested) arithmetic. */
const cellsFor = (occupiedCount: number, heldoutMean = 5): ReturnType<typeof makeCell>[] =>
  Array.from({ length: occupiedCount }, (_, i) => makeCell(i, 10, heldoutMean));

const entry = (
  graphId: string,
  arm: RepertoireEvaluatedEntry['arm'],
  rewiringSeed: number | null,
  searchSeed: number,
  occupiedCount: number
): RepertoireEvaluatedEntry => ({
  graphId,
  arm,
  rewiringSeed,
  searchSeed,
  searchOptions: { ...SEARCH_OPTIONS, seed: searchSeed },
  gpuArchiveSize: occupiedCount,
  occupied: occupiedCount,
  collisions: 0,
  cells: cellsFor(occupiedCount)
});

/** Distinct, deterministic occupied-cell counts per biological search seed -- arbitrary but fixed, so every test computes the same expected numbers by hand. */
const BIO_OCCUPIED: Readonly<Record<number, number>> = { 1729: 4, 1730: 3, 1731: 5, 1732: 2, 1733: 6 };

/** Deterministic, non-degenerate occupied-cell count per (rewiring seed, search seed) pair -- cycles 2..6. */
const rewiredOccupiedCount = (rewiringSeed: number, searchSeed: number): number => 2 + ((rewiringSeed * 7 + searchSeed) % 5);

const buildFixtureGraphs = (): RepertoireEvaluatedEntry[] => {
  const entries: RepertoireEvaluatedEntry[] = [];
  for (const seed of [PRIMARY_SEARCH_SEED, ...EXTRA_SEARCH_SEEDS]) {
    entries.push(entry('biological', 'biological', null, seed, BIO_OCCUPIED[seed]));
  }
  entries.push(entry('disconnected', 'disconnected', null, PRIMARY_SEARCH_SEED, 3));
  for (let rewiringSeed = 0; rewiringSeed < REWIRED_COUNT; rewiringSeed += 1) {
    const seeds = rewiringSeed < REWIRED_SEED_MATCHED_COUNT ? [PRIMARY_SEARCH_SEED, ...EXTRA_SEARCH_SEEDS] : [PRIMARY_SEARCH_SEED];
    for (const seed of seeds) {
      entries.push(entry(`rewired-${rewiringSeed}`, 'rewired', rewiringSeed, seed, rewiredOccupiedCount(rewiringSeed, seed)));
    }
  }
  return entries;
};

const fixtureEvaluated = (graphs: RepertoireEvaluatedEntry[] = buildFixtureGraphs()): RepertoireEvaluatedArtifact => ({
  schemaVersion: 1,
  graphs
});

/**
 * Only `options`/`discoverySeeds`/`heldoutSeeds` are ever read by
 * `buildRepertoireNullArtifact` -- the cast below stands in for the rest of
 * `SearchArtifact`'s real, much larger shape (candidates, history, runtime,
 * ...), which no test here needs a real value for.
 */
const fixtureAtlas = (): { readonly source: SearchArtifact } =>
  ({
    source: {
      options: { seed: PRIMARY_SEARCH_SEED, ...SEARCH_OPTIONS },
      discoverySeeds: DISCOVERY_SEEDS,
      heldoutSeeds: HELDOUT_SEEDS
    }
  }) as unknown as { readonly source: SearchArtifact };

const fixtureRewiringNull = () => ({ sourceGraphSha256: SHA('bio') });

const fixtureGraphsIndex = () => ({ sourceSha256: SHA('bio'), seeds: Array.from({ length: REWIRED_COUNT }, (_, seed) => ({ seed })) });

/** A complete, self-consistent `BuildArtifactInputs` -- every cross-check the builder performs passes against this fixture unless a test deliberately perturbs one field. */
const fixtureInputs = (overrides: Partial<BuildArtifactInputs> = {}): BuildArtifactInputs => {
  const evaluated = overrides.evaluated ?? fixtureEvaluated();
  const evaluatedBytes = overrides.evaluatedBytes ?? Buffer.from(JSON.stringify(evaluated));
  const rewiringNullParsed = overrides.rewiringNullParsed ?? fixtureRewiringNull();
  const rewiringNullBytes = overrides.rewiringNullBytes ?? Buffer.from(JSON.stringify(rewiringNullParsed));
  const atlasParsed = overrides.atlasParsed ?? fixtureAtlas();
  const atlasBytes = overrides.atlasBytes ?? Buffer.from(JSON.stringify(atlasParsed));
  const graphsIndexParsed = overrides.graphsIndexParsed ?? fixtureGraphsIndex();
  return {
    evaluated,
    evaluatedBytes,
    manifestBiologicalSha: overrides.manifestBiologicalSha ?? SHA('bio'),
    manifestRewiringNullSha: overrides.manifestRewiringNullSha ?? sha256Hex(rewiringNullBytes),
    rewiringNullBytes,
    rewiringNullParsed,
    atlasBytes,
    atlasParsed,
    graphsIndexBytes: overrides.graphsIndexBytes ?? Buffer.from(JSON.stringify(graphsIndexParsed)),
    graphsIndexParsed
  };
};

describe('buildRepertoireNullArtifact', () => {
  it('computes primary/robustness/disconnected identically to independently re-deriving them from the same fixture', () => {
    const artifact = buildRepertoireNullArtifact(fixtureInputs());

    // Primary (seed 1729): the same computation, done independently here
    // from the identical fixture values, not merely re-asserting whatever
    // the builder happened to produce.
    const rewiredAt1729 = Array.from({ length: REWIRED_COUNT }, (_, r) => rewiredOccupiedCount(r, PRIMARY_SEARCH_SEED));
    const occDist = rewiredDistribution(rewiredAt1729);
    const qdDist = rewiredDistribution(rewiredAt1729.map((n) => n * 10));
    const category = categorize(
      metricVerdict(BIO_OCCUPIED[PRIMARY_SEARCH_SEED], occDist),
      metricVerdict(BIO_OCCUPIED[PRIMARY_SEARCH_SEED] * 10, qdDist)
    );
    expect(artifact.primary.bio.occupied).toBe(BIO_OCCUPIED[PRIMARY_SEARCH_SEED]);
    expect(artifact.primary.bio.qd).toBe(BIO_OCCUPIED[PRIMARY_SEARCH_SEED] * 10);
    expect(artifact.primary.rewiredDistribution.occupied).toEqual(occDist);
    expect(artifact.primary.rewiredDistribution.qd).toEqual(qdDist);
    expect(artifact.primary.category).toBe(category.category);
    expect(artifact.primary.tie).toBe(category.tie);

    // Robustness: every seed's category, independently re-derived.
    const perSeed = [PRIMARY_SEARCH_SEED, ...EXTRA_SEARCH_SEEDS].map((seed) => {
      const rewiredAtSeed = Array.from(
        { length: seed === PRIMARY_SEARCH_SEED ? REWIRED_COUNT : REWIRED_SEED_MATCHED_COUNT },
        (_, r) => rewiredOccupiedCount(r, seed)
      );
      const seedOccDist = rewiredDistribution(rewiredAtSeed);
      const seedQdDist = rewiredDistribution(rewiredAtSeed.map((n) => n * 10));
      const seedCategory = categorize(
        metricVerdict(BIO_OCCUPIED[seed], seedOccDist),
        metricVerdict(BIO_OCCUPIED[seed] * 10, seedQdDist)
      ).category;
      return { seed, category: seedCategory, n: rewiredAtSeed.length };
    });
    const expectedRobustness = robustness(perSeed.map(({ seed, category: c }) => ({ seed, category: c })));
    expect(artifact.robustness.perSeed).toEqual(expectedRobustness.perSeed);
    expect(artifact.robustness.robust).toBe(expectedRobustness.robust);
    for (const { seed, n } of perSeed) expect(artifact.robustness.seedMatchedCounts[seed]).toBe(n);

    // Disconnected: reported, never categorized.
    expect(artifact.disconnected.occupied).toBe(3);
    expect(artifact.disconnected.qd).toBe(30);

    // Every graph summary carries its occupied cell indices, ascending.
    const bio1729 = artifact.graphs.find((g) => g.id === 'biological' && g.seed === PRIMARY_SEARCH_SEED);
    expect(bio1729?.cells).toEqual([0, 1, 2, 3]);

    // Sources/search/host provenance.
    expect(artifact.sources.biologicalSha).toBe(SHA('bio'));
    expect(artifact.search.population).toBe(SEARCH_OPTIONS.population);
    expect(artifact.search.rewiredCount).toBe(REWIRED_COUNT);
    expect(artifact.search.rewiredSeedMatchedCount).toBe(REWIRED_SEED_MATCHED_COUNT);
    expect(artifact.host).toEqual({ arch: process.arch, node: process.version });
    expect(artifact.sources.producer.script).toBe('scripts/atlas/repertoire-report.ts');
    expect(artifact.sources.producer.sourceSha256).toMatch(/^[0-9a-f]{64}$/);
  });

  it('rejects an evaluated.json missing one of the planned 46 (graph, seed) pairs', () => {
    const graphs = buildFixtureGraphs();
    graphs.pop();
    expect(() => buildRepertoireNullArtifact(fixtureInputs({ evaluated: fixtureEvaluated(graphs) }))).toThrow(
      /does not cover exactly the planned 46/
    );
  });

  it('rejects an evaluated.json with an unexpected extra (graph, seed) pair', () => {
    const graphs = buildFixtureGraphs();
    graphs.push(entry('rewired-19', 'rewired', 19, 1730, 3));
    expect(() => buildRepertoireNullArtifact(fixtureInputs({ evaluated: fixtureEvaluated(graphs) }))).toThrow(
      /does not cover exactly the planned 46/
    );
  });

  it('rejects a duplicated (graphId, searchSeed) row, even though the coverage Set alone would collapse it', () => {
    const graphs = buildFixtureGraphs();
    graphs.push({ ...graphs[graphs.length - 1] }); // a real duplicate: rewired-19@1729, appended a second time
    expect(() => buildRepertoireNullArtifact(fixtureInputs({ evaluated: fixtureEvaluated(graphs) }))).toThrow(/duplicated: rewired-19@1729/);
  });

  it('rejects an entry whose arm/rewiringSeed disagree with its own graphId', () => {
    const graphs = buildFixtureGraphs();
    const index = graphs.findIndex((g) => g.graphId === 'rewired-3' && g.searchSeed === PRIMARY_SEARCH_SEED);
    graphs[index] = { ...graphs[index], arm: 'biological', rewiringSeed: null };
    expect(() => buildRepertoireNullArtifact(fixtureInputs({ evaluated: fixtureEvaluated(graphs) }))).toThrow(
      /rewired-3@1729 has arm=biological\/rewiringSeed=null, expected arm=rewired\/rewiringSeed=3/
    );
  });

  it('rejects a graphs/index.json whose sourceSha256 does not match manifest.binarySha256', () => {
    expect(() =>
      buildRepertoireNullArtifact(fixtureInputs({ graphsIndexParsed: { ...fixtureGraphsIndex(), sourceSha256: SHA('other') } }))
    ).toThrow(/graphs\/index\.json sourceSha256 .* does not match manifest\.binarySha256/);
  });

  it('rejects a graphs/index.json with the wrong number of rewired seeds', () => {
    expect(() =>
      buildRepertoireNullArtifact(fixtureInputs({ graphsIndexParsed: { sourceSha256: SHA('bio'), seeds: [] } }))
    ).toThrow(/graphs\/index\.json has 0 rewired graph\(s\), expected 20/);
  });

  it('rejects a cell list whose recorded occupied count disagrees with cells.length', () => {
    const graphs = buildFixtureGraphs();
    graphs[0] = { ...graphs[0], occupied: 999 };
    expect(() => buildRepertoireNullArtifact(fixtureInputs({ evaluated: fixtureEvaluated(graphs) }))).toThrow(
      /inconsistent, duplicated, or out-of-range cell list/
    );
  });

  it('rejects a cell list with an out-of-range cell index', () => {
    const graphs = buildFixtureGraphs();
    graphs[0] = { ...graphs[0], cells: [...graphs[0].cells.slice(0, -1), makeCell(CELL_COUNT, 10, 5)] };
    expect(() => buildRepertoireNullArtifact(fixtureInputs({ evaluated: fixtureEvaluated(graphs) }))).toThrow(
      /inconsistent, duplicated, or out-of-range cell list/
    );
  });

  it('rejects a search options mismatch against the shipped atlas budget', () => {
    const graphs = buildFixtureGraphs();
    graphs[0] = { ...graphs[0], searchOptions: { ...graphs[0].searchOptions, ticks: 99999 } };
    expect(() => buildRepertoireNullArtifact(fixtureInputs({ evaluated: fixtureEvaluated(graphs) }))).toThrow(
      /search options diverge/
    );
  });

  it('rejects a rewiring-null sha256 that does not match the manifest', () => {
    expect(() => buildRepertoireNullArtifact(fixtureInputs({ manifestRewiringNullSha: SHA('wrong') }))).toThrow(
      /does not match the manifest's rewiringNull\.sha256/
    );
  });

  it('rejects a rewiring-null sourceGraphSha256 that does not match manifest.binarySha256', () => {
    expect(() =>
      buildRepertoireNullArtifact(fixtureInputs({ rewiringNullParsed: { sourceGraphSha256: SHA('other') } }))
    ).toThrow(/does not match the published null's sourceGraphSha256/);
  });

  it('rejects a shipped atlas whose discoverySeeds/heldoutSeeds no longer match src/lib/atlas/types.ts', () => {
    const badDiscovery = fixtureAtlas();
    (badDiscovery.source as { discoverySeeds: number[] }).discoverySeeds = [1, 2, 3];
    expect(() => buildRepertoireNullArtifact(fixtureInputs({ atlasParsed: badDiscovery }))).toThrow(/discoverySeeds no longer match/);

    const badHeldout = fixtureAtlas();
    (badHeldout.source as { heldoutSeeds: number[] }).heldoutSeeds = [1, 2, 3];
    expect(() => buildRepertoireNullArtifact(fixtureInputs({ atlasParsed: badHeldout }))).toThrow(/heldoutSeeds no longer match/);
  });

  it('rejects an unsupported evaluated.json schemaVersion', () => {
    const evaluated = { ...fixtureEvaluated(), schemaVersion: 2 as unknown as 1 };
    expect(() => buildRepertoireNullArtifact(fixtureInputs({ evaluated }))).toThrow(/unsupported schemaVersion/);
  });

  it('produces a degenerate ("tie") rewired distribution when every rewired occupied/qd value is identical', () => {
    // Force every rewired graph at every seed to the same occupied count as
    // biological -- a fully degenerate p25 = p75 = bio on both metrics.
    const graphs = buildFixtureGraphs().map((g) =>
      g.arm === 'rewired' ? { ...g, occupied: BIO_OCCUPIED[g.searchSeed], cells: cellsFor(BIO_OCCUPIED[g.searchSeed]) } : g
    );
    const artifact = buildRepertoireNullArtifact(fixtureInputs({ evaluated: fixtureEvaluated(graphs) }));
    expect(artifact.primary.category).toBe('typical');
    expect(artifact.primary.tie).toBe(true);
  });
});

describe('renderRepertoireNullReportMarkdown', () => {
  it('states the predeclared categories verbatim and includes every graph in the audit table', () => {
    const artifact = buildRepertoireNullArtifact(fixtureInputs());
    const markdown = renderRepertoireNullReportMarkdown(artifact);
    expect(markdown).toContain('**Wider repertoire:** biological is at or above the rewired 75th percentile');
    expect(markdown).toContain('**Narrower repertoire:** biological is at or below the 25th percentile');
    expect(markdown).toContain(`Category at seed ${PRIMARY_SEARCH_SEED}: ${artifact.primary.category}`);
    expect(markdown).toContain(`Robust across all ${1 + EXTRA_SEARCH_SEEDS.length} biological search seeds: ${artifact.robustness.robust ? 'yes' : 'no'}`);
    for (const g of artifact.graphs.filter((entry) => entry.arm !== 'disconnected')) {
      expect(markdown).toContain(`| ${g.id} | ${g.seed} |`);
    }
    expect(markdown).toContain(`| ${artifact.disconnected.occupied} | ${artifact.disconnected.qd.toFixed(2)} |`);
    expect(markdown).toContain('disconnected graph may produce degenerate behavior');
    expect(markdown).toContain('no biological claim about fly behavior');
  });

  it('states the seed-matched sample\'s percentile resolution correctly (20% for a 5-point sample, never the primary\'s 5%)', () => {
    const artifact = buildRepertoireNullArtifact(fixtureInputs());
    const markdown = renderRepertoireNullReportMarkdown(artifact);
    // A dual review (Important) caught an earlier version stating "a 5%
    // percentile resolution" for the 5-point seed-matched sample (it is
    // 20%); assert both the Method section and the Limitations section
    // state it correctly, and never repeat the wrong figure.
    expect(markdown).toContain('that sample has only 5 points (a 20% percentile');
    expect(markdown).toContain('a 5% percentile resolution; only');
    expect(markdown).toContain('(a 20% percentile resolution at those seeds)');
    expect(markdown).not.toContain('5 points (a 5%');
  });

  it('renders the occupancy maps as real GitHub-flavored Markdown tables, with row/column labels', () => {
    const artifact = buildRepertoireNullArtifact(fixtureInputs());
    const markdown = renderRepertoireNullReportMarkdown(artifact);
    const occupancySection = markdown.slice(markdown.indexOf('### Occupancy maps'), markdown.indexOf('### Disconnected control'));
    // A dual review (Important) found this grid rendered with no header row
    // and no `| --- |` delimiter row, so it showed up as literal pipes on
    // GitHub -- both are now required.
    expect(occupancySection).toContain('| turning \\ coverage |');
    expect(occupancySection.match(/\| --- \|( --- \|){6}/g)?.length).toBe(2); // one delimiter row per grid
    expect(occupancySection).toContain('0.67..1.00'); // a real TURN_EDGES row label, not a bare row index
  });
});

describe('repertoireNullProducer', () => {
  it('names this file as the producer and hashes over its real import-graph closure', () => {
    const producer = repertoireNullProducer();
    expect(producer.script).toBe('scripts/atlas/repertoire-report.ts');
    expect(producer.sourceSha256).toMatch(/^[0-9a-f]{64}$/);
    expect(producer.dependencies).toContain('scripts/atlas/repertoire-report.ts');
    expect(producer.dependencies).toContain('scripts/atlas/repertoire-metrics.ts');
    expect(producer.dependencies).toContain('scripts/atlas/repertoire-plan.ts');
  });

  it('reproduces the identical sourceSha256 on a second call (deterministic, no filesystem race)', () => {
    expect(repertoireNullProducer().sourceSha256).toBe(repertoireNullProducer().sourceSha256);
  });

  /**
   * A dual review (Important): a prior regeneration in this branch's own
   * history left the committed `behavior-repertoire-null-v1.json` stamped
   * with an *earlier* revision's `producer.sourceSha256`, and nothing
   * caught it -- the artifact's whole contract is that `sources.producer`
   * identifies the exact code that produced it. This guard fails loudly the
   * next time a producer-dependency edit lands without a matching
   * regeneration, instead of silently shipping a stale stamp.
   */
  it('the committed artifact was produced by the producer at HEAD', () => {
    const committed = JSON.parse(readFileSync(resolve(here, '../../public/data/behavior-repertoire-null-v1.json'), 'utf8')) as {
      sources: { producer: { sourceSha256: string } };
    };
    expect(committed.sources.producer.sourceSha256).toBe(repertoireNullProducer().sourceSha256);
  });
});

describe('parseRepertoireReportArgs', () => {
  it('rejects --out overwriting an input file', () => {
    expect(() => parseRepertoireReportArgs(['--out', '--evaluated', 'x.json'])).toThrow(); // malformed flag value
    expect(() =>
      parseRepertoireReportArgs(['--evaluated', '/tmp/e.json', '--out', '/tmp/e.json'])
    ).toThrow(/--out must not overwrite an input file/);
  });

  it('rejects --report-md overwriting an input file', () => {
    expect(() =>
      parseRepertoireReportArgs(['--rewiring-null', '/tmp/n.json', '--report-md', '/tmp/n.json'])
    ).toThrow(/--report-md must not overwrite an input file/);
  });

  it('rejects --out and --report-md being the same path', () => {
    expect(() =>
      parseRepertoireReportArgs(['--out', '/tmp/same.json', '--report-md', '/tmp/same.json'])
    ).toThrow(/--out and --report-md must not be the same path/);
  });

  it('rejects --out in a different directory than --manifest (the manifest records only --out\'s basename)', () => {
    // A dual review (Important): `--out /tmp/x.json` against the default
    // `--manifest` would otherwise silently repoint the shipped manifest's
    // `behaviorRepertoireNull.artifact` at a file that does not exist next
    // to it.
    expect(() =>
      parseRepertoireReportArgs(['--manifest', '/tmp/manifest-dir/manifest.json', '--out', '/tmp/other-dir/out.json'])
    ).toThrow(/--out must be in the same directory as --manifest/);
  });

  it('accepts --out in the same directory as --manifest', () => {
    const args = parseRepertoireReportArgs(['--manifest', '/tmp/manifest-dir/manifest.json', '--out', '/tmp/manifest-dir/out.json']);
    expect(args.out).toBe(resolve('/tmp/manifest-dir/out.json'));
  });
});

describe('runRepertoireReport (CLI layer, real scratch directory)', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'repertoire-report-'));
  });

  afterEach(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  const writeFixtureFiles = (): { evaluated: string; graphsIndex: string; rewiringNull: string; atlas: string; manifest: string } => {
    const evaluated = resolve(root, 'evaluated.json');
    const graphsIndex = resolve(root, 'index.json');
    const rewiringNull = resolve(root, 'rewiring-null-v1.json');
    const atlas = resolve(root, 'behavior-atlas-v1.json');
    const manifest = resolve(root, 'manifest.json');

    writeFileSync(evaluated, JSON.stringify(fixtureEvaluated()));
    writeFileSync(graphsIndex, JSON.stringify(fixtureGraphsIndex()));
    const rewiringNullContents = JSON.stringify(fixtureRewiringNull());
    writeFileSync(rewiringNull, rewiringNullContents);
    writeFileSync(atlas, JSON.stringify(fixtureAtlas()));
    // Sorted-keys, 2-space-indent, trailing-newline -- the exact convention
    // `verifyManifestRoundTrips` (`scripts/null/null-report.ts`) requires the
    // real manifest to already be in, so its preflight round-trip check
    // passes against this fixture too.
    writeFileSync(
      manifest,
      `${JSON.stringify(
        { binarySha256: SHA('bio'), rewiringNull: { sha256: sha256Hex(Buffer.from(rewiringNullContents)) } },
        null,
        2
      )}\n`
    );
    return { evaluated, graphsIndex, rewiringNull, atlas, manifest };
  };

  it('writes the artifact and report, updates the manifest, and refuses to overwrite an input', () => {
    const files = writeFixtureFiles();
    const out = resolve(root, 'behavior-repertoire-null-v1.json');
    const reportMd = resolve(root, 'behavior-repertoire-null-report.md');

    const result = runRepertoireReport({ ...files, out, reportMd });

    expect(result.artifactSha256).toMatch(/^[0-9a-f]{64}$/);
    const writtenBytes = readFileSync(out, 'utf8');
    expect(sha256Hex(Buffer.from(writtenBytes))).toBe(result.artifactSha256);

    const manifest = JSON.parse(readFileSync(files.manifest, 'utf8')) as { behaviorRepertoireNull?: { artifact: string; sha256: string } };
    expect(manifest.behaviorRepertoireNull).toEqual({
      artifact: 'behavior-repertoire-null-v1.json',
      sha256: result.artifactSha256
    });

    const reportText = readFileSync(reportMd, 'utf8');
    expect(reportText).toContain('# Behavior repertoire, biological vs. degree-preserving rewirings');
  });

  // The "refuses to overwrite an input" guard lives in `parseRepertoireReportArgs`
  // (the CLI-args layer -- see that describe block above), not in
  // `runRepertoireReport` itself, mirroring `pathwayInterventions.ts`'s
  // identical split between `parseIntoArtifactArgs`'s guard and
  // `runPathwayInterventionsArtifact`'s own unchecked run function. This
  // test exercises the guard through the real CLI argv path end-to-end.
  it('the real CLI argv path refuses to let --out overwrite an input file', () => {
    const files = writeFixtureFiles();
    expect(() =>
      parseRepertoireReportArgs([
        '--evaluated',
        files.evaluated,
        '--graphs-index',
        files.graphsIndex,
        '--rewiring-null',
        files.rewiringNull,
        '--atlas',
        files.atlas,
        '--manifest',
        files.manifest,
        '--out',
        files.evaluated
      ])
    ).toThrow(/must not overwrite an input file/);
  });

  it('regenerating from the same inputs is byte-identical, including re-running against the manifest the first run already rewrote', () => {
    const files = writeFixtureFiles();
    const outA = resolve(root, 'a.json');
    const reportA = resolve(root, 'a-report.md');
    const resultA = runRepertoireReport({ ...files, out: outA, reportMd: reportA });

    // Re-run against the *same* manifest file `runRepertoireReport` already
    // rewrote once (`updateManifestWithRepertoireNull` always fully
    // overwrites its own `behaviorRepertoireNull` key, and
    // `verifyManifestRoundTrips`'s preflight only requires the file to
    // already be in canonical sorted-keys form, which run A's own write
    // leaves it in) -- proving regeneration is stable even when the
    // manifest is not pristine, not just from a fresh copy.
    const outB = resolve(root, 'b.json');
    const reportB = resolve(root, 'b-report.md');
    const resultB = runRepertoireReport({ ...files, out: outB, reportMd: reportB });

    expect(readFileSync(outA, 'utf8')).toBe(readFileSync(outB, 'utf8'));
    expect(resultA.artifactSha256).toBe(resultB.artifactSha256);
    expect(readFileSync(reportA, 'utf8')).toBe(readFileSync(reportB, 'utf8'));
  });
});
