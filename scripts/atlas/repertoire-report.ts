import { mkdirSync, readFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import type { ArmName } from '../training/arms';
import { collectRepoRelativeDependencies, computeSourceIdentitySha256 } from '../lib/import-graph';
import { sortKeysDeep, verifyManifestRoundTrips } from '../null/null-report';
import {
  occupied,
  qd,
  span,
  heldoutOwnMedian,
  rewiredDistribution,
  metricVerdict,
  categorize,
  robustness,
  type RepertoireCategory,
  type RewiredDistribution,
  type RepertoireCellMetricInput
} from './repertoire-metrics';
import type { RepertoireEvaluatedArtifact } from './repertoire-evaluate';
import { PRIMARY_SEARCH_SEED, EXTRA_SEARCH_SEEDS, REWIRED_COUNT, REWIRED_SEED_MATCHED_COUNT } from './repertoire-plan';
import { ATLAS_FILE, CELL_COUNT, DISCOVERY_SEEDS, HELDOUT_SEEDS, type SearchArtifact } from '../../src/lib/atlas/types';
// `renderRepertoireNullReportMarkdown` lives in `./repertoire-report-markdown.ts`
// (thermo-maintainability review, Suggestion S1 -- see that file's own doc
// comment for why it was split out). Imported and re-exported here so every
// existing caller/import site (the CLI's own `runRepertoireReport` below,
// and `tests/unit/repertoire-report.test.ts`) keeps working unchanged.
import { renderRepertoireNullReportMarkdown } from './repertoire-report-markdown';
export { renderRepertoireNullReportMarkdown };

/**
 * WP3 of `.agents/plans/repertoire-null` (`03-artifact-and-atlas-strip.md`):
 * combines WP2's `evaluated.json` (`repertoire-evaluate.ts`'s
 * `RepertoireEvaluatedArtifact`, already TS-re-evaluated and re-binned --
 * nothing here re-runs a search or an episode) with the shipped biological
 * manifest, the published rewiring-null distribution, and the shipped
 * behavior atlas (to independently pin the search budget this study reused)
 * into `public/data/behavior-repertoire-null-v1.json` (the
 * `behaviorRepertoireNull` manifest key) and
 * `docs/behavior-repertoire-null-report.md`.
 *
 * Follows `scripts/null/intervention-artifact.ts`'s own precedent
 * (WP4 of `.agents/plans/pathway-interventions`) directly: a pure,
 * side-effect-free `buildRepertoireNullArtifact` over already-read
 * bytes/already-parsed objects (so `runRepertoireReport` can hash the exact
 * bytes it also parses, and so the builder is independently unit-testable
 * against synthetic fixtures with no filesystem at all), `sortKeysDeep`
 * re-serialization of the manifest via `null-report.ts`'s shared helpers,
 * and a `producer.sourceSha256` stamped over the real, walked import-graph
 * closure from this file (`scripts/lib/import-graph.ts`), the same scheme
 * `regime-check.ts`/`intervention-artifact.ts` already established.
 *
 * Deterministic and re-run-safe by construction: every input is read once
 * from disk, `sortKeysDeep`+`JSON.stringify` never iterate a `Map`/`Set`
 * when producing array output, `graphs` is sorted by a fixed, locale-free
 * comparator (`graphOrder` below -- never `String#localeCompare`, whose
 * collation can depend on the host's ICU data), `robustness.perSeed` is
 * built by iterating the plan's own fixed seed ranges, and nothing here
 * reads the clock -- running this CLI twice against the same inputs on the
 * *same host* (same `process.arch`/`process.version`, both stamped into
 * `host` below and hashed as part of the artifact) produces byte-identical
 * `behavior-repertoire-null-v1.json` bytes (`00-overview.md`'s
 * "Regeneration on the arm64 Spark is byte-identical" acceptance
 * criterion). A dual review (Important/Suggestion) flagged both caveats:
 * an earlier version's `graphs` ordering depended on `localeCompare`, and
 * `host` was, and remains, one of this artifact's own hashed inputs -- the
 * byte-identical claim is scoped to a fixed host precisely because
 * `evaluated.json` (WP2's own output) carries no host of its own to thread
 * through instead.
 */

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

const DEFAULT_EVALUATED = resolve(repoRoot, 'training/runs/repertoire/evaluated.json');
const DEFAULT_GRAPHS_INDEX = resolve(repoRoot, 'training/runs/repertoire/graphs/index.json');
const DEFAULT_REWIRING_NULL = resolve(repoRoot, 'public/data/rewiring-null-v1.json');
const DEFAULT_ATLAS = resolve(repoRoot, 'public/data', ATLAS_FILE);
export const DEFAULT_MANIFEST = resolve(repoRoot, 'public/data/malecns-arena-v1.manifest.json');
export const DEFAULT_OUT = resolve(repoRoot, 'public/data/behavior-repertoire-null-v1.json');
export const DEFAULT_REPORT_MD = resolve(repoRoot, 'docs/behavior-repertoire-null-report.md');

// ---------------------------------------------------------------------------
// Producer code identity
// ---------------------------------------------------------------------------

export interface RepertoireNullProducer {
  readonly script: string;
  readonly sourceSha256: string;
  readonly dependencies: readonly string[];
}

/**
 * This run's TS code-identity block, hashed over the *real, walked* import
 * graph from this file (`scripts/lib/import-graph.ts`'s
 * `collectRepoRelativeDependencies`) -- pulls in every module this file
 * actually imports (`repertoire-metrics.ts`, `repertoire-evaluate.ts`,
 * `repertoire-plan.ts`, `null-report.ts` for the manifest helpers, and
 * everything *they* import in turn), not a hand-maintained filename list.
 * Matches `pathwayInterventionsProducer()`'s own doc comment and this
 * bean's own "producer sourceSha256 over the import-graph closure"
 * instruction.
 */
export const repertoireNullProducer = (): RepertoireNullProducer => {
  const dependencies = collectRepoRelativeDependencies(fileURLToPath(import.meta.url), repoRoot);
  return {
    script: 'scripts/atlas/repertoire-report.ts',
    sourceSha256: computeSourceIdentitySha256(repoRoot, dependencies),
    dependencies
  };
};

// ---------------------------------------------------------------------------
// Artifact shape
// ---------------------------------------------------------------------------

export interface RepertoireNullSources {
  /** `manifest.binarySha256` -- the biological graph's compiled binary sha256, cross-checked against the published rewiring null's own recorded source-graph sha. */
  readonly biologicalSha: string;
  /** `manifest.rewiringNull.sha256`, cross-checked against the actual bytes of `public/data/rewiring-null-v1.json`. */
  readonly rewiringNullSha: string;
  /** sha256 of `training/runs/repertoire/evaluated.json`'s exact bytes -- the one file every number in `graphs` below is derived from. */
  readonly evaluatedSha256: string;
  /** sha256 of the shipped `behavior-atlas-v1.json`, whose `source.options`/`discoverySeeds`/`heldoutSeeds` this study's own search budget was independently checked to reuse (`search` below). */
  readonly atlasSha256: string;
  /** sha256 of `training/runs/repertoire/graphs/index.json` (the rewired-graph batch WP2 searched) -- gitignored, never shipped, recorded here only as build-time provenance. */
  readonly graphsIndexSha256: string;
  readonly producer: RepertoireNullProducer;
}

/** `00-overview.md`'s "Reuse the shipped search exactly" decision, restated with its provenance: every field here is cross-checked against the shipped atlas's own `source.options`/`discoverySeeds`/`heldoutSeeds`, never hand-typed. */
export interface RepertoireNullSearchConfig {
  readonly population: number;
  readonly generations: number;
  readonly ticks: number;
  readonly discoverySeeds: readonly number[];
  readonly heldoutSeeds: readonly number[];
  readonly primarySearchSeed: number;
  readonly extraSearchSeeds: readonly number[];
  readonly rewiredCount: number;
  readonly rewiredSeedMatchedCount: number;
}

/** One searched graph's summary: the predeclared metrics (`repertoire-metrics.ts`) plus the audit fields, and the occupied cell indices (for an occupancy map), trimmed of every raw per-candidate discovery/held-out payload `evaluated.json` itself still carries. */
export interface RepertoireGraphSummary {
  /** `biological` | `disconnected` | `rewired-<N>` -- matches `RepertoireEvaluatedEntry.graphId`. */
  readonly id: string;
  readonly arm: ArmName;
  readonly rewiringSeed: number | null;
  /** The search seed this graph was searched and re-evaluated at. */
  readonly seed: number;
  readonly occupied: number;
  readonly qd: number;
  readonly span: number;
  readonly heldoutOwnMedian: number;
  readonly gpuArchiveSize: number;
  readonly collisions: number;
  /** The occupied cell indices, ascending -- `cellIndex` per `00-overview.md`'s cell-packing convention (`cellFor`, `src/lib/atlas/types.ts`). */
  readonly cells: readonly number[];
}

/**
 * `00-overview.md`'s "Predeclared categories" section names a *single*
 * biological value being compared against a *single* rewired distribution
 * per metric, but the category itself is decided jointly from **both**
 * `occupied` and `qd` (`repertoire-metrics.ts#categorize`) -- `03`'s own
 * schema shorthand (`rewiredDistribution: {p25, p50, p75, values[]}`) names
 * only one distribution, which is ambiguous once two metrics are involved.
 * This artifact carries one `RewiredDistribution` per metric explicitly
 * (`occupied`/`qd`), rather than collapsing them, so a reader can see
 * exactly which metric's distribution decided which half of the verdict.
 */
export interface RepertoireNullMetricDistribution {
  readonly occupied: RewiredDistribution;
  readonly qd: RewiredDistribution;
}

export interface RepertoireNullPrimary {
  readonly bio: { readonly occupied: number; readonly qd: number; readonly span: number; readonly heldoutOwnMedian: number };
  readonly rewiredDistribution: RepertoireNullMetricDistribution;
  readonly category: RepertoireCategory;
  /** True if either metric's rewired distribution was degenerate and equal to the biological value (`repertoire-metrics.ts#categorize`'s own tie annotation). */
  readonly tie: boolean;
}

export interface RepertoireNullRobustness {
  readonly perSeed: Readonly<Record<number, RepertoireCategory>>;
  readonly robust: boolean;
  /** The rewired sample size the category at each seed was decided against -- `20` at `1729` (every rewiring), `5` at `1730..1733` (only rewirings `0..4`), so a reader never mistakes the coarser seeds for the same resolution as the primary one. */
  readonly seedMatchedCounts: Readonly<Record<number, number>>;
}

export interface RepertoireNullArtifact {
  readonly version: 1;
  readonly sources: RepertoireNullSources;
  readonly search: RepertoireNullSearchConfig;
  readonly graphs: readonly RepertoireGraphSummary[];
  readonly primary: RepertoireNullPrimary;
  readonly robustness: RepertoireNullRobustness;
  readonly disconnected: RepertoireGraphSummary;
  readonly host: { readonly arch: string; readonly node: string };
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

export interface BuildArtifactInputs {
  readonly evaluated: Readonly<RepertoireEvaluatedArtifact>;
  readonly evaluatedBytes: Buffer;
  readonly manifestBiologicalSha: string;
  readonly manifestRewiringNullSha: string;
  readonly rewiringNullBytes: Buffer;
  readonly rewiringNullParsed: { readonly sourceGraphSha256: string };
  readonly atlasBytes: Buffer;
  readonly atlasParsed: { readonly source: SearchArtifact };
  readonly graphsIndexBytes: Buffer;
  /**
   * `training/runs/repertoire/graphs/index.json`'s own recorded identity --
   * a dual-review finding (Important): this file used to be hashed only
   * (`sources.graphsIndexSha256`), with nothing ever checking its
   * `sourceSha256` against `manifestBiologicalSha` or its `seeds.length`
   * against `REWIRED_COUNT`. That left the published `biologicalSha`
   * asserted from the manifest alone, never bound to the actual rewired
   * batch the search consumed -- an index regenerated from a different
   * biological graph, or with a truncated rewiring count, would have
   * published without complaint.
   */
  readonly graphsIndexParsed: { readonly sourceSha256: string; readonly seeds: readonly unknown[] };
}

const summarize = (entry: {
  readonly graphId: string;
  readonly arm: ArmName;
  readonly rewiringSeed: number | null;
  readonly searchSeed: number;
  readonly occupied: number;
  readonly gpuArchiveSize: number;
  readonly collisions: number;
  readonly cells: readonly RepertoireCellMetricInput[];
}): RepertoireGraphSummary => {
  // A maintainability review (Suggestion) found nothing cross-checked
  // `entry.occupied` (`evaluated.json`'s own recorded count) against
  // `cells.length`, and nothing bounds- or distinctness-checked the raw
  // cell indices before they feed `occupancyFrequency`'s unchecked
  // `counts[cell] += 1` -- an out-of-range or duplicated cell index would
  // otherwise silently corrupt the published occupancy map or the
  // recomputed `occupied`/`span` metrics.
  const cellIds = entry.cells.map((cell) => cell.cell);
  if (
    entry.occupied !== cellIds.length ||
    new Set(cellIds).size !== cellIds.length ||
    cellIds.some((cell) => !Number.isInteger(cell) || cell < 0 || cell >= CELL_COUNT)
  ) {
    throw new Error(`repertoire-report: ${entry.graphId}@${entry.searchSeed} has an inconsistent, duplicated, or out-of-range cell list`);
  }
  return {
    id: entry.graphId,
    arm: entry.arm,
    rewiringSeed: entry.rewiringSeed,
    seed: entry.searchSeed,
    occupied: occupied(entry.cells),
    qd: qd(entry.cells),
    span: span(entry.cells),
    heldoutOwnMedian: heldoutOwnMedian(entry.cells),
    gpuArchiveSize: entry.gpuArchiveSize,
    collisions: entry.collisions,
    cells: [...cellIds].sort((a, b) => a - b)
  };
};

/**
 * The canonical `(graph, seed)` order (`repertoire-plan.ts`'s own
 * "biological ascending search seed, disconnected, then rewired ascending
 * rewiring seed then ascending search seed" convention) as a locale-free
 * sort key -- a dual review (Suggestion) found the previous version used
 * `String#localeCompare`, which both sorted lexically (`rewired-10` before
 * `rewired-2`) and depended on the host's ICU collation, exactly the kind
 * of hidden input this module's own "byte-identical regeneration" claim
 * rules out everywhere else.
 */
const graphOrder = (graph: Readonly<RepertoireGraphSummary>): number =>
  graph.arm === 'biological' ? -2 : graph.arm === 'disconnected' ? -1 : (graph.rewiringSeed ?? 0);

/**
 * Pure builder: every input is already-read bytes/already-parsed objects (no
 * `readFileSync` inside this function), mirroring
 * `buildPathwayInterventionsArtifact`'s own TOCTOU-avoidance discipline, so
 * `runRepertoireReport` below can hash the exact bytes it also parses, and
 * this function is directly unit-testable against synthetic fixtures.
 *
 * Cross-checks every input against every other, and against the exact
 * planned 46 `(graph, seed)` pairs (`repertoire-plan.ts`'s own
 * `PRIMARY_SEARCH_SEED`/`EXTRA_SEARCH_SEEDS`/`REWIRED_COUNT`/
 * `REWIRED_SEED_MATCHED_COUNT` constants -- the single source of truth WP2's
 * own driver used to produce `evaluated.json` in the first place) before
 * building anything, so a stale, truncated, or hand-edited `evaluated.json`
 * cannot silently publish an artifact that describes a different, or
 * incomplete, experiment than the one its `sources` claim.
 */
export const buildRepertoireNullArtifact = (inputs: Readonly<BuildArtifactInputs>): RepertoireNullArtifact => {
  const { evaluated } = inputs;
  if (evaluated.schemaVersion !== 1) {
    throw new Error(`repertoire-report: evaluated.json has unsupported schemaVersion ${String(evaluated.schemaVersion)}, expected 1`);
  }

  const rewiringNullSha = sha256Hex(inputs.rewiringNullBytes);
  if (rewiringNullSha !== inputs.manifestRewiringNullSha) {
    throw new Error(
      `repertoire-report: rewiring-null sha256 ${rewiringNullSha} does not match the manifest's rewiringNull.sha256 (${inputs.manifestRewiringNullSha}) -- stale manifest?`
    );
  }
  if (inputs.rewiringNullParsed.sourceGraphSha256 !== inputs.manifestBiologicalSha) {
    throw new Error(
      `repertoire-report: manifest.binarySha256 (${inputs.manifestBiologicalSha}) does not match the published null's sourceGraphSha256 (${inputs.rewiringNullParsed.sourceGraphSha256})`
    );
  }
  // A dual review (Important) found `graphs/index.json` was only ever
  // hashed (`sources.graphsIndexSha256`), never actually cross-checked --
  // so an index regenerated from a different biological graph, or with a
  // truncated rewiring batch, would have published `sources.biologicalSha`
  // as an unverified assertion rather than a value bound to what the
  // search actually consumed.
  if (inputs.graphsIndexParsed.sourceSha256 !== inputs.manifestBiologicalSha) {
    throw new Error(
      `repertoire-report: graphs/index.json sourceSha256 (${inputs.graphsIndexParsed.sourceSha256}) does not match manifest.binarySha256 (${inputs.manifestBiologicalSha})`
    );
  }
  if (inputs.graphsIndexParsed.seeds.length !== REWIRED_COUNT) {
    throw new Error(
      `repertoire-report: graphs/index.json has ${inputs.graphsIndexParsed.seeds.length} rewired graph(s), expected ${REWIRED_COUNT}`
    );
  }

  const { options: atlasOptions, discoverySeeds: atlasDiscoverySeeds, heldoutSeeds: atlasHeldoutSeeds } = inputs.atlasParsed.source;
  const seedsEqual = (a: readonly number[], b: readonly number[]): boolean => a.length === b.length && a.every((v, i) => v === b[i]);
  if (!seedsEqual(atlasDiscoverySeeds, DISCOVERY_SEEDS)) {
    throw new Error('repertoire-report: the shipped atlas discoverySeeds no longer match src/lib/atlas/types.ts DISCOVERY_SEEDS');
  }
  if (!seedsEqual(atlasHeldoutSeeds, HELDOUT_SEEDS)) {
    throw new Error('repertoire-report: the shipped atlas heldoutSeeds no longer match src/lib/atlas/types.ts HELDOUT_SEEDS');
  }

  // Expected planned (graph, seed) pairs -- the identical enumeration
  // `repertoire-plan.ts#buildRepertoirePlan` used, restated here as a plain
  // key set (never re-imported as a full plan, since that needs a live
  // rewire index/manifest this pure builder does not have) so this function
  // fails loudly on a truncated or duplicated `evaluated.json` rather than
  // silently reporting whatever subset happens to be present.
  const expectedKeys = new Set<string>();
  for (const seed of [PRIMARY_SEARCH_SEED, ...EXTRA_SEARCH_SEEDS]) expectedKeys.add(`biological@${seed}`);
  expectedKeys.add(`disconnected@${PRIMARY_SEARCH_SEED}`);
  for (let rewiringSeed = 0; rewiringSeed < REWIRED_COUNT; rewiringSeed += 1) {
    const seeds = rewiringSeed < REWIRED_SEED_MATCHED_COUNT ? [PRIMARY_SEARCH_SEED, ...EXTRA_SEARCH_SEEDS] : [PRIMARY_SEARCH_SEED];
    for (const seed of seeds) expectedKeys.add(`rewired-${rewiringSeed}@${seed}`);
  }
  // A dual review (Important) demonstrated that a `Set`-based coverage
  // check alone accepts a duplicated `(graphId, searchSeed)` row (it
  // collapses before the missing/unexpected comparison below): appending a
  // second `rewired-19@1729` was accepted and inflated the published
  // rewired sample to `n=22`. Built from the raw key *list*, not only the
  // deduplicated set, so a duplicate is caught before anything downstream
  // (the `graphs` array, `byGraphSeed`, every distribution) ever sees it.
  const actualKeyList = evaluated.graphs.map((entry) => `${entry.graphId}@${entry.searchSeed}`);
  const actualKeys = new Set(actualKeyList);
  const duplicated = [...new Set(actualKeyList.filter((key, index) => actualKeyList.indexOf(key) !== index))];
  const missing = [...expectedKeys].filter((key) => !actualKeys.has(key));
  const unexpected = [...actualKeys].filter((key) => !expectedKeys.has(key));
  if (missing.length > 0 || unexpected.length > 0 || duplicated.length > 0) {
    throw new Error(
      `repertoire-report: evaluated.json does not cover exactly the planned ${expectedKeys.size} (graph, seed) pairs` +
        (missing.length > 0 ? `; missing: ${missing.join(', ')}` : '') +
        (unexpected.length > 0 ? `; unexpected: ${unexpected.join(', ')}` : '') +
        (duplicated.length > 0 ? `; duplicated: ${duplicated.join(', ')}` : '')
    );
  }
  for (const entry of evaluated.graphs) {
    const mismatches = (['population', 'generations', 'ticks'] as const).filter(
      (key) => entry.searchOptions[key] !== atlasOptions[key]
    );
    if (mismatches.length > 0) {
      throw new Error(
        `repertoire-report: ${entry.graphId}@${entry.searchSeed} search options diverge from the shipped atlas budget: ${mismatches
          .map((key) => `${key} expected ${atlasOptions[key]}, got ${entry.searchOptions[key]}`)
          .join('; ')}`
      );
    }
    // A dual review (Important) demonstrated that nothing checked an
    // entry's `arm`/`rewiringSeed` against its own `graphId` -- the
    // coverage check above keys on `graphId` alone, while the null
    // distribution below selects by `entry.arm === 'rewired'`. Relabelling
    // `rewired-3`'s `arm` to `'biological'` passed the coverage check and
    // silently dropped the primary rewired sample to `n=19`, with no error.
    const rewiredMatch = /^rewired-(\d+)$/.exec(entry.graphId);
    const expectedArm: ArmName = rewiredMatch ? 'rewired' : (entry.graphId as ArmName);
    const expectedRewiringSeed = rewiredMatch ? Number(rewiredMatch[1]) : null;
    if (entry.arm !== expectedArm || entry.rewiringSeed !== expectedRewiringSeed) {
      throw new Error(
        `repertoire-report: ${entry.graphId}@${entry.searchSeed} has arm=${entry.arm}/rewiringSeed=${String(entry.rewiringSeed)}, ` +
          `expected arm=${expectedArm}/rewiringSeed=${String(expectedRewiringSeed)}`
      );
    }
  }

  const graphs = evaluated.graphs.map(summarize).sort((a, b) => graphOrder(a) - graphOrder(b) || a.seed - b.seed);

  const byGraphSeed = new Map(graphs.map((entry) => [`${entry.id}@${entry.seed}`, entry]));
  const bio = (seed: number): RepertoireGraphSummary => {
    const entry = byGraphSeed.get(`biological@${seed}`);
    if (!entry) throw new Error(`repertoire-report: missing biological@${seed} after coverage check`);
    return entry;
  };
  // Asserts the exact expected sample size at the point of use (not only
  // in the coverage check above) -- cheap, and catches both I-2- and
  // I-3-shaped bugs a second, independent way: any duplicate, mislabelled,
  // or missing rewired entry that somehow slipped past the checks above
  // would still change this count.
  const rewiredAt = (seed: number): readonly RepertoireGraphSummary[] => {
    const entries = graphs.filter((entry) => entry.arm === 'rewired' && entry.seed === seed);
    const expectedCount = seed === PRIMARY_SEARCH_SEED ? REWIRED_COUNT : REWIRED_SEED_MATCHED_COUNT;
    if (entries.length !== expectedCount) {
      throw new Error(`repertoire-report: expected ${expectedCount} rewired entries at seed ${seed}, got ${entries.length}`);
    }
    return entries;
  };

  const primaryBio = bio(PRIMARY_SEARCH_SEED);
  const primaryRewired = rewiredAt(PRIMARY_SEARCH_SEED);
  const primaryOccupiedDist = rewiredDistribution(primaryRewired.map((entry) => entry.occupied));
  const primaryQdDist = rewiredDistribution(primaryRewired.map((entry) => entry.qd));
  const primaryOccupiedVerdict = metricVerdict(primaryBio.occupied, primaryOccupiedDist);
  const primaryQdVerdict = metricVerdict(primaryBio.qd, primaryQdDist);
  const primaryCategoryResult = categorize(primaryOccupiedVerdict, primaryQdVerdict);

  const perSeedInputs = [PRIMARY_SEARCH_SEED, ...EXTRA_SEARCH_SEEDS].map((seed) => {
    const bioAtSeed = bio(seed);
    const rewiredAtSeed = rewiredAt(seed);
    const occDist = rewiredDistribution(rewiredAtSeed.map((entry) => entry.occupied));
    const qdDist = rewiredDistribution(rewiredAtSeed.map((entry) => entry.qd));
    const category = categorize(metricVerdict(bioAtSeed.occupied, occDist), metricVerdict(bioAtSeed.qd, qdDist)).category;
    return { seed, category, n: rewiredAtSeed.length };
  });
  const robustnessResult = robustness(perSeedInputs.map(({ seed, category }) => ({ seed, category })));

  const disconnected = graphs.find((entry) => entry.id === 'disconnected');
  if (!disconnected) throw new Error('repertoire-report: missing disconnected graph after coverage check');

  return {
    version: 1,
    sources: {
      biologicalSha: inputs.manifestBiologicalSha,
      rewiringNullSha,
      evaluatedSha256: sha256Hex(inputs.evaluatedBytes),
      atlasSha256: sha256Hex(inputs.atlasBytes),
      graphsIndexSha256: sha256Hex(inputs.graphsIndexBytes),
      producer: repertoireNullProducer()
    },
    search: {
      population: atlasOptions.population,
      generations: atlasOptions.generations,
      ticks: atlasOptions.ticks,
      discoverySeeds: DISCOVERY_SEEDS,
      heldoutSeeds: HELDOUT_SEEDS,
      primarySearchSeed: PRIMARY_SEARCH_SEED,
      extraSearchSeeds: EXTRA_SEARCH_SEEDS,
      rewiredCount: REWIRED_COUNT,
      rewiredSeedMatchedCount: REWIRED_SEED_MATCHED_COUNT
    },
    graphs,
    primary: {
      bio: { occupied: primaryBio.occupied, qd: primaryBio.qd, span: primaryBio.span, heldoutOwnMedian: primaryBio.heldoutOwnMedian },
      rewiredDistribution: { occupied: primaryOccupiedDist, qd: primaryQdDist },
      category: primaryCategoryResult.category,
      tie: primaryCategoryResult.tie
    },
    robustness: {
      perSeed: robustnessResult.perSeed,
      robust: robustnessResult.robust,
      seedMatchedCounts: Object.fromEntries(perSeedInputs.map(({ seed, n }) => [seed, n]))
    },
    disconnected,
    host: { arch: process.arch, node: process.version }
  };
};

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/** Add/overwrite the manifest's `behaviorRepertoireNull` key in place -- touches only this one key, mirrors `null-report.ts#updateManifestWithRewiringNull`/`pathwayInterventions.ts#updateManifestWithPathwayInterventions`'s identical re-serialization scheme. */
export const updateManifestWithRepertoireNull = (
  manifestPath: string,
  entry: { readonly artifact: string; readonly sha256: string }
): void => {
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8')) as Record<string, unknown>;
  manifest.behaviorRepertoireNull = entry;
  atomicWriteFileSync(manifestPath, `${JSON.stringify(sortKeysDeep(manifest), null, 2)}\n`);
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

export interface RepertoireReportArgs {
  readonly evaluated: string;
  readonly graphsIndex: string;
  readonly rewiringNull: string;
  readonly atlas: string;
  readonly manifest: string;
  readonly out: string;
  readonly reportMd: string;
}

export const parseRepertoireReportArgs = (argv: readonly string[]): RepertoireReportArgs => {
  let evaluated = DEFAULT_EVALUATED;
  let graphsIndex = DEFAULT_GRAPHS_INDEX;
  let rewiringNull = DEFAULT_REWIRING_NULL;
  let atlas = DEFAULT_ATLAS;
  let manifest = DEFAULT_MANIFEST;
  let out = DEFAULT_OUT;
  let reportMd = DEFAULT_REPORT_MD;

  let i = 0;
  while (i < argv.length) {
    const flag = argv[i];
    if (flag === '--evaluated') {
      evaluated = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--graphs-index') {
      graphsIndex = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--rewiring-null') {
      rewiringNull = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--atlas') {
      atlas = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--manifest') {
      manifest = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--out') {
      out = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--report-md') {
      reportMd = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  // `--out`/`--report-md` are write targets, not merely inputs -- neither
  // may collide with each other or with any input file (a hand-typed
  // `--out` pointed at, say, `behavior-atlas-v1.json` would silently
  // overwrite the shipped atlas, breaking `npm run atlas:verify`; this bean's
  // own "the producer refuses to overwrite other artifacts" requirement).
  // Mirrors `pathwayInterventions.ts#parseIntoArtifactArgs`'s identical guard.
  for (const target of [out, reportMd]) {
    for (const input of [evaluated, graphsIndex, rewiringNull, atlas, manifest]) {
      if (resolve(target) === resolve(input)) {
        throw new Error(`repertoire-report: ${target === out ? '--out' : '--report-md'} must not overwrite an input file (${input})`);
      }
    }
  }
  if (resolve(out) === resolve(reportMd)) {
    throw new Error('repertoire-report: --out and --report-md must not be the same path');
  }
  // A dual review (Important) found `updateManifestWithRepertoireNull`
  // records only `basename(args.out)` under `behaviorRepertoireNull.artifact`
  // -- correct for the default, same-directory `--out`, but a scratch run
  // such as `--out /tmp/x.json` against the default `--manifest` would
  // silently repoint the real shipped manifest at a same-named file that
  // does not exist next to it (`public/data/x.json`), which the browser
  // would then report as `unavailable`. Requiring `--out` to sit beside
  // `--manifest` keeps the recorded basename meaningful.
  if (dirname(resolve(out)) !== dirname(resolve(manifest))) {
    throw new Error("repertoire-report: --out must be in the same directory as --manifest (the manifest records only --out's basename)");
  }

  return { evaluated, graphsIndex, rewiringNull, atlas, manifest, out, reportMd };
};

export interface RunRepertoireReportResult {
  readonly out: string;
  readonly reportMdPath: string;
  readonly artifactSha256: string;
  readonly artifact: RepertoireNullArtifact;
}

export const runRepertoireReport = (args: Readonly<RepertoireReportArgs>): RunRepertoireReportResult => {
  const evaluatedBytes = readFileSync(args.evaluated);
  const evaluated = JSON.parse(evaluatedBytes.toString('utf8')) as RepertoireEvaluatedArtifact;
  const rewiringNullBytes = readFileSync(args.rewiringNull);
  const rewiringNullParsed = JSON.parse(rewiringNullBytes.toString('utf8')) as { readonly sourceGraphSha256: string };
  const atlasBytes = readFileSync(args.atlas);
  const atlasParsed = JSON.parse(atlasBytes.toString('utf8')) as { readonly source: SearchArtifact };
  const graphsIndexBytes = readFileSync(args.graphsIndex);
  const graphsIndexParsed = JSON.parse(graphsIndexBytes.toString('utf8')) as {
    readonly sourceSha256: string;
    readonly seeds: readonly unknown[];
  };

  const manifest = JSON.parse(readFileSync(args.manifest, 'utf8')) as {
    readonly binarySha256?: string;
    readonly rewiringNull?: { readonly sha256?: string };
  };
  if (!manifest.binarySha256) throw new Error(`repertoire-report: ${args.manifest} is missing binarySha256`);
  if (!manifest.rewiringNull?.sha256) throw new Error(`repertoire-report: ${args.manifest} is missing rewiringNull.sha256`);

  const artifact = buildRepertoireNullArtifact({
    evaluated,
    evaluatedBytes,
    manifestBiologicalSha: manifest.binarySha256,
    manifestRewiringNullSha: manifest.rewiringNull.sha256,
    rewiringNullBytes,
    rewiringNullParsed,
    atlasBytes,
    atlasParsed,
    graphsIndexBytes,
    graphsIndexParsed
  });

  verifyManifestRoundTrips(args.manifest);

  const artifactContents = JSON.stringify(artifact);
  const artifactSha256 = sha256Hex(artifactContents);
  const reportMdContents = renderRepertoireNullReportMarkdown(artifact);

  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, artifactContents);

  updateManifestWithRepertoireNull(args.manifest, { artifact: basename(args.out), sha256: artifactSha256 });

  mkdirSync(dirname(args.reportMd), { recursive: true });
  atomicWriteFileSync(args.reportMd, reportMdContents);

  return { out: args.out, reportMdPath: args.reportMd, artifactSha256, artifact };
};

const main = (): void => {
  try {
    const args = parseRepertoireReportArgs(process.argv.slice(2));
    const result = runRepertoireReport(args);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `repertoire-report: wrote ${result.out} (sha256 ${result.artifactSha256}) and ${result.reportMdPath}\n` +
        `primary.category=${result.artifact.primary.category} robustness.robust=${result.artifact.robustness.robust}`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`repertoire-report failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
