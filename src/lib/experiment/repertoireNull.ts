/**
 * `public/data/behavior-repertoire-null-v1.json` (WP3 of
 * `.agents/plans/repertoire-null`, `03-artifact-and-atlas-strip.md`): the
 * offline-computed behavior-repertoire comparison between the biological
 * MaleCNS topology and its degree-preserving rewirings, under the shipped
 * behavior-atlas MAP-Elites search (`docs/behavior-repertoire-null-report.md`
 * has the full method and every number below). The real producer's full
 * type is `scripts/atlas/repertoire-report.ts`'s `RepertoireNullArtifact` —
 * that module is a Node-only pipeline (not part of the browser bundle), so
 * this file independently authors and validates just the subset the atlas
 * strip and the Findings panel's step 7 actually render, the same
 * "reimplemented, not imported" discipline `./nullExplanation.ts`/
 * `./rewiringNull.ts`/`./pathwayInterventions.ts` already document for their
 * own artifacts.
 *
 * Lives in `src/lib/experiment/`, beside those three sibling manifest-
 * sidecar loaders (same family: fetch -> sha256-verify -> shape-validate ->
 * cross-check against the central manifest), not in `src/lib/atlas/` —
 * moved there after a maintainability review, Important: this file's own
 * doc comment used to justify the `atlas/` location on "its only consumer
 * is `Atlas.svelte`", which stopped being true the moment `.agents/plans/
 * findings-tour` was wired up to this artifact too. Two independent
 * consumers load it today: `Atlas.svelte`'s strip (fetches the central
 * manifest and this artifact on its own, independently of
 * `ExperimentController#initialize()` -- `03-artifact-and-atlas-strip.md`:
 * "Its state never delays or fails the atlas load") and
 * `ExperimentController` (Findings step 7, `src/lib/findings/steps.ts`).
 * Living beside `rewiringNull.ts`/`pathwayInterventions.ts` also means
 * `ExperimentController` importing this module is an ordinary same-directory
 * dependency, not a new `experiment/ -> atlas/` edge.
 */

import { fetchAndVerifySidecarJson, type ArenaManifest, type SidecarManifestEntry } from './assets';
import type { SidecarLoadResult } from './sidecarResult';

/** `00-overview.md`'s predeclared category vocabulary, verbatim. */
export type RepertoireCategory = 'wider' | 'narrower' | 'typical';

const isCategory = (value: unknown): value is RepertoireCategory =>
  value === 'wider' || value === 'narrower' || value === 'typical';

/** One metric's rewired 25th/50th/75th percentile plus its full sorted value set -- `scripts/atlas/repertoire-metrics.ts`'s `RewiredDistribution`, restated for the browser bundle. */
export interface RepertoireRewiredDistribution {
  readonly n: number;
  readonly p25: number;
  readonly p50: number;
  readonly p75: number;
  readonly values: readonly number[];
}

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

/**
 * A maintainability review (Important) found the previous version of this
 * validator accepted an empty or internally inconsistent `values` array
 * (`n !== values.length`, unsorted, or `p25`/`p50`/`p75` outside the
 * values' own range) -- `Atlas.svelte` reads `values[0]`/`values.at(-1)`
 * for the strip's "range a-b" clause, so an empty array rendered the
 * literal text "range undefined-undefined" as a verified `ok` result.
 * Every structural property `rewiredDistribution()`
 * (`scripts/atlas/repertoire-metrics.ts`) actually guarantees is now
 * re-checked here: non-empty, `values.length === n`, ascending, and the
 * three percentiles ordered and bounded by the values themselves.
 */
const isRewiredDistribution = (value: unknown): value is RepertoireRewiredDistribution => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  if (
    !isFiniteNumber(v.n) ||
    !isFiniteNumber(v.p25) ||
    !isFiniteNumber(v.p50) ||
    !isFiniteNumber(v.p75) ||
    !Array.isArray(v.values) ||
    !v.values.every(isFiniteNumber)
  ) {
    return false;
  }
  const values = v.values as number[];
  return (
    values.length > 0 &&
    values.length === v.n &&
    values.every((value, index) => index === 0 || values[index - 1] <= value) &&
    v.p25 <= v.p50 &&
    v.p50 <= v.p75 &&
    values[0] <= v.p25 &&
    v.p75 <= values[values.length - 1]
  );
};

export interface RepertoireNullSearchConfig {
  readonly population: number;
  readonly generations: number;
  readonly ticks: number;
  readonly primarySearchSeed: number;
  readonly extraSearchSeeds: readonly number[];
  readonly rewiredCount: number;
  readonly rewiredSeedMatchedCount: number;
}

export interface RepertoireNullPrimary {
  readonly bio: { readonly occupied: number; readonly qd: number; readonly span: number; readonly heldoutOwnMedian: number };
  readonly rewiredDistribution: { readonly occupied: RepertoireRewiredDistribution; readonly qd: RepertoireRewiredDistribution };
  readonly category: RepertoireCategory;
  readonly tie: boolean;
}

export interface RepertoireNullRobustness {
  readonly perSeed: Readonly<Record<number, RepertoireCategory>>;
  readonly robust: boolean;
  readonly seedMatchedCounts: Readonly<Record<number, number>>;
}

/**
 * The subset of the producer's `RepertoireNullArtifact` this browser shape
 * carries -- no `graphs`/`host`/`sources.evaluatedSha256`/etc., matching
 * `PathwayInterventionsArtifact`'s own "browser subset, not the producer's
 * full type" precedent (a maintainability review, Suggestion: grepping this
 * type name now finds two structurally different shapes -- see that
 * producer type's own doc comment for the fuller shape). Audit-table/
 * occupancy-map/per-graph detail (`graphs[]`) is intentionally not carried
 * into this browser-side shape; a reader who wants that detail follows the
 * report link.
 */
export interface RepertoireNullArtifact {
  readonly version: 1;
  readonly sources: { readonly biologicalSha: string; readonly rewiringNullSha: string; readonly atlasSha256: string };
  readonly search: RepertoireNullSearchConfig;
  readonly primary: RepertoireNullPrimary;
  readonly robustness: RepertoireNullRobustness;
  readonly disconnected: { readonly occupied: number; readonly qd: number };
}

export type RepertoireNullLoadResult = SidecarLoadResult<RepertoireNullArtifact>;

const validateShape = (value: unknown): { ok: true; data: RepertoireNullArtifact } | { ok: false; reason: string } => {
  if (typeof value !== 'object' || value === null) {
    return { ok: false, reason: 'repertoire-null artifact is not a JSON object' };
  }
  const v = value as Record<string, unknown>;
  if (v.version !== 1) return { ok: false, reason: `repertoire-null artifact has unsupported version ${String(v.version)}` };

  const sources = v.sources as Record<string, unknown> | undefined;
  if (
    !sources ||
    typeof sources.biologicalSha !== 'string' ||
    typeof sources.rewiringNullSha !== 'string' ||
    typeof sources.atlasSha256 !== 'string'
  ) {
    return { ok: false, reason: 'repertoire-null artifact is missing sources.biologicalSha/sources.rewiringNullSha/sources.atlasSha256' };
  }

  const search = v.search as Record<string, unknown> | undefined;
  if (
    !search ||
    !isFiniteNumber(search.population) ||
    !isFiniteNumber(search.generations) ||
    !isFiniteNumber(search.ticks) ||
    !isFiniteNumber(search.primarySearchSeed) ||
    !Array.isArray(search.extraSearchSeeds) ||
    !search.extraSearchSeeds.every(isFiniteNumber) ||
    !isFiniteNumber(search.rewiredCount) ||
    !isFiniteNumber(search.rewiredSeedMatchedCount)
  ) {
    return { ok: false, reason: 'repertoire-null artifact has a malformed "search" field' };
  }

  const primary = v.primary as Record<string, unknown> | undefined;
  const primaryBio = primary?.bio as Record<string, unknown> | undefined;
  const primaryDist = primary?.rewiredDistribution as Record<string, unknown> | undefined;
  if (
    !primary ||
    !primaryBio ||
    !isFiniteNumber(primaryBio.occupied) ||
    !isFiniteNumber(primaryBio.qd) ||
    !isFiniteNumber(primaryBio.span) ||
    !isFiniteNumber(primaryBio.heldoutOwnMedian) ||
    !primaryDist ||
    !isRewiredDistribution(primaryDist.occupied) ||
    !isRewiredDistribution(primaryDist.qd) ||
    !isCategory(primary.category) ||
    typeof primary.tie !== 'boolean'
  ) {
    return { ok: false, reason: 'repertoire-null artifact has a malformed "primary" field' };
  }
  // A maintainability review (Suggestion) noted these two distributions'
  // own `n` must equal the primary search seed's actual rewired sample
  // size -- both the strip and Findings step 7 print `search.rewiredCount`
  // next to this distribution's `p50` as if it were that distribution's own
  // sample size.
  const primaryOccupied = primaryDist.occupied as RepertoireRewiredDistribution;
  const primaryQd = primaryDist.qd as RepertoireRewiredDistribution;
  if (primaryOccupied.n !== search.rewiredCount || primaryQd.n !== search.rewiredCount) {
    return { ok: false, reason: 'repertoire-null artifact primary distribution size disagrees with search.rewiredCount' };
  }

  const robustness = v.robustness as Record<string, unknown> | undefined;
  const perSeedRaw = robustness?.perSeed as Record<string, unknown> | undefined;
  const seedMatchedCountsRaw = robustness?.seedMatchedCounts as Record<string, unknown> | undefined;
  if (!robustness || typeof robustness.robust !== 'boolean' || !perSeedRaw || !seedMatchedCountsRaw) {
    return { ok: false, reason: 'repertoire-null artifact has a malformed "robustness" field' };
  }
  const seeds = [search.primarySearchSeed, ...search.extraSearchSeeds] as number[];
  const perSeed: Record<number, RepertoireCategory> = {};
  const seedMatchedCounts: Record<number, number> = {};
  for (const seed of seeds) {
    const category = perSeedRaw[String(seed)];
    const n = seedMatchedCountsRaw[String(seed)];
    if (!isCategory(category) || !isFiniteNumber(n)) {
      return { ok: false, reason: `repertoire-null artifact is missing a valid robustness entry for seed ${seed}` };
    }
    perSeed[seed] = category;
    seedMatchedCounts[seed] = n;
  }
  // Cross-field consistency, mirroring `pathwayInterventions.ts`'s own
  // `trainedRobust`-vs-`perSeedCategory` check: a hash-valid artifact could
  // still ship a `robustness.robust` that disagrees with its own per-seed
  // categories.
  const categories = Object.values(perSeed);
  const recomputedRobust = categories.every((category) => category === categories[0]);
  if (recomputedRobust !== robustness.robust) {
    return { ok: false, reason: 'repertoire-null artifact robustness.robust disagrees with its own per-seed categories' };
  }
  // A maintainability review (Suggestion) noted this loader had everything
  // it needed to catch a self-consistent-but-truncated artifact whose
  // top-level `primary.category` simply disagrees with its own
  // `robustness.perSeed` entry at the primary seed -- cheaper than fully
  // recomputing the category from `bio`/percentiles (which would require
  // reimplementing `repertoire-metrics.ts#categorize`'s tie rule here too;
  // deferred as a follow-up, matching the lighter-weight cross-field checks
  // every sibling loader in this family already uses instead of a full
  // statistical recompute).
  if (perSeed[search.primarySearchSeed] !== primary.category) {
    return { ok: false, reason: 'repertoire-null artifact primary.category disagrees with robustness.perSeed at the primary seed' };
  }

  const disconnected = v.disconnected as Record<string, unknown> | undefined;
  if (!disconnected || !isFiniteNumber(disconnected.occupied) || !isFiniteNumber(disconnected.qd)) {
    return { ok: false, reason: 'repertoire-null artifact has a malformed "disconnected" field' };
  }

  return {
    ok: true,
    data: {
      version: 1,
      sources: { biologicalSha: sources.biologicalSha, rewiringNullSha: sources.rewiringNullSha, atlasSha256: sources.atlasSha256 },
      search: {
        population: search.population,
        generations: search.generations,
        ticks: search.ticks,
        primarySearchSeed: search.primarySearchSeed,
        extraSearchSeeds: search.extraSearchSeeds as number[],
        rewiredCount: search.rewiredCount,
        rewiredSeedMatchedCount: search.rewiredSeedMatchedCount
      },
      primary: {
        bio: {
          occupied: primaryBio.occupied,
          qd: primaryBio.qd,
          span: primaryBio.span,
          heldoutOwnMedian: primaryBio.heldoutOwnMedian
        },
        rewiredDistribution: { occupied: primaryOccupied, qd: primaryQd },
        category: primary.category,
        tie: primary.tie
      },
      robustness: { perSeed, robust: robustness.robust, seedMatchedCounts },
      disconnected: { occupied: disconnected.occupied, qd: disconnected.qd }
    }
  };
};

/**
 * Fetch, sha256-verify, and structurally validate `behavior-repertoire-null-v1.json`
 * (`manifest.behaviorRepertoireNull`), then cross-check it against the
 * manifest's own pinned biological graph and rewiring-null distribution —
 * never throws, matching `loadNullExplanation`/`loadRewiringNull`/
 * `loadPathwayInterventions`'s own "never throws, always return a reasoned
 * status" contract, so a missing/tampered/malformed artifact only ever
 * hides or degrades the atlas strip / Findings step 7, never the rest of
 * either surface (`03-artifact-and-atlas-strip.md`: "Its state never
 * delays or fails the atlas load").
 *
 * Deliberately does **not** cross-check `sources.atlasSha256` here: the
 * shipped behavior atlas is pinned by its own separate
 * `behavior-atlas-v1.manifest.json`, not by this central `ArenaManifest`,
 * so this loader has no live value to compare it against. `Atlas.svelte`
 * (which independently loads both this artifact and the atlas itself) is
 * where that staleness check actually happens, once both loads have
 * resolved -- a maintainability review, Important: the artifact's own
 * premise is "reuses the shipped atlas search exactly", and this field
 * exists specifically so that premise is checkable, not merely recorded.
 * `ExperimentController` (Findings step 7) never loads the atlas at all, so
 * it has nothing to cross-check `atlasSha256` against either; that step
 * links out to `#atlas` rather than rendering beside the live grid.
 *
 * `dataBaseUrl` must be the same value the caller passes to `loadAtlas`
 * (`${import.meta.env.BASE_URL}data` in production) so this artifact
 * resolves under the app's real deployment base path too.
 */
export const loadRepertoireNull = async (
  manifest: ArenaManifest,
  dataBaseUrl: string
): Promise<RepertoireNullLoadResult> => {
  const entry: SidecarManifestEntry | undefined = manifest.behaviorRepertoireNull;
  const fetched = await fetchAndVerifySidecarJson(entry, dataBaseUrl, 'repertoire-null artifact');
  if (fetched.status === 'no-entry') {
    return { status: 'missing', reason: 'The manifest has no behaviorRepertoireNull artifact entry.' };
  }
  if (fetched.status === 'fetch-error') {
    return { status: 'unavailable', reason: fetched.reason };
  }
  if (fetched.status === 'hash-mismatch' || fetched.status === 'parse-error') {
    return { status: 'invalid', reason: fetched.reason };
  }

  const validated = validateShape(fetched.parsed);
  if (!validated.ok) return { status: 'invalid', reason: validated.reason };
  const data = validated.data;

  // The sha256 check above only proves these bytes are the ones the
  // manifest's `behaviorRepertoireNull` entry pins — it says nothing about
  // whether this study was actually computed against *this* manifest's
  // biological graph or rewiring-null distribution (a hand-edited or
  // merge-conflicted manifest could re-pin `behaviorRepertoireNull` to a
  // study computed against a different biological graph or null
  // distribution). Mirrors `loadPathwayInterventions`'s own staleness
  // checks.
  if (data.sources.biologicalSha !== manifest.binarySha256) {
    return {
      status: 'invalid',
      reason: `repertoire-null sources.biologicalSha does not match the manifest's compiled biological graph (${manifest.binarySha256}) — stale artifact`
    };
  }
  const shippedRewiringNullSha = manifest.rewiringNull?.sha256;
  if (!shippedRewiringNullSha) {
    return { status: 'invalid', reason: 'manifest is missing rewiringNull.sha256, needed to cross-check the repertoire-null artifact' };
  }
  if (data.sources.rewiringNullSha !== shippedRewiringNullSha) {
    return {
      status: 'invalid',
      reason: `repertoire-null sources.rewiringNullSha does not match the manifest's rewiring-null artifact (${shippedRewiringNullSha}) — stale artifact`
    };
  }

  return { status: 'ok', data };
};
