/**
 * The pure computation behind `Atlas.svelte`'s one-line repertoire-
 * comparison strip -- extracted out of that component's `$derived`/
 * `$derived.by` blocks (thermo-maintainability review, Suggestion S2: a
 * template regression here was previously only catchable by a full
 * Playwright run, since `Atlas.svelte` is never imported by any unit/
 * component test in this repo -- `Shell.svelte`'s lazy `import()` is the
 * only reference to it). Both functions here are plain, DOM-free
 * computations over already-resolved data, so `tests/unit/repertoire-strip.test.ts`
 * exercises them directly, in milliseconds, without mounting the
 * component.
 */

import type { RepertoireNullArtifact, RepertoireRewiredDistribution } from '../experiment/repertoireNull';
import { CELL_COUNT } from './types';

/**
 * True when the repertoire study's own recorded atlas identity
 * (`sources.atlasSha256`) disagrees with the atlas actually loaded and on
 * screen (`loaded.sha256`) -- `Atlas.svelte`'s own display-time staleness
 * check (a thermo-methodology review, Important: the study's whole premise
 * is "reuses the shipped atlas search exactly," and this is the only place
 * that premise is checked at all, since `loadRepertoireNull`'s own loader
 * has no independent handle on the live atlas sha -- see that function's
 * doc comment). Never a live rehash: both shas are already-computed
 * sha256 strings from two independent, already-verified loads.
 */
export const isRepertoireStale = (repertoireAtlasSha256: string, loadedAtlasSha256: string): boolean =>
  repertoireAtlasSha256 !== loadedAtlasSha256;

/**
 * The predeclared per-metric tie rule (`.agents/plans/repertoire-null/00-overview.md`'s
 * "Predeclared categories" section), recomputed here from data already on
 * `primary` -- mirrors `scripts/atlas/repertoire-metrics.ts#metricVerdict`
 * exactly (`bio >= p75 && bio > p25` -> wider; `bio <= p25 && bio < p75` ->
 * narrower; else typical). A two-comparison recomputation, not a new fetch
 * or a full statistical recompute -- the same lightweight cross-field-
 * consistency discipline this codebase's sidecar loaders already use
 * elsewhere (e.g. `robustness.robust` recomputed from `perSeed` in
 * `repertoireNull.ts`'s own `validateShape`).
 */
export const metricVerdictLabel = (bio: number, distribution: Readonly<Pick<RepertoireRewiredDistribution, 'p25' | 'p75'>>): string =>
  bio >= distribution.p75 && bio > distribution.p25 ? 'wider' : bio <= distribution.p25 && bio < distribution.p75 ? 'narrower' : 'typical';

/**
 * `03-artifact-and-atlas-strip.md`'s one-line strip, templated only from
 * `data`'s own verified fields — never a hard-coded number. Caller
 * (`Atlas.svelte`) is responsible for the `missing`/`unavailable`/
 * `invalid`/stale gating; this function assumes `data` is an already-
 * verified, non-stale `RepertoireNullArtifact`.
 *
 * A thermo-methodology review (Important) found the strip previously
 * stated the category next to only `occupied` (a one-cell gap inside a
 * 26-32 range), even though the predeclared rule -- and
 * `repertoire-metrics.ts#categorize` -- decides the category jointly from
 * *both* `occupied` and `qd`. A reader could reasonably (and wrongly)
 * conclude the whole verdict rested on that single cell, when the `qd` gap
 * is in fact large and unambiguous. `qd` and its rewired median are now
 * always stated (rounded to the nearest integer for concision -- the raw
 * floats are in the full report), alongside an explicit basis clause: "on
 * both metrics" when the category is wider/narrower (which can only hold
 * when both metrics independently agree), or each metric's own verdict
 * when the category is "typical" (which can mask the two metrics
 * disagreeing).
 */
export const buildRepertoireStripText = (data: Readonly<RepertoireNullArtifact>): string => {
  const { primary, search, robustness } = data;
  const dist = primary.rewiredDistribution.occupied;
  const qdDist = primary.rewiredDistribution.qd;
  // `dist.values` is guaranteed non-empty and ascending by
  // `loadRepertoireNull`'s own shape validation, so `[0]`/`.at(-1)` are
  // always the sample's real min/max (a maintainability review, Important,
  // previously found this could read "undefined" for a malformed artifact
  // -- the loader now rejects that shape outright).
  const range = `${dist.values[0]}–${dist.values.at(-1)}`;
  const seeds = [search.primarySearchSeed, ...search.extraSearchSeeds];
  const seedCount = seeds.length;
  const robustClause = robustness.robust
    ? `robust across all ${seedCount} search seeds`
    : `not robust: seeds give ${seeds.map((seed) => `${seed}: ${robustness.perSeed[seed]}`).join(', ')}`;
  // A maintainability review (Suggestion) found this sentence read "Seeds
  // <blank> compare against only N seed-matched rewirings" whenever there
  // were no extra search seeds at all -- omit the clause entirely in that
  // case instead.
  const coarseClause =
    search.extraSearchSeeds.length > 0
      ? ` Seeds ${search.extraSearchSeeds[0]}–${search.extraSearchSeeds.at(-1)} compare against only ${search.rewiredSeedMatchedCount} seed-matched rewirings.`
      : '';
  // `primary.tie` (a maintainability review, Suggestion): validated by the
  // loader but previously never shown here.
  const tieSuffix = primary.tie ? ' (tie)' : '';
  const basisClause =
    primary.category !== 'typical'
      ? 'on both metrics'
      : `occupied ${metricVerdictLabel(primary.bio.occupied, dist)}, qd ${metricVerdictLabel(primary.bio.qd, qdDist)}`;
  return (
    `Repertoire vs ${search.rewiredCount} rewirings: biological occupies ${primary.bio.occupied} of ${CELL_COUNT} cells ` +
    `(rewired median ${dist.p50}, range ${range}), qd ${Math.round(primary.bio.qd)} vs rewired median ${Math.round(qdDist.p50)} — ` +
    `${primary.category}${tieSuffix} ${basisClause} at search seed ${search.primarySearchSeed}; ` +
    `${robustClause}.${coarseClause}`
  );
};
