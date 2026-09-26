import { rewiredDistribution, type RewiredDistribution } from './repertoire-metrics';
import { PRIMARY_SEARCH_SEED, EXTRA_SEARCH_SEEDS } from './repertoire-plan';
import { CELL_COUNT, COVERAGE_BIN_COUNT, TURN_BIN_COUNT, COVERAGE_EDGES, TURN_EDGES } from '../../src/lib/atlas/types';
import type { RepertoireGraphSummary, RepertoireNullArtifact } from './repertoire-report';

/**
 * `docs/behavior-repertoire-null-report.md`'s markdown renderer -- split
 * out of `repertoire-report.ts` (thermo-maintainability review, Suggestion
 * S1: that file bundled six responsibilities in one module, and this
 * ~235-line render function plus its small helpers were the one
 * self-contained, independently-testable piece with zero coupling to the
 * artifact builder or the CLI glue around it -- confirmed by
 * `tests/unit/repertoire-report.test.ts`'s own `describe('renderRepertoireNullReportMarkdown',
 * ...)` block already testing it in isolation). A pure mechanical move: no
 * behavior change, same exported function name, same output.
 *
 * `RepertoireGraphSummary`/`RepertoireNullArtifact` are imported as
 * `import type` from `./repertoire-report` (the reverse direction:
 * `repertoire-report.ts` imports `renderRepertoireNullReportMarkdown` from
 * *this* file) -- a type-only edge, erased entirely at compile time by
 * `import type`, so this is not a runtime circular dependency. The
 * producer's own `sourceSha256` (`repertoireNullProducer()`,
 * `scripts/atlas/repertoire-report.ts`) is a hash over the real, walked
 * import-graph closure from `repertoire-report.ts`, so this file is
 * automatically included in that closure the same way every other real
 * dependency already is -- no separate registration needed.
 */

const fmt = (value: number): string => value.toFixed(2);

/**
 * The finest percentile granularity an `n`-point sample can express (matches
 * `scripts/null/null-stats.ts#percentileResolution`'s own `1/n` convention,
 * restated as a percentage) -- a dual review (Important) found the report
 * previously hard-coded "5%" next to the templated `rewiredSeedMatchedCount`
 * (a 5-point sample is actually a 20% resolution; the report's own
 * Limitations section already stated the correct 20% two paragraphs later,
 * so the Method section contradicted itself). Deriving both figures here
 * means neither can drift from `REWIRED_COUNT`/`REWIRED_SEED_MATCHED_COUNT`
 * again.
 */
const resolutionPct = (n: number): string => `${Math.round(100 / n)}%`;

const renderDistribution = (label: string, dist: Readonly<RewiredDistribution>): string =>
  `${label}: n=${dist.n}, p25=${fmt(dist.p25)}, p50=${fmt(dist.p50)}, p75=${fmt(dist.p75)}`;

const AUDIT_TABLE_HEADER = '| graph | seed | occupied | qd | span | heldoutOwnMedian | gpuArchiveSize | collisions |';
const AUDIT_TABLE_DIVIDER = '| --- | --- | --- | --- | --- | --- | --- | --- |';

const renderAuditRow = (graph: Readonly<RepertoireGraphSummary>): string =>
  `| ${graph.id} | ${graph.seed} | ${graph.occupied} | ${fmt(graph.qd)} | ${graph.span} | ${fmt(graph.heldoutOwnMedian)} | ${graph.gpuArchiveSize} | ${graph.collisions} |`;

/** A `COVERAGE_BIN_COUNT`x`COVERAGE_BIN_COUNT` occupancy grid (row = turning bin, column = coverage bin, `cellFor`'s own packing) over a set of graphs' occupied-cell lists -- counts how many of `graphs` occupy each of the `CELL_COUNT` cells. */
const occupancyFrequency = (graphs: readonly Readonly<RepertoireGraphSummary>[]): number[] => {
  const counts = new Array(CELL_COUNT).fill(0);
  for (const graph of graphs) {
    for (const cell of graph.cells) counts[cell] += 1;
  }
  return counts;
};

const pct = (value: number): string => `${(value * 100).toFixed(0)}%`;

/**
 * A dual review (Important) found this grid rendered as bare `| a | b | ... |`
 * rows with no header row and no `| --- |` delimiter row, so GitHub-flavored
 * Markdown (which requires both before treating lines as a table) showed it
 * as a run of literal pipes -- and with no row/column labels beyond the
 * section heading above, it was hard to read even once it rendered. Column
 * headers and row labels are the same coverage/turning-bin boundaries
 * `Atlas.svelte`'s own axis captions use (`COVERAGE_EDGES`/`TURN_EDGES`),
 * never a re-derived approximation.
 */
const renderOccupancyGrid = (title: string, counts: readonly number[], format: (n: number) => string): string => {
  const columnHeaders = COVERAGE_EDGES.slice(0, -1).map((edge) => `${pct(edge)}+`);
  const header = `| turning \\ coverage | ${columnHeaders.join(' | ')} |`;
  const divider = `| --- | ${columnHeaders.map(() => '---').join(' | ')} |`;
  const rows: string[] = [`#### ${title}`, '', header, divider];
  for (let row = TURN_BIN_COUNT - 1; row >= 0; row -= 1) {
    const rowLabel = `${TURN_EDGES[row].toFixed(2)}..${TURN_EDGES[row + 1].toFixed(2)}`;
    const cells = Array.from({ length: COVERAGE_BIN_COUNT }, (_, col) => format(counts[row * COVERAGE_BIN_COUNT + col]));
    rows.push(`| ${rowLabel} | ${cells.join(' | ')} |`);
  }
  return rows.join('\n');
};

/**
 * `.agents/plans/repertoire-null/00-overview.md`'s "Predeclared categories"
 * section, quoted byte-verbatim -- follows `pathwayInterventions.ts`'s
 * `AUTHORED_CATEGORY_PROSE` precedent (thermo-methodology review): a
 * report's method section states the predeclared rule exactly as written in
 * the plan, not a paraphrase.
 */
const PREDECLARED_CATEGORIES_PROSE = `- **Wider repertoire:** biological is at or above the rewired 75th percentile on both \`occupied\` and \`qd\`.
- **Narrower repertoire:** biological is at or below the 25th percentile on both.
- **Typical:** otherwise.
- Tie rule (predeclared): **Wider** requires, on both metrics, \`bio ≥ p75\` **and** \`bio > p25\`. **Narrower** requires, on both metrics, \`bio ≤ p25\` **and** \`bio < p75\`. With a degenerate rewired distribution (\`p25 = p75 = bio\`) neither holds, and the result is **Typical** with \`tie: true\`. Wider and Narrower can never both hold.`;

export const renderRepertoireNullReportMarkdown = (artifact: Readonly<RepertoireNullArtifact>): string => {
  const { sources, search, graphs, primary, robustness: robustnessResult, disconnected } = artifact;

  const biologicalRows = [PRIMARY_SEARCH_SEED, ...EXTRA_SEARCH_SEEDS]
    .map((seed) => graphs.find((g) => g.id === 'biological' && g.seed === seed))
    .filter((g): g is RepertoireGraphSummary => g !== undefined)
    .map(renderAuditRow)
    .join('\n');
  const rewiredRows = graphs
    .filter((g) => g.arm === 'rewired')
    .sort((a, b) => (a.rewiringSeed ?? 0) - (b.rewiringSeed ?? 0) || a.seed - b.seed)
    .map(renderAuditRow)
    .join('\n');

  const robustnessRows = [PRIMARY_SEARCH_SEED, ...EXTRA_SEARCH_SEEDS]
    .map((seed) => {
      const rewiredAtSeed = graphs.filter((g) => g.arm === 'rewired' && g.seed === seed);
      const occDist = rewiredDistribution(rewiredAtSeed.map((g) => g.occupied));
      const qdDist = rewiredDistribution(rewiredAtSeed.map((g) => g.qd));
      const category = robustnessResult.perSeed[seed];
      const n = robustnessResult.seedMatchedCounts[seed];
      return `| ${seed} | ${n} | ${occDist.p25}/${occDist.p50}/${occDist.p75} | ${fmt(qdDist.p25)}/${fmt(qdDist.p50)}/${fmt(qdDist.p75)} | ${category} |`;
    })
    .join('\n');

  const biologicalOccupancy = occupancyFrequency(graphs.filter((g) => g.id === 'biological' && g.seed === PRIMARY_SEARCH_SEED));
  const rewiredOccupancy = occupancyFrequency(graphs.filter((g) => g.arm === 'rewired' && g.seed === PRIMARY_SEARCH_SEED));
  const rewiredGraphCountAtPrimary = graphs.filter((g) => g.arm === 'rewired' && g.seed === PRIMARY_SEARCH_SEED).length;

  return `# Behavior repertoire, biological vs. degree-preserving rewirings

## Question

Does the measured MaleCNS topology support a wider or narrower range of behaviors than degree-preserving
rewirings of the same graph, under this model's shipped MAP-Elites behavior-discovery search? This is a
comparison of behavioral *repertoire* (how many distinct behavior cells are reached, and how good the best
controller in each is), not of a single behavior score.

## Method

The shipped behavior-atlas MAP-Elites search (population ${search.population}, generations ${search.generations},
ticks ${search.ticks}, discovery seeds ${JSON.stringify(search.discoverySeeds)}, held-out seeds
${JSON.stringify(search.heldoutSeeds)} -- the same search budget and descriptors as the shipped
\`behavior-atlas-v1.json\`, never a tuned or shrunk one) was run once per graph at search seed
${search.primarySearchSeed} on: the biological graph, the disconnected control, and ${search.rewiredCount}
degree-preserving rewirings (seeds \`0..${search.rewiredCount - 1}\`). Biological and rewirings
\`0..${search.rewiredSeedMatchedCount - 1}\` were additionally searched at seeds
${search.extraSearchSeeds.join(', ')} for search-seed robustness -- **rewired seeds
${search.rewiredSeedMatchedCount}..${search.rewiredCount - 1} were searched at a single search seed
(${search.primarySearchSeed}) only**, never at the extra seeds. Every search was re-evaluated in TypeScript
(authoritative), re-binned into the ${CELL_COUNT}-cell coverage x turning grid, with the diversity gate disabled
for null graphs (a narrow repertoire is a valid result, not an error). The discovery seeds and held-out seeds
above are pinned globally, shared by every graph and every search seed; only population/generations/ticks are
checked per entry against the shipped atlas budget (a maintainability review, Suggestion: this doc comment
previously left that unstated, which could be misread as a per-entry seed check).

### Predeclared metrics

- \`occupied\`: the number of occupied cells among ${CELL_COUNT}, after TS rebinning and collision resolution.
- \`qd\`: the sum over occupied cells of \`max(0, quality)\`.
- \`span\`: the number of distinct coverage bins plus the number of distinct turning bins that are occupied.
- \`heldoutOwnMedian\`: the median over occupied cells of the searched graph's own held-out mean score
  (seeds ${JSON.stringify(search.heldoutSeeds)}, trained decoder on that graph). Reported, not categorized.
- Audit fields per graph: \`gpuArchiveSize\`, \`occupied\` after TS rebinning, and \`collisions\`.

### Predeclared categories

${PREDECLARED_CATEGORIES_PROSE}

Search-seed robustness: the category is **robust** only if it is the same for all ${1 + search.extraSearchSeeds.length}
biological search seeds against the seed-matched rewired distribution. Rewirings
\`0..${search.rewiredSeedMatchedCount - 1}\` are the seed-matched sample for seeds
${search.extraSearchSeeds.join(', ')} -- **that sample has only ${search.rewiredSeedMatchedCount} points (a ${resolutionPct(search.rewiredSeedMatchedCount)} percentile
resolution)**, far coarser than the ${search.rewiredCount}-point primary comparison at seed ${search.primarySearchSeed}.

## Results

### Primary comparison (search seed ${search.primarySearchSeed})

| metric | biological | rewired distribution (n=${rewiredGraphCountAtPrimary}) |
| --- | --- | --- |
| occupied | ${primary.bio.occupied} | ${renderDistribution('rewired', primary.rewiredDistribution.occupied)} |
| qd | ${fmt(primary.bio.qd)} | ${renderDistribution('rewired', primary.rewiredDistribution.qd)} |
| span | ${primary.bio.span} | -- (not categorized) |
| heldoutOwnMedian | ${fmt(primary.bio.heldoutOwnMedian)} | -- (not categorized) |

**Category at seed ${search.primarySearchSeed}: ${primary.category}**${primary.tie ? ' (tie: a rewired distribution was degenerate and equal to biological on at least one metric)' : ''}.

### Search-seed robustness

| search seed | seed-matched rewired n | occupied p25/p50/p75 | qd p25/p50/p75 | category |
| --- | --- | --- | --- | --- |
${robustnessRows}

**Robust across all ${1 + search.extraSearchSeeds.length} biological search seeds: ${robustnessResult.robust ? 'yes' : 'no'}.**
${
  robustnessResult.robust
    ? ''
    : `The category is **not** the same at every search seed (${Object.entries(robustnessResult.perSeed)
        .map(([seed, category]) => `seed ${seed}: ${category}`)
        .join(', ')}) -- the primary seed ${search.primarySearchSeed} result (**${primary.category}**) alone must not be
read as the study's headline without this disclosure: it is not robust across search seeds, and seeds
${search.extraSearchSeeds.join(', ')} compare against only ${search.rewiredSeedMatchedCount} seed-matched rewirings each.`
}

### Occupancy maps (search seed ${search.primarySearchSeed}, row = turning bin (bottom = most negative), column = coverage bin (left = least covered))

${renderOccupancyGrid('Biological (1 = occupied, 0 = empty)', biologicalOccupancy, (n) => String(n))}

${renderOccupancyGrid(`Rewired occupancy frequency (of ${rewiredGraphCountAtPrimary} rewirings)`, rewiredOccupancy, (n) => String(n))}

### Disconnected control (reference only, not part of the null)

| occupied | qd | span | heldoutOwnMedian | gpuArchiveSize | collisions |
| --- | --- | --- | --- | --- | --- |
| ${disconnected.occupied} | ${fmt(disconnected.qd)} | ${disconnected.span} | ${fmt(disconnected.heldoutOwnMedian)} | ${disconnected.gpuArchiveSize} | ${disconnected.collisions} |

The disconnected graph may produce degenerate behavior; it is reported as a reference point, not evaluated
against the predeclared categories above.

### Audit table

#### Biological, every search seed

${AUDIT_TABLE_HEADER}
${AUDIT_TABLE_DIVIDER}
${biologicalRows}

#### Rewired, every searched (rewiring seed, search seed) pair

${AUDIT_TABLE_HEADER}
${AUDIT_TABLE_DIVIDER}
${rewiredRows}

## Limitations

- This experiment covers this model only: the authored encoder, dynamics, arena, descriptors, and the atlas
  readout family. There is no biological claim about fly behavior.
- One search budget only (population ${search.population}, generations ${search.generations}, ticks
  ${search.ticks}); a larger budget was declined and may change the result.
- \`n = ${search.rewiredCount}\` rewirings at the primary seed is a ${resolutionPct(search.rewiredCount)} percentile resolution; only
  ${search.rewiredSeedMatchedCount} rewirings are seed-matched for the search-seed robustness check above
  (a ${resolutionPct(search.rewiredSeedMatchedCount)} percentile resolution at those seeds).
- MAP-Elites coverage depends on the search budget and the descriptor space; a different budget or descriptor
  set could occupy different cells.
- Every claim here is descriptive and bound to this model and this search budget; no causal claim is made
  about the real fly.

## Provenance

- \`sources.biologicalSha\`: \`${sources.biologicalSha}\`
- \`sources.rewiringNullSha\`: \`${sources.rewiringNullSha}\`
- \`sources.evaluatedSha256\`: \`${sources.evaluatedSha256}\`
- \`sources.atlasSha256\`: \`${sources.atlasSha256}\`
- \`sources.graphsIndexSha256\`: \`${sources.graphsIndexSha256}\`
- Producer: \`${sources.producer.script}\`, sourceSha256 \`${sources.producer.sourceSha256}\` (${sources.producer.dependencies.length} files)
- Host: ${artifact.host.arch} / node ${artifact.host.node}
`;
};
