import { quantileIndex } from '../null/null-stats';
import { average, COVERAGE_EDGES } from '../../src/lib/atlas/types';
import type { RepertoireCellResult } from './repertoire-task';

/**
 * WP2's predeclared repertoire metrics
 * (`.agents/plans/repertoire-null/00-overview.md`'s "Predeclared metrics"
 * section) and category logic ("Predeclared categories" section), as pure
 * functions over `repertoire-evaluate.ts`'s already-computed cell/held-out
 * data -- nothing here re-runs a search or an episode, so `repertoire-report.ts`
 * (WP3) can rerun this module against a cached `evaluated.json` and
 * reproduce byte-identical output, the same separation-of-concerns
 * `scripts/null/null-stats.ts`'s own doc comment describes for the
 * authored-null study.
 */

export type RepertoireCellMetricInput = Pick<RepertoireCellResult, 'cell' | 'quality' | 'heldoutOwn'>;

/** Occupied cells among 36, after TS rebinning and collision resolution -- `cells` already *is* the occupied set (`evaluateSearchOnGraph`'s `ordered` array, one entry per distinct occupied cell), so this is just its length. */
export const occupied = (cells: readonly RepertoireCellMetricInput[]): number => cells.length;

/** The sum over occupied cells of `max(0, quality)` -- a QD score with a fixed zero floor (predeclared, not offset by the empirical minimum). */
export const qd = (cells: readonly RepertoireCellMetricInput[]): number =>
  cells.reduce((sum, cell) => sum + Math.max(0, cell.quality), 0);

/** The number of coverage bins `cellFor` (`src/lib/atlas/types.ts`) packs into one `cell` index: `cellFor = binIndex(turning, TURN_EDGES) * COVERAGE_BIN_COUNT + binIndex(coverage, COVERAGE_EDGES)`, so this is also the multiplier that separates the turning bin from the coverage bin below. */
const COVERAGE_BIN_COUNT = COVERAGE_EDGES.length - 1;

/** The number of distinct coverage bins plus the number of distinct turning bins occupied, among *this* re-evaluation's TS-rebinned cells (never the GPU search's own archive, which can rebin candidates into different cells than TS re-evaluation does -- `gpuArchiveSize` vs `occupied` audits that gap). `cell.cell = turningBin * COVERAGE_BIN_COUNT + coverageBin` (`cellFor`'s own packing, above), so `cell % COVERAGE_BIN_COUNT` recovers the coverage bin and `Math.floor(cell / COVERAGE_BIN_COUNT)` the turning bin -- never re-derived from raw `coverage`/`turning` values, which could land on a bin boundary differently than `cellFor`'s own rebinning already did. */
export const span = (cells: readonly RepertoireCellMetricInput[]): number => {
  const coverageBins = new Set(cells.map((cell) => cell.cell % COVERAGE_BIN_COUNT));
  const turningBins = new Set(cells.map((cell) => Math.floor(cell.cell / COVERAGE_BIN_COUNT)));
  return coverageBins.size + turningBins.size;
};

/** The median over occupied cells of the searched graph's own held-out mean score (`heldoutOwn`, seeds 62001-62012, trained decoder on that graph). Reported, not categorized. */
export const heldoutOwnMedian = (cells: readonly RepertoireCellMetricInput[]): number => {
  if (cells.length === 0) throw new Error('repertoire-metrics: heldoutOwnMedian requires at least one cell');
  const means = cells
    .map((cell) => average(cell.heldoutOwn.map((metrics) => metrics.movementScore)))
    .sort((a, b) => a - b);
  const n = means.length;
  return n % 2 === 1 ? means[(n - 1) / 2] : (means[n / 2 - 1] + means[n / 2]) / 2;
};

export interface RewiredDistribution {
  readonly n: number;
  readonly p25: number;
  /** `null-stats.ts`'s `quantileIndex` convention, applied at `p = 0.5`: for an even `n` this is the upper-middle element (`sorted[n/2]`'s neighbor rule), not the mean of the two middle elements -- reported for descriptive display only, never used in `metricVerdict`/`categorize`'s category logic below. */
  readonly p50: number;
  readonly p75: number;
  readonly values: readonly number[];
}

/** `values`' 25th/50th/75th empirical percentiles, using `null-stats.ts`'s `quantileIndex` -- the same low-tail-floor/high-tail-ceil-minus-one convention the authored-null study's own percentiles use, applied here to the rewired repertoire distribution instead of a bootstrap resample. */
export const rewiredDistribution = (values: readonly number[]): RewiredDistribution => {
  if (values.length === 0) throw new Error('repertoire-metrics: rewiredDistribution requires at least one value');
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  return {
    n,
    p25: sorted[quantileIndex(n, 0.25)],
    p50: sorted[quantileIndex(n, 0.5)],
    p75: sorted[quantileIndex(n, 0.75)],
    values: sorted
  };
};

export type RepertoireCategory = 'wider' | 'narrower' | 'typical';

export interface MetricVerdict {
  readonly wider: boolean;
  readonly narrower: boolean;
  /** `p25 === p75 === bio`: a degenerate rewired distribution equal to the biological value on this metric alone -- neither `wider` nor `narrower` can ever hold for it (`00-overview.md`'s tie rule). */
  readonly tie: boolean;
}

/** One metric's wider/narrower/tie verdict, per `00-overview.md`'s predeclared tie rule: "Wider requires, on both metrics, bio >= p75 and bio > p25. Narrower requires, on both metrics, bio <= p25 and bio < p75." Applied here to one metric; `categorize` below combines both. */
export const metricVerdict = (bio: number, distribution: Readonly<RewiredDistribution>): MetricVerdict => ({
  wider: bio >= distribution.p75 && bio > distribution.p25,
  narrower: bio <= distribution.p25 && bio < distribution.p75,
  tie: distribution.p25 === distribution.p75 && distribution.p75 === bio
});

export interface CategoryResult {
  readonly category: RepertoireCategory;
  /** True if either metric's rewired distribution was degenerate and equal to the biological value (the predeclared tie annotation). Never itself changes which `category` is reported -- the ordinary `wider`/`narrower` conditions already resolve to `typical` in that case; this only records why. */
  readonly tie: boolean;
}

/**
 * `00-overview.md`'s predeclared category: **wider** only if biological is
 * at or above the rewired 75th percentile on *both* `occupied` and `qd`
 * (and strictly above the 25th, so a value sitting exactly at a degenerate
 * `p25 === p75` never counts); **narrower** is the mirror; otherwise
 * **typical**. "Wider and Narrower can never both hold" is a property of
 * these two conditions being mutually exclusive on a single metric already
 * (`bio >= p75 && bio > p25` and `bio <= p25 && bio < p75` cannot both hold
 * for finite `p25 <= p75`), so no separate guard is needed here.
 */
export const categorize = (
  occupiedVerdict: Readonly<MetricVerdict>,
  qdVerdict: Readonly<MetricVerdict>
): CategoryResult => {
  const category: RepertoireCategory =
    occupiedVerdict.wider && qdVerdict.wider
      ? 'wider'
      : occupiedVerdict.narrower && qdVerdict.narrower
        ? 'narrower'
        : 'typical';
  return { category, tie: occupiedVerdict.tie || qdVerdict.tie };
};

export interface RobustnessInput {
  readonly seed: number;
  readonly category: RepertoireCategory;
}

export interface RobustnessResult {
  readonly perSeed: Readonly<Record<number, RepertoireCategory>>;
  /** True only if every seed's category is identical (`00-overview.md`: "robust only if it is the same for all 5 biological seeds against the seed-matched rewired distribution"). */
  readonly robust: boolean;
}

/** Search-seed robustness over `perSeed`'s categories -- pure aggregation, no statistics of its own; each entry's `category` is expected to already have been computed by `categorize` against that seed's own (seed-matched, for 1730-1733) rewired distribution. */
export const robustness = (perSeed: readonly RobustnessInput[]): RobustnessResult => {
  if (perSeed.length === 0) throw new Error('repertoire-metrics: robustness requires at least one seed');
  const categories = new Set(perSeed.map((entry) => entry.category));
  return {
    perSeed: Object.fromEntries(perSeed.map((entry) => [entry.seed, entry.category])),
    robust: categories.size === 1
  };
};
