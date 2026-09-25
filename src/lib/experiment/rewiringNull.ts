/**
 * `public/data/rewiring-null-v1.json` (WP4,
 * `.agents/plans/rewiring-null/02-authored-null-evaluation.md`'s "Artifact
 * shape (conceptual)"): every graph's per-seed-averaged authored-decoder
 * score, plus the 500-graph null set's summary statistics and a pre-binned
 * histogram (`src/lib/ui/NullHistogram.svelte` draws bars straight from
 * `bins`, never rebinning `rewired` itself). `trained` is WP3's follow-on
 * addition (the CEM-retrained-readout sample on 20 rewirings) and may not
 * exist yet — deliberately typed `unknown` rather than a guessed shape:
 * `null-report.ts`'s `trained` section had not landed as of this WP, so any
 * hand-authored field list here would be unverified against a real
 * producer. `NullHistogram.svelte` does not render anything from it at all
 * in this WP (round-2 dual review: an earlier version guessed at its shape
 * and rendered a strip from it, which was removed) — this artifact's own
 * validation below never depends on it either. Rendering `trained` belongs
 * to WP3, against `null-report.ts`'s real output.
 *
 * Split out of `./assets.ts` (thermo review, Important): this file's own
 * types/validators/loader were a self-contained ~320-line block that pushed
 * `assets.ts` past the 1000-line "god module" threshold and depended on
 * nothing from that file except the shared `ArenaManifest` type and the
 * `fetchAndVerifySidecarJson` fetch→sha256-verify→JSON.parse helper it
 * shares with `loadPositions`.
 */

import { fetchAndVerifySidecarJson, type ArenaManifest } from './assets';

export interface RewiringNullScoreStats {
  score: number;
  median: number;
  std: number;
  ci: readonly [number, number];
}

export interface RewiringNullRewiredEntry extends RewiringNullScoreStats {
  seed: number;
  gzipSha256: string;
  acceptedSwaps: number;
  attempts: number;
}

export interface RewiringNullSummary {
  n: number;
  mean: number;
  median: number;
  std: number;
  p2_5: number;
  p97_5: number;
  iqr: number;
  degenerate: boolean;
}

export interface RewiringNullBins {
  /** `counts.length + 1` sorted (non-decreasing) bin boundaries. */
  edges: readonly number[];
  /** One non-negative integer count per bin; `counts.length === edges.length - 1`. */
  counts: readonly number[];
}

export interface RewiringNullArtifact {
  version: number;
  condition: string;
  seeds: { start: number; count: number };
  ticks: number;
  substeps: number;
  sourceGraphSha256: string;
  rewireSourceSha256: string;
  shards: number;
  biological: RewiringNullScoreStats;
  disconnected: RewiringNullScoreStats;
  /** Sorted by seed (0…499); the 500-graph null set `NullHistogram.svelte` bins bars from. */
  rewired: readonly RewiringNullRewiredEntry[];
  null: RewiringNullSummary;
  /** Biological's empirical percentile in the null set, in `[0, 1]` (e.g. `0` means biological scored below every rewired graph). */
  bioPercentile: number;
  pLow: number;
  pHigh: number;
  bins: RewiringNullBins;
  /** WP3's trained-sample section; see this interface's own doc comment. */
  trained?: unknown;
}

/**
 * `'absent'`/`'unavailable'`/`'invalid'` are deliberately distinct (thermo
 * review, Suggestion): `'absent'` means the manifest simply has no
 * `rewiringNull` entry (nothing was ever shipped); `'unavailable'` means a
 * genuine runtime failure — a fetch/network error, or an unexpected
 * exception anywhere in the load chain (`controller.ts`'s leading `.catch`)
 * — neither of which is a claim about the artifact's integrity;
 * `'invalid'` is reserved for an artifact that was actually fetched and
 * failed a real verification step (sha256, shape, or a cross-field/
 * cross-manifest consistency check). `LedgerPanel.svelte` words each of
 * these differently so "the network hiccuped" is never described to a
 * visitor as "failed verification".
 */
export type RewiringNullLoadResult =
  | { status: 'ok'; data: RewiringNullArtifact }
  | { status: 'absent'; reason: string }
  | { status: 'unavailable'; reason: string }
  | { status: 'invalid'; reason: string };

/** Exported as a small, reusable guard for any future caller that needs the same one-line predicate (round-2 dual review: the doc comment previously claimed `NullHistogram.svelte` imports this, but its own local copy — used for a since-removed `trained`-section narrowing — was deleted outright rather than replaced with this import). */
export const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

const isPositiveInteger = (value: unknown): value is number => Number.isInteger(value) && (value as number) > 0;

const isUnitInterval = (value: unknown): value is number => isFiniteNumber(value) && value >= 0 && value <= 1;

/** `ci[0] <= ci[1]` — a confidence interval whose bounds are swapped is itself a sign of a producer bug, not a real interval (dual review). */
const isFiniteCiPair = (value: unknown): value is readonly [number, number] =>
  Array.isArray(value) &&
  value.length === 2 &&
  isFiniteNumber(value[0]) &&
  isFiniteNumber(value[1]) &&
  (value[0] as number) <= (value[1] as number);

const isScoreStats = (value: unknown): value is RewiringNullScoreStats => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  // `std` is a standard deviation: never negative for real data (dual review).
  return (
    isFiniteNumber(v.score) && isFiniteNumber(v.median) && isFiniteNumber(v.std) && (v.std as number) >= 0 && isFiniteCiPair(v.ci)
  );
};

const isRewiredEntry = (value: unknown): value is RewiringNullRewiredEntry => {
  if (!isScoreStats(value)) return false;
  const v = value as unknown as Record<string, unknown>;
  return (
    isFiniteNumber(v.seed) &&
    typeof v.gzipSha256 === 'string' &&
    isFiniteNumber(v.acceptedSwaps) &&
    isFiniteNumber(v.attempts)
  );
};

const isSummary = (value: unknown): value is RewiringNullSummary => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    isPositiveInteger(v.n) &&
    isFiniteNumber(v.mean) &&
    isFiniteNumber(v.median) &&
    isFiniteNumber(v.std) &&
    (v.std as number) >= 0 &&
    isFiniteNumber(v.p2_5) &&
    isFiniteNumber(v.p97_5) &&
    (v.p2_5 as number) <= (v.p97_5 as number) &&
    isFiniteNumber(v.iqr) &&
    (v.iqr as number) >= 0 &&
    typeof v.degenerate === 'boolean'
  );
};

/**
 * `edges` must be sorted (non-decreasing — `null-report.ts` writes strictly
 * increasing equal-width edges, but non-decreasing is the weakest check that
 * still catches a shuffled/corrupted array without rejecting a legitimate
 * degenerate bin) and `counts` must have exactly one fewer entry than
 * `edges`, every one a finite, non-negative integer count (dual review:
 * "finite" alone let a fractional count like `0.5` through).
 */
const isBins = (value: unknown): value is RewiringNullBins => {
  if (typeof value !== 'object' || value === null) return false;
  const v = value as Record<string, unknown>;
  if (!Array.isArray(v.edges) || !Array.isArray(v.counts)) return false;
  if (v.edges.length < 2 || v.counts.length !== v.edges.length - 1) return false;
  for (let i = 0; i < v.edges.length; i += 1) {
    if (!isFiniteNumber(v.edges[i])) return false;
    if (i > 0 && (v.edges[i] as number) < (v.edges[i - 1] as number)) return false;
  }
  for (const count of v.counts) {
    if (!Number.isInteger(count) || (count as number) < 0) return false;
  }
  return true;
};

/**
 * The empirical rank statistics `scripts/null/null-stats.ts#rankStatistics`
 * computes server-side, over the artifact's own `rewired[].score`/
 * `biological.score` (thermo review S1: a hash-valid artifact could still
 * ship a `bioPercentile`/`pLow`/`pHigh` that disagrees with its own data —
 * the same class of bug `null-report.ts`'s own comments record happening
 * once before for the bin totals). Reimplemented here rather than imported:
 * `scripts/null` is a Node-only pipeline (not part of the browser bundle)
 * and this is a handful of comparisons over data the loader already has in
 * scope at validation time.
 */
const recomputeRankStatistics = (
  rewiredScores: readonly number[],
  bioScore: number
): { bioPercentile: number; pLow: number; pHigh: number } => {
  const n = rewiredScores.length;
  let kBelow = 0;
  let kEqual = 0;
  for (const score of rewiredScores) {
    if (score < bioScore) kBelow += 1;
    else if (score === bioScore) kEqual += 1;
  }
  return {
    bioPercentile: (kBelow + 0.5 * kEqual) / n,
    pLow: (kBelow + kEqual + 1) / (n + 1),
    pHigh: (n - kBelow + 1) / (n + 1)
  };
};

/** Tolerance for comparing a recomputed rank statistic against the artifact's own — generous relative to float64 round-trip error, tight enough to catch a real producer miscount. */
const RANK_STATISTIC_EPSILON = 1e-9;

/**
 * Structural validation (version 1, sorted bins, finite numbers) *plus* the
 * cross-field consistency the UI relies on but a per-field check alone
 * cannot catch (dual review, both reviewers, Important): a hash-valid
 * artifact can still contain internally-contradictory numbers (a
 * `bioPercentile` outside `[0, 1]`, a bin-count total that disagrees with
 * `null.n`/`rewired.length`, or a marker score that falls outside the
 * histogram's own domain and would render off-canvas while its legend entry
 * still claims it is shown). Mirrors `loadPositions`'s "never throws,
 * always return a reasoned status" contract. `trained` is deliberately left
 * unchecked (see `RewiringNullArtifact.trained`'s doc comment) — a
 * malformed `trained` section never fails the whole artifact.
 */
const validateRewiringNullShape = (value: unknown): { ok: true; data: RewiringNullArtifact } | { ok: false; reason: string } => {
  if (typeof value !== 'object' || value === null) {
    return { ok: false, reason: 'rewiring-null artifact is not a JSON object' };
  }
  const v = value as Record<string, unknown>;
  if (v.version !== 1) return { ok: false, reason: `rewiring-null artifact has unsupported version ${String(v.version)}` };
  if (typeof v.condition !== 'string') return { ok: false, reason: 'rewiring-null artifact is missing "condition"' };
  if (
    typeof v.seeds !== 'object' ||
    v.seeds === null ||
    !isFiniteNumber((v.seeds as Record<string, unknown>).start) ||
    !isPositiveInteger((v.seeds as Record<string, unknown>).count)
  ) {
    return { ok: false, reason: 'rewiring-null artifact has a malformed "seeds" field' };
  }
  if (!isFiniteNumber(v.ticks) || !isFiniteNumber(v.substeps) || !isFiniteNumber(v.shards)) {
    return { ok: false, reason: 'rewiring-null artifact is missing ticks/substeps/shards' };
  }
  if (typeof v.sourceGraphSha256 !== 'string' || typeof v.rewireSourceSha256 !== 'string') {
    return { ok: false, reason: 'rewiring-null artifact is missing sourceGraphSha256/rewireSourceSha256' };
  }
  if (!isScoreStats(v.biological)) return { ok: false, reason: 'rewiring-null artifact has a malformed "biological" field' };
  if (!isScoreStats(v.disconnected)) return { ok: false, reason: 'rewiring-null artifact has a malformed "disconnected" field' };
  if (!Array.isArray(v.rewired) || v.rewired.length === 0 || !v.rewired.every(isRewiredEntry)) {
    return { ok: false, reason: 'rewiring-null artifact has a malformed "rewired" array' };
  }
  const rewired = v.rewired as RewiringNullRewiredEntry[];
  // The bars claim to be "the N rewired graphs only" (`NullHistogram.svelte`'s
  // own non-negotiable) — a duplicated seed would double-count one graph
  // and silently misrepresent the null set.
  if (new Set(rewired.map((entry) => entry.seed)).size !== rewired.length) {
    return { ok: false, reason: 'rewiring-null artifact has duplicate seeds in "rewired"' };
  }
  if (!isSummary(v.null)) return { ok: false, reason: 'rewiring-null artifact has a malformed "null" summary field' };
  const summary = v.null as RewiringNullSummary;
  if (!isUnitInterval(v.bioPercentile) || !isUnitInterval(v.pLow) || !isUnitInterval(v.pHigh)) {
    return { ok: false, reason: 'rewiring-null artifact has bioPercentile/pLow/pHigh outside [0, 1]' };
  }
  if (!isBins(v.bins)) return { ok: false, reason: 'rewiring-null artifact has a malformed or unsorted "bins" field' };
  const bins = v.bins as RewiringNullBins;

  // The bars are drawn straight from `bins.counts` and captioned as "the N
  // rewired graphs" (`data.null.n`) — these three counts must agree, or the
  // chart and its own caption would each tell a different story.
  const binTotal = bins.counts.reduce((sum, count) => sum + count, 0);
  if (summary.n !== rewired.length || binTotal !== rewired.length) {
    return {
      ok: false,
      reason: `rewiring-null counts disagree (null.n=${summary.n}, rewired.length=${rewired.length}, sum(bins.counts)=${binTotal})`
    };
  }

  // Every marker `NullHistogram.svelte` draws (biological, disconnected,
  // and the shipped rewired-seed-0 control, when present) must fall inside
  // the histogram's own domain — `null-report.ts` widens the bin edges to
  // guarantee exactly this, so a marker outside `[edges[0], edges[last]]`
  // means the artifact is internally inconsistent, not merely that this
  // loader forgot to clamp it.
  const domainLow = bins.edges[0];
  const domainHigh = bins.edges[bins.edges.length - 1];
  const inDomain = (score: number): boolean => score >= domainLow && score <= domainHigh;
  const biological = v.biological as RewiringNullScoreStats;
  const disconnected = v.disconnected as RewiringNullScoreStats;
  const seed0 = rewired.find((entry) => entry.seed === 0);
  if (!inDomain(biological.score) || !inDomain(disconnected.score) || (seed0 && !inDomain(seed0.score))) {
    return {
      ok: false,
      reason: 'rewiring-null artifact has a marker (biological/disconnected/rewired-seed-0) outside the histogram bin domain'
    };
  }

  // (thermo review S1) The headline percentile the public-facing caption
  // reports is the one number this section's honesty claim rests on most
  // directly — recompute it (and pLow/pHigh, equally cheap) from the
  // artifact's own scores and reject a mismatch, rather than trusting a
  // separately-authored field that could silently drift from the data.
  const rank = recomputeRankStatistics(
    rewired.map((entry) => entry.score),
    biological.score
  );
  if (
    Math.abs(rank.bioPercentile - (v.bioPercentile as number)) > RANK_STATISTIC_EPSILON ||
    Math.abs(rank.pLow - (v.pLow as number)) > RANK_STATISTIC_EPSILON ||
    Math.abs(rank.pHigh - (v.pHigh as number)) > RANK_STATISTIC_EPSILON
  ) {
    return {
      ok: false,
      reason:
        'rewiring-null bioPercentile/pLow/pHigh do not match the rank statistics recomputed from the artifact\'s own ' +
        `rewired/biological scores (recomputed bioPercentile=${rank.bioPercentile}, pLow=${rank.pLow}, ` +
        `pHigh=${rank.pHigh}; artifact has bioPercentile=${v.bioPercentile}, pLow=${v.pLow}, pHigh=${v.pHigh})`
    };
  }

  return { ok: true, data: value as RewiringNullArtifact };
};

/**
 * Fetch, sha256-verify, and structurally validate `rewiring-null-v1.json`
 * (WP4's counterpart to `loadTrainedReadoutArtifact` in `./assets.ts`).
 * Never throws — every failure mode is a returned `status`, matching
 * `TrainedReadoutLoadResult`'s and `PositionsLoadResult`'s "optional
 * presentation, not a Start gate" contract: a missing/tampered/malformed
 * null-distribution artifact only ever hides or degrades the ledger panel's
 * "Topology null distribution" section (`LedgerPanel.svelte`), never the
 * experiment itself.
 *
 * `dataBaseUrl` must be the same value the caller passes to
 * `loadArenaArtifacts`/`loadTrainedReadoutArtifact` (`ExperimentController#initialize`
 * passes `${import.meta.env.BASE_URL}data`) so this artifact resolves under
 * the app's real deployment base path too.
 */
export const loadRewiringNull = async (
  manifest: ArenaManifest,
  dataBaseUrl: string
): Promise<RewiringNullLoadResult> => {
  // `fetchAndVerifySidecarJson` (`./assets.ts`) owns the shared
  // fetch→sha256-verify→JSON.parse skeleton this used to hand-roll
  // alongside `loadPositions`'s identical copy (thermo-maintainability
  // review, Important); only the status mapping below is specific to this
  // artifact.
  const fetched = await fetchAndVerifySidecarJson(manifest.rewiringNull, dataBaseUrl, 'rewiring-null artifact');
  if (fetched.status === 'no-entry') {
    return { status: 'absent', reason: 'The manifest has no rewiringNull artifact entry.' };
  }
  if (fetched.status === 'fetch-error') {
    return { status: 'unavailable', reason: fetched.reason };
  }
  if (fetched.status === 'hash-mismatch' || fetched.status === 'parse-error') {
    return { status: 'invalid', reason: fetched.reason };
  }

  const validated = validateRewiringNullShape(fetched.parsed);
  if (!validated.ok) return { status: 'invalid', reason: validated.reason };
  const data = validated.data;

  // The sha256 check above only proves these bytes are the ones the
  // manifest's `rewiringNull` entry pins — it says nothing about whether
  // this artifact actually describes *this* manifest's graphs (a
  // hand-edited or merge-conflicted manifest could re-pin `rewiringNull` to
  // a null distribution computed against a different biological/rewired
  // graph, including a "shipped" seed-0 marker that is not really the
  // shipped control arm). `loadPositions` (`./assets.ts`) already runs the
  // equivalent staleness check for its own sidecar artifact
  // ("positions.graphSha256 does not match the manifest's compiled graph
  // gzip sha256") — this mirrors that precedent (dual review, Important).
  if (data.sourceGraphSha256 !== manifest.binarySha256) {
    return {
      status: 'invalid',
      reason: `rewiring-null sourceGraphSha256 ${data.sourceGraphSha256} does not match the manifest's biological graph (${manifest.binarySha256}) — stale artifact`
    };
  }

  // The producer (`null-report.ts`) refuses to write a report at all
  // without a seed-0 entry — "the paired comparison needs the shipped
  // control arm" — so an artifact that hash/shape-verifies but omits it is
  // itself a sign of a stale/hand-edited file, not a legitimate variant to
  // render with one fewer marker (thermo review S3: this used to silently
  // skip the cross-check below instead of rejecting the artifact).
  const seed0 = data.rewired.find((entry) => entry.seed === 0);
  if (!seed0) {
    return {
      status: 'invalid',
      reason: 'rewiring-null artifact is missing the shipped rewired-seed-0 control arm (rewired[].seed === 0)'
    };
  }
  const shippedSeed0GzipSha256 = manifest.rewiredArms.seed0?.gzipSha256;
  if (shippedSeed0GzipSha256 && seed0.gzipSha256 !== shippedSeed0GzipSha256) {
    return {
      status: 'invalid',
      reason: 'rewiring-null seed 0 does not match the shipped rewired control arm (rewiredArms.seed0)'
    };
  }

  return { status: 'ok', data };
};
