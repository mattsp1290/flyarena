import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync, sha256Hex } from '../training/fsio';
import { conditionRng, mean, pairedStats, type ConditionStats, type PairedStats } from '../training/stats';
import { assertConsistentInputs } from './intervention-report-validation';
import { graphStats, quantileIndex, rankStatistics, type RankStatistics } from './null-stats';
import type { NullDecoderKind } from './null-worker';
import type { NullGraphListEvaluationRaw } from './null-evaluate';

/**
 * `.agents/plans/pathway-interventions/03-evaluation.md`'s WP2 "Statistics"
 * section, and `.agents/plans/pathway-interventions/04-report-and-ledger.md`'s
 * "new (started in WP2)" `intervention-report.ts`. This file owns only the
 * *statistics* half of that plan: per-graph mean/CI/paired-difference/
 * null-percentile, the C/M/MQ control-arm distributions, P's/Q's rank
 * statistics, and the predeclared outcome category, computed mechanically
 * from `null-evaluate.ts --graph-list`'s raw per-seed `authored.json` output
 * plus the already-published 500-graph authored null
 * (`public/data/rewiring-null-v1.json`). It deliberately does **not** build
 * the public `pathwayInterventions` artifact, the ledger sentence, or
 * `docs/pathway-interventions-report.md` — those combine this file's output
 * with `attribution.json` (swap counts, `k`/`k_Q`) and WP3's `trained.json`,
 * which is WP4's job (`04-report-and-ledger.md`'s change-surface row for
 * this same file: "it combines `index.json`, `attribution.json`,
 * `authored.json`, and `trained.json`").
 *
 * Every statistic here is a pure function of its inputs (no clock, no
 * environment-dependent iteration order): `rankStatistics`/`graphStats`
 * already give byte-identical results given the same `--bootstrap-seed`
 * (`../training/stats.ts`'s `conditionRng` — see its own doc comment), and
 * this module never iterates a `Map`/`Set` when building output arrays
 * (always the `--graph-list` index's own `sortedGraphListEntries`-style
 * ascending-`id` order) — so running this CLI twice against the same inputs
 * produces byte-identical JSON, per `03-evaluation.md`'s "Running the
 * statistics twice gives byte-identical output on the Spark" acceptance
 * criterion.
 */

// ---------------------------------------------------------------------------
// Graph-list index info (kind + gzipSha256 per id)
// ---------------------------------------------------------------------------

/**
 * The predeclared intervention/control kinds `scripts/analysis/
 * interventions.py`'s `index.json` labels every non-biological graph-list
 * entry with (`00-overview.md`'s "Predeclared interventions and
 * prediction"). `R` is included in the type even though this study's own
 * biological graph produced zero R-eligible edges (`02-intervention-graphs.md`'s
 * "the explanation shows input-restricted in-degree 0.0, so the list is
 * expected to be empty" — confirmed empty in the real `index.json` this run
 * consumes) so a future study where R *is* non-empty doesn't need a type
 * change here. An `R` graph, if present, is reported per-graph only (in
 * `InterventionStatistics.graphs`) — it is never categorized: only `P`/`Q`
 * feed `evaluateCategory`/`evaluateChannelSpecific`, and only `C`/`M`/`MQ`
 * feed the control-arm distributions.
 */
export const GRAPH_KINDS = ['P', 'Q', 'C', 'M', 'MQ', 'R'] as const;
export type GraphKind = (typeof GRAPH_KINDS)[number];

const isGraphKind = (value: unknown): value is GraphKind =>
  typeof value === 'string' && (GRAPH_KINDS as readonly string[]).includes(value);

/** Per-id info this module reads from `scripts/analysis/interventions.py`'s `index.json` — its `kind` (for arm membership) and `gzipSha256` (to detect a stale `authored.json` scored against a different graph file than the current index lists). */
export interface GraphListIndexEntryInfo {
  readonly kind: GraphKind;
  readonly gzipSha256: string;
}

/**
 * `readGraphListIndexInfo`'s full return value: the per-id map, plus the
 * index's own top-level `controlCount` (`scripts/analysis/interventions.py`'s
 * `--control-count`, 100 in this study's real run) — `assertConsistentInputs`
 * uses it to check the C/M/MQ arms actually have `controlCount` graphs each,
 * not merely "at least one" (`armDistribution`'s own floor). Without this, a
 * dev/fixture/truncated index (or one built with a smaller `--control-count`)
 * would silently compute a category at an unplanned, coarser percentile
 * resolution — `quantileIndex(5, 0.95) = 4` (the arm maximum) is this study's
 * own *trained*-decoder cutoff rule, not the authored one (a dual-review
 * finding).
 */
export interface GraphListIndexInfo {
  readonly entries: ReadonlyMap<string, GraphListIndexEntryInfo>;
  readonly controlCount: number;
}

/**
 * Parses the raw JSON text of `scripts/analysis/interventions.py`'s
 * `index.json` (already-read bytes, `utf8`-decoded) into `id ->
 * {kind, gzipSha256}` plus `controlCount`. Split out from
 * `readGraphListIndexInfo` (which reads the file itself) so
 * `runInterventionReport` can parse the exact same bytes it hashed into
 * `InterventionStatistics.inputs.indexSha256` — reading the file a second
 * time here would let `inputs.indexSha256` describe different bytes than
 * what was actually parsed if the file changed between the two reads (a
 * dual-review finding: a TOCTOU gap that defeats the whole point of
 * recording that hash for a downstream consumer to verify against).
 * Throws on an entry with a missing/unrecognized `kind`, or a duplicate
 * `id` — matching `null-evaluate.ts`'s `readGraphListIndex`'s own strict
 * rejection of a repeated id (an earlier version of this function tolerated
 * an exact repeat as long as the `kind` agreed; the two readers of the same
 * file disagreeing on what counts as valid was a maintainability review
 * finding).
 */
export const parseGraphListIndexInfo = (text: string, label: string): GraphListIndexInfo => {
  const parsed = JSON.parse(text) as {
    entries?: readonly { id?: unknown; kind?: unknown; gzipSha256?: unknown }[];
    controlCount?: unknown;
  };
  if (!Array.isArray(parsed.entries) || parsed.entries.length === 0) {
    throw new Error(`intervention-report: ${label} has no entries`);
  }
  if (typeof parsed.controlCount !== 'number' || !Number.isInteger(parsed.controlCount) || parsed.controlCount <= 0) {
    throw new Error(`intervention-report: ${label} is missing a positive integer controlCount`);
  }
  const entries = new Map<string, GraphListIndexEntryInfo>();
  for (const entry of parsed.entries) {
    if (
      typeof entry.id !== 'string' ||
      entry.id.length === 0 ||
      !isGraphKind(entry.kind) ||
      typeof entry.gzipSha256 !== 'string'
    ) {
      throw new Error(`intervention-report: ${label} has a malformed entry: ${JSON.stringify(entry)}`);
    }
    if (entries.has(entry.id)) {
      throw new Error(`intervention-report: ${label} lists id "${entry.id}" more than once`);
    }
    entries.set(entry.id, { kind: entry.kind, gzipSha256: entry.gzipSha256 });
  }
  return { entries, controlCount: parsed.controlCount };
};

/** Reads and parses `indexPath` — see `parseGraphListIndexInfo`'s doc comment for why the two are split. */
export const readGraphListIndexInfo = (indexPath: string): GraphListIndexInfo =>
  parseGraphListIndexInfo(readFileSync(indexPath, 'utf8'), indexPath);

// ---------------------------------------------------------------------------
// Published 500-graph authored null
// ---------------------------------------------------------------------------

export interface PublishedNull {
  readonly biologicalScore: number;
  /** The 500 rewired graphs' mean `movementScore` (`rewiring-null-v1.json` `rewired[].score`) — the null set every graph in this study is ranked against. */
  readonly scores: readonly number[];
  readonly sourceGraphSha256: string;
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly substeps: number;
}

/** Parses already-read JSON text — see `parseGraphListIndexInfo`'s doc comment for why (the same TOCTOU reasoning applies to every input this module hashes into `InterventionStatistics.inputs`). */
export const parsePublishedNull = (text: string, label: string): PublishedNull => {
  const parsed = JSON.parse(text) as {
    biological?: { score?: unknown };
    rewired?: readonly { score?: unknown }[];
    sourceGraphSha256?: unknown;
    seeds?: { start?: unknown; count?: unknown };
    ticks?: unknown;
    substeps?: unknown;
  };
  const biologicalScore = parsed.biological?.score;
  if (typeof biologicalScore !== 'number' || !Number.isFinite(biologicalScore)) {
    throw new Error(`intervention-report: ${label} is missing a finite biological.score`);
  }
  if (!Array.isArray(parsed.rewired) || parsed.rewired.length === 0) {
    throw new Error(`intervention-report: ${label} has no rewired entries`);
  }
  const scores = parsed.rewired.map((entry, i) => {
    if (typeof entry.score !== 'number' || !Number.isFinite(entry.score)) {
      throw new Error(`intervention-report: ${label} rewired[${i}].score is not a finite number`);
    }
    return entry.score;
  });
  if (typeof parsed.sourceGraphSha256 !== 'string') {
    throw new Error(`intervention-report: ${label} is missing sourceGraphSha256`);
  }
  if (typeof parsed.seeds?.start !== 'number' || typeof parsed.seeds?.count !== 'number') {
    throw new Error(`intervention-report: ${label} is missing seeds.start/seeds.count`);
  }
  if (typeof parsed.ticks !== 'number') {
    throw new Error(`intervention-report: ${label} is missing ticks`);
  }
  if (typeof parsed.substeps !== 'number') {
    throw new Error(`intervention-report: ${label} is missing substeps`);
  }
  return {
    biologicalScore,
    scores,
    sourceGraphSha256: parsed.sourceGraphSha256,
    seeds: { start: parsed.seeds.start, count: parsed.seeds.count },
    ticks: parsed.ticks,
    substeps: parsed.substeps
  };
};

/** Reads and parses `path` — see `parsePublishedNull`'s doc comment for why the two are split. */
export const readPublishedNull = (path: string): PublishedNull => parsePublishedNull(readFileSync(path, 'utf8'), path);

/**
 * `.agents/plans/pathway-interventions/03-evaluation.md`'s "the biological
 * reproduction check matches exactly" acceptance criterion: re-scoring
 * biological inside the `--graph-list` run must equal
 * `rewiring-null-v1.json`'s own published `biological.score` (both are the
 * mean `movementScore` over the identical 100 held-out seeds, same ticks,
 * same decoder, same source graph). `computedScore` and `publishedScore` are
 * always returned (never only the boolean) so a mismatch is diagnosable from
 * the report output alone. This check alone is only an indirect proxy for
 * "these two runs describe the same experiment" — `assertConsistentInputs`
 * (below) checks the rest (decoder, seeds/ticks, source graph sha) directly,
 * and `runInterventionReport` refuses to write a category when this check
 * fails, unless `--allow-reproduction-mismatch` is passed.
 */
export interface BiologicalReproductionCheck {
  readonly computedScore: number;
  readonly publishedScore: number;
  readonly matches: boolean;
}

export const checkBiologicalReproduction = (
  biologicalMovementScores: readonly number[],
  publishedNull: Readonly<PublishedNull>
): BiologicalReproductionCheck => {
  const computedScore = mean(biologicalMovementScores);
  return {
    computedScore,
    publishedScore: publishedNull.biologicalScore,
    matches: computedScore === publishedNull.biologicalScore
  };
};

// ---------------------------------------------------------------------------
// Per-graph statistics
// ---------------------------------------------------------------------------

/** Where a graph's score falls in the published 500-graph authored null — `rankStatistics`'s own shape, field-for-field (this module never renames `bioPercentile`/`kBelow`/`kEqual`/`pLow`/`pHigh`; only the *subject* differs — any graph's score, not only biological's). Descriptive only: the predeclared category is decided against `publishedNullFloor` (a quantile *value*), not this percentile — see `evaluateCategory`'s doc comment. */
export type PublishedNullRank = RankStatistics;

export interface GraphOutcomeEntry {
  readonly id: string;
  readonly kind: GraphKind;
  readonly n: number;
  readonly mean: number;
  readonly median: number;
  readonly std: number;
  readonly ci95: readonly [number, number];
  /** Absent only for the biological graph itself (paired against itself is meaningless) — every other graph shares biological's held-out seeds by construction (`--graph-list --biological` scores everything on one shared seed list, verified by `assertConsistentInputs`). */
  readonly pairedVsBiological?: PairedStats;
  readonly publishedNullRank: PublishedNullRank;
}

export const toGraphOutcomeEntry = (
  id: string,
  kind: GraphKind,
  movementScore: readonly number[],
  biologicalMovementScore: readonly number[] | undefined,
  publishedNull: Readonly<PublishedNull>,
  bootstrapSeed: number,
  bootstrapResamples: number
): GraphOutcomeEntry => {
  const stats: ConditionStats = graphStats(movementScore, bootstrapSeed, id, bootstrapResamples);
  return {
    id,
    kind,
    n: stats.n,
    mean: stats.mean,
    median: stats.median,
    std: stats.std,
    ci95: stats.ci95,
    ...(biologicalMovementScore
      ? {
          pairedVsBiological: pairedStats(
            movementScore,
            biologicalMovementScore,
            bootstrapResamples,
            conditionRng(bootstrapSeed, `paired|${id}-vs-biological`)
          )
        }
      : {}),
    publishedNullRank: rankStatistics(publishedNull.scores, stats.mean)
  };
};

// ---------------------------------------------------------------------------
// Control-arm distributions (C, M, MQ)
// ---------------------------------------------------------------------------

export interface ArmDistribution {
  readonly n: number;
  /** Sorted ascending. */
  readonly scores: readonly number[];
  readonly p5: number;
  readonly p50: number;
  readonly p95: number;
}

export const armDistribution = (meanScores: readonly number[]): ArmDistribution => {
  if (meanScores.length === 0) throw new Error('intervention-report: armDistribution requires at least one score');
  const scores = [...meanScores].sort((a, b) => a - b);
  const n = scores.length;
  return {
    n,
    scores,
    p5: scores[quantileIndex(n, 0.05)],
    p50: scores[quantileIndex(n, 0.5)],
    p95: scores[quantileIndex(n, 0.95)]
  };
};

// ---------------------------------------------------------------------------
// Predeclared categories (00-overview.md, authored decoder)
// ---------------------------------------------------------------------------

export const NULL_FLOOR_PERCENTILE = 0.25;

/**
 * The published null's 25th-percentile **value** (not a percentile-rank
 * statistic of some other score): `sorted(publishedNull.scores)[quantileIndex(n, 0.25)]`.
 * This is the same low-tail-floor quantile convention
 * `explain_stats.quantile_index`/`null-stats.ts`'s `quantileIndex` use
 * everywhere else in this study, including
 * `scripts/analysis/interventions.py`'s own `_regenerate_null_targets`
 * (which uses it for *its* stopping-rule targets — but those targets are
 * the transfer-matrix entries `T[thrust, rightClearance]`/
 * `T[thrust, forwardClearance]`, not `movementScore`; P's greedy swap
 * search never sees or optimizes toward this movementScore null at all, so
 * the only real link to WP1 is the shared quantile *convention*, not any
 * claim that P was constructed to land near this specific threshold — a
 * dual-review finding on an earlier version of this doc comment, which
 * incorrectly implied the latter).
 *
 * This module deliberately does **not** compare P's *empirical
 * percentile-rank* within the null (`rankStatistics(...).bioPercentile`, a
 * `(kBelow + 0.5*kEqual)/n` mid-rank statistic) against `0.25` — an earlier
 * version of `evaluateCategory` did exactly that, and it is a different
 * quantity that disagrees with this quantile-value definition for any score
 * strictly between `sorted[124]` and `sorted[125]` (`n = 500`), which could
 * flip the headline category right at that boundary (a separate dual-review
 * finding). The plan's "at or above the null's 25th percentile" is read as
 * a comparison against a null *value*, per this convention.
 */
export const publishedNullFloorValue = (publishedNull: Readonly<PublishedNull>): number => {
  const sorted = [...publishedNull.scores].sort((a, b) => a - b);
  return sorted[quantileIndex(sorted.length, NULL_FLOOR_PERCENTILE)];
};

export type OutcomeCategory = 'pathway-supported' | 'edge-class-effect' | 'generic-rewiring-effect' | 'not-supported';

/**
 * `00-overview.md`'s "Predeclared outcome categories (authored decoder)",
 * evaluated mechanically from already-computed statistics: `nullFloorScore`
 * is `publishedNullFloorValue(publishedNull)` (the plan's "at or above the
 * null's 25th percentile" — `pScore < nullFloorScore` fails, so an *equal*
 * score passes, matching "at or above"), and `cArm.p95`/`mArm.p95` are the
 * C/M arms' own `armDistribution` 95th-percentile *values* (per the plan's
 * literal "above the 95th percentile of both the C and M distributions").
 */
export const evaluateCategory = (
  pScore: number,
  nullFloorScore: number,
  cArm: Readonly<ArmDistribution>,
  mArm: Readonly<ArmDistribution>
): OutcomeCategory => {
  if (pScore < nullFloorScore) return 'not-supported';
  const aboveC = pScore > cArm.p95;
  const aboveM = pScore > mArm.p95;
  if (!aboveC) return 'generic-rewiring-effect';
  return aboveM ? 'pathway-supported' : 'edge-class-effect';
};

/**
 * `00-overview.md`'s "Channel-specific (modifier, authored decoder only)":
 * Q **above** the null's 25th percentile and **above** the 95th percentile
 * of its own size-matched class control MQ (the plan's literal wording is
 * "above" here, not "at or above" as for P's own floor test — both
 * comparisons use strict `>`). Authored-decoder only, and Q is never
 * compared against C or M (`03-evaluation.md`'s own wording) — this function
 * takes only `mqArm`, with no `cArm`/`mArm` parameter to make that
 * structurally impossible to get wrong at a call site.
 */
export const evaluateChannelSpecific = (
  qScore: number,
  nullFloorScore: number,
  mqArm: Readonly<ArmDistribution>
): boolean => qScore > nullFloorScore && qScore > mqArm.p95;

// ---------------------------------------------------------------------------
// Full report
// ---------------------------------------------------------------------------

export interface PArmResult {
  readonly id: 'P';
  readonly score: number;
  /** Descriptive only (not the category gate — see `publishedNullFloor`): P's empirical percentile-rank within the published 500-graph null. */
  readonly percentileInPublishedNull: number;
  /** `(k+1)/(n+1)` rank statistic of P among the C arm — `rankStatistics(C.scores, P.score).pHigh` (`03-evaluation.md`'s own wording: "P's ranks among C and among M ... use the (k+1)/(n+1) statistic"). Smaller means P ranks higher: `1/(n+1)` means P's score exceeds every C draw. Ties count against P. */
  readonly pRankAmongC: number;
  /** Same statistic, against the M arm. */
  readonly pRankAmongM: number;
  readonly category: OutcomeCategory;
}

export interface QArmResult {
  readonly id: 'Q';
  readonly score: number;
  /** Descriptive only (not the category gate — see `publishedNullFloor`): Q's empirical percentile-rank within the published 500-graph null. */
  readonly percentileInPublishedNull: number;
  /** `(k+1)/(n+1)` rank statistic of Q among its own size-matched class control MQ (smaller means Q ranks higher; see `PArmResult.pRankAmongC`'s doc comment). Q is never ranked against C or M. */
  readonly qRankAmongMQ: number;
  readonly channelSpecific: boolean;
}

export interface InterventionStatistics {
  readonly version: 1;
  readonly decoder: NullDecoderKind;
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly bootstrap: { readonly seed: number; readonly resamples: number };
  /** sha256 of each input file's raw bytes, so a downstream consumer (WP4) can verify `statistics.json` was built from the `authored.json`/`index.json`/published-null files it currently has on disk, not stale copies. */
  readonly inputs: {
    readonly authoredSha256: string;
    readonly indexSha256: string;
    readonly publishedNullSha256: string;
  };
  readonly biologicalReproduction: BiologicalReproductionCheck;
  /** Every graph-list entry (P, Q, C000..C099, M1000..M1099, MQ2000..MQ2099), sorted by id ascending. */
  readonly graphs: readonly GraphOutcomeEntry[];
  readonly controls: {
    readonly C: ArmDistribution;
    readonly M: ArmDistribution;
    readonly MQ: ArmDistribution;
  };
  /** The published null's 25th-percentile *value* (`publishedNullFloorValue`) — the single floor both P's and Q's category decisions gate on. */
  readonly publishedNullFloor: number;
  readonly p: PArmResult;
  readonly q: QArmResult;
  readonly host: { readonly arch: string; readonly node: string };
  /**
   * `raw.evaluatorGitRev` passed through — see `NullGraphListEvaluationRaw.evaluatorGitRev`'s
   * own doc comment (`null-evaluate.ts`) for why this exists (WP4 of
   * `.agents/plans/pathway-interventions`, thermo-methodology review I2).
   * Defaults to `null` (not a throw) when reading an `authored.json`
   * produced before this field existed — additive, matching this module's
   * own tolerant-of-older-inputs precedent elsewhere (e.g. `NullEvaluationRaw.decoder`'s
   * "absent on any file produced before this field existed" convention).
   */
  readonly evaluatorGitRev: string | null;
  /**
   * Present (and `true`) only when `runInterventionReport` wrote this file
   * despite a failed `biologicalReproduction` check, via
   * `--allow-reproduction-mismatch` — never set by `buildInterventionStatistics`
   * itself (a pure function with no notion of that CLI flag). A downstream
   * consumer (WP4) should refuse to publish any `statistics.json` carrying
   * this field, rather than relying on remembering to also check
   * `biologicalReproduction.matches` (a dual-review finding: a diagnostic
   * override file was otherwise indistinguishable from a clean run except by
   * that one nested boolean).
   */
  readonly diagnosticOnly?: true;
}

/** `id < id` string ordering — matches `null-evaluate.ts`'s `sortedGraphListEntries`, so this module's output key order is independent of `authored.json`'s own array order (itself already sorted the same way, but this does not assume that). */
const byIdAscending = <T extends { readonly id: string }>(a: T, b: T): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * Exactly one graph of `kind` must exist — used to locate P and Q. Looking
 * up by `kind` (not by `id === 'P'`/`id === 'Q'`) means a hand-edited or
 * corrupted index where the entry with `id: 'P'` was mislabeled `kind: 'C'`
 * is caught here (as "zero P graphs found") rather than silently letting
 * that graph be counted as both the intervention P *and* a C control, and
 * ranked against itself (a dual-review finding).
 */
const exactlyOneOfKind = (graphs: readonly GraphOutcomeEntry[], kind: GraphKind): GraphOutcomeEntry => {
  const matches = graphs.filter((g) => g.kind === kind);
  if (matches.length !== 1) {
    throw new Error(`intervention-report: expected exactly one graph of kind "${kind}", found ${matches.length}`);
  }
  return matches[0];
};

export const buildInterventionStatistics = (
  raw: Readonly<NullGraphListEvaluationRaw>,
  info: Readonly<GraphListIndexInfo>,
  publishedNull: Readonly<PublishedNull>,
  bootstrapSeed: number,
  bootstrapResamples: number,
  inputs: Readonly<InterventionStatistics['inputs']>
): InterventionStatistics => {
  if (raw.version !== 1) {
    throw new Error(`intervention-report: authored.json has unsupported version ${String(raw.version)}, expected 1`);
  }
  // Checked here (not only inside assertConsistentInputs) so this narrows
  // `raw.biological` for the rest of this function without a non-null
  // assertion -- assertConsistentInputs still re-checks it independently
  // (defense in depth for any other caller of that function).
  if (!raw.biological) {
    throw new Error(
      'intervention-report: authored.json has no biological section (run null-evaluate with --biological)'
    );
  }
  const biological = raw.biological;
  assertConsistentInputs(raw, info, publishedNull);
  const biologicalMovementScore = biological.movementScore;

  const graphs: GraphOutcomeEntry[] = [...raw.graphs]
    .sort(byIdAscending)
    .map((graph) => {
      const entryInfo = info.entries.get(graph.id);
      if (!entryInfo) throw new Error(`intervention-report: no kind found for graph id "${graph.id}" in the index`);
      return toGraphOutcomeEntry(
        graph.id,
        entryInfo.kind,
        graph.movementScore,
        biologicalMovementScore,
        publishedNull,
        bootstrapSeed,
        bootstrapResamples
      );
    });

  const meanScoresOfKind = (kind: GraphKind): number[] =>
    graphs.filter((g) => g.kind === kind).map((g) => g.mean);

  const cArm = armDistribution(meanScoresOfKind('C'));
  const mArm = armDistribution(meanScoresOfKind('M'));
  const mqArm = armDistribution(meanScoresOfKind('MQ'));

  const pGraph = exactlyOneOfKind(graphs, 'P');
  const qGraph = exactlyOneOfKind(graphs, 'Q');
  // exactlyOneOfKind locates P/Q by `kind`, not by `id === 'P'`/`id === 'Q'`
  // (see its own doc comment) -- this closes the remaining gap: an index
  // where the *other* id was mislabeled `kind: 'P'` would otherwise pass
  // "exactly one P graph found" while reporting a stranger's score under
  // `p.id: 'P'` (a dual-review finding).
  if (pGraph.id !== 'P' || qGraph.id !== 'Q') {
    throw new Error(
      `intervention-report: the graph(s) with kind "P"/"Q" have id(s) "${pGraph.id}"/"${qGraph.id}", expected "P"/"Q"`
    );
  }

  const pRankAmongC = rankStatistics(cArm.scores, pGraph.mean).pHigh;
  const pRankAmongM = rankStatistics(mArm.scores, pGraph.mean).pHigh;
  const qRankAmongMQ = rankStatistics(mqArm.scores, qGraph.mean).pHigh;

  const nullFloor = publishedNullFloorValue(publishedNull);
  const category = evaluateCategory(pGraph.mean, nullFloor, cArm, mArm);
  const channelSpecific = evaluateChannelSpecific(qGraph.mean, nullFloor, mqArm);

  return {
    version: 1,
    decoder: raw.decoder,
    // Built explicitly (not `seeds: raw.seeds`/`host: raw.host` passthrough)
    // so this output's schema is fixed by this module, not by whatever
    // extra keys a future/hand-edited authored.json's `seeds`/`host` objects
    // happen to carry (a dual-review finding).
    seeds: { start: raw.seeds.start, count: raw.seeds.count },
    ticks: raw.ticks,
    bootstrap: { seed: bootstrapSeed, resamples: bootstrapResamples },
    inputs,
    biologicalReproduction: checkBiologicalReproduction(biologicalMovementScore, publishedNull),
    graphs,
    controls: { C: cArm, M: mArm, MQ: mqArm },
    publishedNullFloor: nullFloor,
    p: {
      id: 'P',
      score: pGraph.mean,
      percentileInPublishedNull: pGraph.publishedNullRank.bioPercentile,
      pRankAmongC,
      pRankAmongM,
      category
    },
    q: {
      id: 'Q',
      score: qGraph.mean,
      percentileInPublishedNull: qGraph.publishedNullRank.bioPercentile,
      qRankAmongMQ,
      channelSpecific
    },
    host: { arch: raw.host.arch, node: raw.host.node },
    evaluatorGitRev: raw.evaluatorGitRev ?? null
  };
};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '../..');

const DEFAULT_AUTHORED = resolve(repoRoot, 'training/runs/interventions/authored.json');
const DEFAULT_INDEX = resolve(repoRoot, 'training/runs/interventions/index.json');
const DEFAULT_PUBLISHED_NULL = resolve(repoRoot, 'public/data/rewiring-null-v1.json');
/** Exported so tests can exercise the `diagnosticOnly`-vs-default-`--out` guard directly, without needing a real 302-graph run just to get there (matches `null-report.ts`'s `DEFAULT_OUT` export convention). */
export const DEFAULT_OUT = resolve(repoRoot, 'training/runs/interventions/statistics.json');

/** `'PATH'` as a fixed default seed (arbitrary but stable across runs) — this study's own bootstrap seed, independent of `null-report.ts`'s `DEFAULT_BOOTSTRAP_SEED` ('NULL'), matching that file's "fixed default" convention (`03-evaluation.md`: "`--bootstrap-seed` with a fixed default"). */
const DEFAULT_BOOTSTRAP_SEED = 0x50415448;
const DEFAULT_BOOTSTRAP_RESAMPLES = 10000;

export interface InterventionReportArgs {
  readonly authored: string;
  readonly index: string;
  readonly publishedNull: string;
  readonly out: string;
  readonly bootstrapSeed: number;
  readonly bootstrapResamples: number;
  /** Escape hatch for diagnosis only: without it, `runInterventionReport` refuses to write `--out` when `biologicalReproduction.matches` is false (a dual-review finding — this acceptance gate was previously advisory only: the CLI printed `matches=false` and still wrote a category). */
  readonly allowReproductionMismatch: boolean;
}

export const parseInterventionReportArgs = (argv: readonly string[]): InterventionReportArgs => {
  let authored = DEFAULT_AUTHORED;
  let index = DEFAULT_INDEX;
  let publishedNull = DEFAULT_PUBLISHED_NULL;
  let out = DEFAULT_OUT;
  let bootstrapSeed = DEFAULT_BOOTSTRAP_SEED;
  let bootstrapResamples = DEFAULT_BOOTSTRAP_RESAMPLES;
  let allowReproductionMismatch = false;

  let i = 0;
  while (i < argv.length) {
    const flag = argv[i];
    if (flag === '--authored') {
      authored = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--index') {
      index = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--null') {
      publishedNull = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--out') {
      out = resolve(process.cwd(), requireValue(flag, argv[i + 1]));
      i += 2;
    } else if (flag === '--bootstrap-seed') {
      bootstrapSeed = requireNonNegativeInt(flag, argv[i + 1]);
      i += 2;
    } else if (flag === '--bootstrap-resamples') {
      bootstrapResamples = requirePositiveInt(flag, argv[i + 1]);
      i += 2;
    } else if (flag === '--allow-reproduction-mismatch') {
      allowReproductionMismatch = true;
      i += 1;
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  // `--out` overwriting one of its own inputs would replace a multi-hour
  // `authored.json` (or `index.json`/the published null) with the much
  // smaller statistics output -- checked before any file is touched (a
  // dual-review finding).
  for (const input of [authored, index, publishedNull]) {
    if (resolve(out) === resolve(input)) {
      throw new Error(`intervention-report: --out must not overwrite an input file (${input})`);
    }
  }

  return { authored, index, publishedNull, out, bootstrapSeed, bootstrapResamples, allowReproductionMismatch };
};

export const runInterventionReport = (
  args: Readonly<InterventionReportArgs>
): { readonly out: string; readonly statistics: InterventionStatistics } => {
  // Each file is read exactly once: the same bytes are both hashed (into
  // `inputs.*Sha256`) and parsed, via `parseGraphListIndexInfo`/
  // `parsePublishedNull` taking already-read text rather than re-reading
  // from disk -- a second `readFileSync` per file would let the recorded
  // hash describe different bytes than what was actually parsed if the file
  // changed in between (a dual-review finding: a TOCTOU gap that would
  // defeat the whole point of `inputs`, which a downstream consumer uses to
  // verify it isn't looking at a stale copy).
  const authoredBytes = readFileSync(args.authored);
  const indexBytes = readFileSync(args.index);
  const publishedNullBytes = readFileSync(args.publishedNull);

  const raw = JSON.parse(authoredBytes.toString('utf8')) as NullGraphListEvaluationRaw;
  const info = parseGraphListIndexInfo(indexBytes.toString('utf8'), args.index);
  const publishedNull = parsePublishedNull(publishedNullBytes.toString('utf8'), args.publishedNull);

  const statistics = buildInterventionStatistics(raw, info, publishedNull, args.bootstrapSeed, args.bootstrapResamples, {
    authoredSha256: sha256Hex(authoredBytes),
    indexSha256: sha256Hex(indexBytes),
    publishedNullSha256: sha256Hex(publishedNullBytes)
  });

  if (!statistics.biologicalReproduction.matches && !args.allowReproductionMismatch) {
    throw new Error(
      'intervention-report: biological reproduction check failed ' +
        `(computed ${statistics.biologicalReproduction.computedScore}, ` +
        `published ${statistics.biologicalReproduction.publishedScore}) -- refusing to write a category computed ` +
        'against an incompatible null. Pass --allow-reproduction-mismatch to write anyway for diagnosis.'
    );
  }
  // `diagnosticOnly` is only ever added here, and only when the override was
  // actually exercised (a failed check plus the flag) -- passing
  // `--allow-reproduction-mismatch` on an otherwise-clean run must not mark
  // it, so a clean run's output is byte-identical whether or not the flag
  // was passed. `statistics` itself (from the pure `buildInterventionStatistics`)
  // never carries this field.
  const output: InterventionStatistics =
    args.allowReproductionMismatch && !statistics.biologicalReproduction.matches
      ? { ...statistics, diagnosticOnly: true }
      : statistics;

  // A diagnostic (`--allow-reproduction-mismatch`) run must never silently
  // clobber the canonical `statistics.json` at the default --out -- mirrors
  // `null-evaluate.ts`'s `guardCanonicalOutDefault` (itself added for the
  // same class of dual-review finding: a non-canonical run overwriting the
  // one path every other script reads by default). Scoped to `diagnosticOnly`
  // specifically, not every `--allow-reproduction-mismatch` invocation: the
  // flag is a no-op marker when the reproduction check actually passes (see
  // `diagnosticOnly`'s own construction above), and that case is a perfectly
  // ordinary canonical run.
  if (output.diagnosticOnly && resolve(args.out) === resolve(DEFAULT_OUT)) {
    throw new Error(
      `intervention-report: refusing to write a diagnosticOnly result to the default --out (${DEFAULT_OUT}) -- ` +
        'pass an explicit --out for this diagnostic run.'
    );
  }

  mkdirSync(dirname(args.out), { recursive: true });
  atomicWriteFileSync(args.out, JSON.stringify(output));
  return { out: args.out, statistics: output };
};

const main = (): void => {
  try {
    const args = parseInterventionReportArgs(process.argv.slice(2));
    const { out, statistics } = runInterventionReport(args);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing output.
    console.log(
      `intervention-report: wrote ${out}\n` +
        `biologicalReproduction.matches=${statistics.biologicalReproduction.matches} ` +
        `P.category=${statistics.p.category} Q.channelSpecific=${statistics.q.channelSpecific}`
    );
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- CLI tool: this is its user-facing error output.
    console.error(`intervention-report failed: ${message}`);
    process.exit(1);
  }
};

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
