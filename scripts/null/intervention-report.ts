import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { requireNonNegativeInt, requirePositiveInt, requireValue } from '../training/cli';
import { atomicWriteFileSync } from '../training/fsio';
import { conditionRng, mean, pairedStats, type ConditionStats, type PairedStats } from '../training/stats';
import { graphStats, rankStatistics, type RankStatistics } from './null-stats';
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
// Graph-list index kind lookup
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
 * change here.
 */
export const GRAPH_KINDS = ['P', 'Q', 'C', 'M', 'MQ', 'R'] as const;
export type GraphKind = (typeof GRAPH_KINDS)[number];

const isGraphKind = (value: unknown): value is GraphKind =>
  typeof value === 'string' && (GRAPH_KINDS as readonly string[]).includes(value);

/**
 * `id -> kind` for every entry in `scripts/analysis/interventions.py`'s
 * `index.json`. Read independently of `null-evaluate.ts`'s own
 * `GraphListIndex`/`GraphListEntry` types (which only carry what
 * `null-evaluate.ts` itself needs — `id`/`path`/`gzipSha256`/`binarySha256`,
 * not `kind`) rather than threading `kind` through the evaluator's wire
 * protocol and raw output, which has no use for it. Throws on an entry with
 * a missing/unrecognized `kind`, or two entries sharing one `id` with
 * different kinds (a hand-edited or corrupted index) — silently picking one
 * would misclassify a control graph into the wrong arm with no error
 * anywhere downstream.
 */
export const readGraphKinds = (indexPath: string): ReadonlyMap<string, GraphKind> => {
  const parsed = JSON.parse(readFileSync(indexPath, 'utf8')) as {
    entries?: readonly { id?: unknown; kind?: unknown }[];
  };
  if (!Array.isArray(parsed.entries)) {
    throw new Error(`intervention-report: ${indexPath} has no entries`);
  }
  const kinds = new Map<string, GraphKind>();
  for (const entry of parsed.entries) {
    if (typeof entry.id !== 'string' || entry.id.length === 0 || !isGraphKind(entry.kind)) {
      throw new Error(`intervention-report: ${indexPath} has a malformed entry: ${JSON.stringify(entry)}`);
    }
    const existing = kinds.get(entry.id);
    if (existing !== undefined && existing !== entry.kind) {
      throw new Error(
        `intervention-report: ${indexPath} lists id "${entry.id}" with two different kinds (${existing}, ${entry.kind})`
      );
    }
    kinds.set(entry.id, entry.kind);
  }
  return kinds;
};

// ---------------------------------------------------------------------------
// Published 500-graph authored null
// ---------------------------------------------------------------------------

export interface PublishedNull {
  readonly biologicalScore: number;
  /** The 500 rewired graphs' mean `movementScore` (`rewiring-null-v1.json` `rewired[].score`) — the null set every graph in this study is ranked against. */
  readonly scores: readonly number[];
}

export const readPublishedNull = (path: string): PublishedNull => {
  const parsed = JSON.parse(readFileSync(path, 'utf8')) as {
    biological?: { score?: unknown };
    rewired?: readonly { score?: unknown }[];
  };
  const biologicalScore = parsed.biological?.score;
  if (typeof biologicalScore !== 'number') {
    throw new Error(`intervention-report: ${path} is missing biological.score`);
  }
  if (!Array.isArray(parsed.rewired) || parsed.rewired.length === 0) {
    throw new Error(`intervention-report: ${path} has no rewired entries`);
  }
  const scores = parsed.rewired.map((entry, i) => {
    if (typeof entry.score !== 'number') {
      throw new Error(`intervention-report: ${path} rewired[${i}].score is not a number`);
    }
    return entry.score;
  });
  return { biologicalScore, scores };
};

/**
 * `.agents/plans/pathway-interventions/03-evaluation.md`'s "the biological
 * reproduction check matches exactly" acceptance criterion: re-scoring
 * biological inside the `--graph-list` run must equal
 * `rewiring-null-v1.json`'s own published `biological.score` (both are the
 * mean `movementScore` over the identical 100 held-out seeds, same ticks,
 * same decoder, same source graph). `computedScore` and `publishedScore` are
 * always returned (never only the boolean) so a mismatch is diagnosable from
 * the report output alone.
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

/** Where a graph's score falls in the published 500-graph authored null — `rankStatistics`'s own shape, field-for-field (this module never renames `bioPercentile`/`kBelow`/`kEqual`/`pLow`/`pHigh`; only the *subject* differs — any graph's score, not only biological's). */
export type PublishedNullRank = RankStatistics;

export interface GraphOutcomeEntry {
  readonly id: string;
  readonly kind: GraphKind;
  readonly n: number;
  readonly mean: number;
  readonly median: number;
  readonly std: number;
  readonly ci95: readonly [number, number];
  /** Absent only for the biological graph itself (paired against itself is meaningless) — every other graph shares biological's held-out seeds by construction (`--graph-list --biological` scores everything on one shared seed list). */
  readonly pairedVsBiological?: PairedStats;
  readonly publishedNullRank: PublishedNullRank;
}

const toGraphOutcomeEntry = (
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

/**
 * The empirical quantile index into a length-`n` sorted array for
 * proportion `p` — the same low-tail-floor / high-tail-ceil-minus-one
 * (clamped) convention `null-stats.ts`'s own `quantileIndex` already uses,
 * reproduced here (rather than importing a private helper) so this module's
 * `--graph-list`-indexed statistics stay self-contained; both give the same
 * answer for the same `(n, p)` by construction.
 */
const quantileIndex = (n: number, p: number): number =>
  p <= 0.5 ? Math.floor(p * n) : Math.min(n - 1, Math.ceil(p * n) - 1);

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

export type OutcomeCategory = 'pathway-supported' | 'edge-class-effect' | 'generic-rewiring-effect' | 'not-supported';

/**
 * `00-overview.md`'s "Predeclared outcome categories (authored decoder)",
 * evaluated mechanically from already-computed statistics: `pPercentile` is
 * P's `publishedNullRank.percentile` (its empirical percentile within the
 * published 500-graph authored null — the plan's "at or above the null's
 * 25th percentile"), and `cP95`/`mP95` are the C/M arms' own `armDistribution`
 * 95th-percentile *values* (not `pHigh`/rank statistics — the category rule
 * compares P's raw score against the arm's empirical p95, per the plan's
 * literal "above the 95th percentile of both the C and M distributions").
 */
export const evaluateCategory = (
  pScore: number,
  pPercentileInPublishedNull: number,
  cArm: Readonly<ArmDistribution>,
  mArm: Readonly<ArmDistribution>
): OutcomeCategory => {
  if (pPercentileInPublishedNull < NULL_FLOOR_PERCENTILE) return 'not-supported';
  const aboveC = pScore > cArm.p95;
  const aboveM = pScore > mArm.p95;
  if (!aboveC) return 'generic-rewiring-effect';
  return aboveM ? 'pathway-supported' : 'edge-class-effect';
};

/**
 * `00-overview.md`'s "Channel-specific (modifier, authored decoder only)":
 * Q above the null's 25th percentile and above the 95th percentile of its
 * own size-matched class control MQ. Authored-decoder only, and Q is never
 * compared against C or M (`03-evaluation.md`'s own wording) — this function
 * takes only `mqArm`, with no `cArm`/`mArm` parameter to make that
 * structurally impossible to get wrong at a call site.
 */
export const evaluateChannelSpecific = (
  qScore: number,
  qPercentileInPublishedNull: number,
  mqArm: Readonly<ArmDistribution>
): boolean => qPercentileInPublishedNull >= NULL_FLOOR_PERCENTILE && qScore > mqArm.p95;

// ---------------------------------------------------------------------------
// Full report
// ---------------------------------------------------------------------------

export interface PArmResult {
  readonly id: 'P';
  readonly score: number;
  readonly percentileInPublishedNull: number;
  /** `(k+1)/(n+1)` rank statistic of P among the C arm — `rankStatistics(C.scores, P.score).pHigh` (`03-evaluation.md`'s own wording: "P's ranks among C and among M ... use the (k+1)/(n+1) statistic"). */
  readonly pRankAmongC: number;
  readonly pRankAmongM: number;
  readonly category: OutcomeCategory;
}

export interface QArmResult {
  readonly id: 'Q';
  readonly score: number;
  readonly percentileInPublishedNull: number;
  /** `(k+1)/(n+1)` rank statistic of Q among its own size-matched class control MQ. Q is never ranked against C or M. */
  readonly qRankAmongMQ: number;
  readonly channelSpecific: boolean;
}

export interface InterventionStatistics {
  readonly version: 1;
  readonly decoder: string;
  readonly seeds: { readonly start: number; readonly count: number };
  readonly ticks: number;
  readonly bootstrap: { readonly seed: number; readonly resamples: number };
  readonly biologicalReproduction: BiologicalReproductionCheck;
  /** Every graph-list entry (P, Q, C000..C099, M1000..M1099, MQ2000..MQ2099), sorted by id ascending. */
  readonly graphs: readonly GraphOutcomeEntry[];
  readonly controls: {
    readonly C: ArmDistribution;
    readonly M: ArmDistribution;
    readonly MQ: ArmDistribution;
  };
  readonly p: PArmResult;
  readonly q: QArmResult;
  readonly host: { readonly arch: string; readonly node: string };
}

/** `id < id` string ordering — matches `null-evaluate.ts`'s `sortedGraphListEntries`, so this module's output key order is independent of `authored.json`'s own array order (itself already sorted the same way, but this does not assume that). */
const byIdAscending = <T extends { readonly id: string }>(a: T, b: T): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

export const buildInterventionStatistics = (
  raw: Readonly<NullGraphListEvaluationRaw>,
  kinds: ReadonlyMap<string, GraphKind>,
  publishedNull: Readonly<PublishedNull>,
  bootstrapSeed: number,
  bootstrapResamples: number
): InterventionStatistics => {
  if (raw.version !== 1) {
    throw new Error(`intervention-report: authored.json has unsupported version ${String(raw.version)}, expected 1`);
  }
  if (!raw.biological) {
    throw new Error(
      'intervention-report: authored.json has no biological section (run null-evaluate with --biological)'
    );
  }
  const biologicalMovementScore = raw.biological.movementScore;

  const graphs: GraphOutcomeEntry[] = [...raw.graphs]
    .sort(byIdAscending)
    .map((graph) => {
      const kind = kinds.get(graph.id);
      if (!kind) throw new Error(`intervention-report: no kind found for graph id "${graph.id}" in the index`);
      return toGraphOutcomeEntry(
        graph.id,
        kind,
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

  const pGraph = graphs.find((g) => g.id === 'P');
  if (!pGraph) throw new Error('intervention-report: no graph with id "P" in the index');
  const qGraph = graphs.find((g) => g.id === 'Q');
  if (!qGraph) throw new Error('intervention-report: no graph with id "Q" in the index');

  const pRankAmongC = rankStatistics(cArm.scores, pGraph.mean).pHigh;
  const pRankAmongM = rankStatistics(mArm.scores, pGraph.mean).pHigh;
  const qRankAmongMQ = rankStatistics(mqArm.scores, qGraph.mean).pHigh;

  const category = evaluateCategory(pGraph.mean, pGraph.publishedNullRank.bioPercentile, cArm, mArm);
  const channelSpecific = evaluateChannelSpecific(qGraph.mean, qGraph.publishedNullRank.bioPercentile, mqArm);

  return {
    version: 1,
    decoder: raw.decoder,
    seeds: raw.seeds,
    ticks: raw.ticks,
    bootstrap: { seed: bootstrapSeed, resamples: bootstrapResamples },
    biologicalReproduction: checkBiologicalReproduction(biologicalMovementScore, publishedNull),
    graphs,
    controls: { C: cArm, M: mArm, MQ: mqArm },
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
    host: raw.host
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
const DEFAULT_OUT = resolve(repoRoot, 'training/runs/interventions/statistics.json');

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
}

export const parseInterventionReportArgs = (argv: readonly string[]): InterventionReportArgs => {
  let authored = DEFAULT_AUTHORED;
  let index = DEFAULT_INDEX;
  let publishedNull = DEFAULT_PUBLISHED_NULL;
  let out = DEFAULT_OUT;
  let bootstrapSeed = DEFAULT_BOOTSTRAP_SEED;
  let bootstrapResamples = DEFAULT_BOOTSTRAP_RESAMPLES;

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
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }

  return { authored, index, publishedNull, out, bootstrapSeed, bootstrapResamples };
};

export const runInterventionReport = (
  args: Readonly<InterventionReportArgs>
): { readonly out: string; readonly statistics: InterventionStatistics } => {
  const raw = JSON.parse(readFileSync(args.authored, 'utf8')) as NullGraphListEvaluationRaw;
  const kinds = readGraphKinds(args.index);
  const publishedNull = readPublishedNull(args.publishedNull);
  const statistics = buildInterventionStatistics(
    raw,
    kinds,
    publishedNull,
    args.bootstrapSeed,
    args.bootstrapResamples
  );
  atomicWriteFileSync(args.out, JSON.stringify(statistics));
  return { out: args.out, statistics };
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
