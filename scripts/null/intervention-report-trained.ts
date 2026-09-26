import { mean, type ConditionStats } from '../training/stats';
import { graphStats, quantileIndex } from './null-stats';
import type { GraphKind, GraphListIndexInfo } from './intervention-report';
import type { NullTrainedInterventionEvaluationRaw, NullTrainedInterventionGraphRaw } from './null-trained-evaluate-graph-list';

/**
 * `.agents/plans/pathway-interventions/00-overview.md`'s trained-decoder
 * categories, computed from `null-trained-evaluate-graph-list.ts`'s
 * `trained.json` (the 13 predeclared runs: P at trainer seeds 101/202/303,
 * plus C000..C004 and M1000..M1004 at trainer seed 101). Deliberately its
 * own module, not folded into `intervention-report.ts` (WP4's "keep it
 * under 1000 lines by splitting into modules if needed" — this is the same
 * "split the file that would otherwise cross the threshold" precedent
 * `graph-list-index.ts`/`null-trained-evaluate-graph-list.ts` already set).
 *
 * The overview's own text states the governing reference precisely: "the
 * freshly trained C000-C004 and M1000-M1004 ... at trainer seed 101 ...
 * With 5 graphs per arm, the trained cutoff is 'above the maximum of that
 * arm's 5'." Unlike the authored decoder, there is **no** trained-null
 * 25th-percentile floor test that this study's predeclared rules can
 * mechanically decide: "The published trained null (`rewiring-null-v1.json`
 * `trained`, 20 full rewirings) is reported for context only and does not
 * decide the category." That leaves the predeclared authored four-way split
 * (`pathway-supported` / `edge-class-effect` / `generic-rewiring-effect` /
 * `not-supported`) under-determined on the trained side whenever P does not
 * clear the C arm's max: the C/M-max test alone can rule pathway-supported
 * and edge-class-effect *out*, but cannot decide between "any perturbation
 * of this size helps" (generic) and "no effect" (not-supported) — that
 * finer split needs a real percentile floor, and this study has none that
 * the predeclared rules license using (see `TrainedOutcomeCategory`'s own
 * doc comment). `evaluateTrainedCategory` reports that undetermined case as
 * `'no-specific-effect'` rather than forcing one of the two labels.
 */

/** The study's fixed trainer seeds for P (`.agents/plans/pathway-interventions/00-overview.md`'s "P × 3 trainer seeds"). Order matters only for iteration; `trainedRobust` compares categories, not this order. */
export const P_TRAINER_SEEDS = [101, 202, 303] as const;
export type PTrainerSeed = (typeof P_TRAINER_SEEDS)[number];

/** The study's fixed control-arm trainer seed and per-arm size ("5 C graphs ... and 5 M graphs ... each at trainer seed 101"). */
export const CONTROL_TRAINER_SEED = 101;
export const TRAINED_ARM_SIZE = 5;

/**
 * The trained side's own category vocabulary — **not** a renamed subset of
 * `OutcomeCategory` (`intervention-report.ts`), because it is not one:
 * `'no-specific-effect'` is the deliberate merge of the authored
 * vocabulary's `'generic-rewiring-effect'` and `'not-supported'`, standing
 * in for "neither pathway-supported nor edge-class; the finer generic-vs-
 * not-supported split is undetermined under this study's predeclared rules"
 * (see this module's own doc comment). Reusing `OutcomeCategory` here would
 * either silently claim a resolved `'generic-rewiring-effect'`/
 * `'not-supported'` distinction this study never actually decided, or
 * require a bogus placeholder value with no home in that type — both worse
 * than a type that says plainly what was and wasn't decided.
 */
export type TrainedOutcomeCategory = 'pathway-supported' | 'edge-class-effect' | 'no-specific-effect';

export interface TrainedArmDistribution {
  readonly n: number;
  /** Sorted ascending mean `movementScore`, one per control-arm graph id. */
  readonly scores: readonly number[];
  readonly max: number;
}

/** `values` need not be sorted; the arm's own descriptive stats are computed independently of the sort `scores` above carries. */
const trainedArmDistribution = (values: readonly number[]): TrainedArmDistribution => {
  if (values.length === 0) throw new Error('intervention-report-trained: trainedArmDistribution requires at least one score');
  const scores = [...values].sort((a, b) => a - b);
  return { n: scores.length, scores, max: scores[scores.length - 1] };
};

/**
 * Context only — **never** part of `evaluateTrainedCategory`'s own
 * decision (see this module's top doc comment): the published trained
 * null's own 25th-percentile *value* (n = 20, `rewiring-null-v1.json`
 * `trained.rewired[].score`), and whether a given P trainer seed's score
 * clears it. Reported alongside each seed's real (C/M-max-governed)
 * category so a reader can see, informally, that this finer comparison
 * would not even be seed-robust — the bean's own predeclared reasoning:
 * P at trainer seed 101 falls below this value while 202/303 fall above
 * it, so treating it as decisive would flip the "generic vs not-supported"
 * read seed to seed.
 */
export interface TrainedNullContext {
  readonly publishedTrainedNullP25: number;
  readonly percentileResolution: number;
  readonly abovePublishedNullP25: boolean;
}

export interface TrainedSeedResult {
  readonly trainerSeed: PTrainerSeed;
  readonly score: number;
  readonly stats: ConditionStats;
  readonly aboveC: boolean;
  readonly aboveM: boolean;
  readonly category: TrainedOutcomeCategory;
  readonly context: TrainedNullContext;
}

export interface TrainedStatistics {
  readonly version: 1;
  readonly graphListSha256: string;
  readonly evaluatorGitRev: string | null;
  readonly d: number;
  readonly cemConfig: Record<string, unknown> | null;
  readonly cemConfigWarnings: readonly string[];
  readonly controls: { readonly C: TrainedArmDistribution; readonly M: TrainedArmDistribution };
  /** Keyed by trainer seed (101/202/303) — `PTrainerSeed` values, serialized as JSON object keys (JSON has no numeric-key vocabulary; every reader must index with the string form, e.g. `perSeed['101']`). */
  readonly perSeed: Readonly<Record<PTrainerSeed, TrainedSeedResult>>;
  /** `true` iff every entry in `perSeed` carries the same `category` — `.agents/plans/pathway-interventions/00-overview.md`'s "robust only if all three P trainer seeds ... agree on the category". */
  readonly trainedRobust: boolean;
  readonly host: { readonly arch: string; readonly node: string };
  /**
   * The plan-gap disclosure this study's coordinator wrote after seeing the
   * trained scores (`bn show flyarena-cyum`'s 2026-09-26 09:39 note): states
   * plainly that the generic-vs-not-supported split is undetermined under
   * 00-overview.md's own predeclared trained rules (which decide only the
   * C/M-max comparison), and that the 'no-specific-effect' merge label for
   * that undetermined case is a post-hoc reporting convention, not itself
   * predeclared — see `TRAINED_CATEGORY_NOTE`'s own doc comment. Carried in
   * the artifact itself (not only in the markdown report) so any consumer
   * of the raw JSON sees the same disclosure the ledger sentence and report
   * are required to state.
   */
  readonly note: string;
}

/**
 * Distinguishes what `00-overview.md` actually predeclares from what this
 * study's coordinator decided afterward (methodology review finding: an
 * earlier version of this note said the *label* itself was "predeclared",
 * which overstates it). `00-overview.md` predeclares only "the same
 * categories" for the trained decoder, plus the C/M-max cutoff and the
 * "robust only if all three seeds agree" rule — it says nothing about what
 * to report when the null-floor prong cannot be evaluated (the trained
 * null being context-only). The 'no-specific-effect' merge label for that
 * undecidable case was adopted by this study's coordinator (`bn show
 * flyarena-cyum`'s 2026-09-26 09:39 note) after the trained scores were
 * already known — a reporting convention for an outcome the predeclared
 * rules leave open, not itself a predeclared rule. What genuinely is
 * predeclared and mechanically decides P's category regardless of when
 * this note was added: the strict C/M-max comparison itself.
 */
export const TRAINED_CATEGORY_NOTE =
  "The trained decoder's predeclared C/M-max comparison can rule pathway-supported and edge-class-effect in or " +
  'out (P above/below the max of the freshly-trained 5-graph C/M arms at trainer seed 101), but 00-overview.md ' +
  'does not say how to report the finer generic-rewiring-effect vs not-supported split when P does not clear the ' +
  "C arm: that split needs a trained-null percentile floor, and this study's only trained null " +
  '(rewiring-null-v1.json\'s published n=20 sample) is reported for context only, per 00-overview.md, not as a ' +
  "decisive threshold. 'no-specific-effect' is a reporting convention this study's coordinator adopted after the " +
  'trained scores were known (methodology review, 2026-09-26), to avoid forcing an unlicensed generic/not-supported ' +
  "label -- it does not change which predeclared comparison P passed or failed, only how the undecidable case is " +
  'named.';

/**
 * `00-overview.md`'s trained cutoff, applied per P trainer seed: `aboveC`/
 * `aboveM` are strict `>` against the freshly-trained arm's own maximum
 * (mirrors `evaluateCategory`'s authored-side strict `>` against
 * `cArm.p95`/`mArm.p95` — "above the maximum" is this study's literal
 * trained-side wording, matching that same strictness convention). `!aboveC`
 * collapses to `'no-specific-effect'` regardless of `aboveM` — with P at or
 * below the C arm's own maximum, there is no daylight left in which
 * "pathway-supported" or "edge-class-effect" could still hold.
 */
export const evaluateTrainedCategory = (pScore: number, cArm: Readonly<TrainedArmDistribution>, mArm: Readonly<TrainedArmDistribution>): TrainedOutcomeCategory => {
  const { aboveC, aboveM } = trainedCategoryFlags(pScore, cArm, mArm);
  if (!aboveC) return 'no-specific-effect';
  return aboveM ? 'pathway-supported' : 'edge-class-effect';
};

/**
 * `evaluateTrainedCategory`'s two intermediate booleans, exposed separately
 * so a caller (the artifact/report builder) can state "P cleared the C arm
 * but not the M arm" in prose without recomputing them.
 * `evaluateTrainedCategory` itself calls this rather than repeating the
 * `pScore > arm.max` comparison a second time (a maintainability-review
 * suggestion: two independent copies of the same strict cutoff could drift
 * apart — e.g. one changed to `>=` — without any error surfacing the
 * disagreement between the reported flags and the reported category).
 */
export const trainedCategoryFlags = (pScore: number, cArm: Readonly<TrainedArmDistribution>, mArm: Readonly<TrainedArmDistribution>): { readonly aboveC: boolean; readonly aboveM: boolean } => ({
  aboveC: pScore > cArm.max,
  aboveM: pScore > mArm.max
});

/**
 * `trainedRobust`: every `perSeed[seed].category` must be identical —
 * checked by direct string comparison against the first seed's category
 * (`P_TRAINER_SEEDS[0]`), not by any weaker "no seed disagrees with the
 * majority" rule; a 2-1 split is not robust.
 */
export const isTrainedRobust = (perSeed: Readonly<Record<PTrainerSeed, TrainedSeedResult>>): boolean => {
  const categories = P_TRAINER_SEEDS.map((seed) => perSeed[seed].category);
  return categories.every((category) => category === categories[0]);
};

/**
 * Classify `raw.runs` into P (by trainer seed)/C/M using `info` (the same
 * `index.json`-derived kind map the authored statistics module already
 * validates against — `GraphListIndexInfo.entries`), and assert this
 * study's exact predeclared run set: P at every one of `P_TRAINER_SEEDS`
 * (no fewer, no more, no duplicate), and `TRAINED_ARM_SIZE` distinct C ids
 * and M ids, every one at `CONTROL_TRAINER_SEED`. Q/MQ/R entries are
 * rejected outright — `00-overview.md`: "Q is not evaluated with trained
 * readouts."
 */
/**
 * `label` identifies the run (its graph id) in the thrown message — mirrors
 * `intervention-report-validation.ts`'s `assertFiniteScores` for the
 * authored side. A non-finite or empty `movementScore` would otherwise
 * silently become `NaN` through `mean()`, and `NaN > arm.max` is always
 * `false`, so a corrupt run would silently score as `'no-specific-effect'`
 * with no error anywhere (a maintainability-review finding).
 */
const assertFiniteMovementScore = (label: string, movementScore: readonly number[]): void => {
  if (!Array.isArray(movementScore) || movementScore.length === 0) {
    throw new Error(`intervention-report-trained: "${label}".movementScore is missing or empty`);
  }
  const badIndex = movementScore.findIndex((value) => typeof value !== 'number' || !Number.isFinite(value));
  if (badIndex !== -1) {
    throw new Error(`intervention-report-trained: "${label}".movementScore[${badIndex}] is not a finite number`);
  }
};

const classifyRuns = (
  raw: Readonly<NullTrainedInterventionEvaluationRaw>,
  info: Readonly<GraphListIndexInfo>
): {
  readonly pBySeed: ReadonlyMap<PTrainerSeed, NullTrainedInterventionGraphRaw>;
  readonly cRuns: readonly NullTrainedInterventionGraphRaw[];
  readonly mRuns: readonly NullTrainedInterventionGraphRaw[];
} => {
  if (!Array.isArray(raw.runs)) throw new Error('intervention-report-trained: trained.json "runs" is not an array');
  const byKind = new Map<GraphKind, NullTrainedInterventionGraphRaw[]>();
  for (const run of raw.runs) {
    const entryInfo = info.entries.get(run.id);
    if (!entryInfo) throw new Error(`intervention-report-trained: no kind found for graph id "${run.id}" in the index`);
    if (entryInfo.kind === 'Q' || entryInfo.kind === 'MQ' || entryInfo.kind === 'R') {
      throw new Error(`intervention-report-trained: trained.json has a kind-"${entryInfo.kind}" run ("${run.id}") -- Q/MQ/R are not evaluated with trained readouts`);
    }
    // The index's own recorded `gzipSha256` for this id must match what this
    // run was actually scored against -- otherwise a `trained.json` scored
    // against a stale/different graph (a re-run of `interventions.py` that
    // reused the same id) would silently be reported as if it described the
    // graph `index.json` currently lists (a maintainability-review finding,
    // mirroring `intervention-report-validation.ts`'s identical check on the
    // authored side).
    if (entryInfo.gzipSha256 !== run.gzipSha256) {
      throw new Error(
        `intervention-report-trained: "${run.id}" was scored from a different graph file than index.json currently lists (trained.json gzipSha256 ${run.gzipSha256}, index.json ${entryInfo.gzipSha256}) -- stale trained.json?`
      );
    }
    assertFiniteMovementScore(run.id, run.movementScore);
    const list = byKind.get(entryInfo.kind) ?? [];
    list.push(run);
    byKind.set(entryInfo.kind, list);
  }

  const pRuns = byKind.get('P') ?? [];
  const pBySeed = new Map<PTrainerSeed, NullTrainedInterventionGraphRaw>();
  for (const run of pRuns) {
    if (!(P_TRAINER_SEEDS as readonly number[]).includes(run.trainerSeed)) {
      throw new Error(`intervention-report-trained: P run "${run.id}" has an unexpected trainer seed ${run.trainerSeed} (expected one of ${P_TRAINER_SEEDS.join(', ')})`);
    }
    if (pBySeed.has(run.trainerSeed as PTrainerSeed)) {
      throw new Error(`intervention-report-trained: more than one P run at trainer seed ${run.trainerSeed}`);
    }
    pBySeed.set(run.trainerSeed as PTrainerSeed, run);
  }
  for (const seed of P_TRAINER_SEEDS) {
    if (!pBySeed.has(seed)) throw new Error(`intervention-report-trained: trained.json is missing the P run at trainer seed ${seed}`);
  }

  const assertArm = (kind: 'C' | 'M', runs: readonly NullTrainedInterventionGraphRaw[]): readonly NullTrainedInterventionGraphRaw[] => {
    if (runs.length !== TRAINED_ARM_SIZE) {
      throw new Error(`intervention-report-trained: expected exactly ${TRAINED_ARM_SIZE} kind-"${kind}" runs, found ${runs.length}`);
    }
    const ids = new Set(runs.map((r) => r.id));
    if (ids.size !== runs.length) throw new Error(`intervention-report-trained: kind-"${kind}" runs have a duplicate id`);
    for (const run of runs) {
      if (run.trainerSeed !== CONTROL_TRAINER_SEED) {
        throw new Error(`intervention-report-trained: kind-"${kind}" run "${run.id}" has trainer seed ${run.trainerSeed}, expected the control trainer seed ${CONTROL_TRAINER_SEED}`);
      }
    }
    return runs;
  };

  return {
    pBySeed,
    cRuns: assertArm('C', byKind.get('C') ?? []),
    mRuns: assertArm('M', byKind.get('M') ?? [])
  };
};

export const buildTrainedStatistics = (
  raw: Readonly<NullTrainedInterventionEvaluationRaw>,
  info: Readonly<GraphListIndexInfo>,
  publishedTrainedNullScores: readonly number[],
  bootstrapSeed: number,
  bootstrapResamples: number
): TrainedStatistics => {
  if (raw.version !== 1) throw new Error(`intervention-report-trained: trained.json has unsupported version ${String(raw.version)}, expected 1`);
  if (publishedTrainedNullScores.length === 0 || !publishedTrainedNullScores.every((value) => Number.isFinite(value))) {
    throw new Error('intervention-report-trained: publishedTrainedNullScores must be a non-empty list of finite numbers');
  }
  const { pBySeed, cRuns, mRuns } = classifyRuns(raw, info);

  const scoreOf = (run: Readonly<NullTrainedInterventionGraphRaw>): number => mean(run.movementScore);
  const cArm = trainedArmDistribution(cRuns.map(scoreOf));
  const mArm = trainedArmDistribution(mRuns.map(scoreOf));

  const sortedPublished = [...publishedTrainedNullScores].sort((a, b) => a - b);
  // Same low-tail-floor quantile convention as `intervention-report.ts`'s
  // `publishedNullFloorValue` (`quantileIndex`, `null-stats.ts`) -- reused
  // directly rather than a hand-typed `Math.floor(0.25 * n)` that could
  // silently drift from it (a methodology-review suggestion).
  const publishedTrainedNullP25 = sortedPublished[quantileIndex(sortedPublished.length, 0.25)];
  const percentileResolutionValue = 1 / sortedPublished.length;

  const perSeedEntries = P_TRAINER_SEEDS.map((seed): [PTrainerSeed, TrainedSeedResult] => {
    const run = pBySeed.get(seed);
    if (!run) throw new Error(`intervention-report-trained: missing P run at trainer seed ${seed}`);
    const score = scoreOf(run);
    const { aboveC, aboveM } = trainedCategoryFlags(score, cArm, mArm);
    return [
      seed,
      {
        trainerSeed: seed,
        score,
        stats: graphStats(run.movementScore, bootstrapSeed, `trained-P-seed${seed}`, bootstrapResamples),
        aboveC,
        aboveM,
        category: evaluateTrainedCategory(score, cArm, mArm),
        context: {
          publishedTrainedNullP25,
          percentileResolution: percentileResolutionValue,
          abovePublishedNullP25: score > publishedTrainedNullP25
        }
      }
    ];
  });
  const perSeed = Object.fromEntries(perSeedEntries) as Record<PTrainerSeed, TrainedSeedResult>;

  return {
    version: 1,
    graphListSha256: raw.graphListSha256,
    evaluatorGitRev: raw.evaluatorGitRev,
    d: raw.d,
    cemConfig: raw.cemConfig,
    cemConfigWarnings: raw.cemConfigWarnings,
    controls: { C: cArm, M: mArm },
    perSeed,
    trainedRobust: isTrainedRobust(perSeed),
    host: raw.host,
    note: TRAINED_CATEGORY_NOTE
  };
};
