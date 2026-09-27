import { nullSummary } from './null-stats';

/**
 * `.agents/plans/task-generality/00-overview.md`'s predeclared degenerate
 * guard, extracted out of `intervention-report.ts` (a thermo-maintainability
 * finding: adding this guard pushed that file to 1076 lines, over the
 * repo's 1000-line review-blocker threshold — mirroring the same
 * extraction `intervention-report-validation.ts` already underwent for the
 * same reason). `intervention-report.ts` imports `evaluateDegenerateGuard`
 * from here; this module never imports anything back from it.
 */

/** `nullSummary`'s own `{iqr, degenerate}` pair, for one arm's scores -- see `evaluateDegenerateGuard`'s doc comment. */
export interface ArmDegeneracyInfo {
  readonly iqr: number;
  readonly degenerate: boolean;
}

export interface DegenerateGuardResult {
  readonly nullArm: ArmDegeneracyInfo;
  readonly cArm: ArmDegeneracyInfo;
  readonly mArm: ArmDegeneracyInfo;
  readonly mqArm: ArmDegeneracyInfo;
  /** `.agents/plans/task-generality/00-overview.md`'s guard, applied to P's category (quoted on `evaluateDegenerateGuard` below): true iff the null, C, or M arm is degenerate. Never looks at MQ -- see `channelSpecificDegenerate`. */
  readonly categoryDegenerate: boolean;
  /**
   * The plan's degenerate-guard text names only "the null, the C arm, or
   * the M arm" -- it does not mention `MQ` (Q's own size-matched class
   * control) at all, because the plan never predeclares a degenerate guard
   * for the channel-specific modifier result. Extending the same principle
   * to `channelSpecific` regardless is this file's own repo-specific
   * choice (a thermo-methodology review finding, task-generality WP2 fix
   * pass): without it, a near-constant `MQ` arm could produce a
   * `channelSpecific: true` resting on exactly the same kind of
   * near-point-mass evidence the plan explicitly guards against for P, with
   * no signal anywhere in the artifact. `nullArm` is included here too (not
   * only `mqArm`) because `evaluateChannelSpecific` also reads
   * `nullFloorScore`, the same quantity `categoryDegenerate` already
   * guards for P -- a degenerate null makes that floor equally untrustworthy
   * for Q's own comparison.
   */
  readonly channelSpecificDegenerate: boolean;
}

/**
 * `.agents/plans/task-generality/00-overview.md`'s predeclared degenerate
 * guard, quoted verbatim: *"a task's authored result is not categorized
 * (reported as `degenerate`) if the null, the C arm, or the M arm has IQR
 * below the existing `DEGENERATE_IQR_THRESHOLD` (`scripts/null/null-stats.ts`)."*
 *
 * Reuses `null-stats.ts`'s own `nullSummary` (not a separately-invented IQR
 * calculation) for all four arms -- the exact same function
 * `null-report.ts`'s `buildArtifact` already calls to set
 * `RewiringNullArtifact.null.degenerate` on the published null-summary
 * artifact, so the two are guaranteed consistent by construction (same
 * function, same `DEGENERATE_IQR_THRESHOLD`, and — for the null arm
 * specifically — the same underlying `publishedNull.scores` array a
 * `--stats-only` run's `--null` was built from).
 *
 * Deliberately scoped to `--stats-only` only (`runInterventionReport` calls
 * this only when `args.statsOnly`): `evaluateCategory`/`evaluateChannelSpecific`
 * themselves stay pure and degenerate-guard-free, so the default task's
 * `intervention:report` output (feeding the already-shipped
 * `pathway-interventions-v1.json`, which predates this guard) is untouched.
 */
export const evaluateDegenerateGuard = (
  publishedNullScores: readonly number[],
  cArmScores: readonly number[],
  mArmScores: readonly number[],
  mqArmScores: readonly number[]
): DegenerateGuardResult => {
  const toArmInfo = (scores: readonly number[]): ArmDegeneracyInfo => {
    const summary = nullSummary(scores);
    return { iqr: summary.iqr, degenerate: summary.degenerate };
  };
  const nullArm = toArmInfo(publishedNullScores);
  const cArm = toArmInfo(cArmScores);
  const mArm = toArmInfo(mArmScores);
  const mqArm = toArmInfo(mqArmScores);
  return {
    nullArm,
    cArm,
    mArm,
    mqArm,
    categoryDegenerate: nullArm.degenerate || cArm.degenerate || mArm.degenerate,
    channelSpecificDegenerate: nullArm.degenerate || mqArm.degenerate
  };
};
