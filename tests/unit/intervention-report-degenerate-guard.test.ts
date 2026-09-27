import { describe, expect, it } from 'vitest';

import { evaluateDegenerateGuard } from '../../scripts/null/intervention-degenerate-guard';
import { DEGENERATE_IQR_THRESHOLD, nullSummary } from '../../scripts/null/null-stats';

/**
 * Coverage for `scripts/null/intervention-degenerate-guard.ts`'s `evaluateDegenerateGuard`
 * (`.agents/plans/task-generality/00-overview.md`'s predeclared degenerate
 * guard, applied only in `--stats-only` mode — a thermo-methodology review
 * finding on the task-generality WP2 branch: `no-movement`'s real C control
 * arm is 83/100 identical to biological, IQR 0, and the shipped
 * `intervention-stats.json` reported `pathway-supported` with no signal of
 * this). Kept in its own file (not `intervention-report.test.ts`, already
 * near the repo's 1000-line review-blocker threshold) since this is a
 * self-contained pure-function suite with no fixture files needed.
 */

const CLEAN_ARM = [1, 2, 3, 4, 5]; // IQR = 2, comfortably non-degenerate
const DEGENERATE_ARM = [7, 7, 7, 7, 7]; // IQR = 0

describe('evaluateDegenerateGuard', () => {
  it('flags nothing when every arm is non-degenerate', () => {
    const result = evaluateDegenerateGuard(CLEAN_ARM, CLEAN_ARM, CLEAN_ARM, CLEAN_ARM);
    expect(result.categoryDegenerate).toBe(false);
    expect(result.channelSpecificDegenerate).toBe(false);
    expect(result.nullArm.degenerate).toBe(false);
    expect(result.cArm.degenerate).toBe(false);
    expect(result.mArm.degenerate).toBe(false);
    expect(result.mqArm.degenerate).toBe(false);
  });

  it("00-overview.md's guard: a degenerate C arm alone marks the category degenerate, but not channelSpecific (MQ/null are clean)", () => {
    const result = evaluateDegenerateGuard(CLEAN_ARM, DEGENERATE_ARM, CLEAN_ARM, CLEAN_ARM);
    expect(result.cArm.degenerate).toBe(true);
    expect(result.categoryDegenerate).toBe(true);
    expect(result.channelSpecificDegenerate).toBe(false);
  });

  it('a degenerate M arm alone marks the category degenerate, but not channelSpecific', () => {
    const result = evaluateDegenerateGuard(CLEAN_ARM, CLEAN_ARM, DEGENERATE_ARM, CLEAN_ARM);
    expect(result.mArm.degenerate).toBe(true);
    expect(result.categoryDegenerate).toBe(true);
    expect(result.channelSpecificDegenerate).toBe(false);
  });

  it('a degenerate null arm marks BOTH categoryDegenerate and channelSpecificDegenerate (the shared nullFloor)', () => {
    const result = evaluateDegenerateGuard(DEGENERATE_ARM, CLEAN_ARM, CLEAN_ARM, CLEAN_ARM);
    expect(result.nullArm.degenerate).toBe(true);
    expect(result.categoryDegenerate).toBe(true);
    expect(result.channelSpecificDegenerate).toBe(true);
  });

  it("a degenerate MQ arm alone marks channelSpecificDegenerate only -- 00-overview.md's guard text names only the null/C/M, never MQ, so this is a repo-specific extension scoped to channelSpecific", () => {
    const result = evaluateDegenerateGuard(CLEAN_ARM, CLEAN_ARM, CLEAN_ARM, DEGENERATE_ARM);
    expect(result.mqArm.degenerate).toBe(true);
    expect(result.categoryDegenerate).toBe(false);
    expect(result.channelSpecificDegenerate).toBe(true);
  });

  it('reproduces the real no-movement C-arm shape: 83/100 identical to biological, IQR 0', () => {
    const cArmScores = [...Array(83).fill(-0.62), ...Array(8).fill(-0.6), ...Array(4).fill(-0.52), ...Array(5).fill(-0.58)];
    const result = evaluateDegenerateGuard(CLEAN_ARM, cArmScores, CLEAN_ARM, CLEAN_ARM);
    expect(result.cArm.iqr).toBe(0);
    expect(result.cArm.degenerate).toBe(true);
    expect(result.categoryDegenerate).toBe(true);
  });

  it('reuses null-stats.ts\'s own DEGENERATE_IQR_THRESHOLD/nullSummary (consistency, not a re-invented threshold)', () => {
    // A hand-built arm whose IQR sits just below the threshold must agree
    // exactly with a direct nullSummary(...) call -- evaluateDegenerateGuard
    // must not apply a looser or stricter cutoff of its own.
    const barelyDegenerate = [0, 0, 0, 0, DEGENERATE_IQR_THRESHOLD / 2];
    expect(nullSummary(barelyDegenerate).degenerate).toBe(true);
    const result = evaluateDegenerateGuard(CLEAN_ARM, barelyDegenerate, CLEAN_ARM, CLEAN_ARM);
    expect(result.cArm.degenerate).toBe(true);
    expect(result.cArm.iqr).toBe(nullSummary(barelyDegenerate).iqr);
  });

  it('reports the exact IQR value for each arm (not just the boolean)', () => {
    const result = evaluateDegenerateGuard(CLEAN_ARM, CLEAN_ARM, CLEAN_ARM, CLEAN_ARM);
    expect(result.nullArm.iqr).toBe(nullSummary(CLEAN_ARM).iqr);
    expect(result.cArm.iqr).toBe(nullSummary(CLEAN_ARM).iqr);
    expect(result.mArm.iqr).toBe(nullSummary(CLEAN_ARM).iqr);
    expect(result.mqArm.iqr).toBe(nullSummary(CLEAN_ARM).iqr);
  });
});
