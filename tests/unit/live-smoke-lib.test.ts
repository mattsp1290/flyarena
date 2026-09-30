import { describe, expect, it } from 'vitest';
import { decideStep, isSettledStepStatus, parseAllowlist } from '../../scripts/verify/live-smoke-lib';

/**
 * WP2 of `.agents/plans/consolidated-release` (`02-release.md`): unit
 * coverage for `scripts/verify/live-smoke.ts`'s own pure decision logic --
 * status classification, the "unavailable" retry decision, and allowlist
 * parsing/validation -- factored into `scripts/verify/live-smoke-lib.ts` so
 * these rules are testable without a real Chromium/Playwright session (the
 * rest of `live-smoke.ts` is a thin shell that calls these functions and
 * does the actual navigating/polling, exercised for real by the manual
 * local-dist run this WP's own verification step performs, not by vitest).
 */

describe('decideStep', () => {
  it('"ok" always passes, on a first look and on a retry look', () => {
    expect(decideStep('rewiring-null', 'ok', [], false)).toEqual({ kind: 'ok' });
    expect(decideStep('rewiring-null', 'ok', [], true)).toEqual({ kind: 'ok' });
  });

  it('"missing" passes only when the step id is in the allowlist', () => {
    expect(decideStep('task-generality', 'missing', ['task-generality'], false)).toEqual({ kind: 'missing-allowed' });
    expect(decideStep('task-generality', 'missing', [], false)).toEqual({
      kind: 'fail',
      reason: 'Findings step "task-generality" is "missing" and not in the smoke allowlist.'
    });
    // A different id in the allowlist does not cover this one.
    expect(decideStep('task-generality', 'missing', ['selection-robustness'], false).kind).toBe('fail');
  });

  it('"invalid" always fails immediately -- never retried, even on a "retry" look, and even if allowlisted', () => {
    expect(decideStep('intervention', 'invalid', [], false)).toEqual({
      kind: 'fail',
      reason: 'Findings step "intervention" is "invalid" (failed verification).'
    });
    expect(decideStep('intervention', 'invalid', ['intervention'], false).kind).toBe('fail');
    expect(decideStep('intervention', 'invalid', [], true).kind).toBe('fail');
  });

  it('"unavailable" asks for exactly one retry on a first look, then fails on a retry look', () => {
    expect(decideStep('trained-null', 'unavailable', [], false)).toEqual({ kind: 'retry-unavailable' });
    expect(decideStep('trained-null', 'unavailable', [], true)).toEqual({
      kind: 'fail',
      reason: 'Findings step "trained-null" is still "unavailable" after the one allowed reload-and-recheck.'
    });
    // Allowlisting is irrelevant to "unavailable" -- it is a retryable
    // fetch failure, not the "never published" case the allowlist covers.
    expect(decideStep('trained-null', 'unavailable', ['trained-null'], true).kind).toBe('fail');
  });

  it('an unrecognized status string fails with a clear, step-id-carrying reason instead of crashing', () => {
    // (dual review, both reviewers independently) `decideStep` is called
    // with the raw string read off a live page's `data-step-status`
    // attribute (`live-smoke.ts#waitForStepSettled`), never a value this
    // codebase controls the shape of at that point -- an entirely missing
    // attribute, a future `FindingStepStatus` value this file's restated
    // `StepStatus` hasn't been taught yet, or simple DOM corruption must
    // all fail loudly and clearly here, not throw an opaque `TypeError`.
    expect(decideStep('rewiring-null', 'degraded', [], false)).toEqual({
      kind: 'fail',
      reason: 'Findings step "rewiring-null" has an unrecognized status ("degraded").'
    });
    expect(decideStep('rewiring-null', '', [], false).kind).toBe('fail');
    // Even allowlisted -- the allowlist only ever covers a real "missing" status.
    expect(decideStep('rewiring-null', 'degraded', ['rewiring-null'], false).kind).toBe('fail');
    // Unrecognized also fails on a retry look, same as "invalid" -- there is
    // no status value for which `isRetry` should ever turn a fail into a pass.
    expect(decideStep('rewiring-null', 'degraded', [], true).kind).toBe('fail');
  });
});

describe('isSettledStepStatus', () => {
  it('accepts exactly the four settled status values', () => {
    expect(isSettledStepStatus('ok')).toBe(true);
    expect(isSettledStepStatus('missing')).toBe(true);
    expect(isSettledStepStatus('unavailable')).toBe(true);
    expect(isSettledStepStatus('invalid')).toBe(true);
  });

  it('rejects "loading" (the caller polls this away before calling decideStep) and any other string', () => {
    expect(isSettledStepStatus('loading')).toBe(false);
    expect(isSettledStepStatus('')).toBe(false);
    expect(isSettledStepStatus('OK')).toBe(false);
    expect(isSettledStepStatus('degraded')).toBe(false);
  });
});

describe('parseAllowlist', () => {
  const KNOWN = ['task-generality', 'readout-attribution', 'selection-robustness'] as const;

  it('parses the committed default: an empty array', () => {
    expect(parseAllowlist('[]', KNOWN)).toEqual([]);
  });

  it('parses a populated array of step-id strings', () => {
    expect(parseAllowlist('["task-generality", "readout-attribution"]', KNOWN)).toEqual(['task-generality', 'readout-attribution']);
  });

  it('rejects invalid JSON', () => {
    expect(() => parseAllowlist('not valid json', KNOWN)).toThrow('live-smoke: smoke-allow-missing.json is not valid JSON.');
  });

  it('rejects a JSON value that is not an array', () => {
    expect(() => parseAllowlist('{"task-generality": true}', KNOWN)).toThrow(
      'live-smoke: smoke-allow-missing.json must be a JSON array of step-id strings.'
    );
    expect(() => parseAllowlist('"task-generality"', KNOWN)).toThrow(
      'live-smoke: smoke-allow-missing.json must be a JSON array of step-id strings.'
    );
    expect(() => parseAllowlist('null', KNOWN)).toThrow('live-smoke: smoke-allow-missing.json must be a JSON array of step-id strings.');
  });

  it('rejects a duplicate step id', () => {
    expect(() => parseAllowlist('["task-generality", "task-generality"]', KNOWN)).toThrow(
      'live-smoke: smoke-allow-missing.json lists a step id more than once.'
    );
  });

  it('rejects a step id that is not a real Findings step', () => {
    expect(() => parseAllowlist('["task-generality", "task-generalty"]', KNOWN)).toThrow(
      'live-smoke: smoke-allow-missing.json lists unknown step ids: ["task-generalty"].'
    );
  });

  it('rejects an array containing a non-string entry', () => {
    expect(() => parseAllowlist('["task-generality", 1]', KNOWN)).toThrow(
      'live-smoke: smoke-allow-missing.json must be a JSON array of step-id strings.'
    );
  });
});
