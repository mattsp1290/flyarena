import { describe, expect, it } from 'vitest';
import { decideStep, parseAllowlist } from '../../scripts/verify/live-smoke-lib';

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
});

describe('parseAllowlist', () => {
  it('parses the committed default: an empty array', () => {
    expect(parseAllowlist('[]')).toEqual([]);
  });

  it('parses a populated array of step-id strings', () => {
    expect(parseAllowlist('["task-generality", "readout-attribution"]')).toEqual(['task-generality', 'readout-attribution']);
  });

  it('rejects invalid JSON', () => {
    expect(() => parseAllowlist('not valid json')).toThrow('live-smoke: smoke-allow-missing.json is not valid JSON.');
  });

  it('rejects a JSON value that is not an array', () => {
    expect(() => parseAllowlist('{"task-generality": true}')).toThrow(
      'live-smoke: smoke-allow-missing.json must be a JSON array of step-id strings.'
    );
    expect(() => parseAllowlist('"task-generality"')).toThrow(
      'live-smoke: smoke-allow-missing.json must be a JSON array of step-id strings.'
    );
    expect(() => parseAllowlist('null')).toThrow('live-smoke: smoke-allow-missing.json must be a JSON array of step-id strings.');
  });

  it('rejects an array containing a non-string entry', () => {
    expect(() => parseAllowlist('["task-generality", 1]')).toThrow(
      'live-smoke: smoke-allow-missing.json must be a JSON array of step-id strings.'
    );
  });
});
