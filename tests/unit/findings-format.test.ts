import { describe, expect, it } from 'vitest';
import { describeTrainedCategory, formatPercentile, formatRho } from '../../src/lib/findings/format';

/**
 * Thermo review (methodology Suggestion S2): a direct, standalone unit test
 * for `formatPercentile`/`formatRho` -- previously exercised only
 * indirectly through `findings-steps.test.ts`/`findings-panel.test.ts`/
 * `null-histogram.test.ts`'s assertions on rendered sentences. Also covers
 * `describeTrainedCategory` (added alongside the two moved clause maps for
 * thermo review methodology I1/I2), since it now lives in this same module.
 */

describe('formatPercentile', () => {
  it('formats the lower bound (0)', () => {
    expect(formatPercentile(0)).toBe('0.0th percentile');
  });

  it('formats the upper bound (1, i.e. 100%)', () => {
    expect(formatPercentile(1)).toBe('100.0th percentile');
  });

  it('formats an ordinary fraction with one decimal of precision', () => {
    expect(formatPercentile(0.337)).toBe('33.7th percentile');
  });

  it('rounds to one decimal place', () => {
    expect(formatPercentile(0.25649)).toBe('25.6th percentile');
  });
});

describe('formatRho', () => {
  it('formats to three decimal places', () => {
    expect(formatRho(0.4669248436993748)).toBe('0.467');
  });

  it('formats zero', () => {
    expect(formatRho(0)).toBe('0.000');
  });

  it('formats a negative value', () => {
    expect(formatRho(-0.353)).toBe('-0.353');
  });
});

describe('describeTrainedCategory', () => {
  it('passes self-descriptive categories through unchanged', () => {
    expect(describeTrainedCategory('pathway-supported')).toBe('pathway-supported');
    expect(describeTrainedCategory('edge-class-effect')).toBe('edge-class-effect');
  });

  it('glosses "no-specific-effect" in plain English, without a caveat by default', () => {
    expect(describeTrainedCategory('no-specific-effect')).toBe('no specific effect (neither pathway-supported nor edge-class)');
  });

  it('appends the reporting-convention caveat only when explicitly asked, inside the same parenthetical', () => {
    expect(describeTrainedCategory('no-specific-effect', { withCaveat: true })).toBe(
      'no specific effect (neither pathway-supported nor edge-class; a reporting convention adopted after the trained scores were known, not a predeclared category)'
    );
  });

  it('withCaveat has no effect on the self-descriptive categories', () => {
    expect(describeTrainedCategory('pathway-supported', { withCaveat: true })).toBe('pathway-supported');
    expect(describeTrainedCategory('edge-class-effect', { withCaveat: true })).toBe('edge-class-effect');
  });
});
