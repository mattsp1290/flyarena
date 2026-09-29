import { describe, expect, it } from 'vitest';
import { hasAnyVisibleStep, isStepVisible, nearestVisibleIndex, type FlattenedFindingStep } from '../../src/lib/findings/navigation';

/**
 * WP1 of `.agents/plans/consolidated-release`, fix pass after thermo review
 * (maintainability S1): direct unit coverage for
 * `src/lib/findings/navigation.ts`'s pure search functions, independent of
 * mounting `FindingsPanel.svelte` -- `tests/unit/findings-panel.test.ts`
 * still covers the same behavior through the real component (the
 * integration point that actually matters to a user), but the search
 * algorithm itself (forward-then-backward-fallback over a flattened,
 * filtered list) is exercised here with plain arrays, no Svelte mount, no
 * `jsdom`, no `fireEvent`.
 */

const steps = (...sectionIds: string[]): FlattenedFindingStep[] => sectionIds.map((sectionId) => ({ sectionId }));

describe('isStepVisible', () => {
  it('is true when the index exists and its section is not collapsed', () => {
    expect(isStepVisible(steps('a', 'b'), 0, new Set())).toBe(true);
    expect(isStepVisible(steps('a', 'b'), 1, new Set(['a']))).toBe(true);
  });

  it('is false when the section is collapsed', () => {
    expect(isStepVisible(steps('a', 'b'), 0, new Set(['a']))).toBe(false);
  });

  it('is false for an out-of-range index, never throwing', () => {
    expect(isStepVisible(steps('a'), -1, new Set())).toBe(false);
    expect(isStepVisible(steps('a'), 5, new Set())).toBe(false);
  });
});

describe('nearestVisibleIndex', () => {
  it('searches forward first, returning the next visible index', () => {
    const flat = steps('a', 'a', 'b', 'b');
    expect(nearestVisibleIndex(flat, 0, new Set())).toBe(1);
  });

  it('falls back to searching backward when nothing visible remains forward', () => {
    // Collapsing section "b" while sitting at the last "a" step (index 1):
    // nothing forward is visible, so it falls back to index 0.
    const flat = steps('a', 'a', 'b', 'b');
    expect(nearestVisibleIndex(flat, 1, new Set(['b']))).toBe(0);
  });

  it('skips over a collapsed section entirely to reach the next open one', () => {
    const flat = steps('a', 'b', 'c');
    expect(nearestVisibleIndex(flat, 0, new Set(['b']))).toBe(2);
  });

  it('returns undefined when every section is collapsed at once', () => {
    const flat = steps('a', 'b', 'c');
    expect(nearestVisibleIndex(flat, 0, new Set(['a', 'b', 'c']))).toBeUndefined();
  });

  it('returns undefined for a single-section fixture whose only section is collapsed', () => {
    const flat = steps('a', 'a', 'a');
    expect(nearestVisibleIndex(flat, 1, new Set(['a']))).toBeUndefined();
  });

  it('returns undefined for an empty flattened list', () => {
    expect(nearestVisibleIndex([], 0, new Set())).toBeUndefined();
  });
});

describe('hasAnyVisibleStep', () => {
  it('is true when at least one step is outside the collapsed set', () => {
    expect(hasAnyVisibleStep(steps('a', 'b'), new Set(['a']))).toBe(true);
  });

  it('is false when every section is collapsed', () => {
    expect(hasAnyVisibleStep(steps('a', 'b'), new Set(['a', 'b']))).toBe(false);
  });

  it('is false for an empty flattened list regardless of the collapsed set', () => {
    expect(hasAnyVisibleStep([], new Set())).toBe(false);
    expect(hasAnyVisibleStep([], new Set(['a']))).toBe(false);
  });

  it('is true with no sections collapsed at all', () => {
    expect(hasAnyVisibleStep(steps('a'), new Set())).toBe(true);
  });
});
