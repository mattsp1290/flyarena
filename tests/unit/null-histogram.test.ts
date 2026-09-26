import { cleanup, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it } from 'vitest';
import NullHistogram from '../../src/lib/ui/NullHistogram.svelte';
import type { RewiringNullArtifact } from '../../src/lib/experiment/rewiringNull';

/**
 * WP4 (`.agents/plans/rewiring-null/04-ledger-histogram.md`) component test
 * for `NullHistogram.svelte`, driven by a small hand-built fixture (never
 * the real ~130 KB artifact — that integration is covered by
 * `LedgerPanel`'s own tests in `tests/unit/ui-panels.test.ts` and the
 * `tests/e2e/arena.spec.ts` "model ledger and provenance" suite). This
 * component assumes its `data` already passed `assets.ts#loadRewiringNull`'s
 * verification (including the dual-review-added cross-consistency checks:
 * `bioPercentile`/`pLow`/`pHigh` in `[0, 1]`, `sum(bins.counts) ===
 * rewired.length === null.n`, and every marker inside the bin domain) — the
 * fixture below is built to already satisfy those invariants, matching what
 * the component actually receives in production.
 */

afterEach(() => cleanup());

const fixture: RewiringNullArtifact = {
  version: 1,
  condition: 'authored, opponent parked',
  seeds: { start: 30001, count: 100 },
  ticks: 1800,
  substeps: 4,
  sourceGraphSha256: 'a'.repeat(64),
  rewireSourceSha256: 'b'.repeat(64),
  shards: 18,
  biological: { score: -0.22, median: -1.41, std: 3.28, ci: [-0.85, 0.45] },
  disconnected: { score: -1.86, median: -2, std: 1.79, ci: [-2.22, -1.52] },
  rewired: [
    { seed: 0, gzipSha256: 'c'.repeat(64), score: 1.02, median: 0.62, std: 3.73, ci: [0.3, 1.77], acceptedSwaps: 764340, attempts: 926220 },
    { seed: 1, gzipSha256: 'd'.repeat(64), score: 2.5, median: 2.5, std: 1.0, ci: [2.0, 3.0], acceptedSwaps: 764340, attempts: 926220 },
    { seed: 2, gzipSha256: 'e'.repeat(64), score: 3.1, median: 3.1, std: 1.0, ci: [2.5, 3.7], acceptedSwaps: 764340, attempts: 926220 }
  ],
  null: { n: 3, mean: 2.2, median: 2.5, std: 0.87, p2_5: 1.1, p97_5: 3.05, iqr: 0.8, degenerate: false },
  bioPercentile: 0,
  pLow: 0.25,
  pHigh: 1,
  bins: { edges: [-2, 0, 2, 4], counts: [0, 1, 2] }
};

describe('NullHistogram', () => {
  it('draws exactly one bar per bin', () => {
    const { container } = render(NullHistogram, { data: fixture });
    const bars = container.querySelectorAll('rect.bar');
    expect(bars.length).toBe(fixture.bins.counts.length);
  });

  it('draws all three markers (biological, rewired seed 0, disconnected) with distinct, non-"none"-duplicated dash patterns', () => {
    const { container } = render(NullHistogram, { data: fixture });
    const lines = Array.from(container.querySelectorAll('svg:not(.swatch) > line.marker'));
    expect(lines.length).toBe(3);
    const dashPatterns = lines.map((line) => line.getAttribute('stroke-dasharray'));
    // Every marker must be identifiable by its dash pattern alone (never
    // color alone) — the three patterns must therefore all differ.
    expect(new Set(dashPatterns).size).toBe(3);
  });

  it('draws only two markers when there is no rewired-seed-0 entry', () => {
    const withoutSeed0: RewiringNullArtifact = {
      ...fixture,
      rewired: [fixture.rewired[1], fixture.rewired[2]],
      null: { ...fixture.null, n: 2 },
      bins: { edges: [-2, 0, 2, 4], counts: [0, 0, 2] }
    };
    const { container } = render(NullHistogram, { data: withoutSeed0 });
    const lines = Array.from(container.querySelectorAll('svg:not(.swatch) > line.marker'));
    expect(lines.length).toBe(2);
    expect(screen.queryByText(/rewired \(seed 0, shipped\)/i)).not.toBeInTheDocument();
  });

  it('labels every marker in a visible legend with its own score, not just via color', () => {
    render(NullHistogram, { data: fixture });
    expect(screen.getByText('Biological (-0.22)')).toBeInTheDocument();
    expect(screen.getByText('Rewired (seed 0, shipped) (1.02)')).toBeInTheDocument();
    expect(screen.getByText('Disconnected (-1.86)')).toBeInTheDocument();
  });

  it('renders a figcaption stating the percentile and condition in plain words, built from the artifact\'s own n/seeds.count/condition, and links the report', () => {
    const { container } = render(NullHistogram, { data: fixture });
    const figcaption = container.querySelector('figcaption');
    // Regression coverage for the dual review finding: an earlier version
    // hardcoded "500" and "100 held-out seeds" regardless of the fixture's
    // actual `null.n`/`seeds.count` (3 and 100 here) — this fixture
    // deliberately uses n=3, not 500, so a hardcoded "500" would fail this.
    expect(figcaption).toHaveTextContent(
      'Biological ranks at the 0.0th percentile (0% = lowest score, 100% = highest) among 3 degree-preserving ' +
        'rewirings (authored, opponent parked, 100 held-out seeds).'
    );
    // Thermo review I1: the static disclaimer must be visible in-product
    // (not only in the linked report), and must name "hand-written"/"not
    // biology"/"not trained" explicitly.
    expect(figcaption).toHaveTextContent(
      'The "authored" decoder is a fixed, hand-written mapping — not biology and not trained. This is a ' +
        'descriptive comparison within this model, not a claim that any topology is better or worse.'
    );
    // Still descriptive only — no causal/superiority claim.
    expect(figcaption?.textContent).not.toMatch(/\bcausal\b/i);
    const reportLink = screen.getByRole('link', { name: /full rewiring-null report/i });
    expect(reportLink).toHaveAttribute(
      'href',
      'https://github.com/mattsp1290/flyarena/blob/main/docs/rewiring-null-report.md'
    );
  });

  // Round-2 dual review (guardian S6): the test above uses `seeds.count:
  // 100`, the exact value an earlier version hardcoded — a regression back
  // to "100 held-out seeds" would still pass it. This asserts a different
  // count actually appears, closing that gap.
  it('renders a seeds.count other than 100 in the caption (regression coverage for the earlier hardcoded value)', () => {
    const { container } = render(NullHistogram, { data: { ...fixture, seeds: { ...fixture.seeds, count: 7 } } });
    const figcaption = container.querySelector('figcaption');
    expect(figcaption).toHaveTextContent('7 held-out seeds');
    expect(figcaption).not.toHaveTextContent('100 held-out seeds');
  });

  it('formats a nonzero percentile correctly', () => {
    const { container } = render(NullHistogram, { data: { ...fixture, bioPercentile: 0.337 } });
    const figcaption = container.querySelector('figcaption');
    expect(figcaption).toHaveTextContent('Biological ranks at the 33.7th percentile');
    expect(figcaption).toHaveTextContent('among 3 degree-preserving rewirings');
  });

  it('gives the SVG role="img" with an aria-label repeating the figcaption sentence plus the disclaimer, and no redundant <desc>', () => {
    const { container } = render(NullHistogram, { data: fixture });
    const svg = container.querySelector('svg[role="img"]');
    expect(svg).toBeTruthy();
    expect(svg).toHaveAttribute(
      'aria-label',
      'Biological ranks at the 0.0th percentile (0% = lowest score, 100% = highest) among 3 degree-preserving ' +
        'rewirings (authored, opponent parked, 100 held-out seeds). The "authored" decoder is a fixed, ' +
        'hand-written mapping — not biology and not trained. This is a descriptive comparison within this ' +
        'model, not a claim that any topology is better or worse.'
    );
    // Dual review, Suggestion: `<desc>` duplicated the same sentence a third
    // time (after `aria-label` and the visible `<figcaption>`) with no extra
    // information — removed. `<title>` stays as a short structural name.
    expect(svg?.querySelector('desc')).toBeNull();
    expect(svg?.querySelector('title')).toHaveTextContent('Topology null distribution');
  });

  it('renders no trained-sample content: WP4 never renders data.trained (deferred to WP3, per the artifact type\'s own doc comment)', () => {
    const withTrained = {
      ...fixture,
      trained: { rewired: [{ seed: 0, score: 1.1 }], biological: [{ trainerSeed: 101, score: 0.5 }] }
    } as unknown as RewiringNullArtifact;
    const { container } = render(NullHistogram, { data: withTrained });
    expect(container.querySelector('.trained-sample-strip')).toBeNull();
    expect(screen.queryByText(/trained sample/i)).not.toBeInTheDocument();
  });

  it('shows a degenerate-distribution note when data.null.degenerate is true, and not otherwise', () => {
    const { container: normal } = render(NullHistogram, { data: fixture });
    expect(normal.querySelector('.degenerate-note')).toBeNull();

    cleanup();
    const { container: degenerate } = render(NullHistogram, {
      data: { ...fixture, null: { ...fixture.null, degenerate: true } }
    });
    expect(degenerate.querySelector('.degenerate-note')).not.toBeNull();
    expect(screen.getByText(/degenerate/i)).toBeInTheDocument();
  });

  it('never divides by zero for a degenerate (zero-width) domain, and clamps every bar and marker coordinate inside the viewBox', () => {
    const degenerate: RewiringNullArtifact = {
      ...fixture,
      biological: { ...fixture.biological, score: 1 },
      disconnected: { ...fixture.disconnected, score: 1 },
      rewired: fixture.rewired.map((entry) => ({ ...entry, score: 1 })),
      bins: { edges: [1, 1, 1], counts: [0, 0] }
    };
    const { container } = render(NullHistogram, { data: degenerate });
    const WIDTH = 320;

    const bars = container.querySelectorAll('rect.bar');
    expect(bars.length).toBe(2);
    for (const bar of bars) {
      const x = Number(bar.getAttribute('x'));
      const width = Number(bar.getAttribute('width'));
      expect(Number.isNaN(x)).toBe(false);
      expect(Number.isNaN(width)).toBe(false);
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(WIDTH);
    }

    const lines = Array.from(container.querySelectorAll('svg:not(.swatch) > line.marker'));
    expect(lines.length).toBe(3);
    for (const line of lines) {
      const x1 = Number(line.getAttribute('x1'));
      expect(Number.isNaN(x1)).toBe(false);
      expect(x1).toBeGreaterThanOrEqual(0);
      expect(x1).toBeLessThanOrEqual(WIDTH);
    }
  });

  /**
   * Round-2 dual review (guardian S2, confirmed by mutation testing): the
   * test above sets every marker's score exactly equal to `domainMin` (a
   * zero-width domain), so `(value - domainMin) / domainSpan` is always `0`
   * and the clamp never actually has anything to clamp — deleting the
   * `Math.min(1, Math.max(0, …))` clamp still passed every test. This test
   * uses a genuinely non-zero domain with a marker score outside it (the
   * loader rejects this in production; this exercises the component's own
   * defensive backstop directly, per its doc comment), and asserts the
   * clamped coordinate lands exactly on the plot's edge, not beyond it.
   */
  it('clamps an out-of-domain marker score to the plot edge instead of drawing it outside the viewBox', () => {
    const PADDING_LEFT = 10;
    const WIDTH = 320;
    const PADDING_RIGHT = 10;
    const outOfDomain: RewiringNullArtifact = {
      ...fixture,
      biological: { ...fixture.biological, score: -99 }, // below bins.edges[0] = -2
      disconnected: { ...fixture.disconnected, score: 99 } // above bins.edges[last] = 4
    };
    const { container } = render(NullHistogram, { data: outOfDomain });
    const lines = Array.from(container.querySelectorAll('svg:not(.swatch) > line.marker'));
    const biologicalLine = lines[0]; // `markers` orders biological first.
    const disconnectedLine = lines[lines.length - 1]; // disconnected last.
    expect(Number(biologicalLine.getAttribute('x1'))).toBe(PADDING_LEFT);
    expect(Number(disconnectedLine.getAttribute('x1'))).toBe(WIDTH - PADDING_RIGHT);
  });

  /**
   * Thermo-maintainability review S1: the per-bar `<title>` tooltip needs a
   * mouse hover and is invisible to a screen reader under `role="img"` —
   * this asserts the visually-hidden table gives keyboard/screen-reader
   * users the same per-bin data (bin range, count) without a mouse.
   */
  it('exposes a visually-hidden table of per-bin ranges and counts (keyboard/screen-reader path to the histogram data)', () => {
    const { container } = render(NullHistogram, { data: fixture });
    const table = container.querySelector('table.sr-only');
    expect(table).toBeTruthy();
    const rows = table?.querySelectorAll('tbody tr') ?? [];
    expect(rows.length).toBe(fixture.bins.counts.length);
    fixture.bins.counts.forEach((count, index) => {
      const cells = rows[index].querySelectorAll('td');
      expect(cells[0]).toHaveTextContent(
        `${fixture.bins.edges[index].toFixed(2)} to ${fixture.bins.edges[index + 1].toFixed(2)}`
      );
      expect(cells[1]).toHaveTextContent(String(count));
    });
  });
});
