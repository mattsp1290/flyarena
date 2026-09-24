import { cleanup, render, screen } from '@testing-library/svelte';
import { afterEach, describe, expect, it } from 'vitest';
import NullHistogram from '../../src/lib/ui/NullHistogram.svelte';
import type { RewiringNullArtifact } from '../../src/lib/experiment/assets';

/**
 * WP4 (`.agents/plans/rewiring-null/04-ledger-histogram.md`) component test
 * for `NullHistogram.svelte`, driven by a small hand-built fixture (never
 * the real ~130 KB artifact — that integration is covered by
 * `LedgerPanel`'s own tests in `tests/unit/ui-panels.test.ts` and the
 * `tests/e2e/arena.spec.ts` "model ledger and provenance" suite).
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

  it('labels every marker in a visible legend, not just via color', () => {
    render(NullHistogram, { data: fixture });
    expect(screen.getByText('Biological')).toBeInTheDocument();
    expect(screen.getByText('Rewired (seed 0, shipped)')).toBeInTheDocument();
    expect(screen.getByText('Disconnected')).toBeInTheDocument();
  });

  it('renders a figcaption stating the percentile and condition in plain words, linking the report', () => {
    const { container } = render(NullHistogram, { data: fixture });
    const figcaption = container.querySelector('figcaption');
    expect(figcaption).toHaveTextContent(
      'Biological scored above 0.0% of 500 degree-preserving rewirings (authored decoder, opponent parked, 100 held-out seeds).'
    );
    // Descriptive only — no causal/superiority language anywhere in the caption.
    expect(figcaption?.textContent).not.toMatch(/causal|better|worse|superior/i);
    const reportLink = screen.getByRole('link', { name: /full rewiring-null report/i });
    expect(reportLink).toHaveAttribute(
      'href',
      'https://github.com/mattsp1290/flyarena/blob/main/docs/rewiring-null-report.md'
    );
  });

  it('formats a nonzero percentile correctly', () => {
    const { container } = render(NullHistogram, { data: { ...fixture, bioPercentile: 0.337 } });
    const figcaption = container.querySelector('figcaption');
    expect(figcaption).toHaveTextContent('Biological scored above 33.7% of 500 degree-preserving rewirings');
  });

  it('gives the SVG role="img" with an aria-label repeating the figcaption sentence', () => {
    const { container } = render(NullHistogram, { data: fixture });
    const svg = container.querySelector('svg[role="img"]');
    expect(svg).toBeTruthy();
    expect(svg).toHaveAttribute(
      'aria-label',
      'Biological scored above 0.0% of 500 degree-preserving rewirings (authored decoder, opponent parked, 100 held-out seeds).'
    );
  });

  it('renders nothing extra for the trained sample when data.trained is absent (degrades gracefully)', () => {
    const { container } = render(NullHistogram, { data: fixture });
    expect(container.querySelector('.trained-sample-strip')).toBeNull();
  });

  it('renders nothing extra for the trained sample when data.trained does not match the expected shape', () => {
    const { container } = render(NullHistogram, { data: { ...fixture, trained: { unexpected: true } } });
    expect(container.querySelector('.trained-sample-strip')).toBeNull();
  });

  it('renders a trained-sample strip when data.trained looks like a valid trained section', () => {
    const withTrained: RewiringNullArtifact = {
      ...fixture,
      trained: {
        rewired: [
          { seed: 0, score: 1.1 },
          { seed: 1, score: 1.4 }
        ],
        biological: [
          { trainerSeed: 101, score: 0.5 },
          { trainerSeed: 202, score: 0.6 },
          { trainerSeed: 303, score: 0.4 }
        ]
      }
    };
    const { container } = render(NullHistogram, { data: withTrained });
    const strip = container.querySelector('.trained-sample-strip');
    expect(strip).not.toBeNull();
    expect(strip).toHaveTextContent('2 rewired graphs');
    expect(strip).toHaveTextContent('3 biological trainer-seed replicas');
  });

  it('never divides by zero for a degenerate (zero-width) domain', () => {
    const degenerate: RewiringNullArtifact = {
      ...fixture,
      bins: { edges: [1, 1, 1], counts: [0, 0] }
    };
    const { container } = render(NullHistogram, { data: degenerate });
    const bars = container.querySelectorAll('rect.bar');
    for (const bar of bars) {
      expect(bar.getAttribute('x')).not.toBe('NaN');
      expect(bar.getAttribute('width')).not.toBe('NaN');
    }
  });
});
