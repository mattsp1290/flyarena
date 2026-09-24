<script lang="ts">
  import type { RewiringNullArtifact } from '../experiment/assets';

  /**
   * WP4: pure SVG histogram of the authored-decoder rewiring null
   * distribution (`.agents/plans/rewiring-null/04-ledger-histogram.md`).
   * `LedgerPanel.svelte` is the only caller, and only renders this once
   * `rewiringNull.status === 'ok'` — this component assumes `data` already
   * passed `assets.ts#loadRewiringNull`'s sha256 and shape verification.
   *
   * Non-negotiables this component is responsible for:
   * - The bars are the 500 rewired graphs only (`data.bins`, pre-binned by
   *   the report generator — never rebinned here). Biological, the shipped
   *   rewired-seed-0 control, and disconnected are drawn as vertical line
   *   markers, not bars, so they are never miscounted as part of the null
   *   set.
   * - Every marker is identified by a distinct dash pattern *and* a text
   *   label (the legend below the chart) — never color alone.
   * - The `<figcaption>` states biological's percentile and the condition in
   *   plain, descriptive words (no causal/superiority claim) and links the
   *   human-readable report. Since `docs/` is not part of the deployed
   *   static site, that link is a GitHub blob URL — the same pattern
   *   `LedgerPanel.svelte` already uses for
   *   `docs/trained-readout-report.md` (WP6/nom6).
   * - The SVG has `role="img"` with an `aria-label` that repeats the same
   *   sentence, plus a `<title>`/`<desc>`, so a screen-reader user gets the
   *   result without parsing the drawing.
   */
  interface Props {
    data: RewiringNullArtifact;
  }

  let { data }: Props = $props();

  const WIDTH = 320;
  const HEIGHT = 150;
  const PADDING = { top: 10, right: 10, bottom: 10, left: 10 };
  const PLOT_WIDTH = WIDTH - PADDING.left - PADDING.right;
  const PLOT_HEIGHT = HEIGHT - PADDING.top - PADDING.bottom;

  const domainMin = $derived(data.bins.edges[0]);
  const domainMax = $derived(data.bins.edges[data.bins.edges.length - 1]);
  // Guards against a (structurally valid but) zero-width domain — e.g. a
  // degenerate null where every score is identical — dividing by zero would
  // otherwise turn every x coordinate into NaN and silently blank the SVG.
  const domainSpan = $derived(Math.max(domainMax - domainMin, 1e-9));

  const xForValue = (value: number): number => PADDING.left + ((value - domainMin) / domainSpan) * PLOT_WIDTH;

  const maxCount = $derived(Math.max(1, ...data.bins.counts));

  interface Bar {
    /** The bin's own index — the `{#each}` key below, since `rangeLabel` alone can collide for a degenerate (zero-width) domain where every edge is identical. */
    index: number;
    x: number;
    width: number;
    y: number;
    height: number;
    count: number;
    rangeLabel: string;
  }

  const bars = $derived<Bar[]>(
    data.bins.counts.map((count, index) => {
      const x0 = xForValue(data.bins.edges[index]);
      const x1 = xForValue(data.bins.edges[index + 1]);
      const barHeight = (count / maxCount) * PLOT_HEIGHT;
      return {
        index,
        x: x0,
        // A hairline gap between adjacent bars (never negative — a
        // zero-width bin, from a repeated edge, would otherwise flip the
        // rect's width negative, which SVG treats as an error).
        width: Math.max(0, x1 - x0 - 0.5),
        y: PADDING.top + (PLOT_HEIGHT - barHeight),
        height: barHeight,
        count,
        rangeLabel: `${data.bins.edges[index].toFixed(2)} to ${data.bins.edges[index + 1].toFixed(2)}`
      };
    })
  );

  const rewiredSeed0 = $derived(data.rewired.find((entry) => entry.seed === 0));

  interface Marker {
    key: string;
    label: string;
    value: number;
    /** SVG `stroke-dasharray`; `'none'` (solid) is itself a distinct pattern from the other two. */
    dash: string;
  }

  // Non-negotiable: distinct dash patterns, never color alone, for
  // biological/rewired-seed-0/disconnected (`04-ledger-histogram.md`'s
  // change-surface row for this component).
  const markers = $derived<Marker[]>(
    [
      { key: 'biological', label: 'Biological', value: data.biological.score, dash: 'none' },
      rewiredSeed0
        ? { key: 'rewired-seed0', label: 'Rewired (seed 0, shipped)', value: rewiredSeed0.score, dash: '6 3' }
        : undefined,
      { key: 'disconnected', label: 'Disconnected', value: data.disconnected.score, dash: '2 3' }
    ].filter((marker): marker is Marker => marker !== undefined)
  );

  const percentileLabel = $derived(`${(data.bioPercentile * 100).toFixed(1)}%`);

  /** The plan's literal sentence (`04-ledger-histogram.md`), descriptive only — no causal or superiority claim. */
  const captionSentence = $derived(
    `Biological scored above ${percentileLabel} of 500 degree-preserving rewirings ` +
      `(authored decoder, opponent parked, 100 held-out seeds).`
  );

  /**
   * A plain GitHub blob link, mirroring `LedgerPanel.svelte`'s own
   * `GITHUB_REPORT_URL` for `docs/trained-readout-report.md`: `docs/` is not
   * part of the deployed static site (only `public/` is served), so a
   * relative `docs/rewiring-null-report.md` link would 404 under any base
   * path.
   */
  const GITHUB_REPORT_URL = 'https://github.com/mattsp1290/flyarena/blob/main/docs/rewiring-null-report.md';

  /**
   * Best-effort, forward-compatible narrowing of WP3's `trained` section —
   * see `RewiringNullArtifact.trained`'s doc comment for why this is not a
   * hand-authored strict type. Renders nothing extra when `data.trained` is
   * absent or does not have this shape (degrade gracefully, per the bean's
   * own instruction), rather than failing or guessing at unconfirmed field
   * names.
   */
  interface TrainedSampleShape {
    rewired: readonly { seed: number; score: number }[];
    biological: readonly { trainerSeed: number; score: number }[];
  }

  const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

  const isTrainedEntryArray = (value: unknown, key: 'seed' | 'trainerSeed'): boolean =>
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (entry) =>
        typeof entry === 'object' &&
        entry !== null &&
        isFiniteNumber((entry as Record<string, unknown>)[key]) &&
        isFiniteNumber((entry as Record<string, unknown>).score)
    );

  const trainedSample = $derived<TrainedSampleShape | undefined>(
    (() => {
      const trained = data.trained;
      if (typeof trained !== 'object' || trained === null) return undefined;
      const v = trained as Record<string, unknown>;
      if (!isTrainedEntryArray(v.rewired, 'seed') || !isTrainedEntryArray(v.biological, 'trainerSeed')) return undefined;
      return { rewired: v.rewired, biological: v.biological } as TrainedSampleShape;
    })()
  );
</script>

<figure class="null-histogram">
  <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={captionSentence} focusable="false">
    <title>Topology null distribution</title>
    <desc>{captionSentence}</desc>
    {#each bars as bar (bar.index)}
      <rect x={bar.x} y={bar.y} width={bar.width} height={bar.height} class="bar">
        <title>{bar.rangeLabel}: {bar.count}</title>
      </rect>
    {/each}
    {#each markers as marker (marker.key)}
      <line
        x1={xForValue(marker.value)}
        x2={xForValue(marker.value)}
        y1={PADDING.top}
        y2={PADDING.top + PLOT_HEIGHT}
        class="marker"
        stroke-dasharray={marker.dash}
      />
    {/each}
  </svg>

  <ul class="marker-legend">
    {#each markers as marker (marker.key)}
      <li>
        <svg class="swatch" viewBox="0 0 24 8" aria-hidden="true" focusable="false">
          <line x1="0" y1="4" x2="24" y2="4" class="marker" stroke-dasharray={marker.dash} />
        </svg>
        <span>{marker.label}</span>
      </li>
    {/each}
  </ul>

  {#if trainedSample}
    <div class="trained-sample-strip">
      <p>
        Trained sample: {trainedSample.rewired.length} rewired graphs retrained with a per-graph readout, versus
        {trainedSample.biological.length} biological trainer-seed replicas — trainer-seed noise at fixed topology,
        not comparable to the topology-only null above (see the full report).
      </p>
    </div>
  {/if}

  <figcaption>
    {captionSentence}
    <a href={GITHUB_REPORT_URL} target="_blank" rel="noreferrer">Full rewiring-null report</a>
  </figcaption>
</figure>

<style>
  .null-histogram {
    margin: 0.9rem 0;
    padding: 0.65rem 0.75rem;
    border: 1px solid #304355;
    border-radius: 0.4rem;
    background: rgb(121 216 208 / 6%);
  }

  .null-histogram svg:not(.swatch) {
    display: block;
    width: 100%;
    height: auto;
  }

  .bar {
    fill: #4f6b85;
  }

  .marker {
    stroke: #edf4ff;
    stroke-width: 1.5;
    fill: none;
  }

  .marker-legend {
    margin: 0.5rem 0 0;
    padding: 0;
    list-style: none;
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem 1rem;
  }

  .marker-legend li {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.75rem;
    color: #9aacc2;
    border: none;
    padding: 0;
  }

  .swatch {
    width: 24px;
    height: 8px;
    flex: none;
  }

  .trained-sample-strip {
    margin-top: 0.6rem;
    font-size: 0.75rem;
    color: #9aacc2;
  }

  .trained-sample-strip p {
    margin: 0;
  }

  figcaption {
    margin-top: 0.6rem;
    font-size: 0.78rem;
    color: #cbd8e7;
    line-height: 1.5;
  }

  figcaption a {
    color: #79d8d0;
  }
</style>
