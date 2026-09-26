<script lang="ts">
  import type { RewiringNullArtifact } from '../experiment/rewiringNull';
  import { githubDocUrl } from './links';
  import { formatPercentile } from '../findings/format';

  /**
   * WP4: pure SVG histogram of the authored-decoder rewiring null
   * distribution (`.agents/plans/rewiring-null/04-ledger-histogram.md`).
   * `LedgerPanel.svelte` is the only caller, and only renders this once
   * `rewiringNull.status === 'ok'` — this component assumes `data` already
   * passed `rewiringNull.ts#loadRewiringNull`'s sha256, shape, and (dual review)
   * cross-consistency verification: `bioPercentile`/`pLow`/`pHigh` are in
   * `[0, 1]`, `sum(bins.counts) === rewired.length === null.n`, and every
   * marker score falls inside `[bins.edges[0], bins.edges[last]]`. `xForValue`
   * below still clamps defensively (belt-and-suspenders, not the primary
   * guard) in case a caller ever passes unverified data directly.
   *
   * Non-negotiables this component is responsible for:
   * - The bars are the rewired graphs only (`data.bins`, pre-binned by the
   *   report generator — never rebinned here). Biological, the shipped
   *   rewired-seed-0 control, and disconnected are drawn as vertical line
   *   markers, not bars, so they are never miscounted as part of the null
   *   set.
   * - Every marker is identified by a distinct dash pattern *and* a text
   *   label with its own score (the legend below the chart) — never color
   *   alone.
   * - The `<figcaption>` states biological's percentile and the condition in
   *   plain, descriptive words (no causal/superiority claim), built from the
   *   verified artifact's own `null.n`/`seeds.count`/`condition` fields
   *   (dual review, Important — an earlier version hard-coded "500"/"100
   *   held-out seeds", which would silently disagree with the data on any
   *   re-run with a different N), and links the human-readable report.
   *   Since `docs/` is not part of the deployed static site, that link is a
   *   GitHub blob URL — the same pattern `LedgerPanel.svelte` already uses
   *   for `docs/trained-readout-report.md` (WP6/nom6).
   * - The SVG has `role="img"` with an `aria-label` that repeats the same
   *   sentence, so a screen-reader user gets the result without parsing the
   *   drawing.
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

  // Clamped to the plot's own domain (dual review, Important): the loader
  // already rejects a *marker* score outside `[domainMin, domainMax]` (bars
  // are built straight from `edges` and are always in-domain by
  // construction, so this backstop is really for markers). This is
  // defense-in-depth for a caller that ever passes unverified `data`
  // directly, not the primary guard — without it, an out-of-domain marker
  // would render invisibly outside the viewBox while its legend entry still
  // claims it is shown.
  const xForValue = (value: number): number =>
    PADDING.left + Math.min(1, Math.max(0, (value - domainMin) / domainSpan)) * PLOT_WIDTH;

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
  // change-surface row for this component). Each label carries its own
  // score too (dual review, Suggestion) — the per-bar `<title>` tooltip is
  // the only other place a value appears, and that needs a mouse hover, so
  // it is invisible to keyboard/screen-reader users; the legend is not.
  const markers = $derived<Marker[]>(
    [
      { key: 'biological', label: `Biological (${data.biological.score.toFixed(2)})`, value: data.biological.score, dash: 'none' },
      rewiredSeed0
        ? {
            key: 'rewired-seed0',
            label: `Rewired (seed 0, shipped) (${rewiredSeed0.score.toFixed(2)})`,
            value: rewiredSeed0.score,
            dash: '6 3'
          }
        : undefined,
      {
        key: 'disconnected',
        label: `Disconnected (${data.disconnected.score.toFixed(2)})`,
        value: data.disconnected.score,
        dash: '2 3'
      }
    ].filter((marker): marker is Marker => marker !== undefined)
  );

  /**
   * WP1 of `.agents/plans/findings-tour`: switched from a bare `…%` label to
   * the shared `formatPercentile` (`src/lib/findings/format.ts`) so this
   * field reads identically here and in the Findings panel's step 1
   * sentence — both render `rewiringNull.data.bioPercentile` verbatim.
   */
  const percentileLabel = $derived(formatPercentile(data.bioPercentile));

  /**
   * Thermo-nuclear review (Important, "public honesty gap"): the earlier
   * "scored above X% of N" phrasing reads as self-contradictory at the
   * boundary value the real data ships with ("scored above 0.0%" means
   * "at the very bottom", the opposite of what "above" normally signals),
   * and named neither "authored" nor its own report's disclaiming language
   * anywhere a visitor could see without leaving the app. This restates the
   * percentile direction explicitly at every value (the `(0% = lowest
   * score, 100% = highest)` parenthetical is itself data-independent, so it
   * never needs a branch) — every other number and the condition text still
   * come from the verified artifact itself (dual review, Important)
   * rather than being hard-coded, so a future re-run with a different
   * rewiring count or seed count can never leave this sentence silently
   * describing the wrong run.
   */
  const captionSentence = $derived(
    `Biological ranks at the ${percentileLabel} (0% = lowest score, 100% = highest) ` +
      `among ${data.null.n} degree-preserving rewirings (${data.condition}, ${data.seeds.count} held-out seeds).`
  );

  /**
   * Static and condition-agnostic on purpose (thermo review I1): defines
   * what "authored" means and forecloses the superiority reading a bare
   * percentile invites, regardless of what `data.condition` says on a
   * future run, so it never needs to parse or duplicate that field. Mirrors
   * `docs/rewiring-null-report.md`'s own opening disclaimer, which — unlike
   * this sentence — is not part of the deployed static site.
   */
  const CAPTION_DISCLAIMER =
    'The "authored" decoder is a fixed, hand-written mapping — not biology and not trained. This is a ' +
    'descriptive comparison within this model, not a claim that any topology is better or worse.';

  /** `docs/` is not part of the deployed static site (only `public/` is served), so a relative `docs/rewiring-null-report.md` link would 404 under any base path — see `./links.ts`. */
  const GITHUB_REPORT_URL = githubDocUrl('rewiring-null-report.md');

  // No local narrowing/rendering of WP3's `trained` section is done in this
  // WP (dual review, Suggestion — an earlier version hand-authored a guess
  // at its shape and rendered a strip from it): `RewiringNullArtifact.trained`
  // is deliberately typed `unknown` because no real producer output exists
  // yet to verify a shape against, and a hand-guessed field list here would
  // carry exactly that same risk — silently rendering nothing, or worse,
  // wrong numbers, once WP3 lands with different real field names. That
  // rendering belongs in WP3, against `null-report.ts`'s real `trained`
  // output.
</script>

<figure class="null-histogram">
  <svg
    viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
    role="img"
    aria-label={`${captionSentence} ${CAPTION_DISCLAIMER}`}
    focusable="false"
  >
    <!-- No `<desc>` (dual review, Suggestion): `aria-label` already carries
         the full result sentence (plus the disclaimer) as this image's
         accessible name, and the same text is repeated again in the visible
         `<figcaption>` below — a `<desc>` here would be a screen reader's
         third reading of the same text. `<title>` stays as a short
         structural name for tools that expose it independently of
         `aria-label` (e.g. a mouse tooltip). Children of a `role="img"`
         element are presentational to most screen readers, which is why the
         per-bin data table below lives outside this `<svg>`, not inside it. -->
    <title>Topology null distribution</title>
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

  <!-- Thermo-maintainability review S1: the per-bar `<title>` tooltip above
       needs a mouse hover and is invisible to a screen reader under
       `role="img"` — this gives keyboard/screen-reader users the same
       per-bin data (bin range, count) sighted-mouse users get, using data
       the component already computes (`bars`). Visually hidden via `.sr-only`
       (clipped, not `display:none`), never rendered on screen. -->
  <table class="sr-only">
    <caption>Per-bin counts of the {data.null.n} rewired graphs shown in the histogram above</caption>
    <thead>
      <tr>
        <th scope="col">Bin range</th>
        <th scope="col">Count</th>
      </tr>
    </thead>
    <tbody>
      {#each bars as bar (bar.index)}
        <tr>
          <td>{bar.rangeLabel}</td>
          <td>{bar.count}</td>
        </tr>
      {/each}
    </tbody>
  </table>

  {#if data.null.degenerate}
    <!-- Placed before the figcaption, so "the percentile below" (not
         "above") is the accurate direction — round-2 dual review. Says "the
         score" rather than naming a specific decoder, matching the same
         data-driven approach `captionSentence` above takes for `condition`:
         hardcoding a decoder name here would risk silently contradicting
         the figcaption on a future run under a different condition. -->
    <p class="degenerate-note">
      This null distribution is degenerate (its interquartile range is effectively zero): the score barely varies
      across rewirings, so the percentile below is not very informative. See the full report.
    </p>
  {/if}

  <figcaption>
    {captionSentence}
    {CAPTION_DISCLAIMER}
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

  /* Visually hidden, never `display:none` — keeps the per-bin data table
     reachable by keyboard/screen-reader navigation while invisible on
     screen (standard "sr-only" clip pattern). */
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
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

  .degenerate-note {
    margin: 0.6rem 0 0;
    padding: 0.4rem 0.6rem;
    border: 1px solid #5a6b3f;
    border-radius: 0.3rem;
    background: rgb(214 197 94 / 8%);
    color: #d6c55e;
    font-size: 0.75rem;
    line-height: 1.4;
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
