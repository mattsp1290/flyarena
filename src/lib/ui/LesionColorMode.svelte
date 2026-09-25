<script lang="ts">
  import type { LesionAtlasGraphKey, LesionAtlasLoadResult } from '../experiment/lesionAtlas';
  import { githubDocUrl } from './links';
  import type { ActivityColorMode } from './activityLesionColorMode.svelte';

  /**
   * The lesion-effect color mode's presentational half (WP3): the radio
   * group that switches between Live/Lesion, its honesty-disclosure label
   * paragraph, the per-arm "no lesion data" coverage messages, the sr-only
   * FDR-significance summary, and the diverging legend — all pulled out of
   * `ActivityPanel.svelte` alongside `activityLesionColorMode.svelte.ts`
   * (that module's own doc comment explains the split). This component owns
   * no state beyond two DOM element refs (`liveRadioEl`/`lesionRadioEl` —
   * see `handleColorModeInputChange`) and is otherwise pure props in, one
   * callback out (`onSwitchColorMode`), the same shape `NullHistogram.svelte`
   * already establishes for a static, offline-computed artifact rendered
   * inside this panel family.
   *
   * Mounted twice by `ActivityPanel.svelte`, distinguished by `part`: once
   * (`'controls'`) where the mode selector used to live (radio group + label
   * + coverage + sr-only summary), and once (`'legend'`) where the diverging
   * legend used to live, just after the canvas — `ActivityPanel.svelte`'s own
   * template still controls *where* each part renders (unchanged from before
   * this extraction); this component only owns *what* renders at each spot.
   */

  interface Props {
    part: 'controls' | 'legend';
    /** `'controls'` only: mirrors `ActivityPanel.svelte`'s own `sceneReady`, gating the fieldset the same way it always did. */
    sceneReady?: boolean;
    colorMode: ActivityColorMode;
    lesionOptionDisabledReason?: string;
    lesionAtlasTransientReason?: string;
    /** `'legend'` only: rendered only once `status === 'ok'` — mirrors the original inline `{:else if colorMode === 'lesion' && lesionAtlasStatus?.status === 'ok'}` guard. */
    lesionAtlasStatus?: LesionAtlasLoadResult;
    lesionSourceLeft?: LesionAtlasGraphKey | 'none';
    lesionSourceRight?: LesionAtlasGraphKey | 'none';
    lesionSignificantSummary?: string;
    onSwitchColorMode?: (next: ActivityColorMode) => void;
  }

  let {
    part,
    sceneReady = false,
    colorMode,
    lesionOptionDisabledReason,
    lesionAtlasTransientReason,
    lesionAtlasStatus,
    lesionSourceLeft = 'none',
    lesionSourceRight = 'none',
    lesionSignificantSummary,
    onSwitchColorMode
  }: Props = $props();

  /** `bind:this` targets must be `$state` in Svelte 5 (`svelte/non_reactive_update`) — see `handleColorModeInputChange`. */
  let liveRadioEl = $state<HTMLInputElement | undefined>(undefined);
  let lesionRadioEl = $state<HTMLInputElement | undefined>(undefined);

  /** Plain GitHub blob link to the human-readable lesion-atlas report (`docs/` is not part of the deployed static site) — shared `githubDocUrl` helper, base-path-safe by construction. */
  const LESION_REPORT_URL = githubDocUrl('lesion-atlas-report.md');

  /**
   * A native radio input flips its own `checked` DOM property (and unchecks
   * its named-group sibling) synchronously on click/keyboard selection,
   * *before* Svelte's own reactive `checked={colorMode === value}` binding
   * ever runs. That reactive binding only re-renders when `colorMode` itself
   * actually changes — so if `onSwitchColorMode` ends up staying on the
   * current mode (e.g. the lazily-loaded atlas turns out missing/invalid),
   * `colorMode` never changes value, and the browser is left showing the
   * *disabled* lesion radio as visually checked while the app's own state
   * still says Live. Both radios' native `checked` are force-synced back to
   * the *current* (pre-switch) `colorMode` right here, synchronously, before
   * the switch runs; if it does end up changing `colorMode` for real, the
   * reactive `checked={...}` bindings correct both radios again on the next
   * render, same as always.
   */
  const handleColorModeInputChange = (next: ActivityColorMode): void => {
    if (liveRadioEl) liveRadioEl.checked = colorMode === 'live';
    if (lesionRadioEl) lesionRadioEl.checked = colorMode === 'lesion';
    onSwitchColorMode?.(next);
  };

  /**
   * Single derived hint (thermo-maintainability S3): the fieldset's one
   * `aria-live="polite"` hint paragraph shows the disabled reason, then the
   * transient-retry reason, only while on Live — collapses what used to be
   * two separately-conditioned `{:else if}` branches sharing the same
   * `colorMode === 'live'` guard into one derived string.
   */
  const hintText = $derived.by(() => {
    if (colorMode !== 'live') return '';
    if (lesionOptionDisabledReason) return `Lesion effect (offline) unavailable: ${lesionOptionDisabledReason}`;
    if (lesionAtlasTransientReason)
      return `Lesion effect (offline) could not be loaded (retrying is available): ${lesionAtlasTransientReason}`;
    return '';
  });

  const legendStatus = $derived(lesionAtlasStatus?.status === 'ok' ? lesionAtlasStatus : undefined);
</script>

{#if part === 'controls'}
  <fieldset class="field fieldset-group" disabled={!sceneReady}>
    <legend>Color</legend>
    <label class="radio-option">
      <input
        bind:this={liveRadioEl}
        type="radio"
        name="activity-color-mode"
        value="live"
        checked={colorMode === 'live'}
        onchange={() => handleColorModeInputChange('live')}
      />
      Live rate
    </label>
    <label class="radio-option">
      <input
        bind:this={lesionRadioEl}
        type="radio"
        name="activity-color-mode"
        value="lesion"
        checked={colorMode === 'lesion'}
        disabled={lesionOptionDisabledReason !== undefined}
        onchange={() => handleColorModeInputChange('lesion')}
      />
      Lesion effect (offline)
    </label>
    <!--
      Thermo-maintainability review I3 (a11y): always mounted (never entered/
      left via its own `{#if}`), with only its text content changing — a
      screen reader observes a *mutation* to an already-present live region,
      rather than a freshly-inserted node that already carries its final text
      on first paint (which several screen readers, notably VoiceOver/Safari,
      do not reliably announce).
    -->
    <p class="hint" aria-live="polite">{hintText}</p>
  </fieldset>

  {#if colorMode === 'lesion'}
    <p class="labels lesion-label">
      Lesion effect: <strong>Computed (offline)</strong> — effect on this model's score when this neuron's rate is clamped to 0 for the whole episode (negative = this model's score drops when the neuron is silenced, positive = it rises; authored decoder, opponent parked, 100 held-out seeds).
      FDR q&nbsp;=&nbsp;0.05; faded/outlined neurons did not survive false-discovery correction and are not reliable effects.
      Sensory neurons are this model's hand-wired encoder inputs.
      This is not a claim about the real fly's neural function — see the
      <a href={LESION_REPORT_URL} target="_blank" rel="noreferrer">full report</a>.
    </p>
    {#if lesionSourceLeft === 'none'}
      <p class="coverage streaming-unavailable" role="status">Left arm: no lesion data (disconnected).</p>
    {/if}
    {#if lesionSourceRight === 'none'}
      <p class="coverage streaming-unavailable" role="status">Right arm: no lesion data (disconnected).</p>
    {/if}
  {/if}

  <!--
    Thermo-maintainability review I3 (a11y): always mounted, same reasoning
    as the hint paragraph above — the plan's own accessibility non-negotiable
    (sr-only summary of FDR-significant neuron counts) must be announced on
    entering lesion mode, not merely present once already inserted.
  -->
  <p class="sr-only" aria-live="polite">{lesionSignificantSummary ?? ''}</p>
{:else if part === 'legend' && legendStatus}
  {@const status = legendStatus}
  {@const absMax = status.absMax}
  {@const bioMax = Math.max(...status.data.graphs.biological.effect.map(Math.abs))}
  {@const rewiredMax = Math.max(...status.data.graphs.rewiredSeed0.effect.map(Math.abs))}
  <div class="legend lesion-legend">
    <div class="legend-scale" aria-hidden="true">
      <span>{(-absMax).toFixed(3)}</span>
      <span class="legend-bar"></span>
      <span>+{absMax.toFixed(3)}</span>
    </div>
    <span class="legend-caption">score change when silenced (0 = no effect)</span>
    <!--
      Thermo-nuclear architecture review I2 (public-facing honesty): states
      that the scale is shared across both graphs, with live per-graph max
      values — a uniformly pale arm (small effects *relative to the other
      graph's scale*) must not read as "no effect here."
    -->
    <span class="legend-caption"
      >One scale, shared across both graphs (max |effect| — biological: {bioMax.toFixed(3)}, rewired seed 0: {rewiredMax.toFixed(3)}) —
      a mostly pale arm means its effects are small on this shared scale, not necessarily zero.</span
    >
    <span class="legend-outline" aria-hidden="true"
      ><span class="swatch outline"></span>Outlined/faded = not FDR-significant (q=0.05)</span
    >
    <span class="legend-no-data" aria-hidden="true"><span class="swatch"></span>No lesion data (disconnected)</span>
  </div>
{/if}

<style>
  /* `.field`/`.fieldset-group`/`.radio-option`/`.hint` live in `src/app.css`,
     shared with `ExperimentPanel.svelte`'s decoder fieldset
     (thermo-maintainability review S1). `.labels`/`.coverage`/
     `.streaming-unavailable`/`.legend`/`.legend-scale`/`.legend-no-data`/
     `.sr-only` below are intentionally duplicated from `ActivityPanel.svelte`
     (Svelte's per-component style scoping means its `<style>` block is not
     reachable from here) — the same "duplicated, not shared" tradeoff
     `NullHistogram.svelte`'s own `.sr-only` rule documents; the two legends'
     bar styling differs enough (diverging colors baked in directly here vs.
     viridis there) that a shared class would need its own modifier anyway. */

  .labels {
    margin: 0.6rem 0 0;
    color: #cbd8e7;
    font-size: 0.78rem;
    line-height: 1.5;
  }

  .coverage {
    margin: 0.3rem 0 0;
    color: #9aacc2;
    font-size: 0.78rem;
  }

  .streaming-unavailable {
    color: #ffcf8a;
  }

  .lesion-label {
    color: #cbd8e7;
  }

  .lesion-label a {
    color: #79d8d0;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 1rem;
    margin-top: 0.7rem;
    font-size: 0.78rem;
    color: #9aacc2;
  }

  .legend-scale {
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }

  .legend-bar {
    display: inline-block;
    width: 90px;
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(to right, #0072b2, #ffffff, #d55e00);
  }

  .legend-caption {
    color: #9aacc2;
    font-size: 0.75rem;
  }

  .legend-outline {
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }

  .legend-outline .swatch.outline {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: transparent;
    border: 2px solid #ffe08a;
  }

  .legend-no-data {
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }

  .legend-no-data .swatch {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 2px;
    /* Matches `ActivityScene.ts#NO_DATA_COLOR` (0.32, 0.35, 0.4 in linear
       0-1 RGB) converted to an sRGB hex approximation for CSS. */
    background: #6b7484;
  }

  /* Visually hidden, never `display:none` — same "sr-only" clip pattern
     `NullHistogram.svelte`'s own `.sr-only` rule documents, kept reachable
     by screen-reader/keyboard navigation while invisible on screen. */
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
</style>
