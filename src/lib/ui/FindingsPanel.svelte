<script lang="ts">
  import { tick } from 'svelte';
  import type { ArenaManifest } from '../experiment/assets';
  import type { RewiringNullLoadResult } from '../experiment/rewiringNull';
  import type { NullExplanationLoadResult } from '../experiment/nullExplanation';
  import type { PathwayInterventionsLoadResult } from '../experiment/pathwayInterventions';
  import { buildFindingSteps, findingStepStatusLabel, type FindingStep } from '../findings/steps';

  /**
   * WP1 of `.agents/plans/findings-tour` (`01-findings-panel.md`): a
   * collapsible ARIA stepper that walks a visitor through the seven-step
   * evidence chain, in order, each step templated from an already
   * fetched/sha256-verified/shape-validated artifact
   * (`src/lib/findings/steps.ts#buildFindingSteps`) — never a hard-coded
   * number. Framing (user constraint, no biological claims): the header
   * reads "Findings under this model — not claims about the real fly," and
   * every templated sentence names its decoder condition and ends with
   * "under this model."
   *
   * Collapsed by default, like `ActivityPanel.svelte`'s own "expand to see
   * more" convention — it never blocks the arena or Start. Placed directly
   * above `LedgerPanel` in `App.svelte`'s sidebar; reuses the controller's
   * already-mirrored `$state` load results (props from `App.svelte`), the
   * same "one fetch and verification per artifact, no re-fetching inside
   * the panel" discipline `LedgerPanel`/`NullExplanationNote` already
   * follow.
   *
   * Every one of the seven `<li>` steps is always rendered with its own
   * full content (sentence, provenance) — nothing is hidden behind a
   * wizard-style single-step view, so the tampering/degradation coverage in
   * `tests/e2e/findings.spec.ts` can assert every step's state at once, and
   * a screen-reader user is never forced through Next clicks to reach a
   * later step's content. "Current step" (`aria-current="step"`, the
   * `<ol>`'s own concept per `01-findings-panel.md`) is a keyboard-walking
   * cursor only: Previous/Next move which step has it and move DOM focus to
   * that step's own heading, and a polite live region announces "Step N of
   * 7" on every move — the accessible way to walk the chain in order
   * without requiring a screen-reader user to tab through six steps' worth
   * of links first.
   */

  interface Props {
    manifest: ArenaManifest | undefined;
    rewiringNull: RewiringNullLoadResult | undefined;
    nullExplanation: NullExplanationLoadResult | undefined;
    pathwayInterventions: PathwayInterventionsLoadResult | undefined;
  }

  let { manifest, rewiringNull, nullExplanation, pathwayInterventions }: Props = $props();

  let expanded = $state(false);
  let currentIndex = $state(0);
  let headingEls = $state<(HTMLElement | undefined)[]>([]);

  // Mirrors `ExperimentController#initialize()`'s own `dataBaseUrl`
  // (`${import.meta.env.BASE_URL}data`) — the same base every loader this
  // panel's steps cite already fetched against, so a provenance link
  // resolves under the app's real deployment base path (including `/fly/`
  // — see `tests/e2e/subpath.spec.ts`).
  const dataBaseUrl = `${import.meta.env.BASE_URL}data`;

  const steps = $derived<readonly FindingStep[]>(
    buildFindingSteps({ manifest, dataBaseUrl, rewiringNull, nullExplanation, pathwayInterventions })
  );

  const toggle = (): void => {
    expanded = !expanded;
  };

  const focusHeading = async (index: number): Promise<void> => {
    await tick();
    headingEls[index]?.focus();
  };

  const goTo = (index: number): void => {
    if (index < 0 || index >= steps.length) return;
    currentIndex = index;
    void focusHeading(index);
  };

  const goPrevious = (): void => goTo(currentIndex - 1);
  const goNext = (): void => goTo(currentIndex + 1);

  const conditionLabel = (condition: FindingStep['condition']): string =>
    condition === 'authored' ? 'Authored decoder' : condition === 'trained' ? 'Trained decoder' : 'Both decoders';
</script>

<section class="panel findings" aria-labelledby="findings-heading">
  <div class="section-heading">
    <div>
      <p class="eyebrow">Evidence chain</p>
      <h2 id="findings-heading">Findings under this model — not claims about the real fly</h2>
    </div>
    <button type="button" onclick={toggle} aria-expanded={expanded}>
      {expanded ? 'Collapse' : 'Expand'}
    </button>
  </div>

  {#if !expanded}
    <p class="reason">Expand to walk the evidence chain: seven steps, each templated from a verified artifact.</p>
  {/if}

  <!-- (thermo review, maintainability Suggestion) Mounted unconditionally
       (present but empty before expansion), not inside `{#if expanded}` —
       some screen readers do not reliably announce a live region's
       *initial* content when the region and its first text arrive in the
       same DOM update; they announce only a later mutation of an
       already-present node. The first "Step 1 of 7" text on expand is then
       a real mutation of an existing node, not a simultaneous insertion. -->
  <div aria-live="polite" class="sr-only">{expanded ? `Step ${currentIndex + 1} of ${steps.length}` : ''}</div>

  {#if expanded}
    <p class="disclaimer">
      Every step below names its decoder condition (authored or trained) and states its result "under this
      model" — a descriptive finding about this synthetic experiment, never a claim about the real fly. The
      authored decoder is a fixed, hand-written mapping, not biology and not trained.
    </p>

    <ol class="steps">
      {#each steps as step, index (step.id)}
        <li aria-current={index === currentIndex ? 'step' : undefined} class="step" class:current={index === currentIndex}>
          <h3
            id={`finding-step-${step.id}-heading`}
            bind:this={headingEls[index]}
            tabindex="-1"
          >
            {index + 1}. {step.title}
            <span class="condition">{conditionLabel(step.condition)}</span>
          </h3>

          {#if step.status === 'ok' && step.sentence}
            <p class="sentence">{step.sentence}</p>
          {:else if step.status !== 'ok'}
            <p class="status-line">{findingStepStatusLabel(step.status)}{step.reason ? `: ${step.reason}` : ''}</p>
          {/if}

          {#if step.id === 'behavior-repertoire'}
            <p class="see-also">
              Once published, this step will compare the measured topology's behavior repertoire against the
              rewired null. See the <a href="#atlas">behavior atlas</a> in the meantime.
            </p>
          {/if}

          {#if step.provenance.length > 0}
            <ul class="provenance">
              {#each step.provenance as entry (entry.label)}
                <li>
                  <span class="provenance-label">{entry.label}</span>
                  <span class="provenance-hash">sha256 {entry.sha256Prefix}…</span>
                  <a href={entry.artifactPath} target="_blank" rel="noreferrer">Pinned JSON</a>
                  <a href={entry.reportPath} target="_blank" rel="noreferrer">Report</a>
                </li>
              {/each}
            </ul>
          {/if}

          {#if index === currentIndex}
            <!-- (dual review, Important; narrowed by thermo review,
                 maintainability Important — the prior comment here
                 overclaimed "the next Tab reaches Next directly," which a
                 real Playwright/Chromium Tab-key session showed is false
                 whenever the current step has its own provenance links:
                 Tab lands on "Pinned JSON"/"Report" first, then Next/
                 Previous) Rendered inside the current step, immediately
                 after its own content — not above the `<ol>` — so Tab no
                 longer walks through *every earlier step's* own links to
                 reach Next again, only the *current* step's own (a normal
                 "read the evidence, then act" order: 1-4 links, one pair
                 per provenance entry, not the six-steps'-worth a control
                 placed above the list would force). See
                 `tests/e2e/findings.spec.ts`'s real-Tab-order test for the
                 guarantee this comment actually makes. -->
            <div class="stepper-controls">
              <button type="button" onclick={goPrevious} disabled={currentIndex === 0}>Previous</button>
              <button type="button" onclick={goNext} disabled={currentIndex === steps.length - 1}>Next</button>
            </div>
          {/if}
        </li>
      {/each}
    </ol>
  {/if}
</section>

<style>
  .reason {
    margin: 0.9rem 0 0;
    color: #9aacc2;
    font-size: 0.8rem;
    line-height: 1.5;
  }

  .disclaimer {
    margin: 0.9rem 0;
    color: #9aacc2;
    font-size: 0.8rem;
    line-height: 1.5;
  }

  .stepper-controls {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.6rem;
  }

  .steps {
    margin: 0;
    padding: 0;
    list-style: none;
    display: grid;
    gap: 0.75rem;
  }

  .step {
    border: 1px solid #1d2b3a;
    border-radius: 0.5rem;
    padding: 0.7rem 0.85rem;
    background: rgb(121 216 208 / 4%);
  }

  .step.current {
    border-color: #3b665b;
    background: rgb(121 216 208 / 10%);
  }

  .step h3 {
    margin: 0 0 0.4rem;
    font-size: 0.85rem;
    color: #edf4ff;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    outline: none;
  }

  .step h3:focus-visible {
    outline: 2px solid #7be5c5;
    outline-offset: 2px;
  }

  .condition {
    font-size: 0.68rem;
    font-weight: 400;
    color: #9aacc2;
    border: 1px solid #2b4354;
    border-radius: 999px;
    padding: 0.1rem 0.5rem;
  }

  .sentence {
    margin: 0.4rem 0;
    color: #cbd8e7;
    font-size: 0.82rem;
    line-height: 1.5;
  }

  .status-line {
    margin: 0.4rem 0;
    color: #9aacc2;
    font-size: 0.8rem;
    font-style: italic;
  }

  .see-also {
    margin: 0.4rem 0;
    color: #9aacc2;
    font-size: 0.8rem;
  }

  .see-also a {
    color: #79d8d0;
  }

  .provenance {
    margin: 0.5rem 0 0;
    padding: 0;
    list-style: none;
    display: grid;
    gap: 0.3rem;
  }

  .provenance li {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    font-size: 0.75rem;
    color: #9aacc2;
    border: none;
    padding: 0;
  }

  .provenance-hash {
    font-family: ui-monospace, 'SFMono-Regular', Menlo, monospace;
  }

  .provenance a {
    color: #79d8d0;
  }

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
